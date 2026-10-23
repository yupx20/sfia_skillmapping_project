import os
import sys
import ast
import time
import logging
from typing import Dict, List, Any, Tuple
import pandas as pd
from googletrans import Translator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure app directory is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from utils.text_preprocessing import preprocess_text
from utils.skill_extraction import (
    extract_skills_skillner,
    extract_skills_skillner_qe,
    extract_keybert_keywords,
    kw_model
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import re

# SFIA 6 Standard Categories
SFIA_CATEGORIES = [
    "Strategy and architecture",
    "Change and transformation",
    "Development and implementation",
    "Delivery and operation",
    "People and skills",
    "Relationships and engagement"
]

def clean_desc(text: str) -> str:
    """Strip Excel OpenXML artifacts like _x000D_ and normalize formatting."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'_x000D_[\n\r]*', '\n', text)
    text = re.sub(r'_x000D_', ' ', text)
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n+', '\n\n', text)
    return text.strip()

class SFIAInferenceEngine:
    def __init__(self, cs_sfia_path: str, is_sfia_path: str, sfia_meta_path: str):
        self.cs_sfia_path = cs_sfia_path
        self.is_sfia_path = is_sfia_path
        self.sfia_meta_path = sfia_meta_path
        self.translator = Translator()
        
        logger.info("Initializing SFIA Knowledge Base...")
        self.sfia_cs_df, self.sfia_is_df = self._load_extracted_sfia()
        self.meta_dict = self._load_meta_dict()
        logger.info(f"Loaded {len(self.sfia_cs_df)} CS entries and {len(self.sfia_is_df)} IS entries.")

    def _safe_literal_eval(self, val):
        if isinstance(val, list):
            return val
        if isinstance(val, str):
            try:
                return ast.literal_eval(val)
            except Exception:
                return [val]
        return []

    def _load_extracted_sfia(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        df_cs = pd.read_excel(self.cs_sfia_path)
        df_is = pd.read_excel(self.is_sfia_path)
        
        # Clean Excel artifacts from level descriptions
        df_cs['Level_Description'] = df_cs['Level_Description'].apply(clean_desc)
        df_is['Level_Description'] = df_is['Level_Description'].apply(clean_desc)
        
        # Pre-parse lists for fast inference
        df_cs['skills_parsed'] = df_cs['SkillNER QE_KeyBERT'].apply(self._safe_literal_eval)
        df_is['skills_parsed'] = df_is['SkillNER QE_KeyBERT'].apply(self._safe_literal_eval)
        
        return df_cs, df_is

    def _load_meta_dict(self) -> Dict[str, Dict[str, Any]]:
        """Load category, subcategory and level descriptions from sfia9_en2025.xlsx"""
        meta = {}
        if os.path.exists(self.sfia_meta_path):
            try:
                df = pd.read_excel(self.sfia_meta_path)
                for _, row in df.iterrows():
                    skill_name = str(row['Skill']).strip()
                    meta[skill_name] = {
                        "category": str(row.get('Category', 'Other')).strip(),
                        "subcategory": str(row.get('Subcategory', '')).strip(),
                        "overall_desc": clean_desc(str(row.get('Overall description', ''))),
                        "levels": {
                            l: clean_desc(str(row.get(f'Level {l} description', '')))
                            for l in range(1, 8) if pd.notna(row.get(f'Level {l} description'))
                        }
                    }
            except Exception as e:
                logger.error(f"Error loading SFIA metadata: {e}")
        return meta

    def translate_if_needed(self, text: str) -> Tuple[str, bool]:
        """Detect Indonesian/non-English words and translate to English."""
        # Quick heuristic check for Indonesian keywords
        id_markers = [" dan ", " yang ", " dengan ", " untuk ", " sebagai ", " pengalaman ", " kemampuan ", " bertanggung jawab "]
        text_lower = text.lower()
        is_indonesian = any(marker in text_lower for marker in id_markers)
        
        if is_indonesian:
            try:
                translated = self.translator.translate(text[:4000], dest='en').text
                return translated, True
            except Exception as e:
                logger.warning(f"Translation failed or timed out: {e}")
                return text, False
        return text, False

    def expand_skill_levels(self, skill_levels: List[str], sfia_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Expand mapped skill level N to all foundational levels 1..N"""
        available_set = set(sfia_df['SFIA_Skill_Level'])
        expanded_records = []
        
        for item in skill_levels:
            if isinstance(item, str) and ' ' in item:
                skill, level_str = item.rsplit(' ', 1)
                try:
                    level = int(level_str)
                    for l in range(1, level + 1):
                        candidate = f"{skill} {l}"
                        if candidate in available_set:
                            expanded_records.append({
                                "skill": skill,
                                "level": l,
                                "sfia_skill_level": candidate,
                                "is_original_target": (l == level)
                            })
                except Exception:
                    continue
        return expanded_records

    def predict(self, job_title: str, job_description: str, cluster: str = "auto", threshold: float = 0.20, model_type: str = "skillner_qe_tfidf") -> Dict[str, Any]:
        """
        Execute model pipeline:
        - skillner_qe_tfidf (User Recommended): SkillNER + Cleaned WordNet QE + TF-IDF
        - skillner_qe_keybert_tfidf: SkillNER + Cleaned QE + KeyBERT + TF-IDF
        - skillner_tfidf: SkillNER Exact Dictionary + TF-IDF
        """
        start_time = time.time()
        timings = {}
        
        # 1. Cleaning & Translation
        t0 = time.time()
        english_text, was_translated = self.translate_if_needed(job_description)
        cleaned_text = preprocess_text(english_text)
        timings["cleaning_translation"] = round(time.time() - t0, 3)

        # 2. Skill Extraction (SkillNER)
        t0 = time.time()
        skillner_skills = extract_skills_skillner(cleaned_text)
        timings["skillner"] = round(time.time() - t0, 3)

        # 3. Query Expansion (Cleaned WordNet Synonyms & Lemmas)
        t0 = time.time()
        expanded_skills = extract_skills_skillner_qe(cleaned_text)
        timings["query_expansion"] = round(time.time() - t0, 3)

        # 4. KeyBERT MMR Keyphrase Ranking & Filtering (if requested)
        t0 = time.time()
        keybert_skills = []
        keybert_scores = {}
        
        if "keybert" in model_type:
            keybert_input = " ".join(expanded_skills) if expanded_skills else cleaned_text
            if keybert_input.strip():
                # Extract keyphrases with score tuples
                keybert_results = kw_model.extract_keywords(
                    keybert_input,
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    use_mmr=True,
                    top_n=20
                )
                from utils.skill_extraction import PROFANITY_SLANG_BLACKLIST
                for kw, score in keybert_results:
                    kw_clean = kw.lower().strip()
                    if not any(bad in kw_clean for bad in PROFANITY_SLANG_BLACKLIST):
                        keybert_skills.append(kw)
                        keybert_scores[kw] = round(float(score), 4)
        timings["keybert"] = round(time.time() - t0, 3)

        # Determine target cluster
        if cluster.upper() in ["CS", "IS"]:
            selected_cluster = cluster.upper()
        else:
            # Auto-detect based on job title / keywords
            cs_keywords = ["software", "developer", "engineer", "algorithm", "fullstack", "backend", "frontend", "devops", "cloud", "security"]
            is_keywords = ["analyst", "business", "data", "product", "manager", "governance", "consultant", "erp", "scrum", "project"]
            title_lower = (job_title + " " + cleaned_text[:200]).lower()
            cs_score = sum(1 for kw in cs_keywords if kw in title_lower)
            is_score = sum(1 for kw in is_keywords if kw in title_lower)
            selected_cluster = "CS" if cs_score >= is_score else "IS"

        sfia_df = self.sfia_cs_df if selected_cluster == "CS" else self.sfia_is_df

        # 5. Semantic Mapping: TF-IDF & Cosine Similarity
        t0 = time.time()
        if model_type == "skillner_tfidf":
            job_corpus_str = " ".join(skillner_skills) if skillner_skills else cleaned_text
            active_model_label = "SkillNER + TF-IDF"
        elif model_type == "skillner_qe_keybert_tfidf":
            job_corpus_str = " ".join(keybert_skills) if keybert_skills else " ".join(expanded_skills)
            active_model_label = "SkillNER + QE + KeyBERT + TF-IDF"
        else: # Default: skillner_qe_tfidf
            job_corpus_str = " ".join(expanded_skills) if expanded_skills else " ".join(skillner_skills)
            active_model_label = "SkillNER + QE + TF-IDF"
        sfia_corpus_series = sfia_df['skills_parsed'].apply(lambda x: " ".join(x))
        vectorizer = TfidfVectorizer().fit(pd.concat([pd.Series([job_corpus_str]), sfia_corpus_series]))
        job_vec = vectorizer.transform([job_corpus_str])
        sfia_mat = vectorizer.transform(sfia_corpus_series)
        
        sim_scores = cosine_similarity(job_vec, sfia_mat)[0]
        
        # Check if threshold yields matches, otherwise adaptively fallback to top-K
        passing_indices = [i for i, s in enumerate(sim_scores) if s >= threshold]
        if not passing_indices:
            # Fallback to top 12 matches with score > 0.05
            top_indices = sim_scores.argsort()[::-1][:12]
            passing_indices = [i for i in top_indices if sim_scores[i] > 0.05]

        job_terms_set = set(job_corpus_str.lower().split())
        matches = []
        for i in passing_indices:
            row = sfia_df.iloc[i]
            sim = float(sim_scores[i])
            skill_name = str(row['Skill'])
            level = int(row['Level'])
            meta_info = self.meta_dict.get(skill_name, {})
            
            sfia_terms_set = set(" ".join(row['skills_parsed']).lower().split())
            matched_tokens = list(job_terms_set.intersection(sfia_terms_set))
            
            matches.append({
                "skill": skill_name,
                "level": level,
                "sfia_skill_level": row['SFIA_Skill_Level'],
                "similarity_score": round(sim, 4),
                "category": meta_info.get("category", "Other"),
                "subcategory": meta_info.get("subcategory", ""),
                "level_description": clean_desc(str(row['Level_Description'])),
                "matched_tokens": matched_tokens[:10]
            })

        # Sort matches by similarity score descending
        matches.sort(key=lambda x: x['similarity_score'], reverse=True)
        timings["tfidf_cosine"] = round(time.time() - t0, 3)

        # 6. SFIA Hierarchical Skill Level Expansion
        t0 = time.time()
        unique_matched_skill_levels = [m['sfia_skill_level'] for m in matches]
        expanded_records = self.expand_skill_levels(unique_matched_skill_levels, sfia_df)
        timings["level_expansion"] = round(time.time() - t0, 3)

        # 7. Aggregate by Skill for UI Presentation
        grouped_skills = {}
        for m in matches:
            s_name = m['skill']
            if s_name not in grouped_skills:
                grouped_skills[s_name] = {
                    "skill": s_name,
                    "category": m['category'],
                    "subcategory": m['subcategory'],
                    "highest_level": m['level'],
                    "highest_score": m['similarity_score'],
                    "target_sfia_level": m['sfia_skill_level'],
                    "matched_levels": [m['level']],
                    "expanded_levels": [],
                    "descriptions": {m['level']: m['level_description']},
                    "matched_tokens": m['matched_tokens']
                }
            else:
                if m['level'] > grouped_skills[s_name]['highest_level']:
                    grouped_skills[s_name]['highest_level'] = m['level']
                if m['similarity_score'] > grouped_skills[s_name]['highest_score']:
                    grouped_skills[s_name]['highest_score'] = m['similarity_score']
                grouped_skills[s_name]['matched_levels'].append(m['level'])
                grouped_skills[s_name]['descriptions'][m['level']] = m['level_description']
                grouped_skills[s_name]['matched_tokens'] = list(set(grouped_skills[s_name]['matched_tokens'] + m['matched_tokens']))

        # Populate expanded levels (1..highest_level)
        for s_name, data in grouped_skills.items():
            max_lvl = data['highest_level']
            meta_levels = self.meta_dict.get(s_name, {}).get("levels", {})
            all_possible_levels = [lvl for lvl in range(1, max_lvl + 1) if lvl in meta_levels or f"{s_name} {lvl}" in set(sfia_df['SFIA_Skill_Level'])]
            data['expanded_levels'] = sorted(list(set(all_possible_levels)))
            
            # Enrich missing level descriptions from meta_dict
            for lvl in data['expanded_levels']:
                if lvl not in data['descriptions'] and lvl in meta_levels:
                    data['descriptions'][lvl] = clean_desc(meta_levels[lvl])

        skill_cards = list(grouped_skills.values())
        skill_cards.sort(key=lambda x: x['highest_score'], reverse=True)

        # 8. SFIA Category Radar Distribution
        category_counts = {cat: 0 for cat in SFIA_CATEGORIES}
        category_scores = {cat: 0.0 for cat in SFIA_CATEGORIES}
        
        for sc in skill_cards:
            cat = sc['category']
            if cat in category_counts:
                category_counts[cat] += 1
                category_scores[cat] += sc['highest_score']

        radar_data = {
            "labels": SFIA_CATEGORIES,
            "counts": [category_counts[cat] for cat in SFIA_CATEGORIES],
            "scores": [round(category_scores[cat], 3) for cat in SFIA_CATEGORIES]
        }

        # Level distribution (Level 1..7)
        level_counts = {lvl: 0 for lvl in range(1, 8)}
        for sc in skill_cards:
            for lvl in sc['expanded_levels']:
                if lvl in level_counts:
                    level_counts[lvl] += 1

        total_time = round(time.time() - start_time, 3)

        return {
            "job_title": job_title,
            "active_model": active_model_label,
            "detected_cluster": selected_cluster,
            "was_translated": was_translated,
            "timings": timings,
            "total_execution_time": total_time,
            "counts": {
                "skillner_count": len(skillner_skills),
                "expanded_skills_count": len(expanded_skills),
                "keybert_count": len(keybert_skills),
                "matched_competencies_count": len(matches),
                "expanded_competencies_count": len(expanded_records),
                "unique_skills_mapped": len(skill_cards)
            },
            "extracted_skills": {
                "skillner": skillner_skills,
                "wordnet_qe": expanded_skills,
                "keybert": [{"keyword": kw, "score": keybert_scores.get(kw, 0.0)} for kw in keybert_skills]
            },
            "mapped_skills": skill_cards,
            "radar_data": radar_data,
            "level_distribution": {f"Level {k}": v for k, v in level_counts.items()}
        }
