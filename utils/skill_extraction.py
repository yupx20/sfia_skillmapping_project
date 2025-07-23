from utils.model_loader import load_models
import nltk
from nltk.corpus import wordnet as wn
nltk.download('wordnet')
nltk.download('omw-1.4')
skill_extractor, rake_extractor, yake_extractor, kw_model = load_models()

def extract_skills_skillner(text):
    if not isinstance(text, str) or not text.strip():
            return []
            
    try:
        annotations = skill_extractor.annotate(text)
        skills = set()
        
        results = annotations.get('results', {})
        if not results:
            return []

        for match in results.get('full_matches', []):
            skills.add(match['doc_node_value'].lower())
            
        for match in results.get('ngram_scored', []):
            skills.add(match['doc_node_value'].lower())
            
        return list(skills)
    
    except Exception as e:
        print(f"Terjadi error saat memproses teks: '{str(text)[:50]}...'. Error: {e}")
        return []

def expand_terms(term):

    synonyms = set()
    for syn in wn.synsets(term):
        for lemma in syn.lemmas():
            name = lemma.name().replace('_', ' ').lower()
            if name != term.lower():
                synonyms.add(name)
    return synonyms

def extract_skills_skillner_qe(text):

    original_skills = extract_skills_skillner(text)
    expanded_skills = set(original_skills)

    for skill in original_skills:
        synonyms = expand_terms(skill)
        expanded_skills.update(synonyms)

    return list(expanded_skills)

def extract_rake_keywords(text):
    if not isinstance(text, str) or not text.strip():
        return []
    rake_extractor.extract_keywords_from_text(text)
    keywords = rake_extractor.get_ranked_phrases()
    return [kw.strip().lower() for kw in keywords]

def extract_yake_keywords(text):
    if not isinstance(text, str): return []
    keywords = yake_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def extract_keybert_keywords(text):
    if not isinstance(text, str) or not text.strip(): return []
    keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), stop_words='english', use_mmr=True)
    return [kw for kw, _ in keywords]