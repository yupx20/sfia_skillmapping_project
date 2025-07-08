import pandas as pd
import time
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- Utilitas ---
def join_skills(skill_lists):
    return [" ".join(skills) if isinstance(skills, list) else "" for skills in skill_lists]

def join_all_skills_column(df, column):
    all_skills = set()
    for skills in df[column]:
        if isinstance(skills, list):
            all_skills.update(skills)
        elif isinstance(skills, str):
            try:
                skills_list = ast.literal_eval(skills)
                if isinstance(skills_list, list):
                    all_skills.update(skills_list)
            except:
                continue
    return " ".join(all_skills)

def combine_columns(df, new_col, base_cols):
    combined = []
    for _, row in df.iterrows():
        merged = set()
        for col in base_cols:
            try:
                skills = eval(row[col]) if isinstance(row[col], str) else []
                merged.update(skills)
            except:
                continue
        combined.append(list(merged))
    df[new_col] = combined

def preprocess_rake_string(rake_entry):
    if isinstance(rake_entry, str):
        phrases = []
        try:
            raw = ast.literal_eval(rake_entry)
            if isinstance(raw, list) and raw:
                long_string = raw[0]
                phrases = [phrase.strip() for phrase in long_string.split('  ') if phrase.strip()]
        except:
            pass
        return phrases
    return []

def parse_yake_list(yake_entry):
    if isinstance(yake_entry, str):
        try:
            return ast.literal_eval(yake_entry)
        except:
            return []
    elif isinstance(yake_entry, list):
        return yake_entry
    return []

# Mapping COSINE
def map_skills_cosine(jobs_df, sfia_df, job_col, sfia_col, cluster_name, model_name, threshold=0.2):
    combined_job_text = join_all_skills_column(jobs_df, job_col)
    sfia_text = join_skills(sfia_df[sfia_col])
    vectorizer = TfidfVectorizer().fit([combined_job_text] + sfia_text)
    job_vector = vectorizer.transform([combined_job_text])
    sfia_vectors = vectorizer.transform(sfia_text)

    similarity = cosine_similarity(job_vector, sfia_vectors)[0]
    predicted_skills = [
        sfia_df['SFIA_Skill_Level'].iloc[i]
        for i, score in enumerate(similarity)
        if score >= threshold
    ]
    unique_predicted = sorted(set(predicted_skills))
    pd.DataFrame({'matched_skills': unique_predicted}).to_csv(
        f"output/{cluster_name}/mapping_cosine_{model_name}_{cluster_name}.csv", index=False
    )
    print(f"Cosine mapping '{model_name}' disimpan ke mapping_cosine_{model_name}_{cluster_name}.csv")

# Mapping JACCARD
def jaccard_similarity(set1, set2):
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def map_skills_jaccard_per_job(jobs_df, sfia_df, job_col, sfia_col, cluster_name, model_name, threshold=0.1):
    matched_sfias = set()

    for job_index, job_skills_raw in enumerate(jobs_df[job_col]):
        try:
            job_skills = set(job_skills_raw) if isinstance(job_skills_raw, list) else set(ast.literal_eval(job_skills_raw))
        except:
            job_skills = set()

        for sfia_index, sfia_skills_raw in enumerate(sfia_df[sfia_col]):
            try:
                sfia_skills = set(sfia_skills_raw) if isinstance(sfia_skills_raw, list) else set(ast.literal_eval(sfia_skills_raw))
            except:
                sfia_skills = set()

            if job_skills and sfia_skills:
                similarity = len(job_skills & sfia_skills) / len(job_skills | sfia_skills)
                if similarity >= threshold:
                    matched_sfias.add(sfia_df['SFIA_Skill_Level'].iloc[sfia_index])

    if matched_sfias:
        result_df = pd.DataFrame({'matched_skills': sorted(matched_sfias)})
        result_df.to_csv(f"output/{cluster_name}/mapping_jaccard_{model_name}_{cluster_name}.csv", index=False)
        print(f"Jaccard mapping '{model_name}' disimpan ke mapping_jaccard_{model_name}_{cluster_name}.csv")
    else:
        print(f"Tidak ada hasil mapping untuk model {model_name}")

# --- MAIN ---
if __name__ == '__main__':
    start_time = time.time()
    cluster_name = "IS"

    jobs_df = pd.read_csv(f"output/{cluster_name}/skills_extracted_jobs_{cluster_name}.csv")
    sfia_df = pd.read_csv(f"output/{cluster_name}/skills_extracted_sfia_{cluster_name}.csv")

    # jobs_df["rake_skills"] = jobs_df["rake_skills"].apply(preprocess_rake_string)
    # sfia_df["rake_skills"] = sfia_df["rake_skills"].apply(preprocess_rake_string)

    jobs_df["yake_skills"] = jobs_df["yake_skills"].apply(lambda x: parse_yake_list(x))
    sfia_df["yake_skills"] = sfia_df["yake_skills"].apply(lambda x: parse_yake_list(x))

    # print(jobs_df["yake_skills"])
    # print(sfia_df["yake_skills"])

    # === GABUNGKAN KOLOM MODEL ===
    print("Menggabungkan kolom model...")
    combine_columns(jobs_df, "skills_skillner_tfidf", ["skills_skillner"])
    combine_columns(jobs_df, "skills_skillner_keybert_tfidf", ["skills_skillner", "keybert_skills"])
    combine_columns(jobs_df, "skills_ner_bert_tfidf", ["skills_ner_bert"])
    combine_columns(jobs_df, "skills_ner_bert_keybert_tfidf", ["skills_ner_bert", "keybert_skills"])
    combine_columns(jobs_df, "skills_skillner_qe_tfidf", ["skills_skillner_qe"])
    combine_columns(jobs_df, "skills_skillner_qe_keybert_tfidf", ["skills_skillner_qe", "keybert_skills"])

    combine_columns(jobs_df, "skills_skillner_rake", ["skills_skillner", "rake_skills"])
    combine_columns(jobs_df, "skills_skillner_yake", ["skills_skillner", "yake_skills"])
    combine_columns(jobs_df, "skills_ner_bert_rake", ["skills_ner_bert", "rake_skills"])
    combine_columns(jobs_df, "skills_ner_bert_yake", ["skills_ner_bert", "yake_skills"])

    combine_columns(sfia_df, "skills_skillner_tfidf", ["skills_skillner"])
    combine_columns(sfia_df, "skills_skillner_keybert_tfidf", ["skills_skillner", "keybert_skills"])
    combine_columns(sfia_df, "skills_ner_bert_tfidf", ["skills_ner_bert"])
    combine_columns(sfia_df, "skills_ner_bert_keybert_tfidf", ["skills_ner_bert", "keybert_skills"])
    combine_columns(sfia_df, "skills_skillner_qe_tfidf", ["skills_skillner_qe"])
    combine_columns(sfia_df, "skills_skillner_qe_keybert_tfidf", ["skills_skillner_qe", "keybert_skills"])

    combine_columns(sfia_df, "skills_skillner_rake", ["skills_skillner", "rake_skills"])
    combine_columns(sfia_df, "skills_skillner_yake", ["skills_skillner", "yake_skills"])
    combine_columns(sfia_df, "skills_ner_bert_rake", ["skills_ner_bert", "rake_skills"])
    combine_columns(sfia_df, "skills_ner_bert_yake", ["skills_ner_bert", "yake_skills"])

    # === MODEL KONFIGURASI ===
    COSINE_MODELS = [
        "skills_skillner_tfidf",
        "skills_skillner_keybert_tfidf",
        "skills_ner_bert_tfidf",
        "skills_ner_bert_keybert_tfidf",
        "skills_skillner_qe_tfidf",
        "skills_skillner_qe_keybert_tfidf"
    ]

    JACCARD_MODELS = [
        "skills_skillner_rake",
        "skills_skillner_yake",
        "skills_ner_bert_rake",
        "skills_ner_bert_yake"
    ]

    print("\nMulai mapping COSINE...")
    for col in COSINE_MODELS:
        try:
            map_skills_cosine(jobs_df, sfia_df, col, col, cluster_name, col)
        except Exception as e:
            print(f"Error model cosine '{col}': {e}")

    print("\nMulai mapping JACCARD...")
    for col in JACCARD_MODELS:
        try:
            map_skills_jaccard_per_job(jobs_df, sfia_df, col, col, cluster_name, col)
        except Exception as e:
            print(f"Error model jaccard '{col}': {e}")

    print(f"\nTotal waktu proses: {time.time() - start_time:.2f} detik")
