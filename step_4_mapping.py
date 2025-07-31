import pandas as pd
import time
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def safe_literal_eval(val):
    try:
        if isinstance(val, str):
            return ast.literal_eval(val)
        elif isinstance(val, list):
            return val
    except (ValueError, SyntaxError):
        return []
    return []

# --- EKSPANSI LEVEL SFIA ---
def expand_skill_levels(skill_levels, sfia_df):
    expanded = set()
    for item in skill_levels:
        if isinstance(item, str) and ' ' in item:
            skill, level_str = item.rsplit(' ', 1)
            try:
                level = int(level_str)
                available_levels = sfia_df[sfia_df['SFIA_Skill_Level'].str.startswith(skill + ' ')]['SFIA_Skill_Level']
                for l in range(1, level + 1):
                    candidate = f"{skill} {l}"
                    if candidate in set(available_levels):
                        expanded.add(candidate)
            except:
                continue
    return expanded


# Mapping COSINE
def map_skills_cosine(jobs_df, sfia_df, job_col, sfia_col, cluster_name, model_name, threshold=0.2):
    all_matches = []
    raw_matches = []

    corpus = jobs_df[job_col].apply(lambda x: " ".join(safe_literal_eval(x)))
    sfia_corpus = sfia_df[sfia_col].apply(lambda x: " ".join(safe_literal_eval(x)))
    vectorizer = TfidfVectorizer().fit(pd.concat([corpus, sfia_corpus]))

    visualized = False  # hanya simpan satu contoh per model

    for _, job_row in jobs_df.iterrows():
        job_skills = safe_literal_eval(job_row[job_col])
        if not job_skills:
            continue

        job_vector = vectorizer.transform([" ".join(job_skills)])
        feature_names = vectorizer.get_feature_names_out()
    
        for _, sfia_row in sfia_df.iterrows():
            sfia_skills = safe_literal_eval(sfia_row[sfia_col])
            if not sfia_skills:
                continue

            sfia_vector = vectorizer.transform([" ".join(sfia_skills)])
            score = cosine_similarity(job_vector, sfia_vector)[0][0]

            if score >= 0.1:
                raw_matches.append({
                    'job_title': job_row['job_title'],
                    'matched_skills': sfia_row['SFIA_Skill_Level'],
                    'similarity_score': score
                })

            if score >= threshold:
                all_matches.append({
                    'job_title': job_row['job_title'],
                    'matched_skills': sfia_row['SFIA_Skill_Level'],
                    'similarity_score': score
                })

            # Simpan contoh perbandingan TF-IDF ke Excel satu kali
            if not visualized:
                visualized = True

                job_tf = job_vector.toarray().flatten()
                sfia_tf = sfia_vector.toarray().flatten()

                data = {
                    'Term': feature_names,
                    'TF-IDF Job': job_tf,
                    'TF-IDF SFIA': sfia_tf
                }
                df_compare = pd.DataFrame(data)
                df_compare = df_compare[(df_compare['TF-IDF Job'] > 0) | (df_compare['TF-IDF SFIA'] > 0)]
                df_compare = df_compare.sort_values(by='Term')

                # Tambahkan info Job Title dan SFIA Skill ke setiap baris
                df_compare['Job Title'] = job_row['job_title']
                df_compare['SFIA Skill Level'] = sfia_row['SFIA_Skill_Level']

                # Metadata ringkasan
                metadata = pd.DataFrame({
                    'Keterangan': ['Job Title', 'SFIA Skill Level', 'Cosine Similarity'],
                    'Nilai': [job_row['job_title'], sfia_row['SFIA_Skill_Level'], score]
                })

                with pd.ExcelWriter(f"output/{cluster_name}/contoh_vektor_perbandingan_{model_name}.xlsx") as writer:
                    df_compare.to_excel(writer, index=False, sheet_name="TF-IDF Comparison")
                    metadata.to_excel(writer, index=False, sheet_name="Metadata")

    if not all_matches:
        print(f"Tidak ada hasil mapping untuk model {model_name} (Cosine)")
        return
    
    raw_predicted = pd.DataFrame(raw_matches).sort_values(by=['job_title', 'similarity_score'], ascending=[True, False])
    raw_predicted.to_excel(
        f"output/{cluster_name}_SFIAFormat/raw_mapping_cosine_{model_name}_{cluster_name}.xlsx", index=False
    )

    predicted_df = pd.DataFrame(all_matches).sort_values(by=['job_title', 'similarity_score'], ascending=[True, False])
    predicted_df.to_excel(
        f"output/{cluster_name}/mapping_cosine_{model_name}_{cluster_name}.xlsx", index=False
    )

    # Ekspansi SFIA skill level
    expanded_rows = []
    for _, row in predicted_df.iterrows():
        original_skill = row['matched_skills']
        expanded_skills_set = expand_skill_levels([original_skill], sfia_df)
        for expanded_skill in expanded_skills_set:
            expanded_rows.append({
                'job_title': row['job_title'],
                'expanded_matched_skills': expanded_skill,
                'similarity_score': row['similarity_score']
            })

    expanded_df = pd.DataFrame(expanded_rows).sort_values(by=['job_title', 'similarity_score'], ascending=[True, False])
    expanded_df.to_excel(
        f"output/{cluster_name}/expanded_mapping_cosine_{model_name}_{cluster_name}.xlsx", index=False
    )

    unique_predicted = predicted_df['matched_skills'].unique().tolist()
    expanded_set = sorted(expand_skill_levels(unique_predicted, sfia_df))
    # unique_expanded = pd.DataFrame({'expanded_matched_skills': expanded_set})


    print(f"Cosine mapping '{model_name}' disimpan. ({len(predicted_df)} baris pemetaan).")
    print(f"Hasil sebelum ekspansi : {len(unique_predicted)}, Setelah ekspansi : {len(expanded_set)}.\n")


# --- MAIN ---
if __name__ == '__main__':
    start_time = time.time()
    cluster_name = "IS" # Ganti CS atau IS

    jobs_df = pd.read_excel(f"output/{cluster_name}/skills_extracted_jobs_{cluster_name}.xlsx")
    sfia_df = pd.read_excel(f"output/{cluster_name}/skills_extracted_sfia_{cluster_name}.xlsx")

    COSINE_MODELS = [
        "SkillNER",
        "SkillNER QE",
        "SkillNER RAKE",
        "SkillNER YAKE",
        "SkillNER_KeyBERT",
        "SkillNER QE_KeyBERT",
        "SkillNER RAKE_KeyBERT",
        "SkillNER YAKE_KeyBERT"
    ]

    print("\nMulai mapping COSINE...")
    for col in COSINE_MODELS:
        try:
            map_skills_cosine(jobs_df, sfia_df, col, col, cluster_name, col)
        except Exception as e:
            print(f"Error model cosine '{col}': {e}")

    print(f"\nTotal waktu proses: {time.time() - start_time:.2f} detik")