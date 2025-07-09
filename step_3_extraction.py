import pandas as pd
import time
from utils.skill_extraction import (
    extract_skills_skillner,
    extract_skills_skillner_qe,
    extract_ner_bert_skills,
    extract_keybert_keywords,
    extract_rake_keywords,
    extract_yake_keywords
)

start_time = time.time()

def count_skills_per_row(skills_column):
    return skills_column.apply(lambda x: len(x) if isinstance(x, list) else 0)

def extract_all_skills(job_file: str, sfia_file: str):
    jobs_df = pd.read_csv(job_file)
    sfia_df = pd.read_csv(sfia_file)

    # === Ekstraksi JOBS ===
    print("Mengekstraksi keterampilan dari data lowongan...")
    jobs_df['skills_skillner'] = jobs_df['job_description_cleaned'].apply(extract_skills_skillner)
    jobs_df['skills_skillner_qe'] = jobs_df['job_description_cleaned'].apply(extract_skills_skillner_qe)
    jobs_df['skills_ner_bert'] = jobs_df['job_description_cleaned'].apply(extract_ner_bert_skills)

    jobs_df['skills_skillner_keybert'] = jobs_df['skills_skillner'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    jobs_df['skills_ner_bert_keybert'] = jobs_df['skills_ner_bert'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    jobs_df['skills_skillner_qe_keybert'] = jobs_df['skills_skillner_qe'].apply(lambda s: extract_keybert_keywords(" ".join(s)))

    jobs_df['skills_skillner_rake'] = jobs_df['skills_skillner'].apply(lambda s: extract_rake_keywords(" ".join(s)))
    jobs_df['skills_skillner_yake'] = jobs_df['skills_skillner'].apply(lambda s: extract_yake_keywords(" ".join(s)))
    jobs_df['skills_ner_bert_rake'] = jobs_df['skills_ner_bert'].apply(lambda s: extract_rake_keywords(" ".join(s)))
    jobs_df['skills_ner_bert_yake'] = jobs_df['skills_ner_bert'].apply(lambda s: extract_yake_keywords(" ".join(s)))

    # === Ekstraksi SFIA ===
    print("Mengekstraksi keterampilan dari data SFIA...")
    sfia_df['skills_skillner'] = sfia_df['Level_Description_cleaned'].apply(extract_skills_skillner)
    sfia_df['skills_skillner_qe'] = sfia_df['Level_Description_cleaned'].apply(extract_skills_skillner_qe)
    sfia_df['skills_ner_bert'] = sfia_df['Level_Description_cleaned'].apply(extract_ner_bert_skills)

    sfia_df['skills_skillner_keybert'] = sfia_df['skills_skillner'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    sfia_df['skills_ner_bert_keybert'] = sfia_df['skills_ner_bert'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    sfia_df['skills_skillner_qe_keybert'] = sfia_df['skills_skillner_qe'].apply(lambda s: extract_keybert_keywords(" ".join(s)))

    sfia_df['skills_skillner_rake'] = sfia_df['skills_skillner'].apply(lambda s: extract_rake_keywords(" ".join(s)))
    sfia_df['skills_skillner_yake'] = sfia_df['skills_skillner'].apply(lambda s: extract_yake_keywords(" ".join(s)))
    sfia_df['skills_ner_bert_rake'] = sfia_df['skills_ner_bert'].apply(lambda s: extract_rake_keywords(" ".join(s)))
    sfia_df['skills_ner_bert_yake'] = sfia_df['skills_ner_bert'].apply(lambda s: extract_yake_keywords(" ".join(s)))

    # Tambahkan kolom jumlah keterampilan per baris
    print("\nStatistik jumlah keterampilan per job (lowongan):")
    for col in ['skills_skillner',
                'skills_skillner_qe', 
                'skills_ner_bert',
                'skills_skillner_keybert',
                'skills_ner_bert_keybert',
                'skills_skillner_qe_keybert',
                'skills_skillner_rake',
                'skills_skillner_yake',
                'skills_ner_bert_rake',
                'skills_ner_bert_yake']:
        count_col = col + '_count'
        jobs_df[count_col] = count_skills_per_row(jobs_df[col])
        print(f"  - {col}:")
        print(f"      Mean: {jobs_df[count_col].mean():.2f}, Min: {jobs_df[count_col].min()}, Max: {jobs_df[count_col].max()}, Non-zero: {(jobs_df[count_col] > 0).sum()}")

    print("\nStatistik jumlah keterampilan per deskripsi SFIA:")
    for col in ['skills_skillner',
                'skills_skillner_qe', 
                'skills_ner_bert',
                'skills_skillner_keybert',
                'skills_ner_bert_keybert',
                'skills_skillner_qe_keybert',
                'skills_skillner_rake',
                'skills_skillner_yake',
                'skills_ner_bert_rake',
                'skills_ner_bert_yake']:
        count_col = col + '_count'
        sfia_df[count_col] = count_skills_per_row(sfia_df[col])
        print(f"  - {col}:")
        print(f"      Mean: {sfia_df[count_col].mean():.2f}, Min: {sfia_df[count_col].min()}, Max: {sfia_df[count_col].max()}, Non-zero: {(sfia_df[count_col] > 0).sum()}")

    return jobs_df, sfia_df

# --- MAIN ---
if __name__ == '__main__':
    cluster_name = "IS"
    jobs_df, sfia_df = extract_all_skills(f"output/{cluster_name}/processed_jobs_{cluster_name}.csv", f"output/{cluster_name}/processed_sfia_{cluster_name}.csv")

    jobs_df.to_csv(f"output/{cluster_name}/skills_extracted_jobs_{cluster_name}.csv", index=False)
    sfia_df.to_csv(f"output/{cluster_name}/skills_extracted_sfia_{cluster_name}.csv", index=False)
    print(f"\nHasil ekstraksi disimpan ke:")
    print(f"   - output/{cluster_name}/skills_extracted_jobs_{cluster_name}.csv")
    print(f"   - output/{cluster_name}/skills_extracted_sfia_{cluster_name}.csv")

    end_time = time.time()
    print(f"\nWaktu yang dibutuhkan: {end_time - start_time:.2f} detik")