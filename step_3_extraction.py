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

def extract_all_skills(job_file: str, sfia_file: str, cluster_name: str):
    jobs_df = pd.read_csv(job_file)
    sfia_df = pd.read_csv(sfia_file)

    print("Mengekstraksi keterampilan dari data lowongan...")
    jobs_df['skills_skillner'] = jobs_df['job_description_cleaned'].apply(extract_skills_skillner)
    jobs_df['skills_skillner_qe'] = jobs_df['job_description_cleaned'].apply(extract_skills_skillner_qe)
    jobs_df['skills_ner_bert'] = jobs_df['job_description_cleaned'].apply(extract_ner_bert_skills)
    jobs_df['keybert_skills'] = jobs_df['job_description_cleaned'].apply(extract_keybert_keywords)
    jobs_df['rake_skills'] = jobs_df['job_description_cleaned'].apply(extract_rake_keywords)
    jobs_df['yake_skills'] = jobs_df['job_description_cleaned'].apply(extract_yake_keywords)

    print("Mengekstraksi keterampilan dari data SFIA...")
    sfia_df['skills_skillner'] = sfia_df['Level_Description_cleaned'].apply(extract_skills_skillner)
    sfia_df['skills_skillner_qe'] = sfia_df['Level_Description_cleaned'].apply(extract_skills_skillner_qe)
    sfia_df['skills_ner_bert'] = sfia_df['Level_Description_cleaned'].apply(extract_ner_bert_skills)
    sfia_df['keybert_skills'] = sfia_df['Level_Description_cleaned'].apply(extract_keybert_keywords)
    sfia_df['rake_skills'] = sfia_df['Level_Description_cleaned'].apply(extract_rake_keywords)
    sfia_df['yake_skills'] = sfia_df['Level_Description_cleaned'].apply(extract_yake_keywords)

    # Simpan hasil
    jobs_df.to_csv(f"skills_extracted_jobs_{cluster_name}.csv", index=False)
    sfia_df.to_csv(f"skills_extracted_sfia_{cluster_name}.csv", index=False)
    print(f"Hasil ekstraksi disimpan ke skills_extracted_jobs_{cluster_name}.csv dan skills_extracted_sfia_{cluster_name}.csv")

if __name__ == '__main__':
    extract_all_skills("output/processed_jobs_CS.csv", "output/processed_sfia_CS.csv", "CS")

    end_time = time.time()
    print(f"Waktu yang dibutuhkan: {end_time - start_time:.2f} detik")
