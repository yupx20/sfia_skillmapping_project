import pandas as pd
from googletrans import Translator
from utils.text_preprocessing import preprocess_text
from utils.sfia_processing import transform_sfia_to_long_format

translator = Translator()

def translate_text(text):
    try:
        return translator.translate(text, dest='en').text
    except:
        return text

def preprocess_jobs_and_sfia(job_file: str, sfia_file: str):
    jobs_df = pd.read_excel(job_file)
    sfia_df = transform_sfia_to_long_format(sfia_file)

    print("Menerjemahkan dan membersihkan deskripsi lowongan...")
    jobs_df['job_description_cleaned'] = jobs_df['job_description'].apply(translate_text).apply(preprocess_text)

    print("Mengubah dan membersihkan deskripsi SFIA...")
    sfia_df['Level_Description_cleaned'] = sfia_df['Level_Description'].apply(preprocess_text) # Tidak perlu translate

    return jobs_df, sfia_df

if __name__ == '__main__':
    cluster_name = "CS"
    jobs_df, sfia_df = preprocess_jobs_and_sfia(f"output/{cluster_name}/cleaned_{cluster_name}Jobs.xlsx", "data/sfia9_en2025.xlsx")
    jobs_df.to_excel(f"output/{cluster_name}/processed_jobs_{cluster_name}.xlsx", index=False)
    sfia_df.to_excel(f"output/{cluster_name}/processed_sfia_{cluster_name}.xlsx", index=False)
    print(f"Data lowongan disimpan ke processed_jobs_{cluster_name}.xlsx")
    print(f"Data SFIA disimpan ke processed_sfia_{cluster_name}.xlsx")

    cluster_name = "IS"
    jobs_df, sfia_df = preprocess_jobs_and_sfia(f"output/{cluster_name}/cleaned_{cluster_name}Jobs.xlsx", "data/sfia9_en2025.xlsx")
    jobs_df.to_excel(f"output/{cluster_name}/processed_jobs_{cluster_name}.xlsx", index=False)
    sfia_df.to_excel(f"output/{cluster_name}/processed_sfia_{cluster_name}.xlsx", index=False)
    print(f"Data lowongan disimpan ke processed_jobs_{cluster_name}.xlsx")
    print(f"Data SFIA disimpan ke processed_sfia_{cluster_name}.xlsx")