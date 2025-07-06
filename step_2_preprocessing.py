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

def preprocess_jobs_and_sfia(job_file: str, sfia_file: str, cluster_name: str):
    # Load cleaned job data
    jobs_df = pd.read_csv(job_file)

    # Translate & clean job descriptions
    print("Menerjemahkan dan membersihkan deskripsi lowongan...")
    jobs_df['job_description_cleaned'] = jobs_df['job_description'].apply(translate_text).apply(preprocess_text)

    # Process SFIA
    print("Mengubah dan membersihkan deskripsi SFIA...")
    sfia_df = transform_sfia_to_long_format(sfia_file)
    sfia_df['Level_Description_cleaned'] = sfia_df['Level_Description'].apply(preprocess_text)

    # Save output
    jobs_df.to_csv(f"processed_jobs_{cluster_name}.csv", index=False)
    sfia_df.to_csv(f"processed_sfia_{cluster_name}.csv", index=False)
    print(f"Data lowongan disimpan ke processed_jobs_{cluster_name}.csv")
    print(f"Data SFIA disimpan ke processed_sfia_{cluster_name}.csv")

if __name__ == '__main__':
    preprocess_jobs_and_sfia("output/cleaned_CSJobs.csv", "data/sfia9_en2025.xlsx", "CS")