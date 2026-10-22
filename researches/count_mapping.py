import pandas as pd
import time

def map_skills_cosine(cluster_name, model_name):

    mapping_df = pd.read_excel(
        f"output/{cluster_name}/mapping_cosine_{model_name}_{cluster_name}.xlsx"
    )
   
    expanded_df = pd.read_excel(
        f"output/{cluster_name}/expanded_mapping_cosine_{model_name}_{cluster_name}.xlsx"
    )

    unique_mapped = mapping_df['matched_skills'].unique().tolist()
    unique_expanded = expanded_df['expanded_matched_skills'].unique().tolist()

    print(f"Cosine mapping '{model_name}' disimpan. ({len(mapping_df)} baris pemetaan).")
    print(f"Hasil sebelum ekspansi : {len(unique_mapped)}, Setelah ekspansi : {len(unique_expanded)}.\n")


if __name__ == '__main__':
    start_time = time.time()
    cluster_name = "IS" # Ganti CS atau IS

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

    print("\nMulai menghitung hasil Mapping COSINE...")
    for col in COSINE_MODELS:
        try:
            map_skills_cosine(cluster_name, col)
        except Exception as e:
            print(f"Error model cosine '{col}': {e}")

    print(f"\nTotal waktu proses: {time.time() - start_time:.2f} detik")