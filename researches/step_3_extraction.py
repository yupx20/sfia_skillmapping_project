import pandas as pd
import time
from utils.skill_extraction import (
    extract_skills_skillner,
    extract_skills_skillner_qe,
    extract_keybert_keywords,
    extract_rake_keywords,
    extract_yake_keywords
)

start_time = time.time()

def count_skills_per_row(skills_column):
    return skills_column.apply(lambda x: len(x) if isinstance(x, list) else 0)

def extract_all_skills(job_file: str, sfia_file: str):
    jobs_df = pd.read_excel(job_file)
    sfia_df = pd.read_excel(sfia_file)

    # === Ekstraksi JOBS ===
    print("Mengekstraksi keterampilan dari data lowongan...")
    jobs_df['SkillNER'] = jobs_df['job_description_cleaned'].apply(extract_skills_skillner)
    jobs_df['SkillNER QE'] = jobs_df['job_description_cleaned'].apply(extract_skills_skillner_qe)

    jobs_df['RAKE'] = jobs_df['job_description_cleaned'].apply(extract_rake_keywords)
    jobs_df['YAKE'] = jobs_df['job_description_cleaned'].apply(extract_yake_keywords)

    jobs_df['SkillNER RAKE'] = jobs_df.apply(
        lambda row: list(set(row['SkillNER']).union(row['RAKE'])),
        axis=1)
    jobs_df['SkillNER YAKE'] = jobs_df.apply(
        lambda row: list(set(row['SkillNER']).union(row['YAKE'])),
        axis=1)

    jobs_df['SkillNER_KeyBERT'] = jobs_df['SkillNER'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    jobs_df['SkillNER QE_KeyBERT'] = jobs_df['SkillNER QE'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    jobs_df['SkillNER RAKE_KeyBERT'] = jobs_df['SkillNER RAKE'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    jobs_df['SkillNER YAKE_KeyBERT'] = jobs_df['SkillNER YAKE'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    
    # === Ekstraksi SFIA ===
    print("Mengekstraksi keterampilan dari data SFIA...")
    sfia_df['SkillNER'] = sfia_df['Level_Description_cleaned'].apply(extract_skills_skillner)
    sfia_df['SkillNER QE'] = sfia_df['Level_Description_cleaned'].apply(extract_skills_skillner_qe)

    sfia_df['RAKE'] = sfia_df['Level_Description_cleaned'].apply(extract_rake_keywords)
    sfia_df['YAKE'] = sfia_df['Level_Description_cleaned'].apply(extract_yake_keywords)

    sfia_df['SkillNER RAKE'] = sfia_df.apply(
        lambda row: list(set(row['SkillNER']).union(row['RAKE'])),
        axis=1)
    sfia_df['SkillNER YAKE'] = sfia_df.apply(
        lambda row: list(set(row['SkillNER']).union(row['YAKE'])),
        axis=1)

    sfia_df['SkillNER_KeyBERT'] = sfia_df['SkillNER'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    sfia_df['SkillNER QE_KeyBERT'] = sfia_df['SkillNER QE'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    sfia_df['SkillNER RAKE_KeyBERT'] = sfia_df['SkillNER RAKE'].apply(lambda s: extract_keybert_keywords(" ".join(s)))
    sfia_df['SkillNER YAKE_KeyBERT'] = sfia_df['SkillNER YAKE'].apply(lambda s: extract_keybert_keywords(" ".join(s)))

    # === Tambahkan nama Skill SFIA ke setiap list hasil ekstraksi SFIA ===
    skill_columns = [
        'SkillNER',
        'SkillNER QE',
        'SkillNER RAKE',
        'SkillNER YAKE',
        'SkillNER_KeyBERT',
        'SkillNER QE_KeyBERT',
        'SkillNER RAKE_KeyBERT',
        'SkillNER YAKE_KeyBERT'
    ]
    for col in skill_columns:
        sfia_df[col] = sfia_df.apply(
            lambda row: row[col] + [row['Skill']] if isinstance(row[col], list) else [row['Skill']],
            axis=1
        )

    # Tambahkan kolom jumlah keterampilan per baris dan simpan statistik
    jobs_stats = []
    sfia_stats = []

    print("\nStatistik jumlah keterampilan per job (lowongan):")
    for col in skill_columns:
        count_col = col + '_count'
        jobs_df[count_col] = count_skills_per_row(jobs_df[col])
        stats = {
            'Model': col,
            'Mean': jobs_df[count_col].mean(),
            'Min': jobs_df[count_col].min(),
            'Max': jobs_df[count_col].max(),
            'Non-zero Count': (jobs_df[count_col] > 0).sum()
        }
        jobs_stats.append(stats)
        print(f"  - {col}:")
        print(f"      Mean: {stats['Mean']:.2f}, Min: {stats['Min']}, Max: {stats['Max']}, Non-zero: {stats['Non-zero Count']}")

    print("\nStatistik jumlah keterampilan per deskripsi SFIA:")
    for col in skill_columns:
        count_col = col + '_count'
        sfia_df[count_col] = count_skills_per_row(sfia_df[col])
        stats = {
            'Model': col,
            'Mean': sfia_df[count_col].mean(),
            'Min': sfia_df[count_col].min(),
            'Max': sfia_df[count_col].max(),
            'Non-zero Count': (sfia_df[count_col] > 0).sum()
        }
        sfia_stats.append(stats)
        print(f"  - {col}:")
        print(f"      Mean: {stats['Mean']:.2f}, Min: {stats['Min']}, Max: {stats['Max']}, Non-zero: {stats['Non-zero Count']}")

    # === SIMPAN STATISTIK KE FILE TERPISAH ===
    cluster_name = job_file.split('/')[-1].split('_')[2].split('.')[0]  # ambil 'CS' atau 'IS' dari nama file
    stats_output_path = f"output/{cluster_name}/skill_count_statistics_{cluster_name}.xlsx"
    with pd.ExcelWriter(stats_output_path, engine='openpyxl') as writer:
        pd.DataFrame(jobs_stats).to_excel(writer, sheet_name="Jobs", index=False)
        pd.DataFrame(sfia_stats).to_excel(writer, sheet_name="SFIA", index=False)
    print(f"\nStatistik jumlah keterampilan disimpan ke: {stats_output_path}")

    return jobs_df, sfia_df

# --- MAIN ---
if __name__ == '__main__':
    cluster_name = "CS"
    jobs_df, sfia_df = extract_all_skills(f"output/{cluster_name}/processed_jobs_{cluster_name}.xlsx", f"output/{cluster_name}/processed_sfia_{cluster_name}.xlsx")

    jobs_df.to_excel(f"output/{cluster_name}/skills_extracted_jobs_{cluster_name}.xlsx", index=False)
    sfia_df.to_excel(f"output/{cluster_name}/skills_extracted_sfia_{cluster_name}.xlsx", index=False)
    print(f"\nHasil ekstraksi disimpan ke:")
    print(f"   - output/{cluster_name}/skills_extracted_jobs_{cluster_name}.xlsx")
    print(f"   - output/{cluster_name}/skills_extracted_sfia_{cluster_name}.xlsx")

    cluster_name = "IS"
    jobs_df, sfia_df = extract_all_skills(f"output/{cluster_name}/processed_jobs_{cluster_name}.xlsx", f"output/{cluster_name}/processed_sfia_{cluster_name}.xlsx")

    jobs_df.to_excel(f"output/{cluster_name}/skills_extracted_jobs_{cluster_name}.xlsx", index=False)
    sfia_df.to_excel(f"output/{cluster_name}/skills_extracted_sfia_{cluster_name}.xlsx", index=False)
    print(f"\nHasil ekstraksi disimpan ke:")
    print(f"   - output/{cluster_name}/skills_extracted_jobs_{cluster_name}.xlsx")
    print(f"   - output/{cluster_name}/skills_extracted_sfia_{cluster_name}.xlsx")

    end_time = time.time()
    print(f"\nWaktu yang dibutuhkan: {end_time - start_time:.2f} detik")