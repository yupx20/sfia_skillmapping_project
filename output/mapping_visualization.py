import pandas as pd

COSINE_MODELS = [
    "skills_skillner",
    "skills_skillner_keybert",
    "skills_skillner_qe",
    "skills_skillner_qe_keybert"
]

JACCARD_MODELS = [
    "skills_skillner_rake",
    "skills_skillner_yake"
]

# --- TAMBAHAN UNTUK VISUALISASI ---
def generate_visual_sfiascore(mapping_file, sfia_file, output_file):
    mapping_df = pd.read_csv(mapping_file)
    sfia_df = pd.read_csv(sfia_file)

    # --- FIX: Pisahkan Skill dan Level secara aman ---
    split_data = mapping_df['matched_skills'].dropna().apply(lambda x: x.rsplit(' ', 1) if ' ' in x else [x, None])
    split_df = pd.DataFrame(split_data.tolist(), columns=['Skill', 'Level'])
    mapping_df = pd.concat([mapping_df.reset_index(drop=True), split_df], axis=1)
    mapping_df = mapping_df.dropna(subset=['Skill', 'Level'])
    mapping_df['Level'] = pd.to_numeric(mapping_df['Level'], errors='coerce')
    mapping_df = mapping_df.dropna(subset=['Level'])
    mapping_df['Level'] = mapping_df['Level'].astype(int)

    # --- Ambil semua skill dari SFIA agar format meniru data aslinya ---
    all_skills = sfia_df['Skill'].unique()
    columns = ['Skill'] + [f'Level {i}' for i in range(1, 8)]
    visual_df = pd.DataFrame(columns=columns)
    visual_df['Skill'] = all_skills

    # --- Masukkan skor untuk skill-level yang cocok ---
    for _, row in mapping_df.iterrows():
        skill = row['Skill']
        level_col = f'Level {row["Level"]}'
        score = round(row['similarity_score'], 4)
        if skill in visual_df['Skill'].values:
            visual_df.loc[visual_df['Skill'] == skill, level_col] = score
        else:
            # Jika skill hasil mapping tidak ada di sfia_df (jarang terjadi), tambahkan
            new_row = pd.Series({**{'Skill': skill}, **{level_col: score}})
            visual_df = pd.concat([visual_df, pd.DataFrame([new_row])], ignore_index=True)

    # --- Simpan hasil ---
    visual_df.to_csv(output_file, index=False)
    print(f"Visualisasi hasil skor disimpan ke: {output_file}")


cluster_name = "IS"

# Jalankan visualisasi setelah COSINE dan JACCARD mapping
for col in COSINE_MODELS:
    try:
        mapping_file = f"{cluster_name}_New/mapping_cosine_{col}_{cluster_name}.csv"
        sfia_file = f"{cluster_name}_New/skills_extracted_sfia_{cluster_name}.csv"
        output_file = f"{cluster_name}_Visual/visual_mapping_cosine_{col}_{cluster_name}.csv"
        generate_visual_sfiascore(mapping_file, sfia_file, output_file)
    except Exception as e:
        print(f"Error visualisasi cosine '{col}': {e}")

for col in JACCARD_MODELS:
    try:
        mapping_file = f"{cluster_name}_New/mapping_jaccard_{col}_{cluster_name}.csv"
        sfia_file = f"{cluster_name}_New/skills_extracted_sfia_{cluster_name}.csv"
        output_file = f"{cluster_name}_Visual/visual_mapping_jaccard_{col}_{cluster_name}.csv"
        generate_visual_sfiascore(mapping_file, sfia_file, output_file)
    except Exception as e:
        print(f"Error visualisasi jaccard '{col}': {e}")
