import pandas as pd

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

# --- TAMBAHAN UNTUK VISUALISASI ---
def generate_visual_sfiascore(mapping_file, sfia_file, output_file):
    mapping_df = pd.read_excel(mapping_file)
    sfia_df = pd.read_excel(sfia_file)

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
    visual_df.to_excel(output_file, index=False)
    print(f"Visualisasi hasil skor disimpan ke: {output_file}")


cluster_name = "IS" # Ubah CS atau IS

for col in COSINE_MODELS:
    try:
        mapping_file = f"{cluster_name}/mapping_cosine_{col}_{cluster_name}.xlsx"
        sfia_file = f"{cluster_name}/skills_extracted_sfia_{cluster_name}.xlsx"
        output_file = f"{cluster_name}_SFIAFormat/sfiaformat_mapping_cosine_{col}_{cluster_name}.xlsx"
        generate_visual_sfiascore(mapping_file, sfia_file, output_file)
    except Exception as e:
        print(f"Error visualisasi cosine '{col}': {e}")
