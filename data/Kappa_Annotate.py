import pandas as pd
import numpy as np

def normalize(skill):

    return skill.strip().lower().replace('-', ' ').replace('_', ' ').replace('  ', ' ')

def manual_cohen_kappa(rater1_vector, rater2_vector):

    # n_items = a + b + c + d
    n_items = len(rater1_vector)
    if n_items == 0:
        return 1.0

    # agreements = a + d
    agreements = sum(1 for i in range(n_items) if rater1_vector[i] == rater2_vector[i])
    po = agreements / n_items

    rater1_yes = sum(rater1_vector) / n_items
    rater1_no = 1 - rater1_yes
    
    rater2_yes = sum(rater2_vector) / n_items
    rater2_no = 1 - rater2_yes
    
    chance_yes = rater1_yes * rater2_yes
    chance_no = rater1_no * rater2_no
    
    pe = chance_yes + chance_no
    
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0

    kappa = (po - pe) / (1 - pe)
    
    return kappa

def calculate_and_print_kappa(file1, file2, sheet_name):

    print(f"--- Memproses Sheet: {sheet_name} ---")

    if sheet_name == 'Job Posting':
        id_column = 'Nama Pekerjaan'
    elif sheet_name == 'SFIA':
        id_column = 'Skill - Level'
    else:
        print(f"Nama sheet '{sheet_name}' tidak dikenali.")
        return

    try:
        df1 = pd.read_excel(file1, sheet_name=sheet_name)
        df2 = pd.read_excel(file2, sheet_name=sheet_name)
    except Exception as e:
        print(f"Gagal membaca sheet '{sheet_name}'. Pastikan nama sheet dan file sudah benar. Error: {e}")
        return

    on_column = 'Korpus'
    annotation_column = 'Hasil Anotasi Pakar'
    annotator1_name = file1.replace('Anotasi Skill - ', '').replace('.xlsx', '').lower()
    annotator2_name = file2.replace('Anotasi Skill - ', '').replace('.xlsx', '').lower()

    df1 = df1.rename(columns={annotation_column: f'{annotator1_name}_annotations'})
    df2 = df2.rename(columns={annotation_column: f'{annotator2_name}_annotations'})

    df1_subset = df1[[id_column, on_column, f'{annotator1_name}_annotations']].dropna(subset=[on_column])
    df2_subset = df2[[on_column, f'{annotator2_name}_annotations']].dropna(subset=[on_column])
    
    merged_df = pd.merge(df1_subset, df2_subset, on=on_column)

    results = []

    def get_normalized_annotated_skills(annotation):
        if pd.isna(annotation):
            return set()
        return {normalize(s) for s in str(annotation).split('\n')}

    for index, row in merged_df.iterrows():
        corpus_text = row[on_column]
        
        # Korpus dari deskripsi per baris
        items_to_rate = sorted(list(set(normalize(word) for word in str(corpus_text).split())))
        
        # Hasil anotasi per baris
        anot1_skills = get_normalized_annotated_skills(row[f'{annotator1_name}_annotations'])
        anot2_skills = get_normalized_annotated_skills(row[f'{annotator2_name}_annotations'])

        anot1_vector = [1 if item in anot1_skills else 0 for item in items_to_rate]
        anot2_vector = [1 if item in anot2_skills else 0 for item in items_to_rate]
        
        kappa = manual_cohen_kappa(anot1_vector, anot2_vector)

        results.append({
            id_column: row[id_column],
            'kappa_score': kappa
        })

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("Tidak ada data yang cocok untuk diproses.")
        return
        
    avg_kappa = results_df['kappa_score'].mean()

    print(f"\nNilai Rata-rata Kappa: {avg_kappa:.4f}\n")
    print("Rincian Nilai Kappa per Baris (dengan normalisasi):")
    print(results_df.to_string())
    print("-" * 50 + "\n")


file_anot1 = 'Anotasi Skill - Gopal.xlsx'
file_anot2 = 'Anotasi Skill - Rizfi.xlsx'

calculate_and_print_kappa(file_anot1, file_anot2, sheet_name='Job Posting')

calculate_and_print_kappa(file_anot1, file_anot2, sheet_name='SFIA')