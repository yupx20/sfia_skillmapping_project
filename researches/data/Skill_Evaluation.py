import pandas as pd
import numpy as np

def normalize(skill):

    if not isinstance(skill, str):
        return ''
    return skill.strip().lower().replace('-', ' ').replace('_', ' ').replace('  ', ' ')

def process_and_evaluate(df_annotator1, df_annotator2, sheet_name):

    results_data = []

    if sheet_name == 'Job Posting':
        id_col_name = 'Nama Pekerjaan'
    elif sheet_name == 'SFIA':
        id_col_name = 'Skill - Level'
    else:
        id_col_name = 'ID'

    use_index_as_id = False
    if id_col_name not in df_annotator1.columns:
        print(f"Kolom '{id_col_name}' tidak ditemukan. Menggunakan nomor indeks.")
        use_index_as_id = True
        id_col_name = 'ID'

    for index, row1 in df_annotator1.iterrows():
        row2 = df_annotator2.loc[index]

        # Ambil dan normalkan skill
        system_skills = {normalize(s) for s in str(row1['Hasil Ekstraksi Sistem']).split('\n') if s.strip()}
        annotator1_skills = {normalize(s) for s in str(row1['Hasil Anotasi Pakar']).split('\n') if s.strip()}
        annotator2_skills = {normalize(s) for s in str(row2['Hasil Anotasi Pakar']).split('\n') if s.strip()}

        # Buat Ground Truth (Union)
        ground_truth = annotator1_skills.union(annotator2_skills)

        # Hitung metrik
        true_positives = len(system_skills.intersection(ground_truth))
        false_positives = len(system_skills - ground_truth)
        false_negatives = len(ground_truth - system_skills)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        # Ambil nilai identifier
        identifier = row1[id_col_name] if not use_index_as_id else index + 1
            
        # Simpan hasil kalkulasi
        result_dict = {
            id_col_name: identifier,
            'Ground Truth': ', '.join(sorted(list(ground_truth))),
            'TP': true_positives,
            'FP': false_positives,
            'FN': false_negatives,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        }
        results_data.append(result_dict)

    # Buat DataFrame dari hasil
    column_order = [id_col_name, 'Ground Truth', 'TP', 'FP', 'FN', 'Precision', 'Recall', 'F1-Score']
    results_df = pd.DataFrame(results_data, columns=column_order)
    
    # --- Tampilkan Hasil di Konsol ---
    print(f"--- Evaluasi untuk Sheet '{sheet_name}' ---")
    print("\nPerhitungan per Baris:")
    print(results_df)

    # Hitung dan tampilkan rata-rata
    avg_precision = results_df['Precision'].mean()
    avg_recall = results_df['Recall'].mean()
    avg_f1 = results_df['F1-Score'].mean()

    print("\n" + "-"*40)
    print("Rata-rata (Macro Average):")
    print(f"  - Rata-rata Precision: {avg_precision:.4f}")
    print(f"  - Rata-rata Recall:    {avg_recall:.4f}")
    print(f"  - Rata-rata F1-Score:  {avg_f1:.4f}")
    print("="*60 + "\n")
    
    return results_df

anot1_excel_file = 'Anotasi Skill - Gopal.xlsx'
anot2_excel_file = 'Anotasi Skill - Rizfi.xlsx'

anot1_jp_df = pd.read_excel(anot1_excel_file, sheet_name='Job Posting')
anot2_jp_df = pd.read_excel(anot2_excel_file, sheet_name='Job Posting')
anot1_sfia_df = pd.read_excel(anot1_excel_file, sheet_name='SFIA')
anot2_sfia_df = pd.read_excel(anot2_excel_file, sheet_name='SFIA')

results_jp_df = process_and_evaluate(anot1_jp_df, anot2_jp_df, 'Job Posting')
results_sfia_df = process_and_evaluate(anot1_sfia_df, anot2_sfia_df, 'SFIA')

output_filename = 'Evaluasi_Anotasi.xlsx'
print(f"\nMenyimpan hasil ke file '{output_filename}'...")

with pd.ExcelWriter(output_filename) as writer:
    results_jp_df.to_excel(writer, sheet_name='Hasil Job Posting', index=False)
    results_sfia_df.to_excel(writer, sheet_name='Hasil SFIA', index=False)

print(f"File '{output_filename}' berhasil disimpan.")