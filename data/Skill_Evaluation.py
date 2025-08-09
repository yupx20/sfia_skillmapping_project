import pandas as pd
import numpy as np

gopal_excel_file = 'Anotasi Skill - Gopal.xlsx'
rizfi_excel_file = 'Anotasi Skill - Rizfi.xlsx'

gopal_jp_df = pd.read_excel(gopal_excel_file, sheet_name='Job Posting')
rizfi_jp_df = pd.read_excel(rizfi_excel_file, sheet_name='Job Posting')

gopal_sfia_df = pd.read_excel(gopal_excel_file, sheet_name='SFIA')
rizfi_sfia_df = pd.read_excel(rizfi_excel_file, sheet_name='SFIA')

def normalize(skill):
    return skill.strip().lower().replace('-', ' ').replace('_', ' ').replace('  ', ' ')

def evaluate_system(df_annotator1, df_annotator2, strategy='intersection'):

    list_precision = []
    list_recall = []
    list_f1 = []

    for index, row1 in df_annotator1.iterrows():
        row2 = df_annotator2.loc[index]

        system_skills = {normalize(s) for s in str(row1['Hasil Ekstraksi Sistem']).strip().split('\n')}
        annotator1_skills = {normalize(s) for s in str(row1['Hasil Anotasi Pakar']).strip().split('\n')}
        annotator2_skills = {normalize(s) for s in str(row2['Hasil Anotasi Pakar']).strip().split('\n')}

        for s in [system_skills, annotator1_skills, annotator2_skills]:
            s.discard('')

        if strategy == 'intersection':
            ground_truth = annotator1_skills.intersection(annotator2_skills)
        elif strategy == 'union':
            ground_truth = annotator1_skills.union(annotator2_skills)
        else:
            raise ValueError("Strategi harus 'intersection' atau 'union'")

        # Hitung TP, FP, FN
        true_positives = len(system_skills.intersection(ground_truth))
        false_positives = len(system_skills - ground_truth)
        false_negatives = len(ground_truth - system_skills)

        # Hitung Precision, Recall, F1-Score untuk baris ini
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        list_precision.append(precision)
        list_recall.append(recall)
        list_f1.append(f1_score)

    # Menghitung rata-rata skor dari semua baris (Macro Average)
    avg_scores = {
        'Precision': np.mean(list_precision),
        'Recall': np.mean(list_recall),
        'F1-Score': np.mean(list_f1)
    }
    
    return avg_scores

print("--- Evaluasi untuk Sheet 'Job Posting' ---")
# Intersection
scores_jp_inter = evaluate_system(gopal_jp_df, rizfi_jp_df, strategy='intersection')
print(f"Hasil (Strategi Irisan):")
print(f"   - Precision: {scores_jp_inter['Precision']:.4f}")
print(f"   - Recall:    {scores_jp_inter['Recall']:.4f}")
print(f"   - F1-Score:  {scores_jp_inter['F1-Score']:.4f}\n")

# Union
scores_jp_union = evaluate_system(gopal_jp_df, rizfi_jp_df, strategy='union')
print(f"Hasil (Strategi Gabungan):")
print(f"   - Precision: {scores_jp_union['Precision']:.4f}")
print(f"   - Recall:    {scores_jp_union['Recall']:.4f}")
print(f"   - F1-Score:  {scores_jp_union['F1-Score']:.4f}")

print("\n" + "="*40 + "\n")

print("--- Evaluasi untuk Sheet 'SFIA' ---")
# Intersection
scores_sfia_inter = evaluate_system(gopal_sfia_df, rizfi_sfia_df, strategy='intersection')
print(f"Hasil (Strategi Irisan):")
print(f"   - Precision: {scores_sfia_inter['Precision']:.4f}")
print(f"   - Recall:    {scores_sfia_inter['Recall']:.4f}")
print(f"   - F1-Score:  {scores_sfia_inter['F1-Score']:.4f}\n")

# Union
scores_sfia_union = evaluate_system(gopal_sfia_df, rizfi_sfia_df, strategy='union')
print(f"Hasil (Strategi Gabungan):")
print(f"   - Precision: {scores_sfia_union['Precision']:.4f}")
print(f"   - Recall:    {scores_sfia_union['Recall']:.4f}")
print(f"   - F1-Score:  {scores_sfia_union['F1-Score']:.4f}")