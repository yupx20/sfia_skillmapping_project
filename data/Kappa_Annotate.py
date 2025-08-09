import pandas as pd
import numpy as np

gopal_excel_file = 'Anotasi Skill - Gopal.xlsx'
rizfi_excel_file = 'Anotasi Skill - Rizfi.xlsx'

gopal_job_posting_df = pd.read_excel(gopal_excel_file, sheet_name='Job Posting')
rizfi_job_posting_df = pd.read_excel(rizfi_excel_file, sheet_name='Job Posting')

gopal_sfia_df = pd.read_excel(gopal_excel_file, sheet_name='SFIA')
rizfi_sfia_df = pd.read_excel(rizfi_excel_file, sheet_name='SFIA')

def manual_cohen_kappa(rater1, rater2):
    a = np.sum((rater1 == 1) & (rater2 == 1))
    b = np.sum((rater1 == 1) & (rater2 == 0))
    c = np.sum((rater1 == 0) & (rater2 == 1))
    d = np.sum((rater1 == 0) & (rater2 == 0))
    
    total = a + b + c + d
    if total == 0: return 1.0
    
    po = (a + d) / total
    p_rater1_yes = (a + b) / total
    p_rater2_yes = (a + c) / total
    p_rater1_no = (c + d) / total
    p_rater2_no = (b + d) / total
    pe = (p_rater1_yes * p_rater2_yes) + (p_rater1_no * p_rater2_no)
    
    if pe == 1.0: return 1.0 if po == 1.0 else 0.0
    kappa = (po - pe) / (1 - pe)
    return kappa

# def calculate_kappa_per_skill(df1, df2, column_name):

#     annotations1 = df1[column_name].astype(str).str.lower().str.split('\n')
#     annotations2 = df2[column_name].astype(str).str.lower().str.split('\n')

#     all_skills = set()
#     for skills_list in annotations1: all_skills.update(s for s in skills_list if s)
#     for skills_list in annotations2: all_skills.update(s for s in skills_list if s)
#     all_skills = sorted(list(all_skills))

#     binary_annotations1_np = np.array([[1 if skill in sl else 0 for sl in annotations1] for skill in all_skills])
#     binary_annotations2_np = np.array([[1 if skill in sl else 0 for sl in annotations2] for skill in all_skills])
    
#     kappa_details = []
#     for i, skill in enumerate(all_skills):
#         score = manual_cohen_kappa(binary_annotations1_np[i], binary_annotations2_np[i])
#         kappa_details.append((skill, score))
        
#     avg_kappa = np.mean([score for _, score in kappa_details])
#     kappa_details.sort(key=lambda x: x[1], reverse=True)
#     return avg_kappa, kappa_details

def calculate_kappa_per_row(df1, df2, annot_col, id_col):

    all_skills = set()
    for skills_list in df1[annot_col].astype(str).str.lower().str.split('\n'): all_skills.update(s for s in skills_list if s)
    for skills_list in df2[annot_col].astype(str).str.lower().str.split('\n'): all_skills.update(s for s in skills_list if s)
    all_skills = sorted(list(all_skills))
    
    kappa_details = []
    for index, row1 in df1.iterrows():
        row2 = df2.loc[index]
        skills1 = set(str(row1[annot_col]).lower().strip().split('\n'))
        skills2 = set(str(row2[annot_col]).lower().strip().split('\n'))
        
        vector1 = np.array([1 if skill in skills1 else 0 for skill in all_skills])
        vector2 = np.array([1 if skill in skills2 else 0 for skill in all_skills])
        
        score = manual_cohen_kappa(vector1, vector2)
        row_id = row1[id_col]
        kappa_details.append((row_id, score))

    avg_kappa = np.mean([score for _, score in kappa_details])
    kappa_details.sort(key=lambda x: x[1], reverse=True)
    return avg_kappa, kappa_details

# avg_jp_skill, details_jp_skill = calculate_kappa_per_skill(gopal_job_posting_df, rizfi_job_posting_df, 'Hasil Anotasi Pakar')
# print(f"\n>>> HASIL METODE PER-SKILL (Validasi Skema Anotasi) <<<")
# print(f"Rata-rata Kappa: {avg_jp_skill:.4f}")
# print("--- Rincian Kappa per Skill ---")
# for skill, score in details_jp_skill:
#     print(f"Kappa = {score:.4f} | Skill: {skill}")

avg_jp_row, details_jp_row = calculate_kappa_per_row(gopal_job_posting_df, rizfi_job_posting_df, 'Hasil Anotasi Pakar', 'Nama Pekerjaan')
print(f"\n\n>>> HASIL KAPPA PER-BARIS <<<")
print(f"Rata-rata Kappa: {avg_jp_row:.4f}")
print("--- Rincian Kappa per Baris ---")
for job, score in details_jp_row:
    print(f"Kappa = {score:.4f} | Nama Pekerjaan: {job}")

print("\n\n")

# avg_sfia_skill, details_sfia_skill = calculate_kappa_per_skill(gopal_sfia_df, rizfi_sfia_df, 'Hasil Anotasi Pakar')
# print(f"\n>>> HASIL METODE PER-SKILL (Validasi Skema Anotasi) <<<")
# print(f"Rata-rata Kappa: {avg_sfia_skill:.4f}")
# print("--- Rincian Kappa per Skill ---")
# for skill, score in details_sfia_skill:
#     print(f"Kappa = {score:.4f} | Skill: {skill}")

avg_sfia_row, details_sfia_row = calculate_kappa_per_row(gopal_sfia_df, rizfi_sfia_df, 'Hasil Anotasi Pakar', 'Skill - Level')
print(f"\n\n>>> HASIL KAPPA PER-BARIS <<<")
print(f"Rata-rata Kappa: {avg_sfia_row:.4f}")
print("--- Rincian Kappa per Baris ---")
for skill_level, score in details_sfia_row:
    print(f"Kappa = {score:.4f} | Skill - Level: {skill_level}")