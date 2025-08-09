import pandas as pd
import numpy as np

gopal_excel_file = 'Anotasi Skill - Gopal.xlsx'
rizfi_excel_file = 'Anotasi Skill - Rizfi.xlsx'

gopal_job_posting_df = pd.read_excel(gopal_excel_file, sheet_name='Job Posting')
rizfi_job_posting_df = pd.read_excel(rizfi_excel_file, sheet_name='Job Posting')

gopal_sfia_df = pd.read_excel(gopal_excel_file, sheet_name='SFIA')
rizfi_sfia_df = pd.read_excel(rizfi_excel_file, sheet_name='SFIA')

def manual_cohen_kappa(rater1, rater2):

    # Membangun matriks kontingensi 2x2
    # a: Penilai 1 = Ada (1), Penilai 2 = Ada (1)
    # b: Penilai 1 = Ada (1), Penilai 2 = Tidak Ada (0)
    # c: Penilai 1 = Tidak Ada (0), Penilai 2 = Ada (1)
    # d: Penilai 1 = Tidak Ada (0), Penilai 2 = Tidak Ada (0)
    a = np.sum((rater1 == 1) & (rater2 == 1))
    b = np.sum((rater1 == 1) & (rater2 == 0))
    c = np.sum((rater1 == 0) & (rater2 == 1))
    d = np.sum((rater1 == 0) & (rater2 == 0))
    
    total = a + b + c + d
    if total == 0:
        return 1.0 # Jika tidak ada data, asumsikan kesepakatan sempurna
    
    # Hitung Observed Agreement (po)
    po = (a + d) / total
    
    # Hitung Expected Agreement (pe)
    p_rater1_yes = (a + b) / total
    p_rater2_yes = (a + c) / total
    pe_yes = p_rater1_yes * p_rater2_yes
    
    p_rater1_no = (c + d) / total
    p_rater2_no = (b + d) / total
    pe_no = p_rater1_no * p_rater2_no
    
    pe = pe_yes + pe_no
    
    # Hitung Kappa
    # Menangani kasus di mana pe = 1 (kesepakatan yang diharapkan sempurna)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
        
    kappa = (po - pe) / (1 - pe)
    
    return kappa


def calculate_kappa_manually(df1, df2, column_name):

    annotations1 = df1[column_name].astype(str).str.lower().str.split('\n')
    annotations2 = df2[column_name].astype(str).str.lower().str.split('\n')

    # Buat vocabulary
    all_skills = set()
    for skills_list in annotations1:
        all_skills.update(s for s in skills_list if s)
    for skills_list in annotations2:
        all_skills.update(s for s in skills_list if s)
    
    all_skills = sorted(list(all_skills))

    # Buat representasi biner
    binary_annotations1 = []
    for skills_list in annotations1:
        binary_annotations1.append([1 if skill in skills_list else 0 for skill in all_skills])

    binary_annotations2 = []
    for skills_list in annotations2:
        binary_annotations2.append([1 if skill in skills_list else 0 for skill in all_skills])

    # Ubah ke format NumPy dan transposisi
    binary_annotations1_np = np.array(binary_annotations1).T
    binary_annotations2_np = np.array(binary_annotations2).T
    
    # Hitung Kappa untuk setiap skill menggunakan fungsi manual
    kappa_scores = []
    for i in range(len(all_skills)):
        score = manual_cohen_kappa(binary_annotations1_np[i], binary_annotations2_np[i])
        kappa_scores.append(score)
        
    # Kembalikan rata-rata dari semua skor Kappa
    return np.mean(kappa_scores)


kappa_job_posting = calculate_kappa_manually(gopal_job_posting_df, rizfi_job_posting_df, 'Hasil Anotasi Pakar')
kappa_sfia = calculate_kappa_manually(gopal_sfia_df, rizfi_sfia_df, 'Hasil Anotasi Pakar')

print(f"Nilai Kappa (Manual) untuk sheet 'Job Posting': {kappa_job_posting}")
print(f"Nilai Kappa (Manual) untuk sheet 'SFIA': {kappa_sfia}")