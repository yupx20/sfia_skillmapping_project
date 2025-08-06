import pandas as pd
from sklearn.metrics import cohen_kappa_score
from sklearn.preprocessing import MultiLabelBinarizer

def convert_multiline_to_list(text):
    if isinstance(text, str):
        return [line.strip().lower() for line in text.strip().split('\n') if line.strip()]
    return []

def convert_list_to_multiline(skill_list):
    return '\n'.join(skill_list)

def process_sheet(sheet_name, df1, df2, kappa_threshold=0.6):
    df1['skills'] = df1['Hasil Anotasi Pakar'].apply(convert_multiline_to_list)
    df2['skills'] = df2['Hasil Anotasi Pakar'].apply(convert_multiline_to_list)

    all_skills = df1['skills'].tolist() + df2['skills'].tolist()
    mlb = MultiLabelBinarizer()
    mlb.fit(all_skills)

    bin1 = mlb.transform(df1['skills'])
    bin2 = mlb.transform(df2['skills'])

    skill_kappas = {}
    for i, skill in enumerate(mlb.classes_):
        kappa = cohen_kappa_score(bin1[:, i], bin2[:, i])
        skill_kappas[skill] = kappa

    selected_skills = {skill for skill, kappa in skill_kappas.items() if kappa >= kappa_threshold}
    print(f"[{sheet_name}] Jumlah skill dengan Kappa ≥ {kappa_threshold}: {len(selected_skills)}")

    filtered_skills_per_row = []
    for skill_list in df1['skills']:
        filtered = [skill for skill in skill_list if skill in selected_skills]
        filtered_skills_per_row.append(convert_list_to_multiline(filtered))

    df1 = df1.copy()
    df1['Hasil Anotasi Pakar'] = filtered_skills_per_row
    df1.drop(columns=['skills'], inplace=True)
    return df1, pd.DataFrame(skill_kappas.items(), columns=['Skill', 'Kappa'])

file1_path = 'Anotasi Skill - Rizfi.xlsx'
file2_path = 'Anotasi Skill - Gopal.xlsx'
output_file = 'Anotasi Skill (Kappa).xlsx'
kappa_threshold = 0.5
sheet_names = ['Job Posting', 'SFIA']

with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    for sheet_name in sheet_names:

        df1 = pd.read_excel(file1_path, sheet_name=sheet_name)
        df2 = pd.read_excel(file2_path, sheet_name=sheet_name)

        processed_df, kappa_df = process_sheet(sheet_name, df1, df2, kappa_threshold)

        processed_df.to_excel(writer, sheet_name=sheet_name, index=False)

        kappa_df.to_excel(writer, sheet_name=f"{sheet_name} Kappa", index=False)

print(f"File berhasil disimpan sebagai: {output_file}")
