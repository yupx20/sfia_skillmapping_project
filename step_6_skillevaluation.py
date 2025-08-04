import pandas as pd

def convert_to_list(text):
    if isinstance(text, str):
        return [line.strip() for line in text.strip().split('\n') if line.strip()]
    return []

def normalize(skill):
    return skill.strip().lower().replace('-', ' ').replace('_', ' ').replace('  ', ' ')

def compute_metrics(system_skills_str, expert_skills_str):
    try:
        predicted_list = convert_to_list(system_skills_str)
        true_list = convert_to_list(expert_skills_str)

        predicted_set = set([normalize(s) for s in predicted_list])
        true_set = set([normalize(s) for s in true_list])

        tp = len(predicted_set & true_set)
        fp = len(predicted_set - true_set)
        fn = len(true_set - predicted_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return tp, fp, fn, precision, recall, f1
    except:
        return 0, 0, 0, 0.0, 0.0, 0.0

def process_sheet(df):
    df['TP'] = 0
    df['FP'] = 0
    df['FN'] = 0
    df['Precision'] = 0.0
    df['Recall'] = 0.0
    df['F1-Score'] = 0.0

    for idx, row in df.iterrows():
        tp, fp, fn, precision, recall, f1 = compute_metrics(
            row['Hasil Ekstraksi Sistem'], row['Hasil Anotasi Pakar']
        )
        df.at[idx, 'TP'] = tp
        df.at[idx, 'FP'] = fp
        df.at[idx, 'FN'] = fn
        df.at[idx, 'Precision'] = round(precision, 4)
        df.at[idx, 'Recall'] = round(recall, 4)
        df.at[idx, 'F1-Score'] = round(f1, 4)

    return df

def get_summary(df, sheet_name):
    return {
        "Sheet": sheet_name,
        "Total TP": df['TP'].sum(),
        "Total FP": df['FP'].sum(),
        "Total FN": df['FN'].sum(),
        "Avg Precision": round(df['Precision'].mean(), 4),
        "Avg Recall": round(df['Recall'].mean(), 4),
        "Avg F1-Score": round(df['F1-Score'].mean(), 4)
    }


file_path = 'data/Anotasi Skill (Kappa).xlsx'

xls = pd.ExcelFile(file_path)
job_posting_df = xls.parse("Job Posting")
sfia_df = xls.parse("SFIA")

job_posting_df = process_sheet(job_posting_df)
sfia_df = process_sheet(sfia_df)

summary_data = []
summary_data.append(get_summary(job_posting_df, "Job Posting"))
summary_data.append(get_summary(sfia_df, "SFIA"))
summary_df = pd.DataFrame(summary_data)

with pd.ExcelWriter(file_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
    job_posting_df.to_excel(writer, sheet_name="Job Posting", index=False)
    sfia_df.to_excel(writer, sheet_name="SFIA", index=False)
    summary_df.to_excel(writer, sheet_name="Rekap Evaluasi", index=False)

print("Evaluasi selesai. Hasil per baris dan rekap ditulis ke sheet 'Rekap Evaluasi'.")
