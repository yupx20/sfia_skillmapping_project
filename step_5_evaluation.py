import pandas as pd
import time
import os

# --- Ground Truth ---
def create_ground_truth(gt_path: str, sheet_name: str) -> set:
    gt_df = pd.read_excel(gt_path, sheet_name=sheet_name)
    gt_df.columns = gt_df.columns.str.strip().str.replace('\n', '', regex=True)
    level_columns = [col for col in gt_df.columns if col.startswith('Level')]

    ground_truth_set = set()
    for _, row in gt_df.iterrows():
        skill_name = row['Skill']
        for col in level_columns:
            if row[col] == 1.0:
                level_number = col.split()[1]
                ground_truth_set.add(f"{skill_name} {level_number}")
    return ground_truth_set

# --- Evaluasi satu model (mengembalikan dataframe) ---
def evaluate_single_mapping(mapping_file: str, gt_set: set,
                            model_name: str, cluster_name: str, expanded: bool = False) -> pd.DataFrame:
    col_name = 'expanded_matched_skills' if expanded else 'matched_skills'
    if not os.path.exists(mapping_file):
        print(f"File tidak ditemukan: {mapping_file}")
        return None

    df = pd.read_csv(mapping_file)
    if col_name not in df.columns:
        print(f"Kolom '{col_name}' tidak ditemukan di {mapping_file}")
        return None

    predicted_set = set(df[col_name].dropna())

    # Hitung metrik
    tp = len(predicted_set & gt_set)
    fp = len(predicted_set - gt_set)
    fn = len(gt_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1_score  = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return pd.DataFrame([{
        'Model': model_name,
        'Cluster': cluster_name,
        'Expanded': expanded,
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'Precision': precision,
        'Recall': recall,
        'F1_score': f1_score
    }])

# --- MAIN ---
if __name__ == '__main__':
    start = time.time()
    cluster_name = "IS"
    gt_file = "data/GT_Pakar1.xlsx"
    gt_sheet = "Ilmu Komputer" if cluster_name == "CS" else "Sistem Informasi"

    COSINE_MODELS = [
        "skills_skillner",
        "skills_skillner_keybert",
        "skills_ner_bert",
        "skills_ner_bert_keybert",
        "skills_skillner_qe",
        "skills_skillner_qe_keybert"
    ]

    JACCARD_MODELS = [
        "skills_skillner_rake",
        "skills_skillner_yake",
        "skills_ner_bert_rake",
        "skills_ner_bert_yake"
    ]

    # Inisialisasi ground truth
    gt_set = create_ground_truth(gt_file, gt_sheet)

    # Simpan semua hasil evaluasi
    all_results = []

    for model in COSINE_MODELS + JACCARD_MODELS:
        sim_type = "cosine" if model in COSINE_MODELS else "jaccard"

        original_path = f"output/{cluster_name}/mapping_{sim_type}_{model}_{cluster_name}.csv"
        expanded_path = f"output/{cluster_name}/expanded_mapping_{sim_type}_{model}_{cluster_name}.csv"

        res_orig = evaluate_single_mapping(original_path, gt_set, model, cluster_name, expanded=False)
        res_expd = evaluate_single_mapping(expanded_path, gt_set, model, cluster_name, expanded=True)

        if res_orig is not None:
            all_results.append(res_orig)
        if res_expd is not None:
            all_results.append(res_expd)

    # Gabungkan semua ke satu DataFrame
    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        out_path = f"output/{cluster_name}/all_evaluation_results_{cluster_name}.csv"
        final_df.to_csv(out_path, index=False)
        print(f"\nSemua hasil evaluasi disimpan ke: {out_path}")
        print(final_df)
    else:
        print("Tidak ada hasil evaluasi yang valid.")

    print(f"\nTotal waktu evaluasi: {time.time() - start:.2f} detik")