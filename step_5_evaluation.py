import pandas as pd
import time

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

def evaluate_single_mapping(mapping_file: str, gt_set: set, model_name: str, cluster_name: str):
    try:
        mapping_df = pd.read_csv(mapping_file)
        predicted_set = set(mapping_df['matched_skills'].dropna())

        tp = len(predicted_set.intersection(gt_set))
        fp = len(predicted_set - gt_set)
        fn = len(gt_set - predicted_set)
        tn = 0  # Not computable

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        return {
            'Model': model_name,
            'Cluster': cluster_name,
            'TP': tp,
            'FP': fp,
            'FN': fn,
            'Precision': precision,
            'Recall': recall,
            'F1_score': f1_score
        }
    except Exception as e:
        print(f"❌ Gagal evaluasi {model_name}: {e}")
        return {
            'Model': model_name,
            'Cluster': cluster_name,
            'TP': 0, 'FP': 0, 'FN': 0,
            'Precision': 0.0, 'Recall': 0.0, 'F1_score': 0.0
        }

if __name__ == '__main__':
    start = time.time()
    cluster_name = "CS"
    gt_file = "GT_Pakar1.xlsx"
    gt_sheet = "Ilmu Komputer"
    gt_set = create_ground_truth(gt_file, gt_sheet)

    COSINE_MODELS = [
        "skills_skillner_tfidf",
        "skills_skillner_keybert_tfidf",
        "skills_ner_bert_tfidf",
        "skills_ner_bert_keybert_tfidf",
        "skills_skillner_qe_tfidf",
        "skills_skillner_qe_keybert_tfidf"
    ]

    JACCARD_MODELS = [
        "skills_skillner_rake",
        "skills_skillner_yake",
        "skills_ner_bert_rake",
        "skills_ner_bert_yake"
    ]

    print("\n🚀 Evaluasi COSINE Models")
    all_results = []
    for model in COSINE_MODELS:
        path = f"mapping_cosine_{model}_{cluster_name}.csv"
        result = evaluate_single_mapping(path, gt_set, model, cluster_name)
        all_results.append(result)

    print("\n🚀 Evaluasi JACCARD Models")
    for model in JACCARD_MODELS:
        path = f"mapping_jaccard_{model}_{cluster_name}.csv"
        result = evaluate_single_mapping(path, gt_set, model, cluster_name)
        all_results.append(result)

    # Simpan seluruh hasil evaluasi
    pd.DataFrame(all_results).to_csv(f"evaluation_all_models_{cluster_name}.csv", index=False)
    print(f"\n📊 Semua hasil evaluasi disimpan ke evaluation_all_models_{cluster_name}.csv")
    print(f"⏱️ Total waktu evaluasi: {time.time() - start:.2f} detik")
