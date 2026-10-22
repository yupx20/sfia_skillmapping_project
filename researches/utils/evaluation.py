import pandas as pd

def evaluate_predictions(predictions_df, ground_truth_set, jobs_df):
    eval_rows = []
    for job_id in jobs_df['job_id']:
        keyword = jobs_df.loc[job_id, 'keyword']
        predicted_skills = set(predictions_df[predictions_df['job_id'] == job_id]['predicted_sfia_skill_level'])
        if not predicted_skills:
            precision, recall, f1 = 0, 0, 0
        else:
            true_positives = len(predicted_skills.intersection(ground_truth_set))
            precision = true_positives / len(predicted_skills)
            recall = true_positives / len(ground_truth_set) if len(ground_truth_set) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        eval_rows.append({'job_id': job_id, 'keyword': keyword, 'precision': precision, 'recall': recall, 'f1_score': f1})
    return pd.DataFrame(eval_rows)