import pandas as pd
import re

def has_experience_requirement(text):
    if not isinstance(text, str): return False
    text = text.lower()
    patterns = [
        r'(\d+\s*(\+)?\s*(tahun|thn|year)s?(\s*of)?\s*experience)',
        r'(minimal|min)\s*\d+\s*(tahun|thn|year)s?',
        r'experience\s*\d+\s*-\s*\d+\s*years',
        r'at\s*least\s*\d+\s*year'
    ]
    return any(re.search(pattern, text) for pattern in patterns)

def clean_job_data(filepath, cluster_name: str):
    df = pd.read_excel(filepath)
    initial_count = len(df)

    df = df.dropna(subset=['job_description'])
    after_dropna = len(df)

    df = df.drop_duplicates(subset=['job_description'])
    after_dedup = len(df)

    df = df[~df['job_description'].apply(has_experience_requirement)]
    after_experience_filter = len(df)

    df = df.reset_index(drop=True)
    df['job_id'] = df.index

    print(f"Statistik pembersihan data {cluster_name}:")
    print(f" - Jumlah awal: {initial_count}")
    print(f" - Setelah drop NA: {after_dropna}")
    print(f" - Setelah hapus duplikat: {after_dedup}")
    print(f" - Setelah filter pengalaman: {after_experience_filter}")

    return df

if __name__ == '__main__':
    cluster_name = "IS"
    df = clean_job_data(f'data/{cluster_name}Jobs.xlsx', cluster_name)
    df.to_csv(f'output/{cluster_name}/cleaned_{cluster_name}Jobs.csv', index=False)
    print(f"Data {cluster_name} cleaned dan disimpan ke cleaned_{cluster_name}Jobs.csv")