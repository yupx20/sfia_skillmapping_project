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

def clean_job_data(filepath):
    df = pd.read_excel(filepath)
    df = df.dropna(subset=['job_description'])
    df = df.drop_duplicates(subset=['job_description'])
    df = df[~df['job_description'].apply(has_experience_requirement)]
    df = df.reset_index(drop=True)
    df['job_id'] = df.index
    return df

if __name__ == '__main__':
    df = clean_job_data('data/CSJobs.xlsx')
    df.to_csv('output/cleaned_CSJobs.csv', index=False)
    print("Data CS cleaned dan disimpan ke cleaned_CSJobs.csv")