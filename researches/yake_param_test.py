from rake_nltk import Rake
import yake
import re
import nltk

nltk.download('stopwords')
yake_extractor = yake.KeywordExtractor(lan="en", n=3, dedupLim=0.2, dedupFunc='seqm', features=None) # dedupLim = 0.2 yang terbaik sejauh ini untuk ekstraksi penelitian ini.
rake_extractor = Rake()

def preprocess_for_jaccard(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.replace('\n', ' ').replace('\r', ' ')
    text = re.sub(r'_x000D_', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_rake_keywords_list(text):
    if not isinstance(text, str) or not text.strip():
        return []
    rake_extractor.extract_keywords_from_text(text)
    keywords = rake_extractor.get_ranked_phrases()

    return [kw.strip().lower() for kw in keywords]

def extract_yake_keywords(text):
    if not isinstance(text, str): return []
    keywords = yake_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]


text = """
Perform basic statistical and data analysis to identify trends and patterns in data.
Support the development and maintenance of data reports and dashboards.
Support ad-hoc data requests from different teams.
Contribute to data quality initiatives and ensure data accuracy.
Manage data modeling design, writing, and optimizing ETL jobs.
Collaborate with the business and product team to build data metrics based on the data warehouse.
Other duties may be assigned by the Company from time to time.
Bachelor's degree in a quantitative field such as Statistics, Mathematics, Computer Science, or a related discipline.
Fresh Graduate - experience for 1 year as analyst data
Proficiency in SQL for query and data manipulation
Experience with BI tools (e.g., Looker Studio, Tableau, Power BI).
Experience with spreadsheet software (e.g., Excel, Google Sheets).
Basic knowledge on Python programming.
Familiarity with cloud-based data platforms (e.g., Bigquery).
Familiarity with ETL tools, especially Airflow, is a plus.
Strong analytical and problem-solving skills.
Excellent attention to detail and a commitment to data accuracy.
Good communication and collaboration skills.
"""

text = preprocess_for_jaccard(text)
yake_keywords = extract_yake_keywords(text)
print(yake_keywords)
