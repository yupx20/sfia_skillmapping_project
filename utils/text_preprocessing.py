import re
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.strip()
    text = re.sub(r'_x000D_[\n\r]*', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)