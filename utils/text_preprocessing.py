import re
# from nltk.corpus import stopwords
# stop_words = set(stopwords.words('english'))

# def preprocess_for_cosine(text):
#     if not isinstance(text, str): return ""
#     text = text.lower()
#     text = re.sub(r"[^a-z\s.,!?\'\"()]", " ", text)
#     text = re.sub(r'\d+', '', text)
#     text = text.strip()
#     text = re.sub(r'_x000D_[\n\r]*', '', text)
#     words = text.split()
#     # words = [word for word in words if word not in stop_words]
#     return ' '.join(words)

# def preprocess_for_jaccard(text):
#     if not isinstance(text, str): return ""
#     text = text.lower()
#     text = re.sub(r'\d+', '', text)
#     text = re.sub(r"[^a-z\s.,!?\'\"()]", " ", text)
#     text = text.replace('\n', ' ').replace('\r', ' ')
#     text = re.sub(r'_x000D_', ' ', text)
#     text = re.sub(r'\s+', ' ', text)
#     return text.strip()

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower() # Case folding
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text) # Menghapus karakter aneh/simbol
    text = re.sub(r'\s+', ' ', text) # Hapus whitespace
    text = re.sub(r'_x000D_', ' ', text) # Hapus sisa artefak excel
    return text.strip()