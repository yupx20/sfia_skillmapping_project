import re

def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = re.sub(r"_x000D_[\n\r]*", "", text) # Hapus sisa artefak excel
    text = text.lower() # Case folding
    text = re.sub(r'[^a-zA-Z0-9\s.,]', '', text) # Menghapus karakter spesial
    text = re.sub(r'\s+', ' ', text) # Hapus whitespace
    return text.strip()