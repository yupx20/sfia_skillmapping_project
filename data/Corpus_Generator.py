import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Pastikan data NLTK sudah diunduh
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
try:
    word_tokenize('test')
except LookupError:
    nltk.download('punkt')

# --- INPUT DIUBAH ---
# Muat dataset dari satu file Excel dengan sheet yang berbeda
file_input_excel = 'Anotasi Skill - Gopal.xlsx'
job_posting_df = pd.read_excel(file_input_excel, sheet_name='Job Posting')
sfia_df = pd.read_excel(file_input_excel, sheet_name='SFIA')

stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Pastikan input adalah string untuk menghindari error pada data kosong (NaN)
    if not isinstance(text, str):
        return ''
    # Ubah teks menjadi huruf kecil
    text = text.lower()
    # Hapus tanda baca
    text = re.sub(r'[().,\-\']', ' ', text)
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    # Tokenisasi teks
    tokens = word_tokenize(text)
    # Hapus stopwords
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# Proses dataframe 'Job Posting'
job_posting_df['Korpus'] = job_posting_df['Deskripsi Pekerjaan'].apply(preprocess_text)
print("Job Posting with Korpus:")
print(job_posting_df[['Deskripsi Pekerjaan', 'Korpus']].head())

# Proses dataframe 'SFIA'
sfia_df['Korpus'] = sfia_df['Deskripsi Level'].apply(preprocess_text)
print("\nSFIA with Korpus:")
print(sfia_df[['Deskripsi Level', 'Korpus']].head())

# --- OUTPUT DIUBAH ---
# Simpan dataframe yang sudah diproses ke satu file Excel dengan sheet yang berbeda
file_output_excel = 'Output_Proses_Gopal.xlsx'
with pd.ExcelWriter(file_output_excel) as writer:
    job_posting_df.to_excel(writer, sheet_name='Job Posting', index=False)
    sfia_df.to_excel(writer, sheet_name='SFIA', index=False)

print(f"\nProses selesai. Data berhasil disimpan ke '{file_output_excel}'")