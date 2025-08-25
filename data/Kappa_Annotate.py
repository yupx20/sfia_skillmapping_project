import pandas as pd

def normalize(skill):
    return skill.strip().lower().replace('-', ' ').replace('_', ' ').replace('  ', ' ')

def manual_cohen_kappa(rater1_vector, rater2_vector):
    a, b, c, d = 0, 0, 0, 0
    for i in range(len(rater1_vector)):
        if rater1_vector[i] == 1 and rater2_vector[i] == 1:
            a += 1
        elif rater1_vector[i] == 1 and rater2_vector[i] == 0:
            b += 1
        elif rater1_vector[i] == 0 and rater2_vector[i] == 1:
            c += 1
        elif rater1_vector[i] == 0 and rater2_vector[i] == 0:
            d += 1

    total = a + b + c + d
    if total == 0:
        return {'a': 0, 'b': 0, 'c': 0, 'd': 0, 'Po': 1.0, 'Pe': 1.0, 'kappa': 1.0}

    po = (a + d) / total
    p_rater1_yes = (a + b) / total
    p_rater2_yes = (a + c) / total
    p_rater1_no = (c + d) / total
    p_rater2_no = (b + d) / total
    pe = (p_rater1_yes * p_rater2_yes) + (p_rater1_no * p_rater2_no)

    if pe == 1.0:
        kappa = 1.0 if po == 1.0 else 0.0
    else:
        kappa = (po - pe) / (1 - pe)
    
    return {'a': a, 'b': b, 'c': c, 'd': d, 'Po': po, 'Pe': pe, 'kappa': kappa}

def calculate_and_save_kappa(file1, file2, sheet_name, writer):
    print(f"--- Memproses Sheet: {sheet_name} ---")

    if sheet_name == 'Job Posting':
        id_column = 'Nama Pekerjaan'
    elif sheet_name == 'SFIA':
        id_column = 'Skill - Level'
    else:
        print(f"Nama sheet '{sheet_name}' tidak dikenali.")
        return

    try:
        df1 = pd.read_excel(file1, sheet_name=sheet_name)
        df2 = pd.read_excel(file2, sheet_name=sheet_name)
    except Exception as e:
        print(f"Gagal membaca sheet '{sheet_name}'. Pastikan nama sheet dan file sudah benar. Error: {e}")
        return

    on_column = 'Korpus'
    annotation_column = 'Hasil Anotasi Pakar'
    annotator1_name = file1.replace('Anotasi Skill - ', '').replace('.xlsx', '').lower()
    annotator2_name = file2.replace('Anotasi Skill - ', '').replace('.xlsx', '').lower()

    df1 = df1.rename(columns={annotation_column: f'{annotator1_name}_annotations'})
    df2 = df2.rename(columns={annotation_column: f'{annotator2_name}_annotations'})

    df1_subset = df1[[id_column, on_column, f'{annotator1_name}_annotations']].dropna(subset=[on_column])
    df2_subset = df2[[on_column, f'{annotator2_name}_annotations']].dropna(subset=[on_column])
    
    merged_df = pd.merge(df1_subset, df2_subset, on=on_column)

    results = []

    def get_normalized_annotated_skills(annotation):
        if pd.isna(annotation):
            return set()
        return {normalize(s) for s in str(annotation).split('\n') if s.strip()}

    for _, row in merged_df.iterrows():
        corpus_terms = set(normalize(word) for word in str(row[on_column]).split() if word.strip())
        anot1_skills = get_normalized_annotated_skills(row[f'{annotator1_name}_annotations'])
        anot2_skills = get_normalized_annotated_skills(row[f'{annotator2_name}_annotations'])
    
        items_to_rate = sorted(list(corpus_terms.union(anot1_skills).union(anot2_skills)))

        anot1_vector = [1 if item in anot1_skills else 0 for item in items_to_rate]
        anot2_vector = [1 if item in anot2_skills else 0 for item in items_to_rate]
        
        kappa_components = manual_cohen_kappa(anot1_vector, anot2_vector)

        result_row = {id_column: row[id_column]}
        result_row.update(kappa_components)
        results.append(result_row)

    results_df = pd.DataFrame(results)

    if results_df.empty:
        print("Tidak ada data yang cocok untuk diproses.")
        return
    
    column_order = [id_column, 'a', 'b', 'c', 'd', 'Po', 'Pe', 'kappa']
    results_df = results_df[column_order]
        
    avg_kappa = results_df['kappa'].mean()
    print(f"\nNilai Rata-rata Kappa: {avg_kappa:.4f}\n")
    print(results_df.to_string())
    print("-" * 80 + "\n")

    # Simpan ke sheet sesuai nama
    results_df.to_excel(writer, sheet_name=sheet_name, index=False)

# File input
file_anot1 = 'Anotasi Skill - Khansa.xlsx'
file_anot2 = 'Anotasi Skill - Rizfi.xlsx'

anot1_name = file_anot1.replace('Anotasi Skill - ', '').replace('.xlsx', '').lower()
anot2_name = file_anot2.replace('Anotasi Skill - ', '').replace('.xlsx', '').lower()

# Menulis ke satu file excel dengan dua sheet
with pd.ExcelWriter(f'Kappa {anot1_name} x {anot2_name}.xlsx', engine='openpyxl') as writer:
    calculate_and_save_kappa(file_anot1, file_anot2, sheet_name='Job Posting', writer=writer)
    calculate_and_save_kappa(file_anot1, file_anot2, sheet_name='SFIA', writer=writer)

print(f"Hasil seluruh sheet tersimpan di file: Kappa {anot1_name} x {anot2_name}.xlsx")