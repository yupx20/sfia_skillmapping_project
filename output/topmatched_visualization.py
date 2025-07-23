import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns

def visualize_top_skills(cluster_name: str, top_n: int = 20):

    # Tentukan direktori input dan output
    input_directory = f"{cluster_name}/"
    output_directory = f"{cluster_name}_Visual/"

    # Buat folder output jika belum ada
    os.makedirs(output_directory, exist_ok=True)
    
    # Cari semua file Excel hasil pemetaan di direktori input
    # Menggunakan glob untuk mencari file yang cocok dengan pola
    mapping_files = glob.glob(f"{input_directory}mapping_*.xlsx") + \
                    glob.glob(f"{input_directory}expanded_mapping_*.xlsx")

    if not mapping_files:
        print(f"Tidak ada file Excel hasil pemetaan yang ditemukan di folder: {input_directory}")
        return

    print(f"Memproses {len(mapping_files)} file untuk klaster {cluster_name}...")

    # Loop untuk setiap file yang ditemukan
    for file_path in mapping_files:
        try:
            # Dapatkan nama file dasar untuk penamaan
            base_filename = os.path.basename(file_path).replace(".xlsx", "")
            
            # Tentukan kolom mana yang akan dianalisis berdasarkan nama file
            if "expanded_mapping" in base_filename:
                skill_column = 'expanded_matched_skills'
            else:
                skill_column = 'matched_skills'

            # Baca file Excel
            df = pd.read_excel(file_path)
            
            if skill_column not in df.columns:
                print(f"Peringatan: Kolom '{skill_column}' tidak ditemukan di file {base_filename}. File dilewati.")
                continue

            # Hitung frekuensi kemunculan setiap skill dan ambil N teratas
            skill_counts = df[skill_column].value_counts().nlargest(top_n)

            if skill_counts.empty:
                print(f"Tidak ada data skill untuk divisualisasikan di file {base_filename}.")
                continue

            # === Proses Visualisasi ===
            plt.figure(figsize=(12, 10))
            
            # Buat diagram batang menyamping (horizontal)
            sns.barplot(x=skill_counts.values, y=skill_counts.index, palette="viridis", orient='h')
            
            # Balik urutan sumbu y agar skill paling sering muncul ada di atas
            plt.gca()

            # Atur judul dan label
            plot_title = f'Top {top_n} Keterampilan SFIA Hasil Pemetaan\n(Model: {base_filename})'
            plt.title(plot_title, fontsize=16)
            plt.xlabel('Frekuensi Kemunculan', fontsize=12)
            plt.ylabel('Keterampilan SFIA', fontsize=12)
            
            # Tambahkan angka frekuensi di ujung setiap bar
            for index, value in enumerate(skill_counts.values):
                plt.text(value, index, f' {value}', va='center', fontsize=10)

            plt.tight_layout()

            # Simpan visualisasi ke file gambar
            output_filename = f"{output_directory}chart_{base_filename.replace('.xlsx', '.png')}"
            plt.savefig(output_filename)
            plt.close() 

            print(f"  -> Visualisasi untuk '{base_filename}' disimpan ke '{output_filename}'")

        except Exception as e:
            print(f"Gagal memproses file {file_path}. Error: {e}")
            
    print(f"\nProses visualisasi untuk klaster {cluster_name} selesai.")


if __name__ == '__main__':
    # visualize_top_skills(cluster_name='CS', top_n=20)
    visualize_top_skills(cluster_name='IS', top_n=20)