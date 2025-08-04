import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np

def visualize_skills_radar_grid(cluster_name: str, top_n: int = 5):

    input_directory = f"{cluster_name}/"
    output_directory = f"{cluster_name}_Visual_Radar/"
    os.makedirs(output_directory, exist_ok=True)

    expanded_files = sorted(glob.glob(f"{input_directory}expanded_mapping_*.xlsx"))

    if not expanded_files:
        print(f"Tidak ada file 'expanded_mapping_*.xlsx' yang ditemukan di: {input_directory}")
        return
    
    fig, axes = plt.subplots(4, 2, figsize=(12, 24), subplot_kw=dict(polar=True))

    axes = axes.flatten()

    print(f"Membuat grid 4x2 untuk {len(expanded_files)} model...")

    for i, file_path in enumerate(expanded_files):
        if i >= 8:
            print("Peringatan: Ditemukan lebih dari 8 file, hanya 8 pertama yang akan divisualisasikan.")
            break

        ax = axes[i]

        try:
            df = pd.read_excel(file_path)
            base_filename = os.path.basename(file_path)

            model_name = base_filename.replace('expanded_mapping_cosine_', '').replace(f'_{cluster_name}.xlsx', '')
            
            skill_counts = df['expanded_matched_skills'].value_counts().nlargest(top_n)

            if skill_counts.empty:
                ax.set_title(f"{model_name}\n(Tidak ada data)", color='red')
                ax.set_yticklabels([])
                ax.set_xticklabels([])
                continue

            labels = skill_counts.index.tolist()
            values = skill_counts.values.tolist()
            num_vars = len(labels)
            
            angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
            angles += angles[:1]
            
            values += values[:1]

            ax.plot(angles, values, linewidth=2, linestyle='solid')
            ax.fill(angles, values, alpha=0.25)
            
            ax.set_yticklabels([])
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(labels, size=12)
            ax.set_title(model_name, size=16, y=1.15)

        except Exception as e:
            ax.set_title(f"Gagal memproses\n{os.path.basename(file_path)}", color='red')
            print(f"Error pada file {file_path}: {e}")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Visualisasi Top {top_n} Skills SFIA untuk Setiap Model {cluster_name} (Expanded)', fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95], pad=2.0)

    output_filename = f"{output_directory}grid_radar_all_models_{cluster_name}.png"
    plt.savefig(output_filename)
    plt.close()
    
    print(f"\nVisualisasi grid berhasil disimpan ke: '{output_filename}'")


if __name__ == '__main__':
    visualize_skills_radar_grid('IS')