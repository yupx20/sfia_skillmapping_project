import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import string
import matplotlib.gridspec as gridspec

def plot_single_radar(ax, df, model_name, top_n=10, color='blue'):

    skill_counts = df['expanded_matched_skills'].value_counts().nlargest(top_n)

    if skill_counts.empty:
        ax.set_title(f"{model_name}\n(Tidak ada data)", color='red', fontsize=18)
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        return []

    labels_raw = skill_counts.index.tolist()
    values = skill_counts.values.tolist()
    num_vars = len(labels_raw)

    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    values_closed = values + [values[0]]
    angles_closed = angles + [angles[0]]

    ax.plot(angles_closed, values_closed, linewidth=2.5, linestyle='solid', color=color, zorder=3)
    ax.fill(angles_closed, values_closed, color=color, alpha=0.15, zorder=2)

    abjad_labels = list(string.ascii_uppercase)[:num_vars]
    ax.set_xticks(angles)
    ax.set_xticklabels(abjad_labels, size=20, fontweight='bold')
    
    max_val = max(values) if values else 1
    tick_step = np.ceil(max_val / 4 / 5) * 5 if max_val > 10 else np.ceil(max_val / 4)
    if tick_step == 0: tick_step = 1
    
    yticks = np.arange(tick_step, max_val + tick_step, tick_step)
    
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(y)) for y in yticks], color="grey", size=10)
    ax.set_ylim(0, max_val * 1.15)
    
    ax.spines['polar'].set_visible(False)
    ax.grid(axis='y', linestyle='--', color='black', alpha=0.8, zorder=1)

    ax.set_title(model_name, size=24, y=1.1, weight='bold')

    legend_info = [f"{abjad_labels[i]}. {labels_raw[i]} ({values[i]})" for i in range(num_vars)]
    return legend_info

def visualize_skills_radar_grid(cluster_name: str, top_n: int):

    input_directory = f"{cluster_name}/"
    output_directory = f"{cluster_name}_Visual_Radar/"
    os.makedirs(output_directory, exist_ok=True)

    expanded_files = sorted(glob.glob(f"{input_directory}expanded_mapping_*.xlsx"))
    if not expanded_files:
        print(f"Tidak ada file 'expanded_mapping_*.xlsx' ditemukan di {input_directory}")
        return

    keybert_models = [f for f in expanded_files if 'keybert' in os.path.basename(f).lower()]
    non_keybert_models = [f for f in expanded_files if 'keybert' not in os.path.basename(f).lower()]

    keybert_models = keybert_models[:4]
    non_keybert_models = non_keybert_models[:4]
    max_pairs = min(len(keybert_models), len(non_keybert_models))

    if max_pairs == 0:
        print("Tidak ditemukan pasangan model yang cocok untuk divisualisasikan.")
        return
        
    print(f"Membuat {max_pairs} visualisasi radar (2 plot per gambar)...")
    
    colors = ['#1f77b4', '#ff7f0e']

    for i in range(max_pairs):
        fig = plt.figure(figsize=(24, 15))
        gs = gridspec.GridSpec(2, 2, 
                               figure=fig, 
                               height_ratios=[10, 5],
                               hspace=0.3,
                               wspace=0.1)

        radar_ax1 = fig.add_subplot(gs[0, 0], polar=True)
        radar_ax2 = fig.add_subplot(gs[0, 1], polar=True)
        legend_ax1 = fig.add_subplot(gs[1, 0])
        legend_ax2 = fig.add_subplot(gs[1, 1])

        legend_ax1.axis("off")
        legend_ax2.axis("off")

        df1 = pd.read_excel(non_keybert_models[i])
        model1 = os.path.basename(non_keybert_models[i]).replace('expanded_mapping_cosine_', '').replace(f'_{cluster_name}.xlsx', '')
        legend_text1 = plot_single_radar(radar_ax1, df1, model1, top_n=top_n, color=colors[0])
        
        legend_ax1.text(0.05, 0.95, '\n'.join(legend_text1), fontsize=22,
                        va='top', ha='left', family='monospace',
                        bbox=dict(boxstyle="round,pad=0.8", fc='aliceblue', ec='lightgray', lw=1))

        df2 = pd.read_excel(keybert_models[i])
        model2 = os.path.basename(keybert_models[i]).replace('expanded_mapping_cosine_', '').replace(f'_{cluster_name}.xlsx', '')
        legend_text2 = plot_single_radar(radar_ax2, df2, model2, top_n=top_n, color=colors[1])
        
        legend_ax2.text(0.05, 0.95, '\n'.join(legend_text2), fontsize=22,
                        va='top', ha='left', family='monospace',
                        bbox=dict(boxstyle="round,pad=0.8", fc='oldlace', ec='lightgray', lw=1))
        
        fig.suptitle(
            f'Top {top_n} Mapped SFIA Skills {cluster_name.upper()}',
            fontsize=28, fontweight='bold', y=1.02
        )
        
        fig.patch.set_edgecolor('black')
        fig.patch.set_linewidth(2)
        
        fig.tight_layout(rect=[0, 0.03, 1, 0.96])

        output_file = f"{output_directory}radar_comparison_{model1}_vs_{model2}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.3)
        plt.close(fig)

        print(f"Gambar {i+1} disimpan: {output_file}")

if __name__ == '__main__':
    CLUSTER_TARGET = 'IS'
    TOP_SKILLS = 10
    visualize_skills_radar_grid(cluster_name=CLUSTER_TARGET, top_n=TOP_SKILLS)