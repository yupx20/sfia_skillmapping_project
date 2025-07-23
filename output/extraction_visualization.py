import pandas as pd
import matplotlib.pyplot as plt

# Load data
df_cs = pd.read_csv("CS_New/skills_extracted_jobs_CS.csv")
df_is = pd.read_csv("IS_New/skills_extracted_jobs_IS.csv")

# Daftar kolom count yang ingin dibandingkan
skill_count_columns = [
    'skills_skillner_count',
    'skills_skillner_qe_count'
]

# Inisialisasi struktur hasil
results = {
    'Model': [],
    'Cluster': [],
    'Total': [],
    'Average': []
}

# Fungsi bantu untuk proses satu cluster
def calculate_cluster_stats(df, cluster_name):
    for col in skill_count_columns:
        total = df[col].sum()
        average = df[col].mean()
        model = col.replace('_count', '')
        results['Model'].append(model)
        results['Cluster'].append(cluster_name)
        results['Total'].append(total)
        results['Average'].append(average)

# Proses masing-masing cluster
calculate_cluster_stats(df_cs, 'CS')
calculate_cluster_stats(df_is, 'IS')

label_mapping = {
    'skills_skillner': 'SkillNER',
    'skills_skillner_qe': 'SkillNER QE'
}

# Buat DataFrame dari hasil
results_df = pd.DataFrame(results)

# ------------------ Plot Total ------------------
fig_total, ax_total = plt.subplots(figsize=(10, 6))

# Plot grouped bar chart untuk Total
bar_width = 0.35
x = range(len(skill_count_columns))

# Filter data
cs_totals = results_df[results_df['Cluster'] == 'CS']['Total']
is_totals = results_df[results_df['Cluster'] == 'IS']['Total']
labels = results_df[results_df['Cluster'] == 'CS']['Model'].map(label_mapping)

# Bar positions
x_pos = list(x)
x_pos2 = [p + bar_width for p in x_pos]

# Plotting
bars_cs = ax_total.bar(x_pos, cs_totals, width=bar_width, label='CS', color='skyblue')
bars_is = ax_total.bar(x_pos2, is_totals, width=bar_width, label='IS', color='salmon')

# Formatting
ax_total.set_xlabel('Model ekstraksi keterampilan')
ax_total.set_ylabel('Total Nilai')
ax_total.set_title('Perbandingan Total Nilai Metode Ekstraksi Keterampilan (CS vs IS)')
ax_total.set_xticks([p + bar_width / 2 for p in x_pos])
ax_total.set_xticklabels(labels, rotation=45, ha='right')
ax_total.grid(axis='y', linestyle='--', alpha=0.7)
ax_total.legend()

# Tambahkan nilai di atas batang
for bar in bars_cs + bars_is:
    yval = bar.get_height()
    ax_total.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

# ------------------ Plot Average ------------------
fig_avg, ax_avg = plt.subplots(figsize=(10, 6))

# Filter data
cs_avg = results_df[results_df['Cluster'] == 'CS']['Average']
is_avg = results_df[results_df['Cluster'] == 'IS']['Average']

# Plotting
bars_cs_avg = ax_avg.bar(x_pos, cs_avg, width=bar_width, label='CS', color='skyblue')
bars_is_avg = ax_avg.bar(x_pos2, is_avg, width=bar_width, label='IS', color='salmon')

# Formatting
ax_avg.set_xlabel('Model ekstraksi keterampilan')
ax_avg.set_ylabel('Rata-rata Nilai')
ax_avg.set_title('Perbandingan Rata-rata Nilai Metode Ekstraksi Keterampilan (CS vs IS)')
ax_avg.set_xticks([p + bar_width / 2 for p in x_pos])
ax_avg.set_xticklabels(labels, rotation=45, ha='right')
ax_avg.grid(axis='y', linestyle='--', alpha=0.7)
ax_avg.legend()

# Tambahkan nilai di atas batang
for bar in bars_cs_avg + bars_is_avg:
    yval = bar.get_height()
    ax_avg.text(bar.get_x() + bar.get_width() / 2, yval + 0.05, round(yval, 2), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()