import pandas as pd
import matplotlib.pyplot as plt

cluster_name = "CS" # Ganti CS atau IS

df = pd.read_excel(f"{cluster_name}/all_evaluation_results_{cluster_name}.xlsx")

df_true = df[df['Expanded'] == True]
df_false = df[df['Expanded'] == False]

# linechart dengan grid
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 8))

# Ubah kolom df sumbu Y untuk menampilkaan 'F1_score' / 'Precision' / 'Recall'

ax.plot(df_true['Model'], df_true['Recall'], marker='o', linestyle='-', label='Expanded=True')
for i, txt in enumerate(df_true['Recall']):
    ax.annotate(f'{txt:.3f}', (df_true['Model'].iloc[i], df_true['Recall'].iloc[i]), textcoords="offset points", xytext=(0,10), ha='center')


ax.plot(df_false['Model'], df_false['Recall'], marker='x', linestyle='--', label='Expanded=False')
for i, txt in enumerate(df_false['Recall']):
    ax.annotate(f'{txt:.3f}', (df_false['Model'].iloc[i], df_false['Recall'].iloc[i]), textcoords="offset points", xytext=(0,-15), ha='center')

ax.set_title(f'Perbandingan Recall Model Sebelum Ekspansi dan Sesudah Ekspansi ({cluster_name})', fontsize=16)
ax.set_xlabel('Model', fontsize=12)
ax.set_ylabel('Recall', fontsize=12)
ax.tick_params(axis='x', rotation=45)
ax.legend()
plt.tight_layout()

plt.show()