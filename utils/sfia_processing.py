import pandas as pd

# # ===== Level description saja =====
# def transform_sfia_to_long_format(sfia_path):
#     sfia_df = pd.read_excel(sfia_path)
#     sfia_long_list = []
#     for index, row in sfia_df.iterrows():
#         for level in range(1, 8):
#             col_name = f'Level {level} description'
#             if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != '':
#                 sfia_long_list.append({
#                     'Skill': row['Skill'],
#                     'Level': level,
#                     'SFIA_Skill_Level': f"{row['Skill']} {level}",
#                     'Level_Description': row[col_name]
#                 })
#     return pd.DataFrame(sfia_long_list)

# ===== Level description + overall description =====
def transform_sfia_to_long_format(sfia_path):
    sfia_df = pd.read_excel(sfia_path)
    sfia_long_list = []

    for index, row in sfia_df.iterrows():
        overall_desc = str(row['Overall description']).strip() if pd.notna(row.get('Overall description')) else ''

        for level in range(1, 8):
            col_name = f'Level {level} description'
            if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != '':
                level_desc = str(row[col_name]).strip()
                combined_desc = f"{overall_desc} {level_desc}".strip() if overall_desc else level_desc

                sfia_long_list.append({
                    'Skill': row['Skill'],
                    'Level': level,
                    'SFIA_Skill_Level': f"{row['Skill']} {level}",
                    'Level_Description': combined_desc
                })

    return pd.DataFrame(sfia_long_list)
