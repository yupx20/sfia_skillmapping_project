import pandas as pd

def transform_sfia_to_long_format(sfia_path):
    sfia_df = pd.read_excel(sfia_path)
    sfia_long_list = []
    for index, row in sfia_df.iterrows():
        for level in range(1, 8):
            col_name = f'Level {level} description'
            if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != '':
                sfia_long_list.append({
                    'Skill': row['Skill'],
                    'Level': level,
                    'SFIA_Skill_Level': f"{row['Skill']} {level}",
                    'Level_Description': row[col_name]
                })
    return pd.DataFrame(sfia_long_list)