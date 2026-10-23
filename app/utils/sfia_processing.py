import pandas as pd
import re

def clean_excel_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'_x000D_[\n\r]*', '\n', text)
    text = re.sub(r'_x000D_', ' ', text)
    text = text.replace('\r', '')
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def transform_sfia_to_long_format(sfia_path):
    sfia_df = pd.read_excel(sfia_path)
    sfia_long_list = []

    for index, row in sfia_df.iterrows():
        overall_desc = clean_excel_text(str(row['Overall description'])) if pd.notna(row.get('Overall description')) else ''

        for level in range(1, 8):
            col_name = f'Level {level} description'
            if col_name in row and pd.notna(row[col_name]) and str(row[col_name]).strip() != '':
                level_desc = clean_excel_text(str(row[col_name]))
                combined_desc = f"{overall_desc} {level_desc}".strip() if overall_desc else level_desc

                sfia_long_list.append({
                    'Skill': row['Skill'],
                    'Level': level,
                    'SFIA_Skill_Level': f"{row['Skill']} {level}",
                    'Level_Description': combined_desc
                })

    return pd.DataFrame(sfia_long_list)
