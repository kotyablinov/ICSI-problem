"""
Предварительное приведение результата исхода к бинарному формату.

В structure_changes.csv результат успешный исход описывается значением 2pn. 
1. Предварительно данный столбец приводится к единому регистру, убираются лишние пробелы 
2. Затем ячейки содержащие 2pn считаются успешными - 1, а остальное 0.
"""

import pandas as pd
import re

input_file = r'D:\diplom_tenserflow\analysis_changes\structure_changes.csv'
output_file = r'D:\diplom_tenserflow\analysis_changes\structure_changes_cleaned.csv'

df = pd.read_csv(input_file)
df = df.dropna()

def is_2pn(value):
    value = str(value).lower()
    normalized = re.sub(r'[\s\-_]', '', value)
    
    return 1 if '2pn' in normalized else 0

df['result'] = df['result'].apply(is_2pn)

df.to_csv(output_file, index=False)