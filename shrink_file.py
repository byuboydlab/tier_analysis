import pandas as pd


# User-set parameters
input_file_name = 'med.xlsx'
output_file_name = 'small_med.xlsx'
max_tiers = 3


df = pd.read_excel(input_file_name, sheet_name="Sheet1", engine='openpyxl')
df = df[df['Tier'] <= max_tiers]
df.to_excel(output_file_name)
