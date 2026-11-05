import pandas as pd
import numpy as np

df_titanic = pd.read_csv("supportfiles/titanic.csv")
print("Loaded dataset shape :", df_titanic.shape)
print(df_titanic.head(3))

# Export DataFrame to Excel spreadsheet
df_titanic.to_excel("supportFiles/titanic.xlsx", sheet_name="passengers", index=False)

# Reload DataFrame from Excel spreadsheet
df_from_excel = pd.read_excel("supportFiles/titanic.xlsx", sheet_name="passengers")
print("\nReloaded Excel Passengers Shape:", df_from_excel.shape)