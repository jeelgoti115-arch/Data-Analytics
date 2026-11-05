import pandas as pd
import numpy as np

df_drinks = pd.read_csv("../02_pandas/supportfiles/drinks.csv")
print("Index Objecct :",df_drinks.index)
print("-"*60)
print("Columns : \n",df_drinks.columns)
print("-"*60)
print("Data types : \n",df_drinks.dtypes)
print("-"*60)
print("Shape :",df_drinks.shape)
print("Axes :",df_drinks.axes) #axes gives column names
print("-"*60)
print("Information : \n",df_drinks.info)
print("-"*60)

print(df_drinks.describe())
print("-"*60)
print(df_drinks.describe().T)
print("-"*60)
print(df_drinks.describe().T[["mean","std","min","max"]])
print("-"*60)

df_subset = df_drinks[['country','total_litres_of_pure_alcohol','continent']].copy()
print(df_subset.head())
print("-"*60)
df_subset["status"] = "Evaluated"
print(df_subset.head())
print("-"*60)
df_subset.insert(1,"region_category",value="Standard Market")
print(df_subset.head())
print("-"*60)
df_subset["boosted_liters"] = df_subset["total_litres_of_pure_alcohol"].add(2.5)
print(df_subset.head())
print("-"*60)