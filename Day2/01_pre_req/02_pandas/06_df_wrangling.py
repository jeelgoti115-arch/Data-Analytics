'''
what is data wrangling?
discovering
structuring
cleaning
enriching
validating
publishing
'''

import pandas as pd
import numpy as np

df_drinks = pd.read_csv("../02_pandas/supportfiles/drinks.csv")
print(df_drinks.head())
print("-"*60)
print(df_drinks.sort_values(by='spirit_servings',ascending=False).head()[["country","spirit_servings"]])

print("-"*60)
print(df_drinks["continent"].value_counts())
print("-"*60)
print("-- high spirit consumption (>200) (european countries)")
mask_europe = (df_drinks["continent"] == "Europe" ) & (df_drinks["spirit_servings"] > 200)
df_eu = df_drinks[mask_europe]
print(df_eu[["country","spirit_servings","continent","total_litres_of_pure_alcohol"]])
print("-"*60)

print("-- high spirit servings  (Asia and africa countries)")
# mask_asia_africa = ((df_drinks["continent"] == "Asia") | (df_drinks["continent"] == "Africa")) 
# df_aa = df_drinks[mask_asia_africa]
# print(df_aa[["country","spirit_servings","continent","total_litres_of_pure_alcohol"]])

continent_mask = df_drinks["continent"].isin(["Asia","Africa"])
df_selected_continent = df_drinks[continent_mask]
print(df_selected_continent[["country","continent"]])
print("shape :",df_selected_continent.shape)
print("-"*60)
print(df_selected_continent["continent"].value_counts())
print("-"*60)
