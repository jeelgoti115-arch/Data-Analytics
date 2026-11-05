'''
what is data wrangling?
discovering
structuring
cleaning
enriching
validating
publishing
'''

'''
Missing values
identify
drop
impute
'''

import pandas as pd
import numpy as np
#load dataset
df_titanic = pd.read_csv("../02_pandas/supportfiles/titanic.csv")
print(df_titanic.head())

print("-"*60)
print(df_titanic.isna().sum())
median_age = df_titanic["Age"].median()
df_titanic["Age_Imputed"] = df_titanic["Age"].fillna(median_age)
df_titanic["Cabin_filled"] = df_titanic["Cabin"].fillna("Unknown")
print("-"*60)
print("After imputation age :",df_titanic["Age_Imputed"].isna().sum())
print("After imputation Cabin :",df_titanic["Cabin_filled"].isna().sum())
print("-"*60)
print(df_titanic.head()[["Age","Age_Imputed","Cabin","Age_Imputed"]])
df_titanic_cleaned = df_titanic.copy()
# 1.
# df_titanic_cleaned["Age"] = df_titanic_cleaned["Age_Imputed"]
# df_titanic_cleaned["Cabin"] = df_titanic_cleaned["Cabin_filled"]

# 2.
# df_titanic_cleaned[["Age","Cabin"]] = df_titanic_cleaned[["Age_Imputed","Cabin_filled"]]

# 3.
df_titanic_cleaned = df_titanic_cleaned.assign(
	Age = df_titanic_cleaned['Age_Imputed'],
	Cabin = df_titanic_cleaned['Cabin_filled']
)

print(df_titanic_cleaned.isna().sum())
df_titanic_cleaned = df_titanic_cleaned.drop(columns=["Age_Imputed","Cabin_filled"])
print("-"*60)
print(df_titanic_cleaned["Embarked"].value_counts())
print("-"*60)
'''
Embarked
S    644	: Southampton(England)
C    168	: Cherbourg (France) - the first stop over
Q     77	: Queenstown (Ireland)
'''
most_used = df_titanic_cleaned["Embarked"].mode()
print(most_used)
print("-"*60)
df_titanic_cleaned["Embarked"] = df_titanic_cleaned["Embarked"].fillna('S')
print(df_titanic_cleaned.isna().sum())