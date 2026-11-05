import pandas as pd
import numpy as np

df_wide = pd.DataFrame({
    'player': ['Kohli', 'Rohit', 'Gill'],
    'fifties': [220, 180, 45],
    'hundreds': [80, 31, 6]
})


print("--- Original Wide Format ---")
print(df_wide)
print("=" * 60)
# Unpivoting into long format
df_melted = df_wide.melt(id_vars='player', var_name='score_type', value_name='count')
print("\n--- Melted (Long) Format ---")
print(df_melted)
print("=" * 60)
df_pivoted = df_melted.pivot(index='player',columns="score_type",values='count')
print("--- Pivoted Back to Wide Format ---")
print(df_pivoted)
print("=" * 60)
df_sales = pd.DataFrame({
    'region': ['North', 'North', 'South', 'South', 'North'],
    'product': ['A', 'B', 'A', 'B', 'A'],
    'revenue': [100, 150, 200, 120, 130]
})
print(df_sales)
# Aggregate revenue across duplicate entries
pt = df_sales.pivot_table(index='region',columns='product',values='revenue',aggfunc='sum', fill_value=0)
print("--- Pivot Table (Sum Revenue) ---")
print(pt)
print("=" * 60)
print(pt.T)
print("=" * 60)
df_multi = pd.DataFrame({
    'Q1': [10, 20],
    'Q2': [15, 25]
}, index=['Company_A', 'Company_B'])
print(df_multi)
print("=" * 60)
# Stack columns into a MultiIndex Series
stacked = df_multi.stack()
print("--- Stacked Series ---")
print(stacked)
print("=" * 60)
# Unstack back to DataFrame
unstacked = stacked.unstack()
print("\n--- Unstacked DataFrame ---")
print(unstacked)
print("=" * 60)
data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Favorite_Sport': ['Cricket', 'Cricket', 'Football', 'Cricket', 'Football', 'Tennis', 'Cricket', 'Tennis']
}
df_fans = pd.DataFrame(data)
sport_crosstab = pd.crosstab(df_fans['Gender'], df_fans['Favorite_Sport'])
print("--- Raw DataFrame ---")
print(df_fans)
print("\n--- Crosstab Breakdown ---")
print(sport_crosstab)
print("=" * 60)
df_titanic = pd.read_csv("supportFiles/titanic.csv")
print(df_titanic.head())
print("=" * 60)
# Cross-tabulation of Pclass vs Survived
ct = pd.crosstab(df_titanic['Pclass'], df_titanic['Survived'], margins=True)
print("--- Passenger Class vs Survival Counts ---")
print(ct)
print("=" * 60)
df_nested = pd.DataFrame({
    'project': ['Alpha', 'Beta'],
    'tags': [['Python', 'C++'], ['Pandas', 'SQL', 'Docker']]
})
print("--- Nested Lists ---")
print(df_nested)
print("=" * 60)
df_exploded = df_nested.explode('tags')
print("\n--- Exploded DataFrame ---")
print(df_exploded)
print("=" * 60)
df_emp = pd.DataFrame({'emp_id': [1, 2, 3], 'name': ['Alice', 'Bob', 'Charlie']})
df_dept = pd.DataFrame({'emp_id': [2, 3, 4], 'dept': ['HR', 'Engineering', 'Finance']})
# Inner Join
merged_inner = pd.merge(df_emp, df_dept, on='emp_id', how='inner')
print("--- Inner Join ---")
print(merged_inner)
print("=" * 60)
# Left Join
merged_left = pd.merge(df_emp, df_dept, on='emp_id', how='left')
print("\n--- Left Join ---")
print(merged_left)
print("=" * 60)

# Right Join
merged_right = pd.merge(df_emp, df_dept, on='emp_id', how='right')
print("\n--- Right Join ---")
print(merged_right)
print("=" * 60)
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
# Row-wise Concatenation
concatenated = pd.concat([df1, df2], ignore_index=True)
print("--- Vertical Concatenation ---")
print(concatenated)

print("=" * 60)
df_primary = pd.DataFrame({'price': [100, np.nan, 300, np.nan]})
df_backup = pd.DataFrame({'price': [105, 200, 305, 400]})

patched = df_primary.combine_first(df_backup)
print("--- Patched Dataset ---")
print(patched)
print("=" * 60)