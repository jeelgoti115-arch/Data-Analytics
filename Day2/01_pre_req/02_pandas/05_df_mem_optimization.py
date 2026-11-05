import pandas as pd
import numpy as np
#setting explicit index column on load
df_indexed = pd.read_csv("../02_pandas/supportfiles/drinks.csv", index_col="country")
print("Custom country index preview : \n", df_indexed.head(3))

#look at ram footprint 
mem_original = df_indexed.memory_usage().sum()
print("Original dataframe memory usage in bytes :",mem_original) #surface level
mem_deep = df_indexed.memory_usage(deep=True).sum()
print("Original dataframe deep memory usage in bytes :",mem_deep) #element level
print("-"*60)
print(df_indexed.info())
print("-"*60)
print("{0:35}{1:15}{2:15}{3:15}".format("Column name","min","max","data type"))
# for col in df_indexed.select_dtypes(include=[np.number]).columns: 
for col in df_indexed.select_dtypes(include=['int64','float64']).columns:
    print("{0:<35}{1:<15}{2:<15}{3:<15}".format(col.capitalize(),
                                         df_indexed[col].min(), 
                                         df_indexed[col].max(),
                                         df_indexed[col].dtypes.__str__()
                                         ))
print("-"*60)
#downcasting
df_optimized = df_indexed.copy()
for col in df_optimized.select_dtypes(include=['int64']).column:
    cast_to_str = "integer" if np.issubdtype(dt,np.integer) else "float"
    df_optimized[col] = pd.to_numeric(df_optimized[col],downcast="integer")
print("-"*60)


#aa file adhuri chhe sir mathi levanu chhe 