import numpy as np
import pandas as pd

'''
What is Dataframe ?
'''
df_cricket = pd.DataFrame({
	"name": ["gill", "rohit", "VK", "Thala"],
	"age" : [77,45,18,7],
	"gender" : ["male","male","male","male"]
})

print("In memory dataframe")
print(df_cricket)

print("Extract the age column")
age_series = df_cricket['age']
print(age_series)
print(type(age_series))
print("describe the age column")
print(age_series.describe())
print("Info about the data frame :",df_cricket.info())
print("shape of the data frame :",df_cricket.shape)
print("data type of name column :",df_cricket['name'].dtype)