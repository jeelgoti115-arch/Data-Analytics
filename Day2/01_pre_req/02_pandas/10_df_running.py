import pandas as pd
import numpy as np

# lst = [2,3,-1,4] #[2,2+3=5,5-1=4,4+4=8] = [2,5,4,8]
# total = np.cumsum(lst)
# print(total) 

np.random.seed(42)

dates = pd.date_range(start="2026-01-01",periods=100,freq="D")
prices = 150 + np.random.rand(100).cumsum()

df_stocks = pd.DataFrame({"Date":dates,"Close":prices}).set_index("Date")
print(df_stocks)
print("-"*60)
#rolling window 
df_stocks["7D_MA"]=df_stocks["Close"].rolling(window=7).mean()
df_stocks["7D_STD"]=df_stocks["Close"].rolling(window=7).std()
print("-"*60)
print(df_stocks.head(15))
print("-"*60)