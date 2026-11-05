import pandas as pd
import numpy as np

#illustration 1
#order quantities matrix (Rows : Order ,Column : item)
df_orders = pd.DataFrame({
    'item_A': [2, 1, 4],
    'item_B': [0, 3, 1],
    'item_C': [5, 2, 0]
}, index=['Order_1', 'Order_2', 'Order_3'])
print(df_orders)
print("-" * 60)
print(df_orders.T)
print("-" * 60)

#unit prices series
s_prices = pd.Series([10.0, 25.0, 5.0], index=['item_A', 'item_B', 'item_C'])
print(s_prices)
print("-" * 60)

# Matrix-vector dot product 
df_totals = df_orders.dot(s_prices)
print("\n--- Total Revenue Per Order (.dot()) ---")
print(df_totals)
print("=" * 60)

#illustration 2
df_metrics = pd.DataFrame({
    'A': np.random.randint(1, 100, 5),
    'B': np.random.randint(1, 100, 5),
    'category': ['Tech', 'Health', 'Tech', 'Finance', 'Health']
})
print(df_metrics)
print("-" * 60)
# Vectorized column assignment via eval()
df_metrics.eval("C = A * 2 + B", inplace=True)
print("Evaluated DataFrame .eval()")
print(df_metrics)
print("-" * 60)
# Clean row selection via query()
filtered_df = df_metrics.query("C > 100 and category == 'Tech'")
print("\n--- Query Result (.query()) ---")
print(filtered_df)
print("=" * 60)

# illustration 3
np.random.seed(42)
df_time_series = pd.DataFrame({
    'daily_sales': [100, 120, 115, 130, 140, 125, 150, 160, 155, 170]
})
# 3-Day Moving Average
df_time_series['3D_Moving_Avg'] = df_time_series['daily_sales'].rolling(window=3).mean()
# Cumulative Running Total
df_time_series['Running_Total'] = df_time_series['daily_sales'].expanding(min_periods=1).sum()

print("--- Rolling & Expanding Window Output ---")
print(df_time_series)
print("=" * 60)

# illustration 4
df_finance = pd.DataFrame({
    'close_price': [200.0, 204.0, 202.0, 210.0, 215.0]
})
# Lag price by 1 period
df_finance['prev_price'] = df_finance['close_price'].shift(1)

# Calculate percentage returns
df_finance['daily_return_%'] = df_finance['close_price'].pct_change() * 100

# Rank daily performance
df_finance['price_rank'] = df_finance['close_price'].rank(ascending=False)
print("--- Shift, Pct_Change, and Rank Summary ---")
print(df_finance)
print("=" * 60)

# illustration 5
np.random.seed(42)
raw_data = {
    "Employee": [f"Emp_{i}" for i in range(1, 21)],
    # Most incomes are around 50k-70k, but we inject a couple of extreme outliers
    "Income": [
        55000,
        62000,
        58000,
        4500,  # Extreme low outlier (e.g., data entry error) # janitor
        61000,
        64000,
        59000,
        53000,
        67000,
        56000,
        60000,
        63000,
        57000,
        250000,  # Extreme high outlier (e.g., CEO salary)
        66000,
        54000,
        62500,
        59500,
        61500,
        60500,
    ],
}

df = pd.DataFrame(raw_data)
print("--- Original Data With Outliers ---")
# print(df.head(20))  # Previewing rows to see the outliers
print("-" * 40)
print(df.sort_values(by='Income',ascending=False))
lower_threshold = df["Income"].quantile(0.05) # 5th percentile
upper_threshold = df["Income"].quantile(0.95)
print(f"5th Percentile Floor (Lower Limit): {lower_threshold}")
print(f"95th Percentile Cap (Upper Limit): {upper_threshold}")
df_clean = df.copy()
df_clean["Income"] = df_clean["Income"].clip(
    lower=lower_threshold, upper=upper_threshold
)
print("--- Winsorized Data (Outliers Capped) ---")
print(df_clean.sort_values(by='Income',ascending=False))
print("-" * 40)
print("=" * 60)
print("=" * 60)
print("=" * 60)