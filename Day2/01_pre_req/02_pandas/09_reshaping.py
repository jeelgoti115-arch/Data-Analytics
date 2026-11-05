import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# pivoted data
cricket_stats = {
    'cricketers': [
        'Sachin tendulkar', 'virat kohli', 'ricky pointing', 'kumar sangakara',
        'jacques kallis', 'hasim amla', 'ms dhoni', 'joe root',
        'kane williamson', 'steve smith'
    ],
    'fifties': [164, 220, 146, 153, 149, 88, 108, 105, 96, 78],
    'hundreds': [100, 80, 71, 63, 62, 55, 16, 49, 45, 44]
}

df_cricket = pd.DataFrame(cricket_stats)
print(df_cricket)
print("-" * 60)

# unpivoted data
df_long = df_cricket.melt(
    id_vars="cricketers",
    var_name="Score Type",
    value_name="Count"
)
print(df_long)

# graphical representation
plt.figure(figsize=(10, 5))
sns.barplot(x="cricketers", y="Count", hue="Score Type", data=df_long)
plt.legend(title="Score Type")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Cricketers")
plt.ylabel("Score Count")
plt.title("International Cricket Performance: Fifties vs Hundreds")
plt.tight_layout()
plt.show()