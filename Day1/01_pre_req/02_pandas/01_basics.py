'''
why pandas?
Pandas is a powerful data manipulation and analysis library for Python. It provides data structures like DataFrames and Series that make it easy to work with structured data. With pandas, you can perform tasks such as data cleaning, transformation, aggregation, and visualization efficiently. It is widely used in data science, machine learning, and statistical analysis due to its ease of use and performance.

Data structures of pandas:
1. Series: A one-dimensional labeled array that can hold any data type (integers, strings, floats, etc.). It is similar to a column in a spreadsheet or a database table.
2. DataFrame: A two-dimensional labeled data structure with columns of potentially different types. It is similar to a spreadsheet or SQL table and is the most commonly used data structure in pandas.


Featyres of pandas:
1. Data Cleaning: Pandas provides functions to handle missing data, duplicate data, and inconsistent data formats.
2. Data Transformation: You can easily reshape, pivot, and merge datasets using pandas.
3. Data Analysis: Pandas offers a wide range of functions for statistical analysis, including grouping, filtering, and applying custom functions to data.   
4. Data Visualization: Pandas integrates well with libraries like Matplotlib and Seaborn for creating visualizations of your data.
5. others : stack, unstack, melt, pivot_table, etc.
'''

# pip install pandas : add entry to requirements.txt file explicitly : pip freeze > requirements.txt
# or
# uv add pandas  : entry will be implicitly added to the .toml file


import pandas as pd
import numpy as np

# Instantiating empty Series
empty_s = pd.Series(dtype=np.float64)
print("Empty Series:", empty_s)

# String payload (Objects)
players = ["sachin", "saurav", "rahul", "sewag"]
players_s = pd.Series(players, name="Players")
print("\n--- String Series ---")
print(players_s)

# Integer payload
scores = [15, 678, 934, 234, 876]
scores_s = pd.Series(scores, name="Scores")
print("\n--- Integer Series ---")
print(scores_s)

# Boolean payload
bool_s = pd.Series([True, False, True, True, False])
print("\n--- Boolean Series ---")
print(bool_s)


# Instantiating from a NumPy array with custom integer index
np_players = np.array(["sachin", "saurav", "rahul", "sewag"])
custom_idx_s = pd.Series(np_players, index=[100, 200, 300, 400])
print("--- Custom Index NumPy Series ---")
print(custom_idx_s)

# Instantiating from a Dictionary
players_roles = {
    "sachin": "Batsman",
    "saurav": "Captain",
    "zaheer": "Bowler",
    "virat": "Fighter",
    "robin": "Fielder"
}
dict_s = pd.Series(players_roles)
print("\n--- Series from Dictionary ---")
print(dict_s)


data_dict = {'a': 0.0, 'b': 1.0, 'c': 2.0}
aligned_s = pd.Series(data_dict, index=['b', 'c', 'd', 'a'])
print("--- Index Alignment with Missing Keys ---")
print(aligned_s)

roles_subset = pd.Series(players_roles, index=["sachin", "robin", "kapil"])
print("\n--- Partial Key Alignment ---")
print(roles_subset)

# vectorization and reduction
costs = pd.Series([3.43, 56.7, 23.4, 34.3, 22.4], name="Costs")
print("Original Costs:\n", costs)
print("\nSum:     ", costs.sum())
print("Product: ", costs.product())
print("Mean:    ", costs.mean())
print("Max:     ", costs.max())

print("\nVectorized Add (+ 5.0):\n", (costs + 5.0).head(3))
print("\nVectorized .add(5.0):\n", costs.add(5.0).head(3))


fruits = ["Apple", "Orange", "Pine", "Grapes", "Berry"]
weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
fruit_s = pd.Series(fruits, index=weekdays, name="Weekly_Fruit")
print("\n--- Fruit Series ---")
print(fruit_s)

# Metadata Inspection
print("\n--- Metadata Inspection ---")
print("Values Array:", fruit_s.values)       # Underlying NumPy array / Extension Array
print("Index Object: ", fruit_s.index)        # Explicit Index labels
print("Data Type:    ", fruit_s.dtype)        # Data type
print("Dimensions:   ", fruit_s.ndim)         # Always 1 for Series
print("Shape:        ", fruit_s.shape)        # Array tuple shape
print("Element Count:", fruit_s.size)         # Total element count

# interoperability with python
print("\n--- Interoperability with Python ---") 
city_s = pd.Series(["Bengaluru", "Mumbai", "Delhi", "Chennai", "Kolkata"])

print("Length: ", len(city_s))
print("Type:   ", type(city_s))
print("List conversion: ", list(city_s))
print("Min Label/Val:   ", min(city_s))
print("Max Label/Val:   ", max(city_s))
print("Sorted List:     ", sorted(city_s))  # Returns a new sorted Python list

# sorting
print("\n--- Sorting ---")
cities = pd.Series(["Bengaluru", "Chennai", "Ahmedabad", "Delhi", "Mumbai"], name="Cities")

print("--- Ascending Sort (Returned Copy) ---")
print(cities.sort_values())
print("\n--- Descending Sort (Returned Copy) ---")
print(cities.sort_values(ascending=False))
print("\nOriginal Series (Unchanged):")
print(cities)
cities.sort_values(inplace=True)
print("\n--- Ascending Sort (Inplace) ---")
print(cities)

# memory usage
print("\n--- Memory Usage ---")
raw_data = np.random.choice(["Low", "Medium", "High", "Critical"], size=10000)

s_object = pd.Series(raw_data)
s_category = s_object.astype("category")

print(f"Object Series RAM Usage:   {s_object.memory_usage(deep=True) / 1024:.2f} KB")
print(f"Category Series RAM Usage: {s_category.memory_usage(deep=True) / 1024:.2f} KB")
print("\nMemory Reduction:", f"{(1 - s_category.memory_usage(deep=True)/s_object.memory_usage(deep=True))*100:.1f}%")