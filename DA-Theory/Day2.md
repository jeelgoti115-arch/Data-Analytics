# Day 2: Advanced Pandas DataFrames, Data Wrangling & Reshaping

---

## 1. Pandas DataFrame Architecture & Core Operations

### 1.1 What is a DataFrame?
A **Pandas DataFrame** is a 2-dimensional, size-mutable, and potentially heterogeneous tabular data structure with labeled axes (rows and columns).
- **Index:** Row labels identifying each observation (default is integer `0` to $N-1$).
- **Columns:** Column labels identifying each feature.
- **Series Component:** Each column in a DataFrame is an individual `pd.Series` sharing a unified index.

```python
import pandas as pd
import numpy as np

# In-memory DataFrame construction
df_cricket = pd.DataFrame({
    "name": ["Gill", "Rohit", "Kohli", "Dhoni"],
    "age": [24, 37, 35, 42],
    "role": ["Batsman", "Batsman", "Batsman", "Keeper"]
})
```

### 1.2 Extracting Columns: Series vs. DataFrame

A critical syntax distinction tested in Python/Pandas exams:

| Syntax | Return Type | Dimensions | Description |
| :--- | :--- | :--- | :--- |
| `df['age']` (Single Bracket) | `pd.Series` | 1-Dimensional (`shape = (4,)`) | Extracts a single column as a 1D Series. |
| `df[['age']]` (Double Brackets) | `pd.DataFrame` | 2-Dimensional (`shape = (4, 1)`) | Subsets columns and preserves DataFrame tabular structure. |
| `df[['age', 'name']]` | `pd.DataFrame` | 2-Dimensional (`shape = (4, 2)`) | Subsets multiple columns as a DataFrame. |

### 1.3 Core Structural Inspection Methods

```python
df.shape        # (num_rows, num_columns) tuple
df.dtypes       # Data types for every column
df.columns      # Index object containing column headers
df.index        # Index object representing row labels
df.axes         # List of [row_index, column_index]
df.head(n=5)    # Returns first n rows (default is 5)
df.tail(n=5)    # Returns last n rows (default is 5)
df.info()       # Summary of non-null counts, column dtypes, and memory footprint
df.describe()   # Summary statistics (count, mean, std, min, 25%, 50%, 75%, max)
```

---

## 2. Dataset Ingestion & Export Formats

Modern analytics workflows interface with multiple file formats:

| Format | Read Function | Write Function | Key Advantages & Best Use Cases |
| :--- | :--- | :--- | :--- |
| **CSV** | `pd.read_csv("file.csv")` | `df.to_csv("file.csv", index=False)` | Human-readable text format; universally supported; slow for large files; loses schema types on disk. |
| **Excel** | `pd.read_excel("file.xlsx", sheet_name="Sheet1")` | `df.to_excel("file.xlsx", sheet_name="passengers", index=False)` | Business reporting; requires `openpyxl`; slower I/O performance. |
| **Parquet** | `pd.read_parquet("file.parquet")` | `df.to_parquet("file.parquet", compression='snappy')` | Columnar binary storage; preserves exact data types; compressed; drastically faster queries. Standard format for Big Data. |

> [!IMPORTANT]
> Always pass `index=False` when calling `.to_csv()` or `.to_excel()` unless you deliberately intend to save the DataFrame's row index as an extra unnamed column.

---

## 3. RAM Footprint & Memory Optimization via Downcasting

In production data engineering and machine learning, datasets frequently exceed available RAM. Pandas defaults to allocating `int64` (8 bytes) and `float64` (8 bytes) even if values only range from 1 to 100.

### 3.1 Surface vs. Deep Memory Inspection
- **`df.memory_usage().sum()`**: Shallow memory estimate. Counts only pointer allocations for `object` types, severely undercounting actual RAM usage.
- **`df.memory_usage(deep=True).sum()`**: Deep memory estimate. Traverses nested Python string objects in memory to report true system byte consumption.

### 3.2 Downcasting Numerical Types with `pd.to_numeric`
```python
df_optimized = df.copy()

# Downcast integer columns to the smallest safe subtype (int8, int16, int32)
for col in df_optimized.select_dtypes(include=['int64']).columns:
    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')

# Downcast float columns to float32
for col in df_optimized.select_dtypes(include=['float64']).columns:
    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
```

### 3.3 Integer Subtype Ranges Reference

| Type | Bytes | Minimum Value | Maximum Value |
| :--- | :--- | :--- | :--- |
| `int8` | 1 | $-128$ | $+127$ |
| `int16` | 2 | $-32,768$ | $+32,767$ |
| `int32` | 4 | $-2,147,483,648$ | $+2,147,483,647$ |
| `int64` | 8 | $\approx -9.22 \times 10^{18}$ | $\approx +9.22 \times 10^{18}$ |

---

## 4. The Data Wrangling Framework

**Data Wrangling** (also called Data Munging) is the structured, iterative process of transforming raw, messy data into an analysis-ready format.

### The 6 Stages of Data Wrangling:
1. **Discovering:** Profiling, examining dimensions, understanding column meanings, and inspecting data types.
2. **Structuring:** Reshaping, reorganizing wide/long layouts, indexing, and normalizing schemas.
3. **Cleaning:** Addressing missing values, standardizing text casing, handling outliers, and deduplicating records.
4. **Enriching:** Creating derived features, calculating ratios, and joining supplemental external datasets.
5. **Validating:** Verifying constraints, statistical checks, unit consistency, and business logic conformance.
6. **Publishing:** Exporting clean, documented data artifacts to production databases or downstream pipelines.

### 4.1 Boolean Filtering with Multiple Conditions
When combining multiple boolean masks in Pandas:
- You **must** use bitwise operators: `&` (AND), `|` (OR), `~` (NOT).
- You **must** enclose each individual condition inside parentheses `()`.
- Standard Python keywords `and`, `or`, `not` **will fail** because they evaluate truthiness of the entire Series object rather than element-wise.

```python
# Filtering European countries with spirit servings > 200
mask = (df["continent"] == "Europe") & (df["spirit_servings"] > 200)
df_filtered = df[mask]
```

### 4.2 Frequency Counts: `value_counts()`
```python
df['continent'].value_counts()                  # Absolute counts
df['continent'].value_counts(normalize=True)    # Relative proportions (percentages sum to 1.0)
df['continent'].value_counts(dropna=False)      # Includes NaN count in distribution
```

---

## 5. Missing Value Management: Detect, Drop & Impute

Missing data is represented in Pandas as `NaN` (Not a Number - IEEE floating-point standard) or `None` (Python object).

### 5.1 Detection
```python
df.isna().sum()         # Count of missing values per column
df.isna().mean() * 100  # Percentage of missingness per column
```

### 5.2 Dropping Missing Data
- `df.dropna(how='any')`: Drops rows if **any** column has a `NaN`.
- `df.dropna(how='all')`: Drops rows only if **all** columns are `NaN`.
- `df.dropna(subset=['Age', 'Embarked'])`: Drops rows if specified subset columns have `NaN`.
- `df.dropna(axis=1)`: Drops entire columns containing missing values.

### 5.3 Imputation Strategies
- **Continuous Numerical Features (Skewed / Outliers):** Impute with the **median**.
- **Continuous Numerical Features (Normal / Symmetric):** Impute with the **mean**.
- **Categorical Features:** Impute with the **mode** (most frequent category) or a sentinel string (e.g., `'Unknown'`).

```python
# Imputing numeric column with median
median_age = df["Age"].median()
df["Age"] = df["Age"].fillna(median_age)

# Imputing categorical column with mode
mode_embarked = df["Embarked"].mode()[0]  # .mode() returns a Series
df["Embarked"] = df["Embarked"].fillna(mode_embarked)
```

---

## 6. The GroupBy Aggregation Engine

Pandas follows the **Split-Apply-Combine** paradigm formalized by Hadley Wickham:
1. **Split:** Data is partitioned into subsets based on grouping keys.
2. **Apply:** Independent aggregation or transformation functions run on each subset.
3. **Combine:** Results merge into an output DataFrame.

```python
# Method 1: Named Aggregation (Produces clean, flat column names)
continent_summary = df.groupby('continent').agg(
    avg_beer = ('beer_servings', 'mean'),
    max_wine = ('wine_servings', 'max'),
    total_alcohol = ('total_litres_of_pure_alcohol', 'sum'),
    country_count = ('country', 'count')
)

# Method 2: Dictionary Aggregation (Produces hierarchical MultiIndex columns)
multi_summary = df.groupby('continent').agg({
    'beer_servings': ['mean', 'std', 'max'],
    'wine_servings': ['sum'],
    'country': ['count']
})
```

---

## 7. Data Reshaping: Wide vs. Long Formats

Understanding reshaping operations is essential for feature engineering and visualization.

```
+--------------------------------------------------------------+
|                     Wide vs. Long Formats                    |
+--------------------------------------------------------------+
| WIDE FORMAT:                                                 |
| Player       Fifties   Hundreds                              |
| Kohli        220       80                                    |
| Rohit        180       31                                    |
|                                                              |
|                     melt() (Unpivoting)                      |
|                           ----->                             |
|                     pivot() (Pivoting)                       |
|                           <-----                             |
|                                                              |
| LONG (TIDY) FORMAT:                                          |
| Player       Score_Type   Count                              |
| Kohli        Fifties      220                                |
| Kohli        Hundreds     80                                 |
| Rohit        Fifties      180                                |
| Rohit        Hundreds     31                                 |
+--------------------------------------------------------------+
```

### 7.1 Melting (Wide $\to$ Long)
`pd.melt()` unpivots a DataFrame from wide format to long format.
- **`id_vars`:** Identifier columns to keep intact as row anchors.
- **`value_vars`:** Specific columns to unpivot (if omitted, unpivots all remaining columns).
- **`var_name`:** Name of the new categorical column containing the old headers.
- **`value_name`:** Name of the new numerical column containing the values.

```python
df_long = pd.melt(
    df_wide,
    id_vars=['player'],
    var_name='score_type',
    value_name='count'
)
```

### 7.2 Pivoting (Long $\to$ Wide)
- **`df.pivot(index, columns, values)`:** Reshapes data from long back to wide.
  - **CRITICAL RESTRICTION:** Fails with a `ValueError: Index contains duplicate entries` if there are multiple records with identical `(index, columns)` pairs!
- **`df.pivot_table(index, columns, values, aggfunc='mean', fill_value=0)`:** Robust pivoting that **aggregates duplicate entries** using a mathematical function (`'mean'`, `'sum'`, `'count'`).

### 7.3 Stacking & Unstacking
- **`df.stack()`:** Pivots column headers into the innermost row index, producing a multi-level indexed `pd.Series`.
- **`df.unstack()`:** Pivots the innermost row index into column headers.

---

## 8. Time Series & Running Metrics

### 8.1 Date Ranges & Cumulative Sums
```python
# Generate continuous daily timeline
dates = pd.date_range(start="2026-01-01", periods=100, freq="D")
df_stock = pd.DataFrame({"Date": dates, "Price": np.random.randn(100).cumsum()}).set_index("Date")

# Cumulative aggregations
df_stock["CumSum"] = df_stock["Price"].cumsum()
df_stock["CumMax"] = df_stock["Price"].cummax()
```

### 8.2 Rolling Window Calculations
Rolling computations create a moving window over ordered sequential data:
```python
# 7-day Moving Average & Moving Standard Deviation
df_stock["7D_MA"] = df_stock["Price"].rolling(window=7).mean()
df_stock["7D_STD"] = df_stock["Price"].rolling(window=7).std()
```
> [!NOTE]
> The first `window - 1` entries of a rolling metric evaluate to `NaN` unless `min_periods=1` is specified.

### 8.3 Lag, Lead & Financial Return Metrics
- **`df['Close'].shift(1)` (Lag):** Shifts values downwards by 1 row (brings previous day's value to the current row).
- **`df['Close'].shift(-1)` (Lead):** Shifts values upwards by 1 row (brings future day's value to the current row).
- **`df['Close'].pct_change()`:** Computes percentage return: $\frac{\text{Close}_t - \text{Close}_{t-1}}{\text{Close}_{t-1}}$.
- **`df['Close'].rank(ascending=False)`:** Assigns ordinal rank based on numerical values.

---

## 9. Vectorized Strings, Dates & Binning

### 9.1 Vectorized String Extraction (`.str`)
Pandas allows vectorized string parsing using regular expressions and named capture groups:
```python
df_raw = pd.DataFrame({'raw_code': ['PROD_101_LAPTOP_V2', 'PROD_102_MOBILE_V1']})

# Standardize text
df_raw['clean_code'] = df_raw['raw_code'].str.strip().str.upper()

# Extract components using named regex capture groups
regex = r"PROD_(?P<ID>\d+)_(?P<Category>[A-Z]+)_(?P<Version>V\d+)"
df_extracted = df_raw['clean_code'].str.extract(regex)
```

### 9.2 Datetime Accessor (`.dt`)
```python
df['timestamp'] = pd.to_datetime(df['timestamp_str'])

# Calendar feature engineering
df['year'] = df['timestamp'].dt.year
df['month_name'] = df['timestamp'].dt.month_name()
df['day_name'] = df['timestamp'].dt.day_name()
df['hour'] = df['timestamp'].dt.hour
df['days_diff'] = (df['end_date'] - df['start_date']).dt.days  # Timedelta in days
```

### 9.3 Binning Continuous Data: `cut` vs. `qcut`

A classic exam topic testing data discretization techniques:

| Function | Method | Bin Widths | Bin Counts (Frequencies) | Primary Use Case |
| :--- | :--- | :--- | :--- | :--- |
| **`pd.cut()`** | **Equal-Width Binning** | Constant interval width: $\frac{\text{Max} - \text{Min}}{k}$. | Variable (depends on distribution shape; bins in dense areas hold more items). | Custom domain intervals (e.g., Age groups: 0-18, 19-35, 36-60, 60+). |
| **`pd.qcut()`** | **Equal-Frequency (Quantile) Binning** | Variable interval widths. | Approximately equal count of observations in every bin. | Percentile segmentation, decile scoring, RFM scoring. |

```python
# Fixed interval binning
df['exp_group'] = pd.cut(df['experience'], bins=[0, 5, 12, 20], labels=['Junior', 'Mid', 'Senior'])

# Equal-frequency quantile binning (tertiles)
df['exp_quantiles'] = pd.qcut(df['experience'], q=3, labels=['Low', 'Mid', 'High'])
```

### 9.4 Outlier Clipping & Winsorization
Clipping restricts extreme values to specified threshold boundaries without removing rows:
```python
lower_bound = df['Income'].quantile(0.05)
upper_bound = df['Income'].quantile(0.95)

# Values below 5th percentile or above 95th percentile are capped
df['Clean_Income'] = df['Income'].clip(lower=lower_bound, upper=upper_bound)
```

---

## 10. MCQ Examination Focus & Common Confusion Points

### Key Facts for MCQs
1. **DataFrame Extraction:** `df['col']` gives a Series; `df[['col']]` gives a 1-column DataFrame.
2. **Boolean Indexing Operators:** Always use `&`, `|`, `~` with parentheses. Standard keywords `and`, `or` cause runtime errors on Series.
3. **`pivot()` vs. `pivot_table()`:** `pivot()` crashes when duplicates exist for index-column combinations. `pivot_table()` aggregates duplicates using `aggfunc` (default: `'mean'`).
4. **`cut()` vs. `qcut()`:** `cut()` yields equal bin ranges; `qcut()` yields equal sample counts per bin.
5. **Memory Usage Deep:** Always pass `deep=True` to `memory_usage()` when profiling DataFrames containing `object` string columns.
6. **`shift(1)` vs. `shift(-1)`:** `shift(1)` is a lag operator (previous value); `shift(-1)` is a lead operator (future value).
7. **`stack()` vs. `unstack()`:** `stack()` moves column headers into the index; `unstack()` moves an index level to the columns.
8. **Parquet vs CSV:** Parquet is a columnar binary format with built-in compression and metadata type retention, making it vastly faster than row-based CSV.

### Common Confusion Traps Table

| Concept A | Concept B | Difference / Confusion Trap |
| :--- | :--- | :--- |
| `pd.cut()` | `pd.qcut()` | `cut` splits based on *value intervals* (equal width). `qcut` splits based on *sample percentiles* (equal population). |
| `df.pivot()` | `df.pivot_table()` | `pivot` requires unique index-column pairs; `pivot_table` allows duplicates and aggregates them. |
| `shift(1)` | `shift(-1)` | `shift(1)` retrieves the *prior* row (lag); `shift(-1)` retrieves the *next* row (lead). |
| `df.memory_usage()` | `df.memory_usage(deep=True)` | Shallow counts only pointer memory (underreports text); `deep=True` scans full heap allocations. |
| `df['col'].mode()` | `df['col'].mean()` | `.mean()` returns a scalar float; `.mode()` returns a `pd.Series` because distributions can be multimodal. |
| `df.dropna(how='any')` | `df.dropna(how='all')` | `how='any'` drops a row if even one null exists; `how='all'` drops only if every column is null. |

