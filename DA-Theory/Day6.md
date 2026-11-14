# Day 6: Bivariate Association, SQL Analytics, Simpson's Paradox & Feature Engineering

---

## 1. Bivariate & Multivariate Association Metrics

Understanding statistical association between pairs of features is essential for exploratory data analysis (EDA), multicollinearity audits, and feature selection.

```
+------------------------------------------------------------------------------------+
|                      Association Metrics Quick Reference                           |
+------------------------------------------------------------------------------------+
| Metric          Variable Types         Relationship Tested    Range      Outliers  |
| ---------------------------------------------------------------------------------- |
| Pearson (r)     Continuous vs. Cont.   Strictly Linear        [-1, +1]   Sensitive |
| Spearman (ρ)    Continuous or Ordinal  Monotonic (Ranked)     [-1, +1]   Robust    |
| Kendall (τ)     Continuous or Ordinal  Concordant / Discordant[-1, +1]   Robust    |
| Cramér's V      Categorical vs. Cat.   Nominal Association    [0, 1]     N/A       |
+------------------------------------------------------------------------------------+
```

---

### 1.1 Covariance vs. Correlation
- **Covariance ($\text{Cov}(X, Y)$):** Measures the joint direction of linear movement between two continuous variables:
  $$\text{Cov}(X, Y) = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})$$
  - Positive covariance: As $X$ increases, $Y$ tends to increase.
  - Negative covariance: As $X$ increases, $Y$ tends to decrease.
  - **Major Flaw:** Covariance is **scale-dependent**; its numerical value depends on the measurement units (e.g., measuring height in meters vs. millimeters changes covariance by a factor of 1000).

- **Pearson Correlation Coefficient ($r$):** Normalized, scale-invariant covariance:
  $$r = \frac{\text{Cov}(X, Y)}{s_x s_y} = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}$$
  - Bound strictly within $[-1, +1]$.
  - $r = +1$: Perfect positive linear relationship.
  - $r = -1$: Perfect negative linear relationship.
  - $r = 0$: **No LINEAR relationship**.

> [!WARNING]
> **The $r = 0$ MCQ Trap:** A Pearson correlation of $r = 0$ means there is **no linear relationship**. It does **NOT** mean the variables are independent! A perfect non-linear relationship (e.g., $y = x^2$ symmetric around zero) yields $r \approx 0$.

---

### 1.2 Spearman Rank Correlation ($\rho$ or $r_s$)
Measures the strength and direction of a **monotonic relationship** between two variables using ranked values.
- **Monotonicity:** As $X$ increases, $Y$ consistently increases (monotonic increasing) or consistently decreases (monotonic decreasing), without requiring a straight line.
- **Formula for Untied Ranks:**
  $$\rho = 1 - \frac{6 \sum d_i^2}{n(n^2 - 1)}$$
  where $d_i = \text{rank}(x_i) - \text{rank}(y_i)$.
- **Advantages:** Non-parametric; does not assume normality; **highly robust to extreme outliers**.

---

### 1.3 Kendall's Tau ($\tau$)
Measures rank association based on the proportion of **concordant pairs** ($C$) versus **discordant pairs** ($D$):
$$\tau = \frac{C - D}{\frac{1}{2} n (n - 1)}$$
- A pair is *concordant* if $x_i > x_j$ and $y_i > y_j$ (ranks agree).
- Preferred over Spearman for **small sample sizes** with a high number of identical (tied) ranks.

---

### 1.4 Cramér's V (Categorical Association)
Measures the strength of association between two **nominal categorical variables** based on the Pearson Chi-Square ($\chi^2$) test statistic:
$$V = \sqrt{\frac{\chi^2}{n \cdot \min(k - 1, r - 1)}}$$
where $n$ is total sample size, $r$ is number of rows, and $k$ is number of columns in the contingency table.
- Range: $[0, 1]$.
- $0$: Complete independence between categories; $1$: Perfect association.

---

### 1.5 Multicollinearity & Variance Inflation Factor (VIF)
**Multicollinearity** occurs when two or more independent predictor features in a regression model are highly correlated, inflating the standard errors of regression coefficients.

- **VIF Formula for Feature $j$:**
  $$VIF_j = \frac{1}{1 - R_j^2}$$
  where $R_j^2$ is the coefficient of determination when feature $j$ is regressed against all other independent features.
- **VIF Interpretation Benchmarks:**
  - $VIF = 1$: No correlation whatsoever.
  - $1 < VIF < 5$: Moderate, acceptable collinearity.
  - $VIF > 5 \text{ or } 10$: **Severe multicollinearity.** The feature should be removed, combined, or regularized.

---

## 2. Simpson's Paradox & The Confounder Trap

### 2.1 What is Simpson's Paradox?
**Simpson's Paradox** is a famous statistical phenomenon in which an observed trend, correlation, or pattern across multiple individual subgroups **completely reverses or disappears** when the subgroups are combined into an aggregated whole!

```
Group 1 (Subgroup A): Positive correlation / higher success rate
Group 2 (Subgroup B): Positive correlation / higher success rate
COMBINED TOTAL:       NEGATIVE correlation / lower success rate!
```

### 2.2 Root Cause: Confounding / Lurking Variables
Simpson's Paradox is caused by an unobserved or ignored **confounding variable** that is systematically associated with both the grouping assignment and the outcome variable.

#### Classic Real-World Example: UC Berkeley Admissions Case
- Overall university admission data showed men were accepted at a significantly higher rate than women (alleged gender bias).
- **Subgroup Breakdown by Department:** When analyzed department by department, women were admitted at equal or higher rates than men in almost every department!
- **Confounder Identified:** Women systematically applied to highly competitive departments with low overall acceptance rates (e.g., English, Humanities), whereas men applied to departments with large capacity and high acceptance rates (e.g., Engineering).

> [!TIP]
> **Takeaway for Data Analysts:** Never draw business conclusions from high-level aggregated summary metrics alone. Always slice and dice data across key demographic and behavioral segments.

---

## 3. SQL for Relational Database Analytics

SQL is the universal query language for enterprise analytics and database extraction.

### 3.1 Advanced Aggregations & Conditional Logic
- **`CASE WHEN`:** Conditional branching:
  ```sql
  SELECT employee_id,
         CASE WHEN salary >= 100000 THEN 'Executive'
              WHEN salary >= 60000  THEN 'Mid-Level'
              ELSE 'Entry-Level' END AS salary_band
  FROM employees;
  ```
- **`COALESCE(val1, val2, ...)`:** Returns the **first non-null** expression in the argument list. Essential for handling missing values.
- **`NULLIF(val1, val2)`:** Returns `NULL` if $val1 = val2$; otherwise returns $val1$. Used to prevent **division-by-zero errors**: `revenue / NULLIF(units_sold, 0)`.
- **`WHERE` vs. `HAVING` (Core MCQ Topic):**
  - `WHERE`: Filters individual rows **before** any aggregation takes place. Cannot contain aggregate functions like `SUM()` or `AVG()`.
  - `HAVING`: Filters aggregated groups **after** `GROUP BY` execution.

---

### 3.2 SQL Window Functions (`OVER (PARTITION BY ... ORDER BY ...)`)

Window functions perform calculations across a set of table rows related to the current row without collapsing them into a single summary row (unlike `GROUP BY`).

#### Ranking Window Functions Compared

| Function | Tie Behavior | Sequential Gaps? | Output for Scores: `[90, 80, 80, 70]` |
| :--- | :--- | :--- | :--- |
| **`ROW_NUMBER()`** | Assigns distinct arbitrary integers to ties. | **No gaps** (strictly sequential: $1, 2, 3, 4$). | `1, 2, 3, 4` |
| **`RANK()`** | Assigns identical rank to ties; skips subsequent ranks. | **Leaves gaps** after ties. | `1, 2, 2, 4` (Rank 3 is skipped!) |
| **`DENSE_RANK()`** | Assigns identical rank to ties; does NOT skip. | **No gaps** after ties. | `1, 2, 2, 3` |
| **`NTILE(n)`** | Divides ordered rows into $n$ equal bucket numbers. | Discrete buckets ($1$ to $n$). | `1, 1, 2, 2` (for `NTILE(2)`) |

#### Lead, Lag & Running Totals
- **`LAG(column, offset, default)`:** Retrieves values from a preceding row within the partition.
- **`LEAD(column, offset, default)`:** Retrieves values from a subsequent row within the partition.
- **Running Cumulative Total:**
  ```sql
  SELECT order_date, daily_sales,
         SUM(daily_sales) OVER (
             ORDER BY order_date 
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
         ) AS cumulative_sales
  FROM daily_revenue;
  ```

---

### 3.3 Common Table Expressions (CTEs)
A **CTE** is a temporary named result set defined within a query using the `WITH` clause:
```sql
WITH regional_revenue AS (
    SELECT region, SUM(amount) AS total_rev
    FROM sales
    GROUP BY region
)
SELECT region, total_rev
FROM regional_revenue
WHERE total_rev > 100000;
```
- **Recursive CTEs:** Contain an anchor query and a recursive member joined by `UNION ALL`. Used for modeling hierarchical data (e.g., manager-employee org charts, bill of materials).

---

## 4. Feature Engineering & Transformations

Feature engineering is the process of using domain knowledge to extract features from raw data that improve the performance of machine learning algorithms.

```
+-------------------------------------------------------------------------------+
|                      Categorical Encoding Strategies                          |
+-------------------------------------------------------------------------------+
| Strategy          Mechanism                         Best Use Case / Warning   |
| ----------------------------------------------------------------------------- |
| One-Hot Encoding  Binary columns (0 or 1)           Low cardinality (< 10).   |
|                   for each unique category.         Trap: Dummy variable trap!|
| ----------------------------------------------------------------------------- |
| Ordinal Encoding  Maps categories to ranked         Natural ordered categories|
|                   integers (1, 2, 3).               (e.g., Education, Tier).  |
| ----------------------------------------------------------------------------- |
| Target (Mean)     Replaces category with mean       High cardinality.         |
| Encoding          of the target variable.           Danger: Severe target     |
|                                                     leakage without smoothing!|
| ----------------------------------------------------------------------------- |
| Frequency EncodingReplaces category with its        High cardinality features |
|                   count or percentage.              where volume matters.     |
+-------------------------------------------------------------------------------+
```

### 4.1 Categorical Encoding Nuances
- **The Dummy Variable Trap:** Including all $k$ binary columns created by One-Hot Encoding introduces perfect multicollinearity because the columns sum to 1. **Solution:** Drop one reference column (`drop_first=True`), producing $k - 1$ columns.

---

### 4.2 Numerical Scaling & Power Transforms

Algorithms relying on distance metrics (K-Means, KNN, SVM) or gradient descent (Logistic Regression, Neural Networks) are severely distorted if features have different numerical scales.

```
+------------------------------------------------------------------------------------+
|                         Feature Scalers Compared                                   |
+------------------------------------------------------------------------------------+
| Scaler            Formula                           Output Range   Outlier Impact  |
| ---------------------------------------------------------------------------------- |
| StandardScaler    z = (x - μ) / σ                   (-∞, +∞)       Distorted       |
| MinMaxScaler      (x - x_min) / (x_max - x_min)     [0, 1]         Extremely Dist. |
| RobustScaler      (x - Q2) / IQR                    (-∞, +∞)       ROBUST / Safe   |
+------------------------------------------------------------------------------------+
```

#### Power Transforms for Skewed Features
Transforms non-normal, skewed continuous features into a Gaussian (bell-shaped) distribution:
1. **Log Transform ($\log(1 + x)$ or `np.log1p`):** Shrinks extreme positive tails. Works only for non-negative values ($x \ge 0$).
2. **Box-Cox Transformation:** Parameterized power transform $\frac{x^\lambda - 1}{\lambda}$. **Strict Requirement: $x > 0$ strictly.** Cannot handle zero or negative values.
3. **Yeo-Johnson Transformation:** Generalized power transform that **accommodates zero and negative values ($x \le 0$)**.

---

### 4.3 Date-Time & Cyclical Feature Engineering
- **Temporal Component Extraction:** `.dt.year`, `.dt.month`, `.dt.day`, `.dt.dayofweek`, `.dt.hour`.
- **Cyclical Encoding (Sine / Cosine Transform):**
  - Linear encoding treats hour 23 and hour 0 as maximally distant ($23 - 0 = 23$), even though they are only 1 hour apart in physical time!
  - **Solution:** Project time onto a 2D circle using sine and cosine:
    $$x_{\sin} = \sin\left(\frac{2\pi \cdot t}{T}\right), \quad x_{\cos} = \cos\left(\frac{2\pi \cdot t}{T}\right)$$
    where $T = 24$ for hours, $T = 12$ for months, $T = 7$ for days of week.

---

### 4.4 Text Feature Extraction Basics
1. **Cleaning:** Lowercasing, regex removing punctuation (`[^\w\s]`), stop-word removal.
2. **Stemming vs. Lemmatization:**
   - *Stemming:* Heuristic chopping of word suffixes (e.g., "studying" $\to$ "studi"). Can produce non-dictionary words.
   - *Lemmatization:* Uses morphological vocabulary analysis to return the true base dictionary form (lemma) (e.g., "better" $\to$ "good").
3. **Bag-of-Words (`CountVectorizer`):** Matrix of raw word frequency counts.
4. **TF-IDF (`TfidfVectorizer`):** Term Frequency $\times$ Inverse Document Frequency:
   $$IDF(t) = \ln\left(\frac{1 + N}{1 + DF(t)}\right) + 1$$
   - Penalizes common words appearing across all documents (e.g., "the", "system") and amplifies rare words distinctive to specific texts.

---

## 5. MCQ Examination Focus & Common Confusion Points

### Key Facts for MCQs
1. **Pearson $r = 0$:** Indicates zero *linear* correlation; a non-linear relationship may still exist.
2. **Spearman Rank Correlation:** Measures *monotonic* relationship; uses ranks; highly robust to outliers.
3. **Cramér's V Metric:** Measures association between *two categorical variables*; range $[0, 1]$.
4. **`RANK()` vs. `DENSE_RANK()`:** `RANK()` leaves gaps after ties ($1, 2, 2, 4$); `DENSE_RANK()` leaves no gaps ($1, 2, 2, 3$).
5. **`WHERE` vs. `HAVING`:** `WHERE` executes before `GROUP BY`; `HAVING` executes after `GROUP BY`.
6. **Box-Cox vs. Yeo-Johnson:** Box-Cox requires strictly positive values ($x > 0$); Yeo-Johnson supports zero and negative values.
7. **Dummy Variable Trap Solution:** Drop one column (`drop_first=True`), producing $k - 1$ columns for $k$ categories.
8. **Simpson's Paradox Root Cause:** Presence of an omitted confounding variable.
9. **Cyclical Encoding Purpose:** Preserves circular boundary continuity between period end and start (e.g., 23:00 and 00:00).

### Common Confusion Traps Table

| Concept A | Concept B | Difference / Confusion Trap |
| :--- | :--- | :--- |
| `RANK()` | `DENSE_RANK()` | `RANK()` skips ranks after ties (e.g., 1, 2, 2, 4); `DENSE_RANK()` does not skip (1, 2, 2, 3). |
| `WHERE` | `HAVING` | `WHERE` filters rows before grouping; `HAVING` filters aggregated groups after `GROUP BY`. |
| Box-Cox | Yeo-Johnson | Box-Cox requires $x > 0$; Yeo-Johnson handles negative values and zero ($x \le 0$). |
| Pearson $r$ | Spearman $\rho$ | Pearson evaluates straight-line linearity; Spearman evaluates monotonic order of ranks. |
| One-Hot Encoding | Ordinal Encoding | One-Hot creates $k$ binary columns (for unordered nominals); Ordinal maps to integers (for ordered ranks). |
| Stemming | Lemmatization | Stemming chops word endings (fast, crude); Lemmatization uses linguistic dictionaries to find valid root words. |

