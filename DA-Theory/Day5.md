# Day 5: Discrete Distributions, Shape Geometry, Data Lifecycle & EDA Profiling

---

## 1. Discrete Probability Distributions

A **discrete random variable** takes distinct, separated values resulting from counting processes. It is governed by a **Probability Mass Function (PMF)**:
$$\sum_{k} P(X = k) = 1, \quad P(X = k) \ge 0$$

```
+------------------------------------------------------------------------------------+
|                    Discrete Distributions Quick Reference                          |
+------------------------------------------------------------------------------------+
| Distribution   PMF Formula: P(X = k)          Mean (μ)     Variance (σ²)   Key Use |
| ---------------------------------------------------------------------------------- |
| Bernoulli      p^k · (1 - p)^(1 - k)           p            p(1 - p)        1 Trial|
| Binomial       C(n, k) · p^k · (1 - p)^(n - k) np           np(1 - p)       n Trial|
| Poisson        (λ^k · e^(-λ)) / k!             λ            λ (Equal to μ!) Rates  |
| Geometric      (1 - p)^(k - 1) · p             1 / p        (1 - p) / p²    To 1st |
+------------------------------------------------------------------------------------+
```

---

### 1.1 Bernoulli Distribution
Models a single trial that results in one of two mutually exclusive binary outcomes:
- Success ($X = 1$) with probability $p$.
- Failure ($X = 0$) with probability $q = 1 - p$.

- **Mean & Variance:**
  $$\mu = p, \quad \sigma^2 = p(1 - p)$$
- Maximum variance occurs at $p = 0.5$ ($\sigma^2_{\max} = 0.25$).

---

### 1.2 Binomial Distribution ($\text{Bin}(n, p)$)
Models the total number of successes ($k$) across $n$ independent, identical trials.

#### The 4 Mandatory Ingredients of a Binomial Experiment (BINS):
1. **B**inary outcomes: Each trial results in strictly Success or Failure.
2. **I**ndependent trials: The outcome of one trial has zero effect on any other.
3. **N**umber of trials fixed: $n$ is fixed in advance.
4. **S**ame probability: $p$ remains constant on every trial.

- **PMF:**
  $$P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k} = \frac{n!}{k!(n - k)!} p^k (1 - p)^{n - k}$$
- **Mean & Variance:**
  $$\mu = np, \quad \sigma^2 = np(1 - p), \quad \sigma = \sqrt{np(1 - p)}$$

---

### 1.3 Poisson Distribution ($\text{Pois}(\lambda)$)
Models the count of rare, independent events occurring within a fixed, continuous interval of time, area, or volume.
- Examples: Number of customer website arrivals per hour, number of defects per square meter of cloth, number of network packet collisions per second.

- **Parameter:** $\lambda$ (lambda), the average arrival rate per interval.
- **PMF:**
  $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!} \quad (k = 0, 1, 2, \dots)$$
- **The Unique Poisson Property (High-Frequency MCQ Fact):**
  $$\text{Mean} = \text{Variance} = \lambda \implies \sigma^2 = \mu = \lambda$$
  > [!NOTE]
  > Whenever a discrete distribution exhibits a **variance equal to its mean**, it is governed by a **Poisson process**.

#### Poisson Approximation to the Binomial
When $n$ is very large ($n \to \infty$) and $p$ is very small ($p \to 0$) such that $\lambda = np$ remains moderate, the Binomial distribution converges to a Poisson distribution with $\lambda = np$.

---

### 1.4 Geometric Distribution ($\text{Geom}(p)$)
Models the total number of independent Bernoulli trials required to observe the **first success**.
- **PMF:**
  $$P(X = k) = (1 - p)^{k - 1} p \quad (k = 1, 2, 3, \dots)$$
- **Mean & Variance:**
  $$\mu = \frac{1}{p}, \quad \sigma^2 = \frac{1 - p}{p^2}$$
- **Memoryless Property:** The Geometric distribution is the **only discrete probability distribution that is memoryless**.

---

## 2. Measures of Position: Quantiles & Z-Scores

Measures of position indicate where a specific observation stands relative to the rest of the distribution.

### 2.1 Percentiles & Quartiles
- **Percentiles ($P_k$):** Split ordered data into 100 equal parts. $P_{50}$ is the median.
- **Quartiles ($Q_1, Q_2, Q_3$):** Split ordered data into 4 equal quarters:
  - $Q_1$ (25th percentile): 25% of data lies at or below this value.
  - $Q_2$ (50th percentile): The median.
  - $Q_3$ (75th percentile): 75% of data lies at or below this value.

### 2.2 Z-Score (Standard Score)
Measures the signed distance between an observation $x$ and the mean $\mu$, expressed in units of standard deviation:
$$z = \frac{x - \mu}{\sigma} \quad \text{or} \quad z = \frac{x - \bar{x}}{s}$$

- **Properties of the Z-Score Distribution:**
  - Mean of Z-scores is always **$0$** ($\mu_z = 0$).
  - Standard deviation and variance of Z-scores is always **$1$** ($\sigma_z = 1, \sigma^2_z = 1$).
  - A positive $z$ means the value is above the mean; a negative $z$ means it is below the mean.
- **Outlier Threshold:** Observations with $|z| > 3$ are statistically classified as potential outliers.

### 2.3 Percentile Rank
The percentage of scores in a distribution that are equal to or lower than a given score:
$$\text{Percentile Rank} = \frac{\text{Count of scores } \le x}{N} \times 100\%$$

---

## 3. Measures of Shape & Distribution Geometry

Descriptive statistics summarizes distributions using four mathematical moments:
1. First Moment: **Mean** (Central Location)
2. Second Moment: **Variance** (Dispersion / Spread)
3. Third Moment: **Skewness** (Asymmetry)
4. Fourth Moment: **Kurtosis** (Tailedness / Peakedness)

---

### 3.1 Skewness (Asymmetry)

Skewness measures the degree and direction of asymmetry in a probability distribution.

```
       Negative / Left-Skewed            Symmetric (Normal)           Positive / Right-Skewed
        Mean < Median < Mode           Mean = Median = Mode             Mode < Median < Mean

               /\                               /\                               /\
             /    \                           /    \                           /    \
           /        \                       /        \                       /        \
_________/            \                   /            \                   /            \_________
   Long Left Tail                     Equal Balanced Tails                     Long Right Tail
```

| Property | Negative (Left) Skew | Symmetric Distribution | Positive (Right) Skew |
| :--- | :--- | :--- | :--- |
| **Tail Direction** | Long tail stretches to the **Left** (negative side). | Balanced, identical tails on both sides. | Long tail stretches to the **Right** (positive side). |
| **Skewness Value** | $\text{Skewness} < 0$ | $\text{Skewness} \approx 0$ | $\text{Skewness} > 0$ |
| **Mean vs. Median vs. Mode** | **$\text{Mean} < \text{Median} < \text{Mode}$** | **$\text{Mean} = \text{Median} = \text{Mode}$** | **$\text{Mean} > \text{Median} > \text{Mode}$** |
| **Real-World Examples** | Human lifespan/age at death; easy exam scores. | Adult heights; manufacturing tolerances. | Household income; house prices; social media followers. |

#### Pearson's Median Skewness Coefficient
A simple non-parametric skewness metric:
$$S_p = \frac{3(\text{Mean} - \text{Median})}{\sigma}$$

---

### 3.2 Kurtosis & Excess Kurtosis (Tailedness)

Kurtosis measures the thickness of the distribution tails and the extremity of outliers relative to a normal distribution.

- **Standard Normal Kurtosis = 3.**
- **Excess Kurtosis Formula:**
  $$\text{Excess Kurtosis} = \text{Kurtosis} - 3$$

```
+-------------------------------------------------------------------------------------+
|                              The Kurtosis Family                                    |
+-------------------------------------------------------------------------------------+
| Type          Kurtosis Value   Excess Kurtosis   Tail Behavior & Outlier Propensity |
| ----------------------------------------------------------------------------------- |
| Mesokurtic    Kurtosis = 3     Excess = 0        Standard Normal distribution tails.|
| Leptokurtic   Kurtosis > 3     Excess > 0        Fat / Heavy tails; sharp center    |
|                                                  peak; frequent extreme outliers.   |
| Platykurtic   Kurtosis < 3     Excess < 0        Thin / Light tails; flatter center |
|                                                  peak; fewer extreme outliers.      |
+-------------------------------------------------------------------------------------+
```

- **Leptokurtic Distributions ($> 3$):** Crucial in financial risk analytics (e.g., stock market returns exhibit heavy tails, leading to frequent "black swan" financial crashes). Examples: Student's $t$-distribution, Laplace distribution.
- **Platykurtic Distributions ($< 3$):** Example: Uniform distribution (has no outliers and no tails).

---

## 4. The Data Analytics Lifecycle

From problem definition to production decision-making, analytics adheres to a standard 8-phase lifecycle:

```
+-----------------------------------------------------------------------------+
|                        Data Analytics Lifecycle                             |
+-----------------------------------------------------------------------------+
| 1. Business Problem Formulation & Scope                                     |
|    └─ Define KPIs, business objectives, and deliverables.                   |
| 2. Data Discovery & Ingestion                                               |
|    └─ Extract raw files, SQL tables, and APIs into raw data lakes.          |
| 3. Data Preprocessing & Cleaning                                            |
|    └─ Handle missing data, deduplicate records, sanitize datatypes.         |
| 4. Exploratory Data Analysis (EDA)                                          |
|    └─ Univariate, bivariate, and multivariate distribution profiling.       |
| 5. Feature Engineering & Selection                                          |
|    └─ Transform, encode, scale, and prune predictive attributes.            |
| 6. Modeling & Hypothesis Testing                                            |
|    └─ Train predictive algorithms or execute statistical inference.         |
| 7. Model Evaluation & Validation                                            |
|    └─ Cross-validation, metric benchmarking, and error diagnostics.         |
| 8. Deployment & Actionable Storytelling                                     |
|    └─ Operationalize dashboards, automate reports, deliver insights.       |
+-----------------------------------------------------------------------------+
```

---

## 5. Missing Data Taxonomy: Rubin's Framework

Missing data mechanisms determine whether dropping rows will introduce catastrophic bias:

```
+------------------------------------------------------------------------------------+
|                         Missing Data Taxonomy (Rubin)                              |
+------------------------------------------------------------------------------------+
| Category   Formal Definition                   Impact of Dropping   Imputation Rule|
| ---------------------------------------------------------------------------------- |
| MCAR       P(Missing) is completely            Safe to drop         Mean, median,  |
|            independent of observed or          (reduces power, but  or random     |
|            unobserved variables.               introduces NO bias). imputation.    |
| ---------------------------------------------------------------------------------- |
| MAR        P(Missing) depends on OBSERVED      Biased if dropped    Model-based    |
|            covariates, but not on the          without controlling  imputation     |
|            missing value itself.               for the covariate.   (MICE, KNN).   |
| ---------------------------------------------------------------------------------- |
| MNAR       P(Missing) depends directly on the  SEVERELY BIASED      Domain modeling|
|            UNOBSERVED missing value itself.    if dropped.          or sensitivity |
|                                                                     analysis.      |
+------------------------------------------------------------------------------------+
```

### Concrete Examples:
- **MCAR (Missing Completely At Random):** A laboratory technician accidentally drops a test tube containing a patient's blood sample. Missingness is pure chance.
- **MAR (Missing At Random):** Male survey respondents are less likely to disclose their weight than female respondents. Missingness depends on the observed feature `Gender`, but conditional on gender, missingness is random.
- **MNAR (Missing Not At Random):** Ultra-high earners (millionaires) refuse to disclose their annual salary on a tax survey. The probability of missingness depends directly on the unobserved high value of `Salary`.

---

## 6. Efficient Large-Dataset I/O & Automated EDA Profiling

### 6.1 CSV Chunking for Out-of-Core Processing
When a CSV file exceeds physical RAM, process it sequentially in chunks:
```python
chunk_iterator = pd.read_csv("massive_telemetry.csv", chunksize=100_000)

total_rows = 0
for chunk in chunk_iterator:
    # Process each 100k chunk independently
    total_rows += len(chunk[chunk['error_code'] != 0])
```

### 6.2 Parquet Columnar Storage vs. CSV
- **CSV:** Row-oriented plain text. Must parse every column from disk even if only analyzing 1 feature. No type retention.
- **Parquet:** Column-oriented binary format with Snappy compression.
  - **Column Pruning:** Can read `df[['revenue']]` from disk without reading any other columns.
  - Retains strict schema types (dates remain dates; floats remain floats).

### 6.3 Automated Profiling & Advanced Distribution Visualizations
- **Profiling Tools:** `ydata-profiling` (formerly `pandas-profiling`), `sweetviz`.
- **Advanced Distribution Visualizations:**
  - **KDE (Kernel Density Estimation):** Smoothed non-parametric estimate of the PDF.
  - **Box Plot:** Displays median, IQR, and Tukey fences.
  - **Violin Plot:** Combines a box plot with a mirror KDE on both sides, revealing multimodal peaks.
  - **Boxen Plot (Letter-Value Plot):** Extends the box plot with multiple quantile boxes; ideal for visualizing distributions in large datasets ($N > 10,000$).
  - **Pairplot:** Grid of scatter plots (off-diagonal) and KDEs (diagonal) displaying pairwise relationships across all continuous features.

### 6.4 The Birthday Paradox
In probability theory, the **Birthday Paradox** states that in a group of just **23 people**, the probability that at least two individuals share the identical birthday exceeds **50%**!
- Why? We compare all possible *pairs* of individuals ($\binom{23}{2} = 253$ pairs), not individual birthdays to a specific date.
  $$P(\text{No shared birthday}) = \frac{365}{365} \times \frac{364}{365} \times \dots \times \frac{365 - n + 1}{365}$$
  $$P(\ge 1 \text{ shared birthday}) = 1 - P(\text{No shared birthday}) \approx 50.73\% \quad (n = 23)$$

---

## 7. MCQ Examination Focus & Common Confusion Points

### Key Facts for MCQs
1. **Poisson Mean Equals Variance:** $\mu = \sigma^2 = \lambda$.
2. **Right-Skewed Ordering:** $\text{Mean} > \text{Median} > \text{Mode}$.
3. **Left-Skewed Ordering:** $\text{Mean} < \text{Median} < \text{Mode}$.
4. **Normal Distribution Kurtosis:** Kurtosis $= 3$; Excess Kurtosis $= 0$.
5. **Leptokurtic Definition:** Kurtosis $> 3$ (Excess $> 0$); exhibits fat tails and sharp peaks.
6. **Z-Score Properties:** Standardized distribution always has $\text{Mean} = 0$ and $\text{Variance} = 1$.
7. **MCAR vs MAR vs MNAR:** Dropping MCAR data causes no bias; dropping MNAR data creates catastrophic bias.
8. **Memoryless Discrete Distribution:** The Geometric distribution is uniquely memoryless.
9. **Birthday Paradox Count:** A group of 23 people yields $> 50\%$ probability of a shared birthday.

### Common Confusion Traps Table

| Concept A | Concept B | Difference / Confusion Trap |
| :--- | :--- | :--- |
| Leptokurtic | Platykurtic | Leptokurtic has *fat tails* ($> 3$); Platykurtic has *thin tails* ($< 3$). |
| Positive Skew | Negative Skew | Positive skew has long tail to the *right* ($\text{Mean} > \text{Median}$); Negative has tail to the *left*. |
| Kurtosis | Excess Kurtosis | Kurtosis of normal is $3$; Excess Kurtosis of normal is $0$ ($\text{Kurtosis} - 3$). |
| MCAR | MAR | MCAR missingness is completely random; MAR missingness depends on other *observed* columns. |
| Binomial Distribution | Poisson Distribution | Binomial has *fixed $n$ trials*; Poisson counts events in a *continuous time/space interval*. |
| Percentile | Percentile Rank | Percentile is the *data value* at that cutoff; Percentile Rank is the *percentage* of data below that value. |

