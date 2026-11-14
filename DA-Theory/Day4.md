# Day 4: Continuous Probability Distributions & Measures of Dispersion

---

## 1. Fundamentals of Continuous Probability

### 1.1 Discrete vs. Continuous Random Variables
- **Discrete Random Variable:** Takes distinct, separate values (e.g., number of website clicks: 0, 1, 2, 3). Governed by a **Probability Mass Function (PMF)** where $P(X = x) > 0$.
- **Continuous Random Variable:** Takes any value within an uncountably infinite continuum or interval (e.g., customer wait time, transaction amount, temperature). Governed by a **Probability Density Function (PDF)**.

### 1.2 The Fundamental Rule of Continuous Probability

A critical theoretical concept tested in nearly all probability exams:

> [!IMPORTANT]
> For any continuous random variable $X$, the probability of observing any **exact, single specific point is identically zero**:
> $$P(X = c) = 0$$
> Probability exists **only over a continuous interval**:
> $$P(a \le X \le b) = \int_a^b f(x) \, dx$$

- The total area under the entire PDF curve must equal exactly 1:
  $$\int_{-\infty}^{\infty} f(x) \, dx = 1$$
- **Cumulative Distribution Function (CDF):**
  $$F(x) = P(X \le x) = \int_{-\infty}^x f(t) \, dt$$
  $F(x)$ monotonically increases from $0$ to $1$.

---

## 2. Continuous Probability Distributions

```
+-------------------------------------------------------------------------------+
|                    Continuous Distributions Quick Reference                   |
+-------------------------------------------------------------------------------+
| Distribution   Domain         Key Parameters      Mean (μ)        Variance (σ²)|
| ----------------------------------------------------------------------------- |
| Normal         (-∞, +∞)       μ (mean), σ (std)   μ               σ²          |
| Uniform        [a, b]         a (min), b (max)    (a + b) / 2     (b - a)² / 12|
| Exponential    [0, +∞)        λ (rate), β = 1/λ   1 / λ = β       1 / λ² = β² |
| Beta           [0, 1]         α, β (shapes)       α / (α + β)     Complex     |
| Gamma          [0, +∞)        k (shape), θ (scale)k · θ           k · θ²      |
+-------------------------------------------------------------------------------+
```

---

### 2.1 Gaussian / Normal Distribution ($\mathcal{N}(\mu, \sigma^2)$)

The cornerstone distribution of statistics and machine learning, justified by the Central Limit Theorem.

- **Probability Density Function (PDF):**
  $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2}$$
- **Key Characteristics:**
  1. Perfectly **symmetric and bell-shaped** around the center $\mu$.
  2. $\text{Mean} = \text{Median} = \text{Mode} = \mu$.
  3. **Inflection points** (where curve changes concavity) occur exactly at $\mu - \sigma$ and $\mu + \sigma$.
  4. Total skewness equals $0$; Kurtosis equals $3$ (Excess Kurtosis equals $0$).

#### The Empirical Rule (68–95–99.7 Rule)
For any normally distributed dataset:
- **$\mu \pm 1\sigma$** covers **$68.27\%$** of all observations.
- **$\mu \pm 2\sigma$** covers **$95.45\%$** of all observations.
- **$\mu \pm 3\sigma$** covers **$99.73\%$** of all observations.
- Any observation falling beyond $\pm 3\sigma$ is statistically classified as an **outlier** (occurs $< 0.27\%$ of the time).

#### Standard Normal Distribution ($Z$)
When $\mu = 0$ and $\sigma = 1$, the variable is denoted by $Z \sim \mathcal{N}(0, 1)$.
- Transformation (Standardization): $Z = \frac{X - \mu}{\sigma}$.

---

### 2.2 Continuous Uniform Distribution ($\mathcal{U}(a, b)$)

Models events where every sub-interval of equal length within $[a, b]$ has an identical likelihood of occurring.

- **PDF:**
  $$f(x) = \begin{cases} \frac{1}{b - a} & \text{for } a \le x \le b \\ 0 & \text{otherwise} \end{cases}$$
- **Expected Value (Mean):**
  $$\mu = \frac{a + b}{2}$$
- **Variance:**
  $$\sigma^2 = \frac{(b - a)^2}{12}$$

---

### 2.3 Exponential Distribution ($\text{Exp}(\lambda)$)

Models the **continuous time elapsed between independent Poisson events** occurring at a constant average rate $\lambda$.
- Examples: Time until next customer enters a store, server time-to-failure, waiting time in a call center queue.

- **PDF:**
  $$f(x) = \lambda e^{-\lambda x} \quad (x \ge 0, \lambda > 0)$$
- **Scale Parameter ($\beta$):** In NumPy/SciPy, the exponential is parameterized by scale $\beta = \frac{1}{\lambda}$ (the average time between events).
- **Mean & Variance Relationship:**
  $$\mu = \frac{1}{\lambda} = \beta, \quad \sigma^2 = \frac{1}{\lambda^2} = \beta^2 \implies \sigma = \mu$$
  > [!NOTE]
  > For an Exponential distribution, the **standard deviation always equals the mean** ($\sigma = \mu$)!

#### The Memoryless Property
The exponential distribution is the **only continuous distribution that is memoryless**:
$$P(X > s + t \mid X > s) = P(X > t)$$
- Meaning: If an electronic component has survived $s$ hours, the probability it will survive an additional $t$ hours is identical to the probability that a brand-new component survives $t$ hours. Past duration provides zero predictive information about future failure.

---

### 2.4 Beta Distribution ($\text{Beta}(\alpha, \beta)$)

A continuous distribution family bounded strictly on the closed interval $[0, 1]$.
- **Parameters:** Two positive shape parameters $\alpha > 0$ and $\beta > 0$.
- **Mean:** $\mu = \frac{\alpha}{\alpha + \beta}$.
- **Primary Applications:**
  - Modeling variables that naturally represent **probabilities, percentages, click-through rates (CTR), and conversion rates**.
  - In Bayesian statistics, the Beta distribution serves as the **conjugate prior** for the Binomial likelihood distribution.
  - Special Case: When $\alpha = 1$ and $\beta = 1$, Beta collapses to the standard continuous Uniform distribution $\mathcal{U}(0, 1)$.

---

### 2.5 Gamma Distribution ($\text{Gamma}(k, \theta)$)

A flexible continuous distribution family on $[0, \infty)$ that generalizes the Exponential distribution.
- **Parameters:** Shape parameter $k > 0$ (or $\alpha$) and scale parameter $\theta > 0$ (or rate $\beta = 1/\theta$).
- **Mean & Variance:**
  $$\mu = k \theta, \quad \sigma^2 = k \theta^2$$
- **Physical Interpretation:** The Gamma distribution models the **total waiting time required to observe $k$ independent Poisson events** (i.e., the sum of $k$ independent exponential random variables).
- **Applications:** Waiting time in multi-server queues, aggregate insurance claim amounts, rainfall depth.

---

## 3. Measures of Dispersion / Variability (Spread)

While central tendency identifies the middle of a dataset, **measures of dispersion quantify how scattered, spread out, or variable the data points are**.

```
Two datasets can have the identical mean (50) but vastly different dispersion:
Dataset A (Low Dispersion):  48, 49, 50, 51, 52   (Mean = 50, Variance = 2.5)
Dataset B (High Dispersion): 10, 30, 50, 70, 90   (Mean = 50, Variance = 1000)
```

---

### 3.1 Range
The absolute difference between maximum and minimum values:
$$\text{Range} = x_{\max} - x_{\min}$$
- **Limitation:** Highly sensitive to outliers; completely ignores all intermediate observations between the extremes.

---

### 3.2 Interquartile Range (IQR)
The difference between the third quartile (75th percentile) and first quartile (25th percentile):
$$IQR = Q_3 - Q_1$$
- **What it measures:** The spread of the central **50%** of observations.
- **Robustness:** Highly robust to extreme outliers because the top 25% and bottom 25% of data are ignored.
- **Tukey's Box Plot Fences for Outlier Detection:**
  - **Lower Fence:** $Q_1 - 1.5 \times IQR$
  - **Upper Fence:** $Q_3 + 1.5 \times IQR$
  - Any observation outside these fences is classified as an outlier.

---

### 3.3 Variance ($\sigma^2$ vs. $s^2$)

Variance measures the average squared deviation of data points from their arithmetic mean.

#### Population Variance ($\sigma^2$)
When analyzing every single member of an entire population of size $N$:
$$\sigma^2 = \frac{1}{N} \sum_{i=1}^N (x_i - \mu)^2$$

#### Sample Variance ($s^2$) & Bessel's Correction ($n - 1$)
When estimating population variance from a sample of size $n$:
$$s^2 = \frac{1}{n - 1} \sum_{i=1}^n (x_i - \bar{x})^2$$

> [!IMPORTANT]
> **Why do we divide by $n - 1$ instead of $n$? (Bessel's Correction)**
> Sample points naturally cluster closer to the sample mean $\bar{x}$ than to the true unknown population mean $\mu$. Dividing by $n$ systematically **underestimates** the true population variance. Dividing by degrees of freedom $n - 1$ provides an **unbiased estimator** of population variance.

#### Code Degrees of Freedom (`ddof`) Trap in Python

A major MCQ confusion point tested repeatedly across libraries:

| Library / Function | Default `ddof` | Mathematical Formula Used | Type Calculated |
| :--- | :--- | :--- | :--- |
| **`numpy.var(x)` / `numpy.std(x)`** | **`ddof=0`** | Divide by $N$ | **Population** Variance / Std Dev |
| **`pandas.Series.var()` / `.std()`** | **`ddof=1`** | Divide by $n - 1$ | **Sample** Variance / Std Dev |
| **`statistics.variance()` / `stdev()`** | **`ddof=1`** | Divide by $n - 1$ | **Sample** Variance / Std Dev |
| **`statistics.pvariance()` / `pstdev()`** | **`ddof=0`** | Divide by $N$ | **Population** Variance / Std Dev |

```python
import numpy as np
import pandas as pd
import statistics

data = [10, 20, 30]

# NumPy defaults to ddof=0 (Population)
print(np.var(data))           # 66.67
print(np.var(data, ddof=1))   # 100.0 (Sample variance with Bessel's correction!)

# Pandas defaults to ddof=1 (Sample)
print(pd.Series(data).var())  # 100.0

# Python statistics module has distinct functions
print(statistics.pvariance(data))  # 66.67 (Population)
print(statistics.variance(data))   # 100.0 (Sample)
```

---

### 3.4 Standard Deviation ($\sigma$ vs. $s$)

Standard deviation is the square root of the variance:
$$\sigma = \sqrt{\sigma^2}, \quad s = \sqrt{s^2}$$
- **Advantage over Variance:** Variance is expressed in **squared units** (e.g., $\text{dollars}^2$), making physical interpretation difficult. Standard deviation restores the spread metric to the **original units of measurement** (e.g., dollars).

---

### 3.5 Mean Absolute Deviation (MAD)
The average absolute distance between each data point and the mean:
$$\text{MAD} = \frac{1}{n} \sum_{i=1}^n |x_i - \bar{x}|$$
- Does not square deviations, avoiding disproportionate penalties on moderate variations.

---

### 3.6 Median Absolute Deviation (Median MAD)
The median of the absolute differences between each observation and the dataset median:
$$\text{Median MAD} = \text{median}\left(|x_i - \text{median}(X)|\right)$$
- **Robustness:** The most outlier-resistant dispersion metric available in statistics.
- For a standard normal distribution, standard deviation is approximated from Median MAD as:
  $$\sigma \approx 1.4826 \times \text{Median MAD}$$

---

### 3.7 Coefficient of Variation (CV)

The ratio of the standard deviation to the mean, expressed as a percentage:
$$CV = \frac{\sigma}{\mu} \times 100\% \quad \text{or} \quad CV = \frac{s}{\bar{x}} \times 100\%$$

- **Key Feature:** **Dimensionless / Unitless metric.**
- **When to use CV over Standard Deviation:**
  1. When comparing variability between datasets with **completely different units of measurement** (e.g., comparing volatility of stock prices in US Dollars vs. volatility of employee weights in kilograms).
  2. When comparing datasets with the same units but **vastly different means** (e.g., comparing price stability of \$10 consumer goods vs. \$500,000 luxury homes).
- **Limitation:** Meaningful only for ratio-scale data where zero represents a true absence of the quantity; invalid if mean $\mu \le 0$.

---

## 4. Summary Comparison of Dispersion Metrics

| Metric | Outlier Resistance | Units of Measure | Primary Application |
| :--- | :--- | :--- | :--- |
| **Range** | None (Extremely Sensitive) | Original Units | Quick boundary check; quality control limits. |
| **IQR** | High (Robust) | Original Units | Box plots; non-normal or skewed data spread. |
| **Variance** | None (Squared Penalties) | Squared Units ($X^2$) | Mathematical modeling; ANOVA; regression loss. |
| **Standard Deviation** | Low (Sensitive) | Original Units | Normal distribution spreads; Z-scores. |
| **Median MAD** | Highest (Immune to Extremes) | Original Units | Robust anomaly detection; noisy sensor data. |
| **Coefficient of Variation** | Low (Inherits from Std Dev) | Unitless (%) | Comparing relative risk or variability across scales. |

---

## 5. MCQ Examination Focus & Common Confusion Points

### Key Facts for MCQs
1. **Probability at a Single Point:** For continuous variables, $P(X = c) = 0$.
2. **Empirical Rule Percentages:** $\pm 1\sigma = 68.27\%$, $\pm 2\sigma = 95.45\%$, $\pm 3\sigma = 99.73\%$.
3. **Bessel's Correction ($n - 1$):** Used in sample variance to correct for the negative bias in estimating population variance.
4. **NumPy vs Pandas `ddof` Defaults:** NumPy defaults to `ddof=0` (population); Pandas defaults to `ddof=1` (sample).
5. **Exponential Distribution Property:** Standard deviation equals the mean ($\sigma = \mu = 1/\lambda$).
6. **Memoryless Continuous Distribution:** The Exponential distribution is uniquely memoryless.
7. **Tukey's IQR Outlier Threshold:** Lower fence $= Q_1 - 1.5 \times IQR$; Upper fence $= Q_3 + 1.5 \times IQR$.
8. **When to Use CV:** To compare relative variation between datasets measured in different units.

### Common Confusion Traps Table

| Concept A | Concept B | Difference / Confusion Trap |
| :--- | :--- | :--- |
| `np.std(x)` | `pd.Series(x).std()` | NumPy computes *population* std (`ddof=0`); Pandas computes *sample* std (`ddof=1`). |
| PDF | CDF | $\text{PDF } f(x)$ represents density (can exceed 1); $\text{CDF } F(x)$ represents cumulative probability (bounded in $[0, 1]$). |
| Standard Deviation | Variance | Std Dev is in *original units*; Variance is in *squared units*. |
| Standard Deviation | Coefficient of Variation | Std Dev measures *absolute* spread in physical units; CV measures *relative* spread as a unitless percentage. |
| Gamma Distribution | Exponential Distribution | Exponential models waiting time for *1* event; Gamma models waiting time for *k* events. |
| IQR | Range | Range is $\text{Max} - \text{Min}$ (all data); IQR is $Q_3 - Q_1$ (middle 50% only). |

