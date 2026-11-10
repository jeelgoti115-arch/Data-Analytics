
# Comprehensive Guide to Descriptive Statistical Measures

Descriptive statistical measures summarize, organize, and quantify key properties of a dataset $X = \{x_1, x_2, \dots, x_n\}$.

---

## 1. Measures of Central Tendency (Location)

These measures quantify the central or typical value around which the data clusters.

### Arithmetic Mean ($\mu, \bar{x}$)
The sum of all observations divided by the total sample size $n$.
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

### Median ($M$ or $\tilde{x}$)
The middle value when data points are arranged in non-decreasing order $x_{(1)} \le x_{(2)} \le \dots \le x_{(n)}$.
$$\tilde{x} = \begin{cases} 
x_{\left(\frac{n+1}{2}\right)}, & \text{if } n \text{ is odd} \\[8pt]
\frac{x_{\left(\frac{n}{2}\right)} + x_{\left(\frac{n}{2} + 1\right)}}{2}, & \text{if } n \text{ is even}
\end{cases}$$

### Mode
The value(s) $x$ that maximize the frequency function (for discrete data) or probability density function $f(x)$ (for continuous data).
$$\text{Mode} = \arg\max_{x} \text{Frequency}(x)$$

### Geometric Mean ($G$)
The $n$-th root of the product of $n$ non-negative observations, optimal for multiplicative processes and growth rates.
$$G = \left( \prod_{i=1}^{n} x_i \right)^{\frac{1}{n}} = \exp\left( \frac{1}{n} \sum_{i=1}^{n} \ln x_i \right)$$

### Harmonic Mean ($H$)
The reciprocal of the arithmetic mean of reciprocals, optimal for rates and ratios.
$$H = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}$$

### Trimmed Mean ($\bar{x}_{\alpha}$)
The arithmetic mean computed after discarding a proportion $\alpha \in [0, 0.5)$ of the lowest and highest observations ($k = \lfloor n\alpha \rfloor$).
$$\bar{x}_{\alpha} = \frac{1}{n - 2k} \sum_{i=k+1}^{n-k} x_{(i)}$$

### Winsorized Mean ($\bar{x}_{w,\alpha}$)
The arithmetic mean computed after replacing the $k = \lfloor n\alpha \rfloor$ smallest and largest observations with $x_{(k+1)}$ and $x_{(n-k)}$ respectively.
$$\bar{x}_{w,\alpha} = \frac{1}{n} \left( k \cdot x_{(k+1)} + \sum_{i=k+1}^{n-k} x_{(i)} + k \cdot x_{(n-k)} \right)$$

---

## 2. Measures of Dispersion / Variability (Spread)

These measures quantify the degree of variation, spread, or scatter in a dataset.

### Range ($R$)
The difference between the maximum and minimum values in the dataset.
$$R = x_{(n)} - x_{(1)} = \max(X) - \min(X)$$

### Interquartile Range ($\text{IQR}$)
The spread of the middle 50% of the dataset, defined as the difference between the 3rd quartile ($Q_3$) and 1st quartile ($Q_1$).
$$\text{IQR} = Q_3 - Q_1$$

### Sample Variance ($s^2$) & Population Variance ($\sigma^2$)
The average of squared deviations from the mean (using Bessel's correction $n-1$ for unbiased sample estimation).
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2, \quad \sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

### Standard Deviation ($s, \sigma$)
The square root of the variance, expressed in the original unit of measurement.
$$s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}$$

### Mean Absolute Deviation ($\text{MAD}_{\text{mean}}$)
The average distance between each data point and the mean.
$$\text{MAD}_{\text{mean}} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \bar{x}|$$

### Median Absolute Deviation ($\text{MAD}_{\text{median}}$)
A robust measure of variability, defined as the median of the absolute deviations from the median.
$$\text{MAD}_{\text{median}} = \text{median}\left( |x_i - \tilde{x}| \right)$$

### Coefficient of Variation ($\text{CV}$)
A dimensionless relative measure of dispersion, defined as the ratio of standard deviation to the mean.
$$\text{CV} = \frac{s}{\bar{x}} \times 100\%$$

---

## 3. Measures of Position / Relative Standing

These measures quantify the relative position of an individual observation $x_i$ within a distribution.

### Quantiles / Percentiles ($P_p$)
For $p \in (0, 1)$, the $p$-th quantile $q_p$ (or $100p$-th percentile) is the value satisfying:
$$P(X \le q_p) \ge p \quad \text{and} \quad P(X \ge q_p) \ge 1 - p$$

* **Quartiles ($Q_k$):** $Q_k = P_{0.25k}$ for $k \in \{1, 2, 3\}$.
* **Deciles ($D_k$):** $D_k = P_{0.10k}$ for $k \in \{1, 2, \dots, 9\}$.

### Standardized Score / Z-Score ($z_i$)
The distance between an observation $x_i$ and the mean in units of standard deviation.
$$z_i = \frac{x_i - \bar{x}}{s}$$

### Percentile Rank ($\text{PR}_x$)
The percentage of scores in a sample that fall strictly below (or below and equal to) a given score $x$.
$$\text{PR}_x = \left( \frac{\sum_{i=1}^{n} I(x_i < x) + 0.5 \sum_{i=1}^{n} I(x_i = x)}{n} \right) \times 100\%$$
*(where $I(\cdot)$ is the indicator function returning $1$ if true and $0$ if false)*

---

## 4. Measures of Shape & Distribution Geometry

These measures quantify asymmetry and tail behavior relative to a theoretical Gaussian distribution.

### Population Moments ($r$-th central moment $\mu_r$)
The foundation for higher-order statistical properties.
$$\mu_r = \mathbb{E}[(X - \mu)^r] = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^r$$

### Skewness ($\gamma_1$)
The normalized 3rd central moment measuring asymmetry around the mean.

* **Sample Skewness ($g_1$):**
  $$g_1 = \frac{m_3}{m_2^{3/2}} = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^3}{\left( \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 \right)^{3/2}}$$

* **Adjusted Fisher-Pearson Skewness ($G_1$):**
  $$G_1 = \frac{\sqrt{n(n-1)}}{n-2} \, g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3$$

### Kurtosis ($\gamma_2$)
The normalized 4th central moment measuring tail weight and extreme value likelihood relative to a Normal distribution.

* **Excess Kurtosis ($g_2$):**
  $$g_2 = \frac{m_4}{m_2^2} - 3 = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^4}{\left( \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 \right)^2} - 3$$

* **Sample Unbiased Excess Kurtosis ($G_2$):**
  $$G_2 = \frac{n-1}{(n-2)(n-3)} \left[ (n+1) g_2 + 6 \right]$$

---


## 4. Moments & Measures of Distribution Shape

Moments are quantitative measures that describe the shape, center, and spread of a probability distribution or dataset.

### 4.1 Statistical Moments

#### Raw Moments ($r$-th moment about zero, $\mu'_r$)
Measures the scale and position relative to the origin ($0$).
$$\mu'_r = \mathbb{E}[X^r] = \frac{1}{n} \sum_{i=1}^{n} x_i^r$$
* **1st Raw Moment ($\mu'_1$):** Sample Mean ($\bar{x}$).

#### Central Moments ($r$-th moment about the mean, $\mu_r$ or $m_r$)
Measures deviations relative to the center ($\bar{x}$).
$$m_r = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^r$$
* **1st Central Moment ($m_1$):** Identically $0$ ($\sum (x_i - \bar{x}) = 0$).
* **2nd Central Moment ($m_2$):** Population Variance ($\sigma^2$).
* **3rd Central Moment ($m_3$):** Unstandardized Skewness.
* **4th Central Moment ($m_4$):** Unstandardized Kurtosis.

#### Standardized Moments ($\alpha_r$)
Dimensionless, scale-invariant properties defined by dividing central moments by standard deviation $\sigma^r$.
$$\alpha_r = \frac{m_r}{s^r} = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^r}{\left( \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2} \right)^r}$$

---

### 4.2 Distribution Geometry (Skewness & Kurtosis)

#### Skewness ($\gamma_1$)
The 3rd standardized moment measuring directional asymmetry around the mean.
* **Sample Population Skewness ($g_1$):**
  $$g_1 = \frac{m_3}{m_2^{3/2}} = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^3}{\left( \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 \right)^{3/2}}$$
* **Adjusted Fisher-Pearson Sample Skewness ($G_1$):** *(Requires $n \ge 3$)*
  $$G_1 = \frac{\sqrt{n(n-1)}}{n-2} \, g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left( \frac{x_i - \bar{x}}{s} \right)^3$$

#### Kurtosis ($\gamma_2$)
The 4th standardized moment measuring tail heaviness and extreme value frequency relative to a Gaussian curve.
* **Excess Kurtosis ($g_2$):** *(Normal Distribution $= 0$)*
  $$g_2 = \frac{m_4}{m_2^2} - 3 = \frac{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^4}{\left( \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 \right)^2} - 3$$
* **Sample Unbiased Excess Kurtosis ($G_2$):** *(Requires $n \ge 4$)*
  $$G_2 = \frac{n-1}{(n-2)(n-3)} \left[ (n+1) g_2 + 6 \right]$$


  ## Summary Matrix

| Measure | Mathematical Form | Domain / Constraints | Robust to Outliers? | Breakdown Point | Key Statistical & Implementation Edge Cases |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Arithmetic Mean ($\bar{x}$)** | $\frac{1}{n}\sum_{i=1}^{n} x_i$ | $x \in \mathbb{R}$ | **No** | $0\%$ | Highly sensitive to extreme values; undefined for infinite-variance distributions (e.g., Cauchy). |
| **Median ($\tilde{x}$)** | $x_{\left(\frac{n+1}{2}\right)}$ | $x \in \mathbb{R}$ | **Yes** | $50\%$ | Non-unique interpolation when $n$ is even; less efficient than mean for Gaussian data. |
| **Mode** | $\arg\max_x \text{Freq}(x)$ | Nominal, Discrete, Continuous | **Yes** | Variable | Can be multimodal or non-existent; continuous estimation requires Kernel Density Estimation (KDE). |
| **Geometric Mean ($G$)** | $\left(\prod_{i=1}^{n} x_i\right)^{\frac{1}{n}}$ | $x_i > 0$ | **No** | $0\%$ | Undefined/fails for $x_i \le 0$; `scipy.stats.gmean` raises `ValueError` on negative inputs. |
| **Harmonic Mean ($H$)** | $\frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}$ | $x_i > 0$ | **No** | $0\%$ | Extremely sensitive to values near $0$; undefined if any $x_i = 0$. |
| **Trimmed Mean ($\bar{x}_\alpha$)** | $\frac{1}{n-2k}\sum_{i=k+1}^{n-k} x_{(i)}$ | $x \in \mathbb{R}, \alpha \in [0, 0.5)$ | **Yes** | $\alpha \cdot 100\%$ | Requires sorted array; choice of trim ratio $\alpha$ alters variance estimation. |
| **Winsorized Mean ($\bar{x}_{w,\alpha}$)** | $\frac{1}{n}\left(k \cdot x_{(k+1)} + \sum_{i=k+1}^{n-k} x_{(i)} + k \cdot x_{(n-k)}\right)$ | $x \in \mathbb{R}, \alpha \in [0, 0.5)$ | **Yes** | $\alpha \cdot 100\%$ | Preserves sample size $n$ by replacing extreme tail values instead of discarding them. |
| **Midrange** | $\frac{\max(X) + \min(X)}{2}$ | $x \in \mathbb{R}$ | **No** | $0\%$ | Extreme sensitivity; a single outlier completely shifts the center. |
| **Range ($R$)** | $\max(X) - \min(X)$ | $x \in \mathbb{R}$ | **No** | $0\%$ | Monotonically increases with sample size $n$; highly sensitive to extreme noise. |
| **Variance ($s^2$) / Standard Deviation ($s$)** | $\frac{1}{n-1}\sum_{i=1}^{n} (x_i - \bar{x})^2$ | $x \in \mathbb{R}$ | **No** | $0\%$ | Requires $n \ge 2$ for sample correction ($n-1$); $s^2 = 0$ for constant arrays causing downstream zero-division. |
| **Interquartile Range ($\text{IQR}$)** | $Q_3 - Q_1$ | $x \in \mathbb{R}$ | **Yes** | $25\%$ | Dependent on quantile interpolation method (`linear`, `nearest`, `weibull`, `excel`). |
| **Semi-IQR (Quartile Dev.)** | $\frac{Q_3 - Q_1}{2}$ | $x \in \mathbb{R}$ | **Yes** | $25\%$ | Represents half the middle 50% spread; useful for skewed non-Gaussian distributions. |
| **Mean Absolute Dev. ($\text{MAD}_{\text{mean}}$)** | $\frac{1}{n}\sum_{i=1}^{n} \Vert{}x_i - \bar{x}\Vert{}$ | $x \in \mathbb{R}$ | **No** | $0\%$ | Less sensitive to extreme values than standard deviation, but mathematically less tractable in calculus. |
| **Median Absolute Dev. ($\text{MAD}_{\text{median}}$)** | $\text{median}(\Vert{}x_i - \tilde{x}\Vert{})$ | $x \in \mathbb{R}$ | **Yes** | $50\%$ | Requires scaling factor ($\approx 1.4826$) to estimate standard deviation for normal distributions. |
| **Coefficient of Variation ($\text{CV}$)** | $\frac{s}{\bar{x}} \times 100\%$ | $\bar{x} \ne 0$ | **No** | $0\%$ | Scale-free relative measure; fails or blows up to infinity as $\bar{x} \to 0$. |
| **Quantiles / Percentiles ($P_p$)** | $P(X \le q_p) \ge p$ | $p \in (0, 1)$ | **Yes** | Variable | Up to 9 distinct interpolation standards in Python/R (`NumPy` default = `linear`). |
| **Percentile Rank ($\text{PR}_x$)** | $\left(\frac{\sum I(x_i < x) + 0.5 \sum I(x_i = x)}{n}\right) \times 100\%$ | $x \in \mathbb{R}$ | **Yes** | Variable | Step function; sensitive to duplicate ties in discrete samples. |
| **Z-Score ($z_i$)** | $\frac{x_i - \bar{x}}{s}$ | $s > 0$ | **No** | $0\%$ | Fails with `ZeroDivisionError` when standard deviation $s = 0$ (constant dataset). |
| **Raw Moment ($\mu'_r$)** | $\frac{1}{n}\sum_{i=1}^{n} x_i^r$ | $x \in \mathbb{R}$ | **No** | $0\%$ | $\mu'_1 = \text{Mean}$; rapidly loses numerical precision for large $r$ without log-transformations. |
| **Central Moment ($m_r$)** | $\frac{1}{n}\sum_{i=1}^{n} (x_i - \bar{x})^r$ | $x \in \mathbb{R}$ | **No** | $0\%$ | $m_1 \equiv 0$; $m_2 = \text{Population Variance}$; higher moments ($r \ge 3$) are extreme-outlier sensitive. |
| **Pearson's Median Skewness** | $\frac{3(\bar{x} - \tilde{x})}{s}$ | $s > 0$ | **Moderately** | Variable | Non-parametric approximation bounded roughly between $[-3, +3]$; robust alternative to 3rd moment skewness. |
| **Sample Skewness ($G_1$)** | $\frac{\sqrt{n(n-1)}}{n-2} \cdot \frac{m_3}{m_2^{3/2}}$ | $s > 0, n \ge 3$ | **No** | $0\%$ | Requires $n \ge 3$; undefined for constant arrays ($s = 0$); return `NaN` in `SciPy` for $n < 3$. |
| **Excess Kurtosis ($G_2$)** | $\frac{n-1}{(n-2)(n-3)}\left[(n+1)g_2 + 6\right]$ | $s > 0, n \ge 4$ | **No** | $0\%$ | Requires $n \ge 4$; Gaussian baseline $= 0$; extreme sensitivity to single tail outliers. |