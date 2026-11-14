# Data Analytics (DA) — Complete 10-Day Theory & MCQ Revision Manual

This repository contains the complete, structured, day-wise theoretical syllabus and MCQ examination preparation guide for the **Data Analytics** curriculum.

Every topic is explained from first principles in clear, beginner-friendly language, accompanied by mathematical formulas, ASCII diagrams, code syntax, practical examples, high-frequency MCQ exam facts, and common confusion traps.

---

## 📚 Curriculum Navigation Table

| Day File | Core Theme & Domains | Primary Concepts & Keywords Covered |
| :---: | :--- | :--- |
| **[Day 1](Day1.md)** | **Python Data Science Prerequisites: NumPy & Pandas Basics** | NumPy memory architecture, C-contiguous buffers vs. Python list pointers, SIMD vectorization, Broadcasting rules, Views vs. Copies, 1D Transpose behavior, Universal Functions (`ufuncs`), Linear Algebra (`@`, `np.dot`, `inv`, `det`, `eig`), Einstein Summation (`np.einsum`), PyTorch framework architecture (`torch.nn`, `torch.optim`, `torch.autograd`), Pandas Series fundamentals, and Categorical memory optimization. |
| **[Day 2](Day2.md)** | **Pandas DataFrames, Wrangling & Reshaping** | DataFrame architecture, Series extraction (`df['col']` vs. `df[['col']]`), I/O formats (CSV, Excel, Parquet), Memory optimization and downcasting, 6-stage Data Wrangling framework, Multi-condition boolean masks, Missing value imputation (mean, median, mode), Split-Apply-Combine GroupBy engine, Reshaping (Melt wide-to-long, Pivot long-to-wide, Stack/Unstack), Time-series metrics (Rolling windows, Cumsum, Shift, Pct_change), Vectorized strings, Datetime manipulation (`.dt`), and Binning (`cut` vs. `qcut`). |
| **[Day 3](Day3.md)** | **Statistical Foundations, Grid Generators & Central Tendency** | Descriptive vs. Inferential statistics, Deterministic grid generators (`np.linspace`, `np.logspace`, `np.geomspace`, `np.arange`, `np.meshgrid`), Functional signals and noise models (Harmonic, Exponential decay, Impulse noise, Homoscedasticity vs. Heteroscedasticity), Measures of Central Tendency (Arithmetic Mean, Median, Mode, Geometric Mean, Harmonic Mean, Trimmed Mean, Winsorized Mean), and the Pythagorean Means Inequality ($HM \le GM \le AM$). |
| **[Day 4](Day4.md)** | **Continuous Probability Distributions & Measures of Dispersion** | Continuous random variables, PDF vs. CDF ($P(X = c) = 0$), Gaussian/Normal distribution, Empirical Rule (68-95-99.7%), Standard Normal $Z$, Continuous Uniform, Exponential distribution and its Memoryless Property, Beta distribution, Gamma distribution, Measures of Dispersion (Range, IQR, Population vs. Sample Variance, Bessel's correction $n-1$, Standard Deviation, MAD, Median MAD, and Coefficient of Variation). |
| **[Day 5](Day5.md)** | **Discrete Distributions, Shape Geometry, Lifecycle & EDA Profiling** | Discrete random variables and PMF, Bernoulli, Binomial (BINS conditions), Poisson ($\mu = \sigma^2 = \lambda$), Geometric distribution, Measures of Position (Percentiles, Quartiles, Z-Score properties), Measures of Shape (Skewness direction and Mean/Median/Mode ordering, Kurtosis: Mesokurtic, Leptokurtic, Platykurtic), 8-stage Data Analytics Lifecycle, Rubin's Missing Data Taxonomy (MCAR, MAR, MNAR), Chunking, Parquet I/O, Advanced EDA plots (KDE, Boxen, Violin, Pairplots), and the Birthday Paradox. |
| **[Day 6](Day6.md)** | **Bivariate Association, SQL Analytics, Simpson's Paradox & Feature Engineering** | Covariance vs. Correlation, Pearson linear $r$ ($r=0$ trap), Spearman monotonic rank correlation, Kendall's Tau, Cramér's V, Multicollinearity and VIF, Simpson's Paradox and confounding variables, Advanced SQL (`CASE WHEN`, `COALESCE`, `NULLIF`, `WHERE` vs. `HAVING`), Window Functions (`ROW_NUMBER`, `RANK`, `DENSE_RANK`, `NTILE`, `LEAD`, `LAG`), Recursive CTEs, Feature Engineering (One-Hot dummy trap, Ordinal, Target encoding, Scalers, Box-Cox vs. Yeo-Johnson, Cyclical Sine/Cosine encoding, Bag-of-Words, TF-IDF). |
| **[Day 7](Day7.md)** | **Production Ingestion Pipelines, Data Hygiene & Advanced Outlier Detection** | Enterprise data ingestion architecture (Batch extractors, DB watermark queries, API scrapers with exponential backoff, Kafka event streaming), Data hygiene (MD5/SHA-256 Checksums, YAML Schema contracts, Quarantine architecture, Ingestion memory downcasting), Parametric Z-score outliers vs. Non-parametric IQR fences, Machine Learning tree-based Isolation Forest (`IsolationForest` path length mechanism), and 10 Industry Domain Case Formulations. |
| **[Day 8](Day8.md)** | **Production Preprocessing Pipelines, Feature Engineering & Selection** | Modular Stage 3 Data Preprocessing (MCAR/MAR/MNAR handlers, MICE iterative imputer, KNN imputer, Structural cleaner, Temporal UTC aligner, Numerical sanitizer), Stage 4 Feature Engineering (Interaction ratios, Smoothed Target Encoding with Bayesian m-estimates, Rare category consolidation), Feature Selection (Filter methods: VarianceThreshold, Correlation, SelectKBest; Wrapper methods: RFE; Embedded methods: L1 Lasso), and Data Leakage Prevention (Strict Fit-Transform separation). |
| **[Day 9](Day9.md)** | **CLT, Business Unit Economics, Clustering & Dimensionality Reduction** | Central Limit Theorem (CLT $n \ge 30$, Standard Error $SE = \sigma/\sqrt{n}$, Square Root Law), $R^2$ vs. Adjusted $R^2$ (penalizing feature bloat), Business Analytics KPIs (CLV/LTV, CAC, Payback period, Churn and Retention, Gold-standard 3:1 LTV:CAC ratio), Unsupervised Clustering (K-Means, Centroids, WCSS, Elbow method, Silhouette score, RFM segmentation, Agglomerative Hierarchical with Linkages, DBSCAN density core/border/noise=-1), and PCA vs. t-SNE. |
| **[Day 10](Day10.md)** | **Predictive Modeling, Hypothesis Testing, A/B Testing & Causal Inference** | Supervised Modeling (Logistic Regression, Sigmoid, Log Loss, Decision Trees CART, Gini vs. Entropy, Bagging Random Forest vs. Sequential Boosting XGBoost/LGBM, SVM margin), Classification Metrics (Confusion Matrix, Accuracy, Precision, Recall, Specificity, F1-Score, ROC-AUC vs. PR-AUC), Regression Regularization (Ridge L2 vs. Lasso L1 feature selection), Stratified K-Fold CV, Optuna Bayesian tuning, SHAP explainability (Game theory Shapley values), Hypothesis Testing (Type I $\alpha$ vs. Type II $\beta$ error matrix, Power, One-Sample/Two-Sample/Paired $t$-tests, One-Way ANOVA $F$-test and Tukey HSD, Chi-Square test of independence), A/B Testing design (MDE, Peeking problem, SRM, Multi-Armed Bandits), and Causal Inference (Difference-in-Differences, PSM, RDD). |

---

## 🎯 High-Yield MCQ Quick Revision Tips

1. **Memory & Performance:**
   - NumPy arrays use contiguous C memory; Python lists store scattered object pointers.
   - Basic slicing creates a **view** (modifies original array); fancy indexing creates an independent **copy**.
   - `s.astype('category')` reduces string memory by up to 90%+.
2. **Statistics & Distributions:**
   - Pythagorean Means: $HM \le GM \le AM$. Equality holds only when all values are equal.
   - For an Exponential distribution: $\sigma = \mu = 1/\lambda$.
   - For a Poisson distribution: $\text{Mean} = \text{Variance} = \lambda$.
   - Right-Skewed distribution: $\text{Mean} > \text{Median} > \text{Mode}$.
   - Standard Normal distribution: Kurtosis $= 3$, Excess Kurtosis $= 0$.
   - Standard Error: $SE = \sigma / \sqrt{n}$. Quadrupling sample size cuts $SE$ in half.
3. **Algorithms & Models:**
   - Random Forest (Bagging) builds trees in **parallel** to reduce **variance**.
   - Gradient Boosting (Boosting) builds trees **sequentially** to reduce **bias**.
   - Ridge (L2) shrinks coefficients; Lasso (L1) drives coefficients strictly to zero (feature selection).
   - DBSCAN labels outliers as cluster `-1` and does not require pre-specifying $K$.
   - Isolation Forest isolates anomalies using fewer random splits (**shorter path length**).
4. **Validation & Leakage:**
   - Always call `fit()` or `fit_transform()` exclusively on the **training set**; call `transform()` on test/validation sets.
   - Use **Stratified K-Fold** for classification to maintain class proportions in every fold.
   - Continuous monitoring in A/B testing (the **Peeking Problem**) inflates the Type I error rate.

