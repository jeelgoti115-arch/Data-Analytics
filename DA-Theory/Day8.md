# Day 8: Production Preprocessing Pipelines, Feature Engineering & Selection

---

## 1. Production Data Preprocessing Architecture (Stage 3)

In professional analytics engineering, preprocessing is organized as a modular, deterministic pipeline orchestrated through configuration rules.

```
+------------------------------------------------------------------------------------+
|                Stage 3: Data Preprocessing & Cleaning Pipeline                     |
+------------------------------------------------------------------------------------+
|                                                                                    |
| Raw Ingested Data  ──► [1. Missingness Handler]  ──► [2. Structural Cleaner]        |
|                         (MCAR/MAR/MNAR Dispatch)      (Dedupe, regex, casing)      |
|                                                               │                    |
|                                                               ▼                    |
| Clean Processed    ◄── [5. EDA Profiler]         ◄── [3. Outlier Detector]         |
| Telemetry Artifact      (Distribution telemetry)      (IQR Capping, IsoForest)     |
|                                ▲                              │                    |
|                                │                              ▼                    |
|                        [Numerical Sanitizer]     ◄── [4. Temporal Aligner]         |
|                         (Inf/-Inf, Zero-var)          (UTC, Resample, Gap-fill)    |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

---

### 1.1 Missingness Handling Strategies (Stage 3 Module 1)

Missing values must be addressed based on their underlying statistical mechanism (MCAR, MAR, MNAR):

| Imputation Strategy | Method / Algorithm | Advantages | Disadvantages & Risks |
| :--- | :--- | :--- | :--- |
| **Median Imputation** | Replaces nulls with the column median. | Fast, non-parametric, robust to extreme outliers. | Distorts the variance and shape of the distribution; deflates standard deviation. |
| **Mean Imputation** | Replaces nulls with the arithmetic mean. | Simple; preserves overall sample sum. | **Severely distorted by outliers**; deflates variance. |
| **Mode / Constant** | Replaces nulls with the most frequent category or `'Unknown'`. | Retains category counts; works for nominal text. | Over-amplifies the mode category. |
| **Forward / Backward Fill** | `ffill()` propagates the last valid value forward; `bfill()` propagates next value backward. | Essential for **ordered time-series telemetry**. | Unsuitable for cross-sectional or non-sequential data. |
| **KNN Imputation** | Replaces nulls with the average of the $k$-nearest neighbors in feature space. | Preserves multivariate relationships and correlations. | Computationally slow ($O(N^2)$); sensitive to feature scaling and outliers. |
| **MICE (Iterative Imputer)**| **Multivariate Imputation by Chained Equations**: Models each feature with missingness as a regression function of all other features iteratively. | Highest statistical precision; preserves complex relationships. | High computational complexity; requires parametric assumptions. |

---

### 1.2 Structural Cleaning & Numerical Sanitization

1. **Text Standardization:** Stripping leading/trailing whitespace (`.str.strip()`), standardizing casing (`.str.lower()`), and normalizing Unicode symbols.
2. **Deduplication:** Dropping duplicate records based on primary business keys (`drop_duplicates(subset=['id'], keep='last')`).
3. **Floating-Point Sanitization:**
   - Replacing mathematical infinities (`np.inf`, `-np.inf`) produced by division-by-zero with `np.nan` before imputation.
   - Enforcing domain value boundaries (e.g., `age` must lie in $[0, 120]$; `percentage` in $[0, 100]$).
4. **Zero-Variance Feature Removal:**
   - Dropping constant columns where all rows have the identical value ($\text{Variance} = 0$). Constant columns provide zero analytical or predictive power.

---

### 1.3 Temporal Alignment
Real-world time-series logs arrive with asynchronous, irregular timestamps:
- **UTC Timezone Normalization:** Converting all timestamps to universal Coordinated Universal Time (`tz_convert('UTC')`).
- **Resampling & Regular Frequency Alignment:** Aggregating asynchronous events into fixed daily, hourly, or 15-minute intervals (`df.resample('1H').mean()`).
- **Gap Imputation:** Filling missing operational intervals using time-weighted interpolation.

---

## 2. Advanced Feature Engineering & Transformation (Stage 4)

Feature engineering bridges raw preprocessed telemetry and high-performance machine learning models.

```
+------------------------------------------------------------------------------------+
|               Stage 4: Feature Engineering & Selection Pipeline                    |
+------------------------------------------------------------------------------------+
|                                                                                    |
| Clean Data ──► [Domain Feature Builder] ──► [Categorical Encoders]                 |
|                (Ratios, Interactions)        (One-Hot, Smoothed Target)            |
|                                                       │                            |
|                                                       ▼                            |
| Final Feature ◄── [Data Leakage Validator] ◄── [Feature Selector]                  |
| Matrix            (Train-fit isolation)        (Filter, Wrapper, Embedded)         |
|                                                       ▲                            |
|                                                       │                            |
|                   [Dimensionality Reducer]  ──► [Numerical Scalers]                |
|                    (PCA, Rare consolidation)    (Robust, Standard, Box-Cox)        |
|                                                                                    |
+------------------------------------------------------------------------------------+
```

---

### 2.1 Domain Feature Engineering & Interaction Terms
- **Interaction Features:** Capturing joint effects by multiplying or dividing features:
  $$\text{Feature}_{\text{interaction}} = X_1 \times X_2$$
  - Example: $\text{Total Load} = \text{Voltage} \times \text{Current}$.
- **Financial & Operational Ratios:**
  $$\text{Debt-to-Income (DTI)} = \frac{\text{Total Monthly Debt Payments}}{\text{Gross Monthly Income}}$$
  $$\text{Average Revenue Per Unit (ARPU)} = \frac{\text{Total Revenue}}{\text{Active Subscriptions}}$$

---

### 2.2 Advanced Categorical Encoding: Smoothed Target Encoding

While One-Hot Encoding works well for low-cardinality features, it causes the **Curse of Dimensionality** when a categorical feature has hundreds of unique levels (e.g., `Postal_Code`, `City`).

- **Naive Target Encoding:** Replaces each category with the average target value ($\bar{y}_i$) of that category.
  - **Fatal Flaw:** Rare categories (e.g., a city with only 1 customer who churned) get an extreme target value ($1.0$), causing severe overfitting and target leakage.
- **Smoothed Target Encoding (Bayesian m-estimate):**
  Blends the category mean with the overall global prior mean, weighted by sample size:
  $$\hat{S}_i = \frac{n_i \cdot \bar{y}_i + m \cdot \bar{y}_{\text{global}}}{n_i + m}$$
  where:
  - $n_i$: Number of observations in category $i$.
  - $\bar{y}_i$: Mean target value for category $i$.
  - $\bar{y}_{\text{global}}$: Overall target mean across the entire dataset.
  - $m$: Smoothing weight (smoothing parameter).
  - **Behavior:** When sample size $n_i$ is small, the estimate shrinks safely toward the global mean; when $n_i$ is large, the category mean dominates.

---

### 2.3 Cardinality & Dimensionality Reduction
- **Rare Category Consolidation:**
  - Categories appearing in less than $1\%$ or $5\%$ of total rows are combined into a single catch-all category: `'Other'`.
  - Prevents sparse one-hot matrices and simplifies categorical encoding.

---

## 3. Multistage Feature Selection Methodologies

Feature selection identifies the optimal subset of relevant predictors, reducing overfitting, improving model interpretability, and accelerating training times.

```
+------------------------------------------------------------------------------------+
|                         Feature Selection Strategies                               |
+------------------------------------------------------------------------------------+
| Paradigm         Mechanism                          Pros               Cons        |
| ---------------------------------------------------------------------------------- |
| Filter Methods   Statistical scoring independent    Fastest; $O(P)$;   Ignores     |
|                  of any ML model (Variance, Corr,   model-agnostic.    feature     |
|                  ANOVA F-test, Mutual Info).                           interactions|
| ---------------------------------------------------------------------------------- |
| Wrapper Methods  Iteratively trains an ML model     Captures feature   Extremely   |
|                  evaluating subsets (RFE,           interactions;      slow; risk  |
|                  Forward/Backward Selection).       high accuracy.     of overfit. |
| ---------------------------------------------------------------------------------- |
| Embedded Methods Built directly into model training Balances speed     Tied to a   |
|                  process (L1 Lasso, Tree            and accuracy;      specific    |
|                  feature importances).              robust.            algorithm.  |
+------------------------------------------------------------------------------------+
```

---

### 3.1 Filter Methods
1. **Variance Threshold:** Drops continuous features whose variance falls below a threshold ($\text{Var}(X) \le \tau$). Removes constants and near-constant noise.
2. **Correlation Filtering:** Computes the pairwise Pearson correlation matrix and drops one feature from any pair where $|r| > 0.85$ to eliminate multicollinearity.
3. **Statistical Hypothesis Selectors (`SelectKBest`):**
   - **ANOVA F-test (`f_classif`):** Tests for linear differences in continuous features across categorical classes.
   - **Chi-Square (`chi2`):** Tests independence between non-negative categorical features and categorical targets.
   - **Mutual Information (`mutual_info_classif`):** Non-parametric measure quantifying the reduction in uncertainty of target $Y$ given feature $X$. **Captures complex non-linear relationships that Pearson correlation misses!**

---

### 3.2 Wrapper Methods: Recursive Feature Elimination (RFE)
- **Algorithm:**
  1. Train an estimator (e.g., Logistic Regression, Random Forest) on the full set of features.
  2. Compute feature importances or coefficient magnitudes ($|w_j|$).
  3. Prune the least important feature(s).
  4. Retrain the model on the remaining features and repeat until the desired $K$ features remain.

---

### 3.3 Embedded Methods: L1 Lasso Regularization
Lasso regression applies an $L_1$ penalty to the sum of absolute coefficients:
$$\text{Loss} = \text{MSE} + \lambda \sum_{j=1}^p |w_j|$$
- The diamond geometry of the $L_1$ penalty drives non-essential feature coefficients **strictly to zero ($w_j = 0$)**, performing automatic feature selection!

---

## 4. The Golden Rule of ML: Data Leakage Prevention

**Data Leakage** occurs when information from outside the training dataset is inadvertently used to train a machine learning model, creating unrealistically high training/validation scores that fail disastrously in production.

```
WRONG PIPELINE (DATA LEAKAGE):
  Full Dataset ──► [Fit Scaler / Imputer] ──► [Train-Test Split] ──► Model Evaluated
                   (Test info leaks into train!)

CORRECT PIPELINE (STRICT ISOLATION):
  Full Dataset ──► [Train-Test Split]
                         │
                         ├── Train Split ──► [Fit & Transform Scaler] ──► Train Model
                         │                                                      │
                         └── Test Split  ──► [Transform ONLY]        ──► Evaluate
```

### 4.1 Strict Fit-Transform Separation
- **`fit()` or `fit_transform()`:** Must be called **ONLY on the Training Split**!
- **`transform()`:** Applied to the Validation and Test splits using the parameters (mean, standard deviation, median, categories) learned from the training set.
- **Classic Leakage Trap:** Computing the dataset-wide mean and filling missing values before splitting into train/test leaks the future distribution of the test set into the training set!

### 4.2 Target Leakage
Occurs when a predictor feature incorporates information that is only known *after* the target event has occurred.
- Example: Predicting whether a bank customer will default on a loan using the feature `Collection_Agency_Referral_Date`. A collection agency is only called *after* a customer defaults!

---

## 5. MCQ Examination Focus & Common Confusion Points

### Key Facts for MCQs
1. **Fit-Transform Separation:** `fit()` is called exclusively on training data; test data is processed using `transform()` only.
2. **L1 Lasso Regularization:** Drives non-essential coefficients *strictly to zero*, performing embedded feature selection.
3. **Filter vs Wrapper vs Embedded:** Filter methods are model-agnostic and fastest; Wrapper methods (RFE) iteratively train models; Embedded methods perform selection during model training.
4. **Mutual Information Advantage:** Detects non-linear dependencies between features and targets where Pearson correlation fails.
5. **Smoothed Target Encoding:** Blends category mean with global prior mean to prevent overfitting on rare categories.
6. **MICE Imputation Principle:** Imputes missing values by iteratively modeling each feature as a function of all other features.
7. **Zero-Variance Features:** Features with $\text{Variance} = 0$ are constants and provide zero predictive signal; they should always be eliminated.

### Common Confusion Traps Table

| Concept A | Concept B | Difference / Confusion Trap |
| :--- | :--- | :--- |
| Filter Methods | Wrapper Methods | Filter selects features using *statistical tests* (fast, no model); Wrapper evaluates subsets by *training models* (slow). |
| Data Leakage | Overfitting | Overfitting is high variance from learning noise; Data Leakage is *contaminating train data with test or future target information*. |
| `fit_transform()` | `transform()` | `fit_transform()` calculates parameters and transforms (train only); `transform()` applies existing parameters (test/val). |
| Target Encoding | One-Hot Encoding | Target encoding replaces categories with *target averages* (1 column); One-hot creates *binary flags* ($k-1$ columns). |
| Box-Cox Transform | Yeo-Johnson Transform | Box-Cox requires strictly positive values ($x > 0$); Yeo-Johnson handles negative and zero values. |
| KNN Imputer | Mean Imputer | Mean imputes a single global value; KNN imputes based on the multivariate similarity of nearest sample neighbors. |

