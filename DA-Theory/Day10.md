# Day 10: Predictive Modeling, Hypothesis Testing, A/B Testing & Causal Inference

---

## 1. Supervised Machine Learning: Classification Paradigms

**Supervised Learning** trains predictive models on paired feature matrices $X$ and ground-truth target vectors $y$.

```
+------------------------------------------------------------------------------------+
|                         Classification Algorithms Compared                         |
+------------------------------------------------------------------------------------+
| Algorithm        Model Class      Key Mechanism               Hyperparameters      |
| ---------------------------------------------------------------------------------- |
| Logistic Reg.    Generalized Lin. Sigmoid activation of      C (regularization),  |
|                                   linear log-odds.            penalty ('l1', 'l2') |
| ---------------------------------------------------------------------------------- |
| Decision Tree    Tree-Based       Recursive partitioning via  max_depth,           |
|                                   Gini or Information Gain.   min_samples_split    |
| ---------------------------------------------------------------------------------- |
| Random Forest    Bagging Ensemble Parallel bootstrap trees;   n_estimators,        |
|                                   averages predictions.       max_features         |
| ---------------------------------------------------------------------------------- |
| Gradient Boost   Boosting EnsembleSequential trees fitting    learning_rate,       |
| (XGBoost/LGBM)                    pseudo-residuals.           n_estimators, depth  |
| ---------------------------------------------------------------------------------- |
| SVM              Margin-Based     Maximizes margin boundary;  C (penalty),         |
|                                   Kernel trick (RBF).         gamma, kernel        |
+------------------------------------------------------------------------------------+
```

---

### 1.1 Logistic Regression
Models the posterior probability of a binary outcome ($y \in \{0, 1\}$).
- **Log-Odds (Logit Transformation):**
  $$\ln\left(\frac{p}{1 - p}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_p X_p$$
- **Sigmoid Function:** Maps real numbers from $(-\infty, +\infty)$ to probabilities in $[0, 1]$:
  $$\sigma(z) = \frac{1}{1 + e^{-z}}$$
- **Loss Function:** **Binary Cross-Entropy (Log Loss)**:
  $$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \ln(p_i) + (1 - y_i) \ln(1 - p_i) \right]$$

---

### 1.2 Decision Trees (CART)
Recursively partitions feature space using orthogonal decision boundaries.
- **Splitting Criteria:**
  - **Gini Impurity:** Measures the probability of misclassifying a randomly chosen element:
    $$\text{Gini} = 1 - \sum_{i=1}^C p_i^2$$
    A pure node has $\text{Gini} = 0.0$.
  - **Entropy & Information Gain:**
    $$\text{Entropy}(S) = -\sum_{i=1}^C p_i \log_2(p_i)$$
    $$\text{Information Gain} = \text{Entropy}(\text{Parent}) - \sum \frac{|S_v|}{|S|} \text{Entropy}(S_v)$$
- **Major Risk:** **Severe Overfitting** (low bias, high variance) if allowed to grow unconstrained $\to$ must prune via `max_depth` and `min_samples_leaf`.

---

### 1.3 Ensemble Architectures: Bagging vs. Boosting

A fundamental comparison tested in all Machine Learning and Data Analytics exams:

| Dimension | Bagging (e.g., Random Forest) | Boosting (e.g., XGBoost, LightGBM) |
| :--- | :--- | :--- |
| **Execution Architecture** | **Parallel:** All trees grow independently at the same time. | **Sequential:** Trees grow one after another in a chained sequence. |
| **Core Objective** | **Variance Reduction:** Averages diverse decorrelated trees to prevent overfitting. | **Bias Reduction:** Focuses new trees on correcting the mistakes/residuals of prior trees. |
| **Base Models** | Deep, unpruned trees (low bias, high variance). | Shallow weak learners / stumps (high bias, low variance). |
| **Sampling Mechanism** | Bootstrap sampling with replacement (Bagging) + random feature subsets. | Reweighting samples (AdaBoost) or gradient descent on residuals (Gradient Boosting). |
| **Overfitting Risk** | Extremely resistant to overfitting as `n_estimators` increases. | Can overfit if `n_estimators` is too large without shrinkage (`learning_rate`). |

---

## 2. Classification Evaluation Metrics & Confusion Matrix

Evaluating classification performance on raw accuracy alone is dangerous, especially in imbalanced datasets (e.g., 99% non-fraud, 1% fraud).

```
+-----------------------------------------------------------------------------+
|                          The Confusion Matrix                               |
+-----------------------------------------------------------------------------+
|                                    Actual Positive (1)   Actual Negative (0)|
| Predicted Positive (1)             True Positive (TP)    False Positive (FP)|
|                                                          (Type I Error)     |
| Predicted Negative (0)             False Negative (FN)   True Negative (TN) |
|                                    (Type II Error)                          |
+-----------------------------------------------------------------------------+
```

### 2.1 The Core Metric Formulas

| Metric | Mathematical Formula | Business Interpretation & Best Use Case |
| :--- | :--- | :--- |
| **Accuracy** | $\frac{TP + TN}{TP + TN + FP + FN}$ | Overall percentage correct. **Misleading for imbalanced datasets!** |
| **Precision** | $\frac{TP}{TP + FP}$ | Proportion of positive calls that were truly positive. **Crucial when False Positives are costly** (e.g., Email Spam filter). |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Proportion of actual positives successfully captured. **Crucial when False Negatives are catastrophic** (e.g., Cancer detection, Fraud). |
| **Specificity** | $\frac{TN}{TN + FP}$ | True Negative Rate. Proportion of actual negatives correctly identified. |
| **F1-Score** | $\frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ | **Harmonic Mean** balancing Precision and Recall. Essential for imbalanced datasets. |

### 2.2 Threshold Curves: ROC-AUC vs. PR-AUC
- **ROC Curve (Receiver Operating Characteristic):** Plots **True Positive Rate (Recall)** against **False Positive Rate ($1 - \text{Specificity}$)** across all possible decision thresholds.
  - $\text{AUC} = 0.5$: Worthless random guess.
  - $\text{AUC} = 1.0$: Flawless separation.
- **PR-AUC (Precision-Recall Curve):** Preferred over ROC-AUC when evaluating **severely imbalanced target classes** (e.g., fraud rate $< 1\%$) because it is not inflated by massive numbers of True Negatives.

---

## 3. Regularized Regression & Goodness of Fit

### 3.1 Linear Regression Gauss-Markov Assumptions
Ordinary Least Squares (OLS) produces the Best Linear Unbiased Estimator (BLUE) if:
1. **Linearity:** Relationship between predictors and target is linear in parameters.
2. **Exogeneity:** Errors have mean zero conditional on $X$: $E[\epsilon \mid X] = 0$.
3. **Homoscedasticity:** Constant error variance across all levels of $X$.
4. **No Autocorrelation:** Errors are independent of each other.
5. **No Multicollinearity:** Independent predictors are not perfectly collinear.

### 3.2 Regularization Techniques Compared

Regularization prevents overfitting by adding a penalty term to the sum of squared errors:

```
+------------------------------------------------------------------------------------+
|                         Regularization Families Compared                           |
+------------------------------------------------------------------------------------+
| Method          Penalty Term                     Coefficient Effect  Feature Select?|
| ---------------------------------------------------------------------------------- |
| Ridge (L2)      λ · Σ w_j²                       Shrinks toward 0;   NO (Keeps all) |
|                                                  never exactly zero.                |
| Lasso (L1)      λ · Σ |w_j|                      Drives non-essentialYES (Sparse    |
|                                                  weights to ZERO!    coefficients)  |
| ElasticNet      r·λ·Σ|w_j| + ((1-r)/2)·λ·Σw_j²   Combines L1 and L2. YES            |
+------------------------------------------------------------------------------------+
```

---

## 4. Cross-Validation, Tuning & Interpretability (SHAP)

### 4.1 Cross-Validation Strategies
- **K-Fold CV:** Randomly splits data into $K$ equal folds; trains on $K-1$, validates on 1.
- **Stratified K-Fold:** Enforces that **each fold contains the exact same class proportion** as the complete dataset. **Mandatory for classification problems!**
- **TimeSeriesSplit (Temporal Split):** Expanding walk-forward window without shuffling to prevent **look-ahead bias** in time-dependent data.

### 4.2 Bayesian Hyperparameter Optimization with Optuna
Unlike brute-force Grid Search or blind Random Search, **Optuna** uses a **Tree-structured Parzen Estimator (TPE)** to model the probability distribution of past trial performances, intelligently focusing search efforts on high-performing hyperparameter spaces.

---

### 4.3 Model Explainability: SHAP (SHapley Additive exPlanations)

Traditional tree feature importance (MDI - Mean Decrease in Impurity) is severely flawed: it is biased toward high-cardinality features and cannot explain *why* an individual prediction was made.

- **SHAP Theoretical Basis:** Grounded in cooperative game theory (Lloyd Shapley).
- **Core Mechanism:** Computes the marginal contribution of feature $i$ across all possible feature coalitions $S$:
  $$\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} \left[ f(S \cup \{i\}) - f(S) \right]$$
- **Additive Feature Attribution:** The model prediction equals the base value plus the sum of all SHAP values:
  $$f(x) = \phi_0 + \sum_{i=1}^P \phi_i$$

#### SHAP Visualizations:
1. **Summary Beeswarm Plot:** Global overview showing feature importance ranking and whether high/low values drive positive or negative predictions.
2. **Waterfall & Force Plots:** Local breakdown explaining a single customer prediction step-by-step.

---

## 5. Statistical Hypothesis Testing Decision Matrix

Hypothesis testing provides formal mathematical criteria to determine whether observed differences are statistically significant or attributable to random sampling noise.

```
+-----------------------------------------------------------------------------+
|                         Hypothesis Testing Terminology                      |
+-----------------------------------------------------------------------------+
| Null Hypothesis (H0):         Status quo; no difference; no treatment effect|
| Alternative Hypothesis (H1):  Research claim; genuine difference exists     |
| Significance Level (α):       Threshold for rejecting H0 (typically 0.05)   |
| p-value:                      Probability of obtaining results as extreme   |
|                               as observed, assuming H0 is true.             |
| Decision Rule:                Reject H0 if p-value <= α                     |
+-----------------------------------------------------------------------------+
```

### 5.1 The Error Matrix (Type I vs. Type II Errors)

A staple concept on all Data Analytics certification exams:

```
+------------------------------------------------------------------------------------+
|                               Reality: H0 is TRUE           Reality: H0 is FALSE   |
| ---------------------------------------------------------------------------------- |
| Decision: Reject H0           TYPE I ERROR (α)              CORRECT DECISION       |
|                               False Positive                Power (1 - β)          |
| ---------------------------------------------------------------------------------- |
| Decision: Fail to Reject H0   CORRECT DECISION              TYPE II ERROR (β)      |
|                               Confidence (1 - α)            False Negative         |
+------------------------------------------------------------------------------------+
```

- **Type I Error ($\alpha$):** False Positive. Rejecting the null hypothesis when it was actually true (e.g., convicting an innocent person; sending healthy patient to surgery).
- **Type II Error ($\beta$):** False Negative. Failing to reject the null hypothesis when it was actually false (e.g., letting a guilty person walk free; missing a malignant tumor).
- **Statistical Power ($1 - \beta$):** The probability of correctly rejecting a false null hypothesis (industry standard benchmark: $\ge 80\%$).

---

### 5.2 Statistical Tests Decision Framework

Selecting the correct statistical test based on data types and sample groups:

```
+------------------------------------------------------------------------------------+
|                         Statistical Tests Decision Matrix                          |
+------------------------------------------------------------------------------------+
| Question / Scenario                Groups Compared        Appropriate Test         |
| ---------------------------------------------------------------------------------- |
| Compare 1 sample mean to constant  1 Sample vs. Constant  One-Sample t-Test        |
| (Population variance unknown)                                                      |
| ---------------------------------------------------------------------------------- |
| Compare means of 2 independent     2 Independent Groups   Two-Sample Independent   |
| groups (e.g., Control vs Variant)                         t-Test (Welch's t-test)  |
| ---------------------------------------------------------------------------------- |
| Compare 2 dependent measurements   Same subjects          Paired-Samples t-Test    |
| on same subjects (Before vs After) (Matched pairs)                                 |
| ---------------------------------------------------------------------------------- |
| Compare means of 3 OR MORE         3+ Independent Groups  One-Way ANOVA            |
| groups simultaneously                                     (Followed by Tukey HSD)  |
| ---------------------------------------------------------------------------------- |
| Compare relationship between 2     Categorical vs. Cat.   Chi-Square (χ²) Test     |
| categorical variables                                     of Independence          |
| ---------------------------------------------------------------------------------- |
| Compare means or proportions when  Large Sample (n >= 30) Z-Test                   |
| population variance is known       or known σ                                      |
+------------------------------------------------------------------------------------+
```

#### One-Way ANOVA ($F$-Test) & Post-Hoc Tukey HSD
- **ANOVA Hypothesis:** $H_0: \mu_1 = \mu_2 = \dots = \mu_k$ (all group means are equal).
- **$F$-Statistic Formula:**
  $$F = \frac{\text{Between-Group Variance (MSB)}}{\text{Within-Group Variance (MSW)}}$$
- **Post-Hoc Analysis:** If the ANOVA $p$-value is significant ($p \le 0.05$), it proves *at least one* group mean differs, but **does not say which one**. Analysts must run **Tukey's HSD (Honestly Significant Difference)** post-hoc test to identify the specific differing group pairs while controlling the Family-Wise Error Rate.

#### Chi-Square ($\chi^2$) Test of Independence
Assesses association in an $r \times c$ contingency table:
$$\chi^2 = \sum \frac{(O - E)^2}{E}, \quad E = \frac{\text{Row Total} \times \text{Column Total}}{\text{Grand Total}}$$
$$\text{Degrees of Freedom} = (r - 1)(c - 1)$$

---

## 6. A/B Testing, Experimentation & Causal Inference

### 6.1 A/B Experimentation Design
- **Primary / Evaluation Metric:** The single North Star KPI determining the test outcome (e.g., Checkout Conversion Rate, Revenue Per User).
- **Guardrail Metrics:** Health metrics that must not degrade (e.g., page load latency, bounce rate, customer churn).
- **Sample Size Determination Formula:**
  $$n = \frac{2(Z_{\alpha/2} + Z_{\beta})^2 \sigma^2}{\delta^2}$$
  where $\delta$ is the **Minimum Detectable Effect (MDE)**. Smaller MDEs require exponentially larger sample sizes!

### 6.2 Common Pitfalls in A/B Testing
1. **The Peeking Problem (Optional Stopping):**
   - Continuously monitoring $p$-values every day and stopping the experiment as soon as $p \le 0.05$ **severely inflates the Type I error rate** from $5\%$ to $30\%+$!
   - Tests must run until the pre-calculated sample size is reached.
2. **Sample Ratio Mismatch (SRM):**
   - When the observed traffic split (e.g., 47% Control, 53% Variant) deviates from the intended design (50/50). Signals technical flaws such as redirect bugs, selective bot filtering, or browser crashes.

### 6.3 Multi-Armed Bandits (MAB)
While traditional A/B tests maintain fixed traffic allocation throughout, **Multi-Armed Bandit** algorithms (e.g., $\epsilon$-Greedy, Thompson Sampling, Upper Confidence Bound) dynamically shift traffic to the winning variant *during the test*, minimizing opportunity loss.

---

### 6.4 Quasi-Experiments & Causal Inference

When randomized controlled trials (A/B tests) are impossible, ethical, or too costly, analysts apply causal inference techniques to observational data:

```
+------------------------------------------------------------------------------------+
|                         Causal Inference Quasi-Experiments                         |
+------------------------------------------------------------------------------------+
| Method                     Core Mechanism                   Key Assumption         |
| ---------------------------------------------------------------------------------- |
| Difference-in-Differences  Compares before-and-after change  PARALLEL TRENDS        |
| (DiD)                      between treatment & control.     assumption.            |
| ---------------------------------------------------------------------------------- |
| Propensity Score Matching  Matches treated units with       No unobserved          |
| (PSM)                      controls having identical        confounders.           |
|                            treatment probability.                                  |
| ---------------------------------------------------------------------------------- |
| Regression Discontinuity   Exploits a strict cut-off rule   Units near cut-off are |
| Design (RDD)               (e.g., credit score >= 700).     as good as randomized. |
+------------------------------------------------------------------------------------+
```

---

## 7. MCQ Examination Focus & Common Confusion Points

### Key Facts for MCQs
1. **Type I Error ($\alpha$):** Rejecting true $H_0$ (False Positive).
2. **Type II Error ($\beta$):** Failing to reject false $H_0$ (False Negative).
3. **Statistical Power:** Equal to $1 - \beta$; benchmark is $\ge 0.80$.
4. **Lasso vs. Ridge:** Lasso (L1) sets coefficients strictly to zero (feature selection); Ridge (L2) shrinks them toward zero.
5. **Precision vs. Recall:** Precision is critical when FP is costly; Recall is critical when FN is dangerous.
6. **ANOVA Requirement:** Used to compare means of 3 or more independent groups; followed by Tukey HSD for pairwise comparisons.
7. **Peeking Problem in A/B Testing:** Inflates the Type I error rate.
8. **Stratified K-Fold:** Preserves target class distribution in every fold; mandatory for classification.
9. **SHAP Foundation:** Built on cooperative game theory Shapley values; provides consistent, additive feature attributions.
10. **Difference-in-Differences Assumption:** Relies strictly on the Parallel Trends assumption.

### Common Confusion Traps Table

| Concept A | Concept B | Difference / Confusion Trap |
| :--- | :--- | :--- |
| Type I Error | Type II Error | Type I is *False Positive* (convict innocent); Type II is *False Negative* (acquit guilty). |
| Ridge Regression (L2) | Lasso Regression (L1) | Ridge shrinks weights towards zero; Lasso drives weights *strictly to zero* (feature selection). |
| Precision | Recall | Precision is $\frac{TP}{TP + FP}$ (accuracy of positive calls); Recall is $\frac{TP}{TP + FN}$ (coverage of actual positives). |
| ROC-AUC | PR-AUC | ROC-AUC is distorted by large True Negatives; PR-AUC is preferred for *heavily imbalanced classes*. |
| Bagging (Random Forest) | Boosting (XGBoost) | Bagging builds trees in *parallel* to reduce *variance*; Boosting builds trees *sequentially* to reduce *bias*. |
| t-Test | ANOVA | t-test compares means of *2* groups; ANOVA compares means of *3 or more* groups. |
| Student's t-test | Welch's t-test | Student's assumes *equal variances*; Welch's accommodates *unequal variances*. |
| Decision Tree MDI | SHAP | MDI is biased toward high-cardinality features; SHAP provides mathematically consistent, unbiased attribution. |

