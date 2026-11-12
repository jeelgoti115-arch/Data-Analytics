"""
O05_feature_selector.py
Multistage feature selection using variance thresholds, correlation filters, and Mutual Information.
"""
"""
================================================================================
MODULE: Multistage Feature Selector Engine
FILE: src/O05_feature_selector.py
PURPOSE:
  Executes multistage feature selection to eliminate noise and multicollinearity:
  1. Variance Threshold filtering (drops constant/near-zero variance columns).
  2. Correlation filtering (drops redundant features where absolute correlation > 0.85).
  3. Statistical scoring (Selects top-K features using Mutual Information).
  Parameterized by 'config/O01_feature_rules.yaml'.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, SelectKBest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("FeatureSelector")


class FeatureSelector:
    """Executes multistage feature selection to isolate high-signal features."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.variance_selector = None
        self.selected_features = []

    def _load_config(self) -> dict:
        """Loads feature selection parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("feature_selection", {})

    def apply_variance_threshold(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Removes features with variance below the configured threshold."""
        var_cfg = self.config.get("variance_threshold", {})
        if not var_cfg.get("enabled", True):
            return df

        threshold = var_cfg.get("threshold", 0.01)
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()

        if not num_cols:
            return df_clean

        if is_train:
            logger.info(f"Fitting VarianceThreshold (threshold={threshold}) on numerical features...")
            self.variance_selector = VarianceThreshold(threshold=threshold)
            self.variance_selector.fit(df_clean[num_cols])
            
            retained_cols = [col for col, keep in zip(num_cols, self.variance_selector.get_support()) if keep]
            dropped_cols = set(num_cols) - set(retained_cols)
            
            if dropped_cols:
                logger.info(f"VarianceThreshold dropped low-variance columns: {list(dropped_cols)}")
            
            non_num_cols = [c for c in df_clean.columns if c not in num_cols]
            df_clean = pd.concat([df_clean[non_num_cols], df_clean[retained_cols]], axis=1)
        else:
            if self.variance_selector is None:
                raise ValueError("VarianceThreshold selector not fitted. Run on train split first.")
            non_num_cols = [c for c in df_clean.columns if c not in num_cols]
            retained_cols = [col for col, keep in zip(num_cols, self.variance_selector.get_support()) if keep]
            df_clean = pd.concat([df_clean[non_num_cols], df_clean[retained_cols]], axis=1)

        return df_clean

    def apply_correlation_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drops one of any feature pair with absolute correlation exceeding threshold."""
        corr_cfg = self.config.get("correlation_filter", {})
        if not corr_cfg.get("enabled", True):
            return df

        threshold = corr_cfg.get("threshold", 0.85)
        df_clean = df.copy()
        num_df = df_clean.select_dtypes(include=[np.number])

        if num_df.empty or num_df.shape[1] <= 1:
            return df_clean

        logger.info(f"Applying correlation filter (|r| > {threshold})...")
        corr_matrix = num_df.corr().abs()
        
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]

        if to_drop:
            logger.info(f"Dropping {len(to_drop)} highly collinear features: {to_drop}")
            df_clean.drop(columns=to_drop, inplace=True)

        return df_clean

    def apply_statistical_scoring(self, df: pd.DataFrame, target_col: str, is_train: bool = True) -> pd.DataFrame:
        """Selects top-K features using statistical scoring (Mutual Information)."""
        stat_cfg = self.config.get("statistical_scoring", {})
        top_k = stat_cfg.get("top_k", 15)
        
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Skipping statistical feature selection.")
            return df

        df_clean = df.copy()
        feature_cols = [c for c in df_clean.columns if c != target_col and pd.api.types.is_numeric_dtype(df_clean[c])]

        if len(feature_cols) <= top_k:
            logger.info("Feature count is less than or equal to top_k limit. Skipping subset truncation.")
            self.selected_features = feature_cols
            return df_clean

        if is_train:
            logger.info(f"Selecting top {top_k} features using Mutual Information classification scoring...")
            X = df_clean[feature_cols]
            y = df_clean[target_col]

            selector = SelectKBest(score_func=mutual_info_classif, k=top_k)
            selector.fit(X, y)
            
            self.selected_features = [col for col, keep in zip(feature_cols, selector.get_support()) if keep]
            logger.info(f"Selected top features: {self.selected_features}")
            
            non_feature_cols = [c for c in df_clean.columns if c not in feature_cols]
            df_clean = pd.concat([df_clean[non_feature_cols], df_clean[self.selected_features]], axis=1)
        else:
            if not self.selected_features:
                raise ValueError("Selected features not fitted. Run on train split first.")
            logger.info("Filtering test split to retain selected features...")
            non_feature_cols = [c for c in df_clean.columns if c not in feature_cols or c in self.selected_features]
            df_clean = df_clean[non_feature_cols]

        return df_clean


if __name__ == "__main__":
    # Local test demonstration
    config_file = Path("config/O01_feature_rules.yaml")
    
    sample_df = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feature_b": [1.01, 2.02, 3.01, 4.02, 5.01], # Highly collinear with feature_a
        "feature_c": [5.0, 4.0, 3.0, 2.0, 1.0],
        "constant_feature": [1.0, 1.0, 1.0, 1.0, 1.0], # Zero variance
        "churn_status": [0, 1, 0, 1, 0]
    })
    
    if config_file.exists():
        selector = FeatureSelector(config_path=config_file)
        df_var = selector.apply_variance_threshold(sample_df, is_train=True)
        df_corr = selector.apply_correlation_filter(df_var)
        df_final = selector.apply_statistical_scoring(df_corr, target_col="churn_status", is_train=True)
        print("\n[RESULT] Selected Features DataFrame Preview:")
        print(df_final.head())