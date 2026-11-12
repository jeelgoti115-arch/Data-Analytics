"""
O04_cardinality_dimensionality_reducer.py
Groups rare categorical levels into 'Other' and performs PCA/MCA dimensionality reduction.
"""
"""
================================================================================
MODULE: Cardinality & Dimensionality Reduction Engine
FILE: src/O04_cardinality_dimensionality_reducer.py
PURPOSE:
  Groups rare categorical levels into an 'Other' category based on frequency 
  thresholds and applies Principal Component Analysis (PCA) for dimensionality 
  reduction. Persists learned category mappings to prevent data leakage between 
  train and test splits, parameterized by 'config/O01_feature_rules.yaml'.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CardinalityReducer")


class CardinalityDimensionalityReducer:
    """Manages rare label grouping and dimensionality reduction."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.frequent_categories = {}
        self.pca_transformer = None

    def _load_config(self) -> dict:
        """Loads cardinality reduction parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("cardinality_reduction", {})

    def group_rare_categories(self, df: pd.DataFrame, categorical_cols: list, is_train: bool = True) -> pd.DataFrame:
        """
        Groups categorical levels with frequency below the rare_label_threshold into 'Other'.
        """
        threshold = self.config.get("rare_label_threshold", 0.02)
        df_clean = df.copy()
        
        existing_cols = [c for c in categorical_cols if c in df_clean.columns]
        if not existing_cols:
            return df_clean

        if is_train:
            logger.info(f"Learning frequent categories (threshold >= {threshold * 100}%) for: {existing_cols}")
            self.frequent_categories = {}
            
            for col in existing_cols:
                frequencies = df_clean[col].value_counts(normalize=True)
                frequent_vals = frequencies[frequencies >= threshold].index.tolist()
                self.frequent_categories[col] = frequent_vals
                
                # Transform train split
                df_clean[col] = df_clean[col].apply(lambda x: x if x in frequent_vals else "Other")
                logger.info(f"Column '{col}': Retained {len(frequent_vals)} frequent categories, grouped remainder into 'Other'.")
        else:
            if not self.frequent_categories:
                raise ValueError("Frequent category maps not fitted. Run on train split first.")
            logger.info("Applying learned frequent categories to test split...")
            
            for col in existing_cols:
                frequent_vals = self.frequent_categories.get(col, [])
                df_clean[col] = df_clean[col].apply(lambda x: x if x in frequent_vals else "Other")

        return df_clean

    def apply_pca_reduction(self, df: pd.DataFrame, variance_ratio: float = 0.95, is_train: bool = True) -> pd.DataFrame:
        """
        Applies Principal Component Analysis (PCA) to compress numerical features 
        while preserving a specified cumulative explained variance ratio (e.g., 95%).
        """
        df_clean = df.copy()
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(num_cols) <= 2:
            return df_clean # PCA not necessary for low-dimensional spaces

        if is_train:
            logger.info(f"Fitting PCA to retain {variance_ratio * 100}% cumulative explained variance...")
            self.pca_transformer = PCA(n_components=variance_ratio, random_state=42)
            pca_array = self.pca_transformer.fit_transform(df_clean[num_cols])
            
            pca_cols = [f"pc_{i+1}" for i in range(pca_array.shape[1])]
            pca_df = pd.DataFrame(pca_array, columns=pca_cols, index=df_clean.index)
            
            # Drop original numerical columns and replace with principal components
            df_clean = pd.concat([df_clean.drop(columns=num_cols), pca_df], axis=1)
            logger.info(f"PCA reduced numerical feature space from {len(num_cols)} to {len(pca_cols)} components.")
        else:
            if self.pca_transformer is None:
                raise ValueError("PCA transformer not fitted. Run on train split first.")
            logger.info("Transforming test dataset using fitted PCA transformer...")
            
            pca_array = self.pca_transformer.transform(df_clean[num_cols])
            pca_cols = [f"pc_{i+1}" for i in range(pca_array.shape[1])]
            pca_df = pd.DataFrame(pca_array, columns=pca_cols, index=df_clean.index)
            
            df_clean = pd.concat([df_clean.drop(columns=num_cols), pca_df], axis=1)

        return df_clean


if __name__ == "__main__":
    # Local test demonstration
    config_file = Path("config/O01_feature_rules.yaml")
    
    sample_df = pd.DataFrame({
        "payment_method": ["electronic_check", "mailed_check", "rare_method_1", "rare_method_2", "electronic_check"],
        "feature_1": [1.2, 2.3, 1.1, 4.5, 2.2],
        "feature_2": [3.4, 1.2, 3.5, 8.9, 2.1]
    })
    
    if config_file.exists():
        reducer = CardinalityDimensionalityReducer(config_path=config_file)
        df_reduced = reducer.group_rare_categories(sample_df, categorical_cols=["payment_method"], is_train=True)
        print("\n[RESULT] Rare Category Grouped DataFrame Preview:")
        print(df_reduced)