"""
O03_numerical_transformer.py
Applies power transformations (Yeo-Johnson/Box-Cox) and scaling (Standard/Robust).
"""
"""
================================================================================
MODULE: Numerical Transformer & Scaler Engine
FILE: src/O03_numerical_transformer.py
PURPOSE:
  Applies power transformations (Yeo-Johnson or Box-Cox) to correct feature skewness 
  and numerical scaling (RobustScaler or StandardScaler) to normalize feature ranges. 
  Enforces strict fit-transform separation between training and test sets, 
  parameterized by 'config/O01_feature_rules.yaml'.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import PowerTransformer, StandardScaler, RobustScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("NumericalTransformer")


class NumericalTransformer:
    """Applies power transformations for skewness and feature scaling."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.power_transformer = None
        self.scaler = None
        self.transformed_cols = []

    def _load_config(self) -> dict:
        """Loads numerical transformation parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("numerical_transformation", {})

    def fit_transform_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Applies power transformations and scaling to numerical features.
        
        Args:
            df (pd.DataFrame): Input DataFrame.
            is_train (bool): True if fitting on training split, False for test split.
            
        Returns:
            pd.DataFrame: Transformed and scaled DataFrame.
        """
        df_clean = df.copy()
        
        # Identify numerical columns (excluding binary dummies or IDs)
        num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        # Exclude columns that are already encoded or binary flags if needed
        num_cols = [c for c in num_cols if not c.endswith("_is_missing") and c != "is_multivariate_anomaly"]

        if not num_cols:
            return df_clean

        power_cfg = self.config.get("power_transform", {})
        scaling_cfg = self.config.get("scaling", {})

        # 1. Power Transformation (Skewness correction)
        if power_cfg.get("enabled", True):
            method = power_cfg.get("method", "yeo-johnson")
            if is_train:
                logger.info(f"Fitting PowerTransformer ({method}) on numerical columns: {num_cols}")
                self.power_transformer = PowerTransformer(method=method, standardize=False)
                transformed_array = self.power_transformer.fit_transform(df_clean[num_cols])
            else:
                if self.power_transformer is None:
                    raise ValueError("PowerTransformer not fitted. Run on train split first.")
                logger.info("Transforming numerical columns using fitted PowerTransformer...")
                transformed_array = self.power_transformer.transform(df_clean[num_cols])
                
            df_clean[num_cols] = transformed_array

        # 2. Scaling (Standardization)
        scale_method = scaling_cfg.get("method", "robust")
        if is_train:
            logger.info(f"Fitting Scaler ({scale_method}) on numerical features...")
            if scale_method == "robust":
                self.scaler = RobustScaler() # Robust to remaining outliers using median/IQR
            else:
                self.scaler = StandardScaler()
                
            scaled_array = self.scaler.fit_transform(df_clean[num_cols])
        else:
            if self.scaler is None:
                raise ValueError("Scaler not fitted. Run on train split first.")
            logger.info(f"Scaling numerical features using fitted {scale_method} scaler...")
            scaled_array = self.scaler.transform(df_clean[num_cols])

        df_clean[num_cols] = scaled_array
        self.transformed_cols = num_cols
        
        logger.info("Numerical transformation and scaling complete successfully.")
        return df_clean


if __name__ == "__main__":
    # Local test demonstration
    config_file = Path("config/O01_feature_rules.yaml")
    
    sample_df = pd.DataFrame({
        "tenure_months": [12, 24, 6, 48, 3],
        "monthly_charges": [29.95, 75.0, 19.0, 105.0, 45.0],
        "total_charges": [359.4, 1800.0, 114.0, 5040.0, 135.0]
    })
    
    if config_file.exists():
        transformer = NumericalTransformer(config_path=config_file)
        df_scaled = transformer.fit_transform_features(sample_df, is_train=True)
        print("\n[RESULT] Transformed & Scaled DataFrame Preview:")
        print(df_scaled.head())