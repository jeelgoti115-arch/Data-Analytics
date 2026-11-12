"""
O02_categorical_encoder.py
Handles low and high cardinality categorical variables via One-Hot and out-of-fold Target Encoding.
"""
"""
================================================================================
MODULE: Categorical Encoder Engine
FILE: src/O02_categorical_encoder.py
PURPOSE:
  Transforms low-cardinality categorical variables via One-Hot Encoding 
  and high-cardinality variables via smoothed Target Encoding. Enforces 
  fit-transform separation to prevent target leakage between train and test splits, 
  parameterized by 'config/O01_feature_rules.yaml'.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.preprocessing import OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("CategoricalEncoder")


class CategoricalEncoder:
    """Encodes categorical features using One-Hot and Target Encoding strategies."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.ohe_encoder = None
        self.target_encoding_maps = {}

    def _load_config(self) -> dict:
        """Loads categorical encoding parameters from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("categorical_encoding", {})

    def fit_transform_low_cardinality(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """
        Encodes low-cardinality categorical features using One-Hot Encoding.
        """
        low_cols = self.config.get("low_cardinality_columns", [])
        existing_cols = [c for c in low_cols if c in df.columns]
        
        if not existing_cols:
            return df

        df_clean = df.copy()
        drop_first = self.config.get("drop_first", True)

        if is_train:
            logger.info(f"Fitting OneHotEncoder on low-cardinality columns: {existing_cols}")
            self.ohe_encoder = OneHotEncoder(
                sparse_output=False,
                drop="first" if drop_first else None,
                handle_unknown="ignore"
            )
            encoded_array = self.ohe_encoder.fit_transform(df_clean[existing_cols])
            encoded_cols = self.ohe_encoder.get_feature_names_out(existing_cols)
        else:
            if self.ohe_encoder is None:
                raise ValueError("OneHotEncoder has not been fitted. Run on train split first.")
            logger.info("Transforming low-cardinality columns using fitted OneHotEncoder...")
            encoded_array = self.ohe_encoder.transform(df_clean[existing_cols])
            encoded_cols = self.ohe_encoder.get_feature_names_out(existing_cols)

        # Create encoded DataFrame and merge
        encoded_df = pd.DataFrame(encoded_array, columns=encoded_cols, index=df_clean.index)
        df_clean = pd.concat([df_clean.drop(columns=existing_cols), encoded_df], axis=1)
        
        logger.info(f"Categorical OHE complete. Added {len(encoded_cols)} encoded columns.")
        return df_clean

    def fit_transform_target_encoding(self, df: pd.DataFrame, target_col: str, is_train: bool = True) -> pd.DataFrame:
        """
        Applies smoothed Target Encoding for high-cardinality categorical columns 
        to prevent dimensionality explosion while retaining predictive signal.
        """
        high_cols = self.config.get("high_cardinality_columns", [])
        existing_cols = [c for c in high_cols if c in df.columns]
        
        if not existing_cols or target_col not in df.columns:
            return df

        df_clean = df.copy()
        smoothing = self.config.get("target_encoding", {}).get("smoothing", 10.0)
        global_mean = df_clean[target_col].mean()

        if is_train:
            logger.info(f"Fitting Target Encoder on columns: {existing_cols} with smoothing={smoothing}")
            self.target_encoding_maps = {}
            
            for col in existing_cols:
                stats = df_clean.groupby(col)[target_col].agg(["count", "mean"])
                # Apply smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
                smoothed = (stats["count"] * stats["mean"] + smoothing * global_mean) / (stats["count"] + smoothing)
                self.target_encoding_maps[col] = smoothed.to_dict()
                
                # Transform train split
                df_clean[f"{col}_encoded"] = df_clean[col].map(self.target_encoding_maps[col]).fillna(global_mean)
                df_clean.drop(columns=[col], inplace=True)
        else:
            if not self.target_encoding_maps:
                raise ValueError("Target encoder maps not fitted. Run on train split first.")
            logger.info("Transforming high-cardinality columns using fitted Target Encoder maps...")
            
            for col in existing_cols:
                mapping = self.target_encoding_maps.get(col, {})
                df_clean[f"{col}_encoded"] = df_clean[col].map(mapping).fillna(global_mean)
                df_clean.drop(columns=[col], inplace=True)

        return df_clean


if __name__ == "__main__":
    # Local test demonstration
    config_file = Path("config/O01_feature_rules.yaml")
    
    sample_df = pd.DataFrame({
        "contract_type": ["month-to-month", "one-year", "two-year", "month-to-month"],
        "payment_method": ["electronic_check", "mailed_check", "credit_card", "mailed_check"],
        "churn_status": [1, 0, 0, 1]
    })
    
    if config_file.exists():
        encoder = CategoricalEncoder(config_path=config_file)
        df_encoded = encoder.fit_transform_low_cardinality(sample_df, is_train=True)
        print("\n[RESULT] Encoded Categorical DataFrame Preview:")
        print(df_encoded.head())