"""
O01_domain_feature_builder.py
Builds domain-specific interaction terms, ratios, and usage intensity indicators.
"""
"""
================================================================================
MODULE: Domain Feature Builder Engine
FILE: src/O01_domain_feature_builder.py
PURPOSE:
  Constructs domain-specific business features, interaction terms, and financial 
  ratios (e.g., charge-per-tenure-month, usage intensity indicators) based on formulas 
  parameterized in 'config/O01_feature_rules.yaml', with built-in safe division handling.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DomainFeatureBuilder")


class DomainFeatureBuilder:
    """Builds business and domain-specific feature interactions."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Loads domain feature engineering rules from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("domain_features", {})

    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Constructs custom domain ratios and interaction terms safely.
        
        Args:
            df (pd.DataFrame): Sanitized input DataFrame from Stage 3.
            
        Returns:
            pd.DataFrame: Enhanced DataFrame with newly engineered domain features.
        """
        logger.info("Constructing domain-specific features and interaction ratios...")
        df_clean = df.copy()
        
        ratios = self.config.get("ratios_to_build", [])
        
        for ratio in ratios:
            num_col = ratio.get("numerator")
            denom_col = ratio.get("denominator")
            new_col = ratio.get("new_feature_name")

            if num_col not in df_clean.columns or denom_col not in df_clean.columns:
                logger.warning(f"Skipping feature '{new_col}': Missing columns '{num_col}' or '{denom_col}'.")
                continue

            # Safe division: replace 0 denominator with NaN to avoid division-by-zero exceptions
            denominator = df_clean[denom_col].replace(0, np.nan)
            df_clean[new_col] = df_clean[num_col] / denominator
            
            # Fill resulting NaNs or infinities with a sensible baseline (e.g., 0.0)
            df_clean[new_col] = df_clean[new_col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
            logger.info(f"Engineered domain feature: '{new_col}' ({num_col} / {denom_col}).")

        return df_clean


if __name__ == "__main__":
    # Local test demonstration
    config_file = Path("config/O01_feature_rules.yaml")
    
    sample_df = pd.DataFrame({
        "customer_id": ["CUST-001", "CUST-002", "CUST-003"],
        "tenure_months": [12, 0, 24],  # 0 tenure tests division-by-zero protection
        "monthly_charges": [29.95, 75.0, 45.0],
        "total_charges": [359.4, 0.0, 1080.0]
    })
    
    if config_file.exists():
        builder = DomainFeatureBuilder(config_path=config_file)
        df_featured = builder.build_features(sample_df)
        print("\n[RESULT] Domain Engineered DataFrame Preview:")
        print(df_featured)