"""
O06_data_leakage_validator.py
Enforces strict train/test splitting before transformer fitting to prevent global data leakage.
"""
"""
================================================================================
MODULE: Data Leakage Validator & Splitter Engine
FILE: src/O06_data_leakage_validator.py
PURPOSE:
  Enforces strict train/test splitting prior to transformer fitting or feature 
  selection to prevent global data leakage. Validates that training and test sets 
  are completely isolated, mutually exclusive, and properly stratified for 
  classification targets.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("DataLeakageValidator")


class DataLeakageValidator:
    """Enforces strict data partitioning and prevents data leakage."""

    def __init__(self, test_size: float = 0.2, random_state: int = 42):
        self.test_size = test_size
        self.random_state = random_state
        logger.info(f"Initialized DataLeakageValidator (test_size={test_size}, random_state={random_state})")

    def stratified_train_test_split(self, df: pd.DataFrame, target_col: str) -> tuple:
        """
        Performs a strictly isolated stratified train/test split to maintain 
        class distribution balance and prevent leakage.
        
        Args:
            df (pd.DataFrame): Input cleaned DataFrame.
            target_col (str): Target column name for stratification.
            
        Returns:
            tuple: (train_df, test_df)
        """
        if target_col not in df.columns:
            logger.warning(f"Target column '{target_col}' not found. Falling back to random split.")
            train_df, test_df = train_test_split(df, test_size=self.test_size, random_state=self.random_state)
            return train_df, test_df

        logger.info(f"Executing stratified train/test split on target '{target_col}' (test_size={self.test_size})...")
        
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=df[target_col]
        )

        # Leakage Audit: Verify zero index overlap between train and test sets
        train_indices = set(train_df.index)
        test_indices = set(test_df.index)
        overlap = train_indices.intersection(test_indices)

        if overlap:
            raise ValueError(f"[CRITICAL DATA LEAKAGE] Found {len(overlap)} overlapping indices between train and test splits!")
        else:
            logger.info("Leakage Audit PASSED: Zero row index overlap between train and test sets.")

        logger.info(f"Split completed successfully -> Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        return train_df, test_df


if __name__ == "__main__":
    # Local test demonstration
    sample_df = pd.DataFrame({
        "feature_1": range(100),
        "feature_2": range(100, 200),
        "churn_status": [0]*80 + [1]*20
    })
    
    validator = DataLeakageValidator(test_size=0.2)
    train_data, test_data = validator.stratified_train_test_split(sample_df, target_col="churn_status")
    print(f"\n[RESULT] Train rows: {len(train_data)}, Test rows: {len(test_data)}")