# Imbalance Handling Module Stub
"""
================================================================================
MODULE: Class Imbalance Handler Engine
FILE: src/O03_imbalance_handler.py
PURPOSE:
  Mitigates class imbalance via SMOTE (Synthetic Minority Over-sampling Technique) 
  and computed class weights. Enforces strict fold-level isolation so that 
  synthetic samples are never generated across validation or test boundaries, 
  parameterized by 'config/O01_model_rules.yaml'.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from imblearn.over_sampling import SMOTE # pip install imbalanced-learn
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("ImbalanceHandler")


class ImbalanceHandler:
    """Handles class imbalance mitigation strategies for tabular classification."""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        """Loads imbalance handling rules from YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"[ERROR] Configuration file not found: {self.config_path}")
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f).get("imbalance_handling", {})

    def apply_smote(self, X: pd.DataFrame, y: pd.Series) -> tuple:
        """
        Applies SMOTE over-sampling to the training split to balance class distribution.
        
        CRITICAL: Must be called exclusively on training folds during cross-validation 
        to prevent data leakage into validation/test splits.
        """
        strategy = self.config.get("strategy", "class_weight")
        
        if strategy != "smote":
            logger.info(f"Imbalance strategy is set to '{strategy}'. Skipping SMOTE transformation.")
            return X, y

        sampling_strategy = self.config.get("smote_sampling_strategy", 0.5)
        random_state = 42

        logger.info(f"Applying SMOTE over-sampling (sampling_strategy={sampling_strategy})...")
        orig_counts = y.value_counts().to_dict()
        logger.info(f"Class distribution before SMOTE: {orig_counts}")

        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
        X_resampled_array, y_resampled_array = smote.fit_resample(X, y)

        X_resampled = pd.DataFrame(X_resampled_array, columns=X.columns)
        y_resampled = pd.Series(y_resampled_array, name=y.name if hasattr(y, "name") else "target")

        new_counts = y_resampled.value_counts().to_dict()
        logger.info(f"Class distribution after SMOTE: {new_counts}")

        return X_resampled, y_resampled

    def get_class_weights(self, y: pd.Series) -> dict:
        """
        Computes balanced class weights for cost-sensitive learning in loss functions.
        """
        classes = np.unique(y)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
        class_weight_dict = dict(zip(classes, weights))
        
        logger.info(f"Computed balanced class weights: {class_weight_dict}")
        return class_weight_dict


if __name__ == "__main__":
    # Local test demonstration
    config_file = Path("config/O01_model_rules.yaml")
    
    np.random.seed(42)
    sample_X = pd.DataFrame(np.random.randn(100, 5), columns=[f"f_{i}" for i in range(5)])
    sample_y = pd.Series([0]*85 + [1]*15, name="churn_status")
    
    if config_file.exists():
        handler = ImbalanceHandler(config_path=config_file)
        weights = handler.get_class_weights(sample_y)
        X_smote, y_smote = handler.apply_smote(sample_X, sample_y)
        print(f"\n[RESULT] SMOTE Resampled shape -> X: {X_smote.shape}, y distribution: {y_smote.value_counts().to_dict()}")