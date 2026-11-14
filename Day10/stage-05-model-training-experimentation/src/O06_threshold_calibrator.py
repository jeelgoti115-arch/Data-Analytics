# Probability Calibration & Threshold Tuning Module Stub
"""
================================================================================
MODULE: Probability Calibration & Threshold Tuning Engine
FILE: src/O06_threshold_calibrator.py
PURPOSE:
  Calibrates raw model prediction probabilities using Isotonic Regression and 
  tunes decision classification thresholds to optimize F1-score or business cost-benefit 
  matrices, moving beyond the default 0.5 threshold assumption.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, precision_recall_curve

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("ThresholdCalibrator")


class ThresholdCalibrator:
    """Calibrates model probabilities and optimizes decision thresholds."""

    def __init__(self, method: str = "isotonic"):
        self.method = method
        self.calibrated_model = None
        self.optimal_threshold = 0.5
        logger.info(f"Initialized ThresholdCalibrator (calibration_method={method})")

    def calibrate_probabilities(self, model, X_val: pd.DataFrame, y_val: pd.Series):
        """
        Wraps a trained model with probability calibration (Isotonic or Sigmoid/Platt).
        """
        logger.info(f"Fitting probability calibrator using method='{self.method}'...")
        self.calibrated_model = CalibratedClassifierCV(estimator=model, method=self.method, cv="prefit")
        self.calibrated_model.fit(X_val, y_val)
        logger.info("Probability calibration completed successfully.")
        return self.calibrated_model

    def find_optimal_threshold(self, y_true: np.ndarray, y_probs: np.ndarray) -> float:
        """
        Searches decision thresholds (0.01 to 0.99) to find the exact threshold 
        that maximizes the F1-score on validation data.
        """
        logger.info("Searching for optimal classification decision threshold...")
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_probs)
        
        # Avoid division by zero
        f1_scores = np.divide(
            2 * precisions * recalls,
            (precisions + recalls),
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0
        )

        # Find threshold that yields maximum F1 score
        best_idx = np.argmax(f1_scores)
        
        # precision_recall_curve returns thresholds of length len(precisions) - 1
        if best_idx < len(thresholds):
            self.optimal_threshold = float(thresholds[best_idx])
        else:
            self.optimal_threshold = 0.5

        best_f1 = f1_scores[best_idx]
        logger.info(f"Optimal Decision Threshold found: {self.optimal_threshold:.4f} (Achieved Validation F1: {best_f1:.4f})")
        
        return self.optimal_threshold

    def predict_with_optimal_threshold(self, y_probs: np.ndarray) -> np.ndarray:
        """Applies the tuned optimal threshold to convert probabilities into binary classes."""
        return (y_probs >= self.optimal_threshold).astype(int)


if __name__ == "__main__":
    # Local test demonstration
    np.random.seed(42)
    sample_y_true = np.array([0, 0, 0, 1, 1, 0, 1, 0, 1, 0])
    sample_y_probs = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.2, 0.7, 0.15, 0.9, 0.4])
    
    calibrator = ThresholdCalibrator(method="isotonic")
    opt_thresh = calibrator.find_optimal_threshold(sample_y_true, sample_y_probs)
    preds = calibrator.predict_with_optimal_threshold(sample_y_probs)
    
    print(f"\n[RESULT] Tuned Threshold: {opt_thresh:.4f}")
    print(f"[RESULT] Custom Predictions: {preds}")