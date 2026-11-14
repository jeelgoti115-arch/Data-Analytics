# Model Interpretability (SHAP) Module Stub
"""
================================================================================
MODULE: Model Interpretability Engine (SHAP)
FILE: src/O07_model_interpretability.py
PURPOSE:
  Extracts SHAP (SHapley Additive exPlanations) values from trained tree models 
  to compute global feature importance rankings and local individual prediction 
  explanations, ensuring transparency and regulatory compliance for customer 
  retention interventions.
================================================================================
"""

import logging
from pathlib import Path
import pandas as pd
import numpy as np
import shap

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("ModelInterpretability")


class ModelInterpreter:
    """Computes global and local model explanations using TreeSHAP."""

    def __init__(self, model, X_sample: pd.DataFrame):
        self.model = model
        self.X_sample = X_sample
        self.explainer = None
        self.shap_values = None
        self._initialize_explainer()

    def _initialize_explainer(self):
        """Initializes the SHAP TreeExplainer for tree-based models."""
        logger.info("Initializing SHAP TreeExplainer for tree-based ensemble...")
        try:
            # TreeExplainer is optimized for XGBoost, LightGBM, CatBoost, and Decision Trees
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer(self.X_sample)
            logger.info("SHAP values computed successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP TreeExplainer: {e}")
            raise e

    def get_global_importance_df(self) -> pd.DataFrame:
        """
        Computes mean absolute SHAP values across all samples to establish 
        global feature importance rankings.
        """
        logger.info("Calculating global feature importance rankings from SHAP values...")
        
        # Handle multi-class vs binary output shapes in shap_values
        vals = self.shap_values.values
        if len(vals.shape) == 3:
            # Binary classification shape: (samples, features, classes) -> take positive class
            vals = vals[:, :, 1]

        mean_abs_shap = np.mean(np.abs(vals), axis=0)
        
        importance_df = pd.DataFrame({
            "feature": self.X_sample.columns,
            "mean_abs_shap": mean_abs_shap
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        logger.info(f"Top 3 most important features: {importance_df['feature'].head(3).tolist()}")
        return importance_df

    def get_local_explanation(self, customer_index: int) -> dict:
        """
        Extracts local SHAP explanation for a single customer prediction, 
        highlighting which features pushed the prediction toward or away from churn.
        """
        if customer_index >= len(self.X_sample):
            raise IndexError(f"Customer index {customer_index} out of bounds for sample size {len(self.X_sample)}")

        logger.info(f"Extracting local SHAP explanation for sample index {customer_index}...")
        
        vals = self.shap_values.values
        if len(vals.shape) == 3:
            vals = vals[:, :, 1]

        customer_shap = vals[customer_index]
        customer_features = self.X_sample.iloc[customer_index]

        explanation = pd.DataFrame({
            "feature": self.X_sample.columns,
            "feature_value": customer_features.values,
            "shap_value": customer_shap
        }).sort_values(by="shap_value", key=abs, ascending=False).reset_index(drop=True)

        return {
            "base_value": float(self.shap_values.base_values[customer_index] if not isinstance(self.shap_values.base_values, np.ndarray) else self.shap_values.base_values[customer_index]),
            "explanation_df": explanation
        }


if __name__ == "__main__":
    # Local test demonstration
    import xgboost as xgb
    
    np.random.seed(42)
    X_train = pd.DataFrame(np.random.randn(100, 5), columns=[f"feat_{i}" for i in range(5)])
    y_train = pd.Series(np.random.choice([0, 1], size=100, p=[0.7, 0.3]))
    
    # Train dummy model
    dummy_model = xgb.XGBClassifier(n_estimators=10, random_state=42)
    dummy_model.fit(X_train, y_train)
    
    interpreter = ModelInterpreter(model=dummy_model, X_sample=X_train.head(10))
    global_imp = interpreter.get_global_importance_df()
    local_exp = interpreter.get_local_explanation(customer_index=0)
    
    print("\n[RESULT] Global Importance Preview:")
    print(global_imp.head())
    print(f"\n[RESULT] Local Explanation Base Value: {local_exp['base_value']:.4f}")