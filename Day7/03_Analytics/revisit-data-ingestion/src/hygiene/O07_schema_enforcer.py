"""
07_schema_enforcer.py
Validates incoming payloads against schema contracts and handles quarantine.
"""

"""
================================================================================
MODULE: Schema Enforcer & Quarantine Router
FILE: src/hygiene/O07_schema_enforcer.py
PURPOSE:
  Loads the schema contract and validates incoming raw DataFrames against 
  strict type, nullability, and boundary rules. Detects upstream schema drift 
  (missing columns) and isolates corrupted or invalid records into a Dead-Letter 
  Queue (Quarantine directory) so bad data never crashes downstream pipelines.
================================================================================
"""


from pathlib import Path
from typing import Tuple
import pandas as pd
import yaml



class SchemaEnforcer:
    """Enforces data contracts and routes invalid payloads to quarantine."""

    def __init__(self, contract_path: Path, quarantine_dir: Path):
        self.contract_path = Path(contract_path)
        self.quarantine_dir = Path(quarantine_dir)
        self.quarantine_dir.mkdir(parents=True, exist_ok=True)
        
        self.contract = self._load_contract()
        

    def _load_contract(self) -> dict:
        """Loads the YAML schema contract specification."""
        if not self.contract_path.exists():
            raise FileNotFoundError(f"[ERROR] Schema contract not found: {self.contract_path}")
            
        with open(self.contract_path, "r", encoding="utf-8") as f:
            contract_data = yaml.safe_load(f)
            
        return contract_data

    def validate_and_quarantine(self, df: pd.DataFrame, batch_id: str = "batch_001") -> pd.DataFrame:
        
        expected_cols = self.contract["expected_columns"]
        
        # 1. Check for Missing Columns (Schema Drift)
        missing_columns = [col for col in expected_cols.keys() if col not in df.columns]
        if missing_columns:
            # If critical columns are missing, quarantine the entire batch
            quarantine_path = self.quarantine_dir / f"quarantined_{batch_id}_schema_drift.csv"
            df.to_csv(quarantine_path, index=False)
            raise ValueError(f"Schema drift detected! Entire batch quarantined to {quarantine_path}")

        valid_indices = []
        quarantine_rows = []

        for idx,row in df.iterrows():
            is_row_valid = True
            error_reason = []
            for col,rules in expected_cols.items():
                val = row[col]
                if not rules.get('nullable',True) and pd.isna(val):
                    is_row_valid = False
                    error_reason.append(f"Column '{col}' cannot be null.")
                    break

                if pd.isna(val):
                    print(f"Allowing {col} to pass")
                    continue

                # Allowed categorical values check
                if "allowed_values" in rules and val not in rules["allowed_values"]:
                    is_row_valid = False
                    error_reason.append(f"Column '{col}' value '{val}' not in allowed set {rules['allowed_values']}.")

                    break
                    
                 # Numerical range checks
                if "min_value" in rules and val < rules["min_value"]:
                    is_row_valid = False
                    error_reason.append(f"Column '{col}' value {val} below minimum {rules['min_value']}.")
                    break

                if "max_value" in rules and val > rules["max_value"]:
                    is_row_valid = False
                    error_reason.append(f"Column '{col}' value {val} above maximum {rules['max_value']}.")
                    break
                # print("row valid ",is_row_valid)
        
            if is_row_valid:
                    valid_indices.append(idx)
            else:
                    row_dict = row.to_dict()
                    row_dict["quarantine_reason"] = " | ".join(error_reason)
                    quarantine_rows.append(row_dict)

        # 3. Route Invalid Rows to Quarantine DLQ
        if quarantine_rows:
            q_df = pd.DataFrame(quarantine_rows)
            q_file = self.quarantine_dir / f"quarantined_{batch_id}.csv"
            q_df.to_csv(q_file, index=False)
            
        print("valid : ",valid_indices)
        clean_df = df.loc[valid_indices].reset_index(drop=True)
        return clean_df

if __name__ == "__main__":
    # Local demonstration test
    contract_file = Path(r"E:\jeel-cdac\Data Analytics\Day7\03_Analytics\revisit-data-ingestion\config\O01_schema_contract.yaml")
    quarantine_folder = Path("data/quarantine")

    # test_data0=pd.DataFrame()
   
    # Create mock test DataFrame with some invalid records
    # test_data1 = pd.DataFrame({
    #     "customer_id": ["CUST-001", "CUST-002", None],
    #     "tenure_months": [12, 150, 6],  # 150 violates max_value of 120
    # })
    import numpy as np
    test_data2 = pd.DataFrame({
    "customer_id": ["CUST-1001"],
    # "customer_id": [np.nan],
    "tenure_months": [12],
    "monthly_charges": [29.95],
    # "monthly_charges": [np.nan],
    # "total_charges": [359.4],
    "total_charges": [np.nan],
    "contract_type": ["month-to-month"],
    "payment_method": ["electronic_check"],
    "churn_status": ["no"]
    })

    test_data3 = pd.DataFrame({
    "customer_id": ["CUST-1001", "CUST-1002", "CUST-1003"],
    "tenure_months": [12, 24, -5],
    "monthly_charges": [29.95, 75.50, 105.00],
    "total_charges": [359.4, 1812.0, 525.0],
    "contract_type": ["month-to-month", "one-year", "two-year"],
    "payment_method": ["electronic_check", "mailed_check", "credit_card"],
    "churn_status": ["no", "yes", "no"]
    })

    enforcer = SchemaEnforcer(contract_path=contract_file, quarantine_dir=quarantine_folder)
    clean_data = enforcer.validate_and_quarantine(test_data3)
    # print(clean_data)
    print("\n[RESULT] Clean DataFrame output:")
    print(clean_data)