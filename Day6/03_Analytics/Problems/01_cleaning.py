from typing import Dict, List, Optional
import numpy as np
import pandas as pd

'''
Demonstrates:
- Object-Oriented Pipeline Design
- Schema Standardization (snake_case) : done
- String Normalization & Structural Mapping 
- Automated Missing Value Imputation (Median / Mode)
- IQR-based Outlier Detection & Capping
- Logging & Type Annotations
'''


class TransactionDataCleaner:
    def __init__(self, region_mapping: Optional[Dict[str, str]]) -> None:
        self.region_mapping = region_mapping

    def standardize_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip().str.lower().str.replace(" ","_")
        print("1. Schema standardized")
        return df

    def sanitize_text(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        string_cols = df.select_dtypes(include=["object","string"]).columns
        df[string_cols] = df[string_cols].apply(lambda col:col.str.strip())
        if "full_name" in df.columns:
            df["full_name"] = df["full_name"].str.title()
        print("2. Text Sanitization")
        return df

    def deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.drop_duplicates().reset_index(drop=True)
        return df

    def map_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "region" in df.columns:
                    df['region'] = df['region'].replace(self.region_mapping)

             
        return df

    def cast_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['purchase_amount'] = (
             df["purchase_amount"]
             .astype(str)
             .str.replace("$","",regex=False)
             .str.replace(",","",regex=False)
             .str.strip()
        )
        df['purchase_amount'] = pd.to_numeric(df['purchase_amount'], errors="coerce")

        if "join_date" in df.columns:
            df['join_date'] = pd.to_datetime(df['join_date'], errors="coerce")
             
        return df
         
    def impute_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if "age" in df.columns and df["age"].isnull().any():
            median_age = df["age"].median()
            df["age"] = df["age"].fillna(median_age)

        if "purchase_amount" in df.columns and df["purchase_amount"].isnull().any():
            median_amount = df["purchase_amount"].median()
            df["purchase_amount"] = df["purchase_amount"].fillna(median_amount)
             
        if "region" in df.columns and df["region"].isnull().any():
            mode_region = df["region"].mode()[0]
            df["region"] = df["region"].fillna(mode_region)
        return df

    def handle_outliers_iqr(self, df: pd.DataFrame, column: str = "purchase_amount") -> pd.DataFrame:
        df = df.copy()
        if column not in df.columns:
            return df
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        output_col = f"{column}_capped"
        df[output_col] = np.where(df[column] > upper_bound, upper_bound, df[column])
        return df
         
         
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df_clean =(
            df.pipe(self.standardize_schema)
            .pipe(self.sanitize_text)
            .pipe(self.deduplicate)
            .pipe(self.map_categoricals)
            .pipe(self.cast_types)
            .pipe(self.impute_missing)
            .pipe(self.handle_outliers_iqr,column="purchase_amount")
        )
        return df_clean
    







if __name__ == "__main__":
    # Sample dirty transaction payload
    raw_data = {
        " Customer_ID ": ["CUST_001", "CUST_002", "CUST_001", "CUST_004", "CUST_005", "CUST_006", "CUST_007"],
        " Full Name ": ["  aarav sharma ", "BHAVNA PATEL", "  aarav sharma ", "chirag gupta", "divya REDDY ", " eshan verma", "Fiona Gallagher"],
        " Purchase_Amount ": [" $1,200.50 ", " $450.00 ", " $1,200.50 ", " $89.99 ", " $15,000.00 ", " $150.25 ", np.nan],
        " Join Date ": ["2024-01-15", "2023-11-20", "2024-01-15", "2022-05-10", "2024-03-01", "2023-08-19", "2024-02-12"],
        " Region ": ["BLR", " Bengaluru ", "BLR", "MUM", " Mumbai ", "DEL", np.nan],
        " Age ": [28, 34, 28, np.nan, 42, 29, 31],
    }

    region_mapping =  {
            "BLR": "Bengaluru",
            "Bengaluru": "Bengaluru",
            "MUM": "Mumbai",
            "Mumbai": "Mumbai",
            "DEL": "Delhi",
        }
    
    df_raw = pd.DataFrame(raw_data)
    cleaner = TransactionDataCleaner(region_mapping)
    df_cleaned = cleaner.fit_transform(df_raw)
    print("=" * 100)
    print("_" * 100)
    print(df_cleaned)
    print("=" * 100)




