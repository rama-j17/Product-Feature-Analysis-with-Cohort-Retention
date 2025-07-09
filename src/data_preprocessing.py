"""
Clean Telco Customer Churn CSV, engineer cohort fields & feature flags,
and save processed dataset for analysis / Power BI.

Run:
python src/data_preprocessing.py \
    --input data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv \
    --output data/processed/telco_clean_cohorts.csv
"""
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

file_path = "telco_clean_cohorts.csv"
df = pd.read_csv(file_path)

TODAY = pd.to_datetime("2025-07-01")   # freeze 'today' for reproducibility
FEATURE_MAP = {
    "OnlineSecurity": "Feature_A",     # Mobile App Activation
    "StreamingTV":    "Feature_B",     # Transaction History
    "TechSupport":    "Feature_C",     # Analytics Tools
}


def clean_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Basic cleaning & type fixes."""
    df.columns = df.columns.str.strip().str.replace(" ", "_")
    # numeric conversion
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"])
    # churn to binary
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer signup date, cohort month, retention flags, and feature flags."""
    # synthetic signup date from tenure (months)
    df["SignupDate"]  = TODAY - pd.to_timedelta(df["tenure"] * 30, unit="D")
    df["CohortMonth"] = df["SignupDate"].dt.to_period("M")

    # feature flags
    for col, new_flag in FEATURE_MAP.items():
        df[new_flag] = df[col].map({"Yes": 1, "No": 0})

    # retention flags
    df["Retained_1M"] = (df["tenure"] >= 1).astype(int)
    df["Retained_3M"] = (df["tenure"] >= 3).astype(int)
    return df


def main(args):
    in_path  = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    df = engineer_features(clean_telco(df))
    df.to_csv(out_path, index=False)
    print(f"[+] Saved processed file to {out_path.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="raw Telco CSV")
    ap.add_argument("--output", required=True, help="processed CSV path")
    main(ap.parse_args())
