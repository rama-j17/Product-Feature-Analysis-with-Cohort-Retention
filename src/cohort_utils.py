"""
Utility functions to build cohort retention matrices
and feature‑impact summaries.
"""
import pandas as pd
import numpy as np


def build_retention_matrix(df: pd.DataFrame,
                           cohort_col: str = "CohortMonth",
                           tenure_col: str = "tenure",
                           horizons=(1, 3, 6)) -> pd.DataFrame:
    """
    Returns a retention matrix (% retained) with rows = cohort,
    cols = horizon (months).
    """
    base = df.groupby(cohort_col)["customerID"].nunique()
    mat = {}
    for h in horizons:
        alive = df[df[tenure_col] >= h]               # still around at month h
        retained = alive.groupby(cohort_col)["customerID"].nunique()
        mat[h] = (retained / base * 100).round(1)
    return pd.DataFrame(mat).fillna(0).rename(columns=lambda m: f"{m}M")


def feature_retention_lift(df: pd.DataFrame,
                           feature_flag: str,
                           horizon: int = 3) -> pd.Series:
    """
    Calculates retention % with vs without a feature at given horizon (months).
    Returns a Series: index = ['With', 'Without'].
    """
    retained_col = f"Retained_{horizon}M"
    with_feat    = df[df[feature_flag] == 1][retained_col].mean() * 100
    without_feat = df[df[feature_flag] == 0][retained_col].mean() * 100
    return pd.Series({"With": round(with_feat, 1),
                      "Without": round(without_feat, 1)})
