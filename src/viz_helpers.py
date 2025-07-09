"""
Plotting helpers: heatmap, cohort curves, feature lift bar.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def heatmap_retention(retention_df: pd.DataFrame, title: str):
    plt.figure(figsize=(10, 6))
    sns.heatmap(retention_df, annot=True, fmt=".1f", cmap="viridis")
    plt.title(title)
    plt.ylabel("Cohort (Signup Month)")
    plt.xlabel("Retention Horizon")
    plt.tight_layout()
    plt.show()


def cohort_curves(df_long: pd.DataFrame, title: str = "Retention Curves"):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_long, x="Months_Since_Signup", y="Retention", hue="Cohort")
    plt.title(title)
    plt.ylim(0, 100)
    plt.ylabel("Retention %")
    plt.tight_layout()
    plt.show()
