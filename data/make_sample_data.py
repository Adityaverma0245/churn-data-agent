"""Generate a synthetic Telco-style churn table (no real customer data).

Usage: python data/make_sample_data.py [rows] > data/sample_churn.csv
To use the real Telco Customer Churn dataset instead, drop in its CSV and keep
the same column names (at minimum a Yes/No 'Churn' column).
"""
import sys

import numpy as np
import pandas as pd


def make(rows: int = 2000, seed: int = 7) -> pd.DataFrame:
    r = np.random.default_rng(seed)
    contract = r.choice(["Month-to-month", "One year", "Two year"], rows, p=[0.55, 0.25, 0.20])
    internet = r.choice(["DSL", "Fiber optic", "No"], rows, p=[0.35, 0.45, 0.20])
    pay = r.choice(["Electronic check", "Mailed check", "Credit card", "Bank transfer"], rows)
    tenure = r.integers(1, 73, rows)
    monthly = np.round(r.normal(65, 25, rows).clip(18, 120), 2)
    senior = r.integers(0, 2, rows)
    z = (-0.6 + 1.3 * (contract == "Month-to-month") - 0.9 * (contract == "Two year")
         + 0.5 * (internet == "Fiber optic") + 0.4 * (pay == "Electronic check")
         - 0.035 * tenure + 0.012 * (monthly - 65) + 0.3 * senior)
    churn = np.where(r.random(rows) < 1 / (1 + np.exp(-z)), "Yes", "No")
    return pd.DataFrame({"tenure": tenure, "MonthlyCharges": monthly, "Contract": contract,
                         "InternetService": internet, "PaymentMethod": pay,
                         "SeniorCitizen": senior, "Churn": churn})


if __name__ == "__main__":
    make(int(sys.argv[1]) if len(sys.argv) > 1 else 2000).to_csv(sys.stdout, index=False)
