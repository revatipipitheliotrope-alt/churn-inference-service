"""Synthetic data generation and model training.

The repository ships with no customer data at all. ``generate_synthetic_customers``
draws a dataset from a fixed random seed with a known relationship between the
features and churn, which is enough to develop and test the API end to end.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from app.model import FEATURE_COLUMNS

MODEL_VERSION = "2026.10.1"


def generate_synthetic_customers(n_rows: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Return a frame with FEATURE_COLUMNS plus a binary ``churned`` label."""
    rng = np.random.default_rng(seed)

    tenure = rng.integers(0, 73, n_rows)
    monthly = rng.normal(65, 25, n_rows).clip(18, 120).round(2)
    total = (monthly * tenure + rng.normal(0, 20, n_rows)).clip(0).round(2)
    tickets = rng.poisson(1.2, n_rows)
    contract = rng.choice([0, 1, 2], n_rows, p=[0.55, 0.25, 0.20])
    tech_support = (rng.random(n_rows) < 0.45).astype(int)

    logit = (
        -1.0
        - 0.04 * tenure
        + 0.015 * monthly
        + 0.35 * tickets
        - 0.9 * contract
        - 0.6 * tech_support
    )
    churned = (rng.random(n_rows) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": total,
            "num_support_tickets": tickets,
            "contract_type": contract,
            "has_tech_support": tech_support,
            "churned": churned,
        }
    )


def train_pipeline(frame: pd.DataFrame) -> Pipeline:
    """Fit the scaler + logistic-regression pipeline on ``frame``."""
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    pipeline.fit(frame[FEATURE_COLUMNS], frame["churned"])
    return pipeline


def save_bundle(pipeline: Pipeline, path: Path, version: str = MODEL_VERSION) -> None:
    """Write the joblib bundle that ``app.model.ChurnModel.load`` expects."""
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": pipeline, "version": version, "feature_columns": FEATURE_COLUMNS},
        path,
    )
