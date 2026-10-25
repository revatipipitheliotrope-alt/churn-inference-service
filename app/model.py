"""Model loading and feature preparation.

The trained model is stored as a joblib bundle::

    {"pipeline": sklearn.pipeline.Pipeline, "version": str, "feature_columns": list[str]}

``features_to_frame`` is the single place where validated request rows are
turned into the numeric frame the pipeline expects. Both the single and any
future batch endpoint must go through it so the two can never drift apart.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.config import Settings
from app.schemas import CustomerFeatures, RiskBucket

FEATURE_COLUMNS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "num_support_tickets",
    "contract_type",
    "has_tech_support",
]

CONTRACT_ENCODING = {"month_to_month": 0, "one_year": 1, "two_year": 2}


class ModelNotFoundError(FileNotFoundError):
    """Raised when the model artifact is missing from the configured path."""


class ChurnModel:
    """Thin wrapper around the trained scikit-learn pipeline."""

    def __init__(self, pipeline, version: str) -> None:
        self.pipeline = pipeline
        self.version = version

    @classmethod
    def load(cls, path: Path) -> ChurnModel:
        """Load the joblib bundle at ``path``.

        Raises ``ModelNotFoundError`` with an actionable message instead of a
        bare ``FileNotFoundError`` so callers can turn it into a 503.
        """
        if not path.exists():
            raise ModelNotFoundError(
                f"Model artifact not found at '{path}'. Run `python scripts/train.py` "
                "or point CHURN_MODEL_PATH at an existing bundle."
            )
        bundle = joblib.load(path)
        return cls(pipeline=bundle["pipeline"], version=bundle["version"])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return the churn probability for every row of ``frame`` in order."""
        return self.pipeline.predict_proba(frame[FEATURE_COLUMNS])[:, 1]


def features_to_frame(rows: list[CustomerFeatures]) -> pd.DataFrame:
    """Convert validated request rows into the numeric frame the pipeline expects.

    Row order is preserved and no row is ever dropped, so the i-th probability
    always belongs to the i-th customer in ``rows``.
    """
    frame = pd.DataFrame([row.model_dump() for row in rows])
    frame["contract_type"] = frame["contract_type"].map(CONTRACT_ENCODING)
    frame["has_tech_support"] = frame["has_tech_support"].astype(int)
    return frame[FEATURE_COLUMNS]


def risk_bucket(probability: float, settings: Settings) -> RiskBucket:
    """Map a probability to the low/medium/high bucket used by the retention team."""
    if probability >= settings.high_risk_threshold:
        return "high"
    if probability >= settings.medium_risk_threshold:
        return "medium"
    return "low"
