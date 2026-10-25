"""Runtime configuration.

Every deployable setting is read from an environment variable so the same
code runs unchanged on a laptop, in CI and in the container image. Nothing in
the application code should hard-code a filesystem path or a threshold.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_MODEL_PATH = Path("models") / "churn_model.joblib"


def _env_path(name: str, default: Path) -> Path:
    return Path(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


@dataclass(frozen=True)
class Settings:
    """Service settings. Each field maps to one environment variable."""

    # CHURN_MODEL_PATH: location of the joblib bundle written by scripts/train.py
    model_path: Path = field(
        default_factory=lambda: _env_path("CHURN_MODEL_PATH", DEFAULT_MODEL_PATH)
    )
    # CHURN_MEDIUM_RISK_THRESHOLD / CHURN_HIGH_RISK_THRESHOLD: probability cut-offs
    medium_risk_threshold: float = field(
        default_factory=lambda: _env_float("CHURN_MEDIUM_RISK_THRESHOLD", 0.4)
    )
    high_risk_threshold: float = field(
        default_factory=lambda: _env_float("CHURN_HIGH_RISK_THRESHOLD", 0.7)
    )


def get_settings() -> Settings:
    """Build settings from the current environment.

    Used as a FastAPI dependency so tests can override variables per test.
    """
    return Settings()
