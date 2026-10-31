"""FastAPI application exposing the churn model."""

from __future__ import annotations

import logging

from fastapi import Depends, FastAPI, HTTPException

from app import __version__
from app.batch import router as batch_router
from app.config import Settings, get_settings
from app.model import ChurnModel, ModelNotFoundError, features_to_frame, risk_bucket
from app.schemas import CustomerFeatures, PredictionResponse

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Inference Service",
    version=__version__,
    description="Scores customers for churn risk using the retention team's model.",
)
app.include_router(batch_router)

_model: ChurnModel | None = None


def get_model(settings: Settings = Depends(get_settings)) -> ChurnModel:
    """Load the model on first use and cache it for the lifetime of the process.

    Loading lazily keeps `import app.main` side-effect free, which is what lets
    the test-suite point CHURN_MODEL_PATH at a throw-away model.
    """
    global _model
    if _model is None:
        try:
            _model = ChurnModel.load(settings.model_path)
        except ModelNotFoundError as exc:
            logger.error("model artifact missing: %s", exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        logger.info("loaded churn model %s from %s", _model.version, settings.model_path)
    return _model


@app.get("/health")
def health() -> dict[str, object]:
    """Liveness probe. Reports whether the model has been loaded yet."""
    return {"status": "ok", "model_loaded": _model is not None, "service_version": __version__}


@app.post("/predict", response_model=PredictionResponse)
def predict(
    features: CustomerFeatures,
    model: ChurnModel = Depends(get_model),
    settings: Settings = Depends(get_settings),
) -> PredictionResponse:
    """Score a single customer."""
    frame = features_to_frame([features])
    probability = float(model.predict_proba(frame)[0])
    return PredictionResponse(
        customer_id=features.customer_id,
        churn_probability=probability,
        churn_risk=risk_bucket(probability, settings),
        model_version=model.version,
    )
