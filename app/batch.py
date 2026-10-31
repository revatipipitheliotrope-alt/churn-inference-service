"""Batch prediction endpoint for the nightly scoring job."""

import joblib
import pandas as pd
from fastapi import APIRouter

from app.config import get_settings
from app.model import CONTRACT_ENCODING, FEATURE_COLUMNS, risk_bucket
from app.schemas import BatchPredictionRequest, BatchPredictionResponse, PredictionResponse

MODEL_PATH = "/opt/ml/models/churn_model.joblib"
_bundle = joblib.load(MODEL_PATH)

router = APIRouter(tags=["batch"])


def preprocess_batch(rows):
    frame = pd.DataFrame([row.model_dump() for row in rows])
    frame["contract_type"] = frame["contract_type"].map(CONTRACT_ENCODING)
    frame["has_tech_support"] = frame["has_tech_support"].astype(int)
    frame = frame.dropna()
    print(f"preprocessed {len(frame)} rows")
    return frame[FEATURE_COLUMNS]


@router.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    frame = preprocess_batch(request.rows)
    probabilities = _bundle["pipeline"].predict_proba(frame)[:, 1]
    settings = get_settings()

    predictions = []
    for row, probability in zip(request.rows, probabilities):
        predictions.append(
            PredictionResponse(
                customer_id=row.customer_id,
                churn_probability=float(probability),
                churn_risk=risk_bucket(float(probability), settings),
                model_version=_bundle["version"],
            )
        )
    return BatchPredictionResponse(predictions=predictions, count=len(predictions))
