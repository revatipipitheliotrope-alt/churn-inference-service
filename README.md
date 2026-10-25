# Churn Inference Service

A small FastAPI service that scores customers for churn risk with a scikit-learn model.
The retention team calls it from the nightly scoring job and from the account-manager dashboard.

## How it is organised

| Path | Purpose |
| --- | --- |
| `app/main.py` | FastAPI app, `/health` and `/predict` endpoints, lazy model loading |
| `app/model.py` | `ChurnModel` wrapper, `features_to_frame` preprocessing, risk buckets |
| `app/schemas.py` | Pydantic request/response models |
| `app/config.py` | Settings read from environment variables (`CHURN_MODEL_PATH`, thresholds) |
| `app/training.py` | Synthetic data generation and training code |
| `scripts/train.py` | Command-line entry point that writes `models/churn_model.joblib` |
| `tests/` | pytest suite; trains a throw-away model, never touches a checked-in artifact |
| `docs/REVIEW_TEMPLATE.md` | Template reviewers copy to `REVIEW_SUMMARY.md` |

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt

python scripts/train.py            # writes models/churn_model.joblib
uvicorn app.main:app --reload      # http://127.0.0.1:8000/docs
```

## API

### `GET /health`

Liveness probe. Returns `{"status": "ok", "model_loaded": false, "service_version": "0.3.0"}`.
`model_loaded` becomes `true` after the first prediction.

### `POST /predict`

Scores one customer.

```bash
curl -s -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "customer_id": "C-1001",
        "tenure_months": 14,
        "monthly_charges": 79.5,
        "total_charges": 1113.0,
        "num_support_tickets": 2,
        "contract_type": "month_to_month",
        "has_tech_support": false
      }'
```

```json
{"customer_id": "C-1001", "churn_probability": 0.61, "churn_risk": "medium", "model_version": "2026.10.1"}
```

Risk buckets come from `CHURN_MEDIUM_RISK_THRESHOLD` (default 0.4) and `CHURN_HIGH_RISK_THRESHOLD` (default 0.7).

## Configuration

| Variable | Default | Meaning |
| --- | --- | --- |
| `CHURN_MODEL_PATH` | `models/churn_model.joblib` | Location of the joblib bundle |
| `CHURN_MEDIUM_RISK_THRESHOLD` | `0.4` | Probability at which a customer becomes `medium` risk |
| `CHURN_HIGH_RISK_THRESHOLD` | `0.7` | Probability at which a customer becomes `high` risk |

Application code must read paths and thresholds through `app.config.Settings`; nothing should be hard-coded.

## Tests and lint

```bash
pytest -q
ruff check .
```

CI runs both on every push and pull request (`.github/workflows/ci.yml`).

## Code review

Every change to `main` goes through a pull request. Reviewers verify the CI result, reproduce
failures locally, and record their findings in `REVIEW_SUMMARY.md` (copy `docs/REVIEW_TEMPLATE.md`).
Blocking findings are anything that can mis-score a customer, break the nightly job, or make the
service depend on a specific machine.

## Data

The repository contains no customer data. `app/training.py` generates a synthetic dataset from a
fixed random seed so the service can be developed, tested and demonstrated end to end.

## License

MIT, see `LICENSE`.
