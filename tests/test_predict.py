from fastapi.testclient import TestClient

from app import main as main_module

LOYAL_CUSTOMER = {
    "customer_id": "C-1001",
    "tenure_months": 60,
    "monthly_charges": 45.0,
    "total_charges": 2700.0,
    "num_support_tickets": 0,
    "contract_type": "two_year",
    "has_tech_support": True,
}

AT_RISK_CUSTOMER = {
    "customer_id": "C-2002",
    "tenure_months": 2,
    "monthly_charges": 105.0,
    "total_charges": 210.0,
    "num_support_tickets": 4,
    "contract_type": "month_to_month",
    "has_tech_support": False,
}


def test_predict_returns_probability_bucket_and_version(client):
    response = client.post("/predict", json=LOYAL_CUSTOMER)

    assert response.status_code == 200
    body = response.json()
    assert body["customer_id"] == "C-1001"
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["churn_risk"] in {"low", "medium", "high"}
    assert body["model_version"] == "test"


def test_at_risk_customer_scores_higher_than_loyal_customer(client):
    loyal = client.post("/predict", json=LOYAL_CUSTOMER).json()
    at_risk = client.post("/predict", json=AT_RISK_CUSTOMER).json()

    assert at_risk["churn_probability"] > loyal["churn_probability"]


def test_negative_tenure_is_rejected(client):
    response = client.post("/predict", json={**LOYAL_CUSTOMER, "tenure_months": -1})

    assert response.status_code == 422


def test_unknown_contract_type_is_rejected(client):
    response = client.post("/predict", json={**LOYAL_CUSTOMER, "contract_type": "lifetime"})

    assert response.status_code == 422


def test_missing_model_artifact_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(tmp_path / "does-not-exist.joblib"))
    monkeypatch.setattr(main_module, "_model", None)

    response = TestClient(main_module.app).post("/predict", json=LOYAL_CUSTOMER)

    assert response.status_code == 503
    assert "CHURN_MODEL_PATH" in response.json()["detail"]
