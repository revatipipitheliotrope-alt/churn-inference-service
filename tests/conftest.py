"""Shared fixtures.

A throw-away model is trained once per test session and the service is pointed
at it through CHURN_MODEL_PATH, so the tests never depend on a checked-in
artifact or on anything outside the repository.
"""

import pytest
from fastapi.testclient import TestClient

from app import main as main_module
from app.training import generate_synthetic_customers, save_bundle, train_pipeline


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("model") / "churn_model.joblib"
    pipeline = train_pipeline(generate_synthetic_customers(n_rows=800, seed=7))
    save_bundle(pipeline, path, version="test")
    return path


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setenv("CHURN_MODEL_PATH", str(model_path))
    monkeypatch.setattr(main_module, "_model", None)
    return TestClient(main_module.app)
