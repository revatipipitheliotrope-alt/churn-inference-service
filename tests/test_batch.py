def test_batch_predict_returns_one_prediction_per_row(client):
    rows = [
        {
            "customer_id": "C-1001",
            "tenure_months": 60,
            "monthly_charges": 45.0,
            "total_charges": 2700.0,
            "num_support_tickets": 0,
            "contract_type": "two_year",
            "has_tech_support": True,
        },
        {
            "customer_id": "C-2002",
            "tenure_months": 2,
            "monthly_charges": 105.0,
            "total_charges": 210.0,
            "num_support_tickets": 4,
            "contract_type": "month_to_month",
            "has_tech_support": False,
        },
    ]

    response = client.post("/predict/batch", json={"rows": rows})

    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 2
    assert [p["customer_id"] for p in body["predictions"]] == ["C-1001", "C-2002"]
