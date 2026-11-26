"""Integration and unit tests for FastAPI endpoints."""

from fastapi.testclient import TestClient
import pytest

from api.main import app

client = TestClient(app)


def test_health_endpoint():
    """Verify /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data


def test_model_info_endpoint():
    """Verify /model/info returns metadata for the active model."""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_name" in data
    assert "categories" in data
    assert len(data["categories"]) == 10


def test_predict_single_endpoint():
    """Verify /predict returns valid prediction, confidence, and alternatives."""
    payload = {
        "description": "UPI/4829102938/SWIGGY@ybl",
        "threshold": 0.70,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_category" in data
    assert "confidence" in data
    assert 0.0 <= data["confidence"] <= 1.0
    assert "swiggy" in data["normalized_description"]
    assert isinstance(data["requires_review"], bool)


def test_predict_batch_endpoint():
    """Verify /predict/batch categorizes multiple transactions."""
    payload = {
        "items": [
            {"description": "UPI/4829102938/SWIGGY@ybl"},
            {"description": "ACH/NETFLIX.COM/SUBSCRIPTION"},
            {"description": "UBER RIDE MUMBAI"},
        ],
        "threshold": 0.70,
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 3
    assert len(data["predictions"]) == 3


def test_feedback_endpoint_valid_and_invalid_category():
    """Verify /feedback persists valid corrections and rejects unknown categories."""
    valid_payload = {
        "transaction_id": "txn_test_999",
        "normalized_description": "swiggy food",
        "original_category": "Shopping & Retail",
        "corrected_category": "Food & Dining",
    }
    res_valid = client.post("/feedback", json=valid_payload)
    assert res_valid.status_code == 200
    assert res_valid.json()["success"] is True

    # Unknown category should return 400 Bad Request
    invalid_payload = {
        "transaction_id": "txn_test_999",
        "normalized_description": "swiggy food",
        "original_category": "Shopping & Retail",
        "corrected_category": "Astronomy & Rockets",  # Invalid category
    }
    res_invalid = client.post("/feedback", json=invalid_payload)
    assert res_invalid.status_code == 400
    assert "Invalid category" in res_invalid.json()["detail"]


@pytest.fixture
def sample_api_transactions():
    return [
        {
            "transaction_id": "tx1",
            "date": "2026-01-01",
            "raw_description": "UPI/SWIGGY@ybl",
            "amount": -450.0,
            "predicted_category": "Food & Dining",
        },
        {
            "transaction_id": "tx2",
            "date": "2026-01-02",
            "raw_description": "SALARY CREDIT INFOSYS",
            "amount": 95000.0,
            "predicted_category": "Income",
        },
        {
            "transaction_id": "tx3",
            "date": "2026-01-05",
            "raw_description": "CRED/CC PAYMENT/HDFC",
            "amount": -15000.0,
            "predicted_category": "Financial Services",
        },
        {
            "transaction_id": "tx4",
            "date": "2026-01-08",
            "raw_description": "APOLLO PHARMACY",
            "amount": -1200.0,
            "predicted_category": "Healthcare & Medical",
        },
        {
            "transaction_id": "tx5",
            "date": "2026-01-10",
            "raw_description": "NETFLIX MUMBAI",
            "amount": -649.0,
            "predicted_category": "Entertainment & Recreation",
        },
        {
            "transaction_id": "tx6",
            "date": "2026-01-12",
            "raw_description": "SPOTIFY INDIA",
            "amount": -119.0,
            "predicted_category": "Entertainment & Recreation",
        },
        {
            "transaction_id": "tx7",
            "date": "2026-01-15",
            "raw_description": "ZERODHA ELSS MUTUAL FUND",
            "amount": -20000.0,
            "predicted_category": "Financial Services",
        },
        {
            "transaction_id": "tx8",
            "date": "2026-01-18",
            "raw_description": "ZOMATO RESTAURANT DINING",
            "amount": -520.0,
            "predicted_category": "Food & Dining",
        },
    ]


def test_analytics_tax_endpoint(sample_api_transactions):
    """Verify /analytics/tax returns deductions and regime simulation."""
    payload = {
        "transactions": sample_api_transactions,
        "annual_gross_income": 1200000.0,
        "is_senior_citizen": False,
    }
    response = client.post("/analytics/tax", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "deductions" in data
    assert "regime_comparison" in data
    assert data["deductions"]["total_80c_eligible"] >= 20000.0
    assert data["regime_comparison"]["recommended_regime"] in ["New Regime", "Old Regime", "Either (Identical Tax)"]


def test_analytics_transfers_endpoint(sample_api_transactions):
    """Verify /analytics/transfers isolates credit card settlement."""
    payload = {"transactions": sample_api_transactions}
    response = client.post("/analytics/transfers", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transfer_count"] >= 1
    assert data["double_counting_prevented"] >= 15000.0
    assert data["effective_net_outflows"] < data["total_raw_debits"]


def test_analytics_budget_endpoint(sample_api_transactions):
    """Verify /analytics/budget calculates 50/30/20 buckets and burn rate."""
    payload = {"transactions": sample_api_transactions, "override_inflow": 95000.0}
    response = client.post("/analytics/budget", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "buckets" in data
    assert "Needs (50%)" in data["buckets"]
    assert "Wants (30%)" in data["buckets"]
    assert "Savings (20%)" in data["buckets"]
    assert "wants_forecast" in data


def test_analytics_health_endpoint(sample_api_transactions):
    """Verify /analytics/health returns 0-100 vitality score and runway."""
    payload = {"transactions": sample_api_transactions, "override_liquid_balance": 300000.0}
    response = client.post("/analytics/health", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert 0 <= data["vitality_score"] <= 100
    assert data["emergency_runway_months"] > 0
    assert len(data["components"]) == 4


def test_analytics_leaks_endpoint(sample_api_transactions):
    """Verify /analytics/leaks surfaces redundant streaming services."""
    payload = {"transactions": sample_api_transactions}
    response = client.post("/analytics/leaks", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "alerts" in data
    assert "total_annual_leak_estimate" in data

