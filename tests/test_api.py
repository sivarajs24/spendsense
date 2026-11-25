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
