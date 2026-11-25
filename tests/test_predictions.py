"""Unit and integration tests for calibrated predictions and review queue logic."""

from pathlib import Path
import pandas as pd
import pytest

from spendsense.config import get_config
from spendsense.data.preprocess import normalize_description
from spendsense.models.calibrate import CalibratedSpendSenseModel, PredictionOutput
from spendsense.models.pipeline import get_model_pipeline
from spendsense.models.registry import ModelRegistry


@pytest.fixture
def trained_calibrated_model():
    """Builds and trains a quick pipeline for testing predictions."""
    config, categories = get_config()
    df = pd.read_csv("data/samples/training_sample.csv")
    df["clean"] = df["raw_description"].apply(normalize_description)

    pipeline = get_model_pipeline("linear_svc", config=config)
    pipeline.fit(df["clean"], df["true_category"])

    return CalibratedSpendSenseModel(
        fitted_pipeline=pipeline,
        model_version="1.0.0-test",
        confidence_threshold=0.70,
    )


def test_predict_single_returns_valid_structure(trained_calibrated_model):
    """Verify single prediction output schema and constraints."""
    res = trained_calibrated_model.predict_single("swiggy food delivery bangalore")

    assert isinstance(res, PredictionOutput)
    assert isinstance(res.predicted_category, str)
    assert 0.0 <= res.confidence <= 1.0
    assert isinstance(res.requires_review, bool)
    assert res.model_version == "1.0.0-test"
    assert len(res.alternatives) <= 2


def test_confidence_threshold_review_queue_trigger(trained_calibrated_model):
    """Verify review queue flags predictions when threshold is not met."""
    # With a high threshold (0.99), almost every prediction should be sent for review
    res_high = trained_calibrated_model.predict_single("swiggy order", threshold=0.99)
    assert res_high.requires_review is True
    assert res_high.review_reason is not None

    # With a low threshold (0.01), predictions should pass without review
    res_low = trained_calibrated_model.predict_single("swiggy order", threshold=0.01)
    assert res_low.requires_review is False
    assert res_low.review_reason is None


def test_batch_prediction_consistency(trained_calibrated_model):
    """Verify batch inference handles multiple inputs properly."""
    inputs = [
        "uber trip to airport",
        "netflix monthly entertainment subscription",
        "salary credit from infosys technologies",
    ]
    outputs = trained_calibrated_model.predict_batch(inputs, threshold=0.60)

    assert len(outputs) == 3
    for out in outputs:
        assert isinstance(out, PredictionOutput)
        assert len(out.alternatives) > 0
        # Assert alternatives are ordered descending by confidence
        if len(out.alternatives) >= 2:
            assert out.alternatives[0]["confidence"] >= out.alternatives[1]["confidence"]


def test_model_registry_loads_saved_artifact():
    """Verify ModelRegistry loads the trained model from models/ directory."""
    model = ModelRegistry.load_active_model(force_reload=True)
    assert isinstance(model, CalibratedSpendSenseModel)

    manifest = ModelRegistry.get_manifest()
    assert "version" in manifest
    assert "categories" in manifest

    # Test an inference run using the loaded artifact
    res = model.predict_single("starbucks coffee bangalore")
    assert res.predicted_category in manifest["categories"]
