"""Unit and integration tests for baseline and classical ML models."""

from pathlib import Path
import tempfile
import joblib
import pandas as pd
import pytest

from spendsense.config import get_config
from spendsense.data.preprocess import normalize_description
from spendsense.models.baseline_rules import KeywordBaselineClassifier
from spendsense.models.evaluate import compute_classification_metrics
from spendsense.models.pipeline import get_model_pipeline
from spendsense.models.train import compare_all_models, train_and_evaluate_model


@pytest.fixture
def sample_training_data():
    """Loads the controlled synthetic training sample."""
    csv_path = Path("data/samples/training_sample.csv")
    df = pd.read_csv(csv_path)
    df["normalized_description"] = df["raw_description"].apply(normalize_description)
    return df


def test_keyword_baseline_classifier_matching_and_fallback():
    """Verify KeywordBaseline matches configured keywords and defaults properly."""
    kw_map = {
        "Food & Dining": ["swiggy", "zomato", "cafe"],
        "Transportation": ["uber", "ola", "metro"],
    }
    clf = KeywordBaselineClassifier(keyword_map=kw_map)
    clf.fit(["random text"], ["Food & Dining"])

    preds = clf.predict(["I ordered food from Swiggy", "Took an Uber ride", "Unknown store purchase"])
    assert preds[0] == "Food & Dining"
    assert preds[1] == "Transportation"
    assert preds[2] == "Food & Dining"  # Fallback to majority class


def test_compute_classification_metrics():
    """Verify metrics calculation returns valid macro-f1 and confusion matrix."""
    y_true = ["Food & Dining", "Food & Dining", "Transportation", "Income"]
    y_pred = ["Food & Dining", "Transportation", "Transportation", "Income"]

    metrics = compute_classification_metrics(y_true, y_pred)
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["macro_f1"] <= 1.0
    assert "Food & Dining" in metrics["per_class"]
    assert len(metrics["confusion_matrix"]) > 0


def test_all_model_pipelines_train_and_predict(sample_training_data):
    """Verify all 5 required model architectures instantiate, train, and predict."""
    config, categories = get_config()
    train_df = sample_training_data.iloc[:60]
    val_df = sample_training_data.iloc[60:]

    models_to_test = ["dummy", "keyword", "naive_bayes", "logistic_regression", "linear_svc"]

    for model_name in models_to_test:
        pipeline, metrics = train_and_evaluate_model(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            config=config,
            categories=categories,
        )
        assert metrics["macro_f1"] >= 0.0
        assert metrics["accuracy"] >= 0.0

        # Predict on single new example
        test_pred = pipeline.predict(["swiggy delivery food order"])
        assert len(test_pred) == 1
        assert isinstance(test_pred[0], str)


def test_model_serialization_consistency(sample_training_data, tmp_path):
    """Verify saved and reloaded pipeline yields 100% identical predictions (Acceptance Criteria #20)."""
    config, categories = get_config()
    train_df = sample_training_data.iloc[:60]
    val_df = sample_training_data.iloc[60:]

    save_file = tmp_path / "test_pipeline.joblib"
    original_pipeline, _ = train_and_evaluate_model(
        model_name="linear_svc",
        train_df=train_df,
        val_df=val_df,
        config=config,
        categories=categories,
        save_path=save_file,
    )

    assert save_file.exists()
    reloaded_pipeline = joblib.load(save_file)

    test_inputs = val_df["normalized_description"].tolist()
    preds_original = original_pipeline.predict(test_inputs)
    preds_reloaded = reloaded_pipeline.predict(test_inputs)

    assert list(preds_original) == list(preds_reloaded)


def test_compare_all_models_summary_table(sample_training_data):
    """Verify compare_all_models returns a ranked benchmark table."""
    config, categories = get_config()
    train_df = sample_training_data.iloc[:50]
    val_df = sample_training_data.iloc[50:]

    df_summary = compare_all_models(
        train_df=train_df,
        val_df=val_df,
        config=config,
        categories=categories,
    )

    assert isinstance(df_summary, pd.DataFrame)
    assert len(df_summary) == 5
    assert "Macro F1" in df_summary.columns
    assert "Model" in df_summary.columns
    # Check that rows are sorted in descending order of Macro F1
    assert df_summary["Macro F1"].is_monotonic_decreasing
