"""Unit tests for SQLite feedback storage and controlled retraining."""

import tempfile
from pathlib import Path
import pandas as pd
import pytest

from spendsense.feedback.retrain import evaluate_and_promote_candidate_model
from spendsense.feedback.store import FeedbackStore


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides an isolated temporary SQLite database path."""
    return str(tmp_path / "test_feedback.db")


def test_feedback_store_save_and_retrieve(temp_db_path):
    """Verify storing user corrections and retrieval."""
    store = FeedbackStore(db_path=temp_db_path)
    assert store.get_corrections_count() == 0

    success = store.save_correction(
        transaction_id="txn_12345",
        normalized_description="swiggy delivery food order",
        original_category="Shopping & Retail",
        corrected_category="Food & Dining",
        model_version="1.0.0",
    )
    assert success is True
    assert store.get_corrections_count() == 1

    corrections = store.get_all_corrections()
    assert len(corrections) == 1
    assert corrections[0].transaction_id == "txn_12345"
    assert corrections[0].corrected_category == "Food & Dining"


def test_feedback_store_deduplication(temp_db_path):
    """Verify inserting identical correction replaces rather than duplicates."""
    store = FeedbackStore(db_path=temp_db_path)
    store.save_correction("txn_1", "uber ride", "Shopping & Retail", "Transportation")
    store.save_correction("txn_1", "uber ride", "Shopping & Retail", "Transportation")

    assert store.get_corrections_count() == 1


def test_retrain_rejection_on_insufficient_samples(temp_db_path):
    """Verify retraining is rejected if fewer than minimum corrections exist."""
    store = FeedbackStore(db_path=temp_db_path)
    store.save_correction("txn_1", "swiggy", "Shopping & Retail", "Food & Dining")

    train_df = pd.DataFrame({
        "normalized_description": ["swiggy food", "uber ride"],
        "true_category": ["Food & Dining", "Transportation"],
    })
    val_df = train_df.copy()

    promoted, report = evaluate_and_promote_candidate_model(
        base_train_df=train_df,
        val_df=val_df,
        db_path=temp_db_path,
        min_corrections_required=5,
    )

    assert promoted is False
    assert report["status"] == "rejected"
    assert "Insufficient" in report["reason"]
