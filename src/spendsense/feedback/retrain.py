"""Controlled model retraining and candidate promotion orchestrator.

Validates that human feedback actually improves or maintains Macro-F1 on the held-out benchmark
before promoting the new model to active serving (Principle #306).
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd

from spendsense.config import get_config
from spendsense.feedback.store import FeedbackStore
from spendsense.models.evaluate import compute_classification_metrics
from spendsense.models.pipeline import get_model_pipeline
from spendsense.models.registry import ModelRegistry


def evaluate_and_promote_candidate_model(
    base_train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    db_path: Optional[str] = None,
    min_corrections_required: int = 5,
) -> Tuple[bool, Dict[str, Any]]:
    """Gathers user corrections, trains a candidate pipeline, and conditionally promotes it.

    Returns:
        (promoted_bool, evaluation_report)
    """
    config, categories = get_config()
    store = FeedbackStore(db_path=db_path)
    corrections = store.get_all_corrections()

    if len(corrections) < min_corrections_required:
        return False, {
            "status": "rejected",
            "reason": f"Insufficient user corrections ({len(corrections)}/{min_corrections_required}) to warrant retraining.",
            "promoted": False,
        }

    # Convert corrections into training rows
    feedback_rows = [
        {
            "normalized_description": c.normalized_description,
            "true_category": c.corrected_category,
        }
        for c in corrections
        if c.corrected_category in categories.canonical_names
    ]
    feedback_df = pd.DataFrame(feedback_rows)

    # Augment base training data with deduplicated human corrections
    augmented_train_df = pd.concat([base_train_df, feedback_df], ignore_index=True)
    augmented_train_df = augmented_train_df.drop_duplicates(subset=["normalized_description"], keep="last")

    # Train candidate model
    candidate_pipeline = get_model_pipeline(config.model.active_model_name, config=config)
    candidate_pipeline.fit(
        augmented_train_df["normalized_description"],
        augmented_train_df["true_category"],
    )

    # Evaluate candidate on the fixed held-out validation set
    y_val = val_df["true_category"].tolist()
    y_pred_candidate = candidate_pipeline.predict(val_df["normalized_description"])
    candidate_metrics = compute_classification_metrics(y_val, list(y_pred_candidate))

    # Evaluate current active model on same validation set
    current_model = ModelRegistry.load_active_model()
    current_preds = [
        current_model.predict_single(desc).predicted_category
        for desc in val_df["normalized_description"]
    ]
    current_metrics = compute_classification_metrics(y_val, current_preds)

    curr_f1 = current_metrics["macro_f1"]
    cand_f1 = candidate_metrics["macro_f1"]

    # Promotion Rule: Candidate must maintain or improve Macro-F1
    should_promote = cand_f1 >= curr_f1

    report = {
        "status": "promoted" if should_promote else "rejected",
        "current_macro_f1": curr_f1,
        "candidate_macro_f1": cand_f1,
        "feedback_samples_used": len(feedback_rows),
        "promoted": should_promote,
    }

    if should_promote:
        model_dir = Path(config.paths.models_dir)
        artifact_path = model_dir / "spendsense_classifier.joblib"
        manifest_path = model_dir / "model_manifest.json"

        joblib.dump(candidate_pipeline, artifact_path)

        manifest = {
            "model_name": config.model.active_model_name,
            "version": f"1.1.0-retrained-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M')}",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "macro_f1": cand_f1,
            "accuracy": candidate_metrics["accuracy"],
            "feedback_corrections_incorporated": len(feedback_rows),
            "confidence_threshold": config.model.confidence_threshold,
            "categories": categories.canonical_names,
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        # Invalidate cached model instance
        ModelRegistry.load_active_model(force_reload=True)

    return should_promote, report
