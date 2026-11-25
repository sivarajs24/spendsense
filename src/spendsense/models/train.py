"""Model training and baseline comparison orchestrator."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from spendsense.config import CategoriesConfig, SpendSenseConfig, get_config
from spendsense.models.evaluate import compute_classification_metrics
from spendsense.models.pipeline import get_model_pipeline


def train_and_evaluate_model(
    model_name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Optional[SpendSenseConfig] = None,
    categories: Optional[CategoriesConfig] = None,
    text_col: str = "normalized_description",
    target_col: str = "true_category",
    save_path: Optional[Path] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Trains a single model pipeline on train_df and evaluates on val_df.

    Returns:
        (fitted_pipeline, metrics_dict)
    """
    if config is None or categories is None:
        cfg, cats = get_config()
        config = config or cfg
        categories = categories or cats

    # Prepare keyword map for baseline rules
    keyword_map = {cat.name: cat.keywords for cat in categories.canonical_categories}

    pipeline = get_model_pipeline(
        model_name=model_name,
        config=config,
        keyword_map=keyword_map,
    )

    X_train = train_df[text_col].astype(str)
    y_train = train_df[target_col].astype(str)

    X_val = val_df[text_col].astype(str)
    y_val = val_df[target_col].astype(str)

    # Train
    start_time = time.perf_counter()
    pipeline.fit(X_train, y_train)
    training_duration = time.perf_counter() - start_time

    # Predict
    predict_start = time.perf_counter()
    y_pred = pipeline.predict(X_val)
    inference_duration = time.perf_counter() - predict_start

    # Evaluate
    canonical_labels = categories.canonical_names
    metrics = compute_classification_metrics(list(y_val), list(y_pred), labels=canonical_labels)
    metrics["model_name"] = model_name
    metrics["training_time_sec"] = round(training_duration, 4)
    metrics["inference_time_sec"] = round(inference_duration, 4)
    metrics["train_sample_size"] = len(train_df)
    metrics["val_sample_size"] = len(val_df)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, save_path)
        metrics["artifact_path"] = str(save_path)

    return pipeline, metrics


def compare_all_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    config: Optional[SpendSenseConfig] = None,
    categories: Optional[CategoriesConfig] = None,
    model_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Trains all baselines and candidates, outputting a side-by-side comparison table."""
    if model_names is None:
        model_names = ["dummy", "keyword", "naive_bayes", "logistic_regression", "linear_svc"]

    rows = []
    for model_name in model_names:
        _, metrics = train_and_evaluate_model(
            model_name=model_name,
            train_df=train_df,
            val_df=val_df,
            config=config,
            categories=categories,
        )
        rows.append({
            "Model": model_name,
            "Macro F1": metrics["macro_f1"],
            "Weighted F1": metrics["weighted_f1"],
            "Accuracy": metrics["accuracy"],
            "Train Time (s)": metrics["training_time_sec"],
            "Inference Time (s)": metrics["inference_time_sec"],
        })

    comparison_df = pd.DataFrame(rows).sort_values(by="Macro F1", ascending=False).reset_index(drop=True)
    return comparison_df
