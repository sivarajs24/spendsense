"""Models package for SpendSense: baselines, linear classifiers, evaluation, and pipelines."""

from spendsense.models.baseline_rules import KeywordBaselineClassifier
from spendsense.models.calibrate import CalibratedSpendSenseModel, PredictionOutput
from spendsense.models.evaluate import compute_classification_metrics
from spendsense.models.pipeline import get_model_pipeline
from spendsense.models.registry import ModelRegistry
from spendsense.models.train import compare_all_models, train_and_evaluate_model

__all__ = [
    "KeywordBaselineClassifier",
    "CalibratedSpendSenseModel",
    "PredictionOutput",
    "ModelRegistry",
    "get_model_pipeline",
    "compute_classification_metrics",
    "train_and_evaluate_model",
    "compare_all_models",
]
