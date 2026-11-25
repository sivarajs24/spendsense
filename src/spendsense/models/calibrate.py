"""Probability calibration and confidence-aware prediction service.

Wraps linear models (e.g. LinearSVC) with Platt Scaling (CalibratedClassifierCV)
to generate defensible, calibrated probabilities and enforce an abstention/review queue.
"""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, Field
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline


class PredictionOutput(BaseModel):
    """Structured response for a single transaction prediction."""
    predicted_category: str
    confidence: float = Field(ge=0.0, le=1.0)
    requires_review: bool
    review_reason: Optional[str] = None
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    model_version: str = "1.0.0"


class CalibratedSpendSenseModel:
    """Wrapper that provides calibrated probabilities and review queue routing."""

    def __init__(
        self,
        fitted_pipeline: Pipeline,
        model_version: str = "1.0.0",
        confidence_threshold: float = 0.70,
    ):
        self.raw_pipeline = fitted_pipeline
        self.model_version = model_version
        self.confidence_threshold = confidence_threshold
        self.calibrated_model_: Optional[Any] = None
        self.classes_: np.ndarray = np.array([])

    def fit_calibration(
        self,
        X_val: Any,
        y_val: Any,
        method: str = "sigmoid",
    ) -> "CalibratedSpendSenseModel":
        """Fits Platt scaling (sigmoid) on an independent validation set.

        Ensures calibration parameters are never learned on the training set.
        """
        calibrator = CalibratedClassifierCV(
            estimator=self.raw_pipeline,
            method=method,
            cv="prefit",
        )
        calibrator.fit(X_val, y_val)
        self.calibrated_model_ = calibrator
        self.classes_ = calibrator.classes_
        return self

    def predict_single(
        self,
        normalized_description: str,
        threshold: Optional[float] = None,
    ) -> PredictionOutput:
        """Predicts category with calibrated confidence for a single description."""
        batch_results = self.predict_batch([normalized_description], threshold=threshold)
        return batch_results[0]

    def predict_batch(
        self,
        descriptions: List[str],
        threshold: Optional[float] = None,
    ) -> List[PredictionOutput]:
        """Generates confidence-aware predictions for a batch of descriptions."""
        active_threshold = threshold if threshold is not None else self.confidence_threshold

        if not descriptions:
            return []

        # If model is calibrated, use predict_proba; otherwise fall back to raw pipeline
        if self.calibrated_model_ is not None:
            proba_matrix = self.calibrated_model_.predict_proba(descriptions)
            classes = self.classes_
        elif hasattr(self.raw_pipeline, "predict_proba"):
            proba_matrix = self.raw_pipeline.predict_proba(descriptions)
            classes = self.raw_pipeline.classes_
        else:
            # Fallback for uncalibrated models: decision function with softmax
            decision = self.raw_pipeline.decision_function(descriptions)
            exp_dec = np.exp(decision - np.max(decision, axis=1, keepdims=True))
            proba_matrix = exp_dec / np.sum(exp_dec, axis=1, keepdims=True)
            classes = self.raw_pipeline.classes_

        results = []
        for i, probs in enumerate(proba_matrix):
            # Sort categories by probability descending
            sorted_indices = np.argsort(probs)[::-1]
            top_idx = sorted_indices[0]
            top_category = str(classes[top_idx])
            top_confidence = float(probs[top_idx])

            # Prepare top-2 alternative candidates
            alternatives = []
            for alt_idx in sorted_indices[1:3]:
                alternatives.append({
                    "category": str(classes[alt_idx]),
                    "confidence": round(float(probs[alt_idx]), 4),
                })

            requires_review = top_confidence < active_threshold
            review_reason = None
            if requires_review:
                alt_info = (
                    f" Alternative candidate: '{alternatives[0]['category']}' ({alternatives[0]['confidence']:.1%})."
                    if alternatives
                    else ""
                )
                review_reason = (
                    f"Prediction confidence ({top_confidence:.1%}) is below the "
                    f"configured review threshold ({active_threshold:.1%}).{alt_info}"
                )

            results.append(PredictionOutput(
                predicted_category=top_category,
                confidence=round(top_confidence, 4),
                requires_review=requires_review,
                review_reason=review_reason,
                alternatives=alternatives,
                model_version=self.model_version,
            ))

        return results
