"""Deterministic keyword baseline classifier for benchmarking classical ML models."""

from collections import Counter
from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class KeywordBaselineClassifier(BaseEstimator, ClassifierMixin):
    """Rule-based text categorization baseline.

    Matches token substrings against a predefined keyword dictionary.
    If no keywords match, falls back to the empirical majority class from training.
    """

    def __init__(self, keyword_map: Optional[Dict[str, List[str]]] = None):
        """Args:

        keyword_map: Dictionary mapping Category Name -> List of keyword
        strings.
        """
        self.keyword_map = keyword_map or {}
        self.default_class_: Optional[str] = None
        self.classes_: np.ndarray = np.array([])

    def fit(self, X: Any, y: Any) -> "KeywordBaselineClassifier":
        """Fits the baseline by identifying available classes and majority class fallback."""
        y_list = list(y)
        if not y_list:
            raise ValueError("Training target y cannot be empty.")

        counts = Counter(y_list)
        self.default_class_ = counts.most_common(1)[0][0]
        self.classes_ = np.array(sorted(list(set(y_list))))
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predicts categories by keyword matching or fallback."""
        if self.default_class_ is None:
            raise RuntimeError("Classifier has not been fitted yet. Call fit() first.")

        predictions = []
        for text in X:
            s = str(text).lower()
            matched_category = None

            # First match wins (deterministic order)
            for cat_name, keywords in self.keyword_map.items():
                for kw in keywords:
                    if kw.lower() in s:
                        matched_category = cat_name
                        break
                if matched_category:
                    break

            predictions.append(matched_category or self.default_class_)

        return np.array(predictions)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Generates pseudo-probabilities (1.0 for keyword match, uniform distribution for fallback)."""
        if self.default_class_ is None or len(self.classes_) == 0:
            raise RuntimeError("Classifier has not been fitted yet.")

        n_classes = len(self.classes_)
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        probas = np.zeros((len(X), n_classes))

        for row_idx, text in enumerate(X):
            s = str(text).lower()
            matched_category = None
            for cat_name, keywords in self.keyword_map.items():
                for kw in keywords:
                    if kw.lower() in s:
                        matched_category = cat_name
                        break
                if matched_category:
                    break

            if matched_category and matched_category in class_to_idx:
                probas[row_idx, class_to_idx[matched_category]] = 1.0
            else:
                # Assign fallback class high weight
                probas[row_idx, class_to_idx[self.default_class_]] = 0.5
                remaining_mass = 0.5 / max(1, n_classes - 1)
                for c_idx in range(n_classes):
                    if c_idx != class_to_idx[self.default_class_]:
                        probas[row_idx, c_idx] = remaining_mass

        return probas
