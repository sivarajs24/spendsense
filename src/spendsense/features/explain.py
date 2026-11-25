"""Model explainability and feature attribution module for SpendSense.

Extracts exact linear attributions (TF-IDF weight * classifier coefficient)
to explain why a transaction was assigned to a specific category.
"""

from typing import Any, Dict, List, Optional
import numpy as np


def explain_text_prediction(
    text: str,
    model: Any,
    target_category: Optional[str] = None,
    top_n: int = 6,
) -> Dict[str, Any]:
    """Explains a model's prediction for an input transaction string.

    Args:
        text: Normalized or raw transaction description.
        model: CalibratedSpendSenseModel or standard sklearn Pipeline.
        target_category: Optional category to explain. If None, uses the predicted class.
        top_n: Maximum number of contributing features to return.

    Returns:
        Dictionary containing predicted category, top positive features, and top negative features.
    """
    if not text or not text.strip():
        return {
            "predicted_category": "Uncategorized",
            "top_features": [],
            "method": "empty_input",
        }

    # Extract raw pipeline from CalibratedSpendSenseModel wrapper if applicable
    pipeline = getattr(model, "raw_pipeline", model)

    if not hasattr(pipeline, "named_steps") or "features" not in pipeline.named_steps:
        return {
            "predicted_category": target_category or "Uncategorized",
            "top_features": [],
            "method": "unsupported_pipeline",
        }

    features_step = pipeline.named_steps["features"]
    classifier_step = pipeline.named_steps.get("classifier")

    try:
        X_vec = features_step.transform([text])
        feature_names = np.array(features_step.get_feature_names_out())
    except Exception:
        return {
            "predicted_category": target_category or "Uncategorized",
            "top_features": [],
            "method": "feature_extraction_failed",
        }

    # Identify non-zero feature indices for this input
    non_zero_indices = X_vec.nonzero()[1]
    if len(non_zero_indices) == 0:
        return {
            "predicted_category": target_category or "Uncategorized",
            "top_features": [],
            "method": "out_of_vocabulary",
        }

    tfidf_values = X_vec.data
    matched_names = feature_names[non_zero_indices]

    # Clean feature names (remove word_tfidf__ and char_tfidf__ prefixes)
    clean_names = []
    for name in matched_names:
        if "__" in name:
            name = name.split("__", 1)[1]
        clean_names.append(name)

    # If linear classifier coefficients are available
    if classifier_step is not None and hasattr(classifier_step, "coef_") and hasattr(classifier_step, "classes_"):
        classes = list(classifier_step.classes_)
        if target_category and target_category in classes:
            class_idx = classes.index(target_category)
        else:
            # Predict category
            pred_class = pipeline.predict([text])[0]
            class_idx = classes.index(pred_class) if pred_class in classes else 0
            target_category = str(classes[class_idx])

        coef_matrix = classifier_step.coef_
        if coef_matrix.ndim == 1:
            class_coefs = coef_matrix[non_zero_indices]
        elif coef_matrix.shape[0] == 1:
            # Binary classification in scikit-learn: row 0 corresponds to classes[1]
            raw_coefs = coef_matrix[0, non_zero_indices]
            class_coefs = raw_coefs if class_idx == 1 else -raw_coefs
        else:
            class_coefs = coef_matrix[class_idx, non_zero_indices]

        # Linear attribution = tfidf * weight
        contributions = tfidf_values * class_coefs

        # Sort by contribution
        sort_order = np.argsort(contributions)[::-1]
        top_indices = sort_order[:top_n]

        top_features: List[Dict[str, Any]] = []
        for idx in top_indices:
            feat_name = clean_names[idx]
            score = float(contributions[idx])
            top_features.append({
                "token": feat_name,
                "importance": round(score, 4),
                "is_positive": score > 0,
            })

        return {
            "predicted_category": target_category,
            "top_features": top_features,
            "method": "linear_coefficient_attribution",
        }

    # Fallback to pure TF-IDF magnitude if non-linear or dummy model
    sort_order = np.argsort(tfidf_values)[::-1][:top_n]
    top_features = [
        {"token": clean_names[i], "importance": round(float(tfidf_values[i]), 4), "is_positive": True}
        for i in sort_order
    ]

    return {
        "predicted_category": target_category or "Uncategorized",
        "top_features": top_features,
        "method": "tfidf_magnitude",
    }
