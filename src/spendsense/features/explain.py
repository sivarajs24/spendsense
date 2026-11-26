"""Model explainability and feature attribution module for SpendSense.

Extracts exact linear attributions (TF-IDF weight * classifier coefficient)
and builds publication-grade SHAP waterfall charts to explain why a transaction
was assigned to a specific category.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import plotly.graph_objects as go


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
        Dictionary containing predicted category, top features, and waterfall chart steps.
    """
    if not text or not text.strip():
        return {
            "predicted_category": "Uncategorized",
            "top_features": [],
            "waterfall_steps": [],
            "method": "empty_input",
        }

    # Extract raw pipeline from CalibratedSpendSenseModel wrapper if applicable
    pipeline = getattr(model, "raw_pipeline", model)

    if not hasattr(pipeline, "named_steps") or "features" not in pipeline.named_steps:
        return {
            "predicted_category": target_category or "Uncategorized",
            "top_features": [],
            "waterfall_steps": [],
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
            "waterfall_steps": [],
            "method": "feature_extraction_failed",
        }

    # Identify non-zero feature indices for this input
    non_zero_indices = X_vec.nonzero()[1]
    if len(non_zero_indices) == 0:
        return {
            "predicted_category": target_category or "Uncategorized",
            "top_features": [],
            "waterfall_steps": [],
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

        # Intercept / Prior Base Bias E[f(x)]
        base_bias = 0.0
        if hasattr(classifier_step, "intercept_"):
            intercept_arr = classifier_step.intercept_
            if intercept_arr.ndim == 1 and len(intercept_arr) == 1:
                base_bias = float(intercept_arr[0] if class_idx == 1 else -intercept_arr[0])
            elif len(intercept_arr) > class_idx:
                base_bias = float(intercept_arr[class_idx])

        # Exact LinearSHAP attribution: phi_i = tfidf_i * w_i
        contributions = tfidf_values * class_coefs

        # Sort features by absolute contribution magnitude
        sort_order = np.argsort(np.abs(contributions))[::-1]
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

        # Construct SHAP Waterfall Steps
        waterfall_steps: List[Dict[str, Any]] = [
            {
                "label": "Base Bias (E[f(x)])",
                "value": round(base_bias, 4),
                "measure": "relative",
            }
        ]

        for idx in top_indices:
            score = float(contributions[idx])
            waterfall_steps.append({
                "label": clean_names[idx],
                "value": round(score, 4),
                "measure": "relative",
            })

        # Bundle remaining tail tokens if any
        if len(sort_order) > top_n:
            rem_indices = sort_order[top_n:]
            rem_score = float(np.sum(contributions[rem_indices]))
            if abs(rem_score) >= 1e-4:
                waterfall_steps.append({
                    "label": f"+{len(rem_indices)} Other Sub-words",
                    "value": round(rem_score, 4),
                    "measure": "relative",
                })

        # Final Net Decision Margin f(x)
        total_margin = base_bias + float(np.sum(contributions))
        waterfall_steps.append({
            "label": "Net Margin f(x)",
            "value": round(total_margin, 4),
            "measure": "total",
        })

        return {
            "predicted_category": target_category,
            "base_value": round(base_bias, 4),
            "total_score": round(total_margin, 4),
            "top_features": top_features,
            "waterfall_steps": waterfall_steps,
            "method": "linear_coefficient_attribution",
        }

    # Fallback to pure TF-IDF magnitude if non-linear or dummy model
    sort_order = np.argsort(tfidf_values)[::-1][:top_n]
    top_features = [
        {"token": clean_names[i], "importance": round(float(tfidf_values[i]), 4), "is_positive": True}
        for i in sort_order
    ]

    waterfall_steps = [
        {"label": "Base Prior", "value": 0.0, "measure": "relative"}
    ]
    for i in sort_order:
        waterfall_steps.append({
            "label": clean_names[i],
            "value": round(float(tfidf_values[i]), 4),
            "measure": "relative",
        })
    waterfall_steps.append({
        "label": "Total TF-IDF Weight",
        "value": round(float(np.sum(tfidf_values)), 4),
        "measure": "total",
    })

    return {
        "predicted_category": target_category or "Uncategorized",
        "base_value": 0.0,
        "total_score": round(float(np.sum(tfidf_values)), 4),
        "top_features": top_features,
        "waterfall_steps": waterfall_steps,
        "method": "tfidf_magnitude",
    }


def build_waterfall_figure(explanation: Dict[str, Any]) -> go.Figure:
    """Builds a high-contrast executive SHAP waterfall chart from explanation steps."""
    steps = explanation.get("waterfall_steps", [])
    if not steps:
        fig = go.Figure()
        fig.update_layout(
            annotations=[dict(text="No attribution tokens available", showarrow=False, font=dict(size=14, color="#64748B"))],
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=200,
        )
        return fig

    labels = [s["label"] for s in steps]
    values = [s["value"] for s in steps]
    measures = [s["measure"] for s in steps]

    text_labels = []
    for s in steps:
        val = s["value"]
        if s["measure"] == "total":
            text_labels.append(f"<b>{val:+.2f}</b>")
        else:
            text_labels.append(f"{val:+.2f}")

    fig = go.Figure(
        go.Waterfall(
            name="SHAP Attribution",
            orientation="v",
            measure=measures,
            x=labels,
            y=values,
            text=text_labels,
            textposition="outside",
            textfont=dict(family="JetBrains Mono, monospace", size=11, color="#0F172A"),
            connector={"line": {"color": "#CBD5E1", "width": 1.5, "dash": "dot"}},
            increasing={"marker": {"color": "#10B981"}},
            decreasing={"marker": {"color": "#EF4444"}},
            totals={"marker": {"color": "#4F46E5"}},
        )
    )

    pred_cat = explanation.get("predicted_category", "Category")
    fig.update_layout(
        title=dict(
            text=f"SHAP Logit Attribution Waterfall • <b>{pred_cat}</b>",
            font=dict(size=13, color="#0F172A", family="Plus Jakarta Sans, sans-serif"),
            x=0.01,
            y=0.96,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=320,
        showlegend=False,
        margin=dict(t=50, b=30, l=40, r=20),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(size=11, color="#334155", family="JetBrains Mono, monospace"),
        ),
        yaxis=dict(
            title=dict(text="Logit Contribution", font=dict(size=11, color="#64748B")),
            showgrid=True,
            gridcolor="#F1F5F9",
            tickfont=dict(size=11, color="#64748B"),
        ),
    )

    return fig
