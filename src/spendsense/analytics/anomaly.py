"""Spending anomaly detection module.

Uses robust statistical fences (Interquartile Range / Median Absolute Deviation)
relative to a user's category-level baseline.
Explicitly avoids labeling alerts as 'fraud' (Principle #9).
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel


class AnomalyAlert(BaseModel):
    """Structured report for an unusual spending transaction."""
    transaction_id: str
    date: Optional[str]
    description: str
    amount: float
    category: str
    baseline_median: float
    baseline_q3: float
    threshold_limit: float
    is_anomaly: bool
    explanation: str


def detect_spending_anomalies(
    df: pd.DataFrame,
    min_history_transactions: int = 5,
    iqr_multiplier: float = 1.5,
) -> List[AnomalyAlert]:
    """Identifies unusually high debit transactions relative to category spending history.

    Args:
        df: DataFrame containing 'transaction_id', 'amount', 'direction', and 'category' (or 'predicted_category').
        min_history_transactions: Minimum transactions needed to establish a category baseline.
        iqr_multiplier: Tukey fence multiplier (typically 1.5 for outliers).

    Returns:
        List of AnomalyAlert objects.
    """
    if df.empty or "amount" not in df.columns:
        return []

    cat_col = "category" if "category" in df.columns else "predicted_category"
    if cat_col not in df.columns:
        return []

    # Focus only on expense outflows (debits)
    debits = df[df["amount"] < 0].copy()
    if debits.empty:
        return []

    debits["abs_amount"] = debits["amount"].abs()
    alerts: List[AnomalyAlert] = []

    for cat_name, group in debits.groupby(cat_col):
        amounts = group["abs_amount"].values
        n_samples = len(amounts)

        if n_samples < min_history_transactions:
            # Insufficient category history — do not raise false alarm
            for _, row in group.iterrows():
                alerts.append(AnomalyAlert(
                    transaction_id=str(row.get("transaction_id", "")),
                    date=str(row.get("date", "")),
                    description=str(row.get("normalized_description", row.get("raw_description", ""))),
                    amount=float(row["amount"]),
                    category=str(cat_name),
                    baseline_median=round(float(np.median(amounts)), 2),
                    baseline_q3=0.0,
                    threshold_limit=0.0,
                    is_anomaly=False,
                    explanation=f"Insufficient category history ({n_samples}/{min_history_transactions} txns) to establish statistical baseline.",
                ))
            continue

        q1 = float(np.percentile(amounts, 25))
        q3 = float(np.percentile(amounts, 75))
        iqr = q3 - q1
        median = float(np.median(amounts))
        upper_fence = q3 + (iqr_multiplier * max(iqr, median * 0.20))

        for _, row in group.iterrows():
            amt = float(row["abs_amount"])
            is_anomaly = amt > upper_fence

            if is_anomaly:
                explanation = (
                    f"Unusually high spending of Rs. {amt:,.2f} in '{cat_name}'. "
                    f"Typical median is Rs. {median:,.2f} (Upper baseline fence: Rs. {upper_fence:,.2f})."
                )
            else:
                explanation = f"Normal spending within expected range for '{cat_name}'."

            alerts.append(AnomalyAlert(
                transaction_id=str(row.get("transaction_id", "")),
                date=str(row.get("date", "")),
                description=str(row.get("normalized_description", row.get("raw_description", ""))),
                amount=float(row["amount"]),
                category=str(cat_name),
                baseline_median=round(median, 2),
                baseline_q3=round(q3, 2),
                threshold_limit=round(upper_fence, 2),
                is_anomaly=is_anomaly,
                explanation=explanation,
            ))

    return alerts
