"""Recurring payment and subscription detection module.

Identifies potential recurring bills by analyzing merchant tokens, amount stability,
and calendar date intervals (weekly, monthly, annual).
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel


class RecurringPaymentItem(BaseModel):
    """Structured report for an identified recurring subscription/bill."""
    merchant_group: str
    sample_description: str
    category: str
    estimated_frequency: str  # 'weekly', 'monthly', 'annual', 'irregular'
    mean_amount: float
    occurrences: int
    interval_days_median: float
    confidence_score: float
    explanation: str


def detect_recurring_payments(
    df: pd.DataFrame,
    min_occurrences: int = 2,
    max_amount_pct_variance: float = 0.15,
) -> List[RecurringPaymentItem]:
    """Detects recurring expenses from transaction history.

    Args:
        df: DataFrame containing 'amount', 'date', 'merchant_group', and 'category'.
        min_occurrences: Minimum number of historical transactions needed.
        max_amount_pct_variance: Max allowed coefficient of variation in transaction amounts.

    Returns:
        List of RecurringPaymentItem objects.
    """
    if df.empty or "date" not in df.columns or "amount" not in df.columns:
        return []

    cat_col = "category" if "category" in df.columns else "predicted_category"
    merchant_col = "merchant_group" if "merchant_group" in df.columns else "normalized_description"

    # Filter for debits (expenses) with valid dates
    debits = df[(df["amount"] < 0) & (pd.notna(df["date"]))].copy()
    if debits.empty:
        return []

    debits["parsed_date"] = pd.to_datetime(debits["date"], errors="coerce")
    debits = debits.dropna(subset=["parsed_date"]).sort_values("parsed_date")

    recurring_candidates: List[RecurringPaymentItem] = []

    for merchant, group in debits.groupby(merchant_col):
        n = len(group)
        if n < min_occurrences:
            continue

        abs_amounts = group["amount"].abs().values
        mean_amt = float(np.mean(abs_amounts))
        std_amt = float(np.std(abs_amounts)) if n > 1 else 0.0

        # Amount stability check
        amount_variance = (std_amt / mean_amt) if mean_amt > 0 else 1.0
        if amount_variance > max_amount_pct_variance and n < 4:
            continue

        # Date interval regularity check
        dates = group["parsed_date"].values
        deltas = np.diff(dates) / np.timedelta64(1, "D")

        if len(deltas) == 0:
            continue

        median_delta = float(np.median(deltas))

        # Classify frequency based on interval
        if 5 <= median_delta <= 9:
            freq = "weekly"
            conf = 0.90
        elif 25 <= median_delta <= 35:
            freq = "monthly"
            conf = 0.95
        elif 80 <= median_delta <= 100:
            freq = "quarterly"
            conf = 0.85
        elif 350 <= median_delta <= 380:
            freq = "annual"
            conf = 0.90
        else:
            freq = "irregular"
            conf = 0.60

        # Skip irregular intervals unless amount is identical and count is high
        if freq == "irregular" and n < 3:
            continue

        sample_desc = str(group["raw_description"].iloc[0]) if "raw_description" in group.columns else str(merchant)
        category_name = str(group[cat_col].iloc[0]) if cat_col in group.columns else "Unknown"

        explanation = (
            f"Detected {n} transactions for '{merchant}' with ~Rs. {mean_amt:,.2f} average amount "
            f"every {median_delta:.0f} days ({freq} schedule)."
        )

        recurring_candidates.append(RecurringPaymentItem(
            merchant_group=str(merchant),
            sample_description=sample_desc,
            category=category_name,
            estimated_frequency=freq,
            mean_amount=round(mean_amt, 2),
            occurrences=n,
            interval_days_median=round(median_delta, 1),
            confidence_score=round(conf, 2),
            explanation=explanation,
        ))

    return sorted(recurring_candidates, key=lambda x: x.confidence_score, reverse=True)
