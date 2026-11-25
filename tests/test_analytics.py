"""Unit tests for spending anomaly detection, recurring payment detection, and summaries."""

import pandas as pd
import pytest

from spendsense.analytics.anomaly import detect_spending_anomalies
from spendsense.analytics.recurring import detect_recurring_payments
from spendsense.analytics.summaries import generate_spending_summary


def test_detect_spending_anomalies_flags_outliers():
    """Verify statistical IQR fence flags an unusually high expense."""
    # 10 normal food transactions around Rs. 300-500, and one extreme Rs. 15,000 transaction
    data = {
        "transaction_id": [f"txn_{i}" for i in range(11)],
        "date": ["2026-01-01"] * 11,
        "raw_description": ["swiggy"] * 10 + ["swiggy luxury party"],
        "normalized_description": ["swiggy"] * 10 + ["swiggy luxury party"],
        "amount": [-350.0, -420.0, -380.0, -490.0, -310.0, -450.0, -390.0, -520.0, -400.0, -410.0, -15000.0],
        "category": ["Food & Dining"] * 11,
    }
    df = pd.DataFrame(data)

    alerts = detect_spending_anomalies(df, min_history_transactions=5, iqr_multiplier=1.5)
    anomalies = [a for a in alerts if a.is_anomaly]

    assert len(anomalies) == 1
    assert anomalies[0].amount == -15000.0
    assert "Unusually high spending" in anomalies[0].explanation


def test_anomaly_detection_handles_insufficient_history():
    """Verify anomaly detection abstains when history is below threshold."""
    df = pd.DataFrame({
        "transaction_id": ["txn_1"],
        "date": ["2026-01-01"],
        "raw_description": ["rare transaction"],
        "amount": [-50000.0],
        "category": ["Government & Legal"],
    })
    alerts = detect_spending_anomalies(df, min_history_transactions=5)
    assert len(alerts) == 1
    assert alerts[0].is_anomaly is False
    assert "Insufficient" in alerts[0].explanation


def test_detect_recurring_payments_identifies_monthly_subscription():
    """Verify recurring payments module identifies regular monthly cadence."""
    # 4 monthly Netflix charges of Rs. 649 on consecutive months
    df = pd.DataFrame({
        "merchant_group": ["netflix"] * 4,
        "raw_description": ["NETFLIX.COM MUMBAI"] * 4,
        "date": ["2026-01-05", "2026-02-05", "2026-03-05", "2026-04-05"],
        "amount": [-649.0, -649.0, -649.0, -649.0],
        "category": ["Entertainment & Recreation"] * 4,
    })

    recurring = detect_recurring_payments(df, min_occurrences=3)
    assert len(recurring) == 1
    assert recurring[0].merchant_group == "netflix"
    assert recurring[0].estimated_frequency == "monthly"
    assert recurring[0].mean_amount == 649.0
    assert recurring[0].occurrences == 4


def test_generate_spending_summary_calculations():
    """Verify total inflows, outflows, and review rate calculations."""
    df = pd.DataFrame({
        "amount": [-500.0, -1500.0, 50000.0],
        "category": ["Food & Dining", "Transportation", "Income"],
        "requires_review": [False, True, False],
        "date": ["2026-01-10", "2026-01-15", "2026-01-31"],
    })

    summary = generate_spending_summary(df)
    assert summary["total_transactions"] == 3
    assert summary["total_debits"] == 2000.0
    assert summary["total_credits"] == 50000.0
    assert summary["net_flow"] == 48000.0
    assert summary["requires_review_count"] == 1
    assert summary["review_rate_pct"] == 33.33
    assert len(summary["category_breakdown"]) == 2  # Food & Transport (outflows)
