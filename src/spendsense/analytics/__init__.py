"""Analytics package: spending anomaly alerts, recurring payment patterns, and summaries."""

from spendsense.analytics.anomaly import AnomalyAlert, detect_spending_anomalies
from spendsense.analytics.recurring import RecurringPaymentItem, detect_recurring_payments
from spendsense.analytics.summaries import generate_spending_summary

__all__ = [
    "detect_spending_anomalies",
    "AnomalyAlert",
    "detect_recurring_payments",
    "RecurringPaymentItem",
    "generate_spending_summary",
]
