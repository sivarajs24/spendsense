"""Money Leak & Subscription Redundancy Sentinel.

Identifies silent financial drains: duplicate memberships, overlapping streaming services,
and high-frequency micro-transactions silently compounding into significant annual burn.
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field

from spendsense.analytics.recurring import detect_recurring_payments


class MoneyLeakAlert(BaseModel):
    """Specific financial leak warning with estimated annual cost."""
    leak_id: str
    leak_type: str  # "Redundant Subscriptions", "Micro-Transaction Leak", "Delivery Duplication"
    severity: str  # "High", "Medium", "Low"
    annual_cost_estimate: float
    affected_merchants: List[str]
    explanation: str
    action_plan: str


class MoneyLeakReport(BaseModel):
    """Consolidated financial leak & subscription audit report."""
    total_annual_leak_estimate: float
    active_leak_count: int
    alerts: List[MoneyLeakAlert] = Field(default_factory=list)


# Overlapping service clusters
STREAMING_CLUSTER = {"netflix", "prime", "hotstar", "disney", "spotify", "apple music", "youtube premium"}
FOOD_DELIVERY_CLUSTER = {"swiggy", "zomato", "eats"}
QUICK_COMMERCE_CLUSTER = {"blinkit", "zepto", "instamart", "bbnow", "dunzo"}


def detect_money_leaks(df: pd.DataFrame) -> MoneyLeakReport:
    """Audits transactions to surface invisible money leaks and redundant commitments.

    Args:
        df: DataFrame containing bank statement records.

    Returns:
        Structured MoneyLeakReport with annual waste estimates and recommendations.
    """
    if df.empty or "amount" not in df.columns:
        return MoneyLeakReport(total_annual_leak_estimate=0.0, active_leak_count=0, alerts=[])

    alerts: List[MoneyLeakAlert] = []
    debits = df[df["amount"] < 0].copy()
    debits["abs_amount"] = debits["amount"].abs()

    # 1. Redundant Subscriptions Check
    recurrings = detect_recurring_payments(df)
    active_merchants = {r.merchant_group.lower(): r for r in recurrings}

    # Check Multiple Streaming
    overlapping_streaming = [m for m in active_merchants if any(s in m for s in STREAMING_CLUSTER)]
    if len(overlapping_streaming) >= 2:
        monthly_cost = sum(active_merchants[m].mean_amount for m in overlapping_streaming)
        annual_cost = monthly_cost * 12.0
        alerts.append(
            MoneyLeakAlert(
                leak_id="leak_streaming_redundancy",
                leak_type="Redundant Subscriptions",
                severity="Medium",
                annual_cost_estimate=round(annual_cost, 2),
                affected_merchants=[m.title() for m in overlapping_streaming],
                explanation=f"{len(overlapping_streaming)} concurrent digital media subscriptions active ({', '.join(m.title() for m in overlapping_streaming)}).",
                action_plan="Consolidate streaming services or rotate quarterly to save up to 50% of annualized media spend.",
            )
        )

    # Check Dual Food Delivery Memberships
    all_text = " ".join(debits.get("raw_description", "").astype(str).str.lower().tolist())
    has_swiggy = "swiggy" in all_text
    has_zomato = "zomato" in all_text
    if has_swiggy and has_zomato:
        # Calculate combined monthly dining spend
        swiggy_spend = debits[debits["raw_description"].str.contains("swiggy", case=False, na=False)]["abs_amount"].sum()
        zomato_spend = debits[debits["raw_description"].str.contains("zomato", case=False, na=False)]["abs_amount"].sum()
        combined = swiggy_spend + zomato_spend
        if combined >= 500.0:
            annual_leak = (combined * 0.15) * 12.0  # 15% estimated packaging/delivery fee overlap
            alerts.append(
                MoneyLeakAlert(
                    leak_id="leak_food_delivery_overlap",
                    leak_type="Delivery Duplication",
                    severity="Medium",
                    annual_cost_estimate=round(annual_leak, 2),
                    affected_merchants=["Swiggy", "Zomato"],
                    explanation=f"Splitting orders across both Swiggy (₹{swiggy_spend:,.0f}) and Zomato (₹{zomato_spend:,.0f}) doubles platform and membership fees.",
                    action_plan="Commit to one primary food ecosystem (Swiggy One or Zomato Gold) to avoid duplicate delivery charges.",
                )
            )

    # 2. High-Frequency Micro-Transaction Leak (< ₹150)
    micro_txns = debits[debits["abs_amount"] <= 150.0]
    if len(micro_txns) >= 8:
        total_micro = micro_txns["abs_amount"].sum()
        annual_micro = total_micro * 12.0
        alerts.append(
            MoneyLeakAlert(
                leak_id="leak_micro_transactions",
                leak_type="Micro-Transaction Leak",
                severity="High" if total_micro > 2500 else "Low",
                annual_cost_estimate=round(annual_micro, 2),
                affected_merchants=["UPI QR Vendors", "Convenience Purchases"],
                explanation=f"{len(micro_txns)} micro-purchases under ₹150 totaled ₹{total_micro:,.2f} this period (projected ₹{annual_micro:,.0f}/year).",
                action_plan="Small friction-free UPI QR payments accumulate invisibly. Set a daily discretionary pocket budget.",
            )
        )

    total_leak = sum(a.annual_cost_estimate for a in alerts)

    return MoneyLeakReport(
        total_annual_leak_estimate=round(total_leak, 2),
        active_leak_count=len(alerts),
        alerts=alerts,
    )
