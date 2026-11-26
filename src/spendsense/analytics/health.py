"""Financial Vitality Score & Emergency Fund Runway Engine.

Calculates a 0-100 holistic financial wellness index and projects liquidity runway
in months of essential living expenses.
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from spendsense.analytics.anomaly import detect_spending_anomalies
from spendsense.analytics.budget import analyze_50_30_20_budget
from spendsense.analytics.recurring import detect_recurring_payments


class VitalityComponent(BaseModel):
    """Sub-score evaluation for a specific financial wellness dimension."""
    component_name: str
    score: float
    max_score: float
    status: str  # "Excellent", "Good", "Needs Attention", "Critical"
    insight: str


class FinancialHealthReport(BaseModel):
    """Master financial health vitality index & cash runway analysis."""
    vitality_score: int  # 0 to 100
    tier: str  # "Elite / Fortress", "Healthy / Resilient", "Vulnerable", "Critical Risk"
    tier_color: str
    components: List[VitalityComponent]
    emergency_runway_months: float
    liquid_reserves: float
    monthly_essential_burn: float
    runway_status: str  # "Fortified (6+ mos)", "Adequate (3-6 mos)", "Underfunded (< 3 mos)"
    executive_verdict: str


def calculate_financial_health(
    df: pd.DataFrame,
    override_liquid_balance: Optional[float] = None,
) -> FinancialHealthReport:
    """Computes a composite financial vitality score and liquid runway.

    Args:
        df: DataFrame containing statement records with 'amount', 'category', 'date'.
        override_liquid_balance: Optional current cash balance. Inferred from 'balance' column if present.

    Returns:
        Structured FinancialHealthReport.
    """
    if df.empty or "amount" not in df.columns:
        return FinancialHealthReport(
            vitality_score=50,
            tier="Neutral / Unassessed",
            tier_color="#94A3B8",
            components=[],
            emergency_runway_months=0.0,
            liquid_reserves=0.0,
            monthly_essential_burn=0.0,
            runway_status="No Data",
            executive_verdict="Import transaction history to evaluate financial vitality.",
        )

    budget_rep = analyze_50_30_20_budget(df)
    income = max(1.0, budget_rep.total_inflow)

    # 1. Savings Rate Dimension (30 points)
    savings_rate = budget_rep.actual_savings_rate_pct
    if savings_rate >= 30.0:
        c1_score = 30.0
        c1_status = "Excellent"
        c1_note = f"Exceptional capital accumulation ({savings_rate:.1f}% savings rate)."
    elif savings_rate >= 20.0:
        c1_score = 25.0
        c1_status = "Good"
        c1_note = f"Meets classic 20% savings target ({savings_rate:.1f}%)."
    elif savings_rate >= 10.0:
        c1_score = 16.0
        c1_status = "Needs Attention"
        c1_note = f"Below 20% benchmark ({savings_rate:.1f}%). Moderate wealth accumulation."
    elif savings_rate > 0.0:
        c1_score = 8.0
        c1_status = "Critical"
        c1_note = f"Thin savings buffer ({savings_rate:.1f}%). High vulnerability to shocks."
    else:
        c1_score = 0.0
        c1_status = "Critical"
        c1_note = "Net negative cash flow. Living beyond monthly inflows."

    # 2. Discretionary / Wants Discipline (25 points)
    wants_pct = budget_rep.buckets["Wants (30%)"].actual_pct
    if wants_pct <= 25.0:
        c2_score = 25.0
        c2_status = "Excellent"
        c2_note = f"Disciplined lifestyle spending ({wants_pct:.1f}% of income)."
    elif wants_pct <= 35.0:
        c2_score = 20.0
        c2_status = "Good"
        c2_note = f"Healthy balance between living and saving ({wants_pct:.1f}%)."
    elif wants_pct <= 50.0:
        c2_score = 10.0
        c2_status = "Needs Attention"
        c2_note = f"Discretionary outlays ({wants_pct:.1f}%) crowd out investments."
    else:
        c2_score = 4.0
        c2_status = "Critical"
        c2_note = f"Excessive lifestyle burn ({wants_pct:.1f}% of total earnings)."

    # 3. Expense Stability & Anomaly Control (25 points)
    alerts = detect_spending_anomalies(df, min_history_transactions=3)
    anomalies = [a for a in alerts if a.is_anomaly]
    n_anomalies = len(anomalies)

    if n_anomalies == 0:
        c3_score = 25.0
        c3_status = "Excellent"
        c3_note = "Zero statistical spending spikes. High predictability."
    elif n_anomalies == 1:
        c3_score = 20.0
        c3_status = "Good"
        c3_note = "1 unusual outlay detected. Well-contained volatility."
    elif n_anomalies == 2:
        c3_score = 14.0
        c3_status = "Needs Attention"
        c3_note = f"2 unexpected high-ticket purchases flagged."
    else:
        c3_score = 6.0
        c3_status = "Critical"
        c3_note = f"{n_anomalies} spending anomalies detected. Erratic outflow volatility."

    # 4. Committed Fixed-Cost Burden (20 points)
    recurrings = detect_recurring_payments(df)
    monthly_fixed = sum(r.mean_amount for r in recurrings if r.estimated_frequency == "monthly")
    fixed_ratio = (monthly_fixed / income) * 100.0

    if fixed_ratio <= 15.0:
        c4_score = 20.0
        c4_status = "Excellent"
        c4_note = f"Light fixed debt & subscription load ({fixed_ratio:.1f}% of income)."
    elif fixed_ratio <= 25.0:
        c4_score = 15.0
        c4_status = "Good"
        c4_note = f"Manageable committed obligations ({fixed_ratio:.1f}%)."
    elif fixed_ratio <= 40.0:
        c4_score = 8.0
        c4_status = "Needs Attention"
        c4_note = f"Heavy recurring bill obligations ({fixed_ratio:.1f}%). High overhead."
    else:
        c4_score = 2.0
        c4_status = "Critical"
        c4_note = f"Dangerously elevated fixed overhead ({fixed_ratio:.1f}%)."

    total_score = int(round(c1_score + c2_score + c3_score + c4_score))
    total_score = max(0, min(100, total_score))

    if total_score >= 85:
        tier = "Elite / Fortress"
        tier_color = "#10B981"
        verdict = "Outstanding capital health. High savings rate, disciplined lifestyle, and resilient financial runway."
    elif total_score >= 70:
        tier = "Healthy / Resilient"
        tier_color = "#3B82F6"
        verdict = "Solid financial fundamentals with sustainable savings and controlled discretionary burn."
    elif total_score >= 50:
        tier = "Vulnerable / Attention Needed"
        tier_color = "#F59E0B"
        verdict = "Elevated discretionary or fixed commitments constrain wealth accumulation. Review leaks."
    else:
        tier = "Critical / High Risk"
        tier_color = "#EF4444"
        verdict = "Expenses closely track or exceed inflows with minimal liquidity cushion."

    # Emergency Runway Calculation
    # Monthly Essential Burn = Needs (Utilities + Medical + Basic Transportation)
    needs_spent = budget_rep.buckets["Needs (50%)"].allocated_amount
    monthly_essential = max(5000.0, needs_spent)

    if override_liquid_balance is not None and override_liquid_balance > 0:
        liquid_reserves = float(override_liquid_balance)
    elif "balance" in df.columns and pd.notna(df["balance"]).any():
        valid_balances = df["balance"].dropna().values
        liquid_reserves = float(valid_balances[-1]) if len(valid_balances) > 0 else (income * 2.5)
    else:
        # Conservative baseline assumption
        liquid_reserves = income * 2.0

    runway_months = round(liquid_reserves / monthly_essential, 1) if monthly_essential > 0 else 0.0

    if runway_months >= 6.0:
        runway_status = "Fortified (6+ Months)"
    elif runway_months >= 3.0:
        runway_status = "Adequate (3–6 Months)"
    else:
        runway_status = "Underfunded (< 3 Months)"

    components = [
        VitalityComponent(
            component_name="Capital Accumulation (Savings Rate)",
            score=c1_score,
            max_score=30.0,
            status=c1_status,
            insight=c1_note,
        ),
        VitalityComponent(
            component_name="Lifestyle Discipline (Wants Allocation)",
            score=c2_score,
            max_score=25.0,
            status=c2_status,
            insight=c2_note,
        ),
        VitalityComponent(
            component_name="Outflow Stability (Anomaly Control)",
            score=c3_score,
            max_score=25.0,
            status=c3_status,
            insight=c3_note,
        ),
        VitalityComponent(
            component_name="Fixed-Cost Overhead (Subscription Burden)",
            score=c4_score,
            max_score=20.0,
            status=c4_status,
            insight=c4_note,
        ),
    ]

    return FinancialHealthReport(
        vitality_score=total_score,
        tier=tier,
        tier_color=tier_color,
        components=components,
        emergency_runway_months=runway_months,
        liquid_reserves=round(liquid_reserves, 2),
        monthly_essential_burn=round(monthly_essential, 2),
        runway_status=runway_status,
        executive_verdict=verdict,
    )
