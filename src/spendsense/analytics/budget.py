"""50/30/20 Budgeting Rule & Mid-Month Burn-Rate Forecaster.

Classifies categorized expenses into Needs (50%), Wants (30%), and Savings (20%),
and projects end-of-month trajectory with breach alert forecasting.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional
import pandas as pd
from pydantic import BaseModel, Field


class BudgetBucket(BaseModel):
    """Structured allocation breakdown for one of the 50/30/20 buckets."""
    bucket_name: str  # "Needs", "Wants", "Savings & Investments"
    target_pct: float
    actual_pct: float
    allocated_amount: float
    target_amount: float
    status: str  # "Optimal", "Under Budget", "Over Budget"
    categories: List[str] = Field(default_factory=list)


class BurnForecast(BaseModel):
    """Pace-meter projection estimating month-end burn and budget breach day."""
    bucket_name: str
    observed_days: int
    total_days_in_month: int
    spent_to_date: float
    daily_burn_rate: float
    projected_month_end_spend: float
    budget_target: float
    projected_overspend: float
    estimated_breach_day: Optional[int] = None
    pace_status: str  # "On Track", "Accelerating", "Critical Overspend"
    recommendation: str


class BudgetReport(BaseModel):
    """Master financial budget and burn forecast analysis."""
    total_inflow: float
    effective_outflow: float
    net_savings: float
    actual_savings_rate_pct: float
    buckets: Dict[str, BudgetBucket]
    wants_forecast: BurnForecast
    needs_forecast: BurnForecast


# Canonical category mappings to 50/30/20 buckets
NEEDS_CATEGORIES = {
    "Utilities & Services",
    "Healthcare & Medical",
    "Government & Legal",
    "Transportation",
}

WANTS_CATEGORIES = {
    "Entertainment & Recreation",
    "Food & Dining",
    "Shopping & Retail",
    "Charity & Donations",
}

SAVINGS_CATEGORIES = {
    "Financial Services",
}


def analyze_50_30_20_budget(
    df: pd.DataFrame,
    override_income: Optional[float] = None,
    override_inflow: Optional[float] = None,
) -> BudgetReport:
    """Analyzes personal financial distribution against the 50/30/20 rule.

    Args:
        df: DataFrame containing 'amount', 'category' (or 'predicted_category'), and 'date'.
        override_income: Optional explicit monthly income. If None, infers from deposits.
        override_inflow: Alias for override_income.

    Returns:
        Structured BudgetReport with allocations and forward-looking burn forecasts.
    """
    if override_inflow is not None and override_income is None:
        override_income = override_inflow

    cat_col = "predicted_category" if "predicted_category" in df.columns else "category"

    # Calculate Inflow & Outflow
    if override_income is not None and override_income > 0:
        total_income = float(override_income)
    else:
        credits = df[df["amount"] > 0]["amount"].sum() if "amount" in df.columns else 0.0
        total_income = float(credits) if credits > 0 else 50000.0  # Conservative default baseline

    debits = df[df["amount"] < 0].copy() if "amount" in df.columns else pd.DataFrame()
    # Exclude internal transfers if already tagged to avoid double-counting
    if "is_internal_transfer" in debits.columns:
        debits = debits[~debits["is_internal_transfer"]].copy()

    debits["abs_amount"] = debits["amount"].abs() if not debits.empty else []

    needs_sum = 0.0
    wants_sum = 0.0
    savings_sum = 0.0

    if not debits.empty and cat_col in debits.columns:
        for cat, group in debits.groupby(cat_col):
            amt = float(group["abs_amount"].sum())
            if cat in NEEDS_CATEGORIES:
                needs_sum += amt
            elif cat in WANTS_CATEGORIES:
                wants_sum += amt
            elif cat in SAVINGS_CATEGORIES:
                savings_sum += amt
            else:
                # Default split for uncategorized
                needs_sum += amt * 0.50
                wants_sum += amt * 0.50

    effective_outflow = needs_sum + wants_sum + savings_sum
    residual_savings = max(0.0, total_income - (needs_sum + wants_sum))
    total_savings_invested = savings_sum + residual_savings
    savings_rate = (total_savings_invested / total_income * 100.0) if total_income > 0 else 0.0

    # Bucket Allocations
    needs_target = total_income * 0.50
    wants_target = total_income * 0.30
    savings_target = total_income * 0.20

    needs_actual_pct = (needs_sum / total_income * 100.0) if total_income > 0 else 0.0
    wants_actual_pct = (wants_sum / total_income * 100.0) if total_income > 0 else 0.0
    savings_actual_pct = (total_savings_invested / total_income * 100.0) if total_income > 0 else 0.0

    def get_status(actual: float, target: float, is_savings: bool = False) -> str:
        if is_savings:
            return "Optimal" if actual >= target else "Under Target"
        if actual > target * 1.05:
            return "Over Budget"
        elif actual < target * 0.90:
            return "Under Budget"
        return "Optimal"

    buckets = {
        "Needs (50%)": BudgetBucket(
            bucket_name="Needs",
            target_pct=50.0,
            actual_pct=round(needs_actual_pct, 1),
            allocated_amount=round(needs_sum, 2),
            target_amount=round(needs_target, 2),
            status=get_status(needs_sum, needs_target),
            categories=list(NEEDS_CATEGORIES),
        ),
        "Wants (30%)": BudgetBucket(
            bucket_name="Wants",
            target_pct=30.0,
            actual_pct=round(wants_actual_pct, 1),
            allocated_amount=round(wants_sum, 2),
            target_amount=round(wants_target, 2),
            status=get_status(wants_sum, wants_target),
            categories=list(WANTS_CATEGORIES),
        ),
        "Savings (20%)": BudgetBucket(
            bucket_name="Savings & Investments",
            target_pct=20.0,
            actual_pct=round(savings_actual_pct, 1),
            allocated_amount=round(total_savings_invested, 2),
            target_amount=round(savings_target, 2),
            status=get_status(total_savings_invested, savings_target, is_savings=True),
            categories=list(SAVINGS_CATEGORIES),
        ),
    }

    # Mid-Month Burn-Rate Forecast (using observed dates)
    observed_days = 15
    total_days = 30
    if "date" in df.columns and pd.notna(df["date"]).any():
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        if not dates.empty:
            day_nums = dates.dt.day.values
            observed_days = int(max(1, min(30, max(day_nums))))

    def project_burn(bucket_name: str, spent: float, target: float) -> BurnForecast:
        daily_rate = spent / observed_days if observed_days > 0 else 0.0
        projected_total = daily_rate * total_days
        overspend = max(0.0, projected_total - target)

        breach_day = None
        if daily_rate > 0 and target > 0:
            potential_day = int(target / daily_rate)
            if potential_day < total_days:
                breach_day = max(1, potential_day)

        if overspend > target * 0.15:
            status = "Critical Overspend"
            rec = f"Curtail discretionary {bucket_name.lower()} spend by ₹{overspend / (total_days - observed_days):,.0f}/day to stay solvent."
        elif overspend > 0:
            status = "Accelerating"
            rec = f"Spend is pacing slightly ahead. Projected to breach target by Day {breach_day or 28}."
        else:
            status = "On Track"
            rec = f"Comfortably within monthly limit. Surplus capacity: ₹{abs(target - projected_total):,.0f}."

        return BurnForecast(
            bucket_name=bucket_name,
            observed_days=observed_days,
            total_days_in_month=total_days,
            spent_to_date=round(spent, 2),
            daily_burn_rate=round(daily_rate, 2),
            projected_month_end_spend=round(projected_total, 2),
            budget_target=round(target, 2),
            projected_overspend=round(overspend, 2),
            estimated_breach_day=breach_day,
            pace_status=status,
            recommendation=rec,
        )

    wants_forecast = project_burn("Wants", wants_sum, wants_target)
    needs_forecast = project_burn("Needs", needs_sum, needs_target)

    return BudgetReport(
        total_inflow=round(total_income, 2),
        effective_outflow=round(effective_outflow, 2),
        net_savings=round(total_savings_invested, 2),
        actual_savings_rate_pct=round(savings_rate, 1),
        buckets=buckets,
        wants_forecast=wants_forecast,
        needs_forecast=needs_forecast,
    )
