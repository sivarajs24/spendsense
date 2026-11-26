"""Comprehensive test suite for advanced financial analytics modules:
Tax Intelligence, Transfer Reconciliation, 50/30/20 Budgeting, Financial Health, and Money Leaks.
"""

import pandas as pd
import pytest

from spendsense.analytics.budget import analyze_50_30_20_budget
from spendsense.analytics.health import calculate_financial_health
from spendsense.analytics.leaks import detect_money_leaks
from spendsense.analytics.tax import compare_tax_regimes, detect_tax_deductions
from spendsense.analytics.transfers import reconcile_inter_account_transfers


@pytest.fixture
def sample_financial_df() -> pd.DataFrame:
    """Fixture providing a diverse multi-category financial transaction statement."""
    data = [
        {"transaction_id": "t1", "date": "2026-01-01", "raw_description": "UPI/SWIGGY@ybl", "normalized_description": "swiggy", "amount": -450.0, "predicted_category": "Food & Dining"},
        {"transaction_id": "t2", "date": "2026-01-02", "raw_description": "ACH/NETFLIX.COM/MUMBAI", "normalized_description": "netflix", "amount": -649.0, "predicted_category": "Entertainment & Recreation"},
        {"transaction_id": "t3", "date": "2026-01-03", "raw_description": "SALARY CREDIT INFOSYS", "normalized_description": "salary credit infosys", "amount": 95000.0, "predicted_category": "Income"},
        {"transaction_id": "t4", "date": "2026-01-05", "raw_description": "ZERODHA BROKING SIP MUTUAL FUND", "normalized_description": "zerodha sip mutual fund", "amount": -15000.0, "predicted_category": "Financial Services"},
        {"transaction_id": "t5", "date": "2026-01-08", "raw_description": "APOLLO PHARMACY BANGALORE", "normalized_description": "apollo pharmacy", "amount": -1850.0, "predicted_category": "Healthcare & Medical"},
        {"transaction_id": "t6", "date": "2026-01-10", "raw_description": "HOUSE RENT PAYMENT TO LANDLORD", "normalized_description": "house rent payment landlord", "amount": -22000.0, "predicted_category": "Utilities & Services"},
        {"transaction_id": "t7", "date": "2026-01-15", "raw_description": "CRED/CREDIT CARD BILL PAYMENT AXIS", "normalized_description": "cred credit card bill payment axis", "amount": -12450.0, "predicted_category": "Financial Services"},
        {"transaction_id": "t8", "date": "2026-01-18", "raw_description": "TRANSFER TO AC 99812 SELF SAVINGS", "normalized_description": "transfer to ac self savings", "amount": -5000.0, "predicted_category": "Financial Services"},
        {"transaction_id": "t9", "date": "2026-01-20", "raw_description": "SPOTIFY INDIA SUBSCRIPTION", "normalized_description": "spotify india subscription", "amount": -119.0, "predicted_category": "Entertainment & Recreation"},
        {"transaction_id": "t10", "date": "2026-01-22", "raw_description": "ZOMATO RESTAURANT DINING", "normalized_description": "zomato restaurant dining", "amount": -520.0, "predicted_category": "Food & Dining"},
        {"transaction_id": "t11", "date": "2026-01-24", "raw_description": "PM CARES FUND DONATION", "normalized_description": "pm cares fund donation", "amount": -1000.0, "predicted_category": "Charity & Donations"},
    ]
    return pd.DataFrame(data)


def test_detect_tax_deductions_80c_80d_hra(sample_financial_df):
    """Verify Section 80C, 80D, HRA and 80G tax tagging with statutory caps."""
    tax_rep = detect_tax_deductions(sample_financial_df, senior_citizen=False)

    # Zerodha SIP (₹15,000) under 80C
    assert tax_rep.total_80c_eligible == 15000.0
    assert tax_rep.total_80c_claimed == 15000.0

    # Apollo Pharmacy (₹1,850) under 80D
    assert tax_rep.total_80d_eligible == 1850.0
    assert tax_rep.total_80d_claimed == 1850.0

    # House rent payment (₹22,000) under HRA
    assert tax_rep.total_hra_rent_paid == 22000.0

    # PM Cares donation (₹1,000 @ 50% eligible = ₹500)
    assert tax_rep.total_80g_donations == 500.0

    assert tax_rep.total_deductions_claimed == 15000.0 + 1850.0 + 500.0
    assert len(tax_rep.deduction_items) == 4


def test_compare_tax_regimes(sample_financial_df):
    """Verify Old vs New tax regime simulator correctly evaluates comparative tax liability."""
    tax_rep = detect_tax_deductions(sample_financial_df)
    comparison = compare_tax_regimes(annual_gross_income=1200000.0, tax_summary=tax_rep)

    assert comparison.annual_gross_income == 1200000.0
    assert comparison.new_regime_tax >= 0.0
    assert comparison.old_regime_tax >= 0.0
    assert comparison.recommended_regime in ["New Regime", "Old Regime", "Either (Identical Tax)"]
    assert comparison.tax_savings >= 0.0


def test_reconcile_inter_account_transfers_prevents_double_counting(sample_financial_df):
    """Verify internal credit card settlements and self-transfers are reconciled."""
    annotated_df, summary = reconcile_inter_account_transfers(sample_financial_df)

    # CRED payment (₹12,450) and Self Transfer (₹5,000) should be flagged
    assert summary.transfer_count == 2
    assert summary.total_transfers_reconciled == 12450.0 + 5000.0
    assert summary.double_counting_prevented == 17450.0

    # Check dataframe annotations (filter for the credit card payment, not salary credit)
    cred_row = annotated_df[annotated_df["raw_description"].str.startswith("CRED/")].iloc[0]
    assert cred_row["is_internal_transfer"] == True
    assert cred_row["transfer_type"] == "Credit Card Settlement"

    self_row = annotated_df[annotated_df["raw_description"].str.contains("SELF SAVINGS")].iloc[0]
    assert self_row["is_internal_transfer"] == True
    assert self_row["transfer_type"] == "Self Transfer"

    # Effective net outflows should exclude internal balance transfers
    assert summary.effective_net_outflows == summary.total_raw_debits - 17450.0


def test_analyze_50_30_20_budget(sample_financial_df):
    """Verify 50/30/20 allocation buckets and mid-month burn rate pacing."""
    budget_rep = analyze_50_30_20_budget(sample_financial_df)

    assert "Needs (50%)" in budget_rep.buckets
    assert "Wants (30%)" in budget_rep.buckets
    assert "Savings (20%)" in budget_rep.buckets

    assert budget_rep.total_inflow == 95000.0
    assert budget_rep.actual_savings_rate_pct > 0.0

    # Burn forecast checks
    assert budget_rep.wants_forecast.bucket_name == "Wants"
    assert budget_rep.wants_forecast.daily_burn_rate >= 0.0
    assert budget_rep.wants_forecast.projected_month_end_spend >= budget_rep.wants_forecast.spent_to_date


def test_calculate_financial_health_and_runway(sample_financial_df):
    """Verify composite 0-100 Vitality score and emergency fund runway calculation."""
    health_rep = calculate_financial_health(sample_financial_df, override_liquid_balance=150000.0)

    assert 0 <= health_rep.vitality_score <= 100
    assert health_rep.tier in ["Elite / Fortress", "Healthy / Resilient", "Vulnerable / Attention Needed", "Critical / High Risk"]
    assert len(health_rep.components) == 4

    # Liquid runway
    assert health_rep.liquid_reserves == 150000.0
    assert health_rep.emergency_runway_months > 0.0
    assert health_rep.monthly_essential_burn > 0.0


def test_detect_money_leaks(sample_financial_df):
    """Verify detection of delivery app duplication and multi-service leaks."""
    leak_rep = detect_money_leaks(sample_financial_df)

    assert leak_rep.active_leak_count >= 1
    # Sample has both Swiggy and Zomato
    leak_types = [a.leak_type for a in leak_rep.alerts]
    assert "Delivery Duplication" in leak_types or "Redundant Subscriptions" in leak_types
    assert leak_rep.total_annual_leak_estimate > 0.0
