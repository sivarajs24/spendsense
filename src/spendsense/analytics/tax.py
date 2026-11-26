"""Indian Tax Intelligence and Deductions Engine (Section 80C, 80D, HRA).

Analyzes financial transactions for tax-deductible expenditures and simulates
Old vs. New Tax Regime comparisons (FY 2024-25 / FY 2025-26 Indian Tax Code).
"""

import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
from pydantic import BaseModel, Field

TaxCalculationResult = Tuple[float, float, Dict[str, Any]]



class TaxDeductionItem(BaseModel):
    """Single tax-deductible expenditure detected from statements."""
    transaction_id: str
    date: Optional[str] = None
    description: str
    amount: float
    section: str  # "80C", "80D", "HRA", "Donations (80G)"
    deductible_amount: float
    explanation: str


class TaxDeductionSummary(BaseModel):
    """Aggregated tax deduction report across statutory sections."""
    total_80c_eligible: float
    total_80c_claimed: float  # Capped at ₹1,50,000
    total_80d_eligible: float
    total_80d_claimed: float  # Capped at ₹25,000 (or ₹50,000 for seniors)
    total_hra_rent_paid: float
    total_80g_donations: float
    total_deductions_claimed: float
    estimated_tax_savings_30pct: float
    deduction_items: List[TaxDeductionItem] = Field(default_factory=list)


class TaxRegimeComparison(BaseModel):
    """Comparative simulation of Old vs. New Tax Regimes."""
    annual_gross_income: float
    old_regime_tax: float
    new_regime_tax: float
    recommended_regime: str  # "New Regime" or "Old Regime"
    tax_savings: float
    old_regime_breakdown: Dict[str, Any]
    new_regime_breakdown: Dict[str, Any]


# Regular expression patterns for Indian statutory tax sections
PATTERNS_80C = re.compile(
    r"(zerodha|groww|kuvera|upstox|elss|mutual\s*fund|sip|ppf|provident\s*fund|"
    r"life\s*insurance|lic|hdfc\s*life|icici\s*pru|sbi\s*life|max\s*life|nps|pension)",
    re.IGNORECASE,
)

PATTERNS_80D = re.compile(
    r"(apollo\s*pharmacy|pharmeasy|netmeds|tata\s*1mg|max\s*healthcare|fortis|"
    r"manipal|apollo\s*hosp|health\s*insurance|mediclaim|care\s*health|star\s*health|"
    r"niva\s*bupa|diagnostic|dr\s*lal\s*path)",
    re.IGNORECASE,
)

PATTERNS_HRA = re.compile(
    r"(rent|house\s*rent|landlord|nobroker|nestaway|housing|society\s*maintenance)",
    re.IGNORECASE,
)

PATTERNS_80G = re.compile(
    r"(pm\s*cares|cry\s*child|unicef|prime\s*minister\s*relief|chief\s*minister\s*relief|"
    r"akshaya\s*patra|give\s*india|donation|charity)",
    re.IGNORECASE,
)


def detect_tax_deductions(
    df: pd.DataFrame,
    senior_citizen: bool = False,
    is_senior_citizen: Optional[bool] = None,
) -> TaxDeductionSummary:
    """Scans categorized bank transactions for tax-deductible expenditures.

    Args:
        df: DataFrame containing 'amount', 'raw_description', 'normalized_description'.
        senior_citizen: Whether the taxpayer is $\ge 60$ years (Section 80D limit ₹50k vs ₹25k).
        is_senior_citizen: Alias for senior_citizen.

    Returns:
        Structured TaxDeductionSummary with statutory caps applied.
    """
    if is_senior_citizen is not None:
        senior_citizen = is_senior_citizen
    if df.empty or "amount" not in df.columns:
        return TaxDeductionSummary(
            total_80c_eligible=0.0,
            total_80c_claimed=0.0,
            total_80d_eligible=0.0,
            total_80d_claimed=0.0,
            total_hra_rent_paid=0.0,
            total_80g_donations=0.0,
            total_deductions_claimed=0.0,
            estimated_tax_savings_30pct=0.0,
            deduction_items=[],
        )

    # Focus on debit outflows
    debits = df[df["amount"] < 0].copy()
    items: List[TaxDeductionItem] = []

    sum_80c = 0.0
    sum_80d = 0.0
    sum_hra = 0.0
    sum_80g = 0.0

    for idx, row in debits.iterrows():
        raw_desc = str(row.get("raw_description", ""))
        norm_desc = str(row.get("normalized_description", ""))
        text_comb = f"{raw_desc} {norm_desc}".lower()
        amt = abs(float(row.get("amount", 0.0)))
        t_id = str(row.get("transaction_id", f"txn_{idx}"))
        t_date = str(row.get("date", "")) if pd.notna(row.get("date")) else None

        # Check Section 80C
        if PATTERNS_80C.search(text_comb):
            items.append(
                TaxDeductionItem(
                    transaction_id=t_id,
                    date=t_date,
                    description=raw_desc,
                    amount=amt,
                    section="80C",
                    deductible_amount=amt,
                    explanation="Investment in ELSS / Mutual Fund SIP / PPF / Life Insurance",
                )
            )
            sum_80c += amt

        # Check Section 80D (Health)
        elif PATTERNS_80D.search(text_comb):
            items.append(
                TaxDeductionItem(
                    transaction_id=t_id,
                    date=t_date,
                    description=raw_desc,
                    amount=amt,
                    section="80D",
                    deductible_amount=amt,
                    explanation="Health insurance premium or approved pharmacy/diagnostic expense",
                )
            )
            sum_80d += amt

        # Check HRA Rent
        elif PATTERNS_HRA.search(text_comb) and amt >= 3000.0:
            pan_note = " (PAN declaration recommended)" if amt > 8333.0 else ""
            items.append(
                TaxDeductionItem(
                    transaction_id=t_id,
                    date=t_date,
                    description=raw_desc,
                    amount=amt,
                    section="HRA",
                    deductible_amount=amt,
                    explanation=f"House Rent Payment{pan_note}",
                )
            )
            sum_hra += amt

        # Check Section 80G (Charity)
        elif PATTERNS_80G.search(text_comb):
            items.append(
                TaxDeductionItem(
                    transaction_id=t_id,
                    date=t_date,
                    description=raw_desc,
                    amount=amt,
                    section="Donations (80G)",
                    deductible_amount=amt * 0.50,  # Typically 50% deduction
                    explanation="Charitable trust or relief fund contribution (50% eligible)",
                )
            )
            sum_80g += amt * 0.50

    cap_80c = 150000.0
    cap_80d = 50000.0 if senior_citizen else 25000.0

    claimed_80c = min(sum_80c, cap_80c)
    claimed_80d = min(sum_80d, cap_80d)
    total_claimed = claimed_80c + claimed_80d + sum_80g

    return TaxDeductionSummary(
        total_80c_eligible=round(sum_80c, 2),
        total_80c_claimed=round(claimed_80c, 2),
        total_80d_eligible=round(sum_80d, 2),
        total_80d_claimed=round(claimed_80d, 2),
        total_hra_rent_paid=round(sum_hra, 2),
        total_80g_donations=round(sum_80g, 2),
        total_deductions_claimed=round(total_claimed, 2),
        estimated_tax_savings_30pct=round(total_claimed * 0.312, 2),  # 30% + 4% cess
        deduction_items=items,
    )


def calculate_tax_new_regime(gross_income: float) -> TaxCalculationResult:
    """Calculates tax under New Tax Regime (Section 115BAC, FY 2024-25 / 2025-26).

    Standard Deduction: ₹75,000
    Slabs:
      0 to 3L: 0%
      3L to 7L: 5%
      7L to 10L: 10%
      10L to 12L: 15%
      12L to 15L: 20%
      Above 15L: 30%
    Section 87A: Full rebate if taxable income <= ₹7,00,000.
    """
    std_deduction = 75000.0
    taxable_income = max(0.0, gross_income - std_deduction)

    if taxable_income <= 700000.0:
        return 0.0, taxable_income, {"std_deduction": std_deduction, "rebate_87a": True}

    tax = 0.0
    # 3L to 7L @ 5%
    if taxable_income > 300000.0:
        tax += min(taxable_income - 300000.0, 400000.0) * 0.05
    # 7L to 10L @ 10%
    if taxable_income > 700000.0:
        tax += min(taxable_income - 700000.0, 300000.0) * 0.10
    # 10L to 12L @ 15%
    if taxable_income > 1000000.0:
        tax += min(taxable_income - 1000000.0, 200000.0) * 0.15
    # 12L to 15L @ 20%
    if taxable_income > 1200000.0:
        tax += min(taxable_income - 1200000.0, 300000.0) * 0.20
    # Above 15L @ 30%
    if taxable_income > 1500000.0:
        tax += (taxable_income - 1500000.0) * 0.30

    cess = tax * 0.04
    total_tax = round(tax + cess, 2)
    return total_tax, taxable_income, {"std_deduction": std_deduction, "cess": cess}


def calculate_tax_old_regime(
    gross_income: float,
    deductions: float,
) -> TaxCalculationResult:
    """Calculates tax under Old Tax Regime with chapter VI-A deductions.

    Standard Deduction: ₹50,000
    Slabs:
      0 to 2.5L: 0%
      2.5L to 5L: 5%
      5L to 10L: 20%
      Above 10L: 30%
    Section 87A: Full rebate if taxable income <= ₹5,00,000.
    """
    std_deduction = 50000.0
    total_exemptions = std_deduction + deductions
    taxable_income = max(0.0, gross_income - total_exemptions)

    if taxable_income <= 500000.0:
        return 0.0, taxable_income, {"exemptions": total_exemptions, "rebate_87a": True}

    tax = 0.0
    # 2.5L to 5L @ 5%
    if taxable_income > 250000.0:
        tax += min(taxable_income - 250000.0, 250000.0) * 0.05
    # 5L to 10L @ 20%
    if taxable_income > 500000.0:
        tax += min(taxable_income - 500000.0, 500000.0) * 0.20
    # Above 10L @ 30%
    if taxable_income > 1000000.0:
        tax += (taxable_income - 1000000.0) * 0.30

    cess = tax * 0.04
    total_tax = round(tax + cess, 2)
    return total_tax, taxable_income, {"exemptions": total_exemptions, "cess": cess}


def compare_tax_regimes(
    annual_gross_income: float,
    tax_summary: Optional[TaxDeductionSummary] = None,
    total_deductions_claimed: Optional[float] = None,
) -> TaxRegimeComparison:
    """Simulates Old vs New Tax Regimes to recommend the most cost-effective path."""
    if total_deductions_claimed is not None:
        claimed_deductions = total_deductions_claimed
    elif tax_summary is not None:
        claimed_deductions = tax_summary.total_deductions_claimed
    else:
        claimed_deductions = 0.0

    new_tax, new_taxable, new_breakdown = calculate_tax_new_regime(annual_gross_income)
    old_tax, old_taxable, old_breakdown = calculate_tax_old_regime(
        annual_gross_income, claimed_deductions
    )

    diff = abs(old_tax - new_tax)
    if new_tax < old_tax:
        recommendation = "New Regime"
    elif old_tax < new_tax:
        recommendation = "Old Regime"
    else:
        recommendation = "Either (Identical Tax)"

    return TaxRegimeComparison(
        annual_gross_income=annual_gross_income,
        old_regime_tax=old_tax,
        new_regime_tax=new_tax,
        recommended_regime=recommendation,
        tax_savings=round(diff, 2),
        old_regime_breakdown={**old_breakdown, "taxable_income": old_taxable},
        new_regime_breakdown={**new_breakdown, "taxable_income": new_taxable},
    )
