"""Inter-Account Transfer Reconciliation Module.

Detects internal settlements (Credit Card Bill Payments, Self-Transfers, Wallet Loads)
and removes them from net expense tallies to eliminate financial double-counting.
"""

import re
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd
from pydantic import BaseModel, Field


class ReconciledTransfer(BaseModel):
    """Details of a single reconciled internal movement."""
    transaction_id: str
    date: Optional[str] = None
    description: str
    amount: float
    transfer_type: str  # "Credit Card Settlement", "Self Transfer", "Wallet Load"
    explanation: str


class TransferReconciliationSummary(BaseModel):
    """Aggregated reconciliation metrics distinguishing net spend from balance movements."""
    total_raw_debits: float
    total_transfers_reconciled: float
    effective_net_outflows: float
    double_counting_prevented: float
    transfer_count: int
    transfers: List[ReconciledTransfer] = Field(default_factory=list)


# Patterns identifying inter-account transfers & credit card repayments
RE_CC_BILL = re.compile(
    r"(cred/|credit\s*card\s*bill|cc\s*payment|card\s*payment|billdesk.*credit|"
    r"sbi\s*cards?|hdfc\s*cards?|icici\s*cards?|axis\s*cards?|onecard)",
    re.IGNORECASE,
)

RE_SELF_TRANSFER = re.compile(
    r"(transfer\s*to\s*a/?c|transfer\s*to\s*own|self\s*transfer|sweep\s*transfer|"
    r"linked\s*account|personal\s*transfer|funds\s*transfer\s*to\s*self)",
    re.IGNORECASE,
)

RE_WALLET_LOAD = re.compile(
    r"(wallet\s*topup|wallet\s*load|paytm\s*wallet|amazon\s*pay\s*balance|mobikwik\s*load)",
    re.IGNORECASE,
)


def reconcile_inter_account_transfers(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, TransferReconciliationSummary]:
    """Identifies and flags internal balance movements to prevent double-counting.

    Args:
        df: DataFrame containing transaction records.

    Returns:
        (annotated_df, summary_report)
    """
    if df.empty or "amount" not in df.columns:
        empty_summary = TransferReconciliationSummary(
            total_raw_debits=0.0,
            total_transfers_reconciled=0.0,
            effective_net_outflows=0.0,
            double_counting_prevented=0.0,
            transfer_count=0,
            transfers=[],
        )
        return df.copy(), empty_summary

    out_df = df.copy()
    is_transfer_flags = [False] * len(out_df)
    transfer_type_flags = [None] * len(out_df)

    reconciled_items: List[ReconciledTransfer] = []
    total_debits = float(out_df[out_df["amount"] < 0]["amount"].abs().sum())
    transfers_sum = 0.0

    for pos, (idx, row) in enumerate(out_df.iterrows()):
        amt = float(row.get("amount", 0.0))
        # Internal balance transfers are debit outflows from primary checking accounts
        if amt >= 0:
            continue

        raw_desc = str(row.get("raw_description", ""))
        norm_desc = str(row.get("normalized_description", ""))
        text = f"{raw_desc} {norm_desc}".lower()
        t_id = str(row.get("transaction_id", f"txn_{idx}"))
        t_date = str(row.get("date", "")) if pd.notna(row.get("date")) else None
        abs_amt = abs(amt)

        t_type = None
        explanation = None

        if RE_CC_BILL.search(text):
            t_type = "Credit Card Settlement"
            explanation = "Credit card repayment. Subsumes individual card purchases; excluded to prevent 2x double-counting."
        elif RE_SELF_TRANSFER.search(text):
            t_type = "Self Transfer"
            explanation = "Internal transfer between own secondary/savings accounts."
        elif RE_WALLET_LOAD.search(text):
            t_type = "Wallet Load"
            explanation = "Prepaid wallet load. Actual expense will occur at point of merchant spend."

        if t_type:
            is_transfer_flags[pos] = True
            transfer_type_flags[pos] = t_type
            transfers_sum += abs_amt
            reconciled_items.append(
                ReconciledTransfer(
                    transaction_id=t_id,
                    date=t_date,
                    description=raw_desc,
                    amount=abs_amt,
                    transfer_type=t_type,
                    explanation=explanation,
                )
            )

    out_df["is_internal_transfer"] = is_transfer_flags
    out_df["transfer_type"] = transfer_type_flags
    effective_net = max(0.0, total_debits - transfers_sum)

    summary = TransferReconciliationSummary(
        total_raw_debits=round(total_debits, 2),
        total_transfers_reconciled=round(transfers_sum, 2),
        effective_net_outflows=round(effective_net, 2),
        double_counting_prevented=round(transfers_sum, 2),
        transfer_count=len(reconciled_items),
        transfers=reconciled_items,
    )

    return out_df, summary
