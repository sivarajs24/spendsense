"""Data validation and transaction schema enforcement.

Validates raw tabular statements and produces strongly-typed normalized records.
"""

from datetime import date, datetime
import hashlib
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field

from spendsense.data.preprocess import extract_merchant_proxy, normalize_description


class NormalizedTransaction(BaseModel):
    """Internal canonical schema for every transaction in SpendSense."""
    transaction_id: str
    date: Optional[str] = None  # ISO format: YYYY-MM-DD
    raw_description: str
    normalized_description: str
    merchant_group: str
    amount: float
    direction: str = Field(description="'debit', 'credit', or 'unknown'")
    currency: str = "INR"
    requires_review: bool = False
    true_category: Optional[str] = None


def generate_transaction_id(date_str: Optional[str], raw_desc: str, amount: float, idx: int) -> str:
    """Generates a stable, reproducible pseudo-hash for internal tracking."""
    payload = f"{date_str or 'no-date'}|{raw_desc.strip().lower()}|{amount:.2f}|{idx}"
    return f"txn_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12]}"


def parse_flexible_date(date_val: Any) -> Optional[str]:
    """Attempts to parse common date formats into ISO YYYY-MM-DD string."""
    if pd.isna(date_val) or date_val is None:
        return None
    if isinstance(date_val, (datetime, date)):
        return date_val.strftime("%Y-%m-%d")

    s = str(date_val).strip()
    if not s:
        return None

    formats = [
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%m/%d/%Y",
        "%d/%m/%y",
        "%d-%m-%y",
        "%d-%b-%Y",
        "%d %b %Y",
        "%Y/%m/%d",
    ]
    for fmt in formats:
        try:
            parsed = datetime.strptime(s, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Fallback to pandas parser
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.notna(ts):
            return ts.strftime("%Y-%m-%d")
    except Exception:
        pass
    return None


def clean_numeric_amount(val: Any) -> float:
    """Converts dirty strings like 'Rs. 1,499.50 CR' or '(500.00)' into signed floats."""
    if pd.isna(val) or val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)

    s = str(val).strip().replace(",", "")
    # Remove currency symbols or non-numeric prefixes
    is_negative = False
    if s.startswith("(") and s.endswith(")"):
        is_negative = True
        s = s[1:-1]
    if "-" in s:
        is_negative = True
        s = s.replace("-", "")

    # Clean out any alphabets like CR, DR, INR
    cleaned = "".join(c for c in s if c.isdigit() or c == ".")
    try:
        amt = float(cleaned) if cleaned else 0.0
        return -amt if is_negative else amt
    except ValueError:
        return 0.0


def validate_raw_dataframe(
    df: pd.DataFrame,
    desc_col: str,
    amount_col: Optional[str] = None,
    debit_col: Optional[str] = None,
    credit_col: Optional[str] = None,
    date_col: Optional[str] = None,
    category_col: Optional[str] = None,
    currency: str = "INR",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Validates and converts an imported statement into the standard SpendSense format.

    Returns:
        (normalized_df, quality_report_dict)
    """
    total_raw_rows = len(df)
    if desc_col not in df.columns:
        raise ValueError(f"Required description column '{desc_col}' not found in dataframe.")

    records: List[Dict[str, Any]] = []
    dropped_empty_descriptions = 0

    for idx, row in df.iterrows():
        raw_desc = str(row[desc_col]) if pd.notna(row[desc_col]) else ""
        if not raw_desc.strip():
            dropped_empty_descriptions += 1
            continue

        norm_desc = normalize_description(raw_desc)
        merchant_group = extract_merchant_proxy(norm_desc)

        # Date parsing
        date_str = parse_flexible_date(row[date_col]) if date_col and date_col in df.columns else None

        # Amount determination: Either separate Debit/Credit cols or single Amount col
        final_amount = 0.0
        direction = "unknown"

        if debit_col and credit_col and debit_col in df.columns and credit_col in df.columns:
            debit_val = clean_numeric_amount(row[debit_col])
            credit_val = clean_numeric_amount(row[credit_col])
            if debit_val > 0:
                final_amount = -abs(debit_val)
                direction = "debit"
            elif credit_val > 0:
                final_amount = abs(credit_val)
                direction = "credit"
            else:
                final_amount = 0.0
                direction = "unknown"
        elif amount_col and amount_col in df.columns:
            raw_amt = clean_numeric_amount(row[amount_col])
            final_amount = raw_amt
            if raw_amt < 0:
                direction = "debit"
            elif raw_amt > 0:
                direction = "credit"
            else:
                direction = "unknown"

        true_cat = str(row[category_col]).strip() if category_col and category_col in df.columns and pd.notna(row[category_col]) else None

        txn_id = generate_transaction_id(date_str, raw_desc, final_amount, int(idx))

        records.append({
            "transaction_id": txn_id,
            "date": date_str,
            "raw_description": raw_desc,
            "normalized_description": norm_desc,
            "merchant_group": merchant_group,
            "amount": final_amount,
            "direction": direction,
            "currency": currency,
            "requires_review": False,
            "true_category": true_cat,
        })

    normalized_df = pd.DataFrame(records)

    quality_report = {
        "total_raw_rows": total_raw_rows,
        "valid_rows": len(normalized_df),
        "dropped_empty_descriptions": dropped_empty_descriptions,
        "null_dates": int(normalized_df["date"].isna().sum()) if not normalized_df.empty else 0,
        "debit_count": int((normalized_df["direction"] == "debit").sum()) if not normalized_df.empty else 0,
        "credit_count": int((normalized_df["direction"] == "credit").sum()) if not normalized_df.empty else 0,
        "unique_merchant_groups": int(normalized_df["merchant_group"].nunique()) if not normalized_df.empty else 0,
    }

    return normalized_df, quality_report
