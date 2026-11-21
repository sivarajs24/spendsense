"""CSV loading and column inference utilities for heterogeneous bank statements."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from spendsense.data.validate import validate_raw_dataframe

# Common column name heuristics across banking exports (HDFC, SBI, ICICI, Mendeley, etc.)
CANDIDATE_DATE_COLS = ["date", "txn_date", "transaction_date", "value_date", "posting_date"]
CANDIDATE_DESC_COLS = [
    "description", "transaction_description", "narration", "particulars",
    "merchant", "details", "transaction_details", "memo"
]
CANDIDATE_AMOUNT_COLS = ["amount", "txn_amount", "transaction_amount", "net_amount", "balance_amt"]
CANDIDATE_DEBIT_COLS = ["debit", "withdrawal", "dr", "debit_amount", "withdrawal_amount"]
CANDIDATE_CREDIT_COLS = ["credit", "deposit", "cr", "credit_amount", "deposit_amount"]
CANDIDATE_CATEGORY_COLS = ["category", "target", "label", "true_category"]


def infer_statement_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Infers standard column roles by inspecting normalized column names."""
    col_map = {c.strip().lower().replace(" ", "_"): c for c in df.columns}

    def _match(candidates: list[str]) -> Optional[str]:
        for candidate in candidates:
            for lower_name, original_name in col_map.items():
                if candidate == lower_name or candidate in lower_name:
                    return original_name
        return None

    inferred = {
        "desc_col": _match(CANDIDATE_DESC_COLS),
        "date_col": _match(CANDIDATE_DATE_COLS),
        "amount_col": _match(CANDIDATE_AMOUNT_COLS),
        "debit_col": _match(CANDIDATE_DEBIT_COLS),
        "credit_col": _match(CANDIDATE_CREDIT_COLS),
        "category_col": _match(CANDIDATE_CATEGORY_COLS),
    }

    # If separate debit and credit columns exist, prefer them over a single ambiguous amount column
    if inferred["debit_col"] and inferred["credit_col"]:
        inferred["amount_col"] = None

    return inferred


def load_statement_csv(
    file_path: Union[str, Path],
    column_mapping: Optional[Dict[str, Optional[str]]] = None,
    currency: str = "INR",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Reads a CSV bank statement and normalizes it into SpendSense format.

    Args:
        file_path: Path to the CSV file.
        column_mapping: Optional explicit mapping overriding auto-detection.
        currency: Currency code (default 'INR').

    Returns:
        (normalized_df, quality_report)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Statement file does not exist: {path}")

    # Try common encodings
    df = None
    for encoding in ["utf-8", "utf-8-sig", "latin1", "cp1252"]:
        try:
            df = pd.read_csv(path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Unable to decode CSV file with standard encodings: {path}")

    mapping = infer_statement_columns(df)
    if column_mapping:
        mapping.update({k: v for k, v in column_mapping.items() if v is not None})

    desc_col = mapping.get("desc_col")
    if not desc_col or desc_col not in df.columns:
        raise ValueError(
            f"Could not automatically detect transaction description column. "
            f"Available columns: {list(df.columns)}"
        )

    return validate_raw_dataframe(
        df=df,
        desc_col=desc_col,
        amount_col=mapping.get("amount_col"),
        debit_col=mapping.get("debit_col"),
        credit_col=mapping.get("credit_col"),
        date_col=mapping.get("date_col"),
        category_col=mapping.get("category_col"),
        currency=currency,
    )
