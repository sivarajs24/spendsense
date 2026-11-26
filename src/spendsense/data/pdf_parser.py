"""Bank PDF Statement Ingestion and Password Unlock Parser.

Extracts tabular transactions from encrypted or plaintext bank e-statements (HDFC, SBI, ICICI, Axis)
using local in-memory decryption without storing credentials on disk.
"""

from io import BytesIO
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd

from spendsense.data.preprocess import normalize_description
from spendsense.data.validate import clean_numeric_amount, parse_flexible_date

try:
    import pypdf
except ImportError:
    pypdf = None


# Regex matching common bank statement transaction rows
# Pattern matches: Date (DD-MM-YYYY or DD/MM/YYYY or YYYY-MM-DD or DD-Mon-YYYY)
DATE_PREFIX_PATTERN = re.compile(
    r"^(\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4}|\d{4}[-/\.]\d{1,2}[-/\.]\d{1,2}|\d{1,2}[-\s][A-Za-z]{3}[-\s]\d{2,4})\b"
)

# Numeric amount pattern (e.g. 1,450.00 or 500.50 or 12450)
AMOUNT_PATTERN = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d{2})\b")


def parse_bank_statement_pdf(
    pdf_source: Union[str, Path, bytes, BytesIO],
    password: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Decrypts and parses a bank statement PDF into canonical transaction schema.

    Args:
        pdf_source: File path, raw bytes, or BytesIO buffer containing the PDF statement.
        password: Optional decryption password (e.g. DOB DDMMYYYY or PAN for encrypted e-statements).

    Returns:
        (parsed_df, report_dict)
    """
    if pypdf is None:
        raise ImportError(
            "PDF statement parsing requires 'pypdf'. Please install it via: pip install pypdf"
        )

    # Load reader
    if isinstance(pdf_source, (str, Path)):
        file_path = Path(pdf_source)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF statement not found: {file_path}")
        reader = pypdf.PdfReader(str(file_path))
    elif isinstance(pdf_source, bytes):
        reader = pypdf.PdfReader(BytesIO(pdf_source))
    elif isinstance(pdf_source, BytesIO):
        reader = pypdf.PdfReader(pdf_source)
    else:
        raise ValueError(f"Unsupported PDF input type: {type(pdf_source)}")

    # Check Encryption & Unlock
    if reader.is_encrypted:
        if not password:
            raise PermissionError(
                "This bank statement is password-protected. Please provide your statement password (e.g. DOB or PAN)."
            )
        decrypt_success = reader.decrypt(password)
        # In pypdf, decrypt returns 0/PasswordType.NOT_DECRYPTED if password is wrong
        if not decrypt_success:
            raise PermissionError("Incorrect statement password. Could not decrypt PDF.")

    extracted_lines: List[str] = []
    num_pages = len(reader.pages)

    for page_idx in range(num_pages):
        page = reader.pages[page_idx]
        text = page.extract_text()
        if text:
            extracted_lines.extend(text.splitlines())

    parsed_rows: List[Dict[str, Any]] = []

    # Parse lines identifying transaction rows
    for line in extracted_lines:
        line_clean = line.strip()
        if not line_clean:
            continue

        date_match = DATE_PREFIX_PATTERN.match(line_clean)
        if not date_match:
            continue

        raw_date = date_match.group(1)
        remainder = line_clean[date_match.end():].strip()

        # Find amounts in remainder (Withdrawal, Deposit, Balance)
        amounts = AMOUNT_PATTERN.findall(remainder)
        if not amounts:
            continue

        # Extract amounts from the tail of the line
        # Typical bank layout: Date | Description ... | Withdrawal | Deposit | Balance
        amounts_numeric = [clean_numeric_amount(a) for a in amounts]

        # Isolate description text by stripping trailing numeric tokens
        desc_text = remainder
        for amt_str in amounts:
            idx = desc_text.rfind(amt_str)
            if idx != -1:
                desc_text = desc_text[:idx].strip()

        desc_clean = re.sub(r"\s+", " ", desc_text).strip()
        if not desc_clean:
            desc_clean = "Bank Transaction"

        amount_val = 0.0
        direction = "debit"

        if len(amounts_numeric) == 1:
            # Single amount: treat as debit unless 'CR' in line
            amt = amounts_numeric[0]
            if " CR" in remainder.upper() or "CREDIT" in remainder.upper():
                amount_val = amt
                direction = "credit"
            else:
                amount_val = -amt
                direction = "debit"
        elif len(amounts_numeric) >= 2:
            # Two or three amounts: [Withdrawal, Deposit] or [Debit/Credit, Balance]
            amt1, amt2 = amounts_numeric[0], amounts_numeric[1]
            if amt1 > 0 and amt2 == 0:
                amount_val = -amt1
                direction = "debit"
            elif amt1 == 0 and amt2 > 0:
                amount_val = amt2
                direction = "credit"
            elif "CR" in remainder.upper():
                amount_val = amt1
                direction = "credit"
            else:
                amount_val = -amt1
                direction = "debit"

        clean_date = parse_flexible_date(raw_date)

        parsed_rows.append({
            "transaction_id": f"pdf_{len(parsed_rows) + 1}",
            "date": clean_date,
            "raw_description": desc_clean,
            "normalized_description": normalize_description(desc_clean),
            "amount": float(amount_val),
            "direction": direction,
            "balance": amounts_numeric[-1] if len(amounts_numeric) >= 3 else None,
        })

    df_out = pd.DataFrame(parsed_rows)
    report = {
        "pages_processed": num_pages,
        "is_encrypted": reader.is_encrypted,
        "valid_rows": len(df_out),
        "debits_found": int((df_out["amount"] < 0).sum()) if not df_out.empty else 0,
        "credits_found": int((df_out["amount"] > 0).sum()) if not df_out.empty else 0,
    }

    return df_out, report
