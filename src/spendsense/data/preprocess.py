"""Text normalization and transaction description cleaning module.

Includes specialized parsers for Indian UPI, NEFT, IMPS, and card narration patterns.
"""

import re
import unicodedata
from typing import Optional

# Regex patterns for Indian banking narrations
UPI_PREFIX_PATTERN = re.compile(r"^(?:UPI|UPI-CR|UPI-DR|UPI/P2M|UPI/P2P|UPI/REV)[/-]", re.IGNORECASE)
UPI_VPA_PATTERN = re.compile(r"([a-zA-Z0-9.\-_]+@[a-zA-Z0-9]+)", re.IGNORECASE)
NEFT_RTGS_PATTERN = re.compile(r"^(?:NEFT|RTGS|IMPS|INB|INF|MOB|MMT)[/-](?:[a-zA-Z0-9]+[/-])?", re.IGNORECASE)
REF_NUMBER_PATTERN = re.compile(r"\b(?:REF|UTR|TXN|NO|ID|RRN)[:/\-]?\s*[a-zA-Z0-9]{6,25}\b", re.IGNORECASE)
MASKED_CARD_PATTERN = re.compile(r"\b\d{4}[X*]{4,8}\d{4}\b", re.IGNORECASE)
HEX_RANDOM_TOKEN_PATTERN = re.compile(r"\b[0-9a-fA-F]{10,}\b")
DIGIT_SEQUENCES_PATTERN = re.compile(r"\b\d{4,}\b")
MULTIPLE_SPACES_PATTERN = re.compile(r"\s+")
SPECIAL_CHARACTERS_PATTERN = re.compile(r"[^a-zA-Z0-9\s/&.-]")

# Common noise words found in bank statements
NOISE_TOKENS = {
    "pos", "ecom", "inr", "cr", "dr", "pur", "txn", "ref", "billdesk", "razorpay",
    "ccav", "payu", "ccavenue", "pg", "charges", "chg", "transfer", "trf", "via",
    "to", "by", "from", "at", "direct", "debit", "credit"
}


def clean_text(text: Optional[str]) -> str:
    """Performs baseline Unicode normalization and whitespace trimming."""
    if not text or not isinstance(text, str):
        return ""
    # Normalize unicode (NFKD decomposes ligatures and accented characters)
    normalized = unicodedata.normalize("NFKD", text)
    # Strip non-ASCII characters to keep token representations standard
    ascii_text = normalized.encode("ascii", "ignore").decode("utf-8")
    return MULTIPLE_SPACES_PATTERN.sub(" ", ascii_text).strip()


def parse_upi_description(text: str) -> tuple[str, Optional[str], Optional[str]]:
    """Extracts merchant tokens and VPA handles from Indian UPI strings.

    Examples:
        "UPI/4829123891/SWIGGY@ybl" -> ("swiggy", "swiggy@ybl", "4829123891")
        "UPI-CR-3120192837-ZOMATO-zomato@icici" -> ("zomato", "zomato@icici", "3120192837")
    """
    cleaned = clean_text(text)
    if not cleaned:
        return "", None, None

    vpa = None
    vpa_match = UPI_VPA_PATTERN.search(cleaned)
    if vpa_match:
        vpa = vpa_match.group(1).lower()

    # Split by standard delimiters used in UPI narrations: /, -, :
    tokens = re.split(r"[/:\-_]", cleaned)
    tokens = [t.strip().lower() for t in tokens if t.strip()]

    # Filter out prefixes, purely numerical IDs, noise tokens, and raw VPAs
    candidate_merchants = []
    ref_id = None

    for token in tokens:
        if token.isdigit() and len(token) >= 6:
            ref_id = token
            continue
        if token in {"upi", "cr", "dr", "p2m", "p2p", "rev", "paytm", "gpay", "phonepe"}:
            continue
        if "@" in token:
            # Extract handle prefix before @ if meaningful (e.g. "swiggy" from "swiggy@ybl")
            handle_name = token.split("@")[0]
            if len(handle_name) >= 3 and not handle_name.isdigit():
                candidate_merchants.append(handle_name)
            continue
        if len(token) >= 3 and not token.isdigit():
            candidate_merchants.append(token)

    merchant = " ".join(candidate_merchants) if candidate_merchants else cleaned
    return merchant, vpa, ref_id


def extract_merchant_proxy(normalized_text: str) -> str:
    """Extracts a high-level merchant group token for unseen-merchant evaluation splits."""
    tokens = normalized_text.split()
    if not tokens:
        return "unknown_merchant"
    # Filter out residual noise words to isolate root merchant name
    meaningful = [t for t in tokens if t not in NOISE_TOKENS and len(t) >= 3]
    return meaningful[0] if meaningful else tokens[0]


def normalize_description(text: Optional[str]) -> str:
    """End-to-end normalization pipeline for any transaction description.

    Steps:
    1. Unicode & whitespace normalization.
    2. Lowercase transformation.
    3. Removal of masked cards (e.g., 4012XXXXXXXX1234).
    4. Removal of transaction/reference numbers (e.g., REF129384758).
    5. Extraction of clean tokens from UPI / NEFT prefixes.
    6. Stripping of isolated digit sequences (terminal IDs, order numbers).
    7. Removal of special punctuation while preserving word boundaries.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return ""

    lowered = cleaned.lower()

    # If it's a UPI transaction, perform specialized UPI token extraction first
    if UPI_PREFIX_PATTERN.search(lowered) or "@" in lowered:
        merchant, vpa, _ = parse_upi_description(lowered)
        lowered = merchant

    # Strip masked credit/debit card numbers
    lowered = MASKED_CARD_PATTERN.sub(" ", lowered)

    # Strip banking prefixes (NEFT, RTGS, IMPS)
    lowered = NEFT_RTGS_PATTERN.sub(" ", lowered)

    # Strip reference IDs & UTR numbers
    lowered = REF_NUMBER_PATTERN.sub(" ", lowered)

    # Strip high-entropy hex sequences and random tokens
    lowered = HEX_RANDOM_TOKEN_PATTERN.sub(" ", lowered)

    # Strip standalone long numeric sequences (order IDs, store branch numbers)
    lowered = DIGIT_SEQUENCES_PATTERN.sub(" ", lowered)

    # Clean remaining unwanted punctuation
    lowered = SPECIAL_CHARACTERS_PATTERN.sub(" ", lowered)

    # Consolidate whitespace
    tokens = lowered.split()
    # Filter out single-character noise and empty tokens
    meaningful_tokens = [t for t in tokens if len(t) > 1 or t.isalpha()]

    return " ".join(meaningful_tokens).strip()
