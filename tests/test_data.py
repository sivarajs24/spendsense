"""Unit and integration tests for data loading, preprocessing, and splitting."""

from pathlib import Path
import pandas as pd
import pytest

from spendsense.data.load import infer_statement_columns, load_statement_csv
from spendsense.data.preprocess import (
    clean_text,
    extract_merchant_proxy,
    normalize_description,
    parse_upi_description,
)
from spendsense.data.split import grouped_merchant_split, train_val_test_split
from spendsense.data.validate import (
    clean_numeric_amount,
    parse_flexible_date,
    validate_raw_dataframe,
)


def test_clean_text_normalizes_unicode_and_spaces():
    raw = "  SWIGGY\u00a0\u00a0DELIVERY   SERVICES\t"
    cleaned = clean_text(raw)
    assert cleaned == "SWIGGY DELIVERY SERVICES"


def test_parse_upi_description_extracts_tokens():
    raw_upi = "UPI/4829102938/SWIGGY@ybl"
    merchant, vpa, ref_id = parse_upi_description(raw_upi)
    assert "swiggy" in merchant.lower()
    assert vpa == "swiggy@ybl"
    assert ref_id == "4829102938"


def test_normalize_description_strips_noise_and_card_numbers():
    raw = "POS 4012XXXXXXXX1234 STARBUCKS CAFE BANGALORE REF88991122"
    normalized = normalize_description(raw)
    assert "starbucks" in normalized
    assert "4012" not in normalized
    assert "REF88991122" not in normalized


def test_clean_numeric_amount_varieties():
    assert clean_numeric_amount("1,499.50") == 1499.50
    assert clean_numeric_amount("(500.00)") == -500.00
    assert clean_numeric_amount("-250.75") == -250.75
    assert clean_numeric_amount(None) == 0.0


def test_parse_flexible_date_varieties():
    assert parse_flexible_date("2026-01-15") == "2026-01-15"
    assert parse_flexible_date("15/01/2026") == "2026-01-15"
    assert parse_flexible_date("15-Jan-2026") == "2026-01-15"
    assert parse_flexible_date(None) is None


def test_load_statement_csv_with_sample_file():
    sample_file = Path("data/samples/sample_bank_statement.csv")
    df, report = load_statement_csv(sample_file)

    assert len(df) == 15
    assert report["valid_rows"] == 15
    assert report["debit_count"] > 0
    assert report["credit_count"] > 0
    assert "normalized_description" in df.columns
    assert "merchant_group" in df.columns

    # Verify Swiggy row is parsed as debit with Swiggy in description
    swiggy_row = df[df["raw_description"].str.contains("SWIGGY")].iloc[0]
    assert swiggy_row["direction"] == "debit"
    assert swiggy_row["amount"] == -450.00
    assert "swiggy" in swiggy_row["normalized_description"]


def test_stratified_train_val_test_split_ratios():
    data = {
        "raw_description": [f"item_{i}" for i in range(100)],
        "true_category": ["Food & Dining"] * 50 + ["Transportation"] * 30 + ["Income"] * 20,
        "merchant_group": [f"merchant_{i % 10}" for i in range(100)],
    }
    df = pd.DataFrame(data)

    train_df, val_df, test_df = train_val_test_split(
        df,
        target_col="true_category",
        train_size=0.70,
        val_size=0.15,
        test_size=0.15,
        random_state=42,
    )

    assert len(train_df) == 70
    assert len(val_df) == 15
    assert len(test_df) == 15

    # Check class proportions are preserved
    train_dist = train_df["true_category"].value_counts(normalize=True)
    test_dist = test_df["true_category"].value_counts(normalize=True)
    assert abs(train_dist["Food & Dining"] - test_dist["Food & Dining"]) < 0.05


def test_grouped_merchant_split_has_zero_leakage():
    data = {
        "raw_description": [f"desc_{i}" for i in range(100)],
        "true_category": ["Food & Dining"] * 50 + ["Transportation"] * 50,
        "merchant_group": [f"merchant_{i // 5}" for i in range(100)],  # 20 distinct merchants
    }
    df = pd.DataFrame(data)

    train_df, test_df = grouped_merchant_split(
        df,
        group_col="merchant_group",
        target_col="true_category",
        train_size=0.70,
        test_size=0.30,
        random_state=42,
    )

    train_merchants = set(train_df["merchant_group"])
    test_merchants = set(test_df["merchant_group"])

    overlap = train_merchants.intersection(test_merchants)
    assert len(overlap) == 0, f"Merchant overlap detected: {overlap}"
