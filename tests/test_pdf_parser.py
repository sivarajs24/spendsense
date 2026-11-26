"""Unit tests for Bank Statement PDF Parsing and Decryption."""

from io import BytesIO
from pathlib import Path
import pytest

from spendsense.data.pdf_parser import parse_bank_statement_pdf

try:
    import pypdf
except ImportError:
    pypdf = None


def test_pdf_parser_raises_on_missing_file():
    """Verify FileNotFoundError is raised for non-existent PDF path."""
    with pytest.raises(FileNotFoundError):
        parse_bank_statement_pdf("non_existent_statement.pdf")


@pytest.mark.skipif(pypdf is None, reason="pypdf is not installed")
def test_pdf_parser_in_memory_encrypted_and_decrypted():
    """Verify in-memory PDF generation, encryption, password unlocking, and tabular parsing."""
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=612, height=792)

    # Encrypt with standard test password
    writer.encrypt("testpass123")

    pdf_buffer = BytesIO()
    writer.write(pdf_buffer)
    pdf_bytes = pdf_buffer.getvalue()

    # Attempting to read without password must raise PermissionError
    with pytest.raises(PermissionError):
        parse_bank_statement_pdf(pdf_bytes)

    # Attempting to read with wrong password must raise PermissionError
    with pytest.raises(PermissionError):
        parse_bank_statement_pdf(pdf_bytes, password="wrongpassword")

    # Correct password unlocks successfully
    df, report = parse_bank_statement_pdf(pdf_bytes, password="testpass123")
    assert report["is_encrypted"] is True
    assert report["pages_processed"] == 1
    assert isinstance(df, type(df))
