"""Pydantic request and response schemas for the SpendSense FastAPI service."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    model_loaded: bool
    version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    version: str
    created_at: Optional[str]
    confidence_threshold: float
    categories: List[str]
    macro_f1: Optional[float] = None
    accuracy: Optional[float] = None


class PredictRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=500, description="Raw transaction narration or description.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Optional override for review queue confidence threshold.")


class AlternativeCategory(BaseModel):
    category: str
    confidence: float


class PredictResponse(BaseModel):
    description: str
    normalized_description: str
    predicted_category: str
    confidence: float
    requires_review: bool
    review_reason: Optional[str] = None
    alternatives: List[AlternativeCategory] = Field(default_factory=list)
    model_version: str


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest] = Field(..., min_length=1, max_length=1000, description="List of transaction items to categorize.")
    threshold: Optional[float] = Field(None, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    total_processed: int
    requires_review_count: int
    predictions: List[PredictResponse]


class FeedbackRequest(BaseModel):
    transaction_id: str = Field(..., min_length=1, description="Unique internal identifier of the transaction.")
    normalized_description: str = Field(..., min_length=1, description="Normalized description.")
    original_category: str = Field(..., min_length=1, description="Model's original prediction.")
    corrected_category: str = Field(..., min_length=1, description="User's corrected canonical category.")


class FeedbackResponse(BaseModel):
    success: bool
    message: str
    transaction_id: str


# --- Advanced Analytics Schemas ---

class TransactionItem(BaseModel):
    """Normalized transaction row for analytical evaluations."""
    transaction_id: str = Field(..., description="Unique transaction identifier.")
    date: Optional[str] = Field(None, description="Transaction date (YYYY-MM-DD or standard bank format).")
    raw_description: str = Field(..., description="Raw transaction narration or description.")
    amount: float = Field(..., description="Transaction amount (negative for debits, positive for credits).")
    predicted_category: Optional[str] = Field(None, description="Assigned category, if already classified.")


class TaxAnalyticsRequest(BaseModel):
    """Input payload for tax deductions and regime simulation."""
    transactions: List[TransactionItem] = Field(..., min_length=1)
    annual_gross_income: float = Field(1200000.0, ge=0.0, description="Annual gross income for regime comparison.")
    is_senior_citizen: bool = Field(False, description="Whether the taxpayer is a senior citizen (60+).")


class TransferReconciliationRequest(BaseModel):
    """Input payload for inter-account transfer reconciliation."""
    transactions: List[TransactionItem] = Field(..., min_length=1)


class BudgetAnalyticsRequest(BaseModel):
    """Input payload for 50/30/20 budgeting and burn rate pacing."""
    transactions: List[TransactionItem] = Field(..., min_length=1)
    override_inflow: Optional[float] = Field(None, ge=0.0, description="Optional explicit monthly income.")


class HealthAnalyticsRequest(BaseModel):
    """Input payload for financial vitality and emergency runway scoring."""
    transactions: List[TransactionItem] = Field(..., min_length=1)
    override_liquid_balance: Optional[float] = Field(None, ge=0.0, description="Optional liquid cash reserve.")


class LeakAnalyticsRequest(BaseModel):
    """Input payload for money leak and redundant subscription detection."""
    transactions: List[TransactionItem] = Field(..., min_length=1)

