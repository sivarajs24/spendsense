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
