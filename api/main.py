"""FastAPI application for SpendSense inference and human feedback management."""

import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import (
    AlternativeCategory,
    BatchPredictRequest,
    BatchPredictResponse,
    BudgetAnalyticsRequest,
    FeedbackRequest,
    FeedbackResponse,
    HealthAnalyticsRequest,
    HealthResponse,
    LeakAnalyticsRequest,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    TaxAnalyticsRequest,
    TransferReconciliationRequest,
)
from spendsense.analytics.budget import BudgetReport, analyze_50_30_20_budget
from spendsense.analytics.health import FinancialHealthReport, calculate_financial_health
from spendsense.analytics.leaks import MoneyLeakReport, detect_money_leaks
from spendsense.analytics.tax import (
    TaxDeductionSummary,
    TaxRegimeComparison,
    compare_tax_regimes,
    detect_tax_deductions,
)
from spendsense.analytics.transfers import (
    TransferReconciliationSummary,
    reconcile_inter_account_transfers,
)
from spendsense.config import get_config
from spendsense.data.preprocess import normalize_description
from spendsense.feedback.store import FeedbackStore
from spendsense.models.registry import ModelRegistry

# Configure privacy-safe logging (no raw transaction amounts or narrations logged)
logger = logging.getLogger("spendsense.api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle manager."""
    logger.info("Initializing SpendSense API service...")
    try:
        ModelRegistry.load_active_model()
        logger.info("Active SpendSense ML model successfully loaded.")
    except Exception as e:
        logger.warning(f"Could not load active model on startup: {e}. Model will load on first request.")
    yield
    logger.info("Shutting down SpendSense API service...")


app = FastAPI(
    title="SpendSense API",
    description="Personal Expense Intelligence — Real-time transaction categorization and feedback learning.",
    version="0.1.0",
    lifespan=lifespan,
)

# Enable CORS for local dashboards and frontend clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["System"])
def health_check():
    """Returns application health and model loading status."""
    config, _ = get_config()
    is_loaded = False
    try:
        ModelRegistry.load_active_model()
        is_loaded = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy",
        model_loaded=is_loaded,
        version=config.app.version,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
def get_model_info():
    """Returns metadata for the currently active classification model."""
    manifest = ModelRegistry.get_manifest()
    config, categories = get_config()

    if not manifest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model manifest not found. Please train and export a model first.",
        )

    return ModelInfoResponse(
        model_name=manifest.get("model_name", config.model.active_model_name),
        version=manifest.get("version", "1.0.0"),
        created_at=manifest.get("created_at"),
        confidence_threshold=manifest.get("confidence_threshold", config.model.confidence_threshold),
        categories=manifest.get("categories", categories.canonical_names),
        macro_f1=manifest.get("macro_f1"),
        accuracy=manifest.get("accuracy"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_transaction(request: PredictRequest):
    """Categorizes a single transaction narration with calibrated confidence and review routing."""
    model = ModelRegistry.load_active_model()
    normalized = normalize_description(request.description)

    prediction = model.predict_single(normalized, threshold=request.threshold)

    return PredictResponse(
        description=request.description,
        normalized_description=normalized,
        predicted_category=prediction.predicted_category,
        confidence=prediction.confidence,
        requires_review=prediction.requires_review,
        review_reason=prediction.review_reason,
        alternatives=[
            AlternativeCategory(category=alt["category"], confidence=alt["confidence"])
            for alt in prediction.alternatives
        ],
        model_version=prediction.model_version,
    )


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch_transactions(request: BatchPredictRequest):
    """Batch inference endpoint for high-throughput statement categorization."""
    model = ModelRegistry.load_active_model()

    normalized_inputs = [normalize_description(item.description) for item in request.items]
    predictions = model.predict_batch(normalized_inputs, threshold=request.threshold)

    response_items = []
    review_count = 0

    for original_item, norm_desc, pred in zip(request.items, normalized_inputs, predictions):
        if pred.requires_review:
            review_count += 1

        response_items.append(PredictResponse(
            description=original_item.description,
            normalized_description=norm_desc,
            predicted_category=pred.predicted_category,
            confidence=pred.confidence,
            requires_review=pred.requires_review,
            review_reason=pred.review_reason,
            alternatives=[
                AlternativeCategory(category=alt["category"], confidence=alt["confidence"])
                for alt in pred.alternatives
            ],
            model_version=pred.model_version,
        ))

    return BatchPredictResponse(
        total_processed=len(response_items),
        requires_review_count=review_count,
        predictions=response_items,
    )


@app.post("/feedback", response_model=FeedbackResponse, tags=["Feedback"])
def record_user_feedback(request: FeedbackRequest):
    """Persists a human correction locally to power future candidate retraining."""
    _, categories = get_config()

    # Validate that the corrected category is part of the canonical taxonomy
    if request.corrected_category not in categories.canonical_names:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid category '{request.corrected_category}'. Must be one of: {categories.canonical_names}",
        )

    store = FeedbackStore()
    success = store.save_correction(
        transaction_id=request.transaction_id,
        normalized_description=request.normalized_description,
        original_category=request.original_category,
        corrected_category=request.corrected_category,
    )

    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to persist correction to local database.",
        )

    logger.info(f"Recorded correction for txn_id: {request.transaction_id[:8]}***")

    return FeedbackResponse(
        success=True,
        message="Correction saved successfully to local feedback database.",
        transaction_id=request.transaction_id,
    )


# --- Advanced Financial Analytics Endpoints ---

def _transactions_to_df(transactions) -> pd.DataFrame:
    """Helper converting Pydantic TransactionItem list to a canonical DataFrame."""
    rows = []
    for t in transactions:
        rows.append({
            "transaction_id": t.transaction_id,
            "date": t.date,
            "raw_description": t.raw_description,
            "normalized_description": normalize_description(t.raw_description),
            "amount": t.amount,
            "predicted_category": t.predicted_category or "Uncategorized",
        })
    return pd.DataFrame(rows)


@app.post("/analytics/tax", tags=["Analytics"])
def analyze_tax(request: TaxAnalyticsRequest) -> Dict[str, Any]:
    """Calculates Section 80C/80D/HRA deductions and simulates Old vs New Tax Regimes."""
    df = _transactions_to_df(request.transactions)
    deduction_summary = detect_tax_deductions(df, is_senior_citizen=request.is_senior_citizen)
    regime_comparison = compare_tax_regimes(
        annual_gross_income=request.annual_gross_income,
        total_deductions_claimed=deduction_summary.total_deductions_claimed,
    )
    return {
        "deductions": deduction_summary.model_dump(),
        "regime_comparison": regime_comparison.model_dump(),
    }


@app.post("/analytics/transfers", response_model=TransferReconciliationSummary, tags=["Analytics"])
def reconcile_transfers(request: TransferReconciliationRequest):
    """Separates internal settlements from real consumption to eliminate double-counting."""
    df = _transactions_to_df(request.transactions)
    _, summary = reconcile_inter_account_transfers(df)
    return summary


@app.post("/analytics/budget", response_model=BudgetReport, tags=["Analytics"])
def analyze_budget(request: BudgetAnalyticsRequest):
    """Evaluates 50/30/20 target allocations and projects mid-month burn rates."""
    df = _transactions_to_df(request.transactions)
    return analyze_50_30_20_budget(df, override_inflow=request.override_inflow)


@app.post("/analytics/health", response_model=FinancialHealthReport, tags=["Analytics"])
def evaluate_financial_health(request: HealthAnalyticsRequest):
    """Generates 0-100 Financial Vitality Score and emergency fund cash runway."""
    df = _transactions_to_df(request.transactions)
    return calculate_financial_health(df, override_liquid_balance=request.override_liquid_balance)


@app.post("/analytics/leaks", response_model=MoneyLeakReport, tags=["Analytics"])
def audit_money_leaks(request: LeakAnalyticsRequest):
    """Identifies recurring subscription redundancies, delivery app leaks, and micro-burn."""
    df = _transactions_to_df(request.transactions)
    return detect_money_leaks(df)

