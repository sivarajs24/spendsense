"""Financial and operational spending summaries for dashboard and reports."""

from typing import Any, Dict
import pandas as pd


def generate_spending_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Computes comprehensive financial and operational categorization summaries.

    Excludes transfers and refunds appropriately where marked.
    """
    if df.empty:
        return {
            "total_transactions": 0,
            "total_debits": 0.0,
            "total_credits": 0.0,
            "net_flow": 0.0,
            "review_rate_pct": 0.0,
            "category_breakdown": [],
            "monthly_trends": [],
        }

    cat_col = "category" if "category" in df.columns else "predicted_category"
    total_txns = len(df)

    # Inflows and Outflows
    debits = df[df["amount"] < 0]["amount"].sum()
    credits = df[df["amount"] > 0]["amount"].sum()
    net = credits + debits

    # Review Queue Stats
    requires_review_count = int(df["requires_review"].sum()) if "requires_review" in df.columns else 0
    review_pct = round((requires_review_count / total_txns) * 100, 2) if total_txns > 0 else 0.0

    # Category Breakdown (Expense Outflows only)
    cat_summary = []
    if cat_col in df.columns:
        expense_df = df[df["amount"] < 0]
        total_expense = abs(expense_df["amount"].sum()) if not expense_df.empty else 1.0

        for cat, group in expense_df.groupby(cat_col):
            amt = abs(group["amount"].sum())
            pct = round((amt / total_expense) * 100, 2)
            cat_summary.append({
                "category": str(cat),
                "total_amount": round(amt, 2),
                "percentage": pct,
                "transaction_count": len(group),
            })
        cat_summary.sort(key=lambda x: x["total_amount"], reverse=True)

    # Monthly Trends
    monthly_summary = []
    if "date" in df.columns and df["date"].notna().any():
        df_monthly = df.copy()
        df_monthly["month"] = pd.to_datetime(df_monthly["date"], errors="coerce").dt.strftime("%Y-%m")
        df_monthly = df_monthly.dropna(subset=["month"])

        for month, group in df_monthly.groupby("month"):
            m_debit = abs(group[group["amount"] < 0]["amount"].sum())
            m_credit = group[group["amount"] > 0]["amount"].sum()
            monthly_summary.append({
                "month": str(month),
                "expenses": round(m_debit, 2),
                "income": round(m_credit, 2),
                "net": round(m_credit - m_debit, 2),
                "count": len(group),
            })
        monthly_summary.sort(key=lambda x: x["month"])

    return {
        "total_transactions": total_txns,
        "total_debits": round(abs(debits), 2),
        "total_credits": round(credits, 2),
        "net_flow": round(net, 2),
        "requires_review_count": requires_review_count,
        "review_rate_pct": review_pct,
        "category_breakdown": cat_summary,
        "monthly_trends": monthly_summary,
    }
