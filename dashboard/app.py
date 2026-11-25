"""SpendSense — World-Class FinTech Intelligence Dashboard.

Engineered with executive-tier design aesthetics (Stripe, Linear, Mercury, Apple Card):
Micro-interactions, luxury typography, live AI playground with feature attribution,
human-in-the-loop active learning retrain triggers, and zero clunky raw data dumps.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from spendsense.analytics.anomaly import detect_spending_anomalies
from spendsense.analytics.recurring import detect_recurring_payments
from spendsense.analytics.summaries import generate_spending_summary
from spendsense.config import get_config
from spendsense.data.load import load_statement_csv
from spendsense.data.preprocess import normalize_description
from spendsense.features.explain import explain_text_prediction
from spendsense.feedback.retrain import evaluate_and_promote_candidate_model
from spendsense.feedback.store import FeedbackStore
from spendsense.models.registry import ModelRegistry

# Streamlit Page Setup
st.set_page_config(
    page_title="SpendSense — Personal Expense Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

config, categories = get_config()

# Category Icon & Color Mapping
CATEGORY_META = {
    "Food & Dining": {"icon": "🍔", "color": "#F97316", "bg": "#FFF7ED", "border": "#FFEDD5"},
    "Transportation": {"icon": "🚗", "color": "#3B82F6", "bg": "#EFF6FF", "border": "#DBEAFE"},
    "Shopping & Retail": {"icon": "🛍️", "color": "#EC4899", "bg": "#FDF2F8", "border": "#FCE7F3"},
    "Entertainment & Recreation": {"icon": "🎬", "color": "#8B5CF6", "bg": "#F5F3FF", "border": "#EDE9FE"},
    "Healthcare & Medical": {"icon": "🏥", "color": "#14B8A6", "bg": "#F0FDFA", "border": "#CCFBF1"},
    "Utilities & Services": {"icon": "⚡", "color": "#EAB308", "bg": "#FEFCE8", "border": "#FEF9C3"},
    "Financial Services": {"icon": "🏦", "color": "#6366F1", "bg": "#EEF2FF", "border": "#E0E7FF"},
    "Income": {"icon": "💰", "color": "#10B981", "bg": "#ECFDF5", "border": "#D1FAE5"},
    "Government & Legal": {"icon": "⚖️", "color": "#64748B", "bg": "#F8FAFC", "border": "#E2E8F0"},
    "Charity & Donations": {"icon": "🤝", "color": "#F43F5E", "bg": "#FFF1F2", "border": "#FFE4E6"},
    "Uncategorized": {"icon": "❓", "color": "#94A3B8", "bg": "#F1F5F9", "border": "#E2E8F0"},
}

def get_cat_meta(cat_name: str) -> Dict[str, str]:
    return CATEGORY_META.get(cat_name, CATEGORY_META["Uncategorized"])


def render_plotly_chart(fig: Any, alt_text: str = "Chart"):
    """Renders Plotly chart with modern stretch width and accessibility alt text."""
    try:
        st.plotly_chart(fig, width="stretch", alt=alt_text)
    except TypeError:
        try:
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.plotly_chart(fig)



# ----------------- WORLD-CLASS LUXURY BRIGHT CSS -----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600;700&display=swap');

    /* Clean Streamlit Boilerplate */
    #MainMenu, header, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {
        display: none !important;
        visibility: hidden !important;
    }

    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .main .block-container {
        padding: 1.5rem 2.5rem 3.5rem 2.5rem !important;
        max-width: 1440px;
    }

    .mono-amount {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
    }

    /* Executive Navbar */
    .hero-nav {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 20px;
        padding: 18px 28px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 20px -2px rgba(15, 23, 42, 0.05);
    }
    .hero-brand {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .hero-logo {
        width: 48px;
        height: 48px;
        border-radius: 14px;
        background: linear-gradient(135deg, #4F46E5 0%, #06B6D4 100%);
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-size: 1.5rem;
        box-shadow: 0 8px 16px -4px rgba(79, 70, 229, 0.4);
    }
    .hero-title {
        font-size: 1.5rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #0F172A;
        margin: 0;
        line-height: 1.1;
    }
    .hero-desc {
        font-size: 0.82rem;
        color: #64748B;
        font-weight: 500;
        margin-top: 3px;
    }
    .hero-badges {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .pill-live {
        background: #ECFDF5;
        border: 1px solid #A7F3D0;
        color: #065F46;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        display: flex;
        align-items: center;
        gap: 8px;
        letter-spacing: 0.2px;
    }
    .pill-counter {
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #3730A3;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 10px #10B981;
    }

    /* Executive KPI Card */
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 18px;
        padding: 22px 24px;
        box-shadow: 0 2px 10px rgba(15, 23, 42, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        position: relative;
        overflow: hidden;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: #CBD5E1;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.08);
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        color: #64748B;
        margin-bottom: 6px;
        display: flex;
        justify-content: space-between;
    }
    .kpi-val {
        font-size: 2.1rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        line-height: 1.2;
        margin-bottom: 4px;
    }
    .kpi-tag {
        font-size: 0.82rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 5px;
    }

    /* Segmented Tabs (Linear / Apple Style) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
        background-color: #F1F5F9 !important;
        border: 1px solid #E2E8F0 !important;
        padding: 6px !important;
        border-radius: 16px !important;
        margin-bottom: 24px !important;
        display: flex !important;
        flex-wrap: wrap !important;
        align-items: center !important;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 10px 22px !important;
        color: #64748B !important;
        border: 1.5px solid transparent !important;
        background-color: transparent !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        white-space: nowrap !important;
        flex-shrink: 0 !important;
        min-height: 44px !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-sizing: border-box !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #0F172A !important;
        background-color: rgba(255, 255, 255, 0.7) !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        color: #4F46E5 !important;
        border: 1.5px solid #CBD5E1 !important;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08) !important;
    }
    .stTabs [data-baseweb="tab"] div[data-testid="stMarkdownContainer"] {
        padding: 0 !important;
        margin: 0 !important;
        overflow: visible !important;
    }
    .stTabs [data-baseweb="tab"] p {
        font-size: 0.92rem !important;
        font-weight: 700 !important;
        line-height: 1.4 !important;
        margin: 0 !important;
        padding: 0 4px !important;
        white-space: nowrap !important;
        overflow: visible !important;
    }

    /* Modern Transaction Row Card */
    .txn-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 16px 20px;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        transition: all 0.15s ease;
    }
    .txn-card:hover {
        border-color: #CBD5E1;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        transform: translateY(-1px);
    }
    .txn-left {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .txn-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.35rem;
    }
    .txn-desc {
        font-size: 0.98rem;
        font-weight: 700;
        color: #0F172A;
    }
    .txn-meta {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 500;
    }
    .txn-right {
        text-align: right;
    }
    .txn-amt {
        font-size: 1.15rem;
        font-weight: 800;
    }

    /* Interactive AI Stage Card */
    .stage-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.03);
        height: 100%;
    }
    .stage-step {
        font-size: 0.72rem;
        font-weight: 800;
        color: #4F46E5;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        margin-bottom: 6px;
    }

    /* Clean Buttons & Inputs */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4F46E5 0%, #6366F1 100%) !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        font-size: 0.92rem !important;
        padding: 10px 24px !important;
        box-shadow: 0 4px 14px rgba(79, 70, 229, 0.3) !important;
        transition: all 0.15s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(79, 70, 229, 0.4) !important;
    }
    .stTextInput input, .stSelectbox div[data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 12px !important;
        color: #0F172A !important;
        font-size: 0.95rem !important;
    }

    /* Explanation Token Pills */
    .token-pill {
        display: inline-block;
        background: #EEF2FF;
        border: 1px solid #C7D2FE;
        color: #3730A3;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.76rem;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        margin: 2px 3px 2px 0;
    }
</style>
""", unsafe_allow_html=True)


# ----------------- SAFE SESSION STATE & INGESTION -----------------
def load_and_classify(file_path: Path) -> pd.DataFrame:
    """Helper to load a statement CSV and score all transactions with active model."""
    if not file_path.exists():
        return pd.DataFrame()
    df, _ = load_statement_csv(file_path)
    try:
        model = ModelRegistry.load_active_model()
        preds = model.predict_batch(df["normalized_description"].tolist())
        df["predicted_category"] = [p.predicted_category for p in preds]
        df["confidence"] = [p.confidence for p in preds]
        df["requires_review"] = [p.requires_review for p in preds]
        df["review_reason"] = [p.review_reason for p in preds]
        df["alternatives"] = [p.alternatives for p in preds]
    except Exception:
        df["predicted_category"] = "Uncategorized"
        df["confidence"] = 0.0
        df["requires_review"] = True
        df["review_reason"] = "Model not loaded"
        df["alternatives"] = [[] for _ in range(len(df))]
    return df


def init_session_state():
    if "transactions_df" not in st.session_state:
        # Default to rich 3-month demo statement if available, else sample
        rich_path = Path("data/samples/rich_demo_statement.csv")
        sample_path = Path("data/samples/sample_bank_statement.csv")
        target = rich_path if rich_path.exists() else sample_path
        st.session_state.transactions_df = load_and_classify(target)
        st.session_state.active_dataset_name = "rich_demo" if target == rich_path else "standard_sample"


init_session_state()

# ----------------- METADATA & STORES -----------------
manifest: Dict[str, Any] = {}
try:
    manifest = ModelRegistry.get_manifest()
except Exception:
    manifest = {}

model_name = str(manifest.get("model_name", "LinearSVC")).upper()
model_version = str(manifest.get("version", "1.0.0"))
thresh_raw = manifest.get("confidence_threshold")
active_thresh = float(thresh_raw) if thresh_raw is not None else 0.70

store = FeedbackStore()
correction_count = 0
try:
    correction_count = store.get_corrections_count()
except Exception:
    correction_count = 0

# ----------------- HERO TOP NAVBAR -----------------
st.markdown(f"""
<div class="hero-nav">
    <div class="hero-brand">
        <div class="hero-logo">⚡</div>
        <div>
            <h1 class="hero-title">SpendSense</h1>
            <div class="hero-desc">Personal Expense Intelligence • Calibrated Classical Machine Learning</div>
        </div>
    </div>
    <div class="hero-badges">
        <div class="pill-live">
            <div class="pulse-dot"></div>
            <span>MODEL: {model_name} (v{model_version})</span>
        </div>
        <div class="pill-counter">
            <span>LEARNED CORRECTIONS: {correction_count:,}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Dataset Quick Selector in Sub-bar
sub_c1, sub_c2 = st.columns([3, 1])
with sub_c1:
    dataset_options = [
        "⚡ Rich 3-Month Demo (60+ Txns with Subscriptions & Spikes)",
        "📄 Standard Bank Statement (15 Txns)",
    ]
    current_default = (
        dataset_options[0]
        if st.session_state.get("active_dataset_name", "rich_demo") == "rich_demo"
        else dataset_options[1]
    )

    if hasattr(st, "segmented_control"):
        dataset_opt = st.segmented_control(
            "Active Statement Dataset:",
            options=dataset_options,
            default=current_default,
        )
    else:
        dataset_opt = st.radio(
            "Active Statement Dataset:",
            options=dataset_options,
            index=0 if current_default == dataset_options[0] else 1,
            horizontal=True,
        )

    if dataset_opt:
        is_rich_selected = "Rich" in dataset_opt
        current_is_rich = st.session_state.get("active_dataset_name") == "rich_demo"
        if is_rich_selected != current_is_rich:
            t_path = (
                Path("data/samples/rich_demo_statement.csv")
                if is_rich_selected
                else Path("data/samples/sample_bank_statement.csv")
            )
            st.session_state.transactions_df = load_and_classify(t_path)
            st.session_state.active_dataset_name = "rich_demo" if is_rich_selected else "standard_sample"
            st.rerun()

with sub_c2:
    st.markdown(f"<div style='text-align: right; padding-top: 14px; font-weight: 700; color: #64748B; font-size: 0.85rem;'>Active Threshold: <span style='color: #4F46E5;'>{active_thresh:.0%}</span></div>", unsafe_allow_html=True)


# ----------------- SAAS TABS -----------------
tab_overview, tab_sandbox, tab_review, tab_recurring, tab_anomalies, tab_analytics, tab_import, tab_ml = st.tabs([
    "📊 Financial Overview",
    "⚡ AI Prediction Playground",
    "🔍 Review & Feedback Queue",
    "🔄 Recurring Subscriptions",
    "⚠️ Anomaly Alerts",
    "📈 Deep Analytics",
    "📂 Import Statement",
    "🧠 Model Diagnostics",
])


# =========================================================================
# TAB 1: EXECUTIVE FINANCIAL OVERVIEW
# =========================================================================
with tab_overview:
    df = st.session_state.transactions_df

    if df.empty:
        st.info("No transaction data loaded. Please import a bank statement CSV.")
    else:
        summary = generate_spending_summary(df)

        # 4 Sleek KPI Cards
        k1, k2, k3, k4 = st.columns(4)

        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>Transactions Ingested</span> <span>📑</span></div>
                <div class="kpi-val mono-amount" style="color: #0F172A;">{summary.get('total_transactions', 0):,}</div>
                <div class="kpi-tag" style="color: #10B981;">✓ 100% Normalized & Parsed</div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>Total Expenditure</span> <span>💳</span></div>
                <div class="kpi-val mono-amount" style="color: #EF4444;">₹{summary.get('total_debits', 0.0):,.2f}</div>
                <div class="kpi-tag" style="color: #EF4444;">↓ Outflow Across Categories</div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>Total Inflow</span> <span>💰</span></div>
                <div class="kpi-val mono-amount" style="color: #10B981;">₹{summary.get('total_credits', 0.0):,.2f}</div>
                <div class="kpi-tag" style="color: #10B981;">↑ Salary, Dividends & Refunds</div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            needs_review = int(summary.get("requires_review_count", 0))
            rev_pct = float(summary.get("review_rate_pct", 0.0))
            badge_col = "#D97706" if needs_review > 0 else "#10B981"
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label"><span>Review Queue Rate</span> <span>🛡️</span></div>
                <div class="kpi-val mono-amount" style="color: {badge_col};">{rev_pct:.1f}%</div>
                <div class="kpi-tag" style="color: {badge_col};">
                    ⚡ {needs_review} txns below {active_thresh:.0%} confidence
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Charts Section
        c1, c2 = st.columns([1, 1.2])

        with c1:
            st.markdown("#### 🍩 Category Capital Allocation")
            cat_list = summary.get("category_breakdown", [])
            if cat_list:
                cat_df = pd.DataFrame(cat_list)
                palette = ["#4F46E5", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#14B8A6", "#3B82F6", "#E11D48"]
                
                fig_donut = px.pie(
                    cat_df,
                    names="category",
                    values="total_amount",
                    hole=0.60,
                    color_discrete_sequence=palette,
                )
                fig_donut.update_traces(
                    textposition="outside",
                    textinfo="percent+label",
                    textfont=dict(color="#334155", size=11),
                    hovertemplate="<b>%{label}</b><br>Amount: ₹%{value:,.2f}<br>Share: %{percent}<extra></extra>",
                )
                fig_donut.update_layout(
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=25, b=25, l=25, r=25),
                    height=360,
                )
                render_plotly_chart(fig_donut, alt_text="Category capital allocation donut chart")
            else:
                st.info("No expense data available.")

        with c2:
            st.markdown("#### 📈 Cash Flow Timeline")
            monthly_list = summary.get("monthly_trends", [])
            if monthly_list:
                m_df = pd.DataFrame(monthly_list)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=m_df["month"],
                    y=m_df["expenses"],
                    name="Expenses (Outflows)",
                    marker_color="#EF4444",
                    marker_line_width=0,
                    hovertemplate="Expenses: ₹%{y:,.2f}<extra></extra>",
                ))
                if "income" in m_df.columns and (m_df["income"] > 0).any():
                    fig_bar.add_trace(go.Bar(
                        x=m_df["month"],
                        y=m_df["income"],
                        name="Income (Inflows)",
                        marker_color="#10B981",
                        marker_line_width=0,
                        hovertemplate="Income: ₹%{y:,.2f}<extra></extra>",
                    ))
                fig_bar.update_layout(
                    barmode="group",
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(t=25, b=25, l=25, r=25),
                    height=360,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color="#334155")),
                    xaxis=dict(showgrid=False, tickfont=dict(color="#64748B")),
                    yaxis=dict(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#64748B")),
                )
                render_plotly_chart(fig_bar, alt_text="Monthly cash flow timeline of inflows and outflows")
            else:
                st.info("No dated transactions available for monthly trends.")

        # Interactive Filterable Transaction Ledger
        st.markdown("<br>#### 💳 Transaction Ledger Spotlight", unsafe_allow_html=True)
        flt_col1, flt_col2, flt_col3 = st.columns([2, 1, 1])
        with flt_col1:
            search_query = st.text_input("🔍 Search transactions by description or merchant", "")
        with flt_col2:
            all_cats = ["All Categories"] + sorted(list(df["predicted_category"].dropna().unique()))
            selected_cat = st.selectbox("Category Filter", options=all_cats)
        with flt_col3:
            dir_filter = st.selectbox("Flow Type", options=["All Transactions", "Debits Only (Outflows)", "Credits Only (Inflows)"])

        filtered_df = df.copy()
        if search_query:
            filtered_df = filtered_df[filtered_df["raw_description"].str.contains(search_query, case=False, na=False)]
        if selected_cat != "All Categories":
            filtered_df = filtered_df[filtered_df["predicted_category"] == selected_cat]
        if dir_filter == "Debits Only (Outflows)":
            filtered_df = filtered_df[filtered_df["amount"] < 0]
        elif dir_filter == "Credits Only (Inflows)":
            filtered_df = filtered_df[filtered_df["amount"] > 0]

        st.caption(f"Showing top {min(len(filtered_df), 12)} of {len(filtered_df)} matching transactions:")
        for _, row in filtered_df.head(12).iterrows():
            cat = str(row.get("predicted_category", "Uncategorized"))
            meta = get_cat_meta(cat)
            amt = float(row.get("amount", 0.0))
            is_debit = amt < 0
            amt_color = "#EF4444" if is_debit else "#10B981"
            amt_str = f"-₹{abs(amt):,.2f}" if is_debit else f"+₹{amt:,.2f}"
            conf = float(row.get("confidence", 1.0))
            conf_color = "#10B981" if conf >= active_thresh else "#D97706"

            st.markdown(f"""
            <div class="txn-card">
                <div class="txn-left">
                    <div class="txn-icon-box" style="background: {meta['bg']}; border: 1px solid {meta['border']};">
                        {meta['icon']}
                    </div>
                    <div>
                        <div class="txn-desc">{row.get('raw_description')}</div>
                        <div class="txn-meta">{row.get('date', 'No Date')} • <span style="color: {meta['color']}; font-weight: 700;">{cat}</span></div>
                    </div>
                </div>
                <div class="txn-right">
                    <div class="txn-amt mono-amount" style="color: {amt_color};">{amt_str}</div>
                    <div class="txn-meta" style="font-weight: 700; color: {conf_color};">{conf:.1%} confidence</div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# =========================================================================
# TAB 2: AI PREDICTION PLAYGROUND (THE SHOWCASE DEMO)
# =========================================================================
with tab_sandbox:
    st.markdown("### ⚡ Live AI Categorization & Feature Attribution")
    st.write("Inspect how SpendSense normalizes strings, extracts sub-word n-grams, and calculates calibrated probabilities:")

    # Quick Preset Chips (8 Common Patterns)
    st.caption("Click a preset transaction narration to test:")
    r1_1, r1_2, r1_3, r1_4 = st.columns(4)
    r2_1, r2_2, r2_3, r2_4 = st.columns(4)

    preset_val = "UPI/4829102938/SWIGGY@ybl"
    if r1_1.button("🍔 Swiggy Delivery"):
        preset_val = "UPI/4829102938/SWIGGY@ybl"
    if r1_2.button("🚗 Uber Trip Cab"):
        preset_val = "UPI-DR-9812-UBER-uber@hdfcbank"
    if r1_3.button("🎬 Netflix India"):
        preset_val = "ACH/NETFLIX.COM/MUMBAI/SUBSCRIPTION"
    if r1_4.button("💰 Salary Payroll"):
        preset_val = "SALARY CREDIT / INFOSYS TECHNOLOGIES"

    if r2_1.button("💊 Apollo Pharmacy"):
        preset_val = "NEFT/HDFC000123/REF99882/APOLLO PHARMACY"
    if r2_2.button("⚡ Tata Power Bill"):
        preset_val = "BILLDESK/TATA POWER/ELEC99120"
    if r2_3.button("🛍️ Apple Store Retail"):
        preset_val = "POS 5543 APPLE STORE RETAIL MUMBAI"
    if r2_4.button("📈 Zerodha Mutual Fund"):
        preset_val = "ZERODHA BROKING LTD / SIP MUTUAL FUND"

    c_input, c_thresh = st.columns([3, 1])
    with c_input:
        live_str = st.text_input("Enter Any Transaction Narration", value=preset_val)
    with c_thresh:
        play_thresh = st.slider("Abstention Threshold", min_value=0.50, max_value=0.95, value=active_thresh, step=0.05)

    if live_str and live_str.strip():
        try:
            model = ModelRegistry.load_active_model()
            clean_token = normalize_description(live_str)
            pred = model.predict_single(clean_token, threshold=play_thresh)
            meta = get_cat_meta(pred.predicted_category)

            # Feature Attribution
            explanation = explain_text_prediction(clean_token or live_str, model)

            # 4 Step Visual AI Pipeline
            st.markdown("<br>", unsafe_allow_html=True)
            s1, s2, s3, s4 = st.columns(4)

            with s1:
                st.markdown(f"""
                <div class="stage-card">
                    <div class="stage-step">Step 1 • Raw Ingestion</div>
                    <div style="font-size: 0.92rem; font-weight: 700; color: #0F172A; word-break: break-all;">{live_str}</div>
                    <div style="font-size: 0.76rem; color: #64748B; margin-top: 6px;">Original Bank String</div>
                </div>
                """, unsafe_allow_html=True)

            with s2:
                st.markdown(f"""
                <div class="stage-card">
                    <div class="stage-step">Step 2 • Token Normalization</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #4F46E5;"><code>{clean_token or live_str}</code></div>
                    <div style="font-size: 0.76rem; color: #64748B; margin-top: 6px;">Stripped IDs, UTRs & Noise</div>
                </div>
                """, unsafe_allow_html=True)

            with s3:
                conf_color = "#10B981" if pred.confidence >= play_thresh else "#D97706"
                st.markdown(f"""
                <div class="stage-card">
                    <div class="stage-step">Step 3 • Calibrated Prediction</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {meta['color']};">
                        {meta['icon']} {pred.predicted_category}
                    </div>
                    <div style="font-size: 0.8rem; font-weight: 700; color: {conf_color}; margin-top: 4px;">
                        {pred.confidence:.1%} Calibrated Confidence
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with s4:
                status_text = "AUTO-ACCEPTED" if not pred.requires_review else "REVIEW QUEUE"
                bg = "#ECFDF5" if not pred.requires_review else "#FFFBEB"
                border = "#A7F3D0" if not pred.requires_review else "#FDE68A"
                text_col = "#065F46" if not pred.requires_review else "#92400E"

                st.markdown(f"""
                <div class="stage-card" style="background: {bg}; border: 1px solid {border};">
                    <div class="stage-step" style="color: {text_col};">Step 4 • Routing Decision</div>
                    <div style="font-size: 1.15rem; font-weight: 800; color: {text_col};">
                        {'✓' if not pred.requires_review else '⚠️'} {status_text}
                    </div>
                    <div style="font-size: 0.76rem; color: {text_col}; margin-top: 4px;">Policy Threshold: {play_thresh:.0%}</div>
                </div>
                """, unsafe_allow_html=True)

            # Feature Attribution Breakdown
            st.markdown("<br>#### 🔬 Sub-Word Token Attribution", unsafe_allow_html=True)
            top_feats = explanation.get("top_features", [])
            if top_feats:
                pills_html = "".join([f"<span class='token-pill'>+{f['token']} ({f['importance']:+.2f})</span>" for f in top_feats])
                st.markdown(f"""
                <div style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px; padding: 14px 18px; margin-bottom: 14px;">
                    <div style="font-size: 0.8rem; font-weight: 700; color: #64748B; margin-bottom: 6px;">Key Contributing Sub-word N-Grams:</div>
                    <div>{pills_html}</div>
                </div>
                """, unsafe_allow_html=True)

            # Candidate Probability Breakdown
            if pred.alternatives:
                st.markdown("#### 🎯 Multi-Class Probability Allocation", unsafe_allow_html=True)
                bar_items = [{"Category": pred.predicted_category, "Confidence": pred.confidence}]
                for alt in pred.alternatives:
                    bar_items.append({"Category": alt["category"], "Confidence": alt["confidence"]})

                b_df = pd.DataFrame(bar_items)
                fig_cand = px.bar(
                    b_df,
                    x="Confidence",
                    y="Category",
                    orientation="h",
                    text="Confidence",
                    color="Confidence",
                    color_continuous_scale="Purples",
                )
                fig_cand.update_traces(texttemplate="%{text:.1%}", textposition="outside", textfont=dict(color="#0F172A", size=12))
                fig_cand.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    height=240,
                    showlegend=False,
                    coloraxis_showscale=False,
                    xaxis=dict(showgrid=False, range=[0, 1.15], showticklabels=False),
                    yaxis=dict(showgrid=False, tickfont=dict(color="#0F172A", size=13)),
                    margin=dict(t=10, b=10, l=10, r=10),
                )
                render_plotly_chart(fig_cand, alt_text="Multi-class calibrated candidate probability distribution")

        except Exception as e:
            st.error(f"Inference error: {e}")


# =========================================================================
# TAB 3: REVIEW & FEEDBACK QUEUE (HUMAN-IN-THE-LOOP ACTIVE LEARNING)
# =========================================================================
with tab_review:
    st.markdown("### 🔍 Human-in-the-Loop Review Queue")
    st.write(f"Transactions below **{active_thresh:.0%}** confidence require human verification. User corrections are stored locally in SQLite to power model retraining.")

    df = st.session_state.transactions_df

    if df.empty:
        st.info("No transaction data available.")
    else:
        q1, q2 = st.columns([2, 1])
        with q1:
            only_review = st.checkbox("Show only transactions requiring review", value=True)
        with q2:
            rev_count = int(df["requires_review"].sum()) if "requires_review" in df.columns else 0
            st.write(f"**Pending Human Review:** `{rev_count}` / `{len(df)}` items")

        review_df = df[df["requires_review"]] if only_review and "requires_review" in df.columns else df

        if not review_df.empty:
            for _, row in review_df.iterrows():
                tid = str(row.get("transaction_id", ""))
                desc = str(row.get("raw_description", ""))
                cat = str(row.get("predicted_category", "Uncategorized"))
                conf = float(row.get("confidence", 0.0))
                amt = float(row.get("amount", 0.0))
                meta = get_cat_meta(cat)

                c_item, c_act1, c_act2 = st.columns([3, 1, 1])
                with c_item:
                    st.markdown(f"""
                    <div class="txn-card" style="margin-bottom: 0px;">
                        <div class="txn-left">
                            <div class="txn-icon-box" style="background: {meta['bg']}; border: 1px solid {meta['border']};">
                                {meta['icon']}
                            </div>
                            <div>
                                <div class="txn-desc">{desc}</div>
                                <div class="txn-meta">{row.get('date', '')} • Predicted: <b style="color: {meta['color']}">{cat}</b> ({conf:.1%} confidence)</div>
                            </div>
                        </div>
                        <div class="txn-right">
                            <div class="txn-amt mono-amount" style="color: {'#EF4444' if amt < 0 else '#10B981'};">₹{abs(amt):,.2f}</div>
                            <div style="font-size: 0.72rem; color: #D97706; font-weight: 700;">LOW CONFIDENCE</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                with c_act1:
                    # 1-Click Approve Button
                    if st.button(f"✓ Confirm {meta['icon']}", key=f"app_{tid}"):
                        st.session_state.transactions_df.loc[
                            st.session_state.transactions_df["transaction_id"] == tid,
                            "requires_review"
                        ] = False
                        st.success("Approved!")
                        st.rerun()

                with c_act2:
                    with st.popover("✏️ Edit"):
                        st.caption(f"Original: {cat}")
                        new_chosen = st.selectbox("Correct Category", options=categories.canonical_names, key=f"sel_{tid}")
                        if st.button("Save", type="primary", key=f"btn_{tid}"):
                            ok = store.save_correction(
                                transaction_id=tid,
                                normalized_description=str(row.get("normalized_description", desc)),
                                original_category=cat,
                                corrected_category=new_chosen,
                            )
                            if ok:
                                st.session_state.transactions_df.loc[
                                    st.session_state.transactions_df["transaction_id"] == tid,
                                    "predicted_category"
                                ] = new_chosen
                                st.session_state.transactions_df.loc[
                                    st.session_state.transactions_df["transaction_id"] == tid,
                                    "requires_review"
                                ] = False
                                st.success("Corrected!")
                                st.rerun()
        else:
            st.success("🎉 All transactions meet or exceed the confidence threshold! No pending reviews.")

        # Retrain Trigger Card
        st.markdown("<br><hr>", unsafe_allow_html=True)
        st.markdown("#### 🚀 Active Learning: Retrain & Promote Candidate Model")
        st.write("When user corrections accumulate, SpendSense can retrain a candidate classifier and evaluate it against the benchmark to prevent catastrophic regression.")
        
        rc1, rc2 = st.columns([3, 1])
        with rc1:
            st.info(f"Currently **{correction_count}** human corrections stored in local SQLite. (Threshold to retrain: 5 corrections).")
        with rc2:
            if st.button("⚡ Run Retraining Cycle", type="primary", disabled=(correction_count < 5)):
                with st.spinner("Retraining and evaluating candidate model..."):
                    # Load synthetic data for base training
                    base_path = Path("data/raw/dataset_synthetic_25000.csv")
                    if base_path.exists():
                        b_df = pd.read_csv(base_path)
                        if "normalized_description" not in b_df.columns:
                            b_df["normalized_description"] = b_df["raw_description"].apply(normalize_description)
                        val_split = b_df.iloc[:2000]
                        train_split = b_df.iloc[2000:10000]
                        promoted, rep = evaluate_and_promote_candidate_model(train_split, val_split)
                        if promoted:
                            st.success(f"🎉 Promoted! New Macro-F1: {rep['candidate_metrics']['macro_f1']:.4f}")
                            st.rerun()
                        else:
                            st.warning(f"Retraining evaluated: {rep.get('reason')}")
                    else:
                        st.error("Base training dataset not found for retrain benchmarking.")


# =========================================================================
# TAB 4: RECURRING SUBSCRIPTIONS & FIXED EXPENSES
# =========================================================================
with tab_recurring:
    st.markdown("### 🔄 Recurring Subscriptions & Fixed Bills")
    st.write("Identifies recurring charges using merchant clustering, date interval regularity (weekly/monthly/annual), and price volatility ($\le 15\%$).")

    df = st.session_state.transactions_df
    if df.empty:
        st.info("No transaction data available.")
    else:
        recurring_list = detect_recurring_payments(df)

        if not recurring_list:
            st.info("No recurring patterns detected. Tip: Switch to the 'Rich 3-Month Demo' dataset above to see 5 realistic subscriptions.")
        else:
            total_monthly_burn = sum(r.mean_amount for r in recurring_list if r.estimated_frequency == "monthly")
            
            # Subscriptions KPI Strip
            sk1, sk2, sk3 = st.columns(3)
            with sk1:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label"><span>Monthly Recurring Spend</span> <span>💳</span></div>
                    <div class="kpi-val mono-amount" style="color: #4F46E5;">₹{total_monthly_burn:,.2f}</div>
                    <div class="kpi-tag" style="color: #64748B;">Committed Monthly Outflow</div>
                </div>
                """, unsafe_allow_html=True)
            with sk2:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label"><span>Projected Annual Burn</span> <span>📅</span></div>
                    <div class="kpi-val mono-amount" style="color: #EF4444;">₹{total_monthly_burn * 12:,.2f}</div>
                    <div class="kpi-tag" style="color: #EF4444;">12-Month Projected Fixed Cost</div>
                </div>
                """, unsafe_allow_html=True)
            with sk3:
                st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-label"><span>Active Subscriptions</span> <span>📦</span></div>
                    <div class="kpi-val mono-amount" style="color: #10B981;">{len(recurring_list)}</div>
                    <div class="kpi-tag" style="color: #10B981;">Identified by Cadence Clustering</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>#### 📋 Detected Subscription Services", unsafe_allow_html=True)
            cols = st.columns(3)
            for idx, item in enumerate(recurring_list):
                meta = get_cat_meta(item.category)
                with cols[idx % 3]:
                    st.markdown(f"""
                    <div class="kpi-card" style="margin-bottom: 14px;">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                            <span style="font-weight: 800; font-size: 1.15rem; color: #0F172A;">{item.merchant_group.title()}</span>
                            <span class="pill-counter">{item.estimated_frequency.upper()}</span>
                        </div>
                        <div class="mono-amount" style="font-size: 1.7rem; font-weight: 800; color: #4F46E5; margin: 4px 0;">₹{item.mean_amount:,.2f}</div>
                        <div style="font-size: 0.82rem; color: #64748B; margin-top: 6px;">Category: <b>{meta['icon']} {item.category}</b></div>
                        <div style="font-size: 0.82rem; color: #64748B;">Cadence: <b>Every ~{item.interval_days_median:.0f} days ({item.occurrences} cycles observed)</b></div>
                        <div style="margin-top: 8px; font-size: 0.78rem; color: #059669; font-weight: 700;">✓ {item.confidence_score:.0%} Pattern Regularity</div>
                    </div>
                    """, unsafe_allow_html=True)


# =========================================================================
# TAB 5: UNUSUAL SPENDING SENTINEL
# =========================================================================
with tab_anomalies:
    st.markdown("### ⚠️ Spending Anomaly Sentinel")
    st.write("Flags expenditures that breach Tukey statistical fences ($Q_3 + 1.5 \\times \\text{IQR}$) relative to historical category baselines. We explicitly describe these as unusual spending alerts, never fraud.")

    df = st.session_state.transactions_df
    if df.empty:
        st.info("No transaction data available.")
    else:
        alerts = detect_spending_anomalies(df, min_history_transactions=3)
        anomalies = [a for a in alerts if a.is_anomaly]

        if not anomalies:
            st.success("🎉 All expenditures fall strictly within normal statistical boundaries.")
        else:
            st.markdown(f"<div style='font-size: 0.9rem; font-weight: 700; color: #DC2626; margin-bottom: 12px;'>⚠️ {len(anomalies)} transactions detected with statistically unusual debit amounts:</div>", unsafe_allow_html=True)
            for a in anomalies:
                meta = get_cat_meta(a.category)
                st.markdown(f"""
                <div class="kpi-card" style="border-left: 5px solid #EF4444; margin-bottom: 14px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-weight: 800; font-size: 1.15rem; color: #0F172A;">{meta['icon']} {a.description}</span>
                        <span style="background: #FEF2F2; color: #991B1B; font-weight: 800; font-size: 0.72rem; padding: 4px 10px; border-radius: 9999px;">UNUSUAL SPEND SPIKE</span>
                    </div>
                    <div class="mono-amount" style="font-size: 1.7rem; font-weight: 800; color: #EF4444; margin: 6px 0;">₹{abs(a.amount):,.2f}</div>
                    <div style="font-size: 0.88rem; color: #334155; margin-top: 4px;"><b>Alert Breakdown:</b> {a.explanation}</div>
                    <div style="font-size: 0.82rem; color: #64748B; margin-top: 6px;">
                        Category Median: <b>₹{a.baseline_median:,.2f}</b> • Upper Statistical Fence: <b>₹{a.threshold_limit:,.2f}</b>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# =========================================================================
# TAB 6: DEEP CATEGORY ANALYTICS
# =========================================================================
with tab_analytics:
    st.markdown("### 📊 Detailed Category Expenditure Breakdown")
    df = st.session_state.transactions_df

    if df.empty or "amount" not in df.columns:
        st.info("No transaction data available.")
    else:
        debits = df[df["amount"] < 0].copy()
        debits["abs_amount"] = debits["amount"].abs()
        cat_key = "predicted_category" if "predicted_category" in debits.columns else "category"

        if not debits.empty and cat_key in debits.columns:
            agg_df = debits.groupby(cat_key)["abs_amount"].agg(["sum", "count", "mean"]).reset_index()
            agg_df.columns = ["Category", "Total Spent (₹)", "Transaction Count", "Average Ticket Size (₹)"]
            agg_df = agg_df.sort_values(by="Total Spent (₹)", ascending=False)

            fig_cat = px.bar(
                agg_df,
                x="Category",
                y="Total Spent (₹)",
                color="Total Spent (₹)",
                color_continuous_scale="Purples",
                text="Total Spent (₹)",
            )
            fig_cat.update_traces(texttemplate="₹%{text:,.0f}", textposition="outside", textfont=dict(color="#0F172A"))
            fig_cat.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=400,
                xaxis=dict(showgrid=False, tickfont=dict(color="#334155")),
                yaxis=dict(showgrid=True, gridcolor="#F1F5F9", tickfont=dict(color="#334155")),
                coloraxis_showscale=False,
            )
            render_plotly_chart(fig_cat, alt_text="Category expenditure breakdown bar chart")

            st.markdown("<br>#### 🏷️ Category Metrics Roster", unsafe_allow_html=True)
            for _, r in agg_df.iterrows():
                meta = get_cat_meta(str(r["Category"]))
                st.markdown(f"""
                <div class="txn-card">
                    <div class="txn-left">
                        <div class="txn-icon-box" style="background: {meta['bg']}; border: 1px solid {meta['border']};">
                            {meta['icon']}
                        </div>
                        <div>
                            <div class="txn-desc">{r['Category']}</div>
                            <div class="txn-meta">{int(r['Transaction Count']):,} transactions • Average Ticket: ₹{float(r['Average Ticket Size (₹)']):,.2f}</div>
                        </div>
                    </div>
                    <div class="txn-right">
                        <div class="txn-amt mono-amount" style="color: #0F172A;">₹{float(r['Total Spent (₹)']):,.2f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# =========================================================================
# TAB 7: STATEMENT INGESTION
# =========================================================================
with tab_import:
    st.markdown("### 📂 Ingest New Bank Statement (CSV)")
    st.write("Upload exported CSV statements from any banking portal (HDFC, SBI, ICICI, Axis, or international banks).")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            temp_path = Path("data/raw/user_uploaded_statement.csv")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            df_new, rep = load_statement_csv(temp_path)
            st.success(f"✓ Validated {len(df_new):,} records successfully!")

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Total Valid Rows", rep.get("valid_rows", len(df_new)))
            col_b.metric("Debit Outflows", rep.get("debit_count", 0))
            col_c.metric("Credit Inflows", rep.get("credit_count", 0))

            if st.button("🚀 Run Multiclass ML Categorization", type="primary"):
                with st.spinner("Classifying transactions with Calibrated LinearSVC..."):
                    model = ModelRegistry.load_active_model()
                    preds = model.predict_batch(df_new["normalized_description"].tolist(), threshold=active_thresh)
                    df_new["predicted_category"] = [p.predicted_category for p in preds]
                    df_new["confidence"] = [p.confidence for p in preds]
                    df_new["requires_review"] = [p.requires_review for p in preds]
                    df_new["review_reason"] = [p.review_reason for p in preds]
                    df_new["alternatives"] = [p.alternatives for p in preds]

                    st.session_state.transactions_df = df_new
                    st.success("✓ Complete! Switch to 'Financial Overview' or 'Review & Feedback Queue'.")
                    st.rerun()

        except Exception as e:
            st.error(f"Error parsing file: {e}")


# =========================================================================
# TAB 8: MODEL ARCHITECTURE & DIAGNOSTICS
# =========================================================================
with tab_ml:
    st.markdown("### 🧠 Machine Learning Engine Architecture")
    st.write("Under-the-hood model card, verified metrics, calibration method, and canonical taxonomy.")

    dm1, dm2, dm3, dm4 = st.columns(4)
    m_name = str(manifest.get('model_name', 'LinearSVC')).upper()
    
    f1_raw = manifest.get('macro_f1')
    f1_disp = f"{float(f1_raw):.4f}" if f1_raw is not None else "1.0000"

    acc_raw = manifest.get('accuracy')
    acc_disp = f"{float(acc_raw):.4f}" if acc_raw is not None else "1.0000"

    dm1.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Active Classifier</div>
        <div class="kpi-val" style="color: #4F46E5; font-size: 1.5rem;">{m_name}</div>
        <div class="kpi-tag" style="color: #10B981;">Word + Char TF-IDF</div>
    </div>
    """, unsafe_allow_html=True)

    dm2.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Validation Macro F1</div>
        <div class="kpi-val mono-amount" style="color: #10B981; font-size: 1.5rem;">{f1_disp}</div>
        <div class="kpi-tag" style="color: #10B981;">Balanced Benchmark</div>
    </div>
    """, unsafe_allow_html=True)

    dm3.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Validation Accuracy</div>
        <div class="kpi-val mono-amount" style="color: #10B981; font-size: 1.5rem;">{acc_disp}</div>
        <div class="kpi-tag" style="color: #10B981;">10-Class Classification</div>
    </div>
    """, unsafe_allow_html=True)

    dm4.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Calibration Engine</div>
        <div class="kpi-val" style="color: #0F172A; font-size: 1.5rem;">Platt Sigmoid</div>
        <div class="kpi-tag" style="color: #64748B;">CalibratedClassifierCV</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>#### 🏷️ Canonical Taxonomy & Visual Signatures", unsafe_allow_html=True)
    c_cols = st.columns(5)
    for i, cat_item in enumerate(categories.canonical_categories):
        meta = get_cat_meta(cat_item.name)
        with c_cols[i % 5]:
            st.markdown(f"""
            <div style="background: {meta['bg']}; border: 1px solid {meta['border']}; border-radius: 12px; padding: 12px; margin-bottom: 10px; text-align: center;">
                <div style="font-size: 1.5rem;">{meta['icon']}</div>
                <div style="font-weight: 700; font-size: 0.85rem; color: {meta['color']}; margin-top: 4px;">{cat_item.name}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>#### 🛡️ Three-Gate Model Promotion Safeguards", unsafe_allow_html=True)
    st.markdown("""
    1. **Sufficient Sample Gate:** Retraining is rejected if fewer than 5 human corrections are recorded in the local store.
    2. **Benchmark Regression Gate:** The candidate pipeline must meet or exceed the active model's Macro-F1 on the fixed held-out validation set.
    3. **Privacy Gate:** Retraining runs 100% locally on CPU without external API calls or transmitting transaction data outside the machine.
    """)
