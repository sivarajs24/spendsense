# SpendSense — Personal Expense Intelligence

SpendSense is a local-first personal finance intelligence system powered by classical machine learning (LinearSVC / TF-IDF), active human-in-the-loop review queues, and rule-based heuristics tailored for messy real-world transaction statements (including Indian UPI & NEFT formats).

---

## Key Pillars

1. **Bank Statement Ingestion & Preprocessing:** Robust parsing of varied CSV formats and UPI/NEFT narration cleanup.
2. **Confidence-Aware Categorization:** Calibrated text classification with an abstention/review queue for low-confidence transactions.
3. **Continuous Local Learning:** Local SQLite correction storage and controlled model retraining.
4. **Recurring Payment Detection:** Interval and price variance tracking for subscriptions and utility bills.
5. **Spending Anomaly Alerts:** Statistical baselines (IQR / Median Absolute Deviation) to highlight unusual expenditures.
6. **Local Deployment:** FastAPI backend and interactive Streamlit UI with zero cloud dependencies.

---

## Setup & Quickstart

See the project development roadmap and execution steps.

