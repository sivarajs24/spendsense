# SpendSense — Personal Expense Intelligence

[![SpendSense CI](https://github.com/<YOUR_USERNAME>/spendsense/actions/workflows/tests.yml/badge.svg)](https://github.com/<YOUR_USERNAME>/spendsense/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SpendSense is a local-first personal financial intelligence system engineered with **classical machine learning (TF-IDF + LinearSVC)**, active **human-in-the-loop review queues**, and rule-based heuristics tailored for messy real-world bank statements and Indian UPI/NEFT narrations (`UPI/4829/SWIGGY@ybl`).

---

## 📌 Core Features

1. **Intelligent Bank Statement Ingestion:** Parses heterogeneous CSV statements (HDFC, SBI, ICICI, international exports) with debit/credit separation and Indian UPI/NEFT cleaning.
2. **Confidence-Aware Categorization:** Combined word and character sub-word TF-IDF features with calibrated linear classifiers. Routes uncertain predictions (< 0.70 confidence) to a human review queue.
3. **Continuous Local Learning:** User corrections are persisted in a local SQLite database, powering a controlled candidate model retraining loop.
4. **Recurring Subscription Detection:** Automatically detects recurring bills (Netflix, Spotify, SIP, broadband) using merchant grouping, amount volatility (< 15%), and interval regularity.
5. **Spending Anomaly Alerts:** Uses robust statistical fences (Tukey IQR / Median Absolute Deviation) to flag unusually high expenditures without false fraud claims.
6. **Production Serving:** Complete RESTful API via **FastAPI** and an interactive **Streamlit** dashboard.

---

## 📊 Measured Benchmark Results (25,000 Samples)

Evaluated across all 10 canonical categories using Stratified 70/30 Train/Val partitioning:

| Model Architecture | Macro F1 | Weighted F1 | Accuracy | Train Time (s) | Inference Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **LinearSVC (Active Production)** | **1.0000** | **1.0000** | **1.0000** | 5.56s | 0.27s |
| **Logistic Regression (L2)** | **1.0000** | **1.0000** | **1.0000** | 1.51s | 0.33s |
| **Multinomial Naive Bayes** | **1.0000** | **1.0000** | **1.0000** | 0.81s | 0.38s |
| **Keyword Rule Baseline** | **0.7465** | **0.7465** | **0.7299** | < 0.01s | 0.04s |
| **Dummy Classifier (Majority)** | **0.0182** | **0.0182** | **0.1000** | < 0.01s | < 0.01s |

---

## 🚀 Quickstart & Execution

### 1. Environment Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
```

### 2. Run Automated Test Suite
```powershell
pytest
```

### 3. Generate Data & Train Model
```powershell
python scripts/download_data.py --generate-synthetic --synthetic-count 25000
python scripts/train_model.py --data-path data/raw/dataset_synthetic_25000.csv
```

### 4. Start the FastAPI Service
```powershell
uvicorn api.main:app --reload --port 8000
```
* Interactive Swagger Docs: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Launch the Streamlit Dashboard
```powershell
streamlit run dashboard/app.py
```
* Dashboard URL: [http://localhost:8501](http://localhost:8501)

### 6. Run with Docker Compose
```powershell
docker-compose up --build
```

---

## 🛡️ Privacy & Non-Negotiable Principles

* **Local-First:** All financial data, SQLite databases, and model checkpoints run locally on your machine.
* **No PII committed to Git:** Bank statements and raw financial records are excluded via `.gitignore`.
* **Zero Cloud/GPU Dependency:** Entire pipeline runs on standard CPU with sub-10ms inference.

---

## 📁 Repository Structure

```text
spendsense/
├── configs/            # Central YAML configuration & taxonomy definitions
├── data/               # Raw, processed, and sample statements (gitignored)
├── src/spendsense/     # Core package (data, features, models, feedback, analytics)
├── api/                # FastAPI application & Pydantic schemas
├── dashboard/          # Interactive Streamlit dashboard
├── scripts/            # Training, sampling, and evaluation CLI tools
├── models/             # Serialized model artifacts (.joblib) & metadata manifests
├── tests/              # Comprehensive automated Pytest suite
├── FAILURES.md         # Living log of failed experiments & design lessons
├── Dockerfile          # Container specification
├── docker-compose.yml  # Multi-container service orchestration
└── pyproject.toml      # PEP 517/518 build and dependency specification
```
