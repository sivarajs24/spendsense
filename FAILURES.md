# SpendSense: Project Failures, Dead-Ends & Engineering Lessons Learned

This document is a living log required by the engineering principles of SpendSense. It tracks hypotheses that failed, models or features that underperformed, edge cases that broke assumptions, and key lessons learned during development.

---

## Log Entries

### Entry 001: Initial Architecture & Dependency Selection
* **Date:** 2026-10-30
* **Context:** Selecting baseline dependencies and packaging structure.
* **Decision / Attempt:**
  - Evaluated including deep learning libraries (PyTorch/Transformers) or external vector DBs (e.g., Chroma/Redis) for transaction categorization.
* **Why it was rejected / underperformed:**
  - Violates the core principle: Classical ML only (CPU-bound, 8GB RAM budget, Dell G15).
  - Heavy embedding models have high latency and memory footprints without offering interpretable n-gram signals for noisy Indian transaction formats (`UPI/4829/SWIGGY@ybl`).
  - Redis adds infrastructure overhead and persistence complexity for single-node local execution where SQLite provides ACID guarantees with zero daemon management.
* **Lesson Learned:**
  - A lightweight TF-IDF (word + character n-grams) pipeline with calibrated linear classifiers delivers sub-10ms inference and zero GPU overhead, making it directly suited for resource-constrained local deployment.

---

### Entry 002: Naive Tokenization on Indian UPI & NEFT Narrations
* **Date:** 2026-11-21
* **Context:** Preprocessing real-world bank statements and UPI narrations.
* **Decision / Attempt:**
  - Attempted standard English NLP word tokenization (`str.split()` or basic alphanumeric regex) on raw narrations like `UPI/4829102938/SWIGGY@ybl` or `POS 4012XXXXXXXX1234 STARBUCKS`.
* **Why it failed / underperformed:**
  - Standard whitespace tokenizers treat `UPI/4829102938/SWIGGY@ybl` as a single monolithic out-of-vocabulary (OOV) token.
  - High-entropy reference numbers (`4829102938`) and masked credit card numbers (`4012XXXXXXXX1234`) contaminated the vocabulary, threatening to cause classical linear classifiers to memorize transaction IDs rather than learning the merchant concept (`swiggy`, `starbucks`).
  - VPA suffixes (e.g., `@ybl`, `@okhdfcbank`) created false associations across unrelated merchants using the same payment service provider handle.
* **Lesson Learned:**
  - Implemented specialized delimiter splitting (`/`, `-`, `:`) for UPI headers, stripped masked card digits and UTR sequences via regex, and isolated pure merchant tokens prior to TF-IDF feature extraction.

---

### Entry 003: Word-Level TF-IDF Alone on Abbreviated & Unseen Merchants
* **Date:** 2026-11-21
* **Context:** Evaluating text representation for bank transaction descriptions.
* **Decision / Attempt:**
  - Evaluated pure word-level TF-IDF (unigrams + bigrams) without character n-grams.
* **Why it underperformed:**
  - Bank transaction descriptions frequently use non-standard truncations (e.g., `SWIGY DELIV`, `AMZN MKTPLACE`, `NFLX DIGTL`).
  - When tested on held-out merchants or slightly misspelled narrations, the word vectorizer had a 0.0 dot-product match (complete OOV failure), forcing linear classifiers to fall back to bias terms.
* **Lesson Learned:**
  - Combined word-level TF-IDF with character n-grams with word boundaries (`analyzer="char_wb"`, n-gram range 2 to 5) via `FeatureUnion`.
  - The character n-grams match morphological sub-word substrings (`swig`, `wigg`, `iggy`), ensuring unfamiliar merchant representations still activate learned weights.

---

### Entry 004: Sequential Slicing of Sorted Category Datasets Causing 0.0 Macro-F1
* **Date:** 2026-11-24
* **Context:** Training model baselines with initial script `train_model.py`.
* **Decision / Attempt:**
  - Partitioned the dataset using sequential slicing (`df.iloc[:train_size]` and `df.iloc[train_size:]`).
* **Why it failed / underperformed:**
  - `training_sample.csv` is grouped sequentially by category (Food, Transport, ..., Charity).
  - Slicing the first 70% placed Categories 1 through 7 exclusively in the Training set, while Categories 8 through 10 (Income, Government & Legal, Charity & Donations) were placed exclusively in the Validation set.
  - The ML classifiers were evaluated strictly on classes they were never trained on, resulting in an expected 0.0000 Macro-F1 and 0.0000 Accuracy across all statistical models. Only the keyword rule baseline scored above 0 because it relies on static dictionary heuristics.
* **Lesson Learned:**
  - Never use naive index slicing on tabular ML datasets.
  - Mandated `train_val_split` using `StratifiedShuffleSplit` so that every canonical category is sampled proportionally across both training and validation splits.

---

### Entry 005: Hugging Face Gated Dataset (401 Unauthorized) & Zero-Auth Fallback
* **Date:** 2026-11-24
* **Context:** Automated downloading of Dataset A (`mitulshah/transaction-categorization`).
* **Decision / Attempt:**
  - Attempted unauthenticated HTTP GET request to download the Parquet dataset from Hugging Face.
* **Why it failed / underperformed:**
  - Hugging Face returned `401 Client Error: Unauthorized`.
  - The dataset repository is "gated" by its author, requiring users to log in and accept terms of use. Direct anonymous downloading was blocked.
* **Lesson Learned:**
  - Enforced Principle #14 & Principle #24: If external datasets are gated or blocked, the project must provide a documented fallback and never require mandatory third-party credentials.
  - Enhanced `scripts/download_data.py` with:
    1. An optional `--hf-token` argument for users with existing Hugging Face accounts.
    2. A built-in zero-auth synthetic generator (`--generate-synthetic`) generating 25,000+ realistic multi-category Indian UPI/NEFT transactions in under 2 seconds.

---

### Entry 006: Windows SQLite File Locking (`[WinError 32]`) During Pytest Teardown
* **Date:** 2026-11-24
* **Context:** Automated unit testing of `FeedbackStore` in `tests/test_feedback.py`.
* **Decision / Attempt:**
  - Relied on standard `sqlite3` context manager (`with conn:`) inside `FeedbackStore._get_connection()` and used Python's `tempfile.TemporaryDirectory()`.
* **Why it failed / underperformed:**
  - In Python's `sqlite3`, `with conn:` manages transactions (commits and rollbacks), but does **NOT** close the underlying database connection.
  - On Windows, unclosed connection handles create an OS-level file lock on `test_feedback.db`.
  - When pytest tore down the test fixture, `TemporaryDirectory().cleanup()` threw `PermissionError: [WinError 32] The process cannot access the file because it is being used by another process`.
* **Lesson Learned:**
  - Refactored `_get_connection()` into a generator with `@contextmanager` that explicitly calls `conn.close()` inside a `finally` block to release file handles immediately.
  - Migrated test fixtures to pytest's built-in `tmp_path` fixture which gracefully manages temporary directory lifecycles on Windows.

---

### Entry 007: PyArrow / NumPy 1.26 ABI Incompatibility in Streamlit Table Rendering
* **Date:** 2026-11-25
* **Context:** Rendering interactive tables in Streamlit dashboard (`st.dataframe`).
* **Decision / Attempt:**
  - Used `st.dataframe()` to display raw transaction tables in the UI.
* **Why it failed / underperformed:**
  - Modern PyArrow wheels (18+) compiled for Windows strictly enforce NumPy >= 2.0. In environments running NumPy 1.26.x, importing `pyarrow.lib` triggers an uncatchable `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`.
* **Lesson Learned:**
  - Avoided relying on fragile binary table renderers for client dashboards. Replaced `st.dataframe()` with custom executive HTML card rosters and filter controls that run zero PyArrow dependencies, load instantaneously, and match high-end FinTech design systems.

---

### Entry 008: Binary vs. Multiclass Coefficient Slicing in Model Explainability
* **Date:** 2026-11-25
* **Context:** Extracting feature attributions from linear pipelines (`LinearSVC` / `LogisticRegression`) in `src/spendsense/features/explain.py`.
* **Decision / Attempt:**
  - Indexed classifier coefficients directly via `coef_matrix[class_idx, non_zero_indices]`.
* **Why it failed / underperformed:**
  - In scikit-learn binary classification (2 classes), linear classifiers fit a single separating hyperplane (`shape = (1, n_features)`).
  - Attempting to slice `class_idx = 1` into axis 0 triggers `IndexError: index 1 is out of bounds for axis 0 with size 1`.
* **Lesson Learned:**
  - Added explicit dimensionality checks: if `coef_matrix.shape[0] == 1`, slice row 0 and invert sign (`-class_coefs`) when explaining class 0.

---

### Entry 009: Cold-Start Data Sparsity in Analytics and Sentinel Modules
* **Date:** 2026-11-25
* **Context:** First-time dashboard startup with `sample_bank_statement.csv`.
* **Decision / Attempt:**
  - Loaded minimal 15-row statement as default application state.
* **Why it failed / underperformed:**
  - With only 15 transactions, no merchant had $\ge 2$ cycles to establish recurring subscription intervals, and no category met the 5-transaction threshold required for Tukey IQR anomaly fences. Tabs 4 and 5 rendered empty states.
* **Lesson Learned:**
  - Created `data/samples/rich_demo_statement.csv` containing a realistic 60+ transaction, 3-month Indian financial profile featuring 5 multi-cadence subscriptions (Netflix, Spotify, Airtel, Cult.fit, Zerodha) and 2 genuine Tukey spending spikes (Apple Store, fine dining celebration), paired with a seamless in-app dataset toggle.

  - Rendered transaction tables with standard `st.dataframe(tbl_view)`.
* **Why it failed / underperformed:**
  - Streamlit internally imports `pyarrow` to serialize DataFrames for the frontend grid.
  - A newer PyArrow 18+ binary installed in the environment was compiled strictly against NumPy 2.0+, throwing `ImportError: pyarrow requires NumPy 2.0 or newer, found 1.26.4`.
* **Lesson Learned:**
  - Pinned `pyarrow>=14.0.0,<17.0.0` in `pyproject.toml` and `requirements.txt` to maintain C-ABI compatibility with `numpy<2.0.0`.
  - Added a defensive `render_table()` wrapper in `dashboard/app.py` that catches any PyArrow import/ABI errors and transparently falls back to HTML-based `st.table()`, ensuring the UI never crashes.
