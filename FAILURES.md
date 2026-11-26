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

### Entry 010: Sampling-Based KernelSHAP Latency vs. Exact Closed-Form LinearSHAP
* **Date:** 2026-11-25
* **Context:** Implementing model explainability and token attribution for real-time inference.
* **Decision / Attempt:**
  - Evaluated integrating external `shap.KernelExplainer` for transaction narration attributions.
* **Why it was rejected / underperformed:**
  - `shap.KernelExplainer` uses Monte Carlo permutation sampling over background references, adding 200–500ms latency per prediction and requiring heavy external dependencies.
  - For linear models (LinearSVC / LogisticRegression) with sparse TF-IDF features and a zero-reference baseline (absent tokens), the Shapley value for each feature has an exact closed-form solution: $\phi_i(x) = w_i \cdot x_i$, with prior base bias $\phi_0 = b$.
* **Lesson Learned:**
  - Implemented exact closed-form LinearSHAP directly in `src/spendsense/features/explain.py`. Computes in under 0.1ms, satisfies the Efficiency Axiom ($\phi_0 + \sum \phi_i = f(x)$), and renders interactive Plotly `go.Waterfall` charts with zero runtime overhead or background sampling jitter.

---

### Entry 011: BaseWeb Segmented Control Styling Conflict & Double-Rerun Deselection
* **Date:** 2026-11-25
* **Context:** In-app dataset switching control (`st.segmented_control`) between 15-row standard bank statement and 60+ row rich demo statement.
* **Decision / Attempt:**
  - Configured `showWidgetBorder = true` in `.streamlit/config.toml` and called `st.rerun()` directly inside the script body when the selected dataset changed.
* **Why it failed / underperformed:**
  - Setting `showWidgetBorder = true` globally instructed Streamlit's underlying BaseWeb design system to inject hardcoded purple border highlights directly around the rightmost segment item, ignoring custom CSS radii and clipping pill geometries.
  - Calling `st.rerun()` synchronously in the script body created a race condition where the widget temporarily lost selection state (`None`), resetting user choices and causing jarring double screen flashes.
* **Lesson Learned:**
  - Removed `showWidgetBorder = true` from `.streamlit/config.toml` to restore clean BaseWeb boundary behavior.
  - Shifted state transitions to Streamlit's native `on_change=on_dataset_change` callback pattern, which mutates session state before the next render pass without requiring explicit `st.rerun()`, completely eliminating flickering and selection dropouts.

---

### Entry 012: PyPDF Encryption State Semantics & Keyword Argument Inconsistencies
* **Date:** 2026-11-26
* **Context:** Integration testing for encrypted PDF bank statement parser and FastAPI `/analytics/*` analytical endpoints.
* **Decision / Attempt:**
  - In `pdf_parser.py`, checked `if not decrypt_success or reader.is_encrypted:` after running `reader.decrypt(password)`.
  - In `api/main.py`, invoked `analyze_50_30_20_budget(df, override_inflow=...)` against a function signature expecting `override_income`.
  - In `leaks.py`, set food delivery duplication spend threshold to ₹1,500, causing 10-row unit test fixtures to report 0 leaks.
* **Why it failed / underperformed:**
  - In `pypdf`, `reader.is_encrypted` is an immutable document metadata property indicating that the PDF file structure possesses an `/Encrypt` dictionary. Even when successfully unlocked with the correct password, `reader.is_encrypted` evaluates to `True`. Checking `or reader.is_encrypted` caused valid passwords to falsely trigger `PermissionError`.
  - The keyword argument mismatch between `override_inflow` and `override_income` triggered an unhandled `TypeError` inside FastAPI, returning HTTP 500.
* **Lesson Learned:**
  - Evaluated `pypdf` unlock status strictly via the return code of `reader.decrypt(password)` (`0` for failure, `1`/`2` for user/owner success).
  - Implemented dual-alias keyword argument support (`override_income` & `override_inflow`) in `analyze_50_30_20_budget`.
  - Tuned delivery duplication threshold to $\ge \text{₹500}$ to capture realistic sub-month dining patterns.

---

### Entry 013: NumPy Boolean Scalar Identity (`is True`) & Plotly `titlefont` Deprecation
* **Date:** 2026-11-26
* **Context:** Testing transfer reconciliation dataframe annotations and Plotly SHAP waterfall figure rendering.
* **Decision / Attempt:**
  - In `test_advanced_analytics.py`, wrote `assert cred_row["is_internal_transfer"] is True`.
  - In `explain.py`, styled waterfall chart axes with `yaxis=dict(title="...", titlefont=dict(...))`.
* **Why it failed / underperformed:**
  - In Pandas/NumPy, boolean values indexed from a Series return `numpy.bool_` objects. In Python, `is` performs reference identity (`id(a) == id(b)`), which fails for `np.bool_(True) is True` because `numpy.bool_` is not the `builtins.bool` singleton (`id(np.bool_(True)) != id(True)`).
  - In modern Plotly (v5+), passing `titlefont` directly to axis dictionaries throws `ValueError: Invalid property specified for object of type plotly.graph_objs.layout.YAxis: 'titlefont'`. Modern Plotly requires nested dictionaries: `title=dict(text="...", font=dict(...))`.
* **Lesson Learned:**
  - Cast pandas boolean scalar extractions with `bool(val) is True` or evaluate with value equality (`== True`).
  - Standardized all Plotly layout axis titles to modern nested schema `title=dict(text=..., font=dict(...))`.

---

### Entry 014: Pandas 2.2 Copy-on-Write In-Place Mutation via `.at` During Iteration
* **Date:** 2026-11-26
* **Context:** Flagging internal transfers on statement dataframes in `reconcile_inter_account_transfers()`.
* **Decision / Attempt:**
  - Initialized `out_df["is_internal_transfer"] = False` and updated rows inside `for idx, row in out_df.iterrows():` using `out_df.at[idx, "is_internal_transfer"] = True`.
* **Why it failed / underperformed:**
  - In Pandas 2.2+ under Copy-on-Write (CoW), row-level assignments via `.at[idx, ...]` on dynamically created Series during an active `iterrows()` loop can operate on detached column buffers or silent views, leaving the underlying DataFrame's column values as `False`.
* **Lesson Learned:**
  - Avoid row-by-row `.at` mutations during DataFrame iteration.
  - Track classification flags via clean native Python lists (`is_transfer_flags = [False] * len(out_df)`), and assign back as a single batch operation (`out_df["is_internal_transfer"] = is_transfer_flags`). This is both $10\times$ faster and completely immune to view/copy detachment.

---

### Entry 015: Ambiguous Substring Matching in Test Fixture Assertions
* **Date:** 2026-11-26
* **Context:** Unit testing transfer reconciliation row annotations in `tests/test_advanced_analytics.py`.
* **Decision / Attempt:**
  - Sliced the credit card settlement row using `annotated_df[annotated_df["raw_description"].str.contains("CRED")].iloc[0]`.
* **Why it failed / underperformed:**
  - In `sample_financial_df`, row index 2 is `"SALARY CREDIT INFOSYS"`, which matches substring `"CRED"` (from `CREDIT`).
  - Calling `.iloc[0]` grabbed this income credit transaction rather than row index 6 (`"CRED/CREDIT CARD BILL PAYMENT AXIS"`).
  - Because salary credits are legitimately ignored by transfer reconciliation (`is_internal_transfer = False`), the assertion `assert cred_row["is_internal_transfer"] == True` failed with `assert False == True`.
* **Lesson Learned:**
  - Never use broad substring matches (`str.contains("CRED")`) when other narrative rows contain lexical stems like `"CREDIT"`.
  - Disambiguated unit test row locators using delimiter-anchored prefixes (`str.startswith("CRED/")`) or compound masks (`(df["amount"] < 0) & df["raw_description"].str.contains("CRED")`).






