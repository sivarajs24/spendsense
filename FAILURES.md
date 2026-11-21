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
