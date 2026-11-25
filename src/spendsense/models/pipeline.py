"""Model pipeline factory for SpendSense.

Assembles scikit-learn Pipelines combining text feature extractors and classifiers.
Ensures preprocessing is encapsulated and fitted strictly on training data.
"""

from typing import Dict, List, Optional
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from spendsense.config import SpendSenseConfig
from spendsense.features.text_features import build_text_feature_union
from spendsense.models.baseline_rules import KeywordBaselineClassifier


def get_model_pipeline(
    model_name: str,
    config: SpendSenseConfig,
    keyword_map: Optional[Dict[str, List[str]]] = None,
) -> Pipeline:
    """Builds and returns an end-to-end scikit-learn Pipeline for the requested model.

    Supported model names:
        - 'dummy': Majority-class baseline (lower bound).
        - 'keyword': Deterministic dictionary matching baseline.
        - 'naive_bayes': Multinomial Naive Bayes with TF-IDF features.
        - 'logistic_regression': L2-regularized Logistic Regression with TF-IDF.
        - 'linear_svc': Calibrated Linear Support Vector Classifier with TF-IDF.
    """
    model_key = model_name.strip().lower()

    if model_key == "dummy":
        return Pipeline([
            ("clf", DummyClassifier(strategy="most_frequent"))
        ])

    if model_key == "keyword":
        return Pipeline([
            ("clf", KeywordBaselineClassifier(keyword_map=keyword_map or {}))
        ])

    # For ML models, build the combined word + char TF-IDF feature union
    feature_union = build_text_feature_union(
        use_word=config.features.use_word_tfidf,
        use_char=config.features.use_char_tfidf,
        word_ngram_range=config.features.word_ngram_range,
        char_ngram_range=config.features.char_ngram_range,
        max_word_features=config.features.max_word_features,
        max_char_features=config.features.max_char_features,
        sublinear_tf=config.features.sublinear_tf,
        strip_accents=config.features.strip_accents,
    )

    if model_key == "naive_bayes":
        clf = MultinomialNB(alpha=0.1)
    elif model_key == "logistic_regression":
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            random_state=config.app.random_seed,
            solver="lbfgs",
        )
    elif model_key == "linear_svc":
        clf = LinearSVC(
            C=1.0,
            max_iter=2000,
            class_weight="balanced",
            random_state=config.app.random_seed,
            dual="auto",
        )
    else:
        raise ValueError(
            f"Unsupported model name '{model_name}'. "
            f"Allowed models: ['dummy', 'keyword', 'naive_bayes', 'logistic_regression', 'linear_svc']"
        )

    return Pipeline([
        ("features", feature_union),
        ("clf", clf),
    ])
