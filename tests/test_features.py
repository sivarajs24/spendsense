"""Unit tests for text feature engineering with combined Word and Character TF-IDF."""

from sklearn.pipeline import FeatureUnion

from spendsense.data.preprocess import normalize_description
from spendsense.features.text_features import build_text_feature_union


def test_build_text_feature_union_structure():
    """Verify FeatureUnion contains both word and character vectorizers."""
    feature_union = build_text_feature_union(
        use_word=True,
        use_char=True,
        max_word_features=500,
        max_char_features=1000,
    )
    assert isinstance(feature_union, FeatureUnion)
    transformer_names = [name for name, _ in feature_union.transformer_list]
    assert "word_tfidf" in transformer_names
    assert "char_tfidf" in transformer_names


def test_text_feature_union_fit_transform():
    """Verify feature union extracts expected dimensional representation."""
    corpus = [
        "swiggy delivery services bangalore",
        "zomato online food ordering",
        "uber trip mumbai",
        "netflix entertainment subscription",
    ]
    cleaned_corpus = [normalize_description(doc) for doc in corpus]

    feature_union = build_text_feature_union(
        use_word=True,
        use_char=True,
        max_word_features=100,
        max_char_features=200,
    )

    X_mat = feature_union.fit_transform(cleaned_corpus)
    assert X_mat.shape[0] == 4
    # Ensure features were extracted
    assert X_mat.shape[1] > 0
    # Ensure sparsity is reasonable
    assert X_mat.nnz > 0


def test_character_wb_captures_subword_patterns():
    """Verify character_wb captures sub-word tokens even with minor typos or abbreviations."""
    train_corpus = ["swiggy food delivery"]
    test_corpus = ["swigy order"]  # Typo in swiggy

    feature_union = build_text_feature_union(
        use_word=False,  # Isolate char n-grams
        use_char=True,
        char_ngram_range=(2, 4),
    )

    feature_union.fit(train_corpus)
    test_mat = feature_union.transform(test_corpus)
    # Even with typo, sub-word character fragments ('sw', 'wi', 'ig') should overlap
    assert test_mat.nnz > 0


def test_explain_text_prediction_linear_model():
    """Verify explain_text_prediction extracts contributing tokens from a trained pipeline."""
    from sklearn.svm import LinearSVC
    from sklearn.pipeline import Pipeline
    from spendsense.features.explain import explain_text_prediction

    train_texts = [
        "swiggy food delivery",
        "uber cab ride",
        "swiggy meal order",
        "uber transport driver",
    ]
    train_labels = ["Food & Dining", "Transportation", "Food & Dining", "Transportation"]

    features = build_text_feature_union(max_word_features=100, max_char_features=100)
    pipeline = Pipeline([
        ("features", features),
        ("classifier", LinearSVC(random_state=42)),
    ])
    pipeline.fit(train_texts, train_labels)

    explanation = explain_text_prediction("swiggy dinner", pipeline)
    assert explanation["predicted_category"] == "Food & Dining"
    assert len(explanation["top_features"]) > 0
    # Top token should relate to swiggy
    tokens = [f["token"] for f in explanation["top_features"]]
    assert any("swiggy" in t or "sw" in t for t in tokens)

