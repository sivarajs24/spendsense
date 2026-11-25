"""Text feature extraction using combined Word-level and Character-level TF-IDF.

Character n-grams with word boundaries ('char_wb') provide resilience against
abbreviations, typographical errors, and unfamiliar merchant tokens common in UPI narrations.
"""

from typing import Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_text_feature_union(
    use_word: bool = True,
    use_char: bool = True,
    word_ngram_range: Tuple[int, int] = (1, 2),
    char_ngram_range: Tuple[int, int] = (2, 5),
    max_word_features: int = 15000,
    max_char_features: int = 25000,
    sublinear_tf: bool = True,
    strip_accents: Optional[str] = "unicode",
) -> FeatureUnion:
    """Builds a scikit-learn FeatureUnion combining word and sub-word character TF-IDF representations.

    Args:
        use_word: Whether to include word n-grams (captures exact merchant names).
        use_char: Whether to include character n-grams with word boundaries.
        word_ngram_range: Tuple (min_n, max_n) for word vectorizer (default unigrams + bigrams).
        char_ngram_range: Tuple (min_n, max_n) for char vectorizer (sub-word fragments).
        max_word_features: Maximum vocabulary cap for word TF-IDF.
        max_char_features: Maximum vocabulary cap for character TF-IDF.
        sublinear_tf: Apply sublinear scaling (1 + log(tf)) to prevent frequent words from dominating.
        strip_accents: Normalize accented characters during tokenization.

    Returns:
        sklearn.pipeline.FeatureUnion ready to be fitted strictly on training data.
    """
    transformers = []

    if use_word:
        word_vec = TfidfVectorizer(
            analyzer="word",
            ngram_range=word_ngram_range,
            max_features=max_word_features,
            sublinear_tf=sublinear_tf,
            strip_accents=strip_accents,
            token_pattern=r"(?u)\b\w+\b",
        )
        transformers.append(("word_tfidf", word_vec))

    if use_char:
        # 'char_wb' generates character n-grams only from text inside word boundaries.
        # This prevents n-grams from spanning whitespace across unrelated tokens.
        char_vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=char_ngram_range,
            max_features=max_char_features,
            sublinear_tf=sublinear_tf,
            strip_accents=strip_accents,
        )
        transformers.append(("char_tfidf", char_vec))

    if not transformers:
        raise ValueError("At least one of use_word or use_char must be True.")

    return FeatureUnion(transformers)
