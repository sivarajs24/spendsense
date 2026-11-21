"""Test configuration loading and category taxonomy schema."""

from spendsense.config import get_config


def test_config_loads_successfully():
    """Verify default.yaml loads into strongly-typed SpendSenseConfig."""
    config, categories = get_config()

    assert config.app.name == "SpendSense"
    assert config.data.train_split + config.data.val_split + config.data.test_split == 1.0
    assert config.model.confidence_threshold == 0.70
    assert config.features.use_word_tfidf is True
    assert config.features.use_char_tfidf is True


def test_categories_taxonomy_count_and_keys():
    """Verify canonical taxonomy contains exactly the 10 target categories."""
    _, categories = get_config()

    expected_categories = [
        "Food & Dining",
        "Transportation",
        "Shopping & Retail",
        "Entertainment & Recreation",
        "Healthcare & Medical",
        "Utilities & Services",
        "Financial Services",
        "Income",
        "Government & Legal",
        "Charity & Donations",
    ]

    canonical_names = categories.canonical_names
    assert len(canonical_names) == 10
    for cat in expected_categories:
        assert cat in canonical_names

    # Check that system categories exist
    system_cat_ids = [c.id for c in categories.system_categories]
    assert "transfer" in system_cat_ids
    assert "uncategorized" in system_cat_ids

