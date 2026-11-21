"""Configuration management module for SpendSense.

Loads, validates, and exposes application and taxonomy configurations
using Pydantic v2 schemas and PyYAML.
"""

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
import yaml


class AppSettings(BaseModel):
    name: str = "SpendSense"
    version: str = "0.1.0"
    description: str = "Personal Expense Intelligence with Classical ML & Feedback Loops"
    debug: bool = False
    random_seed: int = 42


class PathsSettings(BaseModel):
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    samples_dir: Path = Path("data/samples")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    database_url: str = "sqlite:///data/spendsense.db"
    categories_config: Path = Path("configs/categories.yaml")


class DataSettings(BaseModel):
    sample_size: int = 150000
    train_split: float = 0.70
    val_split: float = 0.15
    test_split: float = 0.15
    min_description_length: int = 3
    max_description_length: int = 255


class FeaturesSettings(BaseModel):
    use_word_tfidf: bool = True
    use_char_tfidf: bool = True
    word_ngram_range: tuple[int, int] = (1, 2)
    char_ngram_range: tuple[int, int] = (2, 5)
    max_word_features: int = 15000
    max_char_features: int = 25000
    sublinear_tf: bool = True
    strip_accents: str = "unicode"


class ModelSettings(BaseModel):
    active_model_name: str = "linear_svc"
    confidence_threshold: float = 0.70
    calibration_method: str = "sigmoid"


class RecurringSettings(BaseModel):
    min_occurrences: int = 3
    max_interval_days_variance: int = 4
    max_amount_pct_variance: float = 0.05


class AnomalySettings(BaseModel):
    method: str = "statistical_iqr"
    min_history_transactions: int = 10
    iqr_multiplier: float = 1.5
    contamination_rate: float = 0.05


class MLflowSettings(BaseModel):
    tracking_uri: str = "file:./mlruns"
    experiment_name: str = "spendsense-categorization"


class SpendSenseConfig(BaseModel):
    app: AppSettings = Field(default_factory=AppSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    data: DataSettings = Field(default_factory=DataSettings)
    features: FeaturesSettings = Field(default_factory=FeaturesSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    recurring: RecurringSettings = Field(default_factory=RecurringSettings)
    anomaly: AnomalySettings = Field(default_factory=AnomalySettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)


class CategoryItem(BaseModel):
    id: str
    name: str
    description: str
    keywords: List[str] = Field(default_factory=list)


class CategoriesConfig(BaseModel):
    version: str = "1.0.0"
    canonical_categories: List[CategoryItem] = Field(default_factory=list)
    system_categories: List[CategoryItem] = Field(default_factory=list)

    @property
    def canonical_names(self) -> List[str]:
        return [cat.name for cat in self.canonical_categories]

    @property
    def id_to_name(self) -> Dict[str, str]:
        mapping = {cat.id: cat.name for cat in self.canonical_categories}
        mapping.update({cat.id: cat.name for cat in self.system_categories})
        return mapping

    @property
    def name_to_id(self) -> Dict[str, str]:
        mapping = {cat.name: cat.id for cat in self.canonical_categories}
        mapping.update({cat.name: cat.id for cat in self.system_categories})
        return mapping


def _find_project_root() -> Path:
    """Finds the root directory containing pyproject.toml."""
    current = Path(__file__).resolve().parent
    for parent in [current] + list(current.parents):
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def load_yaml(file_path: Path) -> Dict[str, Any]:
    """Safely loads a YAML file."""
    if not file_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


@lru_cache()
def get_config(
    config_path: Optional[str] = None,
    categories_path: Optional[str] = None,
) -> tuple[SpendSenseConfig, CategoriesConfig]:
    """Loads and caches both main configuration and categories taxonomy."""
    root = _find_project_root()

    cfg_file = (
        Path(config_path)
        if config_path
        else root / os.getenv("CONFIG_PATH", "configs/default.yaml")
    )
    cat_file = (
        Path(categories_path)
        if categories_path
        else root / os.getenv("CATEGORIES_PATH", "configs/categories.yaml")
    )

    raw_cfg = load_yaml(cfg_file)
    raw_cat = load_yaml(cat_file)

    config = SpendSenseConfig(**raw_cfg)
    categories = CategoriesConfig(**raw_cat)

    return config, categories

