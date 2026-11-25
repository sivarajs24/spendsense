"""Model artifact registry and loader for SpendSense inference."""

import json
from pathlib import Path
from typing import Any, Dict, Optional

import joblib

from spendsense.config import SpendSenseConfig, get_config
from spendsense.models.calibrate import CalibratedSpendSenseModel


class ModelRegistry:
    """Manages loading and metadata retrieval for the active deployed model artifact."""

    _cached_instance: Optional[CalibratedSpendSenseModel] = None
    _cached_manifest: Optional[Dict[str, Any]] = None

    @classmethod
    def load_active_model(
        cls,
        model_dir: Optional[Path] = None,
        config: Optional[SpendSenseConfig] = None,
        force_reload: bool = False,
    ) -> CalibratedSpendSenseModel:
        """Loads and caches the active model artifact."""
        if cls._cached_instance is not None and not force_reload:
            return cls._cached_instance

        if config is None:
            config, _ = get_config()

        m_dir = Path(model_dir or config.paths.models_dir)
        artifact_path = m_dir / "spendsense_classifier.joblib"
        manifest_path = m_dir / "model_manifest.json"

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Active model artifact not found at: {artifact_path}. "
                f"Please run 'python scripts/train_model.py' to train and export the model."
            )

        pipeline = joblib.load(artifact_path)

        manifest = {}
        if manifest_path.exists():
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        version = manifest.get("version", "1.0.0")
        threshold = manifest.get("confidence_threshold", config.model.confidence_threshold)

        # Wrap in CalibratedSpendSenseModel
        calibrated_model = CalibratedSpendSenseModel(
            fitted_pipeline=pipeline,
            model_version=version,
            confidence_threshold=threshold,
        )

        cls._cached_instance = calibrated_model
        cls._cached_manifest = manifest
        return calibrated_model

    @classmethod
    def get_manifest(cls, model_dir: Optional[Path] = None) -> Dict[str, Any]:
        """Returns the loaded model manifest metadata."""
        if cls._cached_manifest is None:
            cls.load_active_model(model_dir=model_dir)
        return cls._cached_manifest or {}
