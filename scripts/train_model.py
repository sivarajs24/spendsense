"""CLI training script to benchmark all classical models and export the best pipeline.

Usage:
    python scripts/train_model.py
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import argparse
import pandas as pd

from spendsense.config import get_config
from spendsense.data.preprocess import normalize_description
from spendsense.models.train import compare_all_models, train_and_evaluate_model


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate SpendSense classification models.")
    parser.add_argument("--data-path", type=str, default="data/samples/training_sample.csv", help="Path to labeled CSV.")
    parser.add_argument("--model", type=str, default="linear_svc", help="Model to train and save as active artifact.")
    parser.add_argument("--compare", action="store_true", default=True, help="Run side-by-side comparison of all 5 baselines.")
    args = parser.parse_args()

    config, categories = get_config()

    print("=" * 70)
    print(" SpendSense — Model Training & Baseline Benchmarking")
    print("=" * 70)

    data_file = Path(args.data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Dataset file not found: {data_file}")

    print(f"[*] Loading dataset from: {data_file}")
    df = pd.read_csv(data_file)
    df["normalized_description"] = df["raw_description"].apply(normalize_description)

    # Stratified Train/Val Split to ensure all 10 categories are represented in both splits
    from spendsense.data.split import train_val_split
    train_df, val_df = train_val_split(
        df,
        target_col="true_category",
        train_size=0.70,
        random_state=config.app.random_seed,
    )

    print(f"[*] Training samples: {len(train_df)} | Validation samples: {len(val_df)}")

    if args.compare:
        print("\n[*] Running side-by-side benchmark across all baseline models...")
        comparison_df = compare_all_models(
            train_df=train_df,
            val_df=val_df,
            config=config,
            categories=categories,
        )
        print("\n" + comparison_df.to_string(index=False) + "\n")

    # Train and export chosen candidate model
    export_path = Path(config.paths.models_dir) / "spendsense_classifier.joblib"
    print(f"[*] Training and saving active model '{args.model}' to: {export_path}")

    pipeline, metrics = train_and_evaluate_model(
        model_name=args.model,
        train_df=train_df,
        val_df=val_df,
        config=config,
        categories=categories,
        save_path=export_path,
    )

    manifest = {
        "model_name": args.model,
        "version": "1.0.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "accuracy": metrics["accuracy"],
        "confidence_threshold": config.model.confidence_threshold,
        "categories": categories.canonical_names,
    }

    manifest_path = Path(config.paths.models_dir) / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[+] Model manifest exported to: {manifest_path}")
    print(f"[+] Validation Macro F1: {metrics['macro_f1']} | Accuracy: {metrics['accuracy']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
