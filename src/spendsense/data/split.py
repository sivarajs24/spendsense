"""Data splitting strategies for classical ML evaluation.

Includes stratified splitting and grouped (unseen-merchant) splitting to
rigorously test out-of-distribution generalization.
"""

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit


def train_val_split(
    df: pd.DataFrame,
    target_col: str = "true_category",
    train_size: float = 0.70,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits a dataset into Train and Validation partitions with stratification.

    Ensures all classes are balanced across both splits.
    """
    clean_df = df.dropna(subset=[target_col]).copy()
    if clean_df.empty:
        raise ValueError(f"Target column '{target_col}' has no valid values to split.")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )

    train_idx, val_idx = next(sss.split(clean_df, clean_df[target_col]))
    train_df = clean_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = clean_df.iloc[val_idx].copy().reset_index(drop=True)

    return train_df, val_df


def train_val_test_split(
    df: pd.DataFrame,
    target_col: str = "true_category",
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits a dataset into Train, Validation, and Test partitions with stratification.

    Prevents class imbalance skew across splits without data leakage.
    """
    if abs((train_size + val_size + test_size) - 1.0) > 1e-5:
        raise ValueError("Train, validation, and test sizes must sum to 1.0.")

    clean_df = df.dropna(subset=[target_col]).copy()
    if clean_df.empty:
        raise ValueError(f"Target column '{target_col}' has no valid values to split.")

    # First split: Separate Train from Temp (Val + Test)
    temp_size = val_size + test_size
    sss_initial = StratifiedShuffleSplit(
        n_splits=1,
        test_size=temp_size,
        random_state=random_state,
    )

    train_idx, temp_idx = next(sss_initial.split(clean_df, clean_df[target_col]))
    train_df = clean_df.iloc[train_idx].copy()
    temp_df = clean_df.iloc[temp_idx].copy()

    # Second split: Separate Val and Test proportionally from Temp
    val_ratio_of_temp = val_size / temp_size
    sss_secondary = StratifiedShuffleSplit(
        n_splits=1,
        train_size=val_ratio_of_temp,
        random_state=random_state,
    )

    val_rel_idx, test_rel_idx = next(sss_secondary.split(temp_df, temp_df[target_col]))
    val_df = temp_df.iloc[val_rel_idx].copy()
    test_df = temp_df.iloc[test_rel_idx].copy()

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def grouped_merchant_split(
    df: pd.DataFrame,
    group_col: str = "merchant_group",
    target_col: str = "true_category",
    train_size: float = 0.70,
    test_size: float = 0.30,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Splits data such that whole merchant groups are exclusively in Train OR Test.

    Evaluates if the model generalizes to unseen merchants rather than memorizing tokens.
    """
    clean_df = df.dropna(subset=[target_col]).copy()
    if clean_df.empty:
        raise ValueError("No valid rows for grouped merchant splitting.")

    gss = GroupShuffleSplit(
        n_splits=1,
        train_size=train_size,
        random_state=random_state,
    )

    groups = clean_df[group_col].astype(str)
    train_idx, test_idx = next(gss.split(clean_df, clean_df[target_col], groups=groups))

    train_df = clean_df.iloc[train_idx].copy().reset_index(drop=True)
    test_df = clean_df.iloc[test_idx].copy().reset_index(drop=True)

    # Sanity check: Ensure disjoint merchant groups
    train_groups = set(train_df[group_col].unique())
    test_groups = set(test_df[group_col].unique())
    leakage = train_groups.intersection(test_groups)
    assert not leakage, f"Data leakage detected! Overlapping merchant groups: {leakage}"

    return train_df, test_df
