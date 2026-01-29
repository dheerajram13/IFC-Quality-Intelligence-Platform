"""Feature preprocessing for ML anomaly detection.

Prepares geometry features for machine learning models.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler

from ifcqi.logger import get_logger

logger = get_logger(__name__)


def select_ml_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Select relevant numeric features for ML.

    Args:
        features_df: DataFrame with geometry features

    Returns:
        DataFrame with selected ML features
    """
    # Core geometric features for anomaly detection
    ml_feature_cols = [
        "dim_x",
        "dim_y",
        "dim_z",
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "aspect_ratio_xy",
        "aspect_ratio_yz",
        "aspect_ratio_xz",
    ]

    # Filter to only columns that exist in the DataFrame
    available_cols = [col for col in ml_feature_cols if col in features_df.columns]

    if not available_cols:
        raise ValueError("No ML features available in DataFrame")

    # Select features and filter to rows with geometry
    if "has_geometry" in features_df.columns:
        df = features_df[features_df["has_geometry"] == True].copy()  # noqa: E712
    else:
        df = features_df.copy()

    ml_df = df[available_cols].copy()

    # Replace inf values with NaN
    ml_df = ml_df.replace([np.inf, -np.inf], np.nan)

    logger.info(
        f"Selected {len(available_cols)} ML features from {len(features_df)} elements "
        f"({len(ml_df)} with geometry)"
    )

    return ml_df


def handle_missing_values(
    ml_df: pd.DataFrame, strategy: str = "median", fill_value: float = 0.0
) -> pd.DataFrame:
    """Handle missing values in ML features.

    Args:
        ml_df: DataFrame with ML features
        strategy: 'median', 'mean', or 'constant'
        fill_value: Value to use if strategy='constant'

    Returns:
        DataFrame with missing values handled
    """
    if ml_df.empty:
        return ml_df

    missing_count = ml_df.isnull().sum().sum()
    if missing_count == 0:
        logger.info("No missing values found")
        return ml_df

    df = ml_df.copy()

    if strategy == "median":
        df = df.fillna(df.median())
    elif strategy == "mean":
        df = df.fillna(df.mean())
    elif strategy == "constant":
        df = df.fillna(fill_value)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Fill any remaining NaN with 0 (e.g., if all values in column were NaN)
    df = df.fillna(0.0)

    logger.info(f"Filled {missing_count} missing values using strategy='{strategy}'")

    return df


def scale_features(
    ml_df: pd.DataFrame,
    scaler_type: str = "robust",
    scaler: Optional[RobustScaler | StandardScaler] = None,
) -> tuple[pd.DataFrame, RobustScaler | StandardScaler]:
    """Scale features using RobustScaler or StandardScaler.

    Args:
        ml_df: DataFrame with ML features
        scaler_type: 'robust' or 'standard'
        scaler: Pre-fitted scaler (optional, for transform-only)

    Returns:
        Tuple of (scaled_df, fitted_scaler)
    """
    if ml_df.empty:
        if scaler_type == "robust":
            return ml_df, RobustScaler()
        else:
            return ml_df, StandardScaler()

    if scaler is None:
        # Fit new scaler
        if scaler_type == "robust":
            scaler = RobustScaler()
        elif scaler_type == "standard":
            scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")

        scaled_values = scaler.fit_transform(ml_df.values)
        logger.info(f"Fitted {scaler_type} scaler on {len(ml_df)} samples")
    else:
        # Use pre-fitted scaler
        scaled_values = scaler.transform(ml_df.values)
        logger.info(f"Transformed {len(ml_df)} samples using pre-fitted scaler")

    scaled_df = pd.DataFrame(scaled_values, columns=ml_df.columns, index=ml_df.index)

    return scaled_df, scaler


def preprocess_features(
    features_df: pd.DataFrame,
    missing_strategy: str = "median",
    scaler_type: str = "robust",
    scaler: Optional[RobustScaler | StandardScaler] = None,
) -> tuple[pd.DataFrame, pd.Index, RobustScaler | StandardScaler]:
    """Full preprocessing pipeline: select, clean, and scale features.

    Args:
        features_df: Raw features DataFrame
        missing_strategy: Strategy for handling missing values
        scaler_type: Type of scaler to use
        scaler: Pre-fitted scaler (optional)

    Returns:
        Tuple of (processed_df, original_indices, fitted_scaler)
    """
    # Select ML features
    ml_df = select_ml_features(features_df)

    # Store original indices before preprocessing
    original_indices = ml_df.index

    # Handle missing values
    ml_df = handle_missing_values(ml_df, strategy=missing_strategy)

    # Scale features
    scaled_df, fitted_scaler = scale_features(ml_df, scaler_type=scaler_type, scaler=scaler)

    logger.info(
        f"Preprocessing complete: {len(scaled_df)} samples, {len(scaled_df.columns)} features"
    )

    return scaled_df, original_indices, fitted_scaler


def add_derived_features(ml_df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered features for better anomaly detection.

    Args:
        ml_df: DataFrame with base ML features

    Returns:
        DataFrame with additional derived features
    """
    df = ml_df.copy()

    # Volume (if dimensions available)
    if all(col in df.columns for col in ["dim_x", "dim_y", "dim_z"]):
        df["volume"] = df["dim_x"] * df["dim_y"] * df["dim_z"]

    # Surface area (approximate)
    if all(col in df.columns for col in ["dim_x", "dim_y", "dim_z"]):
        df["surface_area"] = 2 * (
            df["dim_x"] * df["dim_y"] + df["dim_y"] * df["dim_z"] + df["dim_z"] * df["dim_x"]
        )

    # Distance from origin
    if all(col in df.columns for col in ["centroid_x", "centroid_y", "centroid_z"]):
        df["distance_from_origin"] = np.sqrt(
            df["centroid_x"] ** 2 + df["centroid_y"] ** 2 + df["centroid_z"] ** 2
        )

    # Max dimension
    if all(col in df.columns for col in ["dim_x", "dim_y", "dim_z"]):
        df["max_dimension"] = df[["dim_x", "dim_y", "dim_z"]].max(axis=1)

    # Min dimension
    if all(col in df.columns for col in ["dim_x", "dim_y", "dim_z"]):
        df["min_dimension"] = df[["dim_x", "dim_y", "dim_z"]].min(axis=1)

    # Aspect ratio range
    if all(
        col in df.columns for col in ["aspect_ratio_xy", "aspect_ratio_yz", "aspect_ratio_xz"]
    ):
        aspect_cols = ["aspect_ratio_xy", "aspect_ratio_yz", "aspect_ratio_xz"]
        df["aspect_ratio_max"] = df[aspect_cols].max(axis=1)
        df["aspect_ratio_min"] = df[aspect_cols].min(axis=1)

    logger.info(f"Added derived features: {len(df.columns) - len(ml_df.columns)} new features")

    return df
