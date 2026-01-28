"""Feature engineering for IFC elements.

This module combines geometry and metadata into ML-ready feature datasets.
"""

from pathlib import Path
from typing import Optional

import ifcopenshell
import pandas as pd

from ifcqi.geometry import (
    calculate_aspect_ratios,
    calculate_bbox_dimensions,
    calculate_bbox_volume,
    calculate_centroid,
    create_geometry_settings,
    has_geometry,
)
from ifcqi.ifc_loader import extract_elements
from ifcqi.logger import get_logger

logger = get_logger(__name__)


def compute_geometry_features(
    model: ifcopenshell.file,
    elements_df: Optional[pd.DataFrame] = None,
    max_elements: Optional[int] = None,
) -> pd.DataFrame:
    """Compute geometry features for all elements in a model.

    Args:
        model: Loaded IFC model.
        elements_df: Optional DataFrame with elements (from extract_elements).
                    If None, will extract elements first.
        max_elements: Optional limit on number of elements to process.

    Returns:
        DataFrame with geometry features added.

    Columns added:
        - has_geometry: Boolean indicating if geometry was extracted
        - centroid_x, centroid_y, centroid_z: Centroid coordinates
        - bbox_min_x, bbox_min_y, bbox_min_z: Bounding box minimum
        - bbox_max_x, bbox_max_y, bbox_max_z: Bounding box maximum
        - dim_x, dim_y, dim_z: Bounding box dimensions
        - bbox_volume: Volume of bounding box
        - aspect_ratio_xy, aspect_ratio_yz, aspect_ratio_xz: Dimension ratios

    Example:
        >>> model = load_ifc(Path("model.ifc"))
        >>> features_df = compute_geometry_features(model)
        >>> print(features_df[['name', 'has_geometry', 'bbox_volume']].head())
    """
    # Extract elements if not provided
    if elements_df is None:
        from ifcqi.ifc_loader import extract_elements

        logger.info("Extracting elements...")
        elements_df = extract_elements(model, max_elements=max_elements)
    else:
        # Apply max_elements limit if provided
        if max_elements is not None:
            elements_df = elements_df.head(max_elements)

    logger.info(f"Computing geometry features for {len(elements_df)} elements...")

    # Create geometry settings
    settings = create_geometry_settings(use_world_coords=True)

    # Initialize feature columns
    features = []

    # Process each element
    products = model.by_type("IfcProduct")

    # Create lookup dict for fast access
    element_lookup = {p.GlobalId: p for p in products}

    processed = 0
    failed = 0

    for idx, row in elements_df.iterrows():
        global_id = row["global_id"]

        # Get the IFC element
        element = element_lookup.get(global_id)

        if element is None:
            logger.warning(f"Element not found in model: {global_id}")
            features.append(_empty_geometry_features())
            failed += 1
            continue

        try:
            # Check if element has geometry
            if not has_geometry(element):
                features.append(_empty_geometry_features())
                continue

            # Extract geometry features
            centroid = calculate_centroid(element, settings)
            dims = calculate_bbox_dimensions(element, settings)
            volume = calculate_bbox_volume(element, settings)
            ratios = calculate_aspect_ratios(element, settings)

            # Create feature dict
            feature_dict = {
                "has_geometry": True,
                "centroid_x": centroid[0] if centroid is not None else None,
                "centroid_y": centroid[1] if centroid is not None else None,
                "centroid_z": centroid[2] if centroid is not None else None,
                "dim_x": dims[0] if dims is not None else None,
                "dim_y": dims[1] if dims is not None else None,
                "dim_z": dims[2] if dims is not None else None,
                "bbox_volume": volume,
                "aspect_ratio_xy": ratios[0] if ratios is not None else None,
                "aspect_ratio_yz": ratios[1] if ratios is not None else None,
                "aspect_ratio_xz": ratios[2] if ratios is not None else None,
            }

            features.append(feature_dict)
            processed += 1

        except Exception as e:
            logger.warning(f"Failed to extract geometry for {global_id}: {e}")
            features.append(_empty_geometry_features())
            failed += 1

        # Log progress every 100 elements
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed {idx + 1}/{len(elements_df)} elements...")

    # Create features DataFrame
    features_df = pd.DataFrame(features)

    # Combine with original elements
    result_df = pd.concat([elements_df.reset_index(drop=True), features_df], axis=1)

    logger.info(
        f"Geometry extraction complete: {processed} successful, "
        f"{failed} failed, {len(elements_df) - processed - failed} without geometry"
    )

    return result_df


def _empty_geometry_features() -> dict:
    """Create a dict with empty/None geometry features."""
    return {
        "has_geometry": False,
        "centroid_x": None,
        "centroid_y": None,
        "centroid_z": None,
        "dim_x": None,
        "dim_y": None,
        "dim_z": None,
        "bbox_volume": None,
        "aspect_ratio_xy": None,
        "aspect_ratio_yz": None,
        "aspect_ratio_xz": None,
    }


def get_ml_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Extract ML-ready numeric features from the features DataFrame.

    Filters to only include elements with geometry and numeric columns.

    Args:
        features_df: DataFrame from compute_geometry_features().

    Returns:
        DataFrame with only ML-ready numeric features.

    Example:
        >>> features_df = compute_geometry_features(model)
        >>> ml_df = get_ml_features(features_df)
        >>> print(ml_df.columns)
    """
    # Filter to elements with geometry
    df_with_geom = features_df[features_df["has_geometry"] == True].copy()

    # Select numeric geometry features
    ml_columns = [
        "centroid_x",
        "centroid_y",
        "centroid_z",
        "dim_x",
        "dim_y",
        "dim_z",
        "bbox_volume",
        "aspect_ratio_xy",
        "aspect_ratio_yz",
        "aspect_ratio_xz",
    ]

    ml_df = df_with_geom[ml_columns].copy()

    # Drop rows with any NaN values
    ml_df = ml_df.dropna()

    logger.info(f"ML-ready features: {len(ml_df)} elements with complete data")

    return ml_df


def get_geometry_coverage_stats(features_df: pd.DataFrame) -> dict:
    """Calculate statistics about geometry coverage.

    Args:
        features_df: DataFrame from compute_geometry_features().

    Returns:
        Dictionary with coverage statistics.

    Example:
        >>> features_df = compute_geometry_features(model)
        >>> stats = get_geometry_coverage_stats(features_df)
        >>> print(f"Coverage: {stats['coverage_percentage']:.1f}%")
    """
    total = len(features_df)
    with_geometry = features_df["has_geometry"].sum()
    without_geometry = total - with_geometry

    # Check completeness of features
    complete_features = features_df[
        features_df[
            [
                "centroid_x",
                "dim_x",
                "dim_y",
                "dim_z",
                "bbox_volume",
            ]
        ]
        .notna()
        .all(axis=1)
    ]

    stats = {
        "total_elements": total,
        "elements_with_geometry": int(with_geometry),
        "elements_without_geometry": int(without_geometry),
        "coverage_percentage": (with_geometry / total * 100) if total > 0 else 0,
        "complete_features": len(complete_features),
        "complete_percentage": (len(complete_features) / total * 100) if total > 0 else 0,
    }

    return stats


def save_features(features_df: pd.DataFrame, output_path: Path) -> None:
    """Save features DataFrame to parquet file.

    Args:
        features_df: DataFrame with features.
        output_path: Path where to save the parquet file.

    Example:
        >>> features_df = compute_geometry_features(model)
        >>> save_features(features_df, Path("output/features.parquet"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as parquet (more efficient than CSV for numeric data)
    features_df.to_parquet(output_path, index=False)

    logger.info(f"Saved features to {output_path}")


def load_features(input_path: Path) -> pd.DataFrame:
    """Load features DataFrame from parquet file.

    Args:
        input_path: Path to the parquet file.

    Returns:
        DataFrame with features.

    Example:
        >>> features_df = load_features(Path("output/features.parquet"))
    """
    features_df = pd.read_parquet(input_path)
    logger.info(f"Loaded {len(features_df)} elements from {input_path}")

    return features_df
