"""Example script demonstrating geometry feature extraction.

This script shows how to:
1. Load an IFC file
2. Extract geometry features (bounding boxes, centroids, volumes)
3. Calculate coverage statistics
4. Save ML-ready features

Usage:
    python examples/example_geometry_extraction.py path/to/model.ifc
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ifcqi.features import (
    compute_geometry_features,
    get_geometry_coverage_stats,
    get_ml_features,
    save_features,
)
from ifcqi.ifc_loader import load_ifc
from ifcqi.logger import setup_logging


def main() -> None:
    """Main function to demonstrate geometry extraction."""
    # Setup logging
    setup_logging(level="INFO")

    # Get IFC file path from command line
    if len(sys.argv) > 1:
        ifc_path = Path(sys.argv[1])
    else:
        print("Usage: python example_geometry_extraction.py <path_to_ifc_file>")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("Geometry Feature Extraction")
    print(f"{'=' * 60}")

    # Step 1: Load IFC file
    print(f"\nStep 1: Loading IFC file...")
    print(f"{'=' * 60}")

    model = load_ifc(ifc_path)
    print(f"✓ Loaded IFC model (Schema: {model.schema})")

    # Step 2: Extract geometry features
    print(f"\nStep 2: Extracting geometry features...")
    print(f"{'=' * 60}")
    print("This may take a while for large models...")

    features_df = compute_geometry_features(model)
    print(f"✓ Extracted features for {len(features_df)} elements")

    # Step 3: Show sample features
    print(f"\nStep 3: Sample extracted features...")
    print(f"{'=' * 60}")

    # Show elements with geometry
    with_geom = features_df[features_df["has_geometry"] == True].head()

    if len(with_geom) > 0:
        print("\nElements with geometry (first 5):")
        display_cols = [
            "name",
            "ifc_type",
            "has_geometry",
            "centroid_x",
            "centroid_y",
            "centroid_z",
            "dim_x",
            "dim_y",
            "dim_z",
            "bbox_volume",
        ]
        print(with_geom[display_cols].to_string())
    else:
        print("\n⚠ No elements with extractable geometry found")

    # Step 4: Calculate coverage statistics
    print(f"\nStep 4: Geometry coverage statistics...")
    print(f"{'=' * 60}")

    stats = get_geometry_coverage_stats(features_df)

    print(f"\nCoverage Statistics:")
    print(f"  Total Elements:              {stats['total_elements']}")
    print(f"  With Geometry:               {stats['elements_with_geometry']} "
          f"({stats['coverage_percentage']:.1f}%)")
    print(f"  Without Geometry:            {stats['elements_without_geometry']}")
    print(f"  Complete Feature Vectors:    {stats['complete_features']} "
          f"({stats['complete_percentage']:.1f}%)")

    # Step 5: Get ML-ready features
    print(f"\nStep 5: Preparing ML-ready features...")
    print(f"{'=' * 60}")

    ml_df = get_ml_features(features_df)
    print(f"✓ ML-ready features: {len(ml_df)} elements")
    print(f"  Feature columns: {list(ml_df.columns)}")

    if len(ml_df) > 0:
        print(f"\nFeature Statistics:")
        print(ml_df.describe().to_string())

    # Step 6: Save results
    print(f"\nStep 6: Saving results...")
    print(f"{'=' * 60}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save full features
    features_path = output_dir / "features.parquet"
    save_features(features_df, features_path)
    print(f"✓ Saved features to: {features_path}")

    # Save ML-ready features as CSV for easy viewing
    if len(ml_df) > 0:
        ml_path = output_dir / "ml_features.csv"
        ml_df.to_csv(ml_path, index=False)
        print(f"✓ Saved ML features to: {ml_path}")

    # Save feature summary as JSON
    import json

    summary_path = output_dir / "geometry_summary.json"
    with open(summary_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"✓ Saved summary to: {summary_path}")

    # Step 7: Feature insights
    print(f"\nStep 7: Feature insights...")
    print(f"{'=' * 60}")

    if len(features_df[features_df["has_geometry"] == True]) > 0:
        geom_df = features_df[features_df["has_geometry"] == True]

        # Find largest and smallest elements
        if geom_df["bbox_volume"].notna().any():
            largest_idx = geom_df["bbox_volume"].idxmax()
            smallest_idx = geom_df["bbox_volume"].idxmin()

            largest = geom_df.loc[largest_idx]
            smallest = geom_df.loc[smallest_idx]

            print(f"\nLargest Element (by bounding box volume):")
            print(f"  Name: {largest['name']}")
            print(f"  Type: {largest['ifc_type']}")
            print(f"  Volume: {largest['bbox_volume']:.2f} cubic units")
            print(f"  Dimensions: {largest['dim_x']:.2f} × {largest['dim_y']:.2f} × "
                  f"{largest['dim_z']:.2f}")

            print(f"\nSmallest Element (by bounding box volume):")
            print(f"  Name: {smallest['name']}")
            print(f"  Type: {smallest['ifc_type']}")
            print(f"  Volume: {smallest['bbox_volume']:.2f} cubic units")
            print(f"  Dimensions: {smallest['dim_x']:.2f} × {smallest['dim_y']:.2f} × "
                  f"{smallest['dim_z']:.2f}")

        # Check for unusual aspect ratios
        if geom_df["aspect_ratio_xy"].notna().any():
            unusual_ratios = geom_df[
                (geom_df["aspect_ratio_xy"] > 10) | (geom_df["aspect_ratio_xy"] < 0.1)
            ]

            if len(unusual_ratios) > 0:
                print(f"\n⚠ Found {len(unusual_ratios)} elements with unusual aspect ratios")
                print("  (may indicate data quality issues)")

    # Final summary
    print(f"\n{'=' * 60}")
    print("Summary")
    print(f"{'=' * 60}")
    print(f"✓ Geometry extraction completed!")
    print(f"  Elements processed:     {len(features_df)}")
    print(f"  With geometry:          {stats['elements_with_geometry']} "
          f"({stats['coverage_percentage']:.1f}%)")
    print(f"  ML-ready features:      {len(ml_df)}")
    print(f"\nOutput files:")
    print(f"  - {features_path}")
    if len(ml_df) > 0:
        print(f"  - {ml_path}")
    print(f"  - {summary_path}")


if __name__ == "__main__":
    main()
