"""Example: ML-based anomaly detection on IFC geometry features.

This example demonstrates Module 6 (ML Component) capabilities:
1. Load IFC file and extract geometry features
2. Preprocess features for ML
3. Train Isolation Forest anomaly detector
4. Detect and rank anomalies
5. Save model and results

Usage:
    python examples/example_ml_anomaly_detection.py examples/ifc_files/Duplex_MEP_20110907.ifc
"""

import sys
from pathlib import Path

from ifcqi.features import compute_geometry_features
from ifcqi.ifc_loader import extract_elements, load_ifc
from ifcqi.logger import get_logger
from ifcqi.ml import AnomalyDetector, anomalies_to_dataframe

logger = get_logger(__name__)


def main(ifc_path: Path, output_dir: Path = Path("output")) -> None:
    """Run the ML anomaly detection pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IFC Quality Intelligence - ML Anomaly Detection")
    print("=" * 70)
    print()

    # Step 1: Load IFC
    print("Step 1: Loading IFC file...")
    print("-" * 70)
    model = load_ifc(ifc_path)
    print(f"✓ Loaded IFC model: {ifc_path.name}")
    print()

    # Step 2: Extract elements
    print("Step 2: Extracting elements...")
    print("-" * 70)
    elements_df = extract_elements(model)
    print(f"✓ Extracted {len(elements_df)} elements")
    print()

    # Step 3: Extract geometry features
    print("Step 3: Extracting geometry features...")
    print("-" * 70)
    features_df = compute_geometry_features(model, elements_df)
    n_with_geometry = features_df["has_geometry"].sum()
    print(f"✓ Geometry features extracted")
    print(f"  Elements with geometry: {n_with_geometry}/{len(features_df)}")
    print()

    # Step 4: Train anomaly detector
    print("Step 4: Training Isolation Forest anomaly detector...")
    print("-" * 70)
    contamination = 0.05  # Expect 5% anomalies
    detector = AnomalyDetector(contamination=contamination, n_estimators=100, random_state=42)
    detector.fit(features_df)
    print(f"✓ Model trained with contamination={contamination}")
    print(f"  Features used: {len(detector.feature_names)}")
    print(f"  Feature names: {', '.join(detector.feature_names[:5])}...")
    print()

    # Step 5: Detect anomalies
    print("Step 5: Detecting anomalies...")
    print("-" * 70)
    results = detector.predict(features_df)
    anomalies_df = anomalies_to_dataframe(results)
    n_anomalies = anomalies_df["is_anomaly"].sum()
    print(f"✓ Anomaly detection complete")
    print(f"  Total elements analyzed: {len(results)}")
    print(f"  Detected anomalies: {n_anomalies}")
    print(f"  Anomaly rate: {n_anomalies / max(1, len(results)) * 100:.1f}%")
    print()

    # Step 6: Analyze top anomalies
    print("Step 6: Top 10 anomalies (sorted by probability)...")
    print("-" * 70)
    top_anomalies = anomalies_df.nlargest(10, "anomaly_probability")

    if not top_anomalies.empty:
        for idx, row in top_anomalies.iterrows():
            print(
                f"  {row['ifc_type']:20s} | Prob: {row['anomaly_probability']:.3f} | "
                f"Score: {row['anomaly_score']:+.3f} | ID: {row['global_id'][:16]}"
            )
    else:
        print("  No anomalies detected")
    print()

    # Step 7: Analyze by IFC type
    print("Step 7: Anomaly distribution by IFC type...")
    print("-" * 70)
    anomaly_by_type = (
        anomalies_df[anomalies_df["is_anomaly"] == True]  # noqa: E712
        .groupby("ifc_type")
        .size()
        .sort_values(ascending=False)
        .head(5)
    )

    if not anomaly_by_type.empty:
        for ifc_type, count in anomaly_by_type.items():
            total_of_type = (features_df["ifc_type"] == ifc_type).sum()
            pct = count / total_of_type * 100 if total_of_type > 0 else 0
            print(f"  {ifc_type:25s}: {count:3d} anomalies ({pct:5.1f}% of {total_of_type})")
    else:
        print("  No anomalies detected")
    print()

    # Step 8: Save outputs
    print("Step 8: Saving results and model...")
    print("-" * 70)

    # Save anomaly results
    anomalies_df.to_csv(output_dir / "ml_anomalies.csv", index=False)
    print(f"✓ Saved anomalies to: {output_dir / 'ml_anomalies.csv'}")

    # Save model
    model_path = output_dir / "anomaly_detector.pkl"
    detector.save_model(model_path)
    print(f"✓ Saved model to: {model_path}")

    # Save top anomalies for review
    top_anomalies.to_csv(output_dir / "top_anomalies.csv", index=False)
    print(f"✓ Saved top anomalies to: {output_dir / 'top_anomalies.csv'}")
    print()

    # Step 9: Model info
    print("Step 9: Model performance summary...")
    print("-" * 70)
    true_anomalies = anomalies_df[anomalies_df["is_anomaly"] == True]  # noqa: E712
    if not true_anomalies.empty:
        avg_prob = true_anomalies["anomaly_probability"].mean()
        max_prob = true_anomalies["anomaly_probability"].max()
        min_prob = true_anomalies["anomaly_probability"].min()

        print(f"  Anomaly probability stats:")
        print(f"    Mean:    {avg_prob:.3f}")
        print(f"    Max:     {max_prob:.3f}")
        print(f"    Min:     {min_prob:.3f}")

        # Confidence breakdown
        high_confidence = (true_anomalies["anomaly_probability"] >= 0.8).sum()
        medium_confidence = (
            (true_anomalies["anomaly_probability"] >= 0.6)
            & (true_anomalies["anomaly_probability"] < 0.8)
        ).sum()
        low_confidence = (true_anomalies["anomaly_probability"] < 0.6).sum()

        print(f"\n  Confidence breakdown:")
        print(f"    High (≥0.8):   {high_confidence:3d} anomalies")
        print(f"    Medium (0.6-0.8): {medium_confidence:3d} anomalies")
        print(f"    Low (<0.6):    {low_confidence:3d} anomalies")
    else:
        print("  No anomalies to analyze")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"IFC File: {ifc_path.name}")
    print(f"Total Elements: {len(features_df)}")
    print(f"Elements with Geometry: {n_with_geometry}")
    print(f"Detected Anomalies: {n_anomalies} ({n_anomalies / max(1, n_with_geometry) * 100:.1f}%)")
    print()
    print("Model Configuration:")
    print(f"  Algorithm: Isolation Forest")
    print(f"  Contamination: {contamination}")
    print(f"  N Estimators: {detector.n_estimators}")
    print(f"  Features: {len(detector.feature_names)}")
    print()
    print("Outputs:")
    print(f"  Anomalies:      {output_dir / 'ml_anomalies.csv'}")
    print(f"  Top Anomalies:  {output_dir / 'top_anomalies.csv'}")
    print(f"  Trained Model:  {model_path}")
    print()
    print("Next steps:")
    print("  1. Review top_anomalies.csv for high-confidence detections")
    print("  2. Investigate anomalies by IFC type")
    print("  3. Use saved model for batch processing with AnomalyDetector.load_model()")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/example_ml_anomaly_detection.py <ifc_file_path>")
        sys.exit(1)

    ifc_file = Path(sys.argv[1])
    if not ifc_file.exists():
        print(f"Error: File not found: {ifc_file}")
        sys.exit(1)

    main(ifc_file)
