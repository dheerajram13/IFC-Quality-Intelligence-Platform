"""Unit and integration tests for ML anomaly detection module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ifcqi.ml import (
    AnomalyDetector,
    add_derived_features,
    anomalies_to_dataframe,
    detect_anomalies,
    handle_missing_values,
    preprocess_features,
    scale_features,
    select_ml_features,
)


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create a sample features DataFrame with typical geometry data."""
    return pd.DataFrame(
        {
            "global_id": ["id1", "id2", "id3", "id4", "id5"],
            "ifc_type": ["IfcWall", "IfcDoor", "IfcWindow", "IfcBeam", "IfcSlab"],
            "name": ["Wall 1", "Door 1", "Window 1", "Beam 1", "Slab 1"],
            "has_geometry": [True, True, True, True, True],
            "dim_x": [5.0, 1.0, 0.8, 10.0, 8.0],
            "dim_y": [3.0, 2.0, 1.5, 0.3, 8.0],
            "dim_z": [0.3, 2.5, 1.2, 0.5, 0.25],
            "centroid_x": [10.0, 5.0, 8.0, 15.0, 20.0],
            "centroid_y": [20.0, 15.0, 18.0, 12.0, 25.0],
            "centroid_z": [1.5, 1.2, 1.0, 2.0, 1.8],
            "aspect_ratio_xy": [1.67, 0.5, 0.53, 33.33, 1.0],
            "aspect_ratio_yz": [10.0, 0.8, 1.25, 0.6, 32.0],
            "aspect_ratio_xz": [16.67, 0.4, 0.67, 20.0, 32.0],
        }
    )


@pytest.fixture
def sample_features_with_anomaly() -> pd.DataFrame:
    """Create features DataFrame with one clear anomaly."""
    return pd.DataFrame(
        {
            "global_id": ["id1", "id2", "id3", "id4", "id5"],
            "ifc_type": ["IfcWall", "IfcWall", "IfcWall", "IfcWall", "IfcWall"],
            "name": ["Wall 1", "Wall 2", "Wall 3", "Wall 4", "ANOMALY"],
            "has_geometry": [True, True, True, True, True],
            # First 4 are similar, id5 is very different (anomaly)
            "dim_x": [5.0, 5.1, 4.9, 5.2, 100.0],  # Anomaly: 100.0
            "dim_y": [3.0, 3.1, 2.9, 3.2, 0.01],  # Anomaly: 0.01
            "dim_z": [0.3, 0.31, 0.29, 0.32, 50.0],  # Anomaly: 50.0
            "centroid_x": [10.0, 10.5, 9.5, 10.2, 10.0],
            "centroid_y": [20.0, 20.2, 19.8, 20.1, 20.0],
            "centroid_z": [1.5, 1.6, 1.4, 1.55, 1.5],
            "aspect_ratio_xy": [1.67, 1.65, 1.69, 1.63, 10000.0],  # Anomaly
            "aspect_ratio_yz": [10.0, 10.0, 10.0, 10.0, 0.0002],  # Anomaly
            "aspect_ratio_xz": [16.67, 16.5, 16.8, 16.3, 2.0],
        }
    )


class TestSelectMLFeatures:
    """Tests for select_ml_features."""

    def test_select_ml_features(self, sample_features_df: pd.DataFrame) -> None:
        """Test basic feature selection."""
        ml_df = select_ml_features(sample_features_df)

        assert len(ml_df) == 5  # All rows have geometry
        assert "dim_x" in ml_df.columns
        assert "centroid_z" in ml_df.columns
        assert "aspect_ratio_xy" in ml_df.columns
        # Metadata columns should not be included
        assert "name" not in ml_df.columns
        assert "global_id" not in ml_df.columns

    def test_select_ml_features_filters_no_geometry(self) -> None:
        """Test that elements without geometry are filtered out."""
        df = pd.DataFrame(
            {
                "global_id": ["id1", "id2"],
                "ifc_type": ["IfcWall", "IfcDoor"],
                "has_geometry": [True, False],
                "dim_x": [5.0, np.nan],
                "dim_y": [3.0, np.nan],
                "dim_z": [0.3, np.nan],
            }
        )

        ml_df = select_ml_features(df)
        assert len(ml_df) == 1  # Only id1 has geometry

    def test_select_ml_features_handles_inf(self, sample_features_df: pd.DataFrame) -> None:
        """Test that infinite values are replaced with NaN."""
        df = sample_features_df.copy()
        df.loc[0, "dim_x"] = np.inf
        df.loc[1, "aspect_ratio_xy"] = -np.inf

        ml_df = select_ml_features(df)

        assert pd.isna(ml_df.iloc[0]["dim_x"])
        assert pd.isna(ml_df.iloc[1]["aspect_ratio_xy"])


class TestHandleMissingValues:
    """Tests for handle_missing_values."""

    def test_handle_missing_median(self) -> None:
        """Test median imputation."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0], "b": [10.0, np.nan, 30.0, 40.0]})

        filled = handle_missing_values(df, strategy="median")

        assert not filled.isnull().any().any()
        assert filled["a"].iloc[2] == 2.0  # Median of [1, 2, 4]
        assert filled["b"].iloc[1] == 30.0  # Median of [10, 30, 40]

    def test_handle_missing_mean(self) -> None:
        """Test mean imputation."""
        df = pd.DataFrame({"a": [1.0, 2.0, np.nan, 4.0]})

        filled = handle_missing_values(df, strategy="mean")

        assert not filled.isnull().any().any()
        assert filled["a"].iloc[2] == pytest.approx(2.333, abs=0.01)  # Mean of [1, 2, 4]

    def test_handle_missing_constant(self) -> None:
        """Test constant imputation."""
        df = pd.DataFrame({"a": [1.0, np.nan, 3.0]})

        filled = handle_missing_values(df, strategy="constant", fill_value=999.0)

        assert filled["a"].iloc[1] == 999.0

    def test_handle_missing_empty_df(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame()
        filled = handle_missing_values(df)
        assert filled.empty


class TestScaleFeatures:
    """Tests for scale_features."""

    def test_scale_features_robust(self) -> None:
        """Test robust scaling."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})

        scaled, scaler = scale_features(df, scaler_type="robust")

        assert scaled.shape == df.shape
        assert scaler is not None
        # Check that values are scaled (median ~ 0, IQR ~ 1)
        assert abs(scaled["a"].median()) < 0.01

    def test_scale_features_standard(self) -> None:
        """Test standard scaling."""
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})

        scaled, scaler = scale_features(df, scaler_type="standard")

        assert scaled.shape == df.shape
        # Standard scaled data should have mean ~ 0, std ~ 1 (with ddof=0)
        assert abs(scaled["a"].mean()) < 0.01
        # pandas std() uses ddof=1 by default, sklearn uses ddof=0
        assert abs(scaled["a"].std(ddof=0) - 1.0) < 0.01

    def test_scale_features_with_pretrained_scaler(self) -> None:
        """Test using a pre-fitted scaler."""
        df_train = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        df_test = pd.DataFrame({"a": [2.5, 3.5]})

        # Fit scaler on training data
        _, scaler = scale_features(df_train, scaler_type="robust")

        # Use fitted scaler on test data
        scaled_test, _ = scale_features(df_test, scaler=scaler)

        assert len(scaled_test) == 2


class TestPreprocessFeatures:
    """Tests for preprocess_features (full pipeline)."""

    def test_preprocess_features_full_pipeline(self, sample_features_df: pd.DataFrame) -> None:
        """Test complete preprocessing pipeline."""
        processed, indices, scaler = preprocess_features(sample_features_df)

        assert len(processed) == 5
        assert len(indices) == 5
        assert scaler is not None
        # Data should be scaled (values around 0)
        assert abs(processed.mean().mean()) < 2.0

    def test_preprocess_features_with_pretrained_scaler(
        self, sample_features_df: pd.DataFrame
    ) -> None:
        """Test using pre-fitted scaler in preprocessing."""
        # First preprocessing to get scaler
        _, _, scaler = preprocess_features(sample_features_df)

        # Second preprocessing with same scaler
        processed2, _, _ = preprocess_features(sample_features_df, scaler=scaler)

        assert len(processed2) == 5


class TestAddDerivedFeatures:
    """Tests for add_derived_features."""

    def test_add_derived_features(self) -> None:
        """Test adding derived features."""
        df = pd.DataFrame(
            {
                "dim_x": [2.0, 3.0],
                "dim_y": [3.0, 4.0],
                "dim_z": [4.0, 5.0],
                "centroid_x": [10.0, 20.0],
                "centroid_y": [10.0, 20.0],
                "centroid_z": [10.0, 20.0],
                "aspect_ratio_xy": [0.67, 0.75],
                "aspect_ratio_yz": [0.75, 0.8],
                "aspect_ratio_xz": [0.5, 0.6],
            }
        )

        derived = add_derived_features(df)

        # Check new features were added
        assert "volume" in derived.columns
        assert "surface_area" in derived.columns
        assert "distance_from_origin" in derived.columns
        assert "max_dimension" in derived.columns
        assert "min_dimension" in derived.columns

        # Check volume calculation
        assert derived["volume"].iloc[0] == 2.0 * 3.0 * 4.0  # 24.0


class TestAnomalyDetector:
    """Tests for AnomalyDetector class."""

    def test_anomaly_detector_fit(self, sample_features_df: pd.DataFrame) -> None:
        """Test training anomaly detector."""
        detector = AnomalyDetector(contamination=0.1, random_state=42)
        detector.fit(sample_features_df)

        assert detector.is_fitted
        assert detector.model is not None
        assert detector.scaler is not None
        assert len(detector.feature_names) > 0

    def test_anomaly_detector_predict(self, sample_features_with_anomaly: pd.DataFrame) -> None:
        """Test anomaly prediction."""
        detector = AnomalyDetector(contamination=0.2, random_state=42)
        detector.fit(sample_features_with_anomaly)

        results = detector.predict(sample_features_with_anomaly)

        assert len(results) == 5
        assert all(hasattr(r, "global_id") for r in results)
        assert all(hasattr(r, "is_anomaly") for r in results)
        assert all(hasattr(r, "anomaly_score") for r in results)
        assert all(hasattr(r, "anomaly_probability") for r in results)

        # Check that id5 (the clear anomaly) has high probability
        anomaly_result = next(r for r in results if r.global_id == "id5")
        assert anomaly_result.anomaly_probability > 0.5

    def test_anomaly_detector_get_anomalies(
        self, sample_features_with_anomaly: pd.DataFrame
    ) -> None:
        """Test get_anomalies method."""
        detector = AnomalyDetector(contamination=0.2, random_state=42)
        detector.fit(sample_features_with_anomaly)

        anomalies_df = detector.get_anomalies(sample_features_with_anomaly, threshold=0.5)

        assert isinstance(anomalies_df, pd.DataFrame)
        assert "global_id" in anomalies_df.columns
        assert "anomaly_probability" in anomalies_df.columns
        # All returned anomalies should have probability >= threshold
        assert all(anomalies_df["anomaly_probability"] >= 0.5)

    def test_anomaly_detector_save_load(
        self, sample_features_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test saving and loading model."""
        # Train and save
        detector = AnomalyDetector(contamination=0.1, random_state=42)
        detector.fit(sample_features_df)

        model_path = tmp_path / "model.pkl"
        detector.save_model(model_path)

        assert model_path.exists()

        # Load and verify
        loaded_detector = AnomalyDetector.load_model(model_path)

        assert loaded_detector.is_fitted
        assert loaded_detector.contamination == 0.1
        assert loaded_detector.feature_names == detector.feature_names

        # Test that loaded model can predict
        results = loaded_detector.predict(sample_features_df)
        assert len(results) == 5

    def test_anomaly_detector_not_fitted_error(self, sample_features_df: pd.DataFrame) -> None:
        """Test that predict raises error if model not fitted."""
        detector = AnomalyDetector()

        with pytest.raises(ValueError, match="not fitted"):
            detector.predict(sample_features_df)


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_detect_anomalies(self, sample_features_with_anomaly: pd.DataFrame) -> None:
        """Test detect_anomalies convenience function."""
        results, detector = detect_anomalies(
            sample_features_with_anomaly, contamination=0.2, random_state=42
        )

        assert len(results) == 5
        assert detector.is_fitted

    def test_anomalies_to_dataframe(self, sample_features_with_anomaly: pd.DataFrame) -> None:
        """Test converting anomaly results to DataFrame."""
        results, _ = detect_anomalies(sample_features_with_anomaly, random_state=42)

        df = anomalies_to_dataframe(results)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "global_id" in df.columns
        assert "is_anomaly" in df.columns
        assert "anomaly_score" in df.columns
        assert "anomaly_probability" in df.columns

    def test_anomalies_to_dataframe_empty(self) -> None:
        """Test with empty results list."""
        df = anomalies_to_dataframe([])

        assert df.empty
        assert "global_id" in df.columns


def find_ifc_files() -> list[Path]:
    """Find all .ifc files in the examples/ifc_files directory."""
    ifc_dir = Path(__file__).parent.parent / "examples" / "ifc_files"
    if not ifc_dir.exists():
        return []
    return list(ifc_dir.glob("*.ifc"))


@pytest.mark.integration
@pytest.mark.parametrize("ifc_path", find_ifc_files(), ids=lambda p: p.name)
def test_anomaly_detection_end_to_end(ifc_path: Path) -> None:
    """Integration test: full anomaly detection pipeline on real IFC files."""
    ifcopenshell = pytest.importorskip("ifcopenshell")
    assert ifcopenshell is not None

    if not ifc_path.exists():
        pytest.skip(f"IFC file not found: {ifc_path}")

    from ifcqi.features import compute_geometry_features
    from ifcqi.ifc_loader import extract_elements, load_ifc

    # Load and extract features
    model = load_ifc(ifc_path)
    elements_df = extract_elements(model)
    features_df = compute_geometry_features(model, elements_df)

    # Detect anomalies
    results, detector = detect_anomalies(features_df, contamination=0.05, random_state=42)

    # Basic assertions
    assert len(results) > 0
    assert detector.is_fitted

    # Check result structure
    for result in results[:5]:  # Check first 5
        assert result.global_id
        assert result.ifc_type
        assert isinstance(result.is_anomaly, (bool, np.bool_))
        assert isinstance(result.anomaly_score, (float, np.floating))
        assert 0.0 <= result.anomaly_probability <= 1.0

    # Log results
    n_anomalies = sum(1 for r in results if r.is_anomaly)
    print(f"\nTested: {ifc_path.name}")
    print(f"  Total elements: {len(results)}")
    print(f"  Detected anomalies: {n_anomalies}")
    print(f"  Anomaly rate: {n_anomalies / max(1, len(results)) * 100:.1f}%")
