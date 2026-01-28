"""Unit and integration tests for metrics module."""

from pathlib import Path

import pandas as pd
import pytest

from ifcqi.metrics import build_metrics, save_metrics


@pytest.fixture
def sample_features_df() -> pd.DataFrame:
    """Create a sample features DataFrame (elements + optional geometry flag)."""
    return pd.DataFrame(
        {
            "global_id": ["id1", "id2", "id3", "id4"],
            "ifc_type": ["IfcWall", "IfcDoor", "IfcWindow", "IfcBeam"],
            "name": ["Wall 1", None, "", "Beam 1"],
            "object_type": ["LoadBearing", "Single", None, ""],
            "has_geometry": [True, False, True, True],
        }
    )


@pytest.fixture
def sample_issues_df() -> pd.DataFrame:
    """Create a sample issues DataFrame (matches checks.issues_to_dataframe schema)."""
    return pd.DataFrame(
        {
            "global_id": ["id2", "id3", "id3"],
            "ifc_type": ["IfcDoor", "IfcWindow", "IfcWindow"],
            "issue_type": ["missing_name", "missing_name", "missing_object_type"],
            "severity": ["major", "major", "minor"],
            "message": ["Element has no name", "Element has no name", "Element has no ObjectType"],
            "element_name": [None, None, ""],
        }
    )


class TestBuildMetrics:
    """Tests for build_metrics."""

    def test_build_metrics_from_dataframes(
        self, sample_features_df: pd.DataFrame, sample_issues_df: pd.DataFrame
    ) -> None:
        """Test building metrics from DataFrame inputs."""
        metrics, metrics_dfs = build_metrics(sample_features_df, sample_issues_df)

        assert metrics["total_elements"] == 4
        assert metrics["total_issues"] == 3
        assert "quality_score" in metrics
        assert 0.0 <= float(metrics["quality_score"]) <= 100.0

        assert metrics["severity_breakdown"]["major"] == 2
        assert metrics["severity_breakdown"]["minor"] == 1

        assert "metadata_completeness" in metrics
        assert "name_pct" in metrics["metadata_completeness"]
        assert "object_type_pct" in metrics["metadata_completeness"]
        assert "geometry_pct" in metrics["metadata_completeness"]

        assert "kpi" in metrics_dfs
        assert "severity" in metrics_dfs
        assert "issues_by_ifc_type" in metrics_dfs
        assert "issues_by_type" in metrics_dfs
        assert "pareto_issue_types" in metrics_dfs
        assert "pareto_issue_types" in metrics
        assert isinstance(metrics["pareto_issue_types"], list)

    def test_save_metrics_writes_json(
        self, sample_features_df: pd.DataFrame, sample_issues_df: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Test saving metrics to disk."""
        metrics, _ = build_metrics(sample_features_df, sample_issues_df)
        out_path = tmp_path / "metrics.json"
        save_metrics(metrics, out_path)

        assert out_path.exists()

        import json

        with open(out_path) as f:
            loaded = json.load(f)

        assert loaded["total_elements"] == 4
        assert loaded["total_issues"] == 3


def find_ifc_files() -> list[Path]:
    """Find all .ifc files in the examples/ifc_files directory."""
    ifc_dir = Path(__file__).parent.parent / "examples" / "ifc_files"
    if not ifc_dir.exists():
        return []
    return list(ifc_dir.glob("*.ifc"))


@pytest.mark.integration
@pytest.mark.parametrize("ifc_path", find_ifc_files(), ids=lambda p: p.name)
def test_build_metrics_end_to_end(ifc_path: Path, tmp_path: Path) -> None:
    """Integration test running the full pipeline on all IFC files.

    This test is skipped if no IFC files are found.
    """
    ifcopenshell = pytest.importorskip("ifcopenshell")
    assert ifcopenshell is not None

    if not ifc_path.exists():
        pytest.skip(f"IFC file not found: {ifc_path}")

    from ifcqi.checks import run_all_checks
    from ifcqi.features import compute_geometry_features
    from ifcqi.ifc_loader import extract_elements, load_ifc

    model = load_ifc(ifc_path)
    elements_df = extract_elements(model)
    features_df = compute_geometry_features(model, elements_df)
    issues = run_all_checks(features_df)

    metrics, metrics_dfs = build_metrics(features_df, issues)

    # Basic assertions
    assert metrics["total_elements"] == len(features_df)
    assert metrics["total_issues"] >= 0
    assert 0.0 <= float(metrics["quality_score"]) <= 100.0

    # Test saving metrics
    out_path = tmp_path / f"{ifc_path.stem}_metrics.json"
    save_metrics(metrics, out_path)
    assert out_path.exists()

    # Basic shape checks for viz-ready frames
    assert len(metrics_dfs["kpi"]) > 0
    assert set(metrics_dfs["severity"].columns) == {"severity", "count"}
    
    # Log test results for visibility
    print(f"\nTested: {ifc_path.name}")
    print(f"  Elements: {metrics['total_elements']}")
    print(f"  Issues: {metrics['total_issues']}")
    print(f"  Quality Score: {metrics['quality_score']:.1f}")
    print(f"  Metrics saved to: {out_path}")
