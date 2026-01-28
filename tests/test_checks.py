"""Unit tests for quality checks module."""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from ifcqi.checks import (
    Issue,
    Severity,
    check_coordinate_anomalies,
    check_degenerate_geometry,
    check_duplicate_global_ids,
    check_extreme_aspect_ratios,
    check_extreme_dimensions,
    check_missing_name,
    check_missing_object_type,
    get_issues_summary,
    issues_to_dataframe,
    run_all_checks,
)


@pytest.fixture
def sample_elements_df():
    """Create a sample elements DataFrame for testing."""
    return pd.DataFrame(
        {
            "global_id": ["id1", "id2", "id3", "id4"],
            "ifc_type": ["IfcWall", "IfcDoor", "IfcWindow", "IfcBeam"],
            "name": ["Wall 1", None, "", "Beam 1"],
            "object_type": ["LoadBearing", "Single", None, ""],
        }
    )


@pytest.fixture
def sample_geometry_df():
    """Create a sample geometry DataFrame for testing."""
    return pd.DataFrame(
        {
            "global_id": ["id1", "id2", "id3", "id4", "id5"],
            "ifc_type": ["IfcWall", "IfcDoor", "IfcWindow", "IfcBeam", "IfcSlab"],
            "name": ["Wall 1", "Door 1", "Window 1", "Beam 1", "Slab 1"],
            "object_type": ["LoadBearing", "Single", "Fixed", "Steel", "Concrete"],
            "has_geometry": [True, True, True, True, True],
            "dim_x": [5.0, 1.0, 0.0005, 15000.0, 10.0],  # Normal, normal, degenerate, extreme, normal
            "dim_y": [3.0, 2.0, 1.5, 2.0, 10.0],
            "dim_z": [0.3, 2.5, 1.2, 0.5, 0.3],
            "centroid_x": [10.0, 5.0, 8.0, 120000.0, 15.0],  # Normal, normal, normal, anomaly, normal
            "centroid_y": [20.0, 15.0, 18.0, 5.0, 25.0],
            "centroid_z": [1.5, 1.2, 1.0, 2.0, 1.8],
            "aspect_ratio_xy": [1.67, 0.5, 0.0003, 7500.0, 1.0],  # Normal, normal, extreme, extreme, normal
            "aspect_ratio_yz": [10.0, 0.8, 1.25, 4.0, 33.33],
            "aspect_ratio_xz": [16.67, 0.4, 0.0004, 30000.0, 33.33],
        }
    )


class TestIssue:
    """Tests for Issue dataclass."""

    def test_issue_creation(self):
        """Test creating an Issue object."""
        issue = Issue(
            global_id="test-id",
            ifc_type="IfcWall",
            issue_type="missing_name",
            severity=Severity.MAJOR,
            message="Element has no name",
            element_name=None,
        )

        assert issue.global_id == "test-id"
        assert issue.ifc_type == "IfcWall"
        assert issue.severity == Severity.MAJOR

    def test_issue_to_dict(self):
        """Test converting Issue to dictionary."""
        issue = Issue(
            global_id="test-id",
            ifc_type="IfcWall",
            issue_type="missing_name",
            severity=Severity.MAJOR,
            message="Element has no name",
            element_name="Wall 1",
        )

        issue_dict = issue.to_dict()

        assert issue_dict["global_id"] == "test-id"
        assert issue_dict["severity"] == "major"
        assert issue_dict["element_name"] == "Wall 1"


class TestCheckMissingName:
    """Tests for check_missing_name function."""

    def test_check_missing_name(self, sample_elements_df):
        """Test detecting elements with missing names."""
        issues = check_missing_name(sample_elements_df)

        assert len(issues) == 2  # id2 (None) and id3 (empty string)
        assert all(issue.issue_type == "missing_name" for issue in issues)
        assert all(issue.severity == Severity.MAJOR for issue in issues)

    def test_check_missing_name_all_have_names(self):
        """Test when all elements have names."""
        df = pd.DataFrame(
            {
                "global_id": ["id1", "id2"],
                "ifc_type": ["IfcWall", "IfcDoor"],
                "name": ["Wall 1", "Door 1"],
            }
        )

        issues = check_missing_name(df)

        assert len(issues) == 0

    def test_check_missing_name_empty_df(self):
        """Test with empty DataFrame."""
        df = pd.DataFrame(columns=["global_id", "ifc_type", "name"])

        issues = check_missing_name(df)

        assert len(issues) == 0


class TestCheckMissingObjectType:
    """Tests for check_missing_object_type function."""

    def test_check_missing_object_type(self, sample_elements_df):
        """Test detecting elements with missing object types."""
        issues = check_missing_object_type(sample_elements_df)

        assert len(issues) == 2  # id3 (None) and id4 (empty string)
        assert all(issue.issue_type == "missing_object_type" for issue in issues)
        assert all(issue.severity == Severity.MINOR for issue in issues)

    def test_check_missing_object_type_all_have_types(self):
        """Test when all elements have object types."""
        df = pd.DataFrame(
            {
                "global_id": ["id1", "id2"],
                "ifc_type": ["IfcWall", "IfcDoor"],
                "name": ["Wall 1", "Door 1"],
                "object_type": ["LoadBearing", "Single"],
            }
        )

        issues = check_missing_object_type(df)

        assert len(issues) == 0


class TestCheckDuplicateGlobalIds:
    """Tests for check_duplicate_global_ids function."""

    def test_check_duplicate_global_ids(self):
        """Test detecting duplicate GlobalIds."""
        df = pd.DataFrame(
            {
                "global_id": ["id1", "id2", "id2", "id3", "id3", "id3"],
                "ifc_type": ["IfcWall", "IfcDoor", "IfcDoor", "IfcWindow", "IfcWindow", "IfcWindow"],
                "name": ["Wall", "Door1", "Door2", "Win1", "Win2", "Win3"],
            }
        )

        issues = check_duplicate_global_ids(df)

        # Should have 5 issues (2 for id2, 3 for id3)
        assert len(issues) == 5
        assert all(issue.issue_type == "duplicate_global_id" for issue in issues)
        assert all(issue.severity == Severity.CRITICAL for issue in issues)

        # Check messages indicate correct counts
        id2_issues = [i for i in issues if i.global_id == "id2"]
        assert all("appears 2 times" in i.message for i in id2_issues)

        id3_issues = [i for i in issues if i.global_id == "id3"]
        assert all("appears 3 times" in i.message for i in id3_issues)

    def test_check_duplicate_global_ids_no_duplicates(self):
        """Test when all GlobalIds are unique."""
        df = pd.DataFrame(
            {
                "global_id": ["id1", "id2", "id3"],
                "ifc_type": ["IfcWall", "IfcDoor", "IfcWindow"],
                "name": ["Wall", "Door", "Window"],
            }
        )

        issues = check_duplicate_global_ids(df)

        assert len(issues) == 0


class TestCheckDegenerateGeometry:
    """Tests for check_degenerate_geometry function."""

    def test_check_degenerate_geometry(self, sample_geometry_df):
        """Test detecting degenerate geometry."""
        issues = check_degenerate_geometry(sample_geometry_df, min_dimension=0.001)

        # Should detect id3 with dim_x = 0.0005 < 0.001
        assert len(issues) == 1
        assert issues[0].global_id == "id3"
        assert issues[0].issue_type == "degenerate_geometry"
        assert issues[0].severity == Severity.MAJOR

    def test_check_degenerate_geometry_custom_threshold(self, sample_geometry_df):
        """Test with custom threshold."""
        issues = check_degenerate_geometry(sample_geometry_df, min_dimension=0.5)

        # Should detect id3 (0.0005), id1 (0.3), and id5 (0.3)
        assert len(issues) == 3

    def test_check_degenerate_geometry_no_geometry_column(self):
        """Test when DataFrame missing has_geometry column."""
        df = pd.DataFrame({"global_id": ["id1"], "ifc_type": ["IfcWall"]})

        issues = check_degenerate_geometry(df)

        assert len(issues) == 0


class TestCheckExtremeDimensions:
    """Tests for check_extreme_dimensions function."""

    def test_check_extreme_dimensions(self, sample_geometry_df):
        """Test detecting extreme dimensions."""
        issues = check_extreme_dimensions(sample_geometry_df, max_dimension=10000.0)

        # Should detect id4 with dim_x = 15000.0 > 10000.0
        assert len(issues) == 1
        assert issues[0].global_id == "id4"
        assert issues[0].issue_type == "extreme_dimension"
        assert issues[0].severity == Severity.CRITICAL

    def test_check_extreme_dimensions_custom_threshold(self, sample_geometry_df):
        """Test with custom threshold."""
        issues = check_extreme_dimensions(sample_geometry_df, max_dimension=5.0)

        # Should detect multiple elements
        assert len(issues) > 1


class TestCheckCoordinateAnomalies:
    """Tests for check_coordinate_anomalies function."""

    def test_check_coordinate_anomalies(self, sample_geometry_df):
        """Test detecting coordinate anomalies."""
        issues = check_coordinate_anomalies(sample_geometry_df, max_distance=100000.0)

        # Should detect id4 with centroid_x = 120000.0 (distance > 100000)
        assert len(issues) == 1
        assert issues[0].global_id == "id4"
        assert issues[0].issue_type == "coordinate_anomaly"
        assert issues[0].severity == Severity.MINOR

    def test_check_coordinate_anomalies_no_centroid_columns(self):
        """Test when DataFrame missing centroid columns."""
        df = pd.DataFrame(
            {
                "global_id": ["id1"],
                "ifc_type": ["IfcWall"],
                "has_geometry": [True],
            }
        )

        issues = check_coordinate_anomalies(df)

        assert len(issues) == 0


class TestCheckExtremeAspectRatios:
    """Tests for check_extreme_aspect_ratios function."""

    def test_check_extreme_aspect_ratios(self, sample_geometry_df):
        """Test detecting extreme aspect ratios."""
        issues = check_extreme_aspect_ratios(sample_geometry_df, max_ratio=100.0)

        # Should detect id3 (0.0003, 0.0004) and id4 (7500.0, 30000.0)
        assert len(issues) >= 2

        # Check that both very small and very large ratios are detected
        ratios = [float(i.message.split("=")[1].split()[0]) for i in issues]
        assert any(r < 1 / 100.0 for r in ratios)  # Very small ratio
        assert any(r > 100.0 for r in ratios)  # Very large ratio

    def test_check_extreme_aspect_ratios_custom_threshold(self, sample_geometry_df):
        """Test with custom threshold."""
        issues = check_extreme_aspect_ratios(sample_geometry_df, max_ratio=50.0)

        # More issues with stricter threshold
        assert len(issues) > 0


class TestRunAllChecks:
    """Tests for run_all_checks function."""

    def test_run_all_checks_complete_df(self, sample_geometry_df):
        """Test running all checks on complete DataFrame."""
        issues = run_all_checks(sample_geometry_df)

        # Should find multiple types of issues
        issue_types = set(issue.issue_type for issue in issues)
        assert len(issue_types) > 0

    def test_run_all_checks_no_geometry(self, sample_elements_df):
        """Test running checks on DataFrame without geometry."""
        issues = run_all_checks(sample_elements_df)

        # Should only find metadata issues (no geometry checks)
        issue_types = set(issue.issue_type for issue in issues)
        assert "degenerate_geometry" not in issue_types
        assert "extreme_dimension" not in issue_types

    def test_run_all_checks_clean_data(self):
        """Test with clean data (no issues)."""
        df = pd.DataFrame(
            {
                "global_id": ["id1", "id2"],
                "ifc_type": ["IfcWall", "IfcDoor"],
                "name": ["Wall 1", "Door 1"],
                "object_type": ["LoadBearing", "Single"],
                "has_geometry": [True, True],
                "dim_x": [5.0, 1.0],
                "dim_y": [3.0, 2.0],
                "dim_z": [0.3, 2.5],
                "centroid_x": [10.0, 5.0],
                "centroid_y": [20.0, 15.0],
                "centroid_z": [1.5, 1.2],
                "aspect_ratio_xy": [1.67, 0.5],
                "aspect_ratio_yz": [10.0, 0.8],
                "aspect_ratio_xz": [16.67, 0.4],
            }
        )

        issues = run_all_checks(df)

        assert len(issues) == 0


class TestIssuesToDataFrame:
    """Tests for issues_to_dataframe function."""

    def test_issues_to_dataframe(self):
        """Test converting issues list to DataFrame."""
        issues = [
            Issue(
                global_id="id1",
                ifc_type="IfcWall",
                issue_type="missing_name",
                severity=Severity.MAJOR,
                message="No name",
                element_name=None,
            ),
            Issue(
                global_id="id2",
                ifc_type="IfcDoor",
                issue_type="missing_object_type",
                severity=Severity.MINOR,
                message="No type",
                element_name="Door 1",
            ),
        ]

        df = issues_to_dataframe(issues)

        assert len(df) == 2
        assert list(df.columns) == [
            "global_id",
            "ifc_type",
            "issue_type",
            "severity",
            "message",
            "element_name",
        ]
        assert df.iloc[0]["global_id"] == "id1"
        assert df.iloc[1]["severity"] == "minor"

    def test_issues_to_dataframe_empty(self):
        """Test with empty issues list."""
        df = issues_to_dataframe([])

        assert len(df) == 0
        assert list(df.columns) == [
            "global_id",
            "ifc_type",
            "issue_type",
            "severity",
            "message",
            "element_name",
        ]


class TestGetIssuesSummary:
    """Tests for get_issues_summary function."""

    def test_get_issues_summary(self):
        """Test generating issues summary."""
        issues = [
            Issue("id1", "IfcWall", "missing_name", Severity.MAJOR, "No name"),
            Issue("id2", "IfcDoor", "missing_name", Severity.MAJOR, "No name"),
            Issue("id3", "IfcWindow", "missing_object_type", Severity.MINOR, "No type"),
            Issue("id4", "IfcBeam", "duplicate_global_id", Severity.CRITICAL, "Duplicate"),
        ]

        summary = get_issues_summary(issues)

        assert summary["total_issues"] == 4
        assert summary["critical_count"] == 1
        assert summary["major_count"] == 2
        assert summary["minor_count"] == 1
        assert summary["by_type"]["missing_name"] == 2
        assert summary["by_severity"]["major"] == 2

    def test_get_issues_summary_empty(self):
        """Test summary with no issues."""
        summary = get_issues_summary([])

        assert summary["total_issues"] == 0
        assert summary["critical_count"] == 0
        assert summary["major_count"] == 0
        assert summary["minor_count"] == 0
        assert summary["by_type"] == {}
        assert summary["by_severity"] == {}
