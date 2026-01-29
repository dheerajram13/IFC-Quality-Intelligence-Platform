"""Unit tests for visualization module."""

import pandas as pd
import plotly.graph_objects as go
import pytest

from ifcqi.viz import (
    build_issues_by_ifc_type_chart,
    build_issues_by_type_chart,
    build_pareto_chart,
    build_severity_chart,
    export_figures_to_html,
)


@pytest.fixture
def sample_severity_df() -> pd.DataFrame:
    """Sample severity breakdown DataFrame."""
    return pd.DataFrame(
        {
            "severity": ["critical", "major", "minor"],
            "count": [5, 10, 20],
        }
    )


@pytest.fixture
def sample_issues_by_ifc_type_df() -> pd.DataFrame:
    """Sample issues by IFC type DataFrame."""
    return pd.DataFrame(
        {
            "ifc_type": ["IfcWall", "IfcDoor", "IfcWindow", "IfcBeam"],
            "count": [15, 8, 12, 5],
        }
    )


@pytest.fixture
def sample_issues_by_type_df() -> pd.DataFrame:
    """Sample issues by issue type DataFrame."""
    return pd.DataFrame(
        {
            "issue_type": ["missing_name", "missing_object_type", "extreme_aspect_ratio"],
            "count": [20, 15, 10],
        }
    )


@pytest.fixture
def sample_pareto_df() -> pd.DataFrame:
    """Sample Pareto DataFrame with cumulative percentages."""
    df = pd.DataFrame(
        {
            "issue_type": ["missing_name", "missing_object_type", "extreme_aspect_ratio"],
            "count": [20, 15, 10],
        }
    )
    df["pct"] = df["count"] / df["count"].sum()
    df["cum_pct"] = df["pct"].cumsum()
    return df


class TestSeverityChart:
    """Tests for build_severity_chart."""

    def test_build_severity_chart_with_data(self, sample_severity_df: pd.DataFrame) -> None:
        """Test creating severity chart with data."""
        fig = build_severity_chart(sample_severity_df)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Issues by Severity"
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "Severity"
        assert fig.layout.yaxis.title.text == "Count"

    def test_build_severity_chart_empty(self) -> None:
        """Test creating severity chart with empty data."""
        empty_df = pd.DataFrame(columns=["severity", "count"])
        fig = build_severity_chart(empty_df)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Issues by Severity"


class TestIssuesByIfcTypeChart:
    """Tests for build_issues_by_ifc_type_chart."""

    def test_build_issues_by_ifc_type_chart_with_data(
        self, sample_issues_by_ifc_type_df: pd.DataFrame
    ) -> None:
        """Test creating issues by IFC type chart with data."""
        fig = build_issues_by_ifc_type_chart(sample_issues_by_ifc_type_df)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Issues by IFC Type"
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "IFC Type"
        assert fig.layout.yaxis.title.text == "Count"

    def test_build_issues_by_ifc_type_chart_empty(self) -> None:
        """Test creating issues by IFC type chart with empty data."""
        empty_df = pd.DataFrame(columns=["ifc_type", "count"])
        fig = build_issues_by_ifc_type_chart(empty_df)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Issues by IFC Type"


class TestIssuesByTypeChart:
    """Tests for build_issues_by_type_chart."""

    def test_build_issues_by_type_chart_with_data(
        self, sample_issues_by_type_df: pd.DataFrame
    ) -> None:
        """Test creating issues by issue type chart with data."""
        fig = build_issues_by_type_chart(sample_issues_by_type_df)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Issues by Issue Type"
        assert len(fig.data) > 0
        assert fig.layout.xaxis.title.text == "Issue Type"
        assert fig.layout.yaxis.title.text == "Count"

    def test_build_issues_by_type_chart_empty(self) -> None:
        """Test creating issues by issue type chart with empty data."""
        empty_df = pd.DataFrame(columns=["issue_type", "count"])
        fig = build_issues_by_type_chart(empty_df)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Issues by Issue Type"


class TestParetoChart:
    """Tests for build_pareto_chart."""

    def test_build_pareto_chart_with_data(self, sample_pareto_df: pd.DataFrame) -> None:
        """Test creating Pareto chart with data."""
        fig = build_pareto_chart(sample_pareto_df)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Pareto of Issue Types"
        # Should have 2 traces: bar and line
        assert len(fig.data) == 2
        assert fig.layout.xaxis.title.text == "Issue Type"

        # Check dual y-axes
        assert fig.layout.yaxis.title.text == "Count"
        assert fig.layout.yaxis2.title.text == "Cumulative %"
        assert list(fig.layout.yaxis2.range) == [0, 100]

    def test_build_pareto_chart_empty(self) -> None:
        """Test creating Pareto chart with empty data."""
        empty_df = pd.DataFrame(columns=["issue_type", "count", "pct", "cum_pct"])
        fig = build_pareto_chart(empty_df)

        assert isinstance(fig, go.Figure)
        assert fig.layout.title.text == "Pareto of Issue Types"


class TestExportFiguresToHtml:
    """Tests for export_figures_to_html."""

    def test_export_single_figure(
        self, sample_severity_df: pd.DataFrame, tmp_path
    ) -> None:
        """Test exporting a single figure to HTML."""
        fig = build_severity_chart(sample_severity_df)
        output_path = tmp_path / "report.html"

        export_figures_to_html({"Severity Chart": fig}, output_path)

        assert output_path.exists()
        html_content = output_path.read_text(encoding="utf-8")
        assert "<html>" in html_content
        assert "<h2>Severity Chart</h2>" in html_content
        assert "plotly" in html_content.lower()

    def test_export_multiple_figures(
        self,
        sample_severity_df: pd.DataFrame,
        sample_issues_by_type_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        """Test exporting multiple figures to HTML."""
        fig1 = build_severity_chart(sample_severity_df)
        fig2 = build_issues_by_type_chart(sample_issues_by_type_df)
        output_path = tmp_path / "multi_report.html"

        export_figures_to_html(
            {
                "Severity Chart": fig1,
                "Issue Types Chart": fig2,
            },
            output_path,
        )

        assert output_path.exists()
        html_content = output_path.read_text(encoding="utf-8")
        assert "<h2>Severity Chart</h2>" in html_content
        assert "<h2>Issue Types Chart</h2>" in html_content

    def test_export_creates_parent_directory(
        self, sample_severity_df: pd.DataFrame, tmp_path
    ) -> None:
        """Test that export creates parent directories if they don't exist."""
        fig = build_severity_chart(sample_severity_df)
        output_path = tmp_path / "nested" / "dir" / "report.html"

        export_figures_to_html({"Test Chart": fig}, output_path)

        assert output_path.exists()
        assert output_path.parent.exists()


class TestIntegration:
    """Integration tests for viz module."""

    def test_full_visualization_pipeline(
        self,
        sample_severity_df: pd.DataFrame,
        sample_issues_by_ifc_type_df: pd.DataFrame,
        sample_issues_by_type_df: pd.DataFrame,
        sample_pareto_df: pd.DataFrame,
        tmp_path,
    ) -> None:
        """Test creating all charts and exporting to HTML."""
        figures = {
            "Severity Breakdown": build_severity_chart(sample_severity_df),
            "Issues by IFC Type": build_issues_by_ifc_type_chart(sample_issues_by_ifc_type_df),
            "Issues by Type": build_issues_by_type_chart(sample_issues_by_type_df),
            "Pareto Analysis": build_pareto_chart(sample_pareto_df),
        }

        output_path = tmp_path / "full_report.html"
        export_figures_to_html(figures, output_path)

        assert output_path.exists()
        html_content = output_path.read_text(encoding="utf-8")

        # Check all charts are present
        assert "<h2>Severity Breakdown</h2>" in html_content
        assert "<h2>Issues by IFC Type</h2>" in html_content
        assert "<h2>Issues by Type</h2>" in html_content
        assert "<h2>Pareto Analysis</h2>" in html_content
