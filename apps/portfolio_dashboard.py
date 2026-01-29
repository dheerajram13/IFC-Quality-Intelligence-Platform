"""Streamlit dashboard for IFC Quality Intelligence - Portfolio Mode.

Supports two modes:
1. Portfolio Mode - Overview of multiple models with prioritization
2. Single Model Mode - Detailed analysis of individual models
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ifcqi.viz import (
    build_issues_by_ifc_type_chart,
    build_issues_by_type_chart,
    build_pareto_chart,
    build_severity_chart,
)


def load_portfolio_data(portfolio_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load portfolio summary and metrics."""
    summary_path = portfolio_dir / "portfolio_summary.csv"
    metrics_path = portfolio_dir / "portfolio_metrics.json"

    if not summary_path.exists():
        raise FileNotFoundError(f"Portfolio summary not found: {summary_path}")

    summary_df = pd.read_csv(summary_path)

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return summary_df, metrics


def load_model_data(model_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Load individual model data."""
    features_path = model_dir / "features.parquet"
    issues_path = model_dir / "issues.csv"
    metrics_path = model_dir / "metrics.json"

    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    features_df = pd.read_parquet(features_path)

    if issues_path.exists():
        issues_df = pd.read_csv(issues_path)
    else:
        issues_df = pd.DataFrame()

    if metrics_path.exists():
        with open(metrics_path) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return features_df, issues_df, metrics


def build_portfolio_quality_chart(summary_df: pd.DataFrame) -> go.Figure:
    """Bar chart of quality scores by model (sorted)."""
    success_df = summary_df[summary_df["status"] == "success"].copy()

    if success_df.empty:
        return px.bar(title="Quality Score by Model")

    success_df = success_df.sort_values("quality_score", ascending=True)

    fig = px.bar(
        success_df,
        x="quality_score",
        y="model_name",
        orientation="h",
        title="Quality Score by Model",
        color="quality_score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
    )

    fig.update_layout(
        xaxis_title="Quality Score",
        yaxis_title="Model",
        height=max(400, len(success_df) * 30),
    )

    return fig


def build_critical_issues_chart(summary_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Bar chart of top N models by critical issues."""
    success_df = summary_df[summary_df["status"] == "success"].copy()

    if success_df.empty:
        return px.bar(title=f"Top {top_n} Models by Critical Issues")

    # Filter to only models with critical issues
    critical_df = success_df[success_df["critical_issues"] > 0]

    if critical_df.empty:
        # Show top by total issues instead
        top_df = success_df.nlargest(top_n, "total_issues")
        title = f"Top {top_n} Models by Total Issues (No Critical Issues)"
        y_col = "total_issues"
    else:
        top_df = critical_df.nlargest(top_n, "critical_issues")
        title = f"Top {top_n} Models by Critical Issues"
        y_col = "critical_issues"

    top_df = top_df.sort_values(y_col, ascending=True)

    fig = px.bar(
        top_df,
        x=y_col,
        y="model_name",
        orientation="h",
        title=title,
        color=y_col,
        color_continuous_scale="Reds",
    )

    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="Model",
        height=max(300, len(top_df) * 40),
    )

    return fig


def build_issue_rate_distribution(summary_df: pd.DataFrame) -> go.Figure:
    """Histogram/boxplot of issue rate per 1k distribution."""
    success_df = summary_df[summary_df["status"] == "success"].copy()

    if success_df.empty:
        return px.histogram(title="Issue Rate per 1,000 Distribution")

    fig = go.Figure()

    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=success_df["issue_rate_per_1k"],
            name="Distribution",
            nbinsx=20,
            marker_color="lightblue",
        )
    )

    # Add vertical lines for p50 and p95
    p50 = success_df["issue_rate_per_1k"].median()
    p95 = success_df["issue_rate_per_1k"].quantile(0.95)

    fig.add_vline(x=p50, line_dash="dash", line_color="green", annotation_text=f"P50: {p50:.1f}")
    fig.add_vline(x=p95, line_dash="dash", line_color="red", annotation_text=f"P95: {p95:.1f}")

    fig.update_layout(
        title="Issue Rate per 1,000 Distribution",
        xaxis_title="Issue Rate per 1,000 Elements",
        yaxis_title="Count of Models",
        showlegend=False,
    )

    return fig


def build_portfolio_pareto(portfolio_dir: Path) -> go.Figure:
    """Pareto chart of top issue types across portfolio."""
    # Aggregate all issues from per-model issues.csv files
    all_issues = []

    for model_dir in portfolio_dir.iterdir():
        if model_dir.is_dir() and (model_dir / "issues.csv").exists():
            try:
                issues_df = pd.read_csv(model_dir / "issues.csv")
                all_issues.append(issues_df)
            except Exception:
                continue

    if not all_issues:
        return px.bar(title="Portfolio-wide Issue Types (Pareto)")

    combined_issues = pd.concat(all_issues, ignore_index=True)

    # Count by issue_type
    issue_counts = combined_issues["issue_type"].value_counts().reset_index()
    issue_counts.columns = ["issue_type", "count"]

    # Calculate cumulative percentage
    issue_counts["pct"] = issue_counts["count"] / issue_counts["count"].sum()
    issue_counts["cum_pct"] = issue_counts["pct"].cumsum()

    # Create Pareto chart
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=issue_counts["issue_type"],
            y=issue_counts["count"],
            name="Count",
            marker_color="steelblue",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=issue_counts["issue_type"],
            y=issue_counts["cum_pct"] * 100,
            name="Cumulative %",
            yaxis="y2",
            mode="lines+markers",
            marker_color="red",
        )
    )

    fig.update_layout(
        title="Portfolio-wide Issue Types (Pareto)",
        xaxis_title="Issue Type",
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )

    return fig


def show_portfolio_mode(portfolio_dir: Path, quality_threshold: float) -> None:
    """Display portfolio overview mode."""
    st.header("📊 Portfolio Overview")

    # Load portfolio data
    try:
        summary_df, metrics = load_portfolio_data(portfolio_dir)
    except Exception as e:
        st.error(f"Failed to load portfolio data: {e}")
        return

    # A) Portfolio KPI Tiles
    st.subheader("Portfolio KPIs")

    col1, col2, col3, col4 = st.columns(4)

    total_models = metrics.get("total_models", 0)
    success_models = metrics.get("success_models", 0)
    failed_models = metrics.get("failed_models", 0)

    col1.metric("Total Models", total_models)
    col2.metric(
        "Success / Failed",
        f"{success_models} / {failed_models}",
        delta=f"{metrics.get('success_rate', 0):.0%}",
    )
    col3.metric("Avg Quality Score", f"{metrics.get('avg_quality_score', 0):.1f}")

    models_below_threshold = metrics.get("models_below_threshold", 0)
    pct_below = models_below_threshold / max(1, success_models) * 100
    col4.metric(
        f"Below Threshold ({quality_threshold})",
        f"{models_below_threshold} ({pct_below:.0f}%)",
        delta_color="inverse",
    )

    col5, col6, col7 = st.columns(3)
    col5.metric("Total Critical Issues", metrics.get("total_critical_issues", 0))
    col6.metric("Issue Rate P50", f"{metrics.get('median_issue_rate_per_1k', 0):.1f}")
    col7.metric("Issue Rate P95", f"{metrics.get('p95_issue_rate_per_1k', 0):.1f}")

    st.divider()

    # B) Portfolio Charts
    st.subheader("Portfolio Charts")

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            build_portfolio_quality_chart(summary_df),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            build_critical_issues_chart(summary_df, top_n=10),
            use_container_width=True,
        )

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            build_issue_rate_distribution(summary_df),
            use_container_width=True,
        )

    with right:
        st.plotly_chart(
            build_portfolio_pareto(portfolio_dir),
            use_container_width=True,
        )

    st.divider()

    # C) Portfolio Tables
    st.subheader("Actionable Tables")

    # Top offenders table
    st.write("**Top Offenders (Models)**")

    success_df = summary_df[summary_df["status"] == "success"].copy()

    if not success_df.empty:
        top_offenders = success_df.nlargest(10, "total_issues")[
            [
                "model_name",
                "quality_score",
                "critical_issues",
                "total_issues",
                "issue_rate_per_1k",
                "element_count",
            ]
        ].copy()

        # Format for display
        top_offenders["quality_score"] = top_offenders["quality_score"].apply(lambda x: f"{x:.1f}")
        top_offenders["issue_rate_per_1k"] = top_offenders["issue_rate_per_1k"].apply(
            lambda x: f"{x:.1f}"
        )

        st.dataframe(top_offenders, use_container_width=True)

    # Failed models table
    failed_df = summary_df[summary_df["status"] == "failure"]

    if not failed_df.empty:
        st.write("**Failed Models**")
        st.dataframe(
            failed_df[["model_name", "error_message"]],
            use_container_width=True,
        )

    st.divider()

    # D) Model Drilldown
    st.subheader("🔍 Model Drilldown")

    success_models_list = success_df["model_name"].tolist() if not success_df.empty else []

    if success_models_list:
        selected_model = st.selectbox("Select a model to view details:", success_models_list)

        if selected_model:
            model_dir = portfolio_dir / selected_model

            try:
                features_df, issues_df, model_metrics = load_model_data(model_dir)

                # Show model KPIs
                st.write(f"**Model: {selected_model}**")

                m_col1, m_col2, m_col3, m_col4 = st.columns(4)
                m_col1.metric("Elements", model_metrics.get("total_elements", 0))
                m_col2.metric("Quality Score", f"{model_metrics.get('quality_score', 0):.1f}")
                m_col3.metric("Total Issues", model_metrics.get("total_issues", 0))
                m_col4.metric(
                    "Issue Rate/1k", f"{model_metrics.get('issue_rate_per_1000', 0):.1f}"
                )

                # Build metrics DataFrames for charts
                from ifcqi.metrics import build_metrics

                _, metrics_dfs = build_metrics(features_df, issues_df)

                # Show charts
                chart_left, chart_right = st.columns(2)

                with chart_left:
                    st.plotly_chart(
                        build_severity_chart(metrics_dfs["severity"]),
                        use_container_width=True,
                    )

                with chart_right:
                    st.plotly_chart(
                        build_issues_by_ifc_type_chart(metrics_dfs["issues_by_ifc_type"]),
                        use_container_width=True,
                    )

                # Top issues table
                if not issues_df.empty:
                    st.write("**Top Issues (First 20)**")

                    # Sort by severity and show first 20
                    severity_order = {"critical": 0, "major": 1, "minor": 2}
                    issues_display = issues_df.copy()
                    issues_display["severity_order"] = issues_display["severity"].map(
                        severity_order
                    )
                    issues_display = issues_display.sort_values("severity_order").head(20)

                    st.dataframe(
                        issues_display[
                            [
                                "severity",
                                "issue_type",
                                "ifc_type",
                                "global_id",
                                "element_name",
                                "message",
                            ]
                        ],
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Failed to load model data: {e}")


def show_single_model_mode(quality_threshold: float) -> None:
    """Display single model mode (original dashboard)."""
    st.header("📄 Single Model Analysis")

    from ifcqi.checks import issues_to_dataframe, run_all_checks
    from ifcqi.features import compute_geometry_features
    from ifcqi.ifc_loader import extract_elements, load_ifc
    from ifcqi.metrics import build_metrics

    data_source = st.sidebar.radio(
        "Choose data source", ["Use output folder", "Upload files", "Upload IFC"], index=0
    )

    try:
        if data_source == "Use output folder":
            output_dir = Path("output")
            features_path = output_dir / "features.parquet"
            issues_path = output_dir / "issues.csv"

            if not features_path.exists():
                st.warning("No data found in output folder. Please process an IFC file first.")
                return

            features_df = pd.read_parquet(features_path)

            if issues_path.exists():
                issues_df = pd.read_csv(issues_path)
            else:
                issues_df = pd.DataFrame()

            st.sidebar.success(f"Loaded from {output_dir}")

        elif data_source == "Upload files":
            features_file = st.sidebar.file_uploader("Upload features.parquet", type=["parquet"])
            issues_file = st.sidebar.file_uploader("Upload issues.csv", type=["csv"])

            if features_file is None:
                st.warning("Please upload features.parquet file")
                return

            features_df = pd.read_parquet(features_file)

            if issues_file is not None:
                issues_df = pd.read_csv(issues_file)
            else:
                issues_df = pd.DataFrame()

            st.sidebar.success("Files uploaded")

        else:
            ifc_file = st.sidebar.file_uploader("Upload IFC file", type=["ifc"])

            if ifc_file is None:
                st.warning("Please upload an IFC file")
                return

            with st.spinner("Processing IFC..."):
                from tempfile import NamedTemporaryFile

                with NamedTemporaryFile(delete=False, suffix=".ifc") as tmp:
                    tmp.write(ifc_file.read())
                    tmp_path = Path(tmp.name)

                model = load_ifc(tmp_path)
                elements_df = extract_elements(model)
                features_df = compute_geometry_features(model, elements_df)
                issues = run_all_checks(features_df)
                issues_df = issues_to_dataframe(issues)

            st.sidebar.success("IFC processed")

    except Exception as exc:
        st.error(str(exc))
        return

    # Build metrics
    metrics, metrics_dfs = build_metrics(features_df, issues_df)

    # Quality status
    quality_score = metrics["quality_score"]
    status = "✅ PASS" if quality_score >= quality_threshold else "❌ FAIL"

    st.write(f"**Quality Status:** {status} (Score: {quality_score:.1f}, Threshold: {quality_threshold})")

    # KPI tiles
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Elements", metrics["total_elements"])
    col2.metric("Quality Score", f"{quality_score:.1f}")
    col3.metric("Total Issues", metrics["total_issues"])
    col4.metric("Issue Rate / 1,000", f"{metrics['issue_rate_per_1000']:.2f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Name Completeness (%)", f"{metrics['metadata_completeness']['name_pct']:.1f}")
    col6.metric(
        "ObjectType Completeness (%)",
        f"{metrics['metadata_completeness']['object_type_pct']:.1f}",
    )
    col7.metric(
        "Geometry Coverage (%)",
        f"{metrics['metadata_completeness']['geometry_pct']:.1f}",
    )

    st.divider()

    # Charts
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_severity_chart(metrics_dfs["severity"]), use_container_width=True)
    with right:
        st.plotly_chart(
            build_issues_by_ifc_type_chart(metrics_dfs["issues_by_ifc_type"]),
            use_container_width=True,
        )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(
            build_issues_by_type_chart(metrics_dfs["issues_by_type"]),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(
            build_pareto_chart(metrics_dfs["pareto_issue_types"]),
            use_container_width=True,
        )

    st.divider()

    # Top issues table
    if not issues_df.empty:
        st.subheader("Top Issues (First 20)")

        severity_order = {"critical": 0, "major": 1, "minor": 2}
        issues_display = issues_df.copy()
        issues_display["severity_order"] = issues_display["severity"].map(severity_order)
        issues_display = issues_display.sort_values("severity_order").head(20)

        st.dataframe(
            issues_display[
                ["severity", "issue_type", "ifc_type", "global_id", "element_name", "message"]
            ],
            use_container_width=True,
        )


def main() -> None:
    """Main dashboard application."""
    st.set_page_config(page_title="IFC Quality Intelligence", layout="wide")

    st.title("IFC Quality Intelligence Dashboard")
    st.caption("Enterprise BIM Quality Validation & Portfolio Analytics")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Mode selection
    mode = st.sidebar.radio("Dashboard Mode", ["Portfolio", "Single Model"], index=0)

    # Quality threshold slider
    quality_threshold = st.sidebar.slider(
        "Quality Threshold",
        min_value=0.0,
        max_value=100.0,
        value=80.0,
        step=5.0,
        help="Models below this score are flagged as failing",
    )

    st.sidebar.divider()

    # Mode-specific settings
    if mode == "Portfolio":
        portfolio_path = st.sidebar.text_input(
            "Portfolio Directory",
            value="output/portfolio",
            help="Directory containing portfolio_summary.csv",
        )

        portfolio_dir = Path(portfolio_path)

        if portfolio_dir.exists():
            show_portfolio_mode(portfolio_dir, quality_threshold)
        else:
            st.error(f"Portfolio directory not found: {portfolio_dir}")
            st.info("Run batch processing first: `python examples/example_batch_processing.py`")

    else:
        show_single_model_mode(quality_threshold)


if __name__ == "__main__":
    main()
