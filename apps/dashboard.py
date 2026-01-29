"""Streamlit dashboard for IFC Quality Intelligence metrics."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import pandas as pd
import streamlit as st

from ifcqi.checks import issues_to_dataframe, run_all_checks
from ifcqi.features import compute_geometry_features
from ifcqi.ifc_loader import extract_elements, load_ifc
from ifcqi.metrics import build_metrics
from ifcqi.viz import (
    build_issues_by_ifc_type_chart,
    build_issues_by_type_chart,
    build_pareto_chart,
    build_severity_chart,
)


def _load_outputs_from_folder(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_path = folder / "features.parquet"
    issues_path = folder / "issues.csv"

    if not features_path.exists():
        raise FileNotFoundError(f"Missing features file: {features_path}")

    features_df = pd.read_parquet(features_path)

    if issues_path.exists():
        issues_df = pd.read_csv(issues_path)
    else:
        issues_df = pd.DataFrame(
            columns=["global_id", "ifc_type", "issue_type", "severity", "message", "element_name"]
        )

    return features_df, issues_df


def _load_outputs_from_uploads(
    features_file, issues_file
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if features_file is None:
        raise ValueError("Please upload a features.parquet file.")

    features_df = pd.read_parquet(features_file)

    if issues_file is not None:
        issues_df = pd.read_csv(issues_file)
    else:
        issues_df = pd.DataFrame(
            columns=["global_id", "ifc_type", "issue_type", "severity", "message", "element_name"]
        )

    return features_df, issues_df


def _load_outputs_from_ifc_upload(
    ifc_file, *, max_elements: int | None = None, save_outputs: bool = False
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if ifc_file is None:
        raise ValueError("Please upload an IFC file.")

    with NamedTemporaryFile(delete=False, suffix=".ifc") as tmp:
        tmp.write(ifc_file.read())
        tmp_path = Path(tmp.name)

    model = load_ifc(tmp_path)
    elements_df = extract_elements(model, max_elements=max_elements)
    features_df = compute_geometry_features(model, elements_df)
    issues = run_all_checks(features_df)
    issues_df = issues_to_dataframe(issues)

    if save_outputs:
        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True)
        features_df.to_parquet(output_dir / "features.parquet", index=False)
        issues_df.to_csv(output_dir / "issues.csv", index=False)

    return features_df, issues_df


def main() -> None:
    st.set_page_config(page_title="IFC Quality Dashboard", layout="wide")
    st.title("IFC Quality Intelligence Dashboard")
    st.caption("Module 5: Metrics visualization with Plotly + Streamlit")

    st.sidebar.header("Data Source")
    data_source = st.sidebar.radio(
        "Choose data source", ["Use output folder", "Upload files", "Upload IFC"], index=0
    )
    top_n = st.sidebar.slider("Top N (issue types/IFC types)", 5, 25, 10, 1)
    max_elements = st.sidebar.number_input(
        "Max elements (optional)", min_value=0, value=0, step=100
    )
    max_elements_val = None if max_elements == 0 else int(max_elements)
    save_outputs = st.sidebar.checkbox("Save outputs to ./output", value=False)

    features_df: pd.DataFrame
    issues_df: pd.DataFrame

    try:
        if data_source == "Use output folder":
            output_dir = Path("output")
            features_df, issues_df = _load_outputs_from_folder(output_dir)
            st.sidebar.success(f"Loaded from {output_dir}")
        elif data_source == "Upload files":
            features_file = st.sidebar.file_uploader(
                "Upload features.parquet", type=["parquet"]
            )
            issues_file = st.sidebar.file_uploader("Upload issues.csv", type=["csv"])
            features_df, issues_df = _load_outputs_from_uploads(features_file, issues_file)
            st.sidebar.success("Files uploaded")
        else:
            ifc_file = st.sidebar.file_uploader("Upload IFC file", type=["ifc"])
            with st.spinner("Processing IFC..."):
                features_df, issues_df = _load_outputs_from_ifc_upload(
                    ifc_file, max_elements=max_elements_val, save_outputs=save_outputs
                )
            st.sidebar.success("IFC processed")
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    metrics, metrics_dfs = build_metrics(features_df, issues_df, top_n=top_n)

    # KPI tiles
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Elements", metrics["total_elements"])
    col2.metric("Quality Score", f"{metrics['quality_score']:.1f}")
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
    st.subheader("Top Offenders")
    st.dataframe(pd.DataFrame(metrics["top_offenders"]["ifc_type"].items(), columns=["ifc_type", "count"]))
    st.dataframe(pd.DataFrame(metrics["top_offenders"]["issue_type"].items(), columns=["issue_type", "count"]))

    st.subheader("Download Metrics JSON")
    st.download_button(
        "Download metrics.json",
        data=json.dumps(metrics, indent=2),
        file_name="metrics.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
