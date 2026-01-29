"""Visualization utilities for IFC quality metrics.

Provides Plotly figures for dashboards or HTML reports.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio


def build_severity_chart(severity_df: pd.DataFrame) -> go.Figure:
    """Bar chart of issues by severity."""
    if severity_df.empty:
        return px.bar(title="Issues by Severity")
    fig = px.bar(
        severity_df,
        x="severity",
        y="count",
        color="severity",
        title="Issues by Severity",
        text="count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(yaxis_title="Count", xaxis_title="Severity", showlegend=False)
    return fig


def build_issues_by_ifc_type_chart(issues_by_ifc_df: pd.DataFrame) -> go.Figure:
    """Bar chart of issues by IFC type."""
    if issues_by_ifc_df.empty:
        return px.bar(title="Issues by IFC Type")
    fig = px.bar(
        issues_by_ifc_df,
        x="ifc_type",
        y="count",
        title="Issues by IFC Type",
    )
    fig.update_layout(xaxis_title="IFC Type", yaxis_title="Count")
    return fig


def build_issues_by_type_chart(issues_by_type_df: pd.DataFrame) -> go.Figure:
    """Bar chart of issues by issue type."""
    if issues_by_type_df.empty:
        return px.bar(title="Issues by Issue Type")
    fig = px.bar(
        issues_by_type_df,
        x="issue_type",
        y="count",
        title="Issues by Issue Type",
    )
    fig.update_layout(xaxis_title="Issue Type", yaxis_title="Count")
    return fig


def build_pareto_chart(pareto_df: pd.DataFrame) -> go.Figure:
    """Pareto chart of issue types with cumulative percentage."""
    if pareto_df.empty:
        return px.bar(title="Pareto of Issue Types")

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=pareto_df["issue_type"],
            y=pareto_df["count"],
            name="Count",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=pareto_df["issue_type"],
            y=pareto_df["cum_pct"] * 100.0,
            name="Cumulative %",
            yaxis="y2",
            mode="lines+markers",
        )
    )

    fig.update_layout(
        title="Pareto of Issue Types",
        xaxis_title="Issue Type",
        yaxis=dict(title="Count"),
        yaxis2=dict(
            title="Cumulative %",
            overlaying="y",
            side="right",
            range=[0, 100],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def export_figures_to_html(figures: Dict[str, go.Figure], output_path: Path) -> None:
    """Export multiple figures into a single HTML report."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    html_parts = []
    for title, fig in figures.items():
        html_parts.append(f"<h2>{title}</h2>")
        html_parts.append(pio.to_html(fig, include_plotlyjs="cdn", full_html=False))

    html = "<html><head><meta charset='utf-8'></head><body>" + "\n".join(html_parts) + "</body></html>"
    output_path.write_text(html, encoding="utf-8")
