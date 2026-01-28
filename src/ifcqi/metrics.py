"""Metrics and Scoring Layer for IFC models.

This module calculates various metrics and scores to assess the quality of IFC models.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from ifcqi.logger import get_logger

logger = get_logger(__name__)

SEVERITY_WEIGHTS = {
    "critical": 3.0,
    "major": 2.0,
    "minor": 1.0,
}


def _coerce_issues_df(issues: Union[pd.DataFrame, list]) -> pd.DataFrame:
    """Normalize issues input into a DataFrame.

    Args:
        issues: Either a DataFrame (from `issues_to_dataframe`) or a list of Issue objects.

    Returns:
        Issues DataFrame with at least: global_id, ifc_type, issue_type, severity.
    """
    if isinstance(issues, pd.DataFrame):
        return issues

    # Avoid module-level import to keep dependency direction flexible.
    from ifcqi.checks import issues_to_dataframe

    return issues_to_dataframe(issues)


def compute_severity_breakdown(issues_df: pd.DataFrame) -> Dict[str, int]:
    """Compute counts of issues by severity."""
    if issues_df.empty or "severity" not in issues_df.columns:
        return {"critical": 0, "major": 0, "minor": 0}

    counts = issues_df["severity"].value_counts().to_dict()
    return {
        "critical": int(counts.get("critical", 0)),
        "major": int(counts.get("major", 0)),
        "minor": int(counts.get("minor", 0)),
    }


def compute_issue_rate_per_1000(total_elements: int, total_issues: int) -> float:
    """Compute issue rate per 1,000 elements."""
    if total_elements <= 0:
        return 0.0
    return (total_issues / total_elements) * 1000.0


def compute_metadata_completeness(elements_df: pd.DataFrame) -> Dict[str, float]:
    """Compute metadata completeness metrics.

    Returns percentages in [0, 100].
    """
    total = len(elements_df)
    if total == 0:
        return {
            "name_pct": 0.0,
            "object_type_pct": 0.0,
            "geometry_pct": 0.0,
        }

    def _pct_present(col: str) -> float:
        if col not in elements_df.columns:
            return 0.0
        present = elements_df[col].notna() & (elements_df[col].astype(str).str.len() > 0)
        return float(present.mean() * 100.0)

    geometry_pct = (
        float(elements_df["has_geometry"].mean() * 100.0)
        if "has_geometry" in elements_df.columns
        else 0.0
    )

    return {
        "name_pct": _pct_present("name"),
        "object_type_pct": _pct_present("object_type"),
        "geometry_pct": geometry_pct,
    }


def compute_top_offenders(
    issues_df: pd.DataFrame, *, group_col: str, top_n: int = 10
) -> Dict[str, int]:
    """Compute top-N counts for a given grouping column."""
    if issues_df.empty or group_col not in issues_df.columns:
        return {}
    return (
        issues_df[group_col]
        .value_counts()
        .head(top_n)
        .astype(int)
        .to_dict()
    )


def compute_quality_score(
    total_elements: int,
    issues_df: pd.DataFrame,
    severity_weights: Optional[Dict[str, float]] = None,
) -> float:
    """Compute a 0-100 quality score weighted by issue severity."""
    if total_elements <= 0:
        return 0.0
    if issues_df.empty:
        return 100.0

    weights = severity_weights or SEVERITY_WEIGHTS

    if "severity" not in issues_df.columns:
        weighted_issues = float(len(issues_df))
    else:
        weighted_issues = float(
            issues_df["severity"].map(lambda s: weights.get(str(s).lower(), 1.0)).sum()
        )

    score = 100.0 - (weighted_issues / total_elements * 100.0)
    return float(max(0.0, min(100.0, score)))


def build_metrics(
    elements_df: pd.DataFrame,
    issues: Union[pd.DataFrame, list],
    *,
    severity_weights: Optional[Dict[str, float]] = None,
    top_n: int = 10,
) -> Tuple[Dict[str, Any], Dict[str, pd.DataFrame]]:
    """Build Module-4 metrics.

    Returns:
        metrics: JSON-serializable dict (single source of truth).
        metrics_df: Dict of DataFrames to feed visualizations.
    """
    issues_df = _coerce_issues_df(issues)
    total_elements = int(len(elements_df))
    total_issues = int(len(issues_df))

    severity_breakdown = compute_severity_breakdown(issues_df)
    completeness = compute_metadata_completeness(elements_df)
    issue_rate = compute_issue_rate_per_1000(total_elements, total_issues)
    quality_score = compute_quality_score(
        total_elements, issues_df, severity_weights=severity_weights
    )

    by_ifc_type = compute_top_offenders(issues_df, group_col="ifc_type", top_n=top_n)
    by_issue_type = compute_top_offenders(issues_df, group_col="issue_type", top_n=top_n)

    metrics: Dict[str, Any] = {
        "total_elements": total_elements,
        "quality_score": quality_score,
        "total_issues": total_issues,
        "severity_breakdown": severity_breakdown,
        "issue_rate_per_1000": issue_rate,
        "metadata_completeness": completeness,
        "top_offenders": {
            "ifc_type": by_ifc_type,
            "issue_type": by_issue_type,
        },
    }

    kpi_df = pd.DataFrame(
        {
            "metric": [
                "total_elements",
                "quality_score",
                "total_issues",
                "issue_rate_per_1000",
                "name_pct",
                "object_type_pct",
                "geometry_pct",
                "critical",
                "major",
                "minor",
            ],
            "value": [
                metrics["total_elements"],
                metrics["quality_score"],
                metrics["total_issues"],
                metrics["issue_rate_per_1000"],
                completeness["name_pct"],
                completeness["object_type_pct"],
                completeness["geometry_pct"],
                severity_breakdown["critical"],
                severity_breakdown["major"],
                severity_breakdown["minor"],
            ],
        }
    )

    severity_df = pd.DataFrame(
        {
            "severity": ["critical", "major", "minor"],
            "count": [
                severity_breakdown["critical"],
                severity_breakdown["major"],
                severity_breakdown["minor"],
            ],
        }
    )

    offenders_ifc_df = (
        issues_df["ifc_type"].value_counts().rename_axis("ifc_type").reset_index(name="count")
        if (not issues_df.empty and "ifc_type" in issues_df.columns)
        else pd.DataFrame(columns=["ifc_type", "count"])
    )

    issue_types_df = (
        issues_df["issue_type"]
        .value_counts()
        .rename_axis("issue_type")
        .reset_index(name="count")
        if (not issues_df.empty and "issue_type" in issues_df.columns)
        else pd.DataFrame(columns=["issue_type", "count"])
    )

    # Pareto: top issue types + cumulative percentage
    pareto_df = issue_types_df.copy()
    if not pareto_df.empty:
        pareto_df["pct"] = pareto_df["count"] / max(1, pareto_df["count"].sum())
        pareto_df["cum_pct"] = pareto_df["pct"].cumsum()

    # Surface Pareto summary in metrics JSON (top_n only)
    metrics["pareto_issue_types"] = (
        pareto_df.head(top_n).to_dict(orient="records") if not pareto_df.empty else []
    )

    metrics_dfs = {
        "kpi": kpi_df,
        "severity": severity_df,
        "issues_by_ifc_type": offenders_ifc_df,
        "issues_by_type": issue_types_df,
        "pareto_issue_types": pareto_df,
    }

    logger.info(
        "Built metrics: score=%.1f, issues=%d, elements=%d, rate/1000=%.2f",
        quality_score,
        total_issues,
        total_elements,
        issue_rate,
    )

    return metrics, metrics_dfs


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save metrics to a JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {output_path}")
