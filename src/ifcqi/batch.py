"""Batch processing and portfolio analytics for IFC models.

Orchestrates the full quality validation pipeline across multiple IFC files
and computes portfolio-level metrics.
"""

from __future__ import annotations

import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from ifcqi.checks import issues_to_dataframe, run_all_checks
from ifcqi.features import compute_geometry_features
from ifcqi.ifc_loader import extract_elements, load_ifc
from ifcqi.logger import get_logger
from ifcqi.metrics import build_metrics, save_metrics
from ifcqi.viz import (
    build_issues_by_ifc_type_chart,
    build_issues_by_type_chart,
    build_pareto_chart,
    build_severity_chart,
    export_figures_to_html,
)

logger = get_logger(__name__)


@dataclass
class ModelResult:
    """Result of processing a single IFC model."""

    model_name: str
    model_path: str
    status: str  # "success" | "failure"
    element_count: int
    total_issues: int
    critical_issues: int
    major_issues: int
    minor_issues: int
    quality_score: float
    issue_rate_per_1k: float
    metadata_completeness_pct: float
    anomaly_count: Optional[int]
    anomaly_rate: Optional[float]
    runtime_seconds: float
    processed_at: str
    error_message: Optional[str] = None


def discover_ifc_files(input_dir: Path, recursive: bool = False) -> list[Path]:
    """Discover all IFC files in a directory.

    Args:
        input_dir: Directory to scan
        recursive: Scan subdirectories

    Returns:
        List of IFC file paths
    """
    if not input_dir.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")

    if recursive:
        ifc_files = list(input_dir.rglob("*.ifc"))
    else:
        ifc_files = list(input_dir.glob("*.ifc"))

    logger.info(f"Discovered {len(ifc_files)} IFC files in {input_dir}")
    return ifc_files


def process_single_model(
    ifc_path: Path,
    output_dir: Path,
    enable_ml: bool = False,
    enable_html: bool = True,
) -> ModelResult:
    """Process a single IFC model through the full pipeline.

    Args:
        ifc_path: Path to IFC file
        output_dir: Directory for outputs
        enable_ml: Enable ML anomaly detection
        enable_html: Generate HTML report

    Returns:
        ModelResult with metrics and status
    """
    start_time = time.time()
    model_name = ifc_path.stem

    # Create model-specific output directory
    model_output_dir = output_dir / model_name
    model_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Step 1: Load IFC
        logger.info(f"Processing {model_name}...")
        model = load_ifc(ifc_path)

        # Step 2: Extract elements
        elements_df = extract_elements(model)
        element_count = len(elements_df)

        # Step 3: Extract geometry features
        features_df = compute_geometry_features(model, elements_df)

        # Save features
        features_df.to_parquet(model_output_dir / "features.parquet", index=False)

        # Step 4: Run quality checks
        issues = run_all_checks(features_df)
        issues_df = issues_to_dataframe(issues)

        # Save issues
        issues_df.to_csv(model_output_dir / "issues.csv", index=False)

        # Step 5: Build metrics
        metrics, metrics_dfs = build_metrics(features_df, issues_df)

        # Save metrics
        save_metrics(metrics, model_output_dir / "metrics.json")

        # Step 6: ML anomaly detection (optional)
        anomaly_count = None
        anomaly_rate = None

        if enable_ml and element_count > 10:  # Need enough data for ML
            try:
                from ifcqi.ml import detect_anomalies

                anomalies, _ = detect_anomalies(
                    features_df, contamination=0.05, random_state=42
                )
                anomaly_df = pd.DataFrame([vars(a) for a in anomalies])
                anomaly_df.to_csv(model_output_dir / "anomalies.csv", index=False)

                anomaly_count = int(anomaly_df["is_anomaly"].sum())
                anomaly_rate = float(anomaly_count / max(1, element_count))

            except Exception as e:
                logger.warning(f"ML anomaly detection failed for {model_name}: {e}")

        # Step 7: Generate HTML report (optional)
        if enable_html:
            try:
                figures = {
                    "Severity Breakdown": build_severity_chart(metrics_dfs["severity"]),
                    "Issues by IFC Type": build_issues_by_ifc_type_chart(
                        metrics_dfs["issues_by_ifc_type"]
                    ),
                    "Issues by Type": build_issues_by_type_chart(metrics_dfs["issues_by_type"]),
                    "Pareto Analysis": build_pareto_chart(metrics_dfs["pareto_issue_types"]),
                }
                export_figures_to_html(figures, model_output_dir / "report.html")
            except Exception as e:
                logger.warning(f"HTML report generation failed for {model_name}: {e}")

        # Calculate runtime
        runtime = time.time() - start_time

        # Extract metrics for portfolio
        severity_breakdown = metrics["severity_breakdown"]
        metadata_comp = metrics["metadata_completeness"]
        metadata_pct = (
            metadata_comp["name_pct"]
            + metadata_comp["object_type_pct"]
            + metadata_comp["geometry_pct"]
        ) / 3.0

        result = ModelResult(
            model_name=model_name,
            model_path=str(ifc_path),
            status="success",
            element_count=element_count,
            total_issues=metrics["total_issues"],
            critical_issues=severity_breakdown["critical"],
            major_issues=severity_breakdown["major"],
            minor_issues=severity_breakdown["minor"],
            quality_score=float(metrics["quality_score"]),
            issue_rate_per_1k=float(metrics["issue_rate_per_1000"]),
            metadata_completeness_pct=float(metadata_pct),
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_rate,
            runtime_seconds=runtime,
            processed_at=datetime.now().isoformat(),
        )

        logger.info(
            f"✓ {model_name}: {element_count} elements, "
            f"quality={result.quality_score:.1f}, "
            f"issues={result.total_issues}, "
            f"time={runtime:.1f}s"
        )

        return result

    except Exception as e:
        runtime = time.time() - start_time
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(f"✗ {model_name} failed: {error_msg}")
        logger.debug(traceback.format_exc())

        return ModelResult(
            model_name=model_name,
            model_path=str(ifc_path),
            status="failure",
            element_count=0,
            total_issues=0,
            critical_issues=0,
            major_issues=0,
            minor_issues=0,
            quality_score=0.0,
            issue_rate_per_1k=0.0,
            metadata_completeness_pct=0.0,
            anomaly_count=None,
            anomaly_rate=None,
            runtime_seconds=runtime,
            processed_at=datetime.now().isoformat(),
            error_message=error_msg,
        )


def process_batch(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    enable_ml: bool = False,
    enable_html: bool = True,
    max_workers: Optional[int] = None,
) -> list[ModelResult]:
    """Process all IFC files in a directory.

    Args:
        input_dir: Directory containing IFC files
        output_dir: Directory for batch outputs
        recursive: Scan subdirectories
        enable_ml: Enable ML anomaly detection
        enable_html: Generate HTML reports
        max_workers: Number of parallel workers (None = sequential)

    Returns:
        List of ModelResult objects
    """
    # Discover IFC files
    ifc_files = discover_ifc_files(input_dir, recursive=recursive)

    if not ifc_files:
        logger.warning(f"No IFC files found in {input_dir}")
        return []

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    if max_workers and max_workers > 1:
        # Parallel processing
        logger.info(f"Processing {len(ifc_files)} files with {max_workers} workers...")

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    process_single_model, ifc_path, output_dir, enable_ml, enable_html
                ): ifc_path
                for ifc_path in ifc_files
            }

            for future in as_completed(futures):
                result = future.result()
                results.append(result)

    else:
        # Sequential processing
        logger.info(f"Processing {len(ifc_files)} files sequentially...")

        for ifc_path in ifc_files:
            result = process_single_model(ifc_path, output_dir, enable_ml, enable_html)
            results.append(result)

    logger.info(f"Batch processing complete: {len(results)} models processed")

    return results


def compute_portfolio_metrics(results: list[ModelResult]) -> dict[str, Any]:
    """Compute portfolio-level KPIs from model results.

    Args:
        results: List of ModelResult objects

    Returns:
        Dictionary of portfolio metrics
    """
    if not results:
        return {}

    # Convert to DataFrame for easier aggregation
    df = pd.DataFrame([asdict(r) for r in results])

    # Filter to successful models
    success_df = df[df["status"] == "success"]
    n_success = len(success_df)
    n_failed = len(df) - n_success

    if n_success == 0:
        logger.warning("No successful models to analyze")
        return {
            "total_models": len(df),
            "success_models": 0,
            "failed_models": n_failed,
        }

    # Portfolio health metrics
    portfolio_metrics = {
        # Summary
        "total_models": len(df),
        "success_models": n_success,
        "failed_models": n_failed,
        "success_rate": float(n_success / len(df)),
        # Quality scores
        "avg_quality_score": float(success_df["quality_score"].mean()),
        "median_quality_score": float(success_df["quality_score"].median()),
        "min_quality_score": float(success_df["quality_score"].min()),
        "max_quality_score": float(success_df["quality_score"].max()),
        "std_quality_score": float(success_df["quality_score"].std()),
        # Best/worst models
        "best_model": success_df.loc[success_df["quality_score"].idxmax(), "model_name"],
        "worst_model": success_df.loc[success_df["quality_score"].idxmin(), "model_name"],
        # Issue metrics
        "total_issues": int(success_df["total_issues"].sum()),
        "total_critical_issues": int(success_df["critical_issues"].sum()),
        "total_major_issues": int(success_df["major_issues"].sum()),
        "total_minor_issues": int(success_df["minor_issues"].sum()),
        "avg_issue_rate_per_1k": float(success_df["issue_rate_per_1k"].mean()),
        "median_issue_rate_per_1k": float(success_df["issue_rate_per_1k"].median()),
        "p95_issue_rate_per_1k": float(success_df["issue_rate_per_1k"].quantile(0.95)),
        # Element metrics
        "total_elements": int(success_df["element_count"].sum()),
        "avg_elements_per_model": float(success_df["element_count"].mean()),
        # Metadata completeness
        "avg_metadata_completeness": float(success_df["metadata_completeness_pct"].mean()),
        # Runtime metrics
        "total_runtime_seconds": float(success_df["runtime_seconds"].sum()),
        "avg_runtime_seconds": float(success_df["runtime_seconds"].mean()),
        "p95_runtime_seconds": float(success_df["runtime_seconds"].quantile(0.95)),
    }

    # ML metrics (if available)
    if "anomaly_rate" in success_df.columns and success_df["anomaly_rate"].notna().any():
        ml_df = success_df[success_df["anomaly_rate"].notna()]
        portfolio_metrics.update(
            {
                "total_anomalies": int(ml_df["anomaly_count"].sum()),
                "avg_anomaly_rate": float(ml_df["anomaly_rate"].mean()),
                "median_anomaly_rate": float(ml_df["anomaly_rate"].median()),
                "p95_anomaly_rate": float(ml_df["anomaly_rate"].quantile(0.95)),
            }
        )

    # Top offenders
    if n_success >= 5:
        top_critical = (
            success_df.nlargest(5, "critical_issues")[["model_name", "critical_issues"]]
            .to_dict(orient="records")
        )
        top_issues = (
            success_df.nlargest(5, "total_issues")[["model_name", "total_issues"]]
            .to_dict(orient="records")
        )
        portfolio_metrics["top_models_by_critical"] = top_critical
        portfolio_metrics["top_models_by_issues"] = top_issues

    return portfolio_metrics


def save_portfolio_results(
    results: list[ModelResult], output_dir: Path, portfolio_metrics: dict[str, Any]
) -> None:
    """Save portfolio results to disk.

    Args:
        results: List of ModelResult objects
        output_dir: Output directory
        portfolio_metrics: Portfolio-level metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary CSV
    df = pd.DataFrame([asdict(r) for r in results])
    df.to_csv(output_dir / "portfolio_summary.csv", index=False)
    logger.info(f"Saved portfolio summary to {output_dir / 'portfolio_summary.csv'}")

    # Save summary JSON
    summary_json = df.to_dict(orient="records")
    with open(output_dir / "portfolio_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    logger.info(f"Saved portfolio summary to {output_dir / 'portfolio_summary.json'}")

    # Save portfolio metrics
    with open(output_dir / "portfolio_metrics.json", "w") as f:
        json.dump(portfolio_metrics, f, indent=2, default=str)
    logger.info(f"Saved portfolio metrics to {output_dir / 'portfolio_metrics.json'}")


def run_portfolio_analysis(
    input_dir: Path,
    output_dir: Path,
    recursive: bool = False,
    enable_ml: bool = False,
    enable_html: bool = True,
    max_workers: Optional[int] = None,
    quality_threshold: float = 80.0,
) -> dict[str, Any]:
    """Run full portfolio analysis workflow.

    Args:
        input_dir: Directory containing IFC files
        output_dir: Directory for outputs
        recursive: Scan subdirectories
        enable_ml: Enable ML anomaly detection
        enable_html: Generate HTML reports
        max_workers: Number of parallel workers
        quality_threshold: Quality score threshold

    Returns:
        Portfolio metrics dictionary
    """
    logger.info("=" * 70)
    logger.info("IFC Quality Intelligence - Portfolio Analysis")
    logger.info("=" * 70)

    # Process all models
    results = process_batch(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=recursive,
        enable_ml=enable_ml,
        enable_html=enable_html,
        max_workers=max_workers,
    )

    # Compute portfolio metrics
    portfolio_metrics = compute_portfolio_metrics(results)
    portfolio_metrics["quality_threshold"] = quality_threshold

    # Count models below threshold
    success_results = [r for r in results if r.status == "success"]
    below_threshold = sum(1 for r in success_results if r.quality_score < quality_threshold)
    portfolio_metrics["models_below_threshold"] = below_threshold

    # Save results
    save_portfolio_results(results, output_dir, portfolio_metrics)

    logger.info("=" * 70)
    logger.info("Portfolio Analysis Complete")
    logger.info("=" * 70)

    return portfolio_metrics
