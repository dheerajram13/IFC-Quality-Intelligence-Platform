"""Example: Batch processing and portfolio analytics on multiple IFC files.

This example demonstrates batch processing capabilities:
1. Discover all IFC files in a directory
2. Process each file through the full pipeline
3. Aggregate portfolio-level metrics
4. Generate summary reports

Usage:
    python examples/example_batch_processing.py examples/ifc_files
"""

import sys
from pathlib import Path

from ifcqi.batch import run_portfolio_analysis
from ifcqi.logger import get_logger

logger = get_logger(__name__)


def main(input_dir: Path, output_dir: Path = Path("output/portfolio")) -> None:
    """Run portfolio analysis on a directory of IFC files."""
    print("=" * 70)
    print("IFC Quality Intelligence - Batch Processing & Portfolio Analytics")
    print("=" * 70)
    print()

    # Run portfolio analysis
    portfolio_metrics = run_portfolio_analysis(
        input_dir=input_dir,
        output_dir=output_dir,
        recursive=False,  # Don't scan subdirectories
        enable_ml=True,  # Enable ML anomaly detection
        enable_html=True,  # Generate HTML reports
        max_workers=None,  # Sequential processing (set to 2-4 for parallel)
        quality_threshold=80.0,  # Quality score threshold
    )

    # Display results
    print()
    print("=" * 70)
    print("Portfolio Summary")
    print("=" * 70)
    print()

    print(f"📊 Models Processed:")
    print(f"  Total:     {portfolio_metrics.get('total_models', 0)}")
    print(f"  Success:   {portfolio_metrics.get('success_models', 0)}")
    print(f"  Failed:    {portfolio_metrics.get('failed_models', 0)}")
    print(f"  Success Rate: {portfolio_metrics.get('success_rate', 0):.1%}")
    print()

    if portfolio_metrics.get("success_models", 0) > 0:
        print(f"🎯 Quality Metrics:")
        print(f"  Average Score:  {portfolio_metrics.get('avg_quality_score', 0):.1f}/100")
        print(f"  Median Score:   {portfolio_metrics.get('median_quality_score', 0):.1f}/100")
        print(f"  Best Score:     {portfolio_metrics.get('max_quality_score', 0):.1f}/100")
        print(f"  Worst Score:    {portfolio_metrics.get('min_quality_score', 0):.1f}/100")
        print(f"  Std Dev:        {portfolio_metrics.get('std_quality_score', 0):.1f}")
        print()

        print(f"🏆 Best/Worst Models:")
        print(f"  Best:  {portfolio_metrics.get('best_model', 'N/A')}")
        print(f"  Worst: {portfolio_metrics.get('worst_model', 'N/A')}")
        print()

        print(f"🚨 Issues Detected:")
        print(f"  Total Issues:    {portfolio_metrics.get('total_issues', 0):,}")
        print(f"  Critical:        {portfolio_metrics.get('total_critical_issues', 0):,}")
        print(f"  Major:           {portfolio_metrics.get('total_major_issues', 0):,}")
        print(f"  Minor:           {portfolio_metrics.get('total_minor_issues', 0):,}")
        print(
            f"  Avg Rate/1,000:  {portfolio_metrics.get('avg_issue_rate_per_1k', 0):.2f}"
        )
        print()

        print(f"📏 Elements:")
        print(f"  Total Elements:      {portfolio_metrics.get('total_elements', 0):,}")
        print(
            f"  Avg per Model:       {portfolio_metrics.get('avg_elements_per_model', 0):.0f}"
        )
        print()

        print(f"📋 Metadata Completeness:")
        print(
            f"  Average:  {portfolio_metrics.get('avg_metadata_completeness', 0):.1f}%"
        )
        print()

        if "total_anomalies" in portfolio_metrics:
            print(f"🤖 ML Anomaly Detection:")
            print(f"  Total Anomalies:  {portfolio_metrics.get('total_anomalies', 0):,}")
            print(
                f"  Avg Anomaly Rate: {portfolio_metrics.get('avg_anomaly_rate', 0):.1%}"
            )
            print(
                f"  Median Rate:      {portfolio_metrics.get('median_anomaly_rate', 0):.1%}"
            )
            print()

        print(f"⚡ Performance:")
        print(
            f"  Total Runtime:   {portfolio_metrics.get('total_runtime_seconds', 0):.1f}s"
        )
        print(
            f"  Avg per Model:   {portfolio_metrics.get('avg_runtime_seconds', 0):.1f}s"
        )
        print(
            f"  P95 Runtime:     {portfolio_metrics.get('p95_runtime_seconds', 0):.1f}s"
        )
        print()

        # Top offenders
        if "top_models_by_critical" in portfolio_metrics:
            print(f"🔴 Top 5 Models by Critical Issues:")
            for i, model in enumerate(portfolio_metrics["top_models_by_critical"], 1):
                print(
                    f"  {i}. {model['model_name']:40s} {model['critical_issues']:3d} critical"
                )
            print()

        if "top_models_by_issues" in portfolio_metrics:
            print(f"📊 Top 5 Models by Total Issues:")
            for i, model in enumerate(portfolio_metrics["top_models_by_issues"], 1):
                print(
                    f"  {i}. {model['model_name']:40s} {model['total_issues']:3d} issues"
                )
            print()

        # Risk assessment
        below_threshold = portfolio_metrics.get("models_below_threshold", 0)
        threshold = portfolio_metrics.get("quality_threshold", 80.0)
        total_success = portfolio_metrics.get("success_models", 0)

        print(f"⚠️  Risk Assessment:")
        print(f"  Models Below Threshold ({threshold}):")
        print(f"    Count: {below_threshold}/{total_success}")
        if total_success > 0:
            pct = below_threshold / total_success * 100
            print(f"    Percentage: {pct:.1f}%")
        print()

    print("=" * 70)
    print("Output Files")
    print("=" * 70)
    print(f"  Portfolio Summary CSV:  {output_dir / 'portfolio_summary.csv'}")
    print(f"  Portfolio Summary JSON: {output_dir / 'portfolio_summary.json'}")
    print(f"  Portfolio Metrics:      {output_dir / 'portfolio_metrics.json'}")
    print()
    print(f"  Per-Model Outputs:      {output_dir / '<model_name>'}/*")
    print("=" * 70)
    print()
    print("Next Steps:")
    print("  1. Review portfolio_summary.csv for model-by-model comparison")
    print("  2. Check portfolio_metrics.json for aggregate KPIs")
    print("  3. Investigate models below quality threshold")
    print("  4. Review top offenders for priority fixes")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/example_batch_processing.py <input_directory>")
        print("\nExample:")
        print("  python examples/example_batch_processing.py examples/ifc_files")
        sys.exit(1)

    input_directory = Path(sys.argv[1])
    if not input_directory.exists():
        print(f"Error: Directory not found: {input_directory}")
        sys.exit(1)

    main(input_directory)
