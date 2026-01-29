"""Example: Generate visualizations and HTML report from IFC file.

This example demonstrates Module 5 (Dashboard/Visualization) capabilities:
1. Load IFC file
2. Extract elements and geometry features
3. Run quality checks
4. Build metrics
5. Create visualizations
6. Export to HTML report

Usage:
    python examples/example_dashboard.py examples/ifc_files/Duplex_MEP_20110907.ifc
"""

import sys
from pathlib import Path

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


def main(ifc_path: Path, output_dir: Path = Path("output")) -> None:
    """Run the full dashboard generation pipeline."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("IFC Quality Intelligence - Dashboard Generation")
    print("=" * 70)
    print()

    # Step 1: Load IFC
    print("Step 1: Loading IFC file...")
    print("-" * 70)
    model = load_ifc(ifc_path)
    print(f"✓ Loaded IFC model: {ifc_path.name}")
    print()

    # Step 2: Extract elements
    print("Step 2: Extracting elements...")
    print("-" * 70)
    elements_df = extract_elements(model)
    print(f"✓ Extracted {len(elements_df)} elements")
    print()

    # Step 3: Extract geometry features
    print("Step 3: Extracting geometry features...")
    print("-" * 70)
    features_df = compute_geometry_features(model, elements_df)
    print("✓ Geometry features extracted")
    print()

    # Step 4: Run quality checks
    print("Step 4: Running quality checks...")
    print("-" * 70)
    issues = run_all_checks(features_df)
    issues_df = issues_to_dataframe(issues)
    print(f"✓ Found {len(issues_df)} quality issues")
    print()

    # Step 5: Build metrics
    print("Step 5: Building metrics...")
    print("-" * 70)
    metrics, metrics_dfs = build_metrics(features_df, issues_df, top_n=10)
    print(f"✓ Quality Score: {metrics['quality_score']:.1f}/100")
    print(f"  Total Elements: {metrics['total_elements']}")
    print(f"  Total Issues: {metrics['total_issues']}")
    print(f"  Issue Rate: {metrics['issue_rate_per_1000']:.2f} per 1,000 elements")
    print()

    # Step 6: Save outputs
    print("Step 6: Saving data outputs...")
    print("-" * 70)
    features_df.to_parquet(output_dir / "features.parquet", index=False)
    issues_df.to_csv(output_dir / "issues.csv", index=False)
    save_metrics(metrics, output_dir / "metrics.json")
    print(f"✓ Saved features.parquet to {output_dir}")
    print(f"✓ Saved issues.csv to {output_dir}")
    print(f"✓ Saved metrics.json to {output_dir}")
    print()

    # Step 7: Create visualizations
    print("Step 7: Creating visualizations...")
    print("-" * 70)
    figures = {
        "Severity Breakdown": build_severity_chart(metrics_dfs["severity"]),
        "Issues by IFC Type": build_issues_by_ifc_type_chart(metrics_dfs["issues_by_ifc_type"]),
        "Issues by Type": build_issues_by_type_chart(metrics_dfs["issues_by_type"]),
        "Pareto Analysis": build_pareto_chart(metrics_dfs["pareto_issue_types"]),
    }
    print(f"✓ Created {len(figures)} visualizations")
    print()

    # Step 8: Export HTML report
    print("Step 8: Exporting HTML report...")
    print("-" * 70)
    html_path = output_dir / f"{ifc_path.stem}_report.html"
    export_figures_to_html(figures, html_path)
    print(f"✓ HTML report saved to: {html_path}")
    print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"IFC File: {ifc_path.name}")
    print(f"Quality Score: {metrics['quality_score']:.1f}/100")
    print(f"Total Elements: {metrics['total_elements']}")
    print(f"Total Issues: {metrics['total_issues']}")
    print()
    print("Severity Breakdown:")
    print(f"  Critical: {metrics['severity_breakdown']['critical']}")
    print(f"  Major:    {metrics['severity_breakdown']['major']}")
    print(f"  Minor:    {metrics['severity_breakdown']['minor']}")
    print()
    print("Outputs:")
    print(f"  Data:   {output_dir}")
    print(f"  Report: {html_path}")
    print()
    print("Next steps:")
    print(f"  1. Open {html_path} in a browser to view the dashboard")
    print(f"  2. Run: streamlit run apps/dashboard.py")
    print("=" * 70)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/example_dashboard.py <ifc_file_path>")
        sys.exit(1)

    ifc_file = Path(sys.argv[1])
    if not ifc_file.exists():
        print(f"Error: File not found: {ifc_file}")
        sys.exit(1)

    main(ifc_file)
