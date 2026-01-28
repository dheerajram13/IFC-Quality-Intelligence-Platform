"""Example script demonstrating quality checks on IFC models.

This script shows how to:
1. Load an IFC file
2. Extract elements and geometry
3. Run quality validation checks
4. Generate issue reports

Usage:
    python examples/example_quality_checks.py path/to/model.ifc
"""

import json
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ifcqi.checks import (
    get_issues_summary,
    issues_to_dataframe,
    run_all_checks,
)
from ifcqi.features import compute_geometry_features
from ifcqi.ifc_loader import extract_elements, load_ifc
from ifcqi.logger import setup_logging


def main() -> None:
    """Main function to demonstrate quality checks."""
    # Setup logging
    setup_logging(level="INFO")

    # Get IFC file path from command line
    if len(sys.argv) > 1:
        ifc_path = Path(sys.argv[1])
    else:
        print("Usage: python example_quality_checks.py <path_to_ifc_file>")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print("IFC Quality Validation")
    print(f"{'=' * 70}")

    # Step 1: Load IFC file
    print(f"\nStep 1: Loading IFC file...")
    print(f"{'=' * 70}")

    model = load_ifc(ifc_path)
    print(f"✓ Loaded IFC model (Schema: {model.schema})")

    # Step 2: Extract elements
    print(f"\nStep 2: Extracting elements...")
    print(f"{'=' * 70}")

    elements_df = extract_elements(model)
    print(f"✓ Extracted {len(elements_df)} elements")

    # Step 3: Extract geometry features
    print(f"\nStep 3: Extracting geometry features...")
    print(f"{'=' * 70}")

    features_df = compute_geometry_features(model, elements_df)
    print(f"✓ Geometry features extracted")

    # Step 4: Run quality checks
    print(f"\nStep 4: Running quality checks...")
    print(f"{'=' * 70}")

    issues = run_all_checks(
        features_df,
        min_dimension=0.001,  # 1mm
        max_dimension=10000.0,  # 10km
        max_distance=100000.0,  # 100km from origin
        max_aspect_ratio=100.0,  # 100:1 ratio
    )

    print(f"✓ Quality checks complete")

    # Step 5: Analyze results
    print(f"\nStep 5: Quality Check Results")
    print(f"{'=' * 70}")

    if len(issues) == 0:
        print("\n🎉 No issues found! Model quality is excellent.")
    else:
        summary = get_issues_summary(issues)

        print(f"\nIssue Summary:")
        print(f"  Total Issues:     {summary['total_issues']}")
        print(f"  Critical:         {summary['critical_count']}")
        print(f"  Major:            {summary['major_count']}")
        print(f"  Minor:            {summary['minor_count']}")

        print(f"\nIssues by Type:")
        for issue_type, count in sorted(
            summary["by_type"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  {issue_type:30s}: {count:5d}")

        # Show sample critical and major issues
        issues_df = issues_to_dataframe(issues)

        critical = issues_df[issues_df["severity"] == "critical"]
        if len(critical) > 0:
            print(f"\n⚠️  Critical Issues (showing first 5):")
            print(
                critical.head()[
                    ["ifc_type", "issue_type", "message", "element_name"]
                ].to_string(index=False)
            )

        major = issues_df[issues_df["severity"] == "major"]
        if len(major) > 0:
            print(f"\n⚠️  Major Issues (showing first 5):")
            print(
                major.head()[
                    ["ifc_type", "issue_type", "message", "element_name"]
                ].to_string(index=False)
            )

    # Step 6: Save results
    print(f"\nStep 6: Saving results...")
    print(f"{'=' * 70}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save issues DataFrame
    if len(issues) > 0:
        issues_df = issues_to_dataframe(issues)
        issues_path = output_dir / "issues.csv"
        issues_df.to_csv(issues_path, index=False)
        print(f"✓ Saved issues to: {issues_path}")

        # Save summary as JSON
        summary = get_issues_summary(issues)
        summary_path = output_dir / "issues_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary to: {summary_path}")
    else:
        print("✓ No issues to save")

    # Step 7: Quality metrics
    print(f"\nStep 7: Quality Metrics")
    print(f"{'=' * 70}")

    total_elements = len(features_df)
    elements_with_issues = len(issues_df["global_id"].unique()) if len(issues) > 0 else 0
    clean_elements = total_elements - elements_with_issues

    print(f"\nModel Health:")
    print(f"  Total Elements:         {total_elements}")
    print(f"  Elements with Issues:   {elements_with_issues} "
          f"({elements_with_issues / total_elements * 100:.1f}%)")
    print(f"  Clean Elements:         {clean_elements} "
          f"({clean_elements / total_elements * 100:.1f}%)")

    if len(issues) > 0:
        issue_rate = len(issues) / total_elements * 1000
        print(f"  Issue Rate:             {issue_rate:.2f} per 1,000 elements")

    # Quality score (simple version)
    if len(issues) > 0:
        # Weight by severity
        severity_weights = {"critical": 3.0, "major": 2.0, "minor": 1.0}
        weighted_issues = sum(
            severity_weights.get(issue.severity.value, 1.0) for issue in issues
        )
        quality_score = max(0, 100 - (weighted_issues / total_elements * 100))
    else:
        quality_score = 100.0

    print(f"  Quality Score:          {quality_score:.1f}/100")

    # Final summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")

    if quality_score >= 90:
        print(f"✓ Excellent quality! Score: {quality_score:.1f}/100")
    elif quality_score >= 70:
        print(f"⚠️  Good quality with some issues. Score: {quality_score:.1f}/100")
    elif quality_score >= 50:
        print(f"⚠️  Moderate quality, needs attention. Score: {quality_score:.1f}/100")
    else:
        print(f"❌ Poor quality, significant issues found. Score: {quality_score:.1f}/100")

    if len(issues) > 0:
        print(f"\nIssues found: {len(issues)}")
        print(f"  Critical: {summary['critical_count']}")
        print(f"  Major:    {summary['major_count']}")
        print(f"  Minor:    {summary['minor_count']}")
        print(f"\nRecommendation: Review and fix critical and major issues")
    else:
        print("\nNo issues found! Model meets all quality standards.")


if __name__ == "__main__":
    main()
