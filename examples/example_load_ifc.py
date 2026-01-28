"""Example script demonstrating IFC loading and element extraction.

This script shows how to use the ifcqi.ifc_loader module to:
1. Load an IFC file
2. Extract element information
3. Generate summary statistics
4. Save results to CSV and JSON

Usage:
    python examples/example_load_ifc.py path/to/model.ifc
"""

import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ifcqi.ifc_loader import (
    extract_elements,
    generate_summary_report,
    get_element_summary,
    get_model_metadata,
    load_ifc,
    save_elements_to_csv,
    validate_ifc_file,
)
from ifcqi.logger import setup_logging


def main() -> None:
    """Main function to demonstrate IFC loading."""
    # Setup logging
    setup_logging(level="INFO")

    # Get IFC file path from command line or use example
    if len(sys.argv) > 1:
        ifc_path = Path(sys.argv[1])
    else:
        print("Usage: python example_load_ifc.py <path_to_ifc_file>")
        print("\nExample IFC files:")
        print("  - Download from: https://github.com/buildingSMART/Sample-Test-Files")
        sys.exit(1)

    # Validate IFC file
    print(f"\n{'='*60}")
    print("Step 1: Validating IFC file...")
    print(f"{'='*60}")

    is_valid, error = validate_ifc_file(ifc_path)
    if not is_valid:
        print(f"❌ Invalid IFC file: {error}")
        sys.exit(1)

    print(f"✓ Valid IFC file: {ifc_path}")

    # Load IFC file
    print(f"\n{'='*60}")
    print("Step 2: Loading IFC file...")
    print(f"{'='*60}")

    model = load_ifc(ifc_path)
    print(f"✓ Loaded IFC model (Schema: {model.schema})")

    # Extract metadata
    print(f"\n{'='*60}")
    print("Step 3: Extracting metadata...")
    print(f"{'='*60}")

    metadata = get_model_metadata(model)
    print(f"\nModel Information:")
    print(f"  Schema:              {metadata.get('schema')}")
    print(f"  Name:                {metadata.get('name')}")
    print(f"  Author:              {metadata.get('author')}")
    print(f"  Organization:        {metadata.get('organization')}")
    print(f"  Originating System:  {metadata.get('originating_system')}")
    print(f"  Time Stamp:          {metadata.get('time_stamp')}")

    # Extract elements
    print(f"\n{'='*60}")
    print("Step 4: Extracting elements...")
    print(f"{'='*60}")

    df = extract_elements(model)
    print(f"✓ Extracted {len(df)} elements")

    # Show sample elements
    print(f"\nSample Elements (first 5):")
    print(df.head().to_string())

    # Generate summary
    print(f"\n{'='*60}")
    print("Step 5: Generating summary statistics...")
    print(f"{'='*60}")

    summary = get_element_summary(df)
    print(f"\nElement Counts by Type:")
    for ifc_type, count in sorted(summary.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {ifc_type:25s} : {count:5d}")

    if len(summary) > 10:
        print(f"  ... and {len(summary) - 10} more types")

    # Save results
    print(f"\n{'='*60}")
    print("Step 6: Saving results...")
    print(f"{'='*60}")

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Save elements CSV
    csv_path = output_dir / "elements.csv"
    save_elements_to_csv(df, csv_path)
    print(f"✓ Saved elements to: {csv_path}")

    # Save summary JSON
    json_path = output_dir / "summary.json"
    generate_summary_report(model, df, json_path)
    print(f"✓ Saved summary to: {json_path}")

    # Final statistics
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total Elements:       {len(df)}")
    print(f"Unique Types:         {len(summary)}")
    print(f"Elements with Name:   {df['name'].notna().sum()} ({df['name'].notna().sum() / len(df) * 100:.1f}%)")
    print(f"Elements with Type:   {df['object_type'].notna().sum()} ({df['object_type'].notna().sum() / len(df) * 100:.1f}%)")
    print(f"\n✓ IFC loading and extraction completed successfully!")


if __name__ == "__main__":
    main()
