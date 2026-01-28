"""IFC file loading and element extraction.

This module provides functionality to load IFC files and extract element metadata
using IfcOpenShell library.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import ifcopenshell
import pandas as pd

from ifcqi.logger import get_logger

logger = get_logger(__name__)


class IFCLoadError(Exception):
    """Exception raised when IFC file cannot be loaded."""

    pass


def load_ifc(file_path: Path) -> ifcopenshell.file:
    """Load an IFC file using IfcOpenShell.

    Args:
        file_path: Path to the IFC file.

    Returns:
        ifcopenshell.file: Loaded IFC model.

    Raises:
        FileNotFoundError: If the IFC file doesn't exist.
        IFCLoadError: If the file cannot be parsed as valid IFC.

    Example:
        >>> model = load_ifc(Path("model.ifc"))
        >>> print(f"Loaded {model.schema} model")
        Loaded IFC4 model
    """
    if not file_path.exists():
        raise FileNotFoundError(f"IFC file not found: {file_path}")

    logger.info(f"Loading IFC file: {file_path}")

    try:
        model = ifcopenshell.open(str(file_path))
        logger.info(
            f"Successfully loaded IFC file - Schema: {model.schema}, "
            f"Elements: {len(model.by_type('IfcProduct'))}"
        )
        return model
    except Exception as e:
        raise IFCLoadError(f"Failed to load IFC file {file_path}: {str(e)}") from e


def extract_elements(
    model: ifcopenshell.file, max_elements: Optional[int] = None
) -> pd.DataFrame:
    """Extract basic element information from IFC model.

    Args:
        model: Loaded IFC model from IfcOpenShell.
        max_elements: Maximum number of elements to extract (None = all).

    Returns:
        pd.DataFrame: DataFrame with columns:
            - global_id: Unique identifier (GlobalId)
            - ifc_type: IFC entity type (e.g., IfcWall, IfcDoor)
            - name: Element name (may be None)
            - description: Element description (may be None)
            - object_type: ObjectType attribute (may be None)
            - predefined_type: PredefinedType attribute (may be None)

    Example:
        >>> model = load_ifc(Path("model.ifc"))
        >>> df = extract_elements(model)
        >>> print(df.head())
        >>> print(f"Total elements: {len(df)}")
    """
    logger.info("Extracting elements from IFC model...")

    # Get all IFC products (physical elements)
    products = model.by_type("IfcProduct")

    if max_elements is not None:
        products = products[:max_elements]
        logger.info(f"Limited extraction to {max_elements} elements")

    elements = []

    for product in products:
        try:
            element_data = {
                "global_id": product.GlobalId,
                "ifc_type": product.is_a(),
                "name": getattr(product, "Name", None),
                "description": getattr(product, "Description", None),
                "object_type": getattr(product, "ObjectType", None),
                "predefined_type": getattr(product, "PredefinedType", None),
            }
            elements.append(element_data)

        except Exception as e:
            logger.warning(
                f"Failed to extract element {getattr(product, 'GlobalId', 'unknown')}: {e}"
            )
            continue

    df = pd.DataFrame(elements)
    logger.info(f"Extracted {len(df)} elements")

    return df


def get_element_summary(df: pd.DataFrame) -> Dict[str, int]:
    """Generate summary statistics of element types.

    Args:
        df: DataFrame with extracted elements (from extract_elements).

    Returns:
        Dict mapping IFC type to count, sorted by count descending.

    Example:
        >>> df = extract_elements(model)
        >>> summary = get_element_summary(df)
        >>> for ifc_type, count in summary.items():
        ...     print(f"{ifc_type}: {count}")
        IfcWall: 152
        IfcDoor: 48
        IfcWindow: 36
    """
    if df.empty:
        logger.warning("Empty DataFrame provided for summary")
        return {}

    summary = df["ifc_type"].value_counts().to_dict()
    logger.info(f"Generated summary for {len(summary)} unique element types")

    return summary


def get_model_metadata(model: ifcopenshell.file) -> Dict[str, str]:
    """Extract metadata from IFC model header.

    Args:
        model: Loaded IFC model.

    Returns:
        Dict with model metadata including schema, application, author, etc.

    Example:
        >>> model = load_ifc(Path("model.ifc"))
        >>> metadata = get_model_metadata(model)
        >>> print(metadata["schema"])
        IFC4
    """
    try:
        header = model.header
        file_description = header.file_description
        file_name = header.file_name

        metadata = {
            "schema": model.schema,
            "description": file_description.description[0] if file_description.description else None,
            "implementation_level": file_description.implementation_level if file_description else None,
            "name": file_name.name if file_name else None,
            "time_stamp": file_name.time_stamp if file_name else None,
            "author": file_name.author[0] if file_name and file_name.author else None,
            "organization": file_name.organization[0] if file_name and file_name.organization else None,
            "preprocessor_version": file_name.preprocessor_version if file_name else None,
            "originating_system": file_name.originating_system if file_name else None,
            "authorization": file_name.authorization if file_name else None,
        }

        logger.info(f"Extracted metadata - Schema: {metadata['schema']}, Author: {metadata['author']}")
        return metadata

    except Exception as e:
        logger.warning(f"Failed to extract full metadata: {e}")
        return {"schema": model.schema}


def save_elements_to_csv(df: pd.DataFrame, output_path: Path) -> None:
    """Save extracted elements to CSV file.

    Args:
        df: DataFrame with extracted elements.
        output_path: Path where to save the CSV file.

    Example:
        >>> df = extract_elements(model)
        >>> save_elements_to_csv(df, Path("output/elements.csv"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} elements to {output_path}")


def generate_summary_report(
    model: ifcopenshell.file, df: pd.DataFrame, output_path: Path
) -> None:
    """Generate a JSON summary report of the IFC model.

    Args:
        model: Loaded IFC model.
        df: DataFrame with extracted elements.
        output_path: Path where to save the JSON report.

    Example:
        >>> model = load_ifc(Path("model.ifc"))
        >>> df = extract_elements(model)
        >>> generate_summary_report(model, df, Path("output/summary.json"))
    """
    import json

    metadata = get_model_metadata(model)
    summary = get_element_summary(df)

    report = {
        "metadata": metadata,
        "statistics": {
            "total_elements": len(df),
            "unique_types": len(summary),
            "elements_by_type": summary,
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Generated summary report: {output_path}")


def validate_ifc_file(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate that a file is a valid IFC file.

    Args:
        file_path: Path to the file to validate.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is None.

    Example:
        >>> is_valid, error = validate_ifc_file(Path("model.ifc"))
        >>> if not is_valid:
        ...     print(f"Invalid: {error}")
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    if not file_path.is_file():
        return False, f"Path is not a file: {file_path}"

    if file_path.suffix.lower() not in [".ifc", ".ifczip"]:
        return False, f"File extension must be .ifc or .ifczip, got: {file_path.suffix}"

    try:
        model = ifcopenshell.open(str(file_path))
        # Try to access basic model properties
        _ = model.schema
        _ = len(model.by_type("IfcProduct"))
        return True, None

    except Exception as e:
        return False, f"Failed to parse IFC file: {str(e)}"
