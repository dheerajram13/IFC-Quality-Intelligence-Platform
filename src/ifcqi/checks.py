"""Rule-based quality checks for IFC models.

This module implements validation rules to detect data quality issues including
missing metadata, invalid geometry, duplicates, and coordinate anomalies.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

import pandas as pd

from ifcqi.logger import get_logger

logger = get_logger(__name__)


class Severity(str, Enum):
    """Issue severity levels."""

    CRITICAL = "critical"
    MAJOR = "major"
    MINOR = "minor"


@dataclass
class Issue:
    """Represents a quality issue found in an IFC element.

    Attributes:
        global_id: Element's GlobalId
        ifc_type: Element's IFC type
        issue_type: Type of issue (e.g., "missing_name")
        severity: Issue severity (critical, major, minor)
        message: Human-readable description
        element_name: Element name (if available)
    """

    global_id: str
    ifc_type: str
    issue_type: str
    severity: Severity
    message: str
    element_name: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert issue to dictionary."""
        return {
            "global_id": self.global_id,
            "ifc_type": self.ifc_type,
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "message": self.message,
            "element_name": self.element_name,
        }


def check_missing_name(df: pd.DataFrame) -> List[Issue]:
    """Check for elements with missing or empty names.

    Args:
        df: DataFrame with elements (must have 'name', 'global_id', 'ifc_type' columns).

    Returns:
        List of Issue objects for elements with missing names.

    Example:
        >>> issues = check_missing_name(elements_df)
        >>> print(f"Found {len(issues)} elements without names")
    """
    issues = []

    # Find elements with missing names (None or empty string)
    missing_name = df[(df["name"].isna()) | (df["name"] == "")]

    for _, row in missing_name.iterrows():
        issue = Issue(
            global_id=row["global_id"],
            ifc_type=row["ifc_type"],
            issue_type="missing_name",
            severity=Severity.MAJOR,
            message="Element has no name assigned",
            element_name=None,
        )
        issues.append(issue)

    logger.info(f"Found {len(issues)} elements with missing names")
    return issues


def check_missing_object_type(df: pd.DataFrame) -> List[Issue]:
    """Check for elements with missing ObjectType attribute.

    Args:
        df: DataFrame with elements (must have 'object_type' column).

    Returns:
        List of Issue objects for elements with missing object types.
    """
    issues = []

    # Find elements with missing object_type
    missing_type = df[(df["object_type"].isna()) | (df["object_type"] == "")]

    for _, row in missing_type.iterrows():
        issue = Issue(
            global_id=row["global_id"],
            ifc_type=row["ifc_type"],
            issue_type="missing_object_type",
            severity=Severity.MINOR,
            message="Element has no ObjectType assigned",
            element_name=row.get("name"),
        )
        issues.append(issue)

    logger.info(f"Found {len(issues)} elements with missing object types")
    return issues


def check_duplicate_global_ids(df: pd.DataFrame) -> List[Issue]:
    """Check for duplicate GlobalIds (critical data integrity issue).

    Args:
        df: DataFrame with elements (must have 'global_id' column).

    Returns:
        List of Issue objects for duplicate GlobalIds.
    """
    issues = []

    # Find duplicate global_ids
    duplicates = df[df.duplicated(subset=["global_id"], keep=False)]

    if len(duplicates) > 0:
        for global_id in duplicates["global_id"].unique():
            dup_elements = df[df["global_id"] == global_id]
            count = len(dup_elements)

            # Create issue for each duplicate
            for _, row in dup_elements.iterrows():
                issue = Issue(
                    global_id=row["global_id"],
                    ifc_type=row["ifc_type"],
                    issue_type="duplicate_global_id",
                    severity=Severity.CRITICAL,
                    message=f"GlobalId appears {count} times (must be unique)",
                    element_name=row.get("name"),
                )
                issues.append(issue)

    logger.info(f"Found {len(issues)} duplicate GlobalIds")
    return issues


def check_degenerate_geometry(
    df: pd.DataFrame, min_dimension: float = 0.001
) -> List[Issue]:
    """Check for degenerate geometry (zero or near-zero dimensions).

    Args:
        df: DataFrame with geometry features (must have dim_x, dim_y, dim_z columns).
        min_dimension: Minimum valid dimension in meters (default 1mm).

    Returns:
        List of Issue objects for elements with degenerate geometry.
    """
    issues = []

    # Check for elements with geometry
    if "has_geometry" not in df.columns:
        logger.warning("DataFrame missing 'has_geometry' column, skipping check")
        return issues

    geom_df = df[df["has_geometry"] == True]

    # Check each dimension
    for dim in ["dim_x", "dim_y", "dim_z"]:
        if dim not in geom_df.columns:
            continue

        degenerate = geom_df[
            (geom_df[dim].notna()) & (geom_df[dim] < min_dimension)
        ]

        for _, row in degenerate.iterrows():
            issue = Issue(
                global_id=row["global_id"],
                ifc_type=row["ifc_type"],
                issue_type="degenerate_geometry",
                severity=Severity.MAJOR,
                message=f"Dimension {dim} = {row[dim]:.6f} (< {min_dimension}m)",
                element_name=row.get("name"),
            )
            issues.append(issue)

    logger.info(f"Found {len(issues)} elements with degenerate geometry")
    return issues


def check_extreme_dimensions(
    df: pd.DataFrame, max_dimension: float = 10000.0
) -> List[Issue]:
    """Check for unusually large dimensions (possible scale/unit issues).

    Args:
        df: DataFrame with geometry features.
        max_dimension: Maximum reasonable dimension in meters (default 10km).

    Returns:
        List of Issue objects for elements with extreme dimensions.
    """
    issues = []

    if "has_geometry" not in df.columns:
        return issues

    geom_df = df[df["has_geometry"] == True]

    # Check each dimension
    for dim in ["dim_x", "dim_y", "dim_z"]:
        if dim not in geom_df.columns:
            continue

        extreme = geom_df[(geom_df[dim].notna()) & (geom_df[dim] > max_dimension)]

        for _, row in extreme.iterrows():
            issue = Issue(
                global_id=row["global_id"],
                ifc_type=row["ifc_type"],
                issue_type="extreme_dimension",
                severity=Severity.CRITICAL,
                message=f"Dimension {dim} = {row[dim]:.2f}m (> {max_dimension}m, possible scale issue)",
                element_name=row.get("name"),
            )
            issues.append(issue)

    logger.info(f"Found {len(issues)} elements with extreme dimensions")
    return issues


def check_coordinate_anomalies(
    df: pd.DataFrame, max_distance: float = 100000.0
) -> List[Issue]:
    """Check for elements too far from origin (possible coordinate issues).

    Args:
        df: DataFrame with geometry features (centroid columns).
        max_distance: Maximum reasonable distance from origin in meters (default 100km).

    Returns:
        List of Issue objects for elements with anomalous coordinates.
    """
    issues = []

    if "has_geometry" not in df.columns:
        return issues

    geom_df = df[df["has_geometry"] == True]

    # Check if centroid columns exist
    required_cols = ["centroid_x", "centroid_y", "centroid_z"]
    if not all(col in geom_df.columns for col in required_cols):
        logger.warning("Missing centroid columns, skipping coordinate check")
        return issues

    # Calculate distance from origin
    import numpy as np

    geom_df = geom_df.copy()
    geom_df["distance_from_origin"] = np.sqrt(
        geom_df["centroid_x"] ** 2
        + geom_df["centroid_y"] ** 2
        + geom_df["centroid_z"] ** 2
    )

    # Find anomalies
    anomalies = geom_df[geom_df["distance_from_origin"] > max_distance]

    for _, row in anomalies.iterrows():
        issue = Issue(
            global_id=row["global_id"],
            ifc_type=row["ifc_type"],
            issue_type="coordinate_anomaly",
            severity=Severity.MINOR,
            message=f"Element {row['distance_from_origin']:.2f}m from origin (> {max_distance}m)",
            element_name=row.get("name"),
        )
        issues.append(issue)

    logger.info(f"Found {len(issues)} elements with coordinate anomalies")
    return issues


def check_extreme_aspect_ratios(
    df: pd.DataFrame, max_ratio: float = 100.0
) -> List[Issue]:
    """Check for elements with extreme aspect ratios (very thin/flat objects).

    Args:
        df: DataFrame with geometry features (aspect_ratio columns).
        max_ratio: Maximum reasonable aspect ratio (default 100:1).

    Returns:
        List of Issue objects for elements with extreme aspect ratios.
    """
    issues = []

    if "has_geometry" not in df.columns:
        return issues

    geom_df = df[df["has_geometry"] == True]

    # Check aspect ratios
    for ratio_col in ["aspect_ratio_xy", "aspect_ratio_yz", "aspect_ratio_xz"]:
        if ratio_col not in geom_df.columns:
            continue

        extreme = geom_df[
            (geom_df[ratio_col].notna())
            & ((geom_df[ratio_col] > max_ratio) | (geom_df[ratio_col] < 1 / max_ratio))
        ]

        for _, row in extreme.iterrows():
            issue = Issue(
                global_id=row["global_id"],
                ifc_type=row["ifc_type"],
                issue_type="extreme_aspect_ratio",
                severity=Severity.MINOR,
                message=f"{ratio_col} = {row[ratio_col]:.2f} (extreme, possibly incorrect)",
                element_name=row.get("name"),
            )
            issues.append(issue)

    logger.info(f"Found {len(issues)} elements with extreme aspect ratios")
    return issues


def run_all_checks(
    df: pd.DataFrame,
    min_dimension: float = 0.001,
    max_dimension: float = 10000.0,
    max_distance: float = 100000.0,
    max_aspect_ratio: float = 100.0,
) -> List[Issue]:
    """Run all quality checks on a DataFrame.

    Args:
        df: DataFrame with elements and geometry features.
        min_dimension: Minimum valid dimension for degenerate check.
        max_dimension: Maximum valid dimension for extreme check.
        max_distance: Maximum distance from origin for coordinate check.
        max_aspect_ratio: Maximum aspect ratio for thin/flat object check.

    Returns:
        List of all issues found.

    Example:
        >>> from ifcqi.ifc_loader import load_ifc, extract_elements
        >>> from ifcqi.features import compute_geometry_features
        >>> from ifcqi.checks import run_all_checks
        >>>
        >>> model = load_ifc(Path("model.ifc"))
        >>> elements_df = extract_elements(model)
        >>> features_df = compute_geometry_features(model, elements_df)
        >>> issues = run_all_checks(features_df)
        >>> print(f"Found {len(issues)} total issues")
    """
    logger.info("Running all quality checks...")

    all_issues = []

    # Metadata checks (don't require geometry)
    all_issues.extend(check_missing_name(df))
    all_issues.extend(check_missing_object_type(df))
    all_issues.extend(check_duplicate_global_ids(df))

    # Geometry checks (require geometry features)
    if "has_geometry" in df.columns:
        all_issues.extend(check_degenerate_geometry(df, min_dimension))
        all_issues.extend(check_extreme_dimensions(df, max_dimension))
        all_issues.extend(check_coordinate_anomalies(df, max_distance))
        all_issues.extend(check_extreme_aspect_ratios(df, max_aspect_ratio))
    else:
        logger.warning("No geometry features found, skipping geometry checks")

    logger.info(f"Total issues found: {len(all_issues)}")

    return all_issues


def issues_to_dataframe(issues: List[Issue]) -> pd.DataFrame:
    """Convert list of issues to DataFrame.

    Args:
        issues: List of Issue objects.

    Returns:
        DataFrame with issue details.

    Example:
        >>> issues = run_all_checks(features_df)
        >>> issues_df = issues_to_dataframe(issues)
        >>> issues_df.to_csv("output/issues.csv", index=False)
    """
    if not issues:
        return pd.DataFrame(
            columns=[
                "global_id",
                "ifc_type",
                "issue_type",
                "severity",
                "message",
                "element_name",
            ]
        )

    return pd.DataFrame([issue.to_dict() for issue in issues])


def get_issues_summary(issues: List[Issue]) -> dict:
    """Get summary statistics of issues by type and severity.

    Args:
        issues: List of Issue objects.

    Returns:
        Dictionary with summary statistics.

    Example:
        >>> issues = run_all_checks(features_df)
        >>> summary = get_issues_summary(issues)
        >>> print(summary)
    """
    if not issues:
        return {
            "total_issues": 0,
            "by_severity": {},
            "by_type": {},
            "critical_count": 0,
            "major_count": 0,
            "minor_count": 0,
        }

    issues_df = issues_to_dataframe(issues)

    summary = {
        "total_issues": len(issues),
        "by_severity": issues_df["severity"].value_counts().to_dict(),
        "by_type": issues_df["issue_type"].value_counts().to_dict(),
        "critical_count": int((issues_df["severity"] == "critical").sum()),
        "major_count": int((issues_df["severity"] == "major").sum()),
        "minor_count": int((issues_df["severity"] == "minor").sum()),
    }

    return summary
