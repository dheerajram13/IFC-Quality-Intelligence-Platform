"""Geometry extraction and processing for IFC elements.

This module provides functionality to extract 3D geometry features from IFC elements
including bounding boxes, centroids, volumes, and other spatial properties.
"""

from typing import Optional, Tuple

import ifcopenshell
import ifcopenshell.geom
import numpy as np

from ifcqi.logger import get_logger

logger = get_logger(__name__)


class GeometryExtractionError(Exception):
    """Exception raised when geometry cannot be extracted from an element."""

    pass


def get_element_shape(
    element: ifcopenshell.entity_instance,
    settings: Optional[ifcopenshell.geom.settings] = None,
) -> Optional[ifcopenshell.geom.ShapeType]:
    """Extract geometric shape from an IFC element.

    Args:
        element: IFC element instance.
        settings: IfcOpenShell geometry settings. If None, uses default settings.

    Returns:
        Shape object if geometry exists, None otherwise.

    Example:
        >>> element = model.by_guid("2O2Fr$t4X7Zf8NOew3FL9r")
        >>> shape = get_element_shape(element)
        >>> if shape:
        ...     print(f"Shape extracted with {len(shape.verts)} vertices")
    """
    try:
        if settings is None:
            settings = ifcopenshell.geom.settings()
            settings.set(settings.USE_WORLD_COORDS, True)

        shape = ifcopenshell.geom.create_shape(settings, element)
        return shape

    except Exception as e:
        logger.debug(f"Could not extract shape for element {element.GlobalId}: {e}")
        return None


def extract_bounding_box(
    element: ifcopenshell.entity_instance,
    settings: Optional[ifcopenshell.geom.settings] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract bounding box (min and max coordinates) from an IFC element.

    Args:
        element: IFC element instance.
        settings: IfcOpenShell geometry settings.

    Returns:
        Tuple of (min_point, max_point) as numpy arrays [x, y, z], or None if no geometry.

    Example:
        >>> element = model.by_guid("2O2Fr$t4X7Zf8NOew3FL9r")
        >>> bbox = extract_bounding_box(element)
        >>> if bbox:
        ...     min_pt, max_pt = bbox
        ...     print(f"Box: {min_pt} to {max_pt}")
    """
    shape = get_element_shape(element, settings)

    if shape is None:
        return None

    try:
        # Get vertices from shape - handle different shape types
        if hasattr(shape, 'verts'):
            # Standard shape with verts attribute
            verts = np.array(shape.verts).reshape(-1, 3)
        elif hasattr(shape, 'geometry') and hasattr(shape.geometry, 'verts'):
            # Triangulation element with nested geometry
            verts = np.array(shape.geometry.verts).reshape(-1, 3)
        else:
            logger.debug(f"Unknown shape type for {element.GlobalId}: {type(shape)}")
            return None

        if len(verts) == 0:
            return None

        # Calculate bounding box
        min_point = np.min(verts, axis=0)
        max_point = np.max(verts, axis=0)

        return min_point, max_point

    except Exception as e:
        logger.warning(f"Failed to compute bounding box for {element.GlobalId}: {e}")
        return None


def calculate_centroid(
    element: ifcopenshell.entity_instance,
    settings: Optional[ifcopenshell.geom.settings] = None,
) -> Optional[np.ndarray]:
    """Calculate the centroid (geometric center) of an IFC element.

    Args:
        element: IFC element instance.
        settings: IfcOpenShell geometry settings.

    Returns:
        Centroid as numpy array [x, y, z], or None if no geometry.

    Example:
        >>> element = model.by_guid("2O2Fr$t4X7Zf8NOew3FL9r")
        >>> centroid = calculate_centroid(element)
        >>> if centroid is not None:
        ...     print(f"Center: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
    """
    bbox = extract_bounding_box(element, settings)

    if bbox is None:
        return None

    min_point, max_point = bbox

    # Centroid is the center of the bounding box
    centroid = (min_point + max_point) / 2.0

    return centroid


def calculate_bbox_volume(
    element: ifcopenshell.entity_instance,
    settings: Optional[ifcopenshell.geom.settings] = None,
) -> Optional[float]:
    """Calculate the volume of the bounding box of an element.

    Note: This is NOT the actual volume of the element, just the bounding box volume.
    Useful as a quick approximation or for comparing element sizes.

    Args:
        element: IFC element instance.
        settings: IfcOpenShell geometry settings.

    Returns:
        Bounding box volume in cubic units, or None if no geometry.

    Example:
        >>> element = model.by_guid("2O2Fr$t4X7Zf8NOew3FL9r")
        >>> volume = calculate_bbox_volume(element)
        >>> if volume is not None:
        ...     print(f"Bounding box volume: {volume:.2f} cubic units")
    """
    bbox = extract_bounding_box(element, settings)

    if bbox is None:
        return None

    min_point, max_point = bbox
    dimensions = max_point - min_point

    # Volume = width × height × depth
    volume = float(np.prod(dimensions))

    return volume


def calculate_bbox_dimensions(
    element: ifcopenshell.entity_instance,
    settings: Optional[ifcopenshell.geom.settings] = None,
) -> Optional[Tuple[float, float, float]]:
    """Calculate the dimensions (width, height, depth) of the bounding box.

    Args:
        element: IFC element instance.
        settings: IfcOpenShell geometry settings.

    Returns:
        Tuple of (dx, dy, dz) dimensions, or None if no geometry.

    Example:
        >>> element = model.by_guid("2O2Fr$t4X7Zf8NOew3FL9r")
        >>> dims = calculate_bbox_dimensions(element)
        >>> if dims:
        ...     dx, dy, dz = dims
        ...     print(f"Dimensions: {dx:.2f} × {dy:.2f} × {dz:.2f}")
    """
    bbox = extract_bounding_box(element, settings)

    if bbox is None:
        return None

    min_point, max_point = bbox
    dimensions = max_point - min_point

    dx, dy, dz = float(dimensions[0]), float(dimensions[1]), float(dimensions[2])

    return dx, dy, dz


def calculate_aspect_ratios(
    element: ifcopenshell.entity_instance,
    settings: Optional[ifcopenshell.geom.settings] = None,
) -> Optional[Tuple[float, float, float]]:
    """Calculate aspect ratios between bounding box dimensions.

    Useful for detecting unusual proportions (very thin/flat objects, etc.)

    Args:
        element: IFC element instance.
        settings: IfcOpenShell geometry settings.

    Returns:
        Tuple of (dx/dy, dy/dz, dx/dz) ratios, or None if no geometry.
        Returns None for any ratio if denominator is zero.

    Example:
        >>> element = model.by_guid("2O2Fr$t4X7Zf8NOew3FL9r")
        >>> ratios = calculate_aspect_ratios(element)
        >>> if ratios:
        ...     print(f"Aspect ratios: {ratios}")
    """
    dims = calculate_bbox_dimensions(element, settings)

    if dims is None:
        return None

    dx, dy, dz = dims

    # Avoid division by zero
    epsilon = 1e-10

    ratio_xy = dx / (dy + epsilon) if dy > epsilon else None
    ratio_yz = dy / (dz + epsilon) if dz > epsilon else None
    ratio_xz = dx / (dz + epsilon) if dz > epsilon else None

    return ratio_xy, ratio_yz, ratio_xz


def has_geometry(element: ifcopenshell.entity_instance) -> bool:
    """Check if an element has extractable geometry.

    Args:
        element: IFC element instance.

    Returns:
        True if element has geometry, False otherwise.

    Example:
        >>> element = model.by_guid("2O2Fr$t4X7Zf8NOew3FL9r")
        >>> if has_geometry(element):
        ...     print("Element has geometry")
    """
    shape = get_element_shape(element)
    return shape is not None


def get_geometry_summary(element: ifcopenshell.entity_instance) -> dict:
    """Get a comprehensive summary of an element's geometry features.

    Args:
        element: IFC element instance.

    Returns:
        Dictionary with all geometry features, or dict with has_geometry=False if no geometry.

    Example:
        >>> element = model.by_guid("2O2Fr$t4X7Zf8NOew3FL9r")
        >>> summary = get_geometry_summary(element)
        >>> print(summary)
    """
    if not has_geometry(element):
        return {
            "has_geometry": False,
            "centroid": None,
            "bbox_min": None,
            "bbox_max": None,
            "dimensions": None,
            "bbox_volume": None,
            "aspect_ratios": None,
        }

    bbox = extract_bounding_box(element)
    centroid = calculate_centroid(element)
    dims = calculate_bbox_dimensions(element)
    volume = calculate_bbox_volume(element)
    ratios = calculate_aspect_ratios(element)

    min_pt, max_pt = bbox if bbox else (None, None)

    return {
        "has_geometry": True,
        "centroid": centroid.tolist() if centroid is not None else None,
        "bbox_min": min_pt.tolist() if min_pt is not None else None,
        "bbox_max": max_pt.tolist() if max_pt is not None else None,
        "dimensions": dims,
        "bbox_volume": volume,
        "aspect_ratios": ratios,
    }


def create_geometry_settings(use_world_coords: bool = True) -> ifcopenshell.geom.settings:
    """Create geometry extraction settings with recommended defaults.

    Args:
        use_world_coords: If True, use world coordinates. If False, use local coordinates.

    Returns:
        Configured geometry settings object.

    Example:
        >>> settings = create_geometry_settings(use_world_coords=True)
        >>> shape = get_element_shape(element, settings)
    """
    settings = ifcopenshell.geom.settings()
    settings.set(settings.USE_WORLD_COORDS, use_world_coords)
    settings.set(settings.DISABLE_OPENING_SUBTRACTIONS, False)

    return settings
