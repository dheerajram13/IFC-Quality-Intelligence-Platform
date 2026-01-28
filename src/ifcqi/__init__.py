"""IFC Quality Intelligence Platform.

A lightweight Python system for automated BIM model validation,
quality metrics, and anomaly detection.
"""

__version__ = "0.1.0"
__author__ = "Dheeraj Srirama"
__email__ = "your.email@example.com"

from ifcqi.config import get_config, load_config
from ifcqi.logger import get_logger, setup_logging

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "get_config",
    "load_config",
    "get_logger",
    "setup_logging",
]
