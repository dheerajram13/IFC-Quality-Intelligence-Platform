"""Configuration management for IFC Quality Intelligence Platform.

This module handles loading and managing configuration from multiple sources:
- Default configuration (hardcoded)
- YAML configuration files
- Environment variables
- Runtime overrides
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IFCConfig(BaseSettings):
    """IFC processing configuration."""

    extract_geometry: bool = Field(default=True, description="Extract geometry from IFC elements")
    max_elements: Optional[int] = Field(
        default=None, description="Maximum number of elements to process (None = no limit)"
    )


class ChecksGeometryConfig(BaseSettings):
    """Geometry validation thresholds."""

    min_dimension: float = Field(default=0.001, description="Minimum valid dimension in meters")
    max_dimension: float = Field(default=10000.0, description="Maximum valid dimension in meters")
    max_distance_from_origin: float = Field(
        default=100000.0, description="Maximum valid distance from origin in meters"
    )


class ChecksConfig(BaseSettings):
    """Quality checks configuration."""

    severity_weights: Dict[str, float] = Field(
        default={"critical": 3.0, "major": 2.0, "minor": 1.0},
        description="Severity weight multipliers for score calculation",
    )
    geometry: ChecksGeometryConfig = Field(default_factory=ChecksGeometryConfig)


class MLConfig(BaseSettings):
    """Machine learning configuration."""

    enabled: bool = Field(default=True, description="Enable ML-based anomaly detection")
    model_type: str = Field(default="IsolationForest", description="Anomaly detection model type")
    contamination: float = Field(
        default=0.05, description="Expected proportion of outliers in the dataset"
    )
    random_state: int = Field(default=42, description="Random seed for reproducibility")
    n_estimators: int = Field(default=100, description="Number of estimators for ensemble models")


class OutputConfig(BaseSettings):
    """Output configuration."""

    formats: list[str] = Field(
        default=["json", "csv", "html"], description="Output formats to generate"
    )
    include_geometry: bool = Field(
        default=False, description="Include geometry data in outputs (can be large)"
    )


class MLflowConfig(BaseSettings):
    """MLflow tracking configuration."""

    tracking_uri: str = Field(default="./mlruns", description="MLflow tracking URI")
    experiment_name: str = Field(default="ifc-quality", description="MLflow experiment name")
    enable_autolog: bool = Field(
        default=True, description="Enable MLflow autologging for scikit-learn"
    )


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format",
    )
    file_path: Optional[str] = Field(default=None, description="Log file path (None = console only)")


class Config(BaseSettings):
    """Main configuration class for IFC Quality Intelligence Platform."""

    model_config = SettingsConfigDict(
        env_prefix="IFCQI_",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # Sub-configurations
    ifc: IFCConfig = Field(default_factory=IFCConfig)
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    # General settings
    project_name: str = Field(default="IFC Quality Intelligence", description="Project name")
    version: str = Field(default="0.1.0", description="Application version")


def load_config(config_path: Optional[Path] = None) -> Config:
    """Load configuration from file and environment variables.

    Args:
        config_path: Path to YAML configuration file. If None, uses default configuration.

    Returns:
        Config: Configured application settings.

    Raises:
        FileNotFoundError: If config_path is provided but file doesn't exist.
        yaml.YAMLError: If config file is invalid YAML.
    """
    config_dict: Dict[str, Any] = {}

    # Load from YAML file if provided
    if config_path is not None:
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

    # Load from default config location if it exists
    default_config_path = Path("config/default.yaml")
    if config_path is None and default_config_path.exists():
        with open(default_config_path, "r") as f:
            config_dict = yaml.safe_load(f) or {}

    # Create config object (will also read from environment variables)
    return Config(**config_dict)


def get_default_config() -> Config:
    """Get default configuration without loading from files.

    Returns:
        Config: Default configuration object.
    """
    return Config()


# Global config instance
_config: Optional[Config] = None


def get_config(config_path: Optional[Path] = None, force_reload: bool = False) -> Config:
    """Get the global configuration instance.

    Args:
        config_path: Path to configuration file (only used on first call or if force_reload=True).
        force_reload: Force reload configuration from file.

    Returns:
        Config: Configuration instance.
    """
    global _config

    if _config is None or force_reload:
        _config = load_config(config_path)

    return _config


def save_config(config: Config, output_path: Path) -> None:
    """Save configuration to YAML file.

    Args:
        config: Configuration object to save.
        output_path: Path where to save the configuration file.
    """
    # Create parent directories if they don't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert config to dict and save as YAML
    config_dict = config.model_dump()

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
