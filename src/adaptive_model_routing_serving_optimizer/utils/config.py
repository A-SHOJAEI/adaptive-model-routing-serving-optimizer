"""Configuration management utilities."""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Union

import yaml


class Config:
    """Configuration class for managing YAML-based configuration files."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        """Initialize configuration with a dictionary.

        Args:
            config_dict: Dictionary containing configuration parameters.
        """
        self._config = config_dict
        self._setup_logging()

    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config[key]

    def __contains__(self, key: str) -> bool:
        """Check if key exists in configuration."""
        return key in self._config

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default fallback.

        Args:
            key: Configuration key to retrieve.
            default: Default value if key doesn't exist.

        Returns:
            Configuration value or default.
        """
        return self._config.get(key, default)

    def get_nested(self, *keys: str, default: Any = None) -> Any:
        """Get nested configuration value.

        Args:
            *keys: Nested keys to traverse.
            default: Default value if path doesn't exist.

        Returns:
            Configuration value or default.
        """
        current = self._config
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        return current

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values.

        Args:
            updates: Dictionary of updates to apply.
        """
        self._config.update(updates)

    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_level = self.get_nested("monitoring", "log_level", default="INFO")
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )


def load_config(config_path: Union[str, Path] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file. If None, looks for default config.

    Returns:
        Config object with loaded configuration.

    Raises:
        FileNotFoundError: If configuration file doesn't exist.
        yaml.YAMLError: If configuration file is invalid YAML.
    """
    if config_path is None:
        # Look for default config in current directory or project root
        possible_paths = [
            Path("configs/default.yaml"),
            Path("../configs/default.yaml"),
            Path("../../configs/default.yaml"),
        ]

        for path in possible_paths:
            if path.exists():
                config_path = path
                break
        else:
            raise FileNotFoundError(
                "No configuration file found. Please provide config_path or "
                "ensure configs/default.yaml exists."
            )

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Invalid YAML in configuration file: {e}")

    # Set environment variables for reproducibility
    if "seed" in config_dict:
        os.environ["PYTHONHASHSEED"] = str(config_dict["seed"])

    return Config(config_dict)