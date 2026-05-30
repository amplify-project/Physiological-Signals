"""
Configuration management utilities for the concert engagement project.

This module provides functions to load, validate, and manage configuration
files using OmegaConf for structured configuration management.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from omegaconf import OmegaConf, DictConfig
import yaml

logger = logging.getLogger(__name__)


class ConfigManager:
    """
    Configuration manager for handling project configurations.
    
    Supports loading default configurations, overrides, and environment-specific
    configurations using OmegaConf.
    """
    
    def __init__(self, config_dir: str = "configs"):
        """
        Initialize the configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir)
        self.default_config = None
        self.current_config = None
    
    def load_default_config(self) -> DictConfig:
        """
        Load the default configuration file.
        
        Returns:
            Default configuration as DictConfig
        """
        default_path = self.config_dir / "default.yaml"
        if not default_path.exists():
            raise FileNotFoundError(f"Default config not found: {default_path}")
        
        self.default_config = OmegaConf.load(default_path)
        logger.info(f"Loaded default configuration from {default_path}")
        return self.default_config
    
    def load_config(
        self,
        config_name: Optional[str] = None,
        overrides: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> DictConfig:
        """
        Load configuration with optional overrides.
        
        Args:
            config_name: Name of specific config file (without .yaml)
            overrides: Dictionary of configuration overrides
            **kwargs: Additional overrides as keyword arguments
        
        Returns:
            Merged configuration as DictConfig
        """
        # Start with default config
        if self.default_config is None:
            self.load_default_config()
        
        config = self.default_config.copy()
        
        # Load specific config if provided
        if config_name:
            specific_path = self.config_dir / f"{config_name}.yaml"
            if specific_path.exists():
                specific_config = OmegaConf.load(specific_path)
                config = OmegaConf.merge(config, specific_config)
                logger.info(f"Merged configuration from {specific_path}")
            else:
                logger.warning(f"Specific config not found: {specific_path}")
        
        # Apply overrides from dictionary
        if overrides:
            override_config = OmegaConf.create(overrides)
            config = OmegaConf.merge(config, override_config)
            logger.info("Applied configuration overrides from dictionary")
        
        # Apply overrides from kwargs
        if kwargs:
            kwarg_config = OmegaConf.create(kwargs)
            config = OmegaConf.merge(config, kwarg_config)
            logger.info("Applied configuration overrides from kwargs")
        
        # Resolve any interpolations
        config = OmegaConf.create(OmegaConf.to_yaml(config))
        
        # Validate configuration
        self.validate_config(config)
        
        self.current_config = config
        return config
    
    def validate_config(self, config: DictConfig) -> bool:
        """
        Validate the configuration for required fields and consistency.
        
        Args:
            config: Configuration to validate
        
        Returns:
            True if configuration is valid
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check required top-level sections
        required_sections = ["dataset", "dwpose", "features", "models", "paths"]
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Required configuration section missing: {section}")
        
        # Validate dataset configuration
        if "data_root" not in config.dataset:
            raise ValueError("dataset.data_root is required")
        
        # Validate DWPose configuration
        if "device" not in config.dwpose:
            config.dwpose.device = "cuda:0" if self._check_cuda_available() else "cpu"
        
        # Validate paths exist or can be created
        self._validate_paths(config)
        
        # Validate model configuration
        self._validate_model_config(config)
        
        logger.info("Configuration validation passed")
        return True
    
    def _check_cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def _validate_paths(self, config: DictConfig):
        """Validate and create necessary paths."""
        paths_to_create = [
            config.paths.processed_data,
            config.paths.features,
            config.paths.poses,
            config.paths.models,
            config.paths.results,
            config.paths.logs,
            config.paths.plots
        ]
        
        for path in paths_to_create:
            Path(path).mkdir(parents=True, exist_ok=True)
        
        # Check if raw data path exists
        if not Path(config.dataset.data_root).exists():
            logger.warning(f"Raw data path does not exist: {config.dataset.data_root}")
    
    def _validate_model_config(self, config: DictConfig):
        """Validate model configuration."""
        model_type = config.models.engagement_predictor.type
        supported_models = ["random_forest", "xgboost", "neural_network"]
        
        if model_type not in supported_models:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Supported types: {supported_models}")
        
        # Check if model-specific config exists
        if model_type not in config.models.engagement_predictor:
            logger.warning(f"No specific configuration found for model type: {model_type}")
    
    def save_config(self, config: DictConfig, save_path: str):
        """
        Save configuration to a YAML file.
        
        Args:
            config: Configuration to save
            save_path: Path to save the configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            OmegaConf.save(config, f)
        
        logger.info(f"Configuration saved to: {save_path}")
    
    def get_config_summary(self, config: Optional[DictConfig] = None) -> str:
        """
        Get a summary of the current configuration.
        
        Args:
            config: Configuration to summarize (uses current if None)
        
        Returns:
            Configuration summary as string
        """
        if config is None:
            config = self.current_config
        
        if config is None:
            return "No configuration loaded"
        
        summary = []
        summary.append("=== Configuration Summary ===")
        summary.append(f"Dataset: {config.dataset.name}")
        summary.append(f"Data Root: {config.dataset.data_root}")
        summary.append(f"DWPose Device: {config.dwpose.device}")
        summary.append(f"Model Type: {config.models.engagement_predictor.type}")
        summary.append(f"Feature Window: {config.features.temporal.window_size} frames")
        summary.append(f"Tracking Backend: {config.tracking.backend}")
        summary.append("=" * 30)
        
        return "\n".join(summary)
    
    def update_config(self, updates: Dict[str, Any]) -> DictConfig:
        """
        Update current configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        
        Returns:
            Updated configuration
        """
        if self.current_config is None:
            raise ValueError("No configuration loaded to update")
        
        update_config = OmegaConf.create(updates)
        self.current_config = OmegaConf.merge(self.current_config, update_config)
        
        logger.info("Configuration updated")
        return self.current_config


# Global configuration manager instance
config_manager = ConfigManager()


def load_config(
    config_name: Optional[str] = None,
    config_dir: str = "configs",
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> DictConfig:
    """
    Convenience function to load configuration.
    
    Args:
        config_name: Name of specific config file
        config_dir: Directory containing config files
        overrides: Configuration overrides
        **kwargs: Additional overrides
    
    Returns:
        Loaded configuration
    """
    global config_manager
    config_manager.config_dir = Path(config_dir)
    return config_manager.load_config(config_name, overrides, **kwargs)


def get_config() -> Optional[DictConfig]:
    """
    Get the current configuration.
    
    Returns:
        Current configuration or None if not loaded
    """
    return config_manager.current_config


def validate_config_file(config_path: str) -> bool:
    """
    Validate a configuration file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        True if valid
    """
    try:
        config = OmegaConf.load(config_path)
        config_manager.validate_config(config)
        return True
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def create_experiment_config(
    experiment_name: str,
    base_config: str = "default",
    overrides: Optional[Dict[str, Any]] = None
) -> DictConfig:
    """
    Create a configuration for a specific experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_config: Base configuration to extend
        overrides: Experiment-specific overrides
    
    Returns:
        Experiment configuration
    """
    config = load_config(base_config, overrides=overrides)
    
    # Update experiment-specific settings
    experiment_overrides = {
        "tracking": {
            "mlflow": {
                "experiment_name": f"concert_engagement_{experiment_name}",
                "run_name": experiment_name
            }
        },
        "paths": {
            "results": f"results/{experiment_name}",
            "logs": f"logs/{experiment_name}",
            "plots": f"plots/{experiment_name}"
        }
    }
    
    experiment_config = OmegaConf.create(experiment_overrides)
    config = OmegaConf.merge(config, experiment_config)
    
    # Save experiment config
    experiment_config_path = Path("configs") / f"experiment_{experiment_name}.yaml"
    config_manager.save_config(config, experiment_config_path)
    
    return config


if __name__ == "__main__":
    # Example usage
    
    # Load default configuration
    config = load_config()
    print(config_manager.get_config_summary())
    
    # Load with overrides
    config = load_config(
        overrides={
            "models.engagement_predictor.type": "xgboost",
            "features.temporal.window_size": 120
        }
    )
    
    # Create experiment configuration
    experiment_config = create_experiment_config(
        "pilot_study",
        overrides={
            "dataset.video.clip_duration": 3.0,
            "models.engagement_predictor.type": "neural_network"
        }
    )
    
    print("Experiment configuration created successfully")