"""
Configuration module.

This module contains configuration classes and settings for the application.
It includes configurations for MongoDB, stock API, paths, proxies etc.
"""

import os

from fmp.config import Config as FMPConfig


class PathConfig:
    """
    Path configuration class.

    """

    root_directory_path: str = os.getcwd()
    data_directory_path: str = os.path.join(root_directory_path, "data")
    short_term_model_path: str = os.path.join(
        data_directory_path, "short_term_model.h5"
    )
    daily_model_path: str = os.path.join(data_directory_path, "daily_model.h5")


class Config(FMPConfig):
    """
    Application configuration class.

    """

    project_path: PathConfig = PathConfig()


cfg = Config()  # noqa
