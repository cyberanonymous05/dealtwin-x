"""Load the project's config/config.yaml once and cache it."""

import os

import yaml

_CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "config", "config.yaml")
_cached_config = None


def load_config() -> dict:
    global _cached_config
    if _cached_config is None:
        with open(_CONFIG_PATH) as f:
            _cached_config = yaml.safe_load(f)
    return _cached_config
