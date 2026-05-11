from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


DEFAULT_ENDPOINT_CONFIG_PATH = Path("configs/endpoints.json")

DEFAULT_ENDPOINT_SETTINGS: Dict[str, Dict[str, Any]] = {
    "generator": {
        "api_endpoint": "responses",
        "model": "gpt-5.4",
        "max_output_tokens": 1500,
        "temperature": 0.0,
        "reasoning_effort": None,
    },
    "judge": {
        "api_endpoint": "responses",
        "model": "gpt-5.4",
        "max_output_tokens": 16000,
        "temperature": 0.0,
        "reasoning_effort": "high",
    },
}


def load_endpoint_config(path: Path = DEFAULT_ENDPOINT_CONFIG_PATH) -> Dict[str, Dict[str, Any]]:
    if not path.exists():
        return DEFAULT_ENDPOINT_SETTINGS

    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Expected endpoint config object in {path}")

    config: Dict[str, Dict[str, Any]] = {}
    for section, defaults in DEFAULT_ENDPOINT_SETTINGS.items():
        values = raw.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Expected endpoint config section '{section}' to be an object in {path}")
        config[section] = {**defaults, **values}

    return config


def get_endpoint_settings(
    config: Dict[str, Dict[str, Any]],
    section: str,
) -> Dict[str, Any]:
    if section not in DEFAULT_ENDPOINT_SETTINGS:
        raise ValueError(f"Unknown endpoint config section: {section}")

    settings = {**DEFAULT_ENDPOINT_SETTINGS[section], **config.get(section, {})}
    if settings["api_endpoint"] != "responses":
        raise ValueError(
            f"Unsupported api_endpoint for {section}: {settings['api_endpoint']}. "
            "Only 'responses' is currently implemented."
        )
    if not isinstance(settings["model"], str) or not settings["model"].strip():
        raise ValueError(f"Endpoint config section '{section}' must define a non-empty model.")
    if int(settings["max_output_tokens"]) <= 0:
        raise ValueError(f"Endpoint config section '{section}' must define positive max_output_tokens.")

    return settings
