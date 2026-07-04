"""Typed access to ``config.yaml``.

Rather than sprinkle ``cfg["training"]["lr"]`` string lookups across the code
base, we load the YAML once into nested dataclasses. This gives editor
autocomplete, catches typos, and documents every knob in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, get_type_hints

import yaml

# Project root = parent of the directory containing this file (lpr/).
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")


@dataclass
class RecognizerConfig:
    img_height: int = 32
    img_width: int = 128
    channels: int = 1
    rnn_hidden: int = 256
    rnn_layers: int = 2
    cnn_dropout: float = 0.1
    weights_path: str = "models/recognizer/crnn.pt"
    metrics_path: str = "models/recognizer/metrics.json"


@dataclass
class DetectorConfig:
    backend: str = "classical"
    yolo_weights: str = "models/detector/plate_yolo.pt"
    conf_threshold: float = 0.25
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 6.5


@dataclass
class SyntheticConfig:
    out_dir: str = "data/synthetic"
    num_train: int = 8000
    num_val: int = 1500
    num_test: int = 1500
    seed: int = 1234


@dataclass
class HuggingFaceConfig:
    recognition_dataset: str = "keremberke/license-plate-object-detection"
    recognition_config: str = "full"
    detection_dataset: str = "keremberke/license-plate-object-detection"
    cache_dir: str = "data/hf_cache"


@dataclass
class TrainingConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 2
    grad_clip: float = 5.0
    device: str = "auto"
    augment: bool = True
    early_stop_patience: int = 6
    reports_dir: str = "reports"


@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    upload_dir: str = "app/static/uploads"
    max_upload_mb: int = 16


@dataclass
class Config:
    recognizer: RecognizerConfig = field(default_factory=RecognizerConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    app: AppConfig = field(default_factory=AppConfig)

    def abspath(self, path: str) -> str:
        """Resolve a config-relative path against the project root."""
        return path if os.path.isabs(path) else os.path.join(PROJECT_ROOT, path)


def _build(dc_type: type, data: Dict[str, Any]):
    """Instantiate a dataclass from a dict, recursing into nested dataclasses.

    Unknown keys are ignored so an older binary tolerates a newer config file.
    """
    if not is_dataclass(dc_type):
        return data
    kwargs: Dict[str, Any] = {}
    # ``from __future__ import annotations`` stringifies field types, so resolve
    # them back to real classes to detect nested dataclasses.
    hints = get_type_hints(dc_type)
    for f in fields(dc_type):
        name = f.name
        if name not in (data or {}):
            continue
        value = data[name]
        ftype = hints.get(name, f.type)
        if is_dataclass(ftype) and isinstance(value, dict):
            kwargs[name] = _build(ftype, value)
        else:
            kwargs[name] = value
    return dc_type(**kwargs)


def load_config(path: str | None = None) -> Config:
    """Load configuration from YAML, falling back to dataclass defaults."""
    path = path or DEFAULT_CONFIG_PATH
    if not os.path.exists(path):
        return Config()
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}
    return _build(Config, raw)
