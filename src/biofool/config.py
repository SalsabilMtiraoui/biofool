"""YAML configuration with `a.b.c=value` command-line overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """A dict whose keys can also be read as attributes: cfg.generator.steps."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) and not isinstance(value, Config) else value


def _merge(base: dict, other: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in other.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


PROFILE_DIR = Path(__file__).resolve().parents[2] / "configs" / "profiles"


def load_config(path: str | Path, overrides: list[str] | None = None, profile: str | None = None) -> Config:
    """Load `path`. If it has a `base:` key, that file is loaded first and `path` is merged on top.

    Then the hardware profile (configs/profiles/<profile>.yaml, or the BIOFOOL_PROFILE environment variable, e.g.
    "t4" or "rtx2060") is merged, and finally the command-line overrides."""
    cfg = _load(Path(path))
    profile = profile or os.environ.get("BIOFOOL_PROFILE")
    if profile:
        cfg = _merge(cfg, _load(PROFILE_DIR / f"{profile}.yaml"))
        cfg["profile"] = profile
    for item in overrides or []:
        key, value = item.split("=", 1)
        node = cfg
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return Config(cfg)


def _load(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    if "base" in cfg:
        cfg = _merge(_load(path.parent / cfg.pop("base")), cfg)
    return cfg
