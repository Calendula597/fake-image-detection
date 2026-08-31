from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from .paths import resolve_code_root


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(config_path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    cr = resolve_code_root()
    if config_path is None:
        config_path = str(cr / "configs" / "baseline.yaml")
    p = Path(config_path)
    if not p.is_absolute():
        p = (cr / p).resolve()
    with open(p, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if overrides:
        cfg = _deep_update(cfg, overrides)
    cfg = _expand_env(cfg)
    return cfg


def _expand_env(obj: Any) -> Any:
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(v) for v in obj]
    return obj


def parse_overrides(args: list) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for a in args:
        if "=" not in a:
            continue
        k, v = a.split("=", 1)
        try:
            v_parsed: Any = int(v)
        except ValueError:
            try:
                v_parsed = float(v)
            except ValueError:
                if v.lower() in ("true", "false"):
                    v_parsed = v.lower() == "true"
                else:
                    v_parsed = v
        d = out
        parts = k.split(".")
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        d[parts[-1]] = v_parsed
    return out


def cli_config_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--config", default=None, help="YAML config path (relative to CODE_ROOT or absolute)")
    parser.add_argument("--override", nargs="*", default=[], help="key=value overrides, e.g. training.lr=1e-4")
    return parser
