from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

class Cfg(dict):
    """dict with attribute access recursively"""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Cfg(v) if isinstance(v, dict) else v

    def to_dict(self) -> dict:
        return {k: (v.to_dict() if isinstance(v, Cfg) else v) for k, v in self.items()}

def load_config(path: str | Path) -> Cfg:
    with open(path) as f:
        return Cfg(yaml.safe_load(f))