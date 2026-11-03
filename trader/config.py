import copy
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | os.PathLike | None = None) -> dict:
    path = Path(path or os.environ.get("TRADER_CONFIG", ROOT / "config.yaml"))
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    # rutas relativas -> relativas a la raíz del proyecto
    for section, key in (("storage", "db_path"), ("model", "model_dir")):
        p = Path(cfg[section][key])
        if not p.is_absolute():
            cfg[section][key] = str(ROOT / p)
    return cfg


def override(cfg: dict, **changes) -> dict:
    """Copia de cfg con cambios del tipo override(cfg, data__source='synthetic')."""
    out = copy.deepcopy(cfg)
    for k, v in changes.items():
        section, key = k.split("__", 1)
        out[section][key] = v
    return out
