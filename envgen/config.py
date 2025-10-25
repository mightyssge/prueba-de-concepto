import json
from pathlib import Path
from typing import Any, Dict

def load_config(path: str | Path) -> Dict[str, Any]:
    """Carga tu config.json tal cual, sin validación"""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
