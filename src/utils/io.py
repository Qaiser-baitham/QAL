from __future__ import annotations

import json
from pathlib import Path
import tempfile

import pandas as pd


def atomic_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=path.parent, encoding="utf-8", suffix=".tmp") as tmp:
        json.dump(data, tmp, indent=2)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)


def export_excel(path: Path, data: dict[str, list | tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = {
        key: pd.Series(value if isinstance(value, (list, tuple)) else [value])
        for key, value in data.items()
    }
    pd.DataFrame(columns).to_excel(path, index=False)
