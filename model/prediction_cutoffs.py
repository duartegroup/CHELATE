"""Read the committed validation-selected cutoffs, or an explicit cutoff file."""

from __future__ import annotations

import json
import math
from pathlib import Path

METHODS = ("VINA", "SMINA", "GNINA")
DEFAULT_PATH = Path(__file__).resolve().parent / "params" / "pred_cutoffs.json"


def load_prediction_cutoffs(path: Path | None = None) -> dict[str, float]:
    record = json.loads((DEFAULT_PATH if path is None else Path(path)).read_text())
    result = {}
    for method in METHODS:
        value = record.get(method)
        if isinstance(value, bool) or not isinstance(value, (float, int)):
            raise ValueError(f"Missing or invalid prediction cutoff for {method}")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(
                f"Prediction cutoff for {method} must be finite and positive"
            )
        result[method] = float(value)
    return result


def cutoff_tag(cutoff: float) -> str:
    return f"{cutoff:g}".replace(".", "p")


def cutoff_map_tag(cutoffs: dict[str, float], methods: list[str]) -> str:
    values = {cutoffs[method] for method in methods}
    if len(values) == 1:
        return cutoff_tag(next(iter(values)))
    return "_".join(f"{method}-{cutoff_tag(cutoffs[method])}" for method in methods)
