from __future__ import annotations

from dataclasses import dataclass, field
import logging
from pathlib import Path
import re

import numpy as np
import pandas as pd


@dataclass
class SignalTrace:
    file: str
    sheet: str
    kind: str
    conductance: np.ndarray
    pulse: np.ndarray | None = None
    voltage: np.ndarray | None = None
    current: np.ndarray | None = None
    resistance: np.ndarray | None = None
    reason: str = ""
    columns: dict[str, str] = field(default_factory=dict)


@dataclass
class RawAnalysisResult:
    traces: list[SignalTrace]
    logs: list[str]


class RawMemristorAnalyzer:
    """Infer LTP/LTD/IV traces from loosely formatted CSV/Excel memristor data."""

    KEYWORDS = {
        "voltage": ["volt", "voltage", "v", "bias"],
        "current": ["curr", "current", "i", "amp"],
        "resistance": ["res", "ohm", "r"],
        "conductance": ["cond", "conductance", "siemens", "g"],
        "pulse": ["pulse", "cycle", "step", "index", "number", "time"],
    }

    def __init__(self, path: Path):
        self.path = path
        self.logs: list[str] = []

    def analyze(self) -> RawAnalysisResult:
        files = self._files()
        if not files:
            msg = f"No raw memristor files found in {self.path}. Device model will use conservative defaults."
            logging.warning(msg)
            self.logs.append(msg)
            return RawAnalysisResult([], self.logs)

        traces: list[SignalTrace] = []
        for file in files:
            for sheet, df in self._read_tables(file):
                trace = self._analyze_table(file, sheet, df)
                if trace is not None:
                    traces.append(trace)
        if not traces:
            logging.warning("Raw files were found, but no usable LTP/LTD/IV trace was detected.")
        return RawAnalysisResult(traces, self.logs)

    def _files(self) -> list[Path]:
        if self.path.is_file():
            return [self.path]
        if not self.path.exists():
            return []
        patterns = ["*.xlsx", "*.xls", "*.csv", "*.txt"]
        files: list[Path] = []
        for pattern in patterns:
            files.extend(self.path.rglob(pattern))
        return sorted(files)

    def _read_tables(self, file: Path) -> list[tuple[str, pd.DataFrame]]:
        try:
            if file.suffix.lower() in {".xlsx", ".xls"}:
                sheets = pd.read_excel(file, sheet_name=None, header=None)
                return [(str(name), self._promote_header(df)) for name, df in sheets.items()]
            return [(file.stem, self._promote_header(pd.read_csv(file, header=None)))]
        except Exception as exc:
            logging.warning("Could not read %s: %s", file, exc)
            return []

    def _promote_header(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if df.empty:
            return df
        best_row = 0
        best_score = -1
        for idx in range(min(8, len(df))):
            row = df.iloc[idx].astype(str).str.lower().tolist()
            score = sum(any(k in cell for kws in self.KEYWORDS.values() for k in kws) for cell in row)
            score += sum(not _is_number(cell) for cell in row)
            if score > best_score:
                best_row, best_score = idx, score
        header = [str(x).strip() if str(x).strip() else f"col_{i}" for i, x in enumerate(df.iloc[best_row])]
        body = df.iloc[best_row + 1 :].copy()
        body.columns = _dedupe(header)
        return body.reset_index(drop=True)

    def _analyze_table(self, file: Path, sheet: str, df: pd.DataFrame) -> SignalTrace | None:
        if df.empty or df.shape[1] < 2:
            return None
        numeric = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
        if numeric.shape[1] < 2:
            return None
        cols = self._detect_columns(numeric)
        conductance = self._conductance(numeric, cols)
        if conductance is None or len(conductance) < 3:
            return None
        conductance = _clean_array(conductance)
        if len(conductance) < 3:
            return None
        pulse = _clean_array(numeric[cols["pulse"]].to_numpy()) if "pulse" in cols else np.arange(len(conductance))
        kind, reason = self._classify(conductance, numeric, cols)
        message = (
            f"Used file={file.name}, sheet={sheet}, columns={cols}; classified {kind}: {reason}"
        )
        logging.info(message)
        self.logs.append(message)
        return SignalTrace(
            file=str(file),
            sheet=sheet,
            kind=kind,
            conductance=conductance,
            pulse=pulse[: len(conductance)] if pulse is not None else None,
            voltage=_optional_array(numeric, cols, "voltage"),
            current=_optional_array(numeric, cols, "current"),
            resistance=_optional_array(numeric, cols, "resistance"),
            reason=reason,
            columns=cols,
        )

    def _detect_columns(self, df: pd.DataFrame) -> dict[str, str]:
        scores: dict[str, dict[str, float]] = {role: {} for role in self.KEYWORDS}
        for col in df.columns:
            name = str(col).lower()
            values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
            if len(values) == 0:
                continue
            for role, kws in self.KEYWORDS.items():
                scores[role][col] = 2.0 * sum(bool(re.search(rf"\b{re.escape(k)}\b", name)) or k in name for k in kws)
            if _mostly_monotonic(values) and len(np.unique(values)) > 3:
                scores["pulse"][col] += 1.5
            med = np.nanmedian(np.abs(values))
            if 1e-12 <= med <= 1e-1:
                scores["current"][col] += 0.5
                scores["conductance"][col] += 0.5
            if med > 1:
                scores["resistance"][col] += 0.6
            if np.nanmin(values) < 0 < np.nanmax(values):
                scores["voltage"][col] += 1.0
        chosen: dict[str, str] = {}
        used: set[str] = set()
        for role in ["conductance", "resistance", "current", "voltage", "pulse"]:
            ranked = sorted(scores[role].items(), key=lambda item: item[1], reverse=True)
            if ranked and ranked[0][1] > 0 and ranked[0][0] not in used:
                chosen[role] = ranked[0][0]
                used.add(ranked[0][0])
        if "resistance" in chosen:
            reciprocal = self._find_reciprocal_conductance(df, chosen["resistance"], used)
            if reciprocal is not None:
                previous = chosen.get("conductance")
                if previous is not None:
                    used.discard(previous)
                chosen["conductance"] = reciprocal
                used.add(reciprocal)
            elif _is_ambiguous_conductance_name(chosen.get("conductance")):
                previous = chosen.pop("conductance", None)
                if previous is not None:
                    used.discard(previous)
        if "pulse" not in chosen:
            for col in df.columns:
                values = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=float)
                if _mostly_monotonic(values):
                    chosen["pulse"] = col
                    break
        return chosen

    def _find_reciprocal_conductance(self, df: pd.DataFrame, resistance_col: str, used: set[str]) -> str | None:
        resistance = pd.to_numeric(df[resistance_col], errors="coerce").to_numpy(dtype=float)
        expected = np.where(resistance != 0, 1.0 / resistance, np.nan)
        best_col = None
        best_error = float("inf")
        for col in df.columns:
            if col == resistance_col:
                continue
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(values) & np.isfinite(expected) & (expected > 0)
            if mask.sum() < 3:
                continue
            scale = np.nanmedian(np.abs(expected[mask])) + 1e-30
            error = float(np.nanmedian(np.abs(values[mask] - expected[mask])) / scale)
            if error < best_error:
                best_col = col
                best_error = error
        if best_col is not None and best_error < 1e-3:
            return best_col
        return None

    def _conductance(self, df: pd.DataFrame, cols: dict[str, str]) -> np.ndarray | None:
        if "conductance" in cols:
            return pd.to_numeric(df[cols["conductance"]], errors="coerce").to_numpy(dtype=float)
        if "resistance" in cols:
            r = pd.to_numeric(df[cols["resistance"]], errors="coerce").to_numpy(dtype=float)
            return np.where(r != 0, 1.0 / r, np.nan)
        if "current" in cols and "voltage" in cols:
            i = pd.to_numeric(df[cols["current"]], errors="coerce").to_numpy(dtype=float)
            v = pd.to_numeric(df[cols["voltage"]], errors="coerce").to_numpy(dtype=float)
            return np.where(np.abs(v) > 1e-12, np.abs(i / v), np.nan)
        return None

    def _classify(self, conductance: np.ndarray, df: pd.DataFrame, cols: dict[str, str]) -> tuple[str, str]:
        smooth = _moving_average(conductance, min(7, max(3, len(conductance) // 10)))
        slope = float(np.nanmedian(np.diff(smooth))) if len(smooth) > 2 else 0.0
        total = float(smooth[-1] - smooth[0])
        noise = float(np.nanstd(np.diff(smooth))) + 1e-30
        if total > 2.0 * noise or slope > 0:
            return "LTP", f"conductance increases from {smooth[0]:.4g} to {smooth[-1]:.4g}"
        if total < -2.0 * noise or slope < 0:
            return "LTD", f"conductance decreases from {smooth[0]:.4g} to {smooth[-1]:.4g}"
        if "voltage" in cols and "current" in cols:
            v = pd.to_numeric(df[cols["voltage"]], errors="coerce").dropna().to_numpy(dtype=float)
            if len(v) > 4 and np.nanmin(v) < 0 < np.nanmax(v):
                return "IV", "voltage sweeps through both polarities with current present"
        return "UNKNOWN", "trend is smaller than estimated noise"


def _clean_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return values


def _optional_array(df: pd.DataFrame, cols: dict[str, str], key: str) -> np.ndarray | None:
    if key not in cols:
        return None
    return _clean_array(pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(dtype=float))


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def _mostly_monotonic(values: np.ndarray) -> bool:
    if len(values) < 3:
        return False
    diff = np.diff(values)
    return max(np.mean(diff >= 0), np.mean(diff <= 0)) > 0.85


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def _is_ambiguous_conductance_name(name: str | None) -> bool:
    if name is None:
        return False
    lowered = str(name).lower()
    explicit = ["cond", "conductance", "siemens"]
    if any(token in lowered for token in explicit):
        return False
    return True


def _dedupe(names: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    out = []
    for name in names:
        count = seen.get(name, 0)
        out.append(name if count == 0 else f"{name}_{count}")
        seen[name] = count + 1
    return out
