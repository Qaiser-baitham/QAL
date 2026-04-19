"""
Intelligent memristor data loader.
------------------------------------
Reads .xlsx / .csv files from data/raw_memristor/ without depending on
specific header names. Auto-detects columns for voltage, current,
resistance/conductance, and pulse cycles using fuzzy keyword matching
and numeric-range heuristics.

Separates LTP (increasing conductance) from LTD (decreasing conductance)
automatically within the same sheet.
"""
from __future__ import annotations
import os, re, glob
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# ------- heuristic keyword dictionaries -------
KW = {
    "voltage":     [r"\bv\b", "volt", "vread", "vapp", "vg", "vbias"],
    "current":     [r"\bi\b", "curr", "iread", "ids", "amp"],
    "resistance":  ["res", "ohm", "r_"],
    "conductance": ["cond", "sieme", r"\bg\b", "gx"],
    "pulse":       ["pulse", "cycle", "step", "index", "#"],
    "time":        ["time", "sec", r"t\("],
}


def _match(col: str, keys) -> bool:
    s = col.lower()
    for k in keys:
        try:
            if re.search(k, s):
                return True
        except re.error:
            if k.lower() in s:
                return True
    return False


def _infer_type_numeric(series: pd.Series) -> str:
    """Fallback - guess by magnitude range."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty: return "unknown"
    mx = np.abs(s).max()
    if mx < 5: return "voltage"
    if mx < 1e-3: return "current"
    if mx > 1e3: return "resistance"
    return "unknown"


@dataclass
class MemristorTrace:
    name: str
    voltage: Optional[np.ndarray] = None
    current: Optional[np.ndarray] = None
    resistance: Optional[np.ndarray] = None
    conductance: Optional[np.ndarray] = None
    pulse:  Optional[np.ndarray] = None
    ltp: Optional[np.ndarray] = None
    ltd: Optional[np.ndarray] = None
    meta: Dict = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        cols = {k: getattr(self, k) for k in
                ("pulse", "voltage", "current", "resistance", "conductance")
                if getattr(self, k) is not None}
        if not cols: return pd.DataFrame()
        n = max(len(v) for v in cols.values())
        cols = {k: (v if len(v) == n else np.pad(v, (0, n-len(v)), constant_values=np.nan))
                for k, v in cols.items()}
        return pd.DataFrame(cols)


class MemristorLoader:
    """Auto-parse every file under a directory of memristor measurements."""
    def __init__(self, root: str):
        self.root = root

    def discover(self) -> List[str]:
        exts = ("*.xlsx", "*.xls", "*.csv")
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(self.root, "**", e), recursive=True))
        return sorted(files)

    def _read(self, path: str) -> Dict[str, pd.DataFrame]:
        if path.lower().endswith(".csv"):
            return {"sheet": pd.read_csv(path)}
        return pd.read_excel(path, sheet_name=None)

    def _map_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping = {}
        for col in df.columns:
            c = str(col)
            for k, keys in KW.items():
                if k not in mapping.values() and _match(c, keys):
                    mapping[col] = k
                    break
        unmapped = [c for c in df.columns if c not in mapping]
        for c in unmapped:
            t = _infer_type_numeric(df[c])
            if t != "unknown" and t not in mapping.values():
                mapping[c] = t
        return mapping

    @staticmethod
    def _split_ltp_ltd(g: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split conductance sequence into monotonic up (LTP) / down (LTD) segments."""
        if g.size < 4: return g, np.array([])
        diffs = np.diff(g)
        peak = int(np.argmax(g))
        ltp = g[:peak + 1] if peak > 0 else np.array([])
        ltd = g[peak:]     if peak < len(g) - 1 else np.array([])
        if ltp.size < 2 or ltd.size < 2:
            ltp = g[np.r_[True, diffs > 0]]
            ltd = g[np.r_[True, diffs < 0]]
        return ltp, ltd

    def parse_sheet(self, name: str, df: pd.DataFrame) -> MemristorTrace:
        df = df.dropna(how="all").reset_index(drop=True)
        if df.empty:
            return MemristorTrace(name=name)
        cmap = self._map_columns(df)
        inv = {v: k for k, v in cmap.items()}
        trace = MemristorTrace(name=name, meta={"column_map": cmap})

        def _get(kind):
            if kind in inv:
                return pd.to_numeric(df[inv[kind]], errors="coerce").values
            return None

        trace.voltage = _get("voltage")
        trace.current = _get("current")
        trace.resistance = _get("resistance")
        trace.conductance = _get("conductance")
        trace.pulse = _get("pulse")

        if trace.conductance is None and trace.resistance is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                trace.conductance = 1.0 / trace.resistance
        if trace.conductance is None and trace.voltage is not None and trace.current is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                v = np.where(np.abs(trace.voltage) > 1e-9, trace.voltage, np.nan)
                trace.conductance = trace.current / v
        if trace.pulse is None and trace.conductance is not None:
            trace.pulse = np.arange(len(trace.conductance))

        if trace.conductance is not None:
            g = trace.conductance
            g = g[np.isfinite(g)]
            if g.size:
                trace.ltp, trace.ltd = self._split_ltp_ltd(g)
                trace.meta["g_on"] = float(np.nanmax(g))
                trace.meta["g_off"] = float(np.nanmin(g[g > 0])) if np.any(g > 0) else float("nan")
                trace.meta["on_off_ratio"] = (trace.meta["g_on"] / trace.meta["g_off"]
                                              if trace.meta["g_off"] > 0 else float("inf"))
                trace.meta["num_states_est"] = int(min(len(np.unique(np.round(g, 8))), 256))
                trace.meta["std"] = float(np.nanstd(g))
        return trace

    def load_all(self) -> List[MemristorTrace]:
        traces = []
        for f in self.discover():
            try:
                sheets = self._read(f)
            except Exception as e:
                print(f"[WARN] {f}: {e}"); continue
            for sn, df in sheets.items():
                t = self.parse_sheet(f"{os.path.basename(f)}::{sn}", df)
                if t.conductance is not None and t.conductance.size > 3:
                    traces.append(t)
        return traces


def summarize(traces: List[MemristorTrace]) -> pd.DataFrame:
    rows = []
    for t in traces:
        rows.append({
            "trace": t.name,
            "n_points": len(t.conductance) if t.conductance is not None else 0,
            "g_on": t.meta.get("g_on", np.nan),
            "g_off": t.meta.get("g_off", np.nan),
            "on_off_ratio": t.meta.get("on_off_ratio", np.nan),
            "num_states_est": t.meta.get("num_states_est", 0),
            "std": t.meta.get("std", np.nan),
            "has_LTP": t.ltp is not None and t.ltp.size > 0,
            "has_LTD": t.ltd is not None and t.ltd.size > 0,
        })
    return pd.DataFrame(rows)
