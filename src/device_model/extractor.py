from __future__ import annotations

from dataclasses import dataclass, asdict, field
import logging

import numpy as np

from src.data_loader.raw_memristor import RawAnalysisResult, SignalTrace


@dataclass
class DeviceModel:
    g_on: float
    g_off: float
    on_off_ratio: float
    num_states: int
    ltp_nonlinearity: float
    ltd_nonlinearity: float
    cycle_variation_sigma: float
    state_means: list[float]
    state_stds: list[float]
    formulas: dict[str, str]
    source_traces: list[dict] = field(default_factory=list)
    # Phase 2: additional characterization fields
    ltp_symmetry: float = 0.0          # |NL_LTP - NL_LTD| / max(NL_LTP, NL_LTD, 1e-9)
    dynamic_range_db: float = 0.0      # 20*log10(g_on/g_off)
    endurance_stability: float = 0.0   # coefficient of variation across repeated traces
    fitted_ltp_coeffs: list[float] = field(default_factory=list)  # polynomial fit coefficients for LTP
    fitted_ltd_coeffs: list[float] = field(default_factory=list)  # polynomial fit coefficients for LTD

    def to_dict(self) -> dict:
        return asdict(self)


class DeviceModelExtractor:
    """Convert raw LTP/LTD/IV traces into simulator-ready memory-state statistics.

    Phase 2 improvements:
    - Robust state clustering with IQR-based outlier removal
    - Polynomial fitting for LTP/LTD curves (degree 3)
    - Symmetry metric for LTP vs LTD nonlinearity
    - Dynamic range in dB
    - Endurance stability from cycle-to-cycle variation
    """

    def extract(self, result: RawAnalysisResult) -> DeviceModel:
        traces = [t for t in result.traces if t.kind in {"LTP", "LTD", "IV", "UNKNOWN"} and len(t.conductance) > 0]
        if not traces:
            logging.warning("Using default two-state device because no valid conductance trace was available.")
            return DeviceModel(
                g_on=1e-5,
                g_off=1e-6,
                on_off_ratio=10.0,
                num_states=2,
                ltp_nonlinearity=0.0,
                ltd_nonlinearity=0.0,
                cycle_variation_sigma=0.0,
                state_means=[1e-6, 1e-5],
                state_stds=[0.0, 0.0],
                formulas=_formulas(),
                source_traces=[],
            )
        all_g = np.concatenate([t.conductance for t in traces])
        all_g = all_g[np.isfinite(all_g) & (all_g > 0)]
        # Phase 2: IQR-based outlier removal before statistics
        all_g = _remove_outliers_iqr(all_g)
        g_on = float(np.nanmax(all_g))
        g_off = float(np.nanmin(all_g))
        if g_on < g_off:
            logging.warning("Physical warning: g_on < g_off. Check units and column detection.")
        on_off = g_on / g_off if g_off > 0 else float("inf")
        states = self._stable_states(all_g)
        ltp_traces = [t for t in traces if t.kind == "LTP"]
        ltd_traces = [t for t in traces if t.kind == "LTD"]
        ltp = self._nonlinearity(ltp_traces, increasing=True)
        ltd = self._nonlinearity(ltd_traces, increasing=False)
        sigma = self._cycle_variation([t for t in traces if t.kind in {"LTP", "LTD"}])
        # Phase 2: polynomial fitting
        ltp_coeffs = self._fit_polynomial(ltp_traces, degree=3)
        ltd_coeffs = self._fit_polynomial(ltd_traces, degree=3)
        # Phase 2: symmetry metric
        nl_max = max(abs(ltp), abs(ltd), 1e-9)
        symmetry = abs(abs(ltp) - abs(ltd)) / nl_max
        # Phase 2: dynamic range in dB
        dynamic_range = 20.0 * np.log10(on_off) if on_off > 0 and np.isfinite(on_off) else 0.0
        # Phase 2: endurance stability
        endurance = self._endurance_stability([t for t in traces if t.kind in {"LTP", "LTD"}])

        logging.info("Device formulas: %s", _formulas())
        logging.info(
            "Extracted device model: g_on=%.6g S, g_off=%.6g S, on/off=%.4g (%.1f dB), "
            "states=%d, LTP_nl=%.4g, LTD_nl=%.4g, symmetry=%.4g, sigma=%.4g",
            g_on, g_off, on_off, dynamic_range,
            len(states[0]), ltp, ltd, symmetry, sigma,
        )
        return DeviceModel(
            g_on=g_on,
            g_off=g_off,
            on_off_ratio=on_off,
            num_states=len(states[0]),
            ltp_nonlinearity=ltp,
            ltd_nonlinearity=ltd,
            cycle_variation_sigma=sigma,
            state_means=states[0],
            state_stds=states[1],
            formulas=_formulas(),
            source_traces=_source_trace_rows(traces),
            ltp_symmetry=float(symmetry),
            dynamic_range_db=float(dynamic_range),
            endurance_stability=float(endurance),
            fitted_ltp_coeffs=[float(c) for c in ltp_coeffs],
            fitted_ltd_coeffs=[float(c) for c in ltd_coeffs],
        )

    def _stable_states(self, conductance: np.ndarray) -> tuple[list[float], list[float]]:
        """Improved state clustering: IQR-cleaned data + adaptive gap threshold."""
        values = np.sort(conductance[np.isfinite(conductance) & (conductance > 0)])
        if len(values) < 4:
            return values.tolist(), [0.0 for _ in values]
        diff = np.diff(values)
        # Use robust MAD (median absolute deviation) instead of simple median
        mad = np.nanmedian(np.abs(diff - np.nanmedian(diff))) + 1e-30
        # Adaptive threshold: 4*MAD or 1.5% of full range, whichever is larger
        threshold = max(4.0 * mad, 0.015 * (values[-1] - values[0]))
        groups: list[np.ndarray] = []
        start = 0
        for idx, delta in enumerate(diff, start=1):
            if delta > threshold:
                groups.append(values[start:idx])
                start = idx
        groups.append(values[start:])
        # Merge very small groups (< 2 points) into nearest neighbor
        groups = _merge_small_groups(groups, min_size=2)
        if len(groups) < 2:
            # Fallback: use uniform quantile binning
            n_bins = min(32, max(2, int(np.sqrt(len(values)))))
            quantiles = np.linspace(0, 1, n_bins)
            means = np.unique(np.quantile(values, quantiles))
            return means.tolist(), [float(np.std(values) * 0.05) for _ in means]
        means = [float(np.mean(g)) for g in groups if len(g)]
        stds = [float(np.std(g, ddof=1)) if len(g) > 1 else 0.0 for g in groups if len(g)]
        return means, stds

    def _nonlinearity(self, traces: list[SignalTrace], increasing: bool) -> float:
        """Nonlinearity: RMS deviation from ideal linear with outlier-robust normalization."""
        if not traces:
            return 0.0
        scores = []
        for trace in traces:
            g = np.asarray(trace.conductance, dtype=float)
            g = g[np.isfinite(g)]
            if len(g) < 4:
                continue
            g_range = np.nanmax(g) - np.nanmin(g)
            if g_range < 1e-30:
                continue
            y = (g - np.nanmin(g)) / g_range
            if not increasing:
                y = 1.0 - y
            x = np.linspace(0.0, 1.0, len(y))
            scores.append(float(np.sqrt(np.mean((y - x) ** 2))))
        return float(np.mean(scores)) if scores else 0.0

    def _cycle_variation(self, traces: list[SignalTrace]) -> float:
        if not traces:
            return 0.0
        grouped: dict[str, list[SignalTrace]] = {}
        for trace in traces:
            grouped.setdefault(trace.kind, []).append(trace)
        scores = []
        for same_kind in grouped.values():
            if len(same_kind) > 1:
                length = min(len(t.conductance) for t in same_kind)
                stacked = np.vstack([t.conductance[:length] for t in same_kind])
                denom = np.nanmean(np.abs(stacked)) + 1e-30
                scores.append(float(np.nanmean(np.nanstd(stacked, axis=0)) / denom))
            else:
                diffs = np.diff(same_kind[0].conductance)
                denom = np.nanmean(np.abs(same_kind[0].conductance)) + 1e-30
                scores.append(float(np.nanstd(diffs) / denom))
        return float(np.nanmean(scores)) if scores else 0.0

    def _fit_polynomial(self, traces: list[SignalTrace], degree: int = 3) -> list[float]:
        """Fit polynomial to averaged conductance trace for smooth curve generation."""
        if not traces:
            return []
        # Average all traces of the same kind
        min_len = min(len(t.conductance) for t in traces)
        if min_len < degree + 1:
            return []
        stacked = np.vstack([np.asarray(t.conductance[:min_len], dtype=float) for t in traces])
        avg = np.nanmean(stacked, axis=0)
        x = np.linspace(0.0, 1.0, len(avg))
        try:
            coeffs = np.polyfit(x, avg, degree)
            return coeffs.tolist()
        except (np.linalg.LinAlgError, ValueError):
            return []

    def _endurance_stability(self, traces: list[SignalTrace]) -> float:
        """Coefficient of variation of final conductance across repeated traces.
        Lower = more stable device."""
        if len(traces) < 2:
            return 0.0
        finals = []
        for t in traces:
            g = np.asarray(t.conductance, dtype=float)
            g = g[np.isfinite(g)]
            if len(g) > 0:
                finals.append(float(g[-1]))
        if len(finals) < 2:
            return 0.0
        mean_val = np.mean(finals)
        if abs(mean_val) < 1e-30:
            return 0.0
        return float(np.std(finals, ddof=1) / abs(mean_val))


def _remove_outliers_iqr(values: np.ndarray, factor: float = 2.0) -> np.ndarray:
    """Remove outliers using IQR method. Factor=2.0 is less aggressive than 1.5."""
    if len(values) < 4:
        return values
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr < 1e-30:
        return values
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    mask = (values >= lower) & (values <= upper)
    cleaned = values[mask]
    if len(cleaned) < 4:
        return values  # Don't over-remove
    return cleaned


def _merge_small_groups(groups: list[np.ndarray], min_size: int = 2) -> list[np.ndarray]:
    """Merge groups with fewer than min_size points into their nearest neighbor."""
    if len(groups) <= 1:
        return groups
    merged = []
    for g in groups:
        if len(g) < min_size and merged:
            merged[-1] = np.concatenate([merged[-1], g])
        else:
            merged.append(g)
    # Also merge any trailing small group
    if len(merged) > 1 and len(merged[-1]) < min_size:
        merged[-2] = np.concatenate([merged[-2], merged[-1]])
        merged.pop()
    return merged


def _formulas() -> dict[str, str]:
    return {
        "g_on": "max(conductance) after IQR outlier removal",
        "g_off": "min(conductance) after IQR outlier removal",
        "on_off_ratio": "g_on / g_off",
        "dynamic_range_db": "20 * log10(on_off_ratio)",
        "num_states": "count of conductance clusters (MAD-adaptive gap threshold, small-group merging)",
        "ltp_ltd_nonlinearity": "RMS deviation of normalized conductance trajectory from ideal linear pulse response",
        "symmetry": "|NL_LTP - NL_LTD| / max(NL_LTP, NL_LTD) — 0=symmetric, 1=highly asymmetric",
        "cycle_variation_sigma": "mean standard deviation across repeated cycles normalized by mean conductance",
        "endurance_stability": "coefficient of variation of final conductance across repeated traces",
        "fitted_coefficients": "degree-3 polynomial fit to averaged LTP/LTD traces",
    }


def _source_trace_rows(traces: list[SignalTrace]) -> list[dict]:
    rows = []
    for trace in traces:
        rows.append(
            {
                "file": trace.file,
                "sheet": trace.sheet,
                "kind": trace.kind,
                "reason": trace.reason,
                "pulse": _array_to_list(trace.pulse),
                "conductance": _array_to_list(trace.conductance),
                "voltage": _array_to_list(trace.voltage),
                "current": _array_to_list(trace.current),
                "resistance": _array_to_list(trace.resistance),
            }
        )
    return rows


def _array_to_list(values) -> list[float]:
    if values is None:
        return []
    return [float(v) for v in np.asarray(values, dtype=float) if np.isfinite(v)]
