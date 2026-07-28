from __future__ import annotations

import numpy as np

from ._lib import addr, f64, lib
from .core import frames_to_samples, frames_to_time
from .feature.rhythm import tempo as estimate_tempo
from .onset import onset_strength


def _track_one(
    onset: np.ndarray,
    bpm: float,
    frame_rate: float,
    tightness: float,
    trim: bool,
) -> np.ndarray:
    if bpm <= 0 or tightness <= 0:
        raise ValueError("bpm and tightness must be strictly positive")
    native = f64(onset)
    if native.ndim != 1 or native.size < 2:
        raise ValueError("onset envelope must contain at least two frames")
    if not np.all(np.isfinite(native)):
        raise ValueError("onset envelope is not finite everywhere")
    std = native.std(ddof=1)
    normalized = np.ascontiguousarray(native / (std + np.finfo(native.dtype).tiny))
    n = normalized.size
    period = int(np.round(frame_rate * 60.0 / bpm))
    if period < 1:
        raise ValueError("bpm is too large for the selected sample and hop rates")
    local = np.empty(n, dtype=np.float64)
    cumulative = np.empty(n, dtype=np.float64)
    backlink = np.empty(n, dtype=np.int64)
    lib().mls_beat_dp(
        addr(normalized),
        addr(local),
        addr(cumulative),
        addr(backlink),
        n,
        period,
        float(tightness),
    )
    maxima = np.zeros(n, dtype=bool)
    if n > 1:
        maxima[1:-1] = (cumulative[1:-1] > cumulative[:-2]) & (
            cumulative[1:-1] >= cumulative[2:]
        )
        maxima[-1] = cumulative[-1] > cumulative[-2]
    threshold = 0.5 * np.median(cumulative[maxima]) if maxima.any() else -np.inf
    candidates = np.flatnonzero(maxima & (cumulative >= threshold))
    tail = int(candidates[-1]) if candidates.size else n - 1
    beats = np.zeros(n, dtype=bool)
    while tail >= 0:
        beats[tail] = True
        tail = int(backlink[tail])
    beat_indices = np.flatnonzero(beats)
    if beat_indices.size:
        smooth = np.convolve(local[beats], np.hanning(5))[2 : n + 2]
        cutoff = 0.5 * np.sqrt(np.mean(smooth**2)) if trim else 0.0
        first = 0
        while first < n and local[first] <= cutoff:
            beats[first] = False
            first += 1
        last = n - 1
        while last >= 0 and local[last] <= cutoff:
            beats[last] = False
            last -= 1
    return beats


def beat_track(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    onset_envelope: np.ndarray | None = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    tightness: float = 100,
    trim: bool = True,
    bpm: float | np.ndarray | None = None,
    prior: object | None = None,
    units: str = "frames",
    sparse: bool = True,
) -> tuple[float | np.ndarray, np.ndarray]:
    if prior is not None:
        raise NotImplementedError("probabilistic tempo priors are not implemented")
    if sr <= 0 or hop_length <= 0 or start_bpm <= 0 or tightness <= 0:
        raise ValueError(
            "sr, hop_length, start_bpm, and tightness must be strictly positive"
        )
    if onset_envelope is None:
        if y is None:
            raise ValueError("y or onset_envelope must be provided")
        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset = np.asarray(onset_envelope)
    if onset.ndim == 0 or onset.shape[-1] == 0:
        raise ValueError("onset envelope must contain at least one frame")
    if not np.all(np.isfinite(onset)):
        raise ValueError("onset envelope is not finite everywhere")
    if sparse and onset.ndim != 1:
        raise ValueError("sparse=True does not support multi-dimensional inputs")
    if not onset.any():
        if sparse:
            return 0.0, np.array([], dtype=int)
        return np.zeros(onset.shape[:-1]), np.zeros_like(onset, dtype=bool)
    if onset.shape[-1] < 2:
        raise ValueError("non-empty onset envelope must contain at least two frames")
    if bpm is None:
        bpm_result: float | np.ndarray = estimate_tempo(
            onset_envelope=onset,
            sr=sr,
            hop_length=hop_length,
            start_bpm=start_bpm,
        )
    else:
        bpm_result = bpm
    frame_rate = float(sr) / hop_length
    bpm_array = np.asarray(bpm_result)
    if bpm_array.size == 0 or not np.all(np.isfinite(bpm_array)) or np.any(bpm_array <= 0):
        raise ValueError("bpm must contain finite, strictly positive values")
    if onset.ndim == 1:
        dense = _track_one(onset, float(bpm_array.reshape(-1)[0]), frame_rate, tightness, trim)
    else:
        rows = onset.reshape(-1, onset.shape[-1])
        bpms = np.broadcast_to(bpm_array, onset.shape[:-1]).reshape(-1)
        dense = np.stack(
            [
                _track_one(row, float(row_bpm), frame_rate, tightness, trim)
                for row, row_bpm in zip(rows, bpms)
            ]
        ).reshape(onset.shape)
    if not sparse:
        return bpm_result, dense
    beats = np.flatnonzero(dense)
    if units == "samples":
        beats = frames_to_samples(beats, hop_length=hop_length)
    elif units == "time":
        beats = frames_to_time(beats, sr=sr, hop_length=hop_length)
    elif units != "frames":
        raise ValueError(f"Invalid unit type: {units}")
    return bpm_result, beats
