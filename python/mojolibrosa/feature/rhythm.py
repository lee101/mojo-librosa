from __future__ import annotations

import numpy as np

from .._lib import addr, f64, lib


def tempo(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    onset_envelope: np.ndarray | None = None,
    hop_length: int = 512,
    start_bpm: float = 120.0,
    max_tempo: float | None = 320.0,
    **kwargs: object,
) -> np.ndarray:
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise NotImplementedError(f"tempo options are not implemented: {names}")
    if onset_envelope is None:
        if y is None:
            raise ValueError("y or onset_envelope must be provided")
        from ..onset import onset_strength

        onset_envelope = onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset = np.asarray(onset_envelope)
    if onset.ndim == 0 or onset.shape[-1] < 2:
        raise ValueError("onset envelope must contain at least two frames")
    if not np.all(np.isfinite(onset)):
        raise ValueError("onset envelope is not finite everywhere")
    if sr <= 0 or hop_length <= 0 or start_bpm <= 0:
        raise ValueError("sr, hop_length, and start_bpm must be strictly positive")
    if onset.ndim != 1:
        return np.asarray(
            [
                tempo(
                    onset_envelope=row,
                    sr=sr,
                    hop_length=hop_length,
                    start_bpm=start_bpm,
                    max_tempo=max_tempo,
                ).item()
                for row in onset.reshape(-1, onset.shape[-1])
            ]
        ).reshape(onset.shape[:-1])
    frame_rate = float(sr) / hop_length
    min_bpm = 30.0
    upper_bpm = 320.0 if max_tempo is None else float(max_tempo)
    if upper_bpm <= 0:
        raise ValueError("max_tempo must be strictly positive")
    min_period = max(1, int(np.ceil(frame_rate * 60.0 / upper_bpm)))
    max_period = max(min_period, int(np.floor(frame_rate * 60.0 / min_bpm)))
    start_period = frame_rate * 60.0 / float(start_bpm)
    native = f64(onset)
    period = lib().mls_tempo_period(
        addr(native), native.size, min_period, max_period, start_period
    )
    return np.atleast_1d(frame_rate * 60.0 / period)
