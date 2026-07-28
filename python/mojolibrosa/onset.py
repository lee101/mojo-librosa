from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import lfilter

from .core import power_to_db
from .feature import melspectrogram


def onset_strength(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    S: np.ndarray | None = None,
    lag: int = 1,
    max_size: int = 1,
    ref: np.ndarray | None = None,
    detrend: bool = False,
    center: bool = True,
    feature: Callable[..., np.ndarray] | None = None,
    aggregate: Callable[..., np.ndarray] | None = None,
    **kwargs: object,
) -> np.ndarray:
    if lag <= 0 or max_size <= 0:
        raise ValueError("lag and max_size must be positive integers")
    n_fft = int(kwargs.pop("n_fft", 2048))
    hop_length = int(kwargs.pop("hop_length", 512))
    if S is None:
        if y is None:
            raise ValueError("y or S must be provided")
        feature = melspectrogram if feature is None else feature
        kwargs.setdefault("fmax", 0.5 * sr)
        S = power_to_db(
            np.abs(
                feature(
                    y=y,
                    sr=sr,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    **kwargs,
                )
            )
        )
    S = np.atleast_2d(S)
    if ref is None:
        ref = S if max_size == 1 else maximum_filter1d(S, max_size, axis=-2)
    elif ref.shape != S.shape:
        raise ValueError("Reference spectrum shape must match S")
    onset = np.maximum(0.0, S[..., lag:] - ref[..., :-lag])
    aggregate = np.mean if aggregate is None else aggregate
    onset = aggregate(onset, axis=-2)
    pad_width = lag + (n_fft // (2 * hop_length) if center else 0)
    onset = np.pad(onset, [(0, 0)] * (onset.ndim - 1) + [(pad_width, 0)])
    if detrend:
        onset = lfilter([1.0, -1.0], [1.0, -0.99], onset, axis=-1)
    if center:
        onset = onset[..., : S.shape[-1]]
    return onset
