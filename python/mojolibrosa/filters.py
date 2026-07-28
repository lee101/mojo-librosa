from __future__ import annotations

import warnings

import numpy as np


def hz_to_mel(frequencies: object, *, htk: bool = False) -> np.ndarray:
    frequencies = np.asanyarray(frequencies)
    if htk:
        return 2595.0 * np.log10(1.0 + frequencies / 700.0)
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = np.log(6.4) / 27.0
    if frequencies.ndim:
        mels = np.array(mels, copy=True)
        mask = frequencies >= min_log_hz
        mels[mask] = min_log_mel + np.log(frequencies[mask] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        mels = min_log_mel + np.log(frequencies / min_log_hz) / logstep
    return mels


def mel_to_hz(mels: object, *, htk: bool = False) -> np.ndarray:
    mels = np.asanyarray(mels)
    if htk:
        return 700.0 * (10.0 ** (mels / 2595.0) - 1.0)
    f_sp = 200.0 / 3
    frequencies = f_sp * mels
    min_log_mel = 1000.0 / f_sp
    logstep = np.log(6.4) / 27.0
    if mels.ndim:
        frequencies = np.array(frequencies, copy=True)
        mask = mels >= min_log_mel
        frequencies[mask] = 1000.0 * np.exp(logstep * (mels[mask] - min_log_mel))
    elif mels >= min_log_mel:
        frequencies = 1000.0 * np.exp(logstep * (mels - min_log_mel))
    return frequencies


def mel_frequencies(
    n_mels: int = 128,
    *,
    fmin: float = 0.0,
    fmax: float = 11025.0,
    htk: bool = False,
) -> np.ndarray:
    minimum = hz_to_mel(fmin, htk=htk)
    maximum = hz_to_mel(fmax, htk=htk)
    return mel_to_hz(np.linspace(minimum, maximum, int(n_mels)), htk=htk)


def mel(
    *,
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    htk: bool = False,
    norm: str | float | None = "slaney",
    dtype: np.dtype | type | str = np.float32,
) -> np.ndarray:
    fmax = float(sr) / 2 if fmax is None else float(fmax)
    n_mels = int(n_mels)
    fftfreqs = np.fft.rfftfreq(int(n_fft), 1.0 / float(sr))
    mel_f = mel_frequencies(n_mels + 2, fmin=fmin, fmax=fmax, htk=htk)
    fdiff = np.diff(mel_f)
    ramps = np.subtract.outer(mel_f, fftfreqs)
    weights = np.zeros((n_mels, 1 + int(n_fft) // 2), dtype=dtype)
    for index in range(n_mels):
        lower = -ramps[index] / fdiff[index]
        upper = ramps[index + 2] / fdiff[index + 1]
        weights[index] = np.maximum(0, np.minimum(lower, upper))
    if norm == "slaney":
        weights *= (2.0 / (mel_f[2:] - mel_f[:-2]))[:, np.newaxis]
    elif norm is not None:
        p = float(norm)
        if np.isposinf(p):
            lengths = np.max(np.abs(weights), axis=-1, keepdims=True)
        elif np.isneginf(p):
            lengths = np.min(np.abs(weights), axis=-1, keepdims=True)
        elif p == 0:
            lengths = np.sum(weights != 0, axis=-1, keepdims=True)
        else:
            lengths = np.sum(np.abs(weights) ** p, axis=-1, keepdims=True) ** (1.0 / p)
        weights /= np.where(lengths > 0, lengths, 1)
    if not np.all((mel_f[:-2] == 0) | (weights.max(axis=1) > 0)):
        warnings.warn("Empty filters detected in mel frequency basis", stacklevel=2)
    return weights
