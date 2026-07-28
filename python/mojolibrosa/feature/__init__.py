from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy.fftpack import dct

from .. import filters
from .._lib import addr, lib
from ..core import power_to_db, stft


def _project(matrix: np.ndarray, source: np.ndarray) -> np.ndarray:
    source_array = np.asarray(source)
    if source_array.ndim < 2:
        raise ValueError("Spectrogram input must have at least two dimensions")
    if source_array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("Spectrogram input must use float32 or float64")
    if not np.all(np.isfinite(source_array)):
        raise ValueError("Spectrogram input is not finite everywhere")
    leading = source_array.shape[:-2]
    inner, columns = source_array.shape[-2:]
    if inner == 0 or columns == 0 or any(size == 0 for size in leading):
        raise ValueError("Spectrogram dimensions must be non-empty")
    if matrix.ndim != 2 or matrix.shape[1] != inner or matrix.shape[0] == 0:
        raise ValueError("Projection matrix shape is incompatible with the spectrogram")
    rows = matrix.shape[0]
    batch = int(np.prod(leading)) if leading else 1
    use_f32 = source_array.dtype == np.float32
    native_dtype = np.float32 if use_f32 else np.float64
    matrix_native = np.ascontiguousarray(matrix, dtype=native_dtype)
    source_native = np.ascontiguousarray(source_array, dtype=native_dtype)
    result = np.empty((batch, rows, columns), dtype=native_dtype)
    function = lib().mls_project_f32 if use_f32 else lib().mls_project
    function(
        addr(matrix_native),
        addr(source_native),
        addr(result),
        batch,
        rows,
        inner,
        columns,
    )
    return result.reshape(leading + (rows, columns))


@lru_cache(maxsize=16)
def _dct_basis(
    size: int, n_mfcc: int, dct_type: int, norm: str | None
) -> np.ndarray:
    basis = dct(
        np.eye(size, dtype=np.float64),
        type=dct_type,
        axis=0,
        norm=norm,
    )[:n_mfcc]
    basis.setflags(write=False)
    return basis


def melspectrogram(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    S: np.ndarray | None = None,
    n_fft: int = 2048,
    hop_length: int = 512,
    win_length: int | None = None,
    window: object = "hann",
    center: bool = True,
    pad_mode: object = "constant",
    power: float = 2.0,
    **kwargs: object,
) -> np.ndarray:
    if S is None:
        if y is None:
            raise ValueError("Either y or S must be provided")
        spectrum = stft(
            y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window=window,
            center=center,
            pad_mode=pad_mode,
        )
        S = np.abs(spectrum) ** power
    else:
        S = np.asarray(S)
        if S.ndim < 2 or S.shape[-2] < 2:
            raise ValueError("S must have at least two frequency bins and one time frame")
        n_fft = 2 * (S.shape[-2] - 1)
    basis = filters.mel(sr=sr, n_fft=n_fft, **kwargs)
    projected = _project(basis, S)
    return projected.astype(np.result_type(S.dtype, basis.dtype), copy=False)


def mfcc(
    *,
    y: np.ndarray | None = None,
    sr: float = 22050,
    S: np.ndarray | None = None,
    n_mfcc: int = 20,
    dct_type: int = 2,
    norm: str | None = "ortho",
    lifter: float = 0,
    mel_norm: str | float | None = "slaney",
    **kwargs: object,
) -> np.ndarray:
    if S is None:
        S = power_to_db(
            melspectrogram(y=y, sr=sr, norm=mel_norm, **kwargs)
        )
    else:
        S = np.asarray(S)
    if S.ndim < 2:
        raise ValueError("S must have frequency and time dimensions")
    n_mfcc = int(n_mfcc)
    if n_mfcc <= 0 or n_mfcc > S.shape[-2]:
        raise ValueError("n_mfcc must be between 1 and the number of input bands")
    if dct_type not in (1, 2, 3):
        raise ValueError("dct_type must be 1, 2, or 3")
    if dct_type == 1 and norm is not None:
        raise ValueError("DCT-I is not compatible with norm='ortho'")
    basis = _dct_basis(S.shape[-2], n_mfcc, dct_type, norm)
    result = _project(basis, S)
    if lifter > 0:
        index = np.arange(1, 1 + int(n_mfcc), dtype=result.dtype)
        shape = (1,) * (result.ndim - 2) + (int(n_mfcc), 1)
        result *= (
            1 + np.sin(np.pi * index / lifter) * lifter / 2
        ).reshape(shape)
    elif lifter < 0:
        raise ValueError("MFCC lifter must be a non-negative number")
    return result.astype(S.dtype, copy=False)


from .rhythm import tempo

__all__ = ["melspectrogram", "mfcc", "tempo"]
