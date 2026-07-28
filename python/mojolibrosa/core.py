from __future__ import annotations

import math
import warnings
from collections.abc import Callable
from functools import lru_cache

import numpy as np
from scipy.signal import get_window

from ._lib import addr, f64, lib


def _positive_int(value: int, name: str) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _pad_center(data: np.ndarray, size: int) -> np.ndarray:
    if data.size > size:
        raise ValueError(f"window size={data.size} cannot exceed n_fft={size}")
    left = (size - data.size) // 2
    return np.pad(data, (left, size - data.size - left))


@lru_cache(maxsize=32)
def _resample_kernel(
    orig_sr: int, target_sr: int, half_width: int, dtype: str
) -> tuple[np.ndarray, int, int]:
    common = math.gcd(orig_sr, target_sr)
    input_step = orig_sr // common
    phase_count = target_sr // common
    cutoff = min(float(target_sr) / float(orig_sr), 1.0)
    support = half_width / cutoff
    radius = int(math.ceil(support))
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    phases = np.arange(phase_count, dtype=np.int64)
    fractions = ((phases * input_step) % phase_count) / phase_count
    distances = fractions[:, None] - offsets
    phase = distances / support
    weights = cutoff * np.sinc(cutoff * distances)
    weights *= 0.5 + 0.5 * np.cos(np.pi * phase)
    weights[np.abs(phase) > 1.0] = 0.0
    weights /= weights.sum(axis=1, keepdims=True)
    weights = np.ascontiguousarray(weights, dtype=np.dtype(dtype))
    weights.setflags(write=False)
    return weights, radius, input_step


def stft(
    y: np.ndarray,
    *,
    n_fft: int = 2048,
    hop_length: int | None = None,
    win_length: int | None = None,
    window: object = "hann",
    center: bool = True,
    dtype: np.dtype | type | str | None = None,
    pad_mode: object = "constant",
    out: np.ndarray | None = None,
) -> np.ndarray:
    y_array = np.asarray(y)
    if y_array.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("Audio data must use float32 or float64")
    if np.iscomplexobj(y_array):
        raise ValueError("Audio data must be real-valued")
    if y_array.ndim == 0:
        raise ValueError("Audio data must have at least one dimension")
    if not np.all(np.isfinite(y_array)):
        raise ValueError("Audio buffer is not finite everywhere")

    n_fft = _positive_int(n_fft, "n_fft")
    win_length = n_fft if win_length is None else _positive_int(win_length, "win_length")
    hop_length = win_length // 4 if hop_length is None else _positive_int(hop_length, "hop_length")
    if win_length > n_fft:
        raise ValueError("win_length must not exceed n_fft")
    if n_fft & (n_fft - 1):
        raise ValueError("mojolibrosa.stft currently requires a power-of-two n_fft")

    if isinstance(window, (str, tuple)) or np.isscalar(window):
        fft_window = get_window(window, win_length, fftbins=True)
    elif callable(window):
        fft_window = np.asarray(window(win_length))
    else:
        fft_window = np.asarray(window)
        if fft_window.size != win_length:
            raise ValueError("Window size mismatch")
    fft_window = f64(_pad_center(np.asarray(fft_window), n_fft))

    if center:
        if n_fft > y_array.shape[-1]:
            warnings.warn(
                f"n_fft={n_fft} is too large for input signal of length={y_array.shape[-1]}",
                stacklevel=2,
            )
        padding = [(0, 0)] * y_array.ndim
        padding[-1] = (n_fft // 2, n_fft // 2)
        y_array = np.pad(y_array, padding, mode=pad_mode)
    elif n_fft > y_array.shape[-1]:
        raise ValueError(
            f"n_fft={n_fft} is too large for uncentered signal of length={y_array.shape[-1]}"
        )

    y_native = f64(y_array)
    leading = y_native.shape[:-1]
    if any(size == 0 for size in leading):
        raise ValueError("Audio data must not have empty channel dimensions")
    n_samples = y_native.shape[-1]
    channels = int(np.prod(leading)) if leading else 1
    n_frames = 1 + (n_samples - n_fft) // hop_length
    work = np.empty((channels, n_frames, n_fft), dtype=np.complex128)
    lib().mls_stft(
        addr(y_native),
        addr(fft_window),
        addr(work),
        channels,
        n_samples,
        n_frames,
        n_fft,
        hop_length,
    )
    result = np.moveaxis(work[..., : 1 + n_fft // 2], -1, -2)
    result = result.reshape(leading + (1 + n_fft // 2, n_frames))

    if dtype is None:
        dtype = np.complex64 if y_array.dtype == np.float32 else np.complex128
    result = result.astype(dtype, copy=False)
    if out is not None:
        if out.shape[:-1] != result.shape[:-1] or out.shape[-1] < result.shape[-1]:
            raise ValueError(f"output shape={out.shape} is incompatible with {result.shape}")
        target = out[..., : result.shape[-1]]
        target[...] = result
        return target
    return result


def resample(
    y: np.ndarray,
    *,
    orig_sr: float,
    target_sr: float,
    res_type: str = "soxr_hq",
    fix: bool = True,
    scale: bool = False,
    axis: int = -1,
    **kwargs: object,
) -> np.ndarray:
    if kwargs:
        names = ", ".join(sorted(kwargs))
        raise NotImplementedError(f"resampler options are not implemented: {names}")
    supported_res_types = {
        "soxr_vhq", "soxr_hq", "kaiser_best", "soxr_mq",
        "kaiser_fast", "soxr_lq", "linear",
    }
    if res_type not in supported_res_types:
        raise ValueError(f"Unsupported res_type: {res_type}")
    if orig_sr <= 0 or target_sr <= 0:
        raise ValueError("Sample rates must be positive")
    source = np.asarray(y)
    if source.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("Audio data must use float32 or float64")
    if source.ndim == 0:
        raise ValueError("Audio data must have at least one dimension")
    if not np.all(np.isfinite(source)):
        raise ValueError("Audio buffer is not finite everywhere")
    if orig_sr == target_sr:
        return np.array(source, copy=True)

    axis = int(axis)
    if axis < 0:
        axis += source.ndim
    if axis < 0 or axis >= source.ndim:
        raise np.exceptions.AxisError(axis, source.ndim)
    moved = np.moveaxis(source, axis, -1)
    n_in = moved.shape[-1]
    if n_in == 0:
        raise ValueError("Cannot resample an empty signal")
    n_out = int(math.ceil(n_in * float(target_sr) / float(orig_sr)))
    channels = int(np.prod(moved.shape[:-1])) if moved.ndim > 1 else 1
    if channels == 0:
        raise ValueError("Audio data must not have empty channel dimensions")
    orig_integer = int(orig_sr)
    target_integer = int(target_sr)
    integral_rates = (
        float(orig_integer) == float(orig_sr)
        and float(target_integer) == float(target_sr)
    )
    phase_count = (
        target_integer // math.gcd(orig_integer, target_integer)
        if integral_rates
        else 0
    )
    use_phase_table = integral_rates and phase_count <= 4096
    use_f32 = source.dtype == np.float32 and use_phase_table
    native_dtype = np.float32 if use_f32 else np.float64
    native = np.ascontiguousarray(moved, dtype=native_dtype)
    result = np.empty((channels, n_out), dtype=native_dtype)
    widths = {
        "soxr_vhq": 48,
        "soxr_hq": 32,
        "kaiser_best": 32,
        "soxr_mq": 20,
        "kaiser_fast": 12,
        "soxr_lq": 8,
        "linear": 2,
    }
    half_width = widths[res_type]
    if use_phase_table:
        weights, radius, input_step = _resample_kernel(
            orig_integer, target_integer, half_width, np.dtype(native_dtype).str
        )
        function = lib().mls_resample_f32 if use_f32 else lib().mls_resample
        function(
            addr(native),
            addr(weights),
            addr(result),
            channels,
            n_in,
            n_out,
            radius,
            weights.shape[0],
            input_step,
        )
    else:
        lib().mls_resample_slow(
            addr(native),
            addr(result),
            channels,
            n_in,
            n_out,
            float(orig_sr),
            float(target_sr),
            half_width,
        )
    if scale:
        result /= math.sqrt(float(target_sr) / float(orig_sr))
    result = result.reshape(moved.shape[:-1] + (n_out,))
    if not fix:
        natural = int(round(n_in * float(target_sr) / float(orig_sr)))
        result = result[..., :natural]
    result = np.moveaxis(result, -1, axis)
    return result.astype(source.dtype, copy=False)


def power_to_db(
    S: np.ndarray,
    *,
    ref: float | Callable[[np.ndarray], object] = 1.0,
    amin: float = 1e-10,
    top_db: float | None = 80.0,
) -> np.ndarray:
    S_array = np.asarray(S)
    if amin <= 0:
        raise ValueError("amin must be strictly positive")
    if np.iscomplexobj(S_array):
        warnings.warn(
            "power_to_db was called on complex input so phase information will be discarded",
            stacklevel=2,
        )
        magnitude = np.abs(S_array)
    else:
        magnitude = S_array
    ref_value = ref(magnitude) if callable(ref) else abs(ref)
    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec


def frames_to_samples(frames: np.ndarray, *, hop_length: int = 512) -> np.ndarray:
    return np.asarray(frames) * hop_length


def frames_to_time(
    frames: np.ndarray, *, sr: float = 22050, hop_length: int = 512
) -> np.ndarray:
    return np.asarray(frames) * hop_length / float(sr)
