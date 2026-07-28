from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIB = os.path.join(ROOT, "dist", "libmojo-librosa.so")
SOURCE = os.path.join(ROOT, "src", "librosa.mojo")

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "mls_initialize": ([], None),
    "mls_stft": ([I] * 8, None),
    "mls_project": ([I] * 7, None),
    "mls_project_f32": ([I] * 7, None),
    "mls_resample": ([I] * 9, None),
    "mls_resample_f32": ([I] * 9, None),
    "mls_resample_slow": ([I, I, I, I, I, F, F, I], None),
    "mls_beat_dp": ([I, I, I, I, I, I, F], None),
    "mls_tempo_period": ([I, I, I, I, F], I),
}


class BuildError(RuntimeError):
    pass


def build(force: bool = False) -> str:
    if (
        not force
        and os.path.exists(LIB)
        and os.path.getmtime(LIB) >= os.path.getmtime(SOURCE)
    ):
        return LIB
    proc = subprocess.run(
        ["bash", os.path.join(ROOT, "build", "build.sh")],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if proc.returncode or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_library: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _library
    if _library is None:
        _library = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_library, name)
            function.argtypes = argtypes
            function.restype = restype
        _library.mls_initialize()
    return _library


def f64(a: object) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def addr(a: np.ndarray) -> int:
    if not isinstance(a, np.ndarray):
        raise TypeError("FFI buffers must be NumPy arrays")
    if a.size == 0:
        raise ValueError("FFI buffers must not be empty")
    if not a.flags.c_contiguous:
        raise ValueError("FFI buffers must be C-contiguous")
    address = int(a.ctypes.data)
    if address == 0:
        raise ValueError("FFI buffers must have a non-null address")
    return address
