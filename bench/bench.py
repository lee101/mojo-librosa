from __future__ import annotations

import math
import os
import platform
import sys
import time

import librosa
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

import mojolibrosa as ml


def best_time(function, repeat: int = 5) -> float:
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as stream:
            for line in stream:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def cases():
    rng = np.random.default_rng(0)

    y_stft = rng.normal(size=22050 * 12).astype(np.float32)
    yield (
        "STFT, 12 s, n_fft=2048",
        lambda: ml.stft(y_stft, n_fft=2048, hop_length=512),
        lambda: librosa.stft(y_stft, n_fft=2048, hop_length=512),
    )

    S = rng.random((1025, 600), dtype=np.float32)
    yield (
        "Mel projection, 128 x 600",
        lambda: ml.feature.melspectrogram(S=S, sr=22050, n_mels=128),
        lambda: librosa.feature.melspectrogram(S=S, sr=22050, n_mels=128),
    )

    log_mel = rng.normal(size=(128, 1500)).astype(np.float32)
    yield (
        "MFCC DCT, 20 x 1500",
        lambda: ml.feature.mfcc(S=log_mel, n_mfcc=20),
        lambda: librosa.feature.mfcc(S=log_mel, n_mfcc=20),
    )

    y_resample = rng.normal(size=48000 * 5).astype(np.float32)
    yield (
        "Resample, 5 s, 48 kHz to 16 kHz",
        lambda: ml.resample(y_resample, orig_sr=48000, target_sr=16000),
        lambda: librosa.resample(y_resample, orig_sr=48000, target_sr=16000),
    )

    onset = np.maximum(0, rng.normal(size=20000))
    yield (
        "Beat DP, 20k onset frames, fixed BPM",
        lambda: ml.beat.beat_track(onset_envelope=onset, bpm=120),
        lambda: librosa.beat.beat_track(onset_envelope=onset, bpm=120),
    )


def main() -> None:
    rows = []
    for name, mojo_function, upstream_function in cases():
        mojo_function()
        upstream_function()
        mojo_seconds = best_time(mojo_function)
        upstream_seconds = best_time(upstream_function)
        rows.append((name, mojo_seconds, upstream_seconds, upstream_seconds / mojo_seconds))

    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; Python {platform.python_version()}")
    print()
    print("| Case | mojolibrosa | librosa | Speedup |")
    print("|---|---:|---:|---:|")
    for name, mojo_seconds, upstream_seconds, ratio in rows:
        ratio_text = f"{ratio:.3f}x" if ratio < 0.1 else f"{ratio:.2f}x"
        print(
            f"| {name} | {mojo_seconds * 1000:.3f} ms | "
            f"{upstream_seconds * 1000:.3f} ms | {ratio_text} |"
        )


if __name__ == "__main__":
    main()
