# mojo-librosa

`mojo-librosa` is a standalone Mojo port of the compute-heavy core of
[librosa](https://librosa.org/). It exposes a Python package named
`mojolibrosa` whose covered functions use the same names and keyword-oriented
signatures as librosa. Python handles validation and array orchestration; the
FFT, dense spectral projections, resampling filter, tempo autocorrelation, and
beat dynamic program execute in a Mojo shared library.

This is an independent implementation, not a wrapper around librosa. Librosa
is installed only as a development dependency for parity tests and benchmarks.

## Coverage

Implemented:

- `stft`: real mono or multi-channel input, centered and uncentered framing,
  librosa-compatible windows, padding modes, dtype selection, and `out`
- `filters.mel` and mel/Hz conversion
- `feature.melspectrogram`, from either audio or a power spectrogram
- `feature.mfcc`, including DCT types I/II/III, orthogonal normalization, and
  liftering
- `resample`: multi-channel and arbitrary-axis Hann-windowed band-limited
  interpolation, dtype preservation, output length fixing, and energy scaling
- `onset.onset_strength`, `feature.tempo`, and `beat.beat_track`, including
  fixed or estimated global tempo, frame/sample/time units, trimming, and
  dense output
- `power_to_db`, `frames_to_samples`, and `frames_to_time`

Not implemented:

- audio loading, effects, harmonic/percussive separation, chroma, CQT/VQT,
  reassignment, display helpers, or the rest of librosa's broad API
- non-power-of-two STFT sizes
- time-varying beat tempo and probabilistic tempo priors
- soxr's exact resampling modes; `res_type` selects native filter widths, but
  the resampler is this project's windowed-sinc implementation

The tests compare representative paths for every item above against librosa
0.11.0. STFT, mel, MFCC, filter-bank, tempo, and fixed-tempo beat results agree
to the tolerances encoded in the tests. Resampling is compared on band-limited
signals because its filter design intentionally differs from soxr. The suite
also exercises both native dtypes, multi-channel layouts, SIMD tails, parallel
thresholds, supported resampling widths, and invalid FFI inputs.

## Install

The repository pins the tested Mojo nightly:

```bash
pixi install
pixi run build
```

The build produces `dist/libmojo-librosa.so`. Pixi activates `python/` on
`PYTHONPATH`; the Python loader also rebuilds a missing or stale library.

## Usage

```python
import numpy as np
import mojolibrosa as librosa

sr = 22050
time = np.arange(sr, dtype=np.float32) / sr
audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)

stft = librosa.stft(audio, n_fft=1024, hop_length=256)
mel = librosa.feature.melspectrogram(
    y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=64
)
mfcc = librosa.feature.mfcc(
    y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=64, n_mfcc=13
)
audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000)

print(stft.shape, mel.shape, mfcc.shape, audio_16k.shape)
# (513, 87) (64, 87) (13, 87) (16000,)
```

## How it works

`src/librosa.mojo` is one compilation unit with non-parametric C ABI exports.
Python validates non-empty C-contiguous NumPy buffers and exact float32/float64
dtypes, then passes their addresses and dimensions through `ctypes`; the local
Python references keep every buffer alive for the synchronous call. Mojo reconstructs
`UnsafePointer[..., AnyOrigin[mut=True]]` values. Python owns every allocation,
so the shared library has no cross-language allocation or lifetime protocol.

The STFT uses an in-place radix-2 Cooley-Tukey FFT over interleaved complex
float64 work buffers and parallelizes independent frames above a size
threshold. Spectrograms use row-major `(..., frequency, time)` layout; mel and
DCT projections use native-width SIMD across the contiguous time axis, retain
float32 buffers without conversion, and parallelize sufficiently large row
sets. Results are cast back to librosa-compatible float32/complex64 when the
input is float32.

The resampler caches bounded rational-phase Hann-windowed sinc tables, then
evaluates each output with a SIMD dot product and parallelizes only very large
outputs. Exact integer decimation avoids per-output phase division. Mel
projection bounds each filter to its nonzero frequency support before running
the SIMD time-axis loop.
Unusual rates which would require a large phase table retain the scalar
fallback. The beat tracker precomputes its Gaussian and transition penalties,
vectorizes local scores, parallelizes them above a threshold, and preserves
the sequential Ellis dynamic-programming recurrence.

No GPU path is shipped. A light STFT GPU prototype used about 19 MB of device
memory and was measured through the locked benchmark at 12.495 ms, versus
5.003 ms for the final CPU path. Transfer, launch, and global-memory FFT costs
outweighed the available parallelism. The remaining candidates either have low
arithmetic intensity after sparse support pruning or complete in about 1 ms on
CPU, where launch overhead dominates.

## Tests

```bash
pixi run build
pixi run test
```

The current suite contains 69 parity, boundary-safety, and behavior tests using the real
upstream package.

## Benchmarks

Run benchmarks only through `pixi run bench`; the task uses a machine-wide
lock. Lower time is better, and speedup is `librosa / mojolibrosa`.

Measured by `pixi run bench` on this machine on August 24, 2026. Each cell is the
best of five warmed runs. Machine: Intel Xeon E5-2697 v4 at 2.30 GHz, Linux
6.8.0-136-generic, Python 3.13.14.

| Case | mojolibrosa | librosa | Speedup |
|---|---:|---:|---:|
| STFT, 12 s, n_fft=2048 | 5.003 ms | 12.143 ms | 2.43x |
| Mel projection, 128 x 600 | 6.988 ms | 28.993 ms | 4.15x |
| MFCC DCT, 20 x 1500 | 0.954 ms | 1.468 ms | 1.54x |
| Resample, 5 s, 48 kHz to 16 kHz | 4.011 ms | 2.689 ms | 0.67x |
| Beat DP, 20k onset frames, fixed BPM | 2.383 ms | 10.512 ms | 4.41x |

## License

MIT
