import inspect

import librosa
import numpy as np
import pytest

import mojolibrosa as ml


@pytest.mark.parametrize("orig_sr,target_sr", [(22050, 16000), (16000, 22050), (48000, 8000)])
def test_resample_bandlimited_parity(orig_sr, target_sr):
    duration = 0.5
    t = np.arange(int(orig_sr * duration)) / orig_sr
    cutoff = 0.35 * min(orig_sr, target_sr)
    y = (
        0.6 * np.sin(2 * np.pi * 440 * t)
        + 0.25 * np.sin(2 * np.pi * min(1700, cutoff) * t + 0.2)
    ).astype(np.float32)
    ours = ml.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    theirs = librosa.resample(y, orig_sr=orig_sr, target_sr=target_sr)
    assert ours.shape == theirs.shape
    assert ours.dtype == theirs.dtype
    edge = min(100, ours.size // 10)
    assert np.allclose(ours[edge:-edge], theirs[edge:-edge], rtol=2e-3, atol=2e-4)


def test_resample_multichannel_axis_and_scale():
    sr = 16000
    t = np.arange(4000) / sr
    y = np.stack([np.sin(2 * np.pi * 220 * t), np.sin(2 * np.pi * 880 * t)], axis=1)
    ours = ml.resample(y, orig_sr=sr, target_sr=22050, axis=0, scale=True)
    theirs = librosa.resample(y, orig_sr=sr, target_sr=22050, axis=0, scale=True)
    assert ours.shape == theirs.shape
    assert np.allclose(ours[100:-100], theirs[100:-100], rtol=2e-3, atol=2e-4)


@pytest.mark.parametrize("n_in", [786429, 786430])
def test_resample_parallel_threshold_and_simd_tail(n_in):
    sr = 48000
    t = np.arange(n_in) / sr
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ours = ml.resample(y, orig_sr=sr, target_sr=16000)
    theirs = librosa.resample(y, orig_sr=sr, target_sr=16000)
    assert np.allclose(ours[100:-100], theirs[100:-100], rtol=2e-3, atol=2e-4)


def test_resample_noninteger_rate_fallback():
    t = np.arange(2000) / 8000.5
    y = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    result = ml.resample(y, orig_sr=8000.5, target_sr=6000.25)
    assert result.dtype == y.dtype
    assert result.shape == (int(np.ceil(y.size * 6000.25 / 8000.5)),)
    assert np.all(np.isfinite(result))


def test_resample_large_phase_count_fallback():
    t = np.arange(2000) / 8000
    y = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    result = ml.resample(y, orig_sr=8000, target_sr=8001)
    assert result.dtype == y.dtype
    assert result.shape == (2001,)
    assert np.all(np.isfinite(result))


@pytest.mark.parametrize(
    "audio",
    [
        np.ones(32, dtype=np.float16),
        np.full(32, np.inf, dtype=np.float64),
        np.empty((0, 32), dtype=np.float32),
    ],
)
def test_resample_rejects_unsafe_ffi_inputs(audio):
    with pytest.raises((TypeError, ValueError)):
        ml.resample(audio, orig_sr=48000, target_sr=16000)


def test_resample_signature_matches_upstream():
    assert tuple(inspect.signature(ml.resample).parameters) == tuple(
        inspect.signature(librosa.resample).parameters
    )


@pytest.mark.parametrize(
    "res_type",
    [
        "soxr_vhq",
        "soxr_hq",
        "kaiser_best",
        "soxr_mq",
        "kaiser_fast",
        "soxr_lq",
        "linear",
    ],
)
def test_resample_supported_filter_widths(res_type):
    audio = np.linspace(-1, 1, 31, dtype=np.float32)
    result = ml.resample(
        audio, orig_sr=3, target_sr=2, res_type=res_type, fix=False
    )
    assert result.shape == (round(audio.size * 2 / 3),)
    assert result.dtype == audio.dtype
    assert np.all(np.isfinite(result))


def test_resample_rejects_unsupported_options():
    audio = np.ones(32, dtype=np.float32)
    with pytest.raises(ValueError, match="Unsupported res_type"):
        ml.resample(audio, orig_sr=2, target_sr=1, res_type="not-a-resampler")
    with pytest.raises(NotImplementedError, match="resampler options"):
        ml.resample(audio, orig_sr=2, target_sr=1, window="hann")


@pytest.mark.parametrize("trim", [False, True])
def test_beat_tracking_explicit_bpm_parity(trim):
    onset = np.zeros(240)
    onset[10:230:20] = np.linspace(0.5, 1.5, 11)
    ours = ml.beat.beat_track(
        onset_envelope=onset, sr=100, hop_length=1, bpm=300, trim=trim
    )
    theirs = librosa.beat.beat_track(
        onset_envelope=onset, sr=100, hop_length=1, bpm=300, trim=trim
    )
    assert ours[0] == theirs[0]
    assert np.array_equal(ours[1], theirs[1])


@pytest.mark.parametrize("bpm", [89, 120, 160])
def test_beat_dp_random_envelope_parity(bpm):
    rng = np.random.default_rng(bpm)
    onset = np.maximum(0, rng.normal(size=400))
    ours = ml.beat.beat_track(onset_envelope=onset, bpm=bpm)
    theirs = librosa.beat.beat_track(onset_envelope=onset, bpm=bpm)
    assert np.array_equal(ours[1], theirs[1])


def test_tempo_and_default_beat_parity():
    sr = 22050
    hop = 512
    frames_per_beat = round((sr / hop) * 60 / 120)
    onset = np.zeros(400)
    onset[10::frames_per_beat] = 1
    ours_tempo, ours_beats = ml.beat.beat_track(
        onset_envelope=onset, sr=sr, hop_length=hop, trim=False
    )
    their_tempo, their_beats = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, hop_length=hop, trim=False
    )
    assert np.allclose(ours_tempo, their_tempo)
    assert np.array_equal(ours_beats, their_beats)


def test_beat_units_and_dense_output():
    onset = np.zeros(200)
    onset[10:190:20] = 1
    _, frames = ml.beat.beat_track(onset_envelope=onset, bpm=300, sr=100, hop_length=4)
    _, samples = ml.beat.beat_track(
        onset_envelope=onset, bpm=300, sr=100, hop_length=4, units="samples"
    )
    assert np.array_equal(samples, frames * 4)
    _, dense = ml.beat.beat_track(
        onset_envelope=onset, bpm=300, sr=100, hop_length=4, sparse=False
    )
    assert dense.dtype == bool
    assert np.array_equal(np.flatnonzero(dense), frames)
    _, times = ml.beat.beat_track(
        onset_envelope=onset, bpm=300, sr=100, hop_length=4, units="time"
    )
    assert np.array_equal(times, frames * 4 / 100)


def test_tempo_rejects_unsupported_options():
    onset = np.zeros(100)
    onset[::10] = 1
    with pytest.raises(NotImplementedError, match="tempo options"):
        ml.feature.tempo(onset_envelope=onset, aggregate=np.mean)


def test_beat_empty_matches_upstream():
    onset = np.zeros(100)
    ours = ml.beat.beat_track(onset_envelope=onset)
    theirs = librosa.beat.beat_track(onset_envelope=onset)
    assert ours[0] == theirs[0] == 0
    assert np.array_equal(ours[1], theirs[1])


@pytest.mark.parametrize(
    "onset,bpm",
    [
        (np.array([1.0]), 120),
        (np.array([0.0, np.nan, 1.0]), 120),
        (np.array([0.0, 1.0, 0.0]), 1e12),
    ],
)
def test_beat_rejects_unsafe_ffi_inputs(onset, bpm):
    with pytest.raises(ValueError):
        ml.beat.beat_track(onset_envelope=onset, bpm=bpm)


def test_onset_and_beat_from_audio_parity():
    sr = 22050
    y = np.zeros(sr * 3, dtype=np.float32)
    for timestamp in np.arange(0.25, 3, 0.5):
        start = int(timestamp * sr)
        y[start : start + 100] += np.hanning(100)
    ours_onset = ml.onset.onset_strength(y=y, sr=sr)
    their_onset = librosa.onset.onset_strength(y=y, sr=sr)
    assert np.allclose(ours_onset, their_onset, rtol=2e-6, atol=1e-6)
    ours_tempo, ours_beats = ml.beat.beat_track(y=y, sr=sr, trim=False)
    their_tempo, their_beats = librosa.beat.beat_track(y=y, sr=sr, trim=False)
    assert np.allclose(ours_tempo, their_tempo)
    assert ours_beats.shape == their_beats.shape
    assert np.max(np.abs(ours_beats - their_beats)) <= 1
