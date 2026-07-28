import inspect

import librosa
import numpy as np
import pytest

import mojolibrosa as ml


@pytest.fixture(scope="module")
def audio():
    rng = np.random.default_rng(7)
    return rng.normal(size=8192)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_stft_default_parity(audio, dtype):
    y = audio.astype(dtype)
    ours = ml.stft(y, n_fft=512, hop_length=128)
    theirs = librosa.stft(y, n_fft=512, hop_length=128)
    assert ours.shape == theirs.shape
    assert ours.dtype == theirs.dtype
    assert np.allclose(ours, theirs, rtol=2e-6, atol=2e-6)


@pytest.mark.parametrize("pad_mode", ["constant", "reflect", "edge"])
def test_stft_padding_parity(audio, pad_mode):
    ours = ml.stft(audio, n_fft=256, hop_length=100, pad_mode=pad_mode)
    theirs = librosa.stft(audio, n_fft=256, hop_length=100, pad_mode=pad_mode)
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_stft_uncentered_short_window_parity(audio):
    ours = ml.stft(
        audio, n_fft=512, win_length=300, hop_length=75, center=False, window="hamming"
    )
    theirs = librosa.stft(
        audio, n_fft=512, win_length=300, hop_length=75, center=False, window="hamming"
    )
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_stft_multichannel_parity(audio):
    y = np.stack([audio, 0.5 * audio])
    ours = ml.stft(y, n_fft=512, hop_length=128)
    theirs = librosa.stft(y, n_fft=512, hop_length=128)
    assert ours.shape == (2, 257, 65)
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("n_frames", [511, 512])
def test_stft_parallel_threshold_parity(n_frames):
    rng = np.random.default_rng(n_frames)
    n_fft = 256
    hop_length = 64
    n_samples = n_fft + (n_frames - 1) * hop_length
    y = rng.normal(size=n_samples)
    ours = ml.stft(
        y, n_fft=n_fft, hop_length=hop_length, center=False
    )
    theirs = librosa.stft(
        y, n_fft=n_fft, hop_length=hop_length, center=False
    )
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_stft_out_parameter(audio):
    expected = librosa.stft(audio, n_fft=256, hop_length=64)
    storage = np.empty(expected.shape[:-1] + (expected.shape[-1] + 3,), dtype=complex)
    result = ml.stft(audio, n_fft=256, hop_length=64, out=storage)
    assert np.shares_memory(result, storage)
    assert np.allclose(result, expected, rtol=1e-12, atol=1e-12)


def test_stft_signature_matches_upstream():
    ours = inspect.signature(ml.stft)
    theirs = inspect.signature(librosa.stft)
    assert tuple(ours.parameters) == tuple(theirs.parameters)


def test_stft_rejects_non_power_of_two(audio):
    with pytest.raises(ValueError, match="power-of-two"):
        ml.stft(audio, n_fft=300)


@pytest.mark.parametrize(
    "audio",
    [np.arange(32, dtype=np.int16), np.empty((0, 32), dtype=np.float32)],
)
def test_stft_rejects_unsafe_ffi_inputs(audio):
    with pytest.raises((TypeError, ValueError)):
        ml.stft(audio, n_fft=16)
