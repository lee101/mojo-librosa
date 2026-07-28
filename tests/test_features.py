import librosa
import numpy as np
import pytest

import mojolibrosa as ml


@pytest.fixture(scope="module")
def audio():
    rng = np.random.default_rng(11)
    return rng.normal(size=11025).astype(np.float32)


@pytest.mark.parametrize("htk", [False, True])
def test_mel_filter_parity(htk):
    ours = ml.filters.mel(sr=22050, n_fft=1024, n_mels=64, fmin=40, fmax=8000, htk=htk)
    theirs = librosa.filters.mel(
        sr=22050, n_fft=1024, n_mels=64, fmin=40, fmax=8000, htk=htk
    )
    assert np.array_equal(ours, theirs)


@pytest.mark.parametrize("norm", [None, 1, 2, np.inf])
def test_mel_filter_numeric_norm_parity(norm):
    ours = ml.filters.mel(sr=16000, n_fft=512, n_mels=32, norm=norm)
    theirs = librosa.filters.mel(sr=16000, n_fft=512, n_mels=32, norm=norm)
    assert np.allclose(ours, theirs, rtol=2e-6, atol=1e-8)


def test_mel_frequency_conversion_parity():
    hz = np.geomspace(20, 20000, 100)
    assert np.allclose(ml.filters.hz_to_mel(hz), librosa.hz_to_mel(hz))
    mels = np.linspace(0, 50, 100)
    assert np.allclose(ml.filters.mel_to_hz(mels), librosa.mel_to_hz(mels))


def test_melspectrogram_from_audio_parity(audio):
    ours = ml.feature.melspectrogram(
        y=audio, sr=22050, n_fft=512, hop_length=128, n_mels=40
    )
    theirs = librosa.feature.melspectrogram(
        y=audio, sr=22050, n_fft=512, hop_length=128, n_mels=40
    )
    assert ours.dtype == theirs.dtype
    assert np.allclose(ours, theirs, rtol=2e-6, atol=2e-6)


def test_melspectrogram_from_s_parity():
    rng = np.random.default_rng(3)
    S = rng.random((2, 257, 30)).astype(np.float64)
    ours = ml.feature.melspectrogram(S=S, sr=16000, n_mels=24, fmax=7000)
    theirs = librosa.feature.melspectrogram(S=S, sr=16000, n_mels=24, fmax=7000)
    assert ours.shape == (2, 24, 30)
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_projection_simd_tail_parity(dtype):
    rng = np.random.default_rng(31)
    S = rng.random((257, 13)).astype(dtype)
    ours = ml.feature.melspectrogram(S=S, sr=16000, n_mels=17)
    theirs = librosa.feature.melspectrogram(S=S, sr=16000, n_mels=17)
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    assert np.allclose(ours, theirs, rtol=tolerance, atol=tolerance)


def test_mfcc_from_audio_parity(audio):
    ours = ml.feature.mfcc(
        y=audio, sr=22050, n_fft=512, hop_length=128, n_mels=40, n_mfcc=13
    )
    theirs = librosa.feature.mfcc(
        y=audio, sr=22050, n_fft=512, hop_length=128, n_mels=40, n_mfcc=13
    )
    assert np.allclose(ours, theirs, rtol=2e-5, atol=5e-5)


@pytest.mark.parametrize("dct_type,norm", [(1, None), (2, "ortho"), (3, "ortho")])
def test_mfcc_from_log_s_parity(dct_type, norm):
    rng = np.random.default_rng(5)
    S = rng.normal(size=(40, 50))
    ours = ml.feature.mfcc(S=S, n_mfcc=18, dct_type=dct_type, norm=norm)
    theirs = librosa.feature.mfcc(S=S, n_mfcc=18, dct_type=dct_type, norm=norm)
    assert np.allclose(ours, theirs, rtol=1e-12, atol=1e-12)


def test_mfcc_lifter_parity():
    rng = np.random.default_rng(13)
    S = rng.normal(size=(2, 40, 25)).astype(np.float32)
    ours = ml.feature.mfcc(S=S, n_mfcc=13, lifter=22)
    theirs = librosa.feature.mfcc(S=S, n_mfcc=13, lifter=22)
    assert np.allclose(ours, theirs, rtol=2e-6, atol=2e-5)


def test_power_to_db_parity():
    rng = np.random.default_rng(17)
    S = rng.lognormal(size=(30, 20)).astype(np.float32)
    assert np.array_equal(
        ml.power_to_db(S, ref=np.max, top_db=60),
        librosa.power_to_db(S, ref=np.max, top_db=60),
    )


@pytest.mark.parametrize(
    "S",
    [
        np.empty((257, 0), dtype=np.float32),
        np.ones((257, 3), dtype=np.float16),
        np.full((257, 3), np.nan, dtype=np.float64),
    ],
)
def test_projection_rejects_unsafe_ffi_inputs(S):
    with pytest.raises((TypeError, ValueError)):
        ml.feature.melspectrogram(S=S)
