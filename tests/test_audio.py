import numpy as np
import pytest


def test_resample_48k_to_16k():
    from src.audio import resample_pcm
    samples_48k = np.zeros(480, dtype=np.int16)
    result = resample_pcm(samples_48k.tobytes(), from_rate=48000, to_rate=16000)
    result_samples = np.frombuffer(result, dtype=np.int16)
    assert len(result_samples) == 160


def test_resample_24k_to_48k():
    from src.audio import resample_pcm
    samples_24k = np.zeros(240, dtype=np.int16)
    result = resample_pcm(samples_24k.tobytes(), from_rate=24000, to_rate=48000)
    result_samples = np.frombuffer(result, dtype=np.int16)
    assert len(result_samples) == 480


def test_stereo_to_mono():
    from src.audio import stereo_to_mono
    stereo = np.array([100, 200, 100, 200], dtype=np.int16)
    result = stereo_to_mono(stereo.tobytes())
    mono = np.frombuffer(result, dtype=np.int16)
    assert len(mono) == 2
    assert mono[0] == 150


def test_mono_to_stereo():
    from src.audio import mono_to_stereo
    mono = np.array([100, 200], dtype=np.int16)
    result = mono_to_stereo(mono.tobytes())
    stereo = np.frombuffer(result, dtype=np.int16)
    assert len(stereo) == 4
    assert stereo[0] == 100
    assert stereo[1] == 100
