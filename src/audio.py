import numpy as np
import discord


def resample_pcm(pcm_data: bytes, from_rate: int, to_rate: int) -> bytes:
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    ratio = to_rate / from_rate
    new_length = int(len(samples) * ratio)
    indices = np.linspace(0, len(samples) - 1, new_length).astype(int)
    resampled = samples[indices]
    return resampled.astype(np.int16).tobytes()


def stereo_to_mono(pcm_data: bytes) -> bytes:
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    left = samples[0::2]
    right = samples[1::2]
    mono = ((left.astype(np.int32) + right.astype(np.int32)) // 2).astype(np.int16)
    return mono.tobytes()


def mono_to_stereo(pcm_data: bytes) -> bytes:
    samples = np.frombuffer(pcm_data, dtype=np.int16)
    stereo = np.empty(len(samples) * 2, dtype=np.int16)
    stereo[0::2] = samples
    stereo[1::2] = samples
    return stereo.tobytes()


class PCMStreamSource(discord.AudioSource):
    """Plays raw PCM data (48kHz stereo int16) into Discord voice."""

    def __init__(self):
        self._buffer = bytearray()
        self._finished = False

    def feed(self, pcm_48k_stereo: bytes):
        self._buffer.extend(pcm_48k_stereo)

    def finish(self):
        self._finished = True

    def read(self) -> bytes:
        frame_size = 3840  # 20ms at 48kHz stereo 16-bit
        if len(self._buffer) >= frame_size:
            frame = bytes(self._buffer[:frame_size])
            del self._buffer[:frame_size]
            return frame
        if self._finished:
            return b""
        return b"\x00" * frame_size

    def is_opus(self) -> bool:
        return False
