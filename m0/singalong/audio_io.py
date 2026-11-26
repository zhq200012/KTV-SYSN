"""音频 I/O：读写 WAV、单声道化、重采样。采样统一 16k / 单声道 / float32 [-1,1]。"""
from __future__ import annotations

import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly

_SAMPLE_WIDTH_DTYPE = {1: np.int8, 2: np.int16, 4: np.int32}


def read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV -> (samples float32 [-1,1] 单声道, sample_rate)。"""
    try:
        with wave.open(path, "rb") as w:
            sr = w.getframerate()
            nch = w.getnchannels()
            sw = w.getsampwidth()
            raw = w.readframes(w.getnframes())
    except (wave.Error, EOFError, OSError) as e:
        raise ValueError(f"无法读取音频 {path}: {e}") from e

    if sw not in _SAMPLE_WIDTH_DTYPE:
        raise ValueError(f"不支持的采样位宽 {sw} 字节: {path}")
    samples = np.frombuffer(raw, dtype=_SAMPLE_WIDTH_DTYPE[sw]).astype(np.float32)
    samples /= float(1 << (8 * sw - 1))
    if nch > 1:
        samples = samples.reshape(-1, nch).mean(axis=1)
    return samples.astype(np.float32), sr


def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None:
    """float32 [-1,1] 单声道 -> 16-bit PCM WAV。"""
    samples = np.asarray(samples, dtype=np.float32)
    samples = np.clip(samples, -1.0, 1.0)
    pcm = (samples * 32767.0).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes(pcm.tobytes())


def to_mono(samples: np.ndarray) -> np.ndarray:
    """多声道 -> 单声道（均值）。"""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    return samples.astype(np.float32)


def resample(samples: np.ndarray, from_rate: int, to_rate: int = 16000) -> np.ndarray:
    """多相重采样到目标采样率。"""
    samples = np.asarray(samples, dtype=np.float32)
    if from_rate == to_rate:
        return samples
    g = gcd(int(from_rate), int(to_rate))
    up = int(to_rate) // g
    down = int(from_rate) // g
    return resample_poly(samples, up, down).astype(np.float32)


def load_audio(path: str, target_rate: int = 16000) -> np.ndarray:
    """读取并统一为 target_rate / 单声道 / float32 [-1,1]。"""
    samples, sr = read_wav(path)
    samples = to_mono(samples)
    return resample(samples, sr, target_rate)


def rms(samples: np.ndarray) -> float:
    """均方根（float32 输入）。"""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
