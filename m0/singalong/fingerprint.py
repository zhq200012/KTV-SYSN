"""指纹提取（识别与定位共用）：频谱峰值 + landmark 两两配对哈希。

Shazam 式：FFT -> 频域局部极大峰值 -> 每个峰值与 target zone 内后续峰值配对
-> 哈希 (f1_bucket, f2_bucket, dt_frames)。识别与定位是同一次匹配的两个输出。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from .config import DEFAULT_CONFIG, Config


def _window_array(cfg: Config) -> np.ndarray:
    if cfg.window == "hann":
        return np.hanning(cfg.fft_size).astype(np.float32)
    if cfg.window == "hamming":
        return np.hamming(cfg.fft_size).astype(np.float32)
    return np.ones(cfg.fft_size, dtype=np.float32)


def _spectrogram(samples: np.ndarray, cfg: Config) -> np.ndarray:
    """返回幅度谱 (n_frames, n_bins)。"""
    samples = np.asarray(samples, dtype=np.float32)
    frame_len = cfg.fft_size
    hop = cfg.hop
    if len(samples) < frame_len:
        return np.empty((0, cfg.fft_size // 2 + 1), dtype=np.float32)
    n_frames = 1 + (len(samples) - frame_len) // hop
    idx = np.arange(frame_len)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = samples[idx] * _window_array(cfg)
    return np.abs(np.fft.rfft(frames, n=cfg.fft_size, axis=1)).astype(np.float32)


def extract_peaks(samples: np.ndarray, cfg: Config = DEFAULT_CONFIG) -> list[tuple[float, float]]:
    """提取频谱峰值 -> [(t_sec, f_hz), ...]，按时间升序。"""
    spec = _spectrogram(samples, cfg)
    n_frames, n_bins = spec.shape
    if n_frames == 0:
        return []

    # 频域局部极大（邻域 ±peak_neighborhood），并超过幅度下限
    size = 2 * cfg.peak_neighborhood + 1
    maxf = maximum_filter(spec, size=(1, size), mode="constant")
    is_peak = (spec == maxf) & (spec > cfg.peak_amp_min)
    is_peak[:, : cfg.peak_neighborhood] = False
    is_peak[:, n_bins - cfg.peak_neighborhood:] = False

    freqs = np.fft.rfftfreq(cfg.fft_size, 1.0 / cfg.sample_rate)
    times = np.arange(n_frames) * (cfg.hop / cfg.sample_rate)

    peaks: list[tuple[float, float]] = []
    for i in range(n_frames):
        cols = np.nonzero(is_peak[i])[0]
        for b in cols:
            peaks.append((float(times[i]), float(freqs[b])))
    return peaks


def _hash_value(f1_bucket: int, f2_bucket: int, dt_frames: int) -> int:
    """将 (f1, f2, dt) 编码为 int：各占 12/12/8 bit（f 为 FFT bin 索引，≤2048；dt ≤255）。"""
    return (f1_bucket & 0xFFF) | ((f2_bucket & 0xFFF) << 12) | ((dt_frames & 0xFF) << 24)


def generate_hashes(
    peaks: list[tuple[float, float]], cfg: Config = DEFAULT_CONFIG
) -> list[tuple[int, float]]:
    """由峰值生成哈希 -> [(hash_value, t_sec), ...]，t_sec 为配对中第一个峰值的时间。

    频率量化到 FFT bin 索引（bin 宽 = sample_rate/fft_size ≈ 3.9Hz），
    与 dejavu 一致；bin 级量化使变调 ±1 半音（约 6%）即可移动 ≥2 bin，从而被哈希拒绝（RISKS R4）。
    """
    hashes: list[tuple[int, float]] = []
    n = len(peaks)
    hop_sec = cfg.hop / cfg.sample_rate
    bin_width = cfg.sample_rate / cfg.fft_size
    for i in range(n):
        t1, f1 = peaks[i]
        f1_bucket = int(round(f1 / bin_width))
        paired = 0
        for j in range(i + 1, n):
            t2, f2 = peaks[j]
            dt_frames = int(round((t2 - t1) / hop_sec))
            if dt_frames < cfg.target_zone[0]:
                continue
            if dt_frames > cfg.target_zone[1]:
                break  # peaks 按时间升序，超过 target zone 即可停止
            f2_bucket = int(round(f2 / bin_width))
            hashes.append((_hash_value(f1_bucket, f2_bucket, dt_frames), t1))
            paired += 1
            if paired >= cfg.fan_value:
                break
    return hashes


def fingerprint(samples: np.ndarray, cfg: Config = DEFAULT_CONFIG) -> list[tuple[int, float]]:
    """提取指纹 = extract_peaks + generate_hashes。"""
    return generate_hashes(extract_peaks(samples, cfg), cfg)
