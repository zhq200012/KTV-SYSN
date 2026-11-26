"""全局配置：指纹 / 匹配 / 同步 / 评测的全部可调参数（默认值）。"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # 采样
    sample_rate: int = 16000

    # 指纹 FFT
    fft_size: int = 4096
    hop: int = 512
    window: str = "hann"

    # 峰值与 landmark 配对
    peak_neighborhood: int = 20      # 频域局部极大窗口（±bin）
    peak_amp_min: float = 10.0       # 幅度下限（绝对谱幅度）
    fan_value: int = 15              # 每个峰值向后配对的峰值数
    target_zone: tuple[int, int] = (1, 15)  # Δt 帧数范围 [min, max]（含）

    # 匹配与置信
    confidence_threshold: float = 0.10   # aligned/query 下限
    min_align: int = 5                   # 最少对齐指纹数
    uniqueness_margin: float = 1.5       # top 与 second 得票边际

    # 同步引擎
    match_window: float = 2.0            # 匹配滚动窗口（秒）
    realign_interval: float = 2.0        # 重对齐周期（秒）
    rate_min_span: float = 10.0          # rate 估计最小测量间隔（秒）
    rate_min: float = 0.85
    rate_max: float = 1.15
    rate_ema: float = 0.10
    correct_small: float = 0.10          # <100ms 不调整
    correct_large: float = 1.0           # >1s 跳转
    slew_duration: float = 1.5           # 渐进校正时长（秒）
    low_energy_rms: float = 0.005        # 判定“音频能量低”的 RMS 阈值
    pause_min_energy_sec: float = 1.0    # 连续低能量触发暂停的时长（秒）
    lost_fail_streak: int = 3            # 连续匹配失败触发失锁的次数（非低能量）
    replay_max_pos: float = 20.0         # 重唱判定：offset 回退到该位置以内视为开头附近

    # 时间轴渲染
    chord_lookahead: float = 0.5         # 和弦提前量（秒）
    countdown_lead: float = 3.0          # 进唱倒数提前量（秒）

    @property
    def offset_bin(self) -> float:
        """offset 直方图 bin 宽（秒）。1 帧 = hop/sample_rate，≤50ms（RISKS R2）。"""
        return self.hop / self.sample_rate

    @property
    def freq_bin_hz(self) -> float:
        """频率量化到 FFT bin 的宽度（Hz）= sample_rate/fft_size ≈ 3.9Hz。"""
        return self.sample_rate / self.fft_size

    @property
    def match_window_samples(self) -> int:
        return int(round(self.match_window * self.sample_rate))


DEFAULT_CONFIG = Config()
