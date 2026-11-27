"""持续同步引擎：单调时钟推进 + 周期重对齐 + rate 估计 + 三级校正 + 状态机 + 瞬时事件。

captured_at 语义（DESIGN 3.3 / RISKS R7）：
- 实时麦克风用 wall clock（WallClock.now() = monotonic）。
- 文件回放用文件内时间戳（FileClock，随 chunk advance）。
- 补偿：position_now = position_capture + (applied_at - captured_at) * rate。

rate 估计（DESIGN 3.3）：rate 测量锚点与位置锚点分离——`_rate_ref/_rate_cap` 只在
「累计间隔 ≥ rate_min_span 且 raw 通过 0.85~1.15 过滤」时才计算并重置，不随每次重对齐
重置，避免 span 恒等于重对齐周期导致 rate 恒为 1.0。
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

import numpy as np

from . import fingerprint
from .audio_io import rms
from .config import DEFAULT_CONFIG, Config
from .matcher import MatchResult, Matcher


class Clock:
    def now(self) -> float:  # pragma: no cover - abstract
        raise NotImplementedError


class WallClock(Clock):
    def __init__(self) -> None:
        self._t0 = time.monotonic()

    def now(self) -> float:
        return time.monotonic() - self._t0


class FileClock(Clock):
    """文件回放时钟：由回放器随 chunk 推进，now() 返回文件内累计时间。"""

    def __init__(self) -> None:
        self._t = 0.0

    def advance(self, dt: float) -> None:
        self._t += dt

    def now(self) -> float:
        return self._t


class SyncState(Enum):
    RECOGNIZING = "recognizing"
    PLAYING = "playing"
    PAUSED = "paused"
    LOST = "lost"


class EventType(Enum):
    RECOGNIZED = "recognized"
    SWITCH = "switch"
    REPLAY = "replay"
    RESUME = "resume"
    PAUSED = "paused"
    LOST = "lost"


@dataclass
class Event:
    type: EventType
    song_id: str | None
    position_sec: float | None
    captured_at: float
    detail: str = ""


@dataclass
class Position:
    song_id: str | None
    position: float       # 参考音频播放进度（秒）；未识别为 -1
    rate: float
    captured_at: float
    applied_at: float


@dataclass
class _Slew:
    from_ref: float
    to_ref: float
    start: float
    duration: float


class SyncEngine:
    def __init__(self, matcher: Matcher, clock: Clock, cfg: Config = DEFAULT_CONFIG):
        self.matcher = matcher
        self.clock = clock
        self.cfg = cfg

        self._state = SyncState.RECOGNIZING
        self._buffer = np.zeros(0, dtype=np.float32)

        self._current_song: str | None = None
        self._disp: tuple[float, float] | None = None     # 显示/位置锚点 (ref_at_capture, captured_at)
        self._rate = 1.0
        self._rate_ref: float | None = None               # rate 测量锚点（独立于位置锚点）
        self._rate_cap: float | None = None
        self._slew: _Slew | None = None
        self._frozen_pos: float | None = None

        self._last_realign = float("-inf")
        self._fail_streak = 0                             # 连续失败的匹配尝试次数
        self._low_energy_sec = 0.0                        # 连续低能量时长（秒）
        self._last_energy = 0.0

    # -- 查询接口 ----------------------------------------------------------
    @property
    def state(self) -> SyncState:
        return self._state

    @property
    def current_song(self) -> str | None:
        return self._current_song

    @property
    def rate(self) -> float:
        return self._rate

    def position_at(self, now: float) -> Position:
        if self._disp is None:
            return Position(self._current_song, -1.0, self._rate, now, now)

        if self._state in (SyncState.PAUSED, SyncState.LOST):
            pos = self._frozen_pos if self._frozen_pos is not None else self._disp[0]
            return Position(self._current_song, pos, 1.0, self._disp[1], now)

        ref, cap = self._disp
        pos = ref + (now - cap) * self._rate
        if self._slew is not None:
            t = now - self._slew.start
            if 0.0 <= t < self._slew.duration:
                frac = t / self._slew.duration
                frac = frac * frac * (3.0 - 2.0 * frac)  # smoothstep
                pos_from = self._slew.from_ref + t * self._rate
                pos_to = self._slew.to_ref + t * self._rate
                pos = pos_from + (pos_to - pos_from) * frac
        return Position(self._current_song, pos, self._rate, cap, now)

    # -- 数据入口 ----------------------------------------------------------
    def push(self, samples: np.ndarray, captured_at: float) -> list[Event]:
        """喂入一段已重采样的音频（16k 单声道），captured_at = 该段采集结束时刻。"""
        samples = np.asarray(samples, dtype=np.float32)
        events: list[Event] = []
        self._last_energy = rms(samples)
        chunk_dur = len(samples) / self.cfg.sample_rate

        self._buffer = np.concatenate([self._buffer, samples])
        # 仅「识别中」允许窗口增长到 match_window_max，用更多上下文消歧重复段；
        # 失锁/播放/暂停保持短窗口：失锁恢复要能尽快从噪声中锁定回真实音频，
        # 播放中短窗口保证重唱/切歌检测的低延迟。
        if self._state == SyncState.RECOGNIZING:
            max_len = self.cfg.match_window_max_samples + self.cfg.hop * 4
        else:
            max_len = self.cfg.match_window_samples + self.cfg.hop * 4
        if len(self._buffer) > max_len:
            self._buffer = self._buffer[-max_len:]

        window_samples = self.cfg.match_window_samples
        low_energy = self._last_energy < self.cfg.low_energy_rms

        should_match = False
        if self._state in (SyncState.RECOGNIZING, SyncState.LOST):
            should_match = len(self._buffer) >= window_samples
        else:  # PLAYING / PAUSED
            should_match = (captured_at - self._last_realign) >= self.cfg.realign_interval

        if should_match and len(self._buffer) >= window_samples and not low_energy:
            res = self._try_match()
            if res is None:
                self._fail_streak += 1
            elif self._state == SyncState.RECOGNIZING and res.ambiguous:
                # 识别中歧义：等更多上下文（缓冲继续增长到 match_window_max），暂不识别
                pass
            else:
                self._fail_streak = 0
                events.extend(self._on_match(res, captured_at))
                self._last_realign = captured_at

        # 低能量时长追踪（用于暂停判定），音频恢复即清零
        if low_energy:
            self._low_energy_sec += chunk_dur
        else:
            self._low_energy_sec = 0.0

        events.extend(self._update_state(captured_at))
        return events

    def _try_match(self) -> MatchResult | None:
        hashes = fingerprint.fingerprint(self._buffer, self.cfg)
        return self.matcher.match(hashes)

    def _on_match(self, res: MatchResult, captured_at: float) -> list[Event]:
        events: list[Event] = []
        prev_state = self._state
        ref_at_capture = res.offset + len(self._buffer) / self.cfg.sample_rate

        # 首次识别
        if self._current_song is None:
            self._current_song = res.song_id
            self._rate = 1.0
            self._set_anchor(ref_at_capture, captured_at)
            self._reset_rate_anchor(ref_at_capture, captured_at)
            self._state = SyncState.PLAYING
            self._frozen_pos = None
            events.append(Event(EventType.RECOGNIZED, res.song_id, ref_at_capture, captured_at,
                                f"confidence={res.confidence:.3f}"))
            return events

        # 切歌
        if res.song_id != self._current_song:
            old = self._current_song
            self._current_song = res.song_id
            self._rate = 1.0
            self._set_anchor(ref_at_capture, captured_at)
            self._reset_rate_anchor(ref_at_capture, captured_at)
            self._state = SyncState.PLAYING
            self._frozen_pos = None
            events.append(Event(EventType.SWITCH, res.song_id, ref_at_capture, captured_at,
                                f"from={old} confidence={res.confidence:.3f}"))
            return events

        # 恢复事件（暂停/失锁 -> 播放中）
        if prev_state in (SyncState.PAUSED, SyncState.LOST):
            self._state = SyncState.PLAYING
            self._frozen_pos = None
            self._reset_rate_anchor(ref_at_capture, captured_at)
            events.append(Event(EventType.RESUME, self._current_song, ref_at_capture, captured_at,
                                f"from={prev_state.value}"))

        # 重复段歧义：不据此更新 rate / 判重唱 / 校正位置，保持当前进度
        if res.ambiguous:
            return events

        # 同一首歌：rate 估计（rate 锚点独立，只在累计 ≥10s 时测量并重置）
        if self._rate_ref is not None:
            span = captured_at - self._rate_cap
            if span >= self.cfg.rate_min_span:
                raw = (ref_at_capture - self._rate_ref) / span
                if self.cfg.rate_min <= raw <= self.cfg.rate_max:
                    self._rate = self._rate * (1 - self.cfg.rate_ema) + raw * self.cfg.rate_ema
                self._rate_ref = ref_at_capture
                self._rate_cap = captured_at

        expected = self.position_at(captured_at).position
        deviation = ref_at_capture - expected

        # 重唱：offset 回退到开头附近
        if deviation < -self.cfg.correct_large and ref_at_capture <= self.cfg.replay_max_pos:
            self._rate = 1.0
            self._set_anchor(ref_at_capture, captured_at)
            self._reset_rate_anchor(ref_at_capture, captured_at)
            self._state = SyncState.PLAYING
            self._frozen_pos = None
            events.append(Event(EventType.REPLAY, self._current_song, ref_at_capture, captured_at,
                                f"back to {ref_at_capture:.2f}s"))
            return events

        if abs(deviation) < self.cfg.correct_small:
            pass  # <100ms 不调整
        elif abs(deviation) < self.cfg.correct_large:
            self._slew = _Slew(from_ref=expected, to_ref=ref_at_capture,
                               start=captured_at, duration=self.cfg.slew_duration)
            self._disp = (ref_at_capture, captured_at)
        else:
            self._slew = None
            self._disp = (ref_at_capture, captured_at)
        return events

    def _set_anchor(self, ref: float, captured_at: float) -> None:
        self._disp = (ref, captured_at)
        self._slew = None

    def _reset_rate_anchor(self, ref: float, captured_at: float) -> None:
        self._rate_ref = ref
        self._rate_cap = captured_at

    def _update_state(self, captured_at: float) -> list[Event]:
        events: list[Event] = []
        low_energy = self._last_energy < self.cfg.low_energy_rms

        if self._state == SyncState.PLAYING:
            if self._low_energy_sec >= self.cfg.pause_min_energy_sec:
                # 音频能量持续低 -> 暂停（进度冻结）
                frozen = self.position_at(captured_at).position  # 先取当前进度，再切状态
                self._state = SyncState.PAUSED
                self._frozen_pos = frozen
                events.append(Event(EventType.PAUSED, self._current_song, self._frozen_pos,
                                    captured_at, "低能量持续"))
            elif (not low_energy) and self._fail_streak >= self.cfg.lost_fail_streak:
                # 匹配持续失败但非低能量 -> 失锁（后台重试识别）
                frozen = self.position_at(captured_at).position  # 先取当前进度，再切状态
                self._state = SyncState.LOST
                self._frozen_pos = frozen
                events.append(Event(EventType.LOST, self._current_song, None, captured_at,
                                    "匹配持续失败但非暂停"))
        return events
