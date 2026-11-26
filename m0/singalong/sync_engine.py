"""持续同步引擎：单调时钟推进 + 周期重对齐 + rate 估计 + 三级校正 + 状态机 + 瞬时事件。

captured_at 语义（DESIGN 3.3 / RISKS R7）：
- 实时麦克风用 wall clock（WallClock.now() = monotonic）。
- 文件回放用文件内时间戳（FileClock，随 chunk advance）。
- 补偿：position_now = position_capture + (applied_at - captured_at) * rate。
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
        self._truth: tuple[float, float] | None = None   # (ref_at_capture, captured_at)
        self._disp: tuple[float, float] | None = None
        self._rate = 1.0
        self._slew: _Slew | None = None
        self._frozen_pos: float | None = None

        self._last_realign = float("-inf")
        self._fail_streak = 0
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

        self._buffer = np.concatenate([self._buffer, samples])
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
            if res is not None:
                self._fail_streak = 0
                events.extend(self._on_match(res, captured_at))
                self._last_realign = captured_at
            else:
                self._fail_streak += 1
        elif low_energy:
            self._fail_streak += 1
        else:
            self._fail_streak = 0

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
            self._state = SyncState.PLAYING
            self._frozen_pos = None
            events.append(Event(EventType.SWITCH, res.song_id, ref_at_capture, captured_at,
                                f"from={old} confidence={res.confidence:.3f}"))
            return events

        # 恢复事件（暂停/失锁 -> 播放中）
        if prev_state in (SyncState.PAUSED, SyncState.LOST):
            self._state = SyncState.PLAYING
            self._frozen_pos = None
            events.append(Event(EventType.RESUME, self._current_song, ref_at_capture, captured_at,
                                f"from={prev_state.value}"))

        # 同一首歌：rate 估计
        if self._truth is not None:
            span = captured_at - self._truth[1]
            if span >= self.cfg.rate_min_span:
                raw = (ref_at_capture - self._truth[0]) / span
                if self.cfg.rate_min <= raw <= self.cfg.rate_max:
                    self._rate = self._rate * (1 - self.cfg.rate_ema) + raw * self.cfg.rate_ema
        self._truth = (ref_at_capture, captured_at)

        expected = self.position_at(captured_at).position
        deviation = ref_at_capture - expected

        # 重唱：offset 回退到开头附近
        if deviation < -self.cfg.correct_large and ref_at_capture <= self.cfg.replay_max_pos:
            self._rate = 1.0
            self._set_anchor(ref_at_capture, captured_at)
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
        self._truth = (ref, captured_at)
        self._disp = (ref, captured_at)
        self._slew = None

    def _update_state(self, captured_at: float) -> list[Event]:
        events: list[Event] = []
        low_energy = self._last_energy < self.cfg.low_energy_rms

        if self._state == SyncState.PLAYING:
            if low_energy and self._fail_streak >= self.cfg.pause_fail_streak:
                self._state = SyncState.PAUSED
                self._frozen_pos = self.position_at(captured_at).position
                events.append(Event(EventType.PAUSED, self._current_song, self._frozen_pos,
                                    captured_at, "低能量且匹配持续失败"))
            elif (not low_energy) and self._fail_streak >= self.cfg.lost_fail_streak:
                self._state = SyncState.LOST
                self._frozen_pos = self.position_at(captured_at).position
                events.append(Event(EventType.LOST, self._current_song, None, captured_at,
                                    "匹配持续失败但非暂停"))
        return events
