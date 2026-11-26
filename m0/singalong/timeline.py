"""时间轴：解析 4.4 节时间轴 JSON + 终端渲染（歌词 / 和弦 / 进唱倒数）。

M0 解析时保留 words / chords / sections / bpm 字段供 P1 使用，不丢弃。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .config import DEFAULT_CONFIG, Config


@dataclass
class Word:
    t: float
    text: str


@dataclass
class Chord:
    t: float
    name: str


@dataclass
class Line:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    chords: list[Chord] = field(default_factory=list)


@dataclass
class Section:
    type: str
    start: float
    end: float


@dataclass
class Timeline:
    song_id: str
    title: str
    reference_version: str
    lines: list[Line]
    sections: list[Section]
    bpm: float | None = None


def load_timeline(path: str) -> Timeline:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lines = []
    for ln in data.get("lines", []):
        words = [Word(w["t"], w["text"]) for w in ln.get("words", [])]
        chords = [Chord(c["t"], c["name"]) for c in ln.get("chords", [])]
        lines.append(Line(ln["start"], ln["end"], ln.get("text", ""), words, chords))
    sections = [Section(s["type"], s["start"], s["end"]) for s in data.get("sections", [])]
    return Timeline(
        song_id=data.get("song_id", ""),
        title=data.get("title", ""),
        reference_version=data.get("reference_version", ""),
        lines=lines,
        sections=sections,
        bpm=data.get("bpm"),
    )


def current_line(tl: Timeline, position: float) -> Line | None:
    """返回 position 秒处的当前行（start <= position < end）。"""
    for ln in tl.lines:
        if ln.start <= position < ln.end:
            return ln
    return None


def next_line(tl: Timeline, position: float) -> Line | None:
    """返回 position 之后（含当前即将开始）的下一行。"""
    for ln in tl.lines:
        if ln.start >= position:
            return ln
    return None


def render_line(line: Line, position: float) -> str:
    """P1 接口：M0 返回整行高亮（逐字分段留 P1）。"""
    return line.text


# ---------------------------------------------------------------------------
# 终端渲染（UI_SPEC.md）
# ---------------------------------------------------------------------------

_ANSI = {
    "dim_black": "\x1b[2;90m",
    "dim_white": "\x1b[2;37m",
    "current": "\x1b[1;93m",
    "chord_cur": "\x1b[92m",
    "chord_next": "\x1b[96m",
    "countdown_last": "\x1b[1;91m",
    "st_recognizing": "\x1b[33m",
    "st_playing": "\x1b[32m",
    "st_unknown": "\x1b[31m",
    "st_paused": "\x1b[90m",
    "reset": "\x1b[0m",
    "clear": "\x1b[2J\x1b[H",
}


def _display_width(text: str) -> int:
    """按显示列（CJK=2，ASCII=1）计算宽度。"""
    w = 0
    for ch in text:
        o = ord(ch)
        if 0x1100 <= o <= 0x115F or 0x2E80 <= o <= 0xA4CF or 0xAC00 <= o <= 0xD7A3 or 0xF900 <= o <= 0xFAFF or 0xFE30 <= o <= 0xFE4F or 0xFF00 <= o <= 0xFF60 or 0xFFE0 <= o <= 0xFFE6:
            w += 2
        elif o >= 0x20000:
            w += 2
        else:
            w += 1
    return w


def _truncate(text: str, width: int) -> str:
    if _display_width(text) <= width:
        return text
    out = ""
    for ch in text:
        if _display_width(out + ch) > width - 1:
            return out + "…"
        out += ch
    return out + "…"


def _fmt_time(sec: float) -> str:
    if sec < 0:
        sec = 0.0
    m = int(sec) // 60
    s = sec - m * 60
    return f"{m}:{s:04.1f}"


def _countdown_at(tl: Timeline, position: float, cfg: Config) -> str | None:
    """前奏/间奏进唱倒数：段落 end 前 countdown_lead 秒内显示 3→2→1→进。"""
    for sec in tl.sections:
        if sec.type in ("intro", "interlude"):
            remaining = sec.end - position
            if 0.0 < remaining <= cfg.countdown_lead:
                n = int(remaining) + 1
                return f"{n} · "
    return None


def _chord_at(line: Line | None, position: float, cfg: Config) -> str | None:
    """和弦行：当前和弦 → 下一和弦（提前 lookahead）+ 剩余秒数。"""
    if line is None:
        return None
    chords = line.chords
    if not chords:
        return None
    cur = None
    nxt = None
    for c in chords:
        if c.t <= position:
            cur = c
        elif nxt is None:
            nxt = c
    if cur is None and nxt is not None:
        cur, nxt = None, nxt
    if cur is None:
        return None
    if nxt is not None:
        remain = nxt.t - position
        return f"{cur.name} ──▶ {nxt.name}  ({remain:.1f}s)"
    return f"{cur.name}"


def render_screen(
    tl: Timeline,
    position: float,
    state_label: str,
    total_duration: float,
    hint: str = "",
    width: int = 80,
    color: bool = True,
    cfg: Config = DEFAULT_CONFIG,
) -> str:
    """按 UI_SPEC 渲染整屏（5 个分区）。返回字符串，供终端打印或评测快照。"""
    idx = -1
    for i, ln in enumerate(tl.lines):
        if ln.start <= position < ln.end:
            idx = i
            break
    prev_line = tl.lines[idx - 1] if idx - 1 >= 0 else None
    cur_line = tl.lines[idx] if idx >= 0 else None
    next_ln = tl.lines[idx + 1] if 0 <= idx < len(tl.lines) - 1 else None

    def sty(text: str, key: str) -> str:
        if not color:
            return text
        return _ANSI[key] + text + _ANSI["reset"]

    def render_lyric(ln: Line | None, kind: str) -> str:
        if ln is None:
            return " " * (width - 1)
        prefix = "▶ " if kind == "current" else "  "
        text = prefix + _truncate(ln.text, width - _display_width(prefix))
        if kind == "current":
            text = sty(text, "current")
        elif kind == "next":
            text = sty(text, "dim_white")
        else:
            text = sty(text, "dim_black")
        pad = width - 1 - _display_width(_strip_ansi(text))
        return text + " " * max(0, pad)

    # 分区 ① 状态行
    status = f"{state_label} · {tl.title} · {_fmt_time(position)} / {_fmt_time(total_duration)}"
    st_key = "st_playing"
    if "识别中" in state_label:
        st_key = "st_recognizing"
    elif "未识别" in state_label or "失锁" in state_label or "失去" in state_label:
        st_key = "st_unknown"
    elif "暂停" in state_label:
        st_key = "st_paused"
    line1 = sty(_truncate(status, width - 1), st_key)

    # 分区 ② 歌词窗口
    line2 = render_lyric(prev_line, "prev")
    line3 = render_lyric(cur_line, "current")
    line4 = render_lyric(next_ln, "next")

    # 分区 ③ 和弦行
    chord = _chord_at(cur_line, position, cfg)
    line5 = "" if chord is None else sty("和弦 " + _truncate(chord, width - 5), "chord_next") \
        if "──▶" in chord else sty("和弦 " + chord, "chord_cur")

    # 分区 ④ 倒数行
    cd = _countdown_at(tl, position, cfg)
    line6 = ""
    if cd is not None:
        if cd.startswith("1"):
            line6 = sty("倒数 " + cd + "进", "countdown_last")
        else:
            line6 = "倒数 " + cd + "进"

    # 分区 ⑤ 提示行
    line7 = _truncate(hint, width - 1)

    sep = "─" * (width - 1)
    return "\n".join([line1, sep, line2, line3, line4, sep, line5, line6, sep, line7])


def _strip_ansi(text: str) -> str:
    import re

    return re.sub(r"\x1b\[[0-9;]*m", "", text)
