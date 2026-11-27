"""歌词源文件（LRC / SRT）解析 -> 行级时间轴。M0 只做行级；词级强制对齐留 P1。"""
from __future__ import annotations

import re


def _parse_ts(text: str) -> float | None:
    """解析 mm:ss.xx 或 mm:ss,xx 为秒。"""
    m = re.match(r"^\s*(\d+):(\d{1,2})(?:[.,](\d{1,3}))?\s*$", text)
    if not m:
        return None
    mm = int(m.group(1))
    ss = int(m.group(2))
    frac = m.group(3) or "0"
    frac = float("0." + frac.ljust(3, "0"))
    return mm * 60 + ss + frac


_METADATA_PREFIXES = (
    "词", "曲", "作词", "作曲", "编曲", "演唱", "歌手", "专辑",
    "制作人", "制作", "监制", "发行", "混音", "录音", "和声",
    "吉他", "贝斯", "鼓", "键盘", "弦乐", "歌词",
)


def _is_lrc_metadata(text: str, seen_lyric: bool) -> bool:
    """判断 LRC 行是否为元数据（标题 / 词曲作者等），而非歌词。"""
    for p in _METADATA_PREFIXES:
        if text.startswith(p) and (len(text) == len(p) or text[len(p)] in "：: "):
            return True
    # 首行 "标题 - 演唱者"（出现在任何歌词之前）
    if not seen_lyric and " - " in text:
        return True
    return False


def parse_lrc(content: str) -> list[tuple[float, str]]:
    """LRC -> [(start_sec, text), ...]，按时间升序。跳过标题/词曲等元数据行。"""
    lines: list[tuple[float, str]] = []
    tag_re = re.compile(r"\[(\d{1,2}:\d{1,2}(?:[.,]\d{1,3})?)\]")
    seen_lyric = False
    for raw in content.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        tags = list(tag_re.finditer(raw))
        if not tags:
            continue
        text = raw[tags[-1].end():].strip()
        if not text:
            continue
        if _is_lrc_metadata(text, seen_lyric):
            continue
        seen_lyric = True
        for tag in tags:
            t = _parse_ts(tag.group(1))
            if t is not None:
                lines.append((t, text))
    lines.sort(key=lambda x: x[0])
    return lines


def parse_srt(content: str) -> list[tuple[float, str]]:
    """SRT -> [(start_sec, text), ...]，按时间升序。"""
    lines: list[tuple[float, str]] = []
    blocks = re.split(r"\n\s*\n", content.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        m = re.search(
            r"(\d{1,2}:\d{1,2}:\d{1,2}[,.]\d{1,3})\s*-->\s*(\d{1,2}:\d{1,2}:\d{1,2}[,.]\d{1,3})",
            block,
        )
        if not m:
            continue
        text = block[m.end():].strip().replace("\n", " ")
        if not text:
            continue
        t = _parse_hms(m.group(1))
        if t is not None:
            lines.append((t, text))
    lines.sort(key=lambda x: x[0])
    return lines


def _parse_hms(text: str) -> float | None:
    m = re.match(r"^(\d{1,2}):(\d{1,2}):(\d{1,2})[,.](\d{1,3})$", text)
    if not m:
        return None
    hh, mm, ss, frac = (int(m.group(1)), int(m.group(2)), int(m.group(3)), m.group(4))
    return hh * 3600 + mm * 60 + ss + float("0." + frac.ljust(3, "0"))


def lines_to_timeline(song_id: str, title: str, lines: list[tuple[float, str]],
                      duration_sec: float) -> dict:
    """行级歌词 -> timeline.json（words/chords 空，sections 简单划分）。"""
    out = []
    for i, (t, text) in enumerate(lines):
        end = lines[i + 1][0] if i + 1 < len(lines) else duration_sec
        out.append({"start": round(t, 3), "end": round(end, 3), "text": text,
                    "words": [], "chords": []})
    first = out[0]["start"] if out else 0.0
    sections = [{"type": "intro", "start": 0.0, "end": first}] if first > 0 else []
    sections.append({"type": "verse", "start": first, "end": duration_sec})
    return {
        "song_id": song_id,
        "title": title,
        "reference_version": "",
        "lines": out,
        "sections": sections,
        "bpm": None,
    }
