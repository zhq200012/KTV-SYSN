"""指纹匹配：曲库索引匹配 + offset 直方图投票取峰 + 置信度。

输出 song_id + offset + confidence；低置信明确返回 None（未识别，不猜测）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import DEFAULT_CONFIG, Config

# hash -> list[(song_id, t_ref_sec)]
Index = dict[int, list[tuple[str, float]]]


@dataclass
class MatchResult:
    song_id: str
    offset: float          # 参考音频中，查询窗口起点的秒数
    confidence: float      # = aligned_count / query_hashes
    aligned_count: int
    query_hashes: int
    second_count: int      # 得票第二候选的 aligned 数


class Matcher:
    def __init__(self, index: Index, cfg: Config = DEFAULT_CONFIG):
        self.index = index
        self.cfg = cfg

    def match(self, query_hashes: list[tuple[int, float]]) -> MatchResult | None:
        if not query_hashes:
            return None

        # 逐 hash 投票：offset = t_ref - t_query（= 查询窗口起点对应的参考秒数）
        votes: dict[str, dict[int, int]] = {}
        for h, tq in query_hashes:
            entries = self.index.get(h)
            if not entries:
                continue
            for song_id, tr in entries:
                offset = tr - tq
                b = int(round(offset / self.cfg.offset_bin))
                d = votes.setdefault(song_id, {})
                d[b] = d.get(b, 0) + 1

        best_song: str | None = None
        best_bin = 0
        best_count = 0
        second_count = 0
        for song_id, d in votes.items():
            if not d:
                continue
            b, c = max(d.items(), key=lambda kv: kv[1])
            if c > best_count:
                second_count = best_count
                best_song, best_bin, best_count = song_id, b, c
            elif c > second_count:
                second_count = c

        if best_song is None or best_count == 0:
            return None

        offset = self._refine_offset(votes[best_song], best_bin)
        confidence = best_count / len(query_hashes)

        # 低置信 / 唯一性不足 -> 未识别
        if best_count < self.cfg.min_align:
            return None
        if confidence < self.cfg.confidence_threshold:
            return None
        if best_count < self.cfg.uniqueness_margin * second_count:
            return None

        return MatchResult(
            song_id=best_song,
            offset=offset,
            confidence=confidence,
            aligned_count=best_count,
            query_hashes=len(query_hashes),
            second_count=second_count,
        )

    def _refine_offset(self, d: dict[int, int], b: int) -> float:
        """峰位抛物线插值细化（RISKS R2）。"""
        y0 = d.get(b - 1, 0)
        y1 = d.get(b, 0)
        y2 = d.get(b + 1, 0)
        denom = y0 - 2 * y1 + y2
        delta = 0.0
        if denom != 0:
            delta = 0.5 * (y0 - y2) / denom
            delta = max(-1.0, min(1.0, delta))
        return (b + delta) * self.cfg.offset_bin
