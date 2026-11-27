"""索引存储：manifest.json 与 library.db 的读写、校验与指纹库重建。"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

from . import fingerprint
from .audio_io import load_audio
from .config import DEFAULT_CONFIG, Config
from .matcher import Index


@dataclass
class SongMeta:
    song_id: str
    title: str
    language: str
    license: str
    source: str
    audio_path: str            # 相对 data/ 的指纹参考音频路径
    duration_sec: float
    timeline_path: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "SongMeta":
        return cls(
            song_id=d["song_id"],
            title=d.get("title", d["song_id"]),
            language=d.get("language", ""),
            license=d.get("license", ""),
            source=d.get("source", ""),
            audio_path=d["audio_path"],
            duration_sec=float(d["duration_sec"]),
            timeline_path=d.get("timeline_path"),
        )

    def to_dict(self) -> dict:
        return asdict(self)


def load_manifest(path: str) -> list[SongMeta]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [SongMeta.from_dict(d) for d in data.get("songs", [])]


def save_manifest(path: str, metas: list[SongMeta]) -> None:
    payload = {"songs": [m.to_dict() for m in metas]}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_index(songs: list[SongMeta], data_root: str, cfg: Config = DEFAULT_CONFIG) -> Index:
    """对每首歌提取指纹并并入索引（不落盘，由 dataset-build 落盘）。"""
    index: Index = {}
    for meta in songs:
        audio_path = os.path.join(data_root, meta.audio_path)
        samples = load_audio(audio_path, cfg.sample_rate)
        hashes = fingerprint.fingerprint(samples, cfg)
        for h, t in hashes:
            index.setdefault(h, []).append((meta.song_id, t))
    return index


def save_index(path: str, index: Index) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in index.items()}, f)
    os.replace(tmp, path)


def load_index(path: str) -> Index:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): [(sid, float(t)) for sid, t in v] for k, v in data.items()}
