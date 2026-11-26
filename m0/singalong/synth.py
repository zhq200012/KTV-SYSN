"""合成样本生成（dev 工具）：伴奏(自制旋律) + 人声(独立旋律) + 噪声。

只生成合成素材，不读真实录音。合成歌曲为确定性正弦旋律 + 谐波，无版权问题。
每个音符的频率在 [F_MIN, F_MAX] 内按对数均匀随机取值（连续、非量化），使不同歌曲
的频谱指纹基本不重叠，避免跨曲误匹配；同一首歌的样本是其精确切片，可稳定命中。
两个正交参数：
- freq_scale：变调（旋律时程不变，所有频率 ×p），总时长不变。
- time_scale：变速（旋律时程压缩 s 倍，频率不变，即“同一音高唱/放得更快”），总时长 /s。
"""
from __future__ import annotations

import json
import os

import numpy as np

from .audio_io import write_wav

SAMPLE_RATE = 16000

# 各曲使用互不重叠的频带 + 非谐波分音（比率 <1.5），保证不同歌曲频谱指纹零碰撞。
# 带 k 的基频在 [BAND_BASE[k], BAND_BASE[k]*BAND_WIDTH]，分音最高 1.4x，
# 峰值上限 = BAND_BASE[k]*BAND_WIDTH*1.4 = 1.82*BAND_BASE[k] < 2*BAND_BASE[k]（下一带起点）。
BAND_BASE = [150.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0]
BAND_WIDTH = 1.3
PARTIALS = [(1.0, 1.0), (1.2, 0.5), (1.4, 0.35)]


def _note_schedule(rng: np.random.Generator, duration: float,
                   band_index: int) -> list[tuple[float, float, float]]:
    """确定性音符时程 -> [(ref_start, ref_dur, freq_hz), ...]，基频在 band_index 对应的带内。"""
    notes: list[tuple[float, float, float]] = []
    base = BAND_BASE[band_index % len(BAND_BASE)]
    lo = base
    hi = base * BAND_WIDTH
    t = 0.0
    while t < duration:
        d = float(rng.uniform(0.35, 0.65))
        if t + d > duration:
            d = duration - t
        freq = float(rng.uniform(lo, hi))
        notes.append((t, d, freq))
        t += d
    return notes


def _envelope(w: np.ndarray, wd: float, attack: float = 0.01, release: float = 0.06) -> np.ndarray:
    env = np.ones_like(w, dtype=np.float32)
    if wd > attack:
        env = np.minimum(env, (w / attack).astype(np.float32))
    if wd > release:
        env = np.minimum(env, ((wd - w) / release).astype(np.float32))
    return np.clip(env, 0.0, 1.0)


def generate_song(
    seed: int,
    ref_duration: float,
    freq_scale: float = 1.0,
    time_scale: float = 1.0,
    band_index: int = 0,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """生成一首合成“伴奏”：旋律（非谐波分音），时长 ref_duration/time_scale 秒。"""
    rng = np.random.default_rng(seed)
    notes = _note_schedule(rng, ref_duration, band_index)
    wall_dur = ref_duration / time_scale
    n = int(wall_dur * sr)
    signal = np.zeros(n, dtype=np.float32)
    t_wall = np.arange(n) / sr

    for rs, rd, f in notes:
        ws = rs / time_scale
        wd = rd / time_scale
        i0 = int(ws * sr)
        i1 = int((ws + wd) * sr)
        if i1 <= i0:
            continue
        i0 = max(0, i0)
        i1 = min(n, i1)
        w = t_wall[i0:i1] - ws
        env = _envelope(w, wd)
        base_f = f * freq_scale
        for ratio, amp in PARTIALS:
            signal[i0:i1] += (amp * np.sin(2 * np.pi * base_f * ratio * w) * env).astype(np.float32)

    peak = float(np.max(np.abs(signal))) + 1e-9
    signal = (signal / peak * 0.8).astype(np.float32)
    return signal


def build_timeline(song_id: str, title: str, ref_duration: float, line_interval: float = 4.0) -> dict:
    """生成合成时间轴 JSON（含 lines/chords/sections），供 timeline.py 渲染。"""
    lines = []
    t = line_interval
    i = 1
    chords_cycle = ["C", "Am", "F", "G"]
    while t < ref_duration:
        lines.append({
            "start": round(t, 3),
            "end": round(t + line_interval, 3),
            "text": f"跟唱测试歌词 · 第{i}行",
            "words": [],
            "chords": [
                {"t": round(t, 3), "name": chords_cycle[(i - 1) % len(chords_cycle)]},
            ],
        })
        t += line_interval
        i += 1
    sections = [
        {"type": "intro", "start": 0.0, "end": line_interval},
        {"type": "verse", "start": line_interval, "end": ref_duration},
    ]
    return {
        "song_id": song_id,
        "title": title,
        "reference_version": "合成伴奏A",
        "lines": lines,
        "sections": sections,
        "bpm": None,
    }


def _fit(seg: np.ndarray, n: int) -> np.ndarray:
    """将切片裁齐/补零到长度 n。"""
    seg = np.asarray(seg, dtype=np.float32)
    if len(seg) >= n:
        return seg[:n]
    out = np.zeros(n, dtype=np.float32)
    out[: len(seg)] = seg
    return out


def build_sample(
    seed: int,
    ref_duration: float,
    offset_sec: float,
    duration_sec: float,
    band_index: int = 0,
    snr_db: float = 20.0,
    vocal_gain_db: float = -6.0,
    vocal_offset: float = 0.0,
    freq_scale: float = 1.0,
    time_scale: float = 1.0,
) -> np.ndarray:
    """生成一段合成样本：performance 切片 + 人声(独立旋律) + 高斯噪声。

    offset_sec 为参考时间（真值）；performance 切片按 wall_time = offset_sec/time_scale。
    """
    sr = SAMPLE_RATE
    inst = generate_song(seed, ref_duration, freq_scale, time_scale, band_index)
    vocal = generate_song(seed + 999983, ref_duration, freq_scale, time_scale, band_index)
    n = int(duration_sec * sr)

    def seg(sig: np.ndarray, off_ref: float) -> np.ndarray:
        w0 = off_ref / time_scale
        i0 = int(w0 * sr)
        i1 = int((w0 + duration_sec) * sr)
        return _fit(sig[i0:i1], n)

    inst_seg = seg(inst, offset_sec)
    vocal_seg = seg(vocal, offset_sec + vocal_offset)
    vocal_lin = 10 ** (vocal_gain_db / 20.0)
    sample = inst_seg + vocal_lin * vocal_seg

    rng = np.random.default_rng(seed ^ 0x5EED ^ int(offset_sec * 1000))
    signal_power = float(np.mean(sample.astype(np.float64) ** 2)) + 1e-12
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, float(np.sqrt(noise_power)), n).astype(np.float32)
    sample = sample + noise

    peak = float(np.max(np.abs(sample))) + 1e-9
    return (sample / peak * 0.8).astype(np.float32)


def _write_song(data_dir: str, song_id: str, title: str, seed: int, duration: float,
                band_index: int) -> None:
    sdir = os.path.join(data_dir, "songs", song_id)
    os.makedirs(sdir, exist_ok=True)
    samples = generate_song(seed, duration, band_index=band_index)
    write_wav(os.path.join(sdir, "reference.wav"), samples, SAMPLE_RATE)
    meta = {
        "song_id": song_id,
        "title": title,
        "language": "zh",
        "license": "self-made",
        "source": "synth",
        "audio_path": f"songs/{song_id}/reference.wav",
        "duration_sec": round(duration, 3),
        "timeline_path": f"songs/{song_id}/timeline.json",
    }
    with open(os.path.join(sdir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    with open(os.path.join(sdir, "timeline.json"), "w", encoding="utf-8") as f:
        json.dump(build_timeline(song_id, title, duration), f, ensure_ascii=False, indent=2)


def _write_sample_and_label(data_dir: str, sample_id: str, samples: np.ndarray, label: dict) -> None:
    spath = os.path.join(data_dir, "eval", "synth", f"{sample_id}.wav")
    lpath = os.path.join(data_dir, "eval", "labels", f"{sample_id}.json")
    os.makedirs(os.path.dirname(spath), exist_ok=True)
    os.makedirs(os.path.dirname(lpath), exist_ok=True)
    write_wav(spath, samples, SAMPLE_RATE)
    with open(lpath, "w", encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False, indent=2)


def generate_all(data_dir: str, seed: int = 20261126) -> list[dict]:
    """生成合成曲库 + 合成样本 + 真值标注，返回 eval manifest（samples 列表）。"""
    sr = SAMPLE_RATE
    duration = 40.0

    songs = [("S001", "合成歌曲一", seed + 1, 0), ("S002", "合成歌曲二", seed + 2, 1),
             ("S003", "合成歌曲三", seed + 3, 2), ("S004", "合成歌曲四", seed + 4, 3),
             ("S005", "合成歌曲五", seed + 5, 4)]
    seed_by_id = {sid: sd for sid, title, sd, bi in songs}
    band_by_id = {sid: bi for sid, title, sd, bi in songs}
    for sid, title, sd, bi in songs:
        _write_song(data_dir, sid, title, sd, duration, bi)

    samples: list[dict] = []

    def add(sample_id, wav, label):
        _write_sample_and_label(data_dir, sample_id, wav, label)
        samples.append(label)

    # --- 曲库内：识别 / 定位 / 同步（干净，含人声+噪声） ---
    clean_offsets = [2.0, 10.0, 20.0, 30.0]  # 有旋律段
    for sid, title, sd, bi in songs:
        for off in clean_offsets:
            sid_ = f"synth_{sid}_off{int(off)}"
            wav = build_sample(sd, duration, off, 8.0, band_index=band_by_id[sid], snr_db=20.0, vocal_gain_db=-6.0)
            add(sid_, wav, {
                "sample_id": sid_, "kind": "synth",
                "audio_path": f"eval/synth/{sid_}.wav",
                "song_id": sid, "start_offset_sec": off, "duration_sec": 8.0,
                "events": [], "freq_scale": 1.0, "time_scale": 1.0,
                "notes": "干净(SNR20/人声-6dB)",
            })

    # 起播位置 0%（合成无静音前奏，仅观察起播位置覆盖）
    wav = build_sample(seed + 1, duration, 0.0, 8.0, band_index=band_by_id["S001"], snr_db=20.0)
    add("synth_S001_start0", wav, {
        "sample_id": "synth_S001_start0", "kind": "synth",
        "audio_path": "eval/synth/synth_S001_start0.wav",
        "song_id": "S001", "start_offset_sec": 0.0, "duration_sec": 8.0,
        "events": [], "freq_scale": 1.0, "time_scale": 1.0,
        "notes": "起播位置0%（合成无静音前奏）",
    })

    # --- 整曲同步（无可见跳动） ---
    for sid in ["S001", "S002"]:
        sd = seed_by_id[sid]
        wav = build_sample(sd, duration, 2.0, duration - 4.0, band_index=band_by_id[sid], snr_db=25.0)
        add(f"synth_{sid}_full", wav, {
            "sample_id": f"synth_{sid}_full", "kind": "synth",
            "audio_path": f"eval/synth/synth_{sid}_full.wav",
            "song_id": sid, "start_offset_sec": 2.0, "duration_sec": duration - 4.0,
            "events": [], "freq_scale": 1.0, "time_scale": 1.0, "notes": "整曲回放",
        })

    # --- 变调 ±1 半音（R4 / 3.6） ---
    for semi in (-1, 1):
        p = 2 ** (semi / 12.0)
        wav = build_sample(seed + 1, duration, 10.0, 8.0, band_index=band_by_id["S001"], freq_scale=p)
        add(f"synth_S001_pitch{semi:+d}", wav, {
            "sample_id": f"synth_S001_pitch{semi:+d}", "kind": "synth",
            "audio_path": f"eval/synth/synth_S001_pitch{semi:+d}.wav",
            "song_id": "S001", "start_offset_sec": 10.0, "duration_sec": 8.0,
            "events": [], "freq_scale": p, "time_scale": 1.0, "notes": f"变调{semi:+d}半音",
        })

    # --- 变速 ±5%（R3 / 3.6） ---
    for ts in (0.95, 1.05):
        wav = build_sample(seed + 1, duration, 10.0, 8.0, band_index=band_by_id["S001"], time_scale=ts)
        add(f"synth_S001_speed{ts}", wav, {
            "sample_id": f"synth_S001_speed{ts}", "kind": "synth",
            "audio_path": f"eval/synth/synth_S001_speed{ts}.wav",
            "song_id": "S001", "start_offset_sec": 10.0, "duration_sec": 8.0,
            "events": [], "freq_scale": 1.0, "time_scale": ts, "notes": f"变速{ts}",
        })

    # --- 事件样本：暂停 / 重唱 / 切歌 ---
    for sid in ("S001", "S002", "S003"):
        sd = seed_by_id[sid]
        # 暂停：播 6s -> 静音 2s -> 再播 6s
        seg_a = build_sample(sd, duration, 10.0, 6.0, band_index=band_by_id[sid], snr_db=25.0)
        silence = np.zeros(int(2.0 * sr), dtype=np.float32)
        seg_b = build_sample(sd, duration, 16.0, 6.0, band_index=band_by_id[sid], snr_db=25.0)
        wav = np.concatenate([seg_a, silence, seg_b])
        add(f"synth_{sid}_pause", wav, {
            "sample_id": f"synth_{sid}_pause", "kind": "synth",
            "audio_path": f"eval/synth/synth_{sid}_pause.wav",
            "song_id": sid, "start_offset_sec": 10.0, "duration_sec": 14.0,
            "events": [{"t": 6.0, "type": "pause", "dur_sec": 2.0}],
            "freq_scale": 1.0, "time_scale": 1.0, "notes": "暂停 2s",
        })
        # 重唱：播 8s -> 回到开头播 6s
        seg_a = build_sample(sd, duration, 10.0, 8.0, band_index=band_by_id[sid], snr_db=25.0)
        seg_b = build_sample(sd, duration, 0.0, 6.0, band_index=band_by_id[sid], snr_db=25.0)
        wav = np.concatenate([seg_a, seg_b])
        add(f"synth_{sid}_replay", wav, {
            "sample_id": f"synth_{sid}_replay", "kind": "synth",
            "audio_path": f"eval/synth/synth_{sid}_replay.wav",
            "song_id": sid, "start_offset_sec": 10.0, "duration_sec": 14.0,
            "events": [{"t": 8.0, "type": "replay"}],
            "freq_scale": 1.0, "time_scale": 1.0, "notes": "重唱(回开头)",
        })
        # 失锁：播 6s -> 高能量无法匹配噪声 8s -> 再播 6s（LOST -> 恢复原曲）
        seg_a = build_sample(sd, duration, 10.0, 6.0, band_index=band_by_id[sid], snr_db=25.0)
        noise_rng = np.random.default_rng(sd ^ 0xC0FFEE)
        noise = noise_rng.normal(0.0, 0.2, int(8.0 * sr)).astype(np.float32)
        seg_b = build_sample(sd, duration, 16.0, 6.0, band_index=band_by_id[sid], snr_db=25.0)
        wav = np.concatenate([seg_a, noise, seg_b])
        add(f"synth_{sid}_lost", wav, {
            "sample_id": f"synth_{sid}_lost", "kind": "synth",
            "audio_path": f"eval/synth/synth_{sid}_lost.wav",
            "song_id": sid, "start_offset_sec": 10.0, "duration_sec": 20.0,
            "events": [{"t": 6.0, "type": "lost", "dur_sec": 8.0}],
            "freq_scale": 1.0, "time_scale": 1.0, "notes": "失锁：高能量噪声 8s 后恢复",
        })

    # 切歌：S001 播 8s -> S002 从开头播 6s
    seg_a = build_sample(seed + 1, duration, 10.0, 8.0, band_index=band_by_id["S001"], snr_db=25.0)
    seg_b = build_sample(seed + 2, duration, 0.0, 6.0, band_index=band_by_id["S002"], snr_db=25.0)
    wav = np.concatenate([seg_a, seg_b])
    add("synth_switch_S001_S002", wav, {
        "sample_id": "synth_switch_S001_S002", "kind": "synth",
        "audio_path": "eval/synth/synth_switch_S001_S002.wav",
        "song_id": "S001", "start_offset_sec": 10.0, "duration_sec": 14.0,
        "events": [{"t": 8.0, "type": "switch", "song_id": "S002", "start_offset_sec": 0.0}],
        "freq_scale": 1.0, "time_scale": 1.0, "notes": "切歌 S001->S002",
    })

    # --- 曲库外（拒识） ---
    out_seed = seed + 9000
    _write_song(data_dir, "OUT1", "合成曲库外", out_seed, duration, 5)
    for k in range(2):
        wav = build_sample(out_seed, duration, 10.0 + k * 5, 8.0, band_index=5, snr_db=20.0)
        add(f"synth_OUT1_{k}", wav, {
            "sample_id": f"synth_OUT1_{k}", "kind": "synth",
            "audio_path": f"eval/synth/synth_OUT1_{k}.wav",
            "song_id": None, "start_offset_sec": None, "duration_sec": 8.0,
            "events": [], "freq_scale": 1.0, "time_scale": 1.0, "notes": "曲库外",
        })

    return samples
