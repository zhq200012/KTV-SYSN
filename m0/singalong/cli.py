"""命令行入口：import / align / dataset-build / eval / synth / demo。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import synth
from .audio_io import load_audio, write_wav
from .config import DEFAULT_CONFIG, Config
from .eval_runner import render_report, run_eval
from .index_store import (
    SongMeta,
    build_index,
    load_index,
    load_manifest,
    save_index,
    save_manifest,
)
from .lyrics import lines_to_timeline, parse_lrc, parse_srt
from .matcher import Matcher


def _data_root(args: argparse.Namespace) -> str:
    root = getattr(args, "data_root", None) or os.environ.get("SINGALONG_DATA_ROOT") or "data"
    return root


def _next_song_id(data_root: str) -> str:
    manifest_path = os.path.join(data_root, "manifest.json")
    metas = load_manifest(manifest_path) if os.path.exists(manifest_path) else []
    n = len(metas) + 1
    return f"S{n:03d}"


def cmd_import(args: argparse.Namespace) -> int:
    root = _data_root(args)
    src = args.audio
    if not os.path.exists(src):
        print(f"错误：音频不存在 {src}", file=sys.stderr)
        return 1

    song_id = args.song_id or _next_song_id(root)
    sdir = os.path.join(root, "songs", song_id)
    os.makedirs(sdir, exist_ok=True)

    samples = load_audio(src, DEFAULT_CONFIG.sample_rate)
    duration = len(samples) / DEFAULT_CONFIG.sample_rate
    write_wav(os.path.join(sdir, "reference.wav"), samples, DEFAULT_CONFIG.sample_rate)

    timeline_path = None
    if args.lyrics:
        with open(args.lyrics, "r", encoding="utf-8") as f:
            content = f.read()
        if args.lyrics.lower().endswith(".srt"):
            lines = parse_srt(content)
        else:
            lines = parse_lrc(content)
        title = args.title or song_id
        tl = lines_to_timeline(song_id, title, lines, duration)
        timeline_path = f"songs/{song_id}/timeline.json"
        with open(os.path.join(sdir, "timeline.json"), "w", encoding="utf-8") as f:
            json.dump(tl, f, ensure_ascii=False, indent=2)
        print(f"歌词已解析：{len(lines)} 行")

    meta = SongMeta(
        song_id=song_id,
        title=args.title or song_id,
        language=args.language,
        license="user",
        source=os.path.basename(src),
        audio_path=f"songs/{song_id}/reference.wav",
        duration_sec=round(duration, 3),
        timeline_path=timeline_path,
    )
    manifest_path = os.path.join(root, "manifest.json")
    metas = load_manifest(manifest_path) if os.path.exists(manifest_path) else []
    metas = [m for m in metas if m.song_id != song_id]
    metas.append(meta)
    metas.sort(key=lambda m: m.song_id)
    save_manifest(manifest_path, metas)
    print(f"已导入 {song_id}（{meta.title}），时长 {duration:.2f}s。下一步：python -m singalong dataset-build")
    return 0


def cmd_align(args: argparse.Namespace) -> int:
    root = _data_root(args)
    if not os.path.exists(args.audio):
        print(f"错误：音频不存在 {args.audio}", file=sys.stderr)
        return 1
    with open(args.lyrics, "r", encoding="utf-8") as f:
        content = f.read()
    if args.lyrics.lower().endswith(".srt"):
        lines = parse_srt(content)
    else:
        lines = parse_lrc(content)

    samples = load_audio(args.audio, DEFAULT_CONFIG.sample_rate)
    duration = len(samples) / DEFAULT_CONFIG.sample_rate
    tl = lines_to_timeline("", "", lines, duration)
    out = args.out
    path = out if out.endswith(".json") else out + ".timeline.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tl, f, ensure_ascii=False, indent=2)
    print(f"已生成时间轴 {path}（{len(lines)} 行）。词级对齐留 P1。")
    return 0


def _validate_timeline(tl: dict, song_id: str, duration: float) -> list[str]:
    errs: list[str] = []
    lines = tl.get("lines", [])
    prev_end = None
    for i, ln in enumerate(lines):
        s, e = ln.get("start"), ln.get("end")
        if s is None or e is None or s > e:
            errs.append(f"{song_id} 行 {i} 起止无效")
        if e is not None and e > duration + 0.5:
            errs.append(f"{song_id} 行 {i} 超出音频时长")
        if prev_end is not None and s is not None and s < prev_end:
            errs.append(f"{song_id} 行 {i} 时序重叠")
        prev_end = e if e is not None else prev_end
    return errs


def cmd_dataset_build(args: argparse.Namespace) -> int:
    root = _data_root(args)
    cfg = DEFAULT_CONFIG
    manifest_path = os.path.join(root, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"错误：manifest 不存在 {manifest_path}，先 import 或 synth。", file=sys.stderr)
        return 1
    metas = load_manifest(manifest_path)
    if not metas:
        print("错误：曲库为空。", file=sys.stderr)
        return 1

    errs: list[str] = []
    valid: list[SongMeta] = []
    for meta in metas:
        audio_path = os.path.join(root, meta.audio_path)
        if not os.path.exists(audio_path):
            errs.append(f"{meta.song_id} 音频缺失 {meta.audio_path}")
            continue
        samples = load_audio(audio_path, cfg.sample_rate)
        dur = len(samples) / cfg.sample_rate
        if abs(dur - meta.duration_sec) > 1.0:
            meta.duration_sec = round(dur, 3)
        if meta.timeline_path:
            tp = os.path.join(root, meta.timeline_path)
            if os.path.exists(tp):
                with open(tp, "r", encoding="utf-8") as f:
                    tl = json.load(f)
                errs.extend(_validate_timeline(tl, meta.song_id, dur))
        valid.append(meta)

    if errs:
        for e in errs:
            print(f"校验失败：{e}", file=sys.stderr)
        return 1

    save_manifest(manifest_path, valid)
    index = build_index(valid, root, cfg)
    save_index(os.path.join(root, "library.db"), index)
    n_hashes = sum(len(v) for v in index.values())
    print(f"重建完成：{len(valid)} 首，指纹条目 {n_hashes}。")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    root = _data_root(args)
    cfg = DEFAULT_CONFIG
    manifest_path = os.path.join(root, "manifest.json")
    index_path = os.path.join(root, "library.db")
    eval_manifest = os.path.join(root, "eval", "manifest.json")

    if not os.path.exists(index_path):
        print("错误：缺少 library.db，先 dataset-build。", file=sys.stderr)
        return 1
    if not os.path.exists(eval_manifest):
        print(f"错误：缺少评测样本索引 {eval_manifest}，先 synth。", file=sys.stderr)
        return 1

    index = load_index(index_path)
    matcher = Matcher(index, cfg)
    with open(eval_manifest, "r", encoding="utf-8") as f:
        labels = json.load(f).get("samples", [])

    commit = _git_commit()
    report = run_eval(matcher, labels, root, cfg)
    md = render_report(report, cfg, commit)

    out = args.report or os.path.join("docs", "EVAL_REPORT.md")
    os.makedirs(os.path.dirname(out) if os.path.dirname(out) else ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    print(f"\n报告已写入 {out}")
    return 0


def cmd_synth(args: argparse.Namespace) -> int:
    root = _data_root(args)
    seed = args.seed
    os.makedirs(os.path.join(root, "eval", "labels"), exist_ok=True)
    samples = synth.generate_all(root, seed)

    # 曲库 manifest（含曲库外 OUT1）
    metas = []
    for sid in sorted(set(s["song_id"] for s in samples if s["song_id"])):
        mp = os.path.join(root, "songs", sid, "meta.json")
        with open(mp, "r", encoding="utf-8") as f:
            metas.append(json.load(f))
    save_manifest(os.path.join(root, "manifest.json"),
                  [SongMeta.from_dict(m) for m in metas])

    with open(os.path.join(root, "eval", "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)
    print(f"合成完成：曲库 {len(metas)} 首，样本 {len(samples)} 段。下一步：python -m singalong dataset-build")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    root = _data_root(args)
    cfg = DEFAULT_CONFIG
    index = load_index(os.path.join(root, "library.db"))
    matcher = Matcher(index, cfg)

    from .sync_engine import FileClock, SyncEngine
    from .timeline import load_timeline, render_screen

    label_path = args.sample
    if not os.path.exists(label_path):
        print(f"错误：样本标注不存在 {label_path}", file=sys.stderr)
        return 1
    with open(label_path, "r", encoding="utf-8") as f:
        label = json.load(f)

    samples = load_audio(os.path.join(root, label["audio_path"]), cfg.sample_rate)
    clock = FileClock()
    engine = SyncEngine(matcher, clock, cfg)
    step = 0.5
    sr = cfg.sample_rate
    pos = 0
    hint = ""
    state_labels = {
        "recognizing": "识别中…", "playing": "跟唱中", "paused": "已暂停", "lost": "失去同步 · 重新识别中…",
    }
    while pos < len(samples):
        chunk = samples[pos:int(pos + step * sr)]
        clock.advance(len(chunk) / sr)
        cap = clock.now()
        for ev in engine.push(chunk, cap):
            if ev.type.value == "recognized":
                hint = f"已识别 {ev.song_id}"
            elif ev.type.value == "switch":
                hint = f"已切歌 {ev.song_id}"
            elif ev.type.value == "replay":
                hint = "重唱"
            elif ev.type.value == "paused":
                hint = "已暂停"
            elif ev.type.value == "lost":
                hint = "失去同步"
            elif ev.type.value == "resume":
                hint = ""
        now = clock.now()
        p = engine.position_at(now)
        if p.song_id:
            tl_path = os.path.join(root, "songs", p.song_id, "timeline.json")
            tl = load_timeline(tl_path)
            dur = tl.lines[-1].end if tl.lines else 0.0
            print(f"── t={now:5.1f}s ─────────────────────────────")
            print(render_screen(tl, p.position, state_labels[engine.state.value], dur, hint, width=64, color=False))
        else:
            print(f"── t={now:5.1f}s ── 识别中…")
        pos += int(step * sr)
    return 0


def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "?"
    except Exception:
        return "?"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="singalong", description="现场跟唱助手 M0 原型")
    p.add_argument("--data-root", help="数据根目录（默认 data/）")
    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("import", help="导入音频入库")
    sp.add_argument("audio")
    sp.add_argument("--lyrics")
    sp.add_argument("--title")
    sp.add_argument("--language", default="zh")
    sp.add_argument("--song-id")
    sp.set_defaults(func=cmd_import)

    sp = sub.add_parser("align", help="歌词源 -> 时间轴")
    sp.add_argument("audio")
    sp.add_argument("lyrics")
    sp.add_argument("--out", required=True)
    sp.add_argument("--language", default="zh")
    sp.set_defaults(func=cmd_align)

    sp = sub.add_parser("dataset-build", help="校验并重建指纹库")
    sp.set_defaults(func=cmd_dataset_build)

    sp = sub.add_parser("eval", help="一键评测")
    sp.add_argument("--report", help="报告输出路径（默认 docs/EVAL_REPORT.md）")
    sp.set_defaults(func=cmd_eval)

    sp = sub.add_parser("synth", help="生成合成曲库与样本")
    sp.add_argument("--seed", type=int, default=20261126)
    sp.set_defaults(func=cmd_synth)

    sp = sub.add_parser("demo", help="跑通识别->同步->终端歌词全链路")
    sp.add_argument("sample", help="样本标注 JSON 路径")
    sp.set_defaults(func=cmd_demo)

    return p


def _fix_stdout_encoding() -> None:
    """Windows 控制台默认 GBK，打印含非 ASCII 的报告会抛 UnicodeEncodeError 并退出码 1。

    强制 stdout/stderr 为 UTF-8（errors=replace 兜底），保证 `python -m singalong eval` / `demo`
    在 GBK 控制台下也能正常退出（报告文件本身始终以 UTF-8 写入）。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            if hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    _fix_stdout_encoding()
    args = build_parser().parse_args(argv)
    return args.func(args)
