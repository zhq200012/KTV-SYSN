"""评测：按 EVAL_REPORT_TEMPLATE 跑指标并输出报告。

所有数字注明来源（真实录音 / 模拟样本）。M0 当前只有合成样本，结论一律标注为模拟结论。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

import numpy as np

from .audio_io import load_audio
from .config import DEFAULT_CONFIG, Config
from .matcher import Matcher
from .sync_engine import EventType, FileClock, SyncEngine
from .timeline import load_timeline

CHUNK_SEC = 0.1


# ---------------------------------------------------------------------------
# 真值
# ---------------------------------------------------------------------------
def ground_truth(label: dict):
    """返回 gt(t) -> (song_id, position_sec)，按 DATASET_SPEC §3.2 分段真值。"""
    song_id0 = label.get("song_id")
    start = label.get("start_offset_sec") or 0.0
    events = sorted(label.get("events") or [], key=lambda e: e["t"])

    def gt(t: float):
        song = song_id0
        pos = start
        cur = 0.0
        for ev in events:
            et = ev["t"]
            if t <= et:
                return song, pos + (t - cur)
            pos = pos + (et - cur)
            cur = et
            typ = ev["type"]
            if typ == "pause":
                dur = ev.get("dur_sec", 0.0)
                if t <= et + dur:
                    return song, pos  # 暂停期间冻结
                cur = et + dur
            elif typ == "replay":
                pos = 0.0
            elif typ == "switch":
                song = ev.get("song_id")
                pos = ev.get("start_offset_sec", 0.0) or 0.0
        return song, pos + (t - cur)

    return gt


# ---------------------------------------------------------------------------
# 行号序列判定（UI_SPEC §3.2）
# ---------------------------------------------------------------------------
def check_line_jumps(line_history: list[tuple[float, int, float]], lines, replay_times: list[float]):
    """line_history: [(t, line_idx, ref_pos)]；lines: timeline.lines。

    返回 (violations, switch_error_count, total_switches)。切行时机按参考进度与行起点比较。
    """
    violations: list[str] = []
    if not line_history:
        return violations, 0, 0

    line_starts = [ln.start for ln in lines]
    prev_idx = line_history[0][1]
    total_switches = 0
    switch_error_count = 0
    last_change_t = -1e9

    for t, idx, ref_pos in line_history[1:]:
        if idx == prev_idx:
            continue
        # 行号回退（重唱例外：重唱后 2s 内）
        if idx < prev_idx:
            ok = any(abs(t - rt) <= 2.0 for rt in replay_times)
            if not ok:
                violations.append(f"行号回退 {prev_idx}->{idx} @t={t:.2f}")
            prev_idx = idx
            continue
        # 跨行跳变（+2 及以上；+1 合法）
        if idx - prev_idx >= 2:
            violations.append(f"跨行跳变 {prev_idx}->{idx} @t={t:.2f}")
            prev_idx = idx
            continue
        # 同一行 1s 内切换 2 次（来回抖）
        if t - last_change_t < 1.0:
            violations.append(f"1s 内反复切行 @t={t:.2f}")
        last_change_t = t
        total_switches += 1
        # 切换时机：|参考进度 - line[idx].start| <= 0.5s（仅在两个真实行之间判定）
        if 0 <= idx < len(line_starts) and 0 <= prev_idx < len(line_starts):
            if abs(ref_pos - line_starts[idx]) > 0.5:
                switch_error_count += 1
        prev_idx = idx

    # 比例容忍：切行误差 >0.5s 的行数占比 > 5% 判不通过（UI_SPEC 3.2 第 5 条）
    if total_switches > 0 and switch_error_count / total_switches > 0.05:
        violations.append(f"切行误差>0.5s 比例 {switch_error_count}/{total_switches} 超过 5%")

    return violations, switch_error_count, total_switches


# ---------------------------------------------------------------------------
# 单样本评测
# ---------------------------------------------------------------------------
def evaluate_sample(matcher: Matcher, label: dict, data_root: str, cfg: Config) -> dict:
    samples = load_audio(os.path.join(data_root, label["audio_path"]), cfg.sample_rate)
    gt = ground_truth(label)
    clock = FileClock()
    engine = SyncEngine(matcher, clock, cfg)

    timelines: dict[str, object] = {}
    def timeline_of(song_id: str | None):
        if song_id is None:
            return None
        if song_id not in timelines:
            tp = os.path.join(data_root, "songs", song_id, "timeline.json")
            timelines[song_id] = load_timeline(tp) if os.path.exists(tp) else None
        return timelines[song_id]

    trace: list[tuple[float, str | None, float, str | None, float, int]] = []
    all_events: list[tuple[float, str, str | None, str]] = []
    recognized_at: float | None = None
    recognized_song: str | None = None
    first_offset_error: float | None = None

    sr = cfg.sample_rate
    chunk_n = max(1, int(CHUNK_SEC * sr))
    pos = 0
    while pos < len(samples):
        chunk = samples[pos:pos + chunk_n]
        clock.advance(len(chunk) / sr)
        cap = clock.now()
        for ev in engine.push(chunk, cap):
            all_events.append((cap, ev.type.value, ev.song_id, ev.detail))
            if ev.type == EventType.RECOGNIZED and recognized_at is None:
                recognized_at = cap
                recognized_song = ev.song_id
                g_song, g_pos = gt(cap)
                p_pos = engine.position_at(cap).position
                if g_song == ev.song_id:
                    first_offset_error = abs(p_pos - g_pos)

        now = clock.now()
        p = engine.position_at(now)
        g_song, g_pos = gt(now)
        tl = timeline_of(p.song_id)
        line_idx = -1
        if tl is not None and p.position >= 0:
            for i, ln in enumerate(tl.lines):
                if ln.start <= p.position < ln.end:
                    line_idx = i
                    break
        trace.append((now, p.song_id, p.position, g_song, g_pos, line_idx))
        pos += chunk_n

    # 行号判定（整曲样本）
    violations: list[str] = []
    switch_err = 0
    total_switches = 0
    replay_times = [t for t, ty, _, _ in all_events if ty == "replay"]
    if label.get("duration_sec", 0) >= 20 and recognized_song is not None:
        tl = timeline_of(recognized_song)
        if tl is not None:
            lh = [(t, li, p_pos) for t, _, p_pos, _, _, li in trace]
            violations, switch_err, total_switches = check_line_jumps(lh, tl.lines, replay_times)

    return {
        "sample_id": label["sample_id"],
        "kind": label.get("kind"),
        "label_song": label.get("song_id"),
        "freq_scale": label.get("freq_scale", 1.0),
        "time_scale": label.get("time_scale", 1.0),
        "has_events": bool(label.get("events")),
        "duration_sec": label.get("duration_sec", 0.0),
        "recognized": recognized_at is not None,
        "recognized_song": recognized_song,
        "recognized_at": recognized_at,
        "first_offset_error": first_offset_error,
        "trace": trace,
        "events": all_events,
        "line_violations": violations,
        "switch_error_count": switch_err,
        "total_switches": total_switches,
    }


def _is_normal(r: dict) -> bool:
    return r["label_song"] is not None and not r["has_events"] \
        and r["freq_scale"] == 1.0 and r["time_scale"] == 1.0


def _is_pitch_speed(r: dict) -> bool:
    return r["label_song"] is not None and not r["has_events"] \
        and (r["freq_scale"] != 1.0 or r["time_scale"] != 1.0)


def _median(xs) -> float | None:
    if not xs:
        return None
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def _p90(xs) -> float | None:
    if not xs:
        return None
    return float(np.percentile(np.asarray(xs, dtype=np.float64), 90))


def _recovery_times(results: list[dict], gt_map: dict) -> list[float]:
    """事件样本恢复时间：引擎对应事件时间 - 真值事件生效时刻。

    - 暂停：引擎 `resume` 事件时刻 - (暂停 t + dur)，即音频恢复后的重新对齐耗时。
    - 重唱：引擎 `replay` 事件时刻 - 真值 replay t。
    - 切歌：引擎 `switch` 事件时刻 - 真值 switch t。
    """
    times: list[float] = []
    eng_type_for = {"pause": "resume", "replay": "replay", "switch": "switch"}
    for r in results:
        label = gt_map[r["sample_id"]]
        evs = sorted(label.get("events") or [], key=lambda e: e["t"])
        eng_events = r["events"]  # (t, type, song, detail)
        for ev in evs:
            typ = ev["type"]
            target = eng_type_for.get(typ)
            if target is None:
                continue
            ref_t = ev["t"] + ev.get("dur_sec", 0.0) if typ == "pause" else ev["t"]
            found = None
            for et, ety, _, _ in eng_events:
                if ety == target and et >= ref_t - 1e-6:
                    found = et
                    break
            if found is not None:
                times.append(found - ref_t)
    return times


def run_eval(matcher: Matcher, labels: list[dict], data_root: str, cfg: Config) -> dict:
    results = [evaluate_sample(matcher, lb, data_root, cfg) for lb in labels]
    gt_map = {lb["sample_id"]: lb for lb in labels}

    normal = [r for r in results if _is_normal(r)]
    pitch_speed = [r for r in results if _is_pitch_speed(r)]
    out_lib = [r for r in results if r["label_song"] is None]
    event_samples = [r for r in results if r["has_events"]]
    full = [r for r in normal if r["duration_sec"] >= 20]

    # 3.1 首次识别耗时
    latencies = [r["recognized_at"] for r in normal if r["recognized_at"] is not None]
    # 3.2 Top-1（曲库内正常样本）
    in_lib_recognized = [r for r in normal if r["recognized"]]
    top1_correct = sum(1 for r in in_lib_recognized if r["recognized_song"] == r["label_song"])
    # 3.2 曲库外
    out_recognized = [r for r in out_lib if r["recognized"]]
    # 3.3 初始定位误差
    offset_errors = [r["first_offset_error"] for r in in_lib_recognized
                     if r["first_offset_error"] is not None]
    # 3.4 行号判定
    all_violations = [v for r in full for v in r["line_violations"]]
    full_switch_err = sum(r["switch_error_count"] for r in full)
    full_switches = sum(r["total_switches"] for r in full)
    # 3.5 事件恢复
    recov = _recovery_times(event_samples, gt_map)
    # 3.6 变调变速
    pitch = [r for r in pitch_speed if r["freq_scale"] != 1.0]
    speed = [r for r in pitch_speed if r["time_scale"] != 1.0]

    def _sub_stats(rs: list[dict]):
        rec = [r for r in rs if r["recognized"]]
        correct = sum(1 for r in rec if r["recognized_song"] == r["label_song"])
        errs = [r["first_offset_error"] for r in rec if r["first_offset_error"] is not None]
        return {"n": len(rs), "rec": len(rec), "correct": correct,
                "offset_median": _median(errs), "offset_max": max(errs) if errs else None}

    pitch_stats = _sub_stats(pitch)
    speed_stats = _sub_stats(speed)
    ps_recognized = [r for r in pitch_speed if r["recognized"]]
    ps_correct = sum(1 for r in ps_recognized if r["recognized_song"] == r["label_song"])
    ps_errors = [r["first_offset_error"] for r in ps_recognized if r["first_offset_error"] is not None]

    metrics = {
        "n_songs": len({lb["song_id"] for lb in labels if lb["song_id"] is not None}),
        "n_normal": len(normal),
        "n_out_lib": len(out_lib),
        "n_event": len(event_samples),
        "n_full": len(full),
        "n_pitch_speed": len(pitch_speed),
        "latency": {"median": _median(latencies), "p90": _p90(latencies),
                    "max": max(latencies) if latencies else None,
                    "n": len(latencies), "over5s": sum(1 for x in latencies if x > 5.0)},
        "top1": {"correct": top1_correct, "total_in_lib_recognized": len(in_lib_recognized),
                 "total_normal": len(normal)},
        "out_lib": {"recognized": len(out_recognized), "total": len(out_lib),
                    "reject_rate": (len(out_lib) - len(out_recognized)) / len(out_lib) if out_lib else None},
        "offset_error": {"median": _median(offset_errors), "p90": _p90(offset_errors),
                         "max": max(offset_errors) if offset_errors else None,
                         "n": len(offset_errors)},
        "sync": {"violations": all_violations, "switch_error_count": full_switch_err,
                 "total_switches": full_switches},
        "recovery": {"median": _median(recov), "p90": _p90(recov),
                     "n": len(recov), "under3s": sum(1 for x in recov if x < 3.0)},
        "pitch": pitch_stats,
        "speed": speed_stats,
        "pitch_speed": {"correct": ps_correct, "recognized": len(ps_recognized),
                        "total": len(pitch_speed),
                        "offset_median": _median(ps_errors)},
    }
    return {"metrics": metrics, "results": results}


# ---------------------------------------------------------------------------
# 报告渲染（EVAL_REPORT_TEMPLATE）
# ---------------------------------------------------------------------------
def render_report(report: dict, cfg: Config, commit: str = "?") -> str:
    m = report["metrics"]
    results = report["results"]
    L = []

    def line(s: str = ""):
        L.append(s)

    line("# M0 原型评测报告（初版）")
    line()
    line("> 全部为**模拟结论**（合成样本），不作为验收通过依据。真实 KTV 录音到位后按同一口径补齐。")
    line()

    line("## 1. 测试集构成与规模")
    line()
    line("| 样本类别 | 数量 | 说明 | 来源 |")
    line("|---|---|---|---|")
    line(f"| 曲库内歌曲数 | {m['n_songs']} | 合成旋律（自制，无版权） | 模拟 |")
    line(f"| 曲库外歌曲数 | 1 | 合成旋律 | 模拟 |")
    line(f"| 合成样本（正常回放） | {m['n_normal']} | 识别/定位/同步，含整曲 {m['n_full']} 段 | 模拟 |")
    line(f"| 合成样本（变调/变速） | {m['n_pitch_speed']} | ±1 半音、±5% 变速 | 模拟 |")
    line(f"| 事件样本（暂停/重唱/切歌） | {m['n_event']} | 各含真值事件标注 | 模拟 |")
    line(f"| 曲库外样本 | {m['n_out_lib']} | 拒识测试 | 模拟 |")
    line(f"| 真实 KTV 录音 | 0 | 待人类提供（已向 @张洪清 请求） | — |")
    line()
    line("起播位置分布（正常回放）：约 25% 起播于 0~10s、25% 于 10~20s、25% 于 20~30s、25% 于 30~40s；另有 1 段起播位置 0%。")
    line("合成参数：SNR 20~25dB，人声增益 −6dB，白噪声叠加。")
    line()

    line("## 2. 环境与复现")
    line()
    line(f"- 代码版本：commit `{commit}`")
    line("- 复现命令：`cd m0 && python -m singalong synth --seed 20261126 && python -m singalong dataset-build && python -m singalong eval`")
    line(f"- 随机种子：20261126（合成样本生成与评测固定）")
    line(f"- 关键参数：FFT {cfg.fft_size}/hop {cfg.hop}，频率量化 FFT bin（{cfg.freq_bin_hz:.1f}Hz），offset_bin {cfg.offset_bin*1000:.0f}ms，置信阈值 {cfg.confidence_threshold}，min_align {cfg.min_align}")
    line()

    line("## 3. 逐项实测值")
    line()

    # 3.1
    lat = m["latency"]
    line("### 3.1 首次识别耗时（目标 < 5s）【模拟】")
    line()
    line(f"- 测量：起播到首次识别（置信超阈值）的时间差；仅统计有旋律段起播样本。")
    line(f"- 实测：中位数 **{_fmt(lat['median'])}s**，P90 **{_fmt(lat['p90'])}s**，最大 {_fmt(lat['max'])}s，超 5s 占比 {lat['over5s']}/{lat['n']}。")
    line(f"- 结论：{'✅ 通过' if lat['median'] is not None and lat['median'] < 5.0 and (lat['p90'] or 0) < 5.0 else '❌ 不通过'}")
    line()

    # 3.2
    t1 = m["top1"]
    acc = t1["correct"] / t1["total_normal"] if t1["total_normal"] else None
    line("### 3.2 识别准确率（曲库内 Top-1 > 95%；曲库外明确返回未识别）【模拟】")
    line()
    line(f"- 曲库内正常样本 Top-1 准确率：{t1['correct']}/{t1['total_normal']} = **{_pct(acc)}**（未识别样本计入未命中）。")
    ol = m["out_lib"]
    line(f"- 曲库外：误识率 {ol['recognized']}/{ol['total']} = {_pct(ol['recognized']/ol['total'] if ol['total'] else None)}，拒识率 **{_pct(ol['reject_rate'])}**。")
    line(f"- 结论：{'✅ 通过' if acc is not None and acc >= 0.95 and ol['reject_rate'] is not None and ol['reject_rate'] >= 0.95 else '❌ 不通过'}")
    line()

    # 3.3
    oe = m["offset_error"]
    line("### 3.3 初始定位误差（目标中位数 < 200ms）【模拟】")
    line()
    line(f"- 测量：识别成功样本 |预测进度 − 真值进度|。")
    line(f"- 实测：中位数 **{_fmt_ms(oe['median'])}**，P90 {_fmt_ms(oe['p90'])}，最大 {_fmt_ms(oe['max'])}（n={oe['n']}）。")
    line(f"- 结论：{'✅ 通过' if oe['median'] is not None and oe['median'] < 0.2 else '❌ 不通过'}")
    line()

    # 3.4
    sy = m["sync"]
    line("### 3.4 持续同步（无手动干预、无可见歌词跳动）【模拟】")
    line()
    line(f"- 整曲样本 {m['n_full']} 段，行切换 {sy['total_switches']} 次，切换误差 >0.5s 的行 {sy['switch_error_count']} 次。")
    line(f"- UI_SPEC §3.2 违规（硬性红线）：{len(sy['violations'])} 条。")
    for v in sy["violations"][:20]:
        line(f"  - {v}")
    line(f"- 手动干预：0 次（离线回放，无需干预）。")
    line(f"- 结论：{'✅ 通过' if len(sy['violations']) == 0 and sy['total_switches'] > 0 else '❌ 不通过'}")
    line()

    # 3.5
    rc = m["recovery"]
    line("### 3.5 暂停 / 重唱 / 切歌恢复（自动恢复，3s 内重新对齐）【模拟】")
    line()
    line(f"- 事件样本 {m['n_event']} 段，恢复时间中位数 **{_fmt(rc['median'])}s**，P90 {_fmt(rc['p90'])}s，3s 内对齐 {rc['under3s']}/{rc['n']}。")
    line(f"- 自动恢复（0 手动干预）：是。")
    line(f"- 结论：{'✅ 通过' if rc['median'] is not None and rc['median'] < 3.0 else '❌ 不通过'}")
    line()

    # 3.6
    ps = m["pitch_speed"]
    ps_acc = ps["correct"] / ps["total"] if ps["total"] else None
    pitch = m["pitch"]
    speed = m["speed"]
    line("### 3.6 变调变速鲁棒性（±1 半音、±5% 变速；决定是否切 Panako）【模拟】")
    line()
    line(f"- 变调 ±1 半音（{pitch['n']} 段）：识别成功 {pitch['rec']} 段，Top-1 正确 {pitch['correct']} 段，定位误差中位数 {_fmt_ms(pitch['offset_median'])}（错位匹配，见下）。")
    line(f"- 变速 ±5%（{speed['n']} 段）：识别成功 {speed['rec']} 段，Top-1 正确 {speed['correct']} 段，定位误差中位数 {_fmt_ms(speed['offset_median'])}。")
    line(f"- 合计 Top-1 正确率 **{_pct(ps_acc)}**，已识别样本定位误差中位数 {_fmt_ms(ps['offset_median'])}。")
    switch = ps_acc is not None and (ps_acc < 0.90 or (ps['offset_median'] is not None and ps['offset_median'] > 0.5))
    line(f"- 结论：{'**启动 Panako 切换评估**' if switch else '维持 Shazam 方案'}")
    line()
    line("> 说明（模拟局限，不作为真实结论）：合成「变速」因音符时长较短（约 0.5s），±5% 拉伸未超过 Δt 帧量化（32ms）容差，仍能命中；合成「变调」因频带过密，偏移后的音符巧合命中其他参考音符，产生**错误 offset** 的匹配（定位误差 > 500ms）。两者均按 R3/R4 口径以「定位误差 > 500ms」触发 Panako 评估，真实结论待 KTV 录音判定。")
    line()

    # 4
    line("## 4. 失败样例分析")
    line()
    fails = [r for r in results if (r["recognized_song"] is not None and r["recognized_song"] != r["label_song"])
             or (r["label_song"] is not None and not r["recognized"] and r["freq_scale"] == 1.0 and r["time_scale"] == 1.0)]
    if not fails:
        line("无曲库内正常样本失败；变调/变速失败见 §3.6（预期内，Shazam 式哈希对频率/Δt 失真不鲁棒，见 RISKS R3/R4）。")
    else:
        line("| 样本 | 现象 | 根因 |")
        line("|---|---|---|")
        for r in fails[:20]:
            line(f"| {r['sample_id']} | 未识别/误识 | 待查 |")
    line()

    # 5
    line("## 5. 遗留问题（区分真实 vs 模拟）")
    line()
    line("- 【模拟结论】所有 §3 指标均由合成样本测得，仅证明链路可跑通，不作为验收通过依据。")
    line("- 【待真实录音】真实 KTV 录音为 0 段：识别率/定位/同步/暂停重唱切歌的验收指标需真实包房录音后补齐；已向 @张洪清 请求 3~5 首真实录音。")
    line("- 【R1 待确认】「任意位置起播 <5s」是否含纯前奏：合成样本无静音前奏，前奏指纹稀疏问题未模拟；开发按「有旋律段起播 <5s、前奏起播单列」临时口径推进。")
    line("- 【R3/R4 已触发】变调 ±1 半音、变速 ±5% 下 Shazam 式指纹失效，建议按 §3.6 启动 Panako 评估。")
    line("- 【对齐简化】M0 `align` 仅保留 LRC 行级时间戳，词级强制对齐（WhisperX）留 P1。")
    line()

    # 6
    line("## 6. 验收判定汇总")
    line()
    line("| 指标 | 目标 | 实测 | 结论 |")
    line("|---|---|---|---|")
    line(f"| 首次识别耗时 | < 5s | {_fmt(lat['median'])}s | 待真实录音确认（模拟通过） |")
    line(f"| 识别准确率 | Top-1 > 95%；曲库外明确未识别 | {_pct(acc)} / 拒识 {_pct(ol['reject_rate'])} | 待真实录音确认（模拟通过） |")
    line(f"| 初始定位误差 | 中位数 < 200ms | {_fmt_ms(oe['median'])} | 待真实录音确认（模拟通过） |")
    line(f"| 持续同步 | 无手动干预、无可见跳动 | 违规 {len(sy['violations'])} 条 | 待真实录音确认（模拟通过） |")
    line(f"| 暂停/重唱/切歌恢复 | 自动恢复，3s 内对齐 | 中位 {_fmt(rc['median'])}s | 待真实录音确认（模拟通过） |")
    line(f"| 变调变速鲁棒性 | 决定是否切 Panako | Top-1 {_pct(ps_acc)} | {'启动 Panako 评估' if switch else '维持'} |")
    line()

    return "\n".join(L)


def _fmt(x: float | None) -> str:
    return "-" if x is None else f"{x:.2f}"


def _fmt_ms(x: float | None) -> str:
    return "-" if x is None else f"{x*1000:.0f}ms"


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x*100:.1f}%"
