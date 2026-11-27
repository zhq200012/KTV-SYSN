"""核心模块单测：指纹/匹配（低置信返回未识别）、同步引擎状态机、时间轴解析。

运行：cd m0 && python -m unittest discover -s tests -v
"""
import unittest

import numpy as np

from singalong import fingerprint, synth
from singalong.config import DEFAULT_CONFIG
from singalong.matcher import Matcher
from singalong.sync_engine import EventType, FileClock, SyncEngine, SyncState
from singalong.timeline import current_line, load_timeline
from singalong.lyrics import parse_lrc
from singalong import eval_runner
from singalong.index_store import load_index
import os, tempfile, json
from unittest import mock


def _index_of(songs, duration=4.0):
    """songs: [(song_id, seed, band)] -> Index（默认索引 duration 秒参考）。"""
    index = {}
    for sid, seed, band in songs:
        samples = synth.generate_song(seed, duration, band_index=band)
        for h, t in fingerprint.fingerprint(samples, DEFAULT_CONFIG):
            index.setdefault(h, []).append((sid, t))
    return index


class TestMatcher(unittest.TestCase):
    def setUp(self):
        self.index = _index_of([("A", 1, 0), ("B", 2, 1)])
        self.matcher = Matcher(self.index, DEFAULT_CONFIG)

    def test_clean_match_correct_offset(self):
        samples = synth.generate_song(1, 4.0, band_index=0)
        win = samples[int(1.0 * 16000):int(3.0 * 16000)]
        res = self.matcher.match(fingerprint.fingerprint(win, DEFAULT_CONFIG))
        self.assertIsNotNone(res)
        self.assertEqual(res.song_id, "A")
        self.assertAlmostEqual(res.offset, 1.0, delta=0.2)

    def test_out_of_library_returns_none(self):
        samples = synth.generate_song(999, 4.0, band_index=5)  # 不在索引
        res = self.matcher.match(fingerprint.fingerprint(samples, DEFAULT_CONFIG))
        self.assertIsNone(res)

    def test_noise_returns_none(self):
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.1, 32000).astype(np.float32)
        res = self.matcher.match(fingerprint.fingerprint(noise, DEFAULT_CONFIG))
        self.assertIsNone(res)

    def test_offset_bin_within_50ms(self):
        self.assertLessEqual(DEFAULT_CONFIG.offset_bin, 0.05)


class TestSyncEngine(unittest.TestCase):
    def _run(self, eng, sig, sr=16000, step=0.1):
        events = []
        pos = 0
        while pos < len(sig):
            chunk = sig[pos:pos + int(step * sr)]
            eng.clock.advance(len(chunk) / sr)
            events.extend(eng.push(chunk, eng.clock.now()))
            pos += int(step * sr)
        return events

    def test_recognize_then_replay(self):
        index = _index_of([("A", 1, 0)])
        matcher = Matcher(index, DEFAULT_CONFIG)
        clock = FileClock()
        eng = SyncEngine(matcher, clock, DEFAULT_CONFIG)

        song = synth.generate_song(1, 8.0, band_index=0)
        sr = 16000
        events = []
        # 播 0~4s
        pos = 0
        while pos < 4 * sr:
            chunk = song[pos:pos + int(0.1 * sr)]
            clock.advance(len(chunk) / sr)
            events += eng.push(chunk, clock.now())
            pos += int(0.1 * sr)
        types = [e.type for e in events]
        self.assertIn(EventType.RECOGNIZED, types)
        self.assertEqual(eng.current_song, "A")

        # 重唱：从 0 再播
        events2 = []
        pos = 0
        while pos < 3 * sr:
            chunk = song[pos:pos + int(0.1 * sr)]
            clock.advance(len(chunk) / sr)
            events2 += eng.push(chunk, clock.now())
            pos += int(0.1 * sr)
        types2 = [e.type for e in events2]
        self.assertIn(EventType.REPLAY, types2)

    def test_rate_estimation_tracks_speed(self):
        # 变速 1.05 连播 >10s，断言 rate 明显偏离 1.0 并朝 1.05 收敛（RISKS R3 / DESIGN 3.3）
        ref = synth.generate_song(1, 60.0, band_index=0)
        index = {}
        for h, t in fingerprint.fingerprint(ref, DEFAULT_CONFIG):
            index.setdefault(h, []).append(("A", t))
        matcher = Matcher(index, DEFAULT_CONFIG)
        eng = SyncEngine(matcher, FileClock(), DEFAULT_CONFIG)

        song = synth.generate_song(1, 60.0, time_scale=1.05, band_index=0)
        sr = 16000
        self._run(eng, song[:int(40.0 * sr)])
        self.assertGreater(eng.rate, 1.01, f"rate 应朝 1.05 收敛，实际 {eng.rate:.4f}")
        self.assertLess(eng.rate, 1.06, f"rate 不应超调，实际 {eng.rate:.4f}")

    def test_lost_then_resume(self):
        # 识别成功 -> 高能量无法匹配噪声 -> LOST -> 恢复原曲 -> RESUME
        # 索引覆盖整首（16s），保证恢复段（6~12s）在曲库内、非靠误匹配触发。
        index = _index_of([("A", 1, 0)], duration=16.0)
        matcher = Matcher(index, DEFAULT_CONFIG)
        eng = SyncEngine(matcher, FileClock(), DEFAULT_CONFIG)
        song = synth.generate_song(1, 16.0, band_index=0)
        sr = 16000

        events = self._run(eng, song[:int(6.0 * sr)])
        self.assertIn(EventType.RECOGNIZED, [e.type for e in events])

        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.2, int(8.0 * sr)).astype(np.float32)
        events += self._run(eng, noise)
        self.assertIn(EventType.LOST, [e.type for e in events])
        self.assertEqual(eng.state, SyncState.LOST)

        events += self._run(eng, song[int(6.0 * sr):int(12.0 * sr)])
        self.assertIn(EventType.RESUME, [e.type for e in events])
        self.assertEqual(eng.state, SyncState.PLAYING)

    def test_frozen_position_equals_extrapolated(self):
        # 回归 7ea3f55：失锁瞬间冻结进度 = 切换前的外推进度，而非识别锚点。
        index = _index_of([("A", 1, 0)], duration=16.0)
        matcher = Matcher(index, DEFAULT_CONFIG)
        eng = SyncEngine(matcher, FileClock(), DEFAULT_CONFIG)
        song = synth.generate_song(1, 16.0, band_index=0)
        sr = 16000

        # 识别（锚点约 2s）并推进到 6s，进度外推
        self._run(eng, song[:int(6.0 * sr)])
        self.assertEqual(eng.current_song, "A")
        pos_before = eng.position_at(eng.clock.now()).position
        self.assertAlmostEqual(pos_before, 6.0, delta=0.3)

        # 喂噪声触发失锁
        rng = np.random.default_rng(0)
        noise = rng.normal(0.0, 0.2, int(8.0 * sr)).astype(np.float32)
        self._run(eng, noise)
        self.assertEqual(eng.state, SyncState.LOST)

        # 冻结进度应 ≈ 失锁时刻外推进度（>识别锚点），而非锚点 ~2s
        frozen = eng.position_at(eng.clock.now()).position
        self.assertGreater(frozen, pos_before + 1.0, "冻结进度应外推，而非停在识别锚点")
        self.assertLess(frozen, pos_before + 9.0)


class TestTimeline(unittest.TestCase):
    def test_load_and_current_line(self):
        tl = synth.build_timeline("S001", "t", 12.0)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
            json.dump(tl, f)
            path = f.name
        try:
            loaded = load_timeline(path)
            self.assertEqual(loaded.song_id, "S001")
            ln = current_line(loaded, 4.5)
            self.assertIsNotNone(ln)
            self.assertEqual(ln.text, "跟唱测试歌词 · 第1行")
        finally:
            os.unlink(path)


class TestLyrics(unittest.TestCase):
    def test_parse_lrc_skips_metadata(self):
        content = "\n".join([
            "[0:00] 夜空中最亮的星 (Live) - 逃跑计划",
            "[0:05] 词：某人",
            "[0:10] 曲：某人",
            "[0:15] 第一句歌词",
            "[0:20] 第二句歌词",
        ])
        lines = parse_lrc(content)
        texts = [t for _, t in lines]
        self.assertEqual(texts, ["第一句歌词", "第二句歌词"])
        self.assertEqual(lines[0][0], 15.0)


class TestEvalSplit(unittest.TestCase):
    """run_eval 按样本 kind 把真实（干净源）与模拟样本分开统计。"""

    def _fake_eval(self, matcher, label, data_root, cfg):
        return {
            "sample_id": label["sample_id"], "kind": label.get("kind"),
            "label_song": label.get("song_id"), "freq_scale": 1.0, "time_scale": 1.0,
            "has_events": bool(label.get("events")),
            "duration_sec": label.get("duration_sec", 8.0),
            "recognized": True, "recognized_song": label.get("song_id"),
            "recognized_at": 2.0, "first_offset_error": 0.01,
            "trace": [], "events": [], "line_violations": [],
            "switch_error_count": 0, "total_switches": 0,
        }

    def _labels(self):
        return [
            {"sample_id": "real_1", "kind": "real", "song_id": "R001",
             "start_offset_sec": 0.0, "duration_sec": 8.0, "events": [],
             "freq_scale": 1.0, "time_scale": 1.0},
            {"sample_id": "real_2", "kind": "real", "song_id": "R001",
             "start_offset_sec": 10.0, "duration_sec": 8.0, "events": [],
             "freq_scale": 1.0, "time_scale": 1.0},
            {"sample_id": "synth_1", "kind": "synth", "song_id": "S001",
             "start_offset_sec": 2.0, "duration_sec": 8.0, "events": [],
             "freq_scale": 1.0, "time_scale": 1.0},
        ]

    def test_run_eval_splits_real_vs_synth(self):
        with mock.patch.object(eval_runner, "evaluate_sample", side_effect=self._fake_eval):
            report = eval_runner.run_eval(None, self._labels(), "data", DEFAULT_CONFIG)
        m = report["metrics"]
        self.assertEqual(m["n_real"], 2)
        self.assertEqual(m["n_synth"], 1)
        self.assertEqual(m["n_real_songs"], 1)
        self.assertEqual(m["real"]["top1"]["correct"], 2)
        self.assertEqual(m["synth"]["top1"]["correct"], 1)

    def test_render_report_marks_real_and_sim(self):
        with mock.patch.object(eval_runner, "evaluate_sample", side_effect=self._fake_eval):
            report = eval_runner.run_eval(None, self._labels(), "data", DEFAULT_CONFIG)
        md = eval_runner.render_report(report, DEFAULT_CONFIG, commit="test")
        self.assertIn("真实（干净源）", md)
        self.assertIn("【模拟】", md)


class TestRepeatDisambiguation(unittest.TestCase):
    """重复副歌段消歧回归：短窗口歧义 -> 更长窗口消歧；offset 唯一性校验生效。"""

    def test_short_window_ambiguous_longer_disambiguates(self):
        sr = 16000
        intro = synth.generate_song(11, 4.0, band_index=0)
        chorus = synth.generate_song(12, 4.0, band_index=0)
        verse = synth.generate_song(13, 4.0, band_index=0)
        # 副歌在 4~8s 与 12~16s 重复出现
        ref = np.concatenate([intro, chorus, verse, chorus])
        index = {}
        for h, t in fingerprint.fingerprint(ref, DEFAULT_CONFIG):
            index.setdefault(h, []).append(("REP", t))
        matcher = Matcher(index, DEFAULT_CONFIG)

        # 2s 窗口（副歌内）：与另一处副歌打成平手 -> 歧义
        r2 = matcher.match(fingerprint.fingerprint(ref[4*sr:6*sr], DEFAULT_CONFIG))
        self.assertIsNotNone(r2)
        self.assertTrue(r2.ambiguous, "2s 窗口在重复副歌段应判定为歧义")

        # 6s 窗口（副歌 + 主歌）：唯一 -> 消歧，offset 落在 4s
        r6 = matcher.match(fingerprint.fingerprint(ref[4*sr:10*sr], DEFAULT_CONFIG))
        self.assertIsNotNone(r6)
        self.assertFalse(r6.ambiguous)
        self.assertAlmostEqual(r6.offset, 4.0, delta=0.2)


def _real_data_available() -> bool:
    return os.path.exists("data/library.db") and os.path.exists("data/eval/real/real_R001_off180.wav")


class TestRealRepeatRegression(unittest.TestCase):
    """真实曲目重复副歌回归（依赖本地不入库数据，缺失则跳过）。"""

    @unittest.skipUnless(_real_data_available(), "需要本地真实曲目数据（不入库）")
    def test_real_r001_off180_offset_within_1s(self):
        """回归：R001 off180（重复副歌段）初始定位误差不得出现 >1s 离群。"""
        index = load_index("data/library.db")
        matcher = Matcher(index, DEFAULT_CONFIG)
        with open("data/eval/manifest.json", encoding="utf-8") as f:
            labels = json.load(f)["samples"]
        lb = [x for x in labels if x["sample_id"] == "real_R001_off180"][0]
        r = eval_runner.evaluate_sample(matcher, lb, "data", DEFAULT_CONFIG)
        self.assertTrue(r["recognized"], "off180 应能识别")
        self.assertIsNotNone(r["first_offset_error"])
        self.assertLess(r["first_offset_error"], 1.0, "off180 初始定位误差不得 >1s")


if __name__ == "__main__":
    unittest.main()
