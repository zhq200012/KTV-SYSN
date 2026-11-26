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
import os, tempfile, json


def _index_of(songs):
    """songs: [(song_id, seed, band)] -> Index。"""
    index = {}
    for sid, seed, band in songs:
        samples = synth.generate_song(seed, 4.0, band_index=band)
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
        index = _index_of([("A", 1, 0)])
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


if __name__ == "__main__":
    unittest.main()
