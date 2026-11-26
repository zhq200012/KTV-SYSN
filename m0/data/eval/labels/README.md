# 真值标注格式

每样本一个 JSON，放本目录（`data/eval/labels/<sample_id>.json`，**不入库**）。格式见 `../../docs/DATASET_SPEC.md` §3。

```json
{
  "sample_id": "real_0001",
  "kind": "real",
  "audio_path": "eval/real/real_0001.wav",
  "song_id": "S001",
  "start_offset_sec": 61.20,
  "duration_sec": 25.00,
  "events": [
    { "t": 8.00,  "type": "pause",  "dur_sec": 2.0 },
    { "t": 15.00, "type": "replay" },
    { "t": 20.00, "type": "switch", "song_id": "S002", "start_offset_sec": 0.0 }
  ],
  "notes": "人声跟唱，轻微碰杯"
}
```

字段：`sample_id` 唯一 ID；`kind` = `real` / `synth`；`audio_path` 相对 `data/`；`song_id` 曲库内 ID（曲库外为 `null`）；`start_offset_sec` 起播点在参考音频中的秒数（曲库外 `null`）；`duration_sec` 时长；`events` 暂停/重唱/切歌事件（可选）；`notes` 备注。

合成样本额外字段：`freq_scale`（变调倍率）、`time_scale`（变速倍率）、`snr_db`。

进度真值：回放样本 = `start_offset_sec + 已回放时长`；真实录音由人工听音标 `start_offset_sec`（允许 ±50ms）。
