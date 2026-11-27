# 数据目录规范

本目录是数据根目录（DESIGN.md 4.5 / DATASET_SPEC.md 1）。**音频与歌词不入库**，`.gitignore` 已排除。

```
data/
  songs/<song_id>/
    meta.json          # id / title / language / license / source / audio_path / duration
    reference.wav      # 指纹参考音频（伴奏优先，无伴奏用完整版）——不入库
    lyrics.lrc         # 歌词源（可选）——不入库
    timeline.json      # 时间轴（含歌词文本）——不入库
  manifest.json        # 曲库索引（纯元数据，入库）
  library.db           # 统一指纹库（dataset-build 重建）——不入库
  eval/
    labels/            # 真值标注 *.json ——不入库；README.md 入库
    real/              # 真实 KTV 录音 ——不入库
    synth/             # 合成样本 ——不入库
    manifest.json      # 样本索引（入库）
```

## 准备真实数据

1. 伴奏放入 `data/songs/<song_id>/instrumental.wav`（或 `reference.wav` / `vocals.wav`），见 DESIGN 4.5。
2. 写 `meta.json` 或用 `python -m singalong import` 自动登记。
3. 歌词 `lyrics.lrc` / `lyrics.srt` 用 `python -m singalong align` 生成 `timeline.json`。
4. `python -m singalong dataset-build` 校验并重建指纹库。

## 版权边界

- 不提交无授权音频与歌词；冷启动只用自制 / 授权数据验证技术。
- 合成样本为自制旋律，可自由生成（`python -m singalong synth`）与复现。
