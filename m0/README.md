# M0 原型：现场跟唱助手（Python 3）

不接任何 KTV 点歌系统，只用麦克风听现场声音：自动识别歌曲 → 定位进度 → 持续同步 → 终端逐行歌词跟唱。

本目录是 M0 原型验证：用 Python + numpy/scipy 验证「识别 + 定位 + 持续同步 + 歌词渲染」链路，**不做手机端**。

## 目录

```
m0/
  docs/                设计文档与接口契约（UI_SPEC / EVAL_REPORT_TEMPLATE / DATASET_SPEC / RISKS / INTERFACES）
  singalong/           原型实现（fingerprint / matcher / sync_engine / timeline / 数据管线 / 评测）
  requirements.txt     依赖
  data/                数据根目录（音频/歌词不入库，见 data/README.md）
```

## 快速开始（合成样本跑通全链路）

```bash
cd m0
pip install -r requirements.txt

# 1) 生成合成曲库 + 合成样本 + 真值标注（自制旋律，无版权）
python -m singalong synth --seed 20261126

# 2) 校验并重建指纹库 library.db
python -m singalong dataset-build

# 3) 一键评测，输出 docs/EVAL_REPORT.md
python -m singalong eval

# 4)（可选）看一遍终端跟唱效果
python -m singalong demo data/eval/labels/synth_S001_full.json
```

## 数据准备（真实音频）

`data/` 目录规范见 `docs/DATASET_SPEC.md` 与 `data/README.md`。真实 KTV 录音 / 伴奏放入对应歌曲目录后：

```bash
# 导入一首伴奏（指纹参考）并可选带歌词
python -m singalong import /path/to/instrumental.wav --title "歌曲名" --language zh
python -m singalong import /path/to/instrumental.wav --lyrics /path/to/lyrics.lrc --title "歌曲名"

# 歌词源（LRC/SRT）单独转时间轴 JSON
python -m singalong align /path/to/song.wav /path/to/lyrics.lrc --out /path/to/prefix

# 校验 + 重建指纹库（新增歌曲后跑这个，指纹库不手工维护）
python -m singalong dataset-build
```

## 评测

- 测试集构成、合成参数、真值标注格式见 `docs/DATASET_SPEC.md`。
- 指标口径与通过条件见 `docs/EVAL_REPORT_TEMPLATE.md`。
- 一键回归：`python -m singalong eval`，报告写到 `docs/EVAL_REPORT.md`。
- **模拟结论不作为验收通过依据**：真实 KTV 录音到位后按同一口径补齐（见报告「遗留问题」）。

## 关键参数

全部在 `singalong/config.py`，可调：指纹 FFT 4096/hop 512、频率量化到 FFT bin（≈3.9Hz）、offset bin 32ms（≤50ms，RISKS R2）、置信阈值 0.10、min_align 5、rate 估计 0.85~1.15 + 10% EMA、三级校正 100ms/1s。

## 常见问题

- **识别不出**：确认采样率/单声道已由 `audio_io.load_audio` 统一；确认曲库已 `dataset-build`；降低 `config.py` 的 `confidence_threshold` 或 `min_align`。
- **低置信返回未识别**：这是预期行为（不猜测）。曲库外音频应返回未识别。
- **变速/变调不识别**：Shazam 式哈希对频率/Δt 失真不鲁棒（RISKS R3/R4），属已知限制，见评测报告 §3.6。
- **`data/` 下的文件没进 git**：`.gitignore` 有意排除音频/歌词/标注（版权边界），准备说明见 `data/README.md`。

## 接口契约

模块边界与函数签名级约定见 `docs/INTERFACES.md`，实现以它为准。
