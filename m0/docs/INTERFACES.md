# M0 原型 · 模块边界与接口契约（INTERFACES.md）

版本：v0.1 · 适用：M0 原型验证阶段

> 本契约定义 `m0/singalong/` 各模块的边界、函数签名级接口、数据模型、低置信与错误返回约定，以及 `captured_at` 的传递方式。实现必须与本文一致；本文只收敛 `DESIGN.md` v0.1、`UI_SPEC.md`、`EVAL_REPORT_TEMPLATE.md`、`DATASET_SPEC.md`、`RISKS.md` 已明确的口径，**不新增产品范围**（不做打分 / 点歌 / 手机端 / 云端）。

## 0. 目录与运行约定

- 原型目录 `m0/`，包 `singalong/`，数据根目录 `data/`（相对运行 CWD，即从 `m0/` 下执行 `python -m singalong ...`）。
- 采样统一：**16kHz、单声道、float32、取值 [-1,1]**（音频 I/O 负责采集后的重采样与单声道化）。
- 时间戳统一为 **float 秒**，浮点时钟（wall clock 或文件回放时钟，见 §5）。

## 1. 模块边界

| 模块 | 职责 | 不做什么 |
|---|---|---|
| `audio_io.py` | 音频读写、重采样、单声道化 | 不碰指纹 |
| `fingerprint.py` | 频谱峰值提取 + landmark 哈希（识别与定位共用） | 不做匹配、不建索引 |
| `matcher.py` | 曲库索引匹配、offset 直方图投票、置信度 | 不做状态机、不做时钟 |
| `sync_engine.py` | 单调时钟推进 + 周期重对齐 + rate 估计 + 三级校正 + 状态机 + 瞬时事件 | 不做渲染、不做匹配本身 |
| `timeline.py` | 时间轴 JSON 解析与终端渲染（歌词/和弦/倒数） | 不做状态机、不推进进度 |
| `index_store.py` | `manifest.json` 与 `library.db` 的读写、校验 | 不做指纹提取/匹配 |
| `synth.py` | 合成样本生成（伴奏+人声+噪声） | 不读真实录音 |
| `eval_runner.py` | 按 EVAL_REPORT_TEMPLATE 跑指标、输出报告 | 不改算法 |
| `cli.py` | 命令行入口 `import / align / dataset-build / eval / synth` | 不含算法逻辑 |

依赖方向（无环）：`audio_io` ← `fingerprint` ← `matcher` ← `sync_engine`；`timeline` 独立；`index_store` 独立；`cli` / `eval_runner` / `synth` 为顶层编排。

## 2. 数据模型

### 2.1 音频

```python
# audio_io.py
def read_wav(path: str) -> tuple[np.ndarray, int]          # -> (samples float32 [-1,1] mono, sample_rate)
def write_wav(path: str, samples: np.ndarray, sample_rate: int) -> None
def to_mono(samples: np.ndarray) -> np.ndarray             # 多声道 -> 单声道（均值）
def resample(samples: np.ndarray, from_rate: int, to_rate: int = 16000) -> np.ndarray
def load_audio(path: str) -> np.ndarray                    # -> 16k 单声道 float32 [-1,1]（内部调用上述）
```

### 2.2 指纹

```python
# fingerprint.py
SAMPLE_RATE = 16000
FFT_SIZE    = 4096
HOP         = 512
WINDOW      = "hann"

# Peak/landmark 参数（DESIGN 4.1，可调）
PEAK_NEIGHBORHOOD = 20        # 频域局部极大窗口（±bin）
PEAK_AMP_MIN      = 10.0      # 幅度下限（相对谱，见实现）
FAN_VALUE         = 15        # 每个峰值向后配对的峰值数
TARGET_ZONE       = (1, 15)   # 时间差 Δt 帧数范围 [min,max]（含）
# 频率量化到 FFT bin 索引（bin 宽 = sample_rate/fft_size ≈ 3.9Hz，与 dejavu 一致）
# 哈希 = (f1_bin, f2_bin, dt_frames)，编码为 int（12/12/8 bit）

def extract_peaks(samples: np.ndarray) -> list[tuple[float, float]]
    # -> [(t_sec, f_hz), ...]，按帧内局部极大 + 幅度阈值筛选
def generate_hashes(peaks: list[tuple[float, float]]) -> list[tuple[int, float]]
    # -> [(hash_value, t_sec), ...]，t_sec 为配对中第一个峰值的时间（用于 offset 反推）
def fingerprint(samples: np.ndarray) -> list[tuple[int, float]]
    # = extract_peaks + generate_hashes
```

### 2.3 匹配结果与索引

```python
# matcher.py
@dataclass
class MatchResult:
    song_id: str
    offset: float          # 参考音频中，查询窗口起点的秒数
    confidence: float      # 0..1，= aligned_count / len(query_hashes)
    aligned_count: int
    query_hashes: int
    second_count: int      # 得票第二的候选 aligned 数（用于可解释性）

# 索引：hash -> list[(song_id, t_ref_sec)]（t_ref_sec = 参考音频中该 hash 第一个峰值的时间）
Index = dict[int, list[tuple[str, float]]]

class Matcher:
    def __init__(self, index: Index, config: "Config"): ...
    def match(self, query_hashes: list[tuple[int, float]]) -> MatchResult | None
        # offset 直方图 bin 宽 ≤ 50ms（实现用 1 帧 = 32ms），峰位抛物线插值细化
        # 低置信返回 None（未识别，不猜测）
```

**低置信与错误返回约定**（贯穿全包）：

- 匹配置信度低于阈值 → `Matcher.match` 返回 `None`；上层状态机进入「识别中/未识别/失锁」，**绝不猜测**。
- 阈值：`confidence = aligned_count / query_hashes`，须同时满足
  `aligned_count >= MIN_ALIGN`（默认 5）且 `confidence >= CONFIDENCE_THRESHOLD`（默认 0.10）
  且 `aligned_count >= 1.5 * second_count`（唯一性边际）。三者为可调参数，写在 `config.py`。
- 文件/数据校验失败抛 `ValueError`（带可读消息）；曲库外音频「明确返回未识别」= 无候选超过阈值。

### 2.4 时间轴

```python
# timeline.py
@dataclass
class Word:   t: float; text: str
@dataclass
class Chord:  t: float; name: str
@dataclass
class Line:
    start: float; end: float; text: str
    words: list[Word] = field(default_factory=list)      # P1 保留，不丢弃
    chords: list[Chord] = field(default_factory=list)
@dataclass
class Section: type: str; start: float; end: float
@dataclass
class Timeline:
    song_id: str; title: str; reference_version: str
    lines: list[Line]; sections: list[Section]; bpm: float | None = None

def load_timeline(path: str) -> Timeline                 # 解析 4.4 JSON，保留 words/chords/sections/bpm
def current_line(tl: Timeline, position: float) -> Line | None
def next_line(tl: Timeline, position: float) -> Line | None
def render_line(line: Line, position: float) -> str      # P1 接口：M0 返回整行
```

### 2.5 同步引擎

```python
# sync_engine.py
class SyncState(Enum):
    RECOGNIZING = "recognizing"   # 识别中
    PLAYING     = "playing"       # 播放中（跟唱中）
    PAUSED      = "paused"        # 暂停
    LOST        = "lost"          # 失锁

class EventType(Enum):
    RECOGNIZED = "recognized"     # 首次识别成功
    SWITCH     = "switch"         # 切歌（命中另一首）
    REPLAY     = "replay"         # 重唱（offset 回退到开头附近）
    RESUME     = "resume"         # 暂停/失锁恢复
    PAUSED     = "paused"
    LOST       = "lost"

@dataclass
class Event:
    type: EventType
    song_id: str | None
    position_sec: float | None
    captured_at: float
    detail: str = ""

@dataclass
class Position:
    song_id: str | None          # None 表示未识别
    position: float              # 参考音频中的播放进度（秒）；未识别时为 -1
    rate: float                  # 播放速率估计（默认 1.0）
    captured_at: float           # 最近锚点的采集时间
    applied_at: float            # 该锚点应用时间

class SyncEngine:
    def __init__(self, matcher: Matcher, clock: "Clock", config: "Config"): ...
    # 喂入一段已重采样的音频（16k 单声道）。captured_at = 该 chunk 采集结束时刻（DESIGN 3.3）。
    def push(self, samples: np.ndarray, captured_at: float) -> list[Event]
    # 在单调时钟时刻 now 的当前播放进度（本地时钟外推）。
    def position_at(self, now: float) -> Position
    @property
    def state(self) -> SyncState
```

**状态机**（DESIGN 3.3 + RISKS R6 事件）：

```
识别中 ──识别成功──▶ 播放中 ──命中另一首歌──▶ 播放中(emit SWITCH)
  ▲                   │──offset 回退到开头附近──▶ 播放中(emit REPLAY, 重置进度)
  │                   │──匹配持续失败且能量低──▶ 暂停(emit PAUSED, 进度冻结)
  │                   │──匹配持续失败但非暂停──▶ 失锁(emit LOST, 后台重试识别)
  └── 恢复且匹配原曲 ──┴── 失锁/暂停 ──音频恢复且匹配原曲位置附近──▶ 播放中(emit RESUME)
```

**校正与 rate**（DESIGN 3.3）：

- 正常推进靠单调时钟外推；每 2~5s（`REALIGN_INTERVAL` 默认 2.0s）或每次 `push` 用匹配结果重对齐。
- rate 估计：**rate 测量锚点与位置锚点分离**——`_rate_ref/_rate_cap` 只在「累计间隔 ≥ `RATE_MIN_SPAN`（10s）」时计算 `rate = Δref / Δcapture` 并重置，过滤 `RATE_MIN=0.85` ~ `RATE_MAX=1.15` 之外，10% 指数平滑（`RATE_EMA=0.10`）；切歌/重唱/恢复时重置。不随每次重对齐重置，避免 span 恒等于重对齐周期导致 rate 恒为 1.0。
- 三级校正：`|偏差| < 0.10s` 不调整；`0.10~1.0s` 渐进（约 1.5s 内追平）；`>1.0s` 直接跳转。跳转仅用于合法场景（切歌/重唱），正常校正不得累积到 >1s（UI_SPEC 3.2 红线）。
- **未实现（M0 接受，已在契约注明）**：DESIGN 3.3「连续多次同向大偏差 → 跳转」未落地——`_on_match` 目前只按单次 `|deviation|` 判级，未累计同向大偏差。

### 2.6 索引存储与数据管线

```python
# index_store.py
@dataclass
class SongMeta:
    song_id: str; title: str; language: str; license: str; source: str
    audio_path: str          # 相对 data/ 的指纹参考音频路径
    duration_sec: float; timeline_path: str | None

def load_manifest(path: str) -> list[SongMeta]
def save_manifest(path: str, metas: list[SongMeta]) -> None
def build_index(songs: list[SongMeta], data_root: str, config: "Config") -> Index
    # 对每首歌 fingerprint -> 并入 Index；写 library.db（序列化）
def load_index(path: str) -> Index
def save_index(path: str, index: Index) -> None
```

数据管线 CLI（`cli.py`，`python -m singalong ...`）：

```text
import <音频.mp3|wav> [--lyrics <.lrc|.srt>] [--title 标题] [--language zh] [--song-id SXXX]
       # 转码/重采样 -> data/songs/<id>/{reference.wav, meta.json, timeline.json?}，注册 manifest
align <音频.wav> <歌词.lrc|srt> --out <前缀> [--language zh]
       # 歌词 -> 行级对齐 -> timeline.json（M0 简化：保留原 LRC 行时间戳，词级 P1 留接口）
dataset-build
       # 校验全部歌曲（时序/起止/不超时长）-> 重建 library.db + manifest.json
eval [--samples <label.json|dir>] [--report <out.md>]
       # 一键评测，输出报告
synth [--seed N] [--songs N] [--out data/]
       # 生成合成曲库 + 合成样本 + 真值标注（dev 工具）
```

- 数据目录规范见 `DATASET_SPEC.md` §1；`.gitignore` 排除音频/歌词/`eval/labels/*.json`。
- 新增歌曲只放目录再 `dataset-build`，指纹库不手工维护。

### 2.7 评测入口

```python
# eval_runner.py
def run_eval(data_root: str, config: "Config", samples: list[dict]) -> EvalReport
@dataclass
class EvalReport:
    metrics: dict        # 按 EVAL_REPORT_TEMPLATE 第 3 节逐项
    failures: list[dict] # 失败样例
    summary: dict        # 第 6 节验收判定汇总
def render_report(report: EvalReport) -> str    # Markdown
```

## 3. 低置信 / 错误返回约定（汇总）

| 情况 | 返回 | 下游 |
|---|---|---|
| 匹配置信度低于阈值 / 无候选 | `Matcher.match -> None` | 状态机保持「识别中」或转「失锁」，渲染「未识别到歌曲」 |
| 曲库外音频 | 无候选超阈值 → `None` | 明确「未识别」，不猜测 |
| 数据文件缺失 / JSON 非法 / 标注越界 | `raise ValueError(msg)` | CLI 打印 `msg` 并退出码非 0 |
| 音频解码失败 | `raise ValueError(msg)` | 同上 |
| 时间轴行时序错乱（start>end 等） | `dataset-build` 校验失败并报错 | 停止建库 |

## 4. `captured_at` 传递方式（RISKS R7）

- 统一 `Clock` 抽象：`Clock.now() -> float` 单调秒。
- **实时麦克风**：`CapturedClock`，`now()` 返回 `time.monotonic()`；`push` 的 `captured_at` = 采集结束 wall 时刻。
- **文件回放**：`FileClock`，`now()` 由回放器随 chunk 推进（`advance(duration)`）；`captured_at` = 文件内累计时间戳。
- 补偿公式（DESIGN 3.3）：`position_now = position_capture + (applied_at - captured_at) * rate`。
- 离线评测统一走 `FileClock`，使延迟补偿可单测、可复现。

## 5. 参数表（`config.py`，全部可调、带默认值）

| 参数 | 默认 | 说明 |
|---|---|---|
| `SAMPLE_RATE` | 16000 | 采样率 |
| `FFT_SIZE` / `HOP` | 4096 / 512 | 指纹 FFT |
| `PEAK_AMP_MIN` | 10.0 | 峰值幅度下限 |
| `FAN_VALUE` / `TARGET_ZONE` | 15 / (1,15) | landmark 配对 |
| 频率量化 | FFT bin（≈3.9Hz） | 频率哈希（dejavu 一致，RISKS R4） |
| `OFFSET_BIN` | 1 帧(32ms) | offset 直方图 bin（≤50ms，RISKS R2） |
| `CONFIDENCE_THRESHOLD` | 0.10 | 识别置信阈值 |
| `MIN_ALIGN` | 5 | 最少对齐指纹数 |
| `UNIQUENESS_MARGIN` | 1.5 | 得票唯一性边际 |
| `MATCH_WINDOW` | 2.0s | 匹配滚动窗口 |
| `REALIGN_INTERVAL` | 2.0s | 重对齐周期 |
| `RATE_MIN_SPAN` / `RATE_MIN` / `RATE_MAX` / `RATE_EMA` | 10s / 0.85 / 1.15 / 0.10 | rate 估计 |
| `CORRECT_SMALL` / `CORRECT_LARGE` | 0.10s / 1.0s | 三级校正阈值 |
| `LOW_ENERGY_RMS` | 0.005 | 低能量 RMS 阈值 |
| `PAUSE_MIN_ENERGY_SEC` | 1.0s | 连续低能量触发暂停时长 |
| `LOST_FAIL_STREAK` | 3 | 连续匹配失败触发失锁次数 |

## 6. 不改变需求的原则

本契约不新增打分、点歌、手机端、云端功能；涉及指标口径的阈值（R1 前奏起播、R3/R8 Panako 切换与误识率）已在队长拍板口径下落地，最终口径以人类确认为准。
