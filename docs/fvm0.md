# FVM0 — FOV Visual Matrix v0

FVM0 是研究和测量工具，不是正式 FOV 编码格式。它测量 Bilibili 1080p 数字视频信道对全屏黑白矩阵的裸误码率（BER）。

默认 1920×1080、30 FPS、cell size 8 时：240×135 cells，即 32,400 bit / 4,050 byte 每帧，raw probe rate 为 121,500 byte/s（约 0.972 Mbit/s）。这不是最终 FOV 有效吞吐。

FVM0 没有 QR、Base64、CRC、ECC、RaptorQ、header、同步、pilot、finder、metadata 或数据重复。整块画面从 `(0,0)` 开始全部用于数据，row-major 排列。bit 0 是黑色，bit 1 是白色。

## 生成

```powershell
python scripts/fvm0_encode.py runs/fvm0-8px.mp4
```

这会生成同名 sidecar manifest。新协议使用 `PCG64_RAW_LSB_V1`：逐帧调用 `PCG64.random_raw()`，将每个 uint64 按 LSB-first 展开。旧 sidecar 的 `PCG64` 保持原有 `Generator.integers()` 解释，绝不静默改释义；视频内部不保存任何 metadata。

## 分析

```powershell
python scripts/fvm0_decode.py `
  runs/fvm0-8px.mp4 `
  runs/fvm0-8px.manifest.json `
  --output-dir runs/fvm0-local-result
```

分析器只接受 manifest 完全匹配的视频分辨率，不 resize、纠偏或自适应阈值。每个 cell 求平均亮度，`mean_luma >= 128` 判为 1，否则判为 0。

输出包括 `fvm0_results.json`、每帧 `fvm0_frames.csv`、空间错误热图和亮度直方图。FER 只是“该帧是否有任意 bit 错误”的统计，不代表帧级解码成功或失败。

FVM0 没有 frame ID；若实际帧数与 manifest 不同，脚本只比较共同帧数并警告：插帧或删帧之后的 index-based BER 不可靠，不会自动对齐。

## 推荐 Bilibili 实验

1. 生成约 12,000 帧（400 秒）的 FVM0 视频并本地分析。
2. 上传到自己的测试稿件，等待 1920×1080 rendition。
3. 使用现有 cookies/yt-dlp 工作流明确下载 1080p rendition。
4. 以同一 manifest 运行 `fvm0_decode.py`。
5. 比较本地与平台的 BER、FER、错误 burst、空间热图和亮度分布。

FVM0 不自动化批量上传，也不处理平台风控或限制规避。
## GOP phase diagnostics

Use an explicitly supplied diagnostic period to inspect BER by `frame_index % gop_size` during decoding:

```powershell
python scripts/fvm0_decode.py VIDEO.mp4 MANIFEST.json --output-dir RESULT --gop-size 150
```

Existing decode results can be re-analysed without decoding the MP4 again:

```powershell
python scripts/fvm0_gop_analyze.py runs/fvm0-5px-1200-bilibili-noLogo --gop-size 150
```

This produces `fvm0_gop_phase.csv`, `fvm0_gop_phase.json`, and `fvm0_gop_phase_ber.png`. The period is an experimental diagnostic assumption, not an FVM protocol constant or proof of the video GOP size. In particular, a low phase-0 BER does not establish that phase 0 is an I-frame; use `ffprobe` frame metadata to verify keyframes.
