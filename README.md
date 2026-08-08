# FOV — File Over Video

FOV 是一个实验性 PoC：将任意二进制文件转为由二维码帧组成的 H.264 视频，并在有损视频通道后恢复原文件。

```text
文件 → RaptorQ → 独立 symbol → QR 帧 → H.264 视频
     → 有损视频通道 → ZXing-C++ → CRC32 → RaptorQ → SHA256 → 文件
```

这不是唯一备份方案；应始终保留原始文件和独立校验副本。

## 环境

需要 Python 3.11、Windows 11 PowerShell，以及系统 `PATH` 中的 FFmpeg。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pyraptorq==0.1.7` 的实际 Python API 是 `Encoder(data, symbol_size, engine)` 和 `Decoder(K, symbol_size, data_size, engine)`。FOV 显式复用一个 `RaptorQCppEngine`。该 wheel 在 Windows 的默认 loader 会将 `AMD64` 与 bundled `x86_64` DLL 名称混淆；FOV 仅在 pyraptorq 包的 `distlib/windows` 中按当前架构寻找 DLL。若需要 MinGW runtime，还会加入存在的 `C:\Program Files\Git\mingw64\bin`，并保留 `os.add_dll_directory()` 返回的 handle，避免搜索路径提前失效。

## 使用

```powershell
python file2video.py test.zip data.mp4
python video2file.py data.mp4
```

```powershell
python file2video.py test.zip data.mp4 `
  --symbol-size 200 `
  --repair 0.4 `
  --fps 30 `
  --width 1280 `
  --height 720
```

QR 视觉基线固定为 `box_size=8`、`border=4`、ECC=M，视频默认 1280×720、30 FPS、H.264 CRF 15。packet 过大导致 QR 放不进画面时会直接报错，不会静默缩小 QR module。

## FOV v1 packet

所有整数均为 big-endian/network byte order。CRC32 覆盖 `MAGIC` 至 `PAYLOAD`，CRC 失败的 packet 被丢弃并当作 erasure，绝不进入 RaptorQ。

```text
SYMBOL (TYPE=1)
MAGIC(4, "FOV1") | TYPE(u8) | FILE_ID(8) | BLOCK_ID(u16) |
SYMBOL_ID(u32) | PAYLOAD_LEN(u16) | PAYLOAD | CRC32(u32)

META (TYPE=0)
MAGIC(4, "FOV1") | TYPE(u8) | FILE_ID(8) | PAYLOAD_LEN(u16) |
UTF-8 JSON payload | CRC32(u32)
```

`FILE_ID` 是原文件 SHA256 的前 8 bytes。每个 binary packet 先 Base64，再编码为 QR。

### 固定大小 META schema

META 不保存随文件 block 数线性增长的 `blocks` 数组：

```json
{
  "version": 1,
  "filename": "example.zip",
  "original_size": 399825,
  "sha256": "...64 hex chars...",
  "encoded_size": 399825,
  "compression": "none",
  "symbol_size": 200,
  "repair_ratio": 0.20,
  "block_size": 8388608,
  "block_count": 1
}
```

对 block `i`，接收端完全由 META 推导：

```text
data_size = min(block_size, original_size - i * block_size)
K = ceil(data_size / symbol_size)
N = ceil(K * (1 + repair_ratio))
```

因此最后一个 block 的真实大小正确，META 大小基本不随 block_count 增长。FOV 验证 `block_count == ceil(original_size / block_size)`、`encoded_size == original_size`（当前仅支持 `compression=none`）以及所有 RaptorQ 和 packet 边界。

## 大文件内存模型

编码端使用流式 SHA256 和 `Path.stat()`，不调用 `read_bytes()`。文件按 `INTERLEAVE_WINDOW=4` 个 source block 分窗：

```text
B0-S0, B1-S0, B2-S0, B3-S0, B0-S1, ...
释放 B0~B3，再处理 B4~B7
```

每个 symbol 在即将生成 QR 帧时才调用 `encoder.gen_symbol()`；不会将所有 RaptorQ symbols 放入列表。编码内存近似为 `O(INTERLEAVE_WINDOW × BLOCK_SIZE)`，而不是整个文件大小。RaptorQ engine 在整个编码/解码进程中复用。

解码扫描不要求 META 先到，但也不会保存整段视频的 symbol payload。META 到达前，`PreMetaBuffer` 按 oldest-first 暂存最多 4,096 个 symbol、16 MiB payload、4 个 file ID；超出的旧 symbol 被淘汰并作为 RaptorQ erasure。META 到达后，匹配的缓存 symbol 立即回灌，其余缓存被释放。

之后每个合法 SYMBOL 直接送入该 block 的 native RaptorQ decoder。block 首次收到 symbol 时才创建 decoder；重复检测使用 `bytearray` 位图而非 `set[int]`。block 恢复后立即按 `block_id * block_size` 随机写入临时文件，释放 decoder 与位图。因而 decoder 内存接近 `O(active decoders + bounded pre-META cache)`，不再是 `O(all received symbol payloads)`。多个合法文件或同一 file_id 的冲突 META 都会明确失败；所有 block 恢复且 SHA256 一致后才会原子性写出恢复文件。

## 默认参数与约束

| 参数 | 默认值 |
| --- | --- |
| 视频 | 1280×720，30 FPS，H.264 CRF 15 |
| source symbol | 200 bytes |
| repair ratio | 20% |
| source block | 8 MiB |
| interleave window | 4 blocks |
| META | 首尾各 10 帧，之后每 300 个 SYMBOL 插入一帧 |

RaptorQ source symbol count 最大为 56,403。这是当前 pyraptorq 所封装 cpp-raptorq 的 RFC 参数表上限；超过时 FOV 在 Python 层拒绝参数，不把输入交给 native DLL。`BLOCK_ID` 最大 65,535，`SYMBOL_ID` 最大 uint32。

QR ECC 处理帧内错误；CRC32 将损坏 packet 转为 erasure；RaptorQ 处理帧间丢失；SHA256 提供文件级最终确认。每个 symbol 只发送一次，不再使用旧 `REPEAT=3`。

## 测试

```powershell
python -m compileall fov.py file2video.py video2file.py tests
python -m pytest -q
```

单元测试不依赖 FFmpeg 或视频平台，覆盖 packet、CRC、RaptorQ 随机丢失、重复 symbol、多 block、固定 META、布局推导、metadata 边界、META 晚到、冲突、多文件、惰性生成和 Windows DLL handle。
