# FOV — File Over Video

FOV 是一个实验性 PoC：把任意二进制文件编码成由二维码帧构成的 H.264 视频，经过普通视频平台的有损转码后，再从下载的视频恢复原始文件。

> 这不是唯一备份方案。请始终保留原文件和独立校验副本。

## 工作链路

```text
文件 → RaptorQ → 独立 encoding symbol → QR 帧 → H.264 视频
     → 有损视频通道 → ZXing-C++ → CRC32 → RaptorQ → SHA256 → 文件
```

- QR 的 ECC-M 处理单帧二维码中的视觉错误。
- CRC32 让已损坏的 packet 在进入纠删码前变为 erasure。
- RaptorQ 负责丢帧/擦除恢复；每个 symbol 只发一帧，不再使用旧方案的 `REPEAT=3`。
- SHA256 是文件级的最终完整性确认，校验失败绝不报告恢复成功。

## 环境

需要 Python 3.11、FFmpeg（系统 `PATH` 中的 `ffmpeg`）和 Windows 11 PowerShell。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`pyraptorq==0.1.7` 在 Windows 上随包提供的 DLL 名称为 `x86_64`，而其默认加载器常按 `AMD64` 查找。FOV 通过同一 pyraptorq 包的 `RaptorQCppEngine` 显式加载该 DLL；没有实现或替代 RaptorQ 算法。该 wheel 还可能依赖 MinGW C++ runtime；代码会识别常见的 Git for Windows 路径 `C:\Program Files\Git\mingw64\bin`，否则会在启动时明确报出 DLL 加载错误。

## 使用

```powershell
python file2video.py test.zip data.mp4
python video2file.py data.mp4
```

可调整可靠性和视频参数：

```powershell
python file2video.py test.zip data.mp4 `
  --symbol-size 200 `
  --repair 0.4 `
  --fps 30 `
  --width 1280 `
  --height 720
```

如果某个 packet 使 QR 超出画面，编码会明确失败并提示减小 `--symbol-size`，不会静默缩小 module。

## FOV v1 默认参数

| 参数 | 默认值 |
| --- | --- |
| 视频 | 1280×720、30 fps、H.264 (`crf 15`) |
| source symbol 大小 | 200 bytes |
| repair ratio | 20% |
| QR | box size 8、border 4、ECC M |
| source block | 8 MiB |
| META | 开头/结尾各 10 帧，之后每 300 个 symbol 插入一次 |

大文件按 8 MiB 拆 block；symbol 发送顺序为 `B0-S0, B1-S0, …, B0-S1, …`，避免连续视频损伤集中毁掉单个 block。

## Binary packet

所有整数使用 big-endian/network byte order，所有 packet 均以 `CRC32(MAGIC…PAYLOAD)` 结尾。

`SYMBOL` (`TYPE=1`):

```text
MAGIC(4, "FOV1") | TYPE(u8) | FILE_ID(8) | BLOCK_ID(u16) |
SYMBOL_ID(u32) | PAYLOAD_LEN(u16) | PAYLOAD | CRC32(u32)
```

`META` (`TYPE=0`):

```text
MAGIC(4, "FOV1") | TYPE(u8) | FILE_ID(8) | PAYLOAD_LEN(u16) |
UTF-8 JSON payload | CRC32(u32)
```

`FILE_ID` 是原始文件 SHA256 的前 8 bytes。META JSON 包括版本、纯文件名、原始/编码尺寸、完整 SHA256、`compression: none`、symbol 参数，以及每一个 block 的真实数据长度、K 和 N。为保持默认 META 在 720p/box size=8 的 QR 内，`blocks` 使用 `[data_size, K, N]` 数组；解码端不信任 filename 中的路径。

## RaptorQ

对每个 block：`K = ceil(data_size / symbol_size)`，`N = ceil(K * (1 + repair_ratio))`，生成 `symbol_id=0..N-1`。

FOV 使用实际 `pyraptorq` 0.1.7 API：`Encoder(data, symbol_size, engine).gen_symbol(id)`；解码使用 `Decoder(K, symbol_size, data_size, engine).add_symbol(id, data)`，在 `may_try_decode()` 后调用 `try_decode()`。符号乱序、重复和丢失均可处理，CRC 错误的包不会加入 decoder。

## 测试

```powershell
python -m pytest -q
python -m compileall fov.py file2video.py video2file.py
```

测试不依赖 FFmpeg 或真实视频平台，覆盖 packet roundtrip、CRC、随机删失和乱序后的 RaptorQ 恢复、重复 symbol、多 block 以及 SHA256。
