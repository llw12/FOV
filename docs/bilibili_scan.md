# Bilibili 真实信道参数扫描

`scripts/bilibili_scan.py` 把 `scan_params.py` 的参数组合思想扩展到真实 Bilibili 信道：每个 case 先生成 FOV `source.mp4` 并做本地解码校验，本地通过后才使用 `biliup` 投稿；稿件进入 `--pubed` 后用 `yt-dlp` 下载平台版本，再运行 `video2file.py` 做 QR / RaptorQ / SHA256 验证。

> 每个成功上传的 case 都会产生一个真实 Bilibili 稿件。建议先用 `--plan-only` 或少量 `--case` 检查计划，不要直接对大量组合运行。只操作自己拥有或已获授权的账号和视频，并遵守平台规则。

## 前置条件

先确认单次回环已经可用：

```powershell
python .\scripts\bilibili_roundtrip.py `
  .\runs\fov_1080_s500.mp4 `
  --biliup "E:\Programs\bbup-app\binaries\biliup.exe" `
  --biliup-cookies "E:\Programs\bbup-app\cookies.json" `
  --tid 171
```

并安装 FOV 依赖，确保 `ffmpeg` / `ffprobe` 在 `PATH`：

```powershell
python -m pip install -r requirements.txt
```

推荐把固定路径放到环境变量：

```powershell
$env:BILIUP_EXE="E:\Programs\bbup-app\binaries\biliup.exe"
$env:BILIUP_COOKIES="E:\Programs\bbup-app\cookies.json"
$env:BILI_TID="171"
```

## 默认扫描

为了避免默认一次发布太多真实稿件，`platform` preset 只包含当前已验证附近的两个 1080p case：

```text
1920x1080:500:0.20:30
1920x1080:520:0.20:30
```

先只查看计划：

```powershell
python .\scripts\bilibili_scan.py --plan-only
```

执行真实扫描：

```powershell
python .\scripts\bilibili_scan.py --input .\test.zip
```

完整流程对每个 case 都是：

```text
input file
  -> file2video.py
  -> source.mp4
  -> video2file.py 本地 SHA256 校验
  -> biliup upload
  -> 解析 aid / BVID
  -> biliup list --pubed / --not-pubed 轮询
  -> yt-dlp 下载公开 rendition
  -> ffprobe
  -> video2file.py
  -> QR / CRC32 / RaptorQ / SHA256
```

本地校验失败的 case 不会上传。

## 自定义参数组合

`--case` 与 `scan_params.py` 使用相同格式：

```text
WIDTHxHEIGHT:SYMBOL[:REPAIR[:FPS]]
```

例如直接比较 500B / 505B / 510B / 520B：

```powershell
python .\scripts\bilibili_scan.py `
  --input .\test.zip `
  --case 1920x1080:500:0.20 `
  --case 1920x1080:505:0.20 `
  --case 1920x1080:510:0.20 `
  --case 1920x1080:520:0.20
```

比较同一 symbol size 的不同 RaptorQ repair ratio：

```powershell
python .\scripts\bilibili_scan.py `
  --case 1920x1080:500:0.10 `
  --case 1920x1080:500:0.15 `
  --case 1920x1080:500:0.20
```

也可以让 preset 与多个 repair ratio 做笛卡尔积：

```powershell
python .\scripts\bilibili_scan.py `
  --preset platform `
  --repairs 0.10 0.15 0.20
```

此时计划是 500/520B × 10/15/20% 共 6 个真实稿件。建议先：

```powershell
python .\scripts\bilibili_scan.py `
  --preset platform `
  --repairs 0.10 0.15 0.20 `
  --plan-only
```

## preset

- `platform`（默认）：1920×1080 的 500B、520B，适合当前真实平台阶段。
- `targeted`：复用 `scan_params.py` 的 targeted 基础集合：720p 200/240/280B + 1080p 400/500/600/700B。
- `full`：复用 `scan_params.py` 的完整 resolution × symbol size 组合。

`targeted` / `full` 会产生较多真实投稿，应配合 `--max-cases` 或先 `--plan-only`。

例如只执行前 3 个计划：

```powershell
python .\scripts\bilibili_scan.py `
  --preset targeted `
  --max-cases 3
```

## 审核和错误处理

默认：

```text
poll interval     = 30 s
approval timeout  = 7200 s
status max pages  = 3
case delay        = 5 s
```

扫描默认遇到单个 case 失败后继续下一项，并把失败阶段写入结果。如果希望第一项失败就停止：

```powershell
--stop-on-error
```

审核期间每次轮询都会更新该 case 的：

```text
cases/<case>/result.json
```

因此即使长时间审核过程中被中断，也能看到已经解析出的 BVID 和审核历史。Ctrl+C 时聚合结果也会尽量保存。

## 下载认证

与 `bilibili_roundtrip.py` 一致，默认把 biliup `cookies.json` 临时转换成 Netscape Cookie 给 `yt-dlp`，扫描结束后删除临时文件。

也可以覆盖：

```powershell
--cookies-from-browser edge
--yt-dlp-cookies .\cookies.txt
--anonymous-download
```

默认下载选择器优先 `<=720p` 的 H.264/AVC rendition，可用 `--download-format` 覆盖。

## 输出

每次扫描生成：

```text
runs/bilibili-scan-YYYYMMDD-HHMMSS/
  config.json
  results.json
  results.csv
  summary.md
  cases/
    1920x1080_s500_r20/
      source.mp4
      encode.log
      upload.log
      review.log
      show.log
      result.json
      local/
        decode.log
        recovered/
      platform/
        platform.mp4
        download_attempt_1.log
        decode.log
        recovered/
```

`summary.md` 直接比较每个 case 的：

```text
local PASS/FAIL
有效吞吐 KB/s
BVID
平台 PASS/FAIL
QR erasure
实际收到的 repair symbols
RaptorQ decoded_at_frame
```

`results.json` / `results.csv` 还保存平台 ffprobe、失败帧索引、CRC、每个 block 的 `K/N/received source/received repair`、Original/Recovered SHA256 等完整诊断信息。

## 测试

该扫描脚本的单元测试不投稿、不访问 Bilibili，只测试参数组合、标题、结果指标等纯 helper：

```powershell
python -m compileall fov.py file2video.py video2file.py scripts tests
python -m pytest -q
```
