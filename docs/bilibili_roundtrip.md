# Bilibili 自动回环验证

`scripts/bilibili_roundtrip.py` 用于把一个 FOV `source.mp4` 通过 `biliup` 以普通 B 站账号投稿，等待稿件进入 `--pubed`，再用 `yt-dlp` 下载公开播放版本，并调用 `video2file.py` 完成 QR / RaptorQ / SHA256 验证。

> `biliup` 使用普通账号的客户端/Web 登录态和非公开投稿接口，不是哔哩哔哩开放平台 Open API。接口可能随平台变化而失效；只应用于自己拥有或已获授权的账号和视频，并遵守平台规则。

## 前置条件

1. FOV Python 环境已安装：

```powershell
python -m pip install -r requirements.txt
```

2. 系统 `PATH` 中有 FFmpeg / `ffprobe`。
3. 单独安装 `biliup` Windows CLI，并确认：

```powershell
E:\Programs\bbup-app\binaries\biliup.exe --version
```

当前自动化按 `biliup-cli 1.2.x` 的 CLI 形态实现。

4. 首次人工登录一次普通 B 站账号：

```powershell
E:\Programs\bbup-app\binaries\biliup.exe `
  -u E:\Programs\bbup-app\cookies.json `
  login
```

推荐扫码登录。`cookies.json` 是账号凭据，不要提交到 Git。

## 基本使用

例如上传一个 1080p / 500B 的 FOV 视频：

```powershell
python .\scripts\bilibili_roundtrip.py `
  .\runs\fov_1080_s500.mp4 `
  --biliup "E:\Programs\bbup-app\binaries\biliup.exe" `
  --biliup-cookies "E:\Programs\bbup-app\cookies.json" `
  --tid 171
```

也可以把固定配置放到环境变量：

```powershell
$env:BILIUP_EXE="E:\Programs\bbup-app\binaries\biliup.exe"
$env:BILIUP_COOKIES="E:\Programs\bbup-app\cookies.json"
$env:BILI_TID="171"

python .\scripts\bilibili_roundtrip.py .\runs\fov_1080_s500.mp4
```

## 自动流程

```text
source.mp4
  -> biliup upload
  -> 从上传日志直接解析 aid / BVID
     -> 若日志格式变化，退化为唯一标题 + biliup list 反查 BVID
  -> 每 30 秒轮询：
       biliup list --pubed
       biliup list --not-pubed
  -> BVID 出现在 --pubed：审核/发布完成
  -> BVID 出现在 --not-pubed：立即失败
  -> biliup show <BVID> 保存稿件详情
  -> yt-dlp 下载公开播放 rendition
  -> ffprobe 记录平台编码信息
  -> video2file.py
  -> QR / CRC32 / RaptorQ / SHA256
```

上传默认使用与手工验证一致的参数：

```text
--submit app
--copyright 1
--limit 3
--tag "FOV,二维码,测试"
--desc "FOV 视频信道自动化实验"
```

不传 `--title` 时，脚本生成包含时间戳和源视频文件名的唯一标题，并限制在 80 字符内。

可覆盖：

```powershell
python .\scripts\bilibili_roundtrip.py .\runs\source.mp4 `
  --biliup "E:\Programs\bbup-app\binaries\biliup.exe" `
  --biliup-cookies "E:\Programs\bbup-app\cookies.json" `
  --tid 171 `
  --title "FOV 自动回环测试 500B" `
  --tag "FOV,二维码,测试" `
  --desc "FOV 视频信道自动化实验" `
  --submit app
```

## BVID 获取

当前 `biliup 1.2.x` 投稿成功日志会出现类似：

```text
ResponseData { code: 0, data: Some(Object {
  "aid": Number(117059015018441),
  "bvid": String("BV1wFug64EZg")
}), message: "OK" }
APP接口投稿成功
```

脚本优先直接解析该 `bvid`。如果未来日志格式变化导致解析失败，则使用本次唯一标题依次查询：

```text
biliup list --is-pubing
biliup list --pubed
biliup list --not-pubed
```

以标题反查 BVID。

## 审核等待

默认：

```text
poll interval     = 30 s
approval timeout  = 7200 s
status max pages  = 3
```

可调整：

```powershell
--poll-interval 30
--approval-timeout 7200
--status-max-pages 3
```

判断只依赖 BVID 是否出现在对应列表中，不依赖 `biliup list` 第三列文本，因为处理中稿件的第三列可能显示简介而不是状态。

所有状态查询原始输出追加保存到：

```text
review.log
```

同时 `result.json` 中保存结构化 `review_history`。

## 下载登录态

默认下载格式：

```text
bestvideo[height<=720][vcodec^=avc1]/bestvideo[height<=720]/best[height<=720]/best
```

即优先拿实验当前关注的 `<=720p H.264` 平台 rendition。

如果没有显式指定下载 Cookie，脚本会把 `biliup cookies.json` 中的 `cookie_info.cookies` 临时转换成 Netscape cookie 文件交给 `yt-dlp`，下载结束后立即删除临时文件。因此通常只需要登录一次 biliup。

如需覆盖：

```powershell
# 直接使用浏览器登录态
--cookies-from-browser edge

# 或显式 Netscape cookies.txt
--yt-dlp-cookies .\cookies.txt

# 或强制匿名下载
--anonymous-download
```

`yt-dlp` 默认重试 5 次，审核刚通过但目标 rendition 尚未就绪时也会等待重试。

## 输出

每次运行生成：

```text
runs/bilibili-YYYYMMDD-HHMMSS/
  upload.log
  review.log
  show.log
  download_attempt_1.log
  platform.mp4
  decode.log
  result.json
  recovered/
```

`result.json` 包含：

```text
source video / bytes
标题 / tid / submit
上传耗时 / 返回码
aid / bvid / share_url
review_history
published_at
download_auth
平台视频大小
ffprobe 信息
QR decoded / failed
failed frame indices
每个 block 的 K / N / received source / received repair / decoded_at_frame
CRC failed
Original / Recovered SHA256
最终 decode_ok
```

账号 Cookie 内容不会写进 `result.json`。

## 测试

```powershell
python -m compileall fov.py file2video.py video2file.py scripts tests
python -m pytest -q
```

单元测试不调用真实 B 站，覆盖 `biliup 1.2.x` 投稿日志解析、稿件列表解析、上传命令构造、biliup Cookie 到临时 Netscape Cookie 的转换，以及真实 decoder 诊断日志解析。
