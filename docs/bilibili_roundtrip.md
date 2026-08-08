# Bilibili 自动回环验证

`scripts/bilibili_roundtrip.py` 用于把一个 FOV `source.mp4` 通过哔哩哔哩开放平台投稿，轮询稿件状态直到开放浏览，再下载公开后的平台转码版本，并调用 `video2file.py` 做最终解码/SHA256 验证。

## 能力边界

- **上传与稿件状态查询**：使用哔哩哔哩开放平台文档化的 Open API，包括上传预处理、视频文件上传、封面上传、稿件提交和单稿件查询。
- **下载平台转码版本**：开放平台当前没有文档化的“下载稿件视频文件”接口，因此脚本使用 `yt-dlp` 下载公开播放版本。这一步不是哔哩哔哩 Open API，可能受登录、地区、账号权益或平台反滥用策略影响。
- 只应对自己拥有或获得授权的视频/账号使用该脚本，并遵守哔哩哔哩当前开放平台协议和社区规则。FOV 实验稿件应透明说明用途，不用于规避平台审核或内容规则。

## 前置条件

1. 在哔哩哔哩开放平台完成开发者入驻并创建应用。
2. 为应用申请视频稿件管理相关权限，并让目标 UP 主账号完成授权。
3. 获取应用的 `client_id`、`app_secret` 和该 UP 主的 `access_token`。
4. 确认一个该应用可投稿的分区 `tid`。
5. 安装依赖：

```powershell
python -m pip install -r requirements.txt
```

系统 `PATH` 还需要 `ffprobe`。FOV 自身编码/解码仍需要 FFmpeg。

## 凭据

推荐只通过环境变量提供，不要把凭据写入仓库：

```powershell
$env:BILI_CLIENT_ID="你的 client_id"
$env:BILI_APP_SECRET="你的 app_secret"
$env:BILI_ACCESS_TOKEN="授权 UP 主的 access_token"
$env:BILI_TID="投稿分区 id"
```

也可以用命令行参数 `--client-id`、`--app-secret`、`--access-token` 和 `--tid` 覆盖。

## 基本使用

例如验证一个 1080p / 520B 的 FOV source：

```powershell
python .\scripts\bilibili_roundtrip.py `
  .\runs\fov_1080_s520.mp4 `
  --tid 你的分区ID `
  --cookies-from-browser edge
```

脚本按以下流程执行：

```text
source.mp4
  -> Open API 上传预处理
  -> 视频上传（<=100 MiB 单文件；更大则 8 MiB 分片）
  -> 自动从首帧生成封面并上传
  -> 提交稿件
  -> 轮询 /arcopen/fn/archive/view
  -> state=0 / 开放浏览
  -> yt-dlp 下载公开播放版本（默认优先 H.264、<=720p）
  -> ffprobe 记录平台视频信息
  -> video2file.py 解码
  -> RaptorQ + SHA256 验证
```

运行目录示例：

```text
runs/bilibili-YYYYMMDD-HHMMSS/
  cover.jpg
  platform.mp4
  download_attempt_1.log
  decode.log
  result.json
  recovered/
```

`result.json` 会保存 BV/resource id、审核状态历史、公开 URL、下载后 `ffprobe` 信息和最终解码结果，不保存 `client_id`、`app_secret` 或 `access_token`。

## 下载格式

默认下载选择器：

```text
bestvideo[height<=720][vcodec^=avc1]/bestvideo[height<=720]/best[height<=720]/best
```

这是为了优先拿到当前实验最关心的 720p H.264 平台 rendition。可以覆盖：

```powershell
python .\scripts\bilibili_roundtrip.py .\runs\source.mp4 `
  --tid 123 `
  --download-format "bestvideo[height=720]/bestvideo" `
  --cookies-from-browser chrome
```

`yt-dlp` 对 Bilibili 可能遇到 403/412、登录或高画质权限限制。脚本默认重试 5 次；如果匿名下载失败，可通过 `--cookies-from-browser chrome|edge|firefox` 或 `--cookies cookies.txt` 使用你自己的浏览器登录态。Open Platform 的 `access_token` 与播放页 Cookie 是两套不同的认证体系。

## 审核等待

默认每 30 秒查询一次，最长等待 2 小时：

```powershell
--poll-interval 30
--approval-timeout 7200
```

当接口返回 `state=0` / `开放浏览` 时开始下载；如果返回明确的退回/失败/删除/锁定状态或 `reject_reason`，脚本立即失败并把最后状态写入 `result.json`。其它未识别的处理中状态继续等待，避免把正常审核中的负状态码误判为失败。

开放平台还提供 `video_open` / `video_fail` WebHook；对于本地一次性实验，轮询无需公网回调地址，因此实现更简单。若以后把回环测试部署为长期服务，可以再改成 WebHook 驱动。

## 可选清理

成功解码后自动删除本次 Bilibili 稿件：

```powershell
--delete-after
```

默认**不会**删除，避免脚本在实验数据尚未确认时自动修改账号内容。

## Open API 文件限制

开放平台文档当前给出的主要视频约束包括：文件最大 4 GB、时长小于 5 小时、推荐 MP4/FLV、峰值码率最大 60 Mbps、分辨率最大 4096×4096、帧率最大 120 fps、YUV420；超过 100 MiB 需走分片上传。脚本会自动在单文件和 8 MiB 分片流程之间选择。
