# FVM real-platform validation

`scripts/fvm_platform_validate.py` validates one already encoded FVM source through one explicitly authorized Bilibili upload. It does not encode a source or change the production wire format, encoder, decoder, RS, RaptorQ, threshold, geometry, or defaults.

Without `--allow-upload`, the command only validates the fixed source/case metadata and runs the production FVM decoder locally:

```powershell
python .\scripts\fvm_platform_validate.py `
  .\runs\fvm-codec-sweep\cases\level-080-176_crf30\source.mp4 `
  --original-file .\runs\fvm-codec-sweep\input\codec-sweep-32MiB.bin `
  --case-result .\runs\fvm-codec-sweep\cases\level-080-176_crf30\case_result.json
```

The sole upload requires explicit opt-in and an existing authorized `biliup` login state:

```powershell
python .\scripts\fvm_platform_validate.py `
  .\runs\fvm-codec-sweep\cases\level-080-176_crf30\source.mp4 `
  --original-file .\runs\fvm-codec-sweep\input\codec-sweep-32MiB.bin `
  --case-result .\runs\fvm-codec-sweep\cases\level-080-176_crf30\case_result.json `
  --allow-upload `
  --biliup "E:\Programs\bbup-app\binaries\biliup.exe" `
  --biliup-cookies "E:\Programs\bbup-app\cookies.json" `
  --tid 171
```

The upload-attempt count is persisted before invoking `biliup upload`; the same run can never upload twice. `--resume-run RUN_DIR` continues only when a BVID was already stored. Login failures, rejection, CAPTCHA, risk control, or missing identity block the run without upload retry.

Review and rendition polling use intervals of at least 60 seconds. Format IDs are discovered dynamically. A 1920×1080 AVC rendition is preferred, otherwise another 1920×1080 video codec is accepted and reported; 720p is never used as a fallback. Temporary Netscape cookies are deleted after download and credential-bearing fields are excluded from JSON and reports.

The platform video is decoded by `fvm_video2file.decode()`. Complete RaptorQ block recovery plus exact recovered SHA-256 is the primary PASS gate. Positional raw BER is calculated only when source/platform frame counts match and production diagnostics report no duplicate or out-of-order transport indices. Otherwise raw BER is marked unavailable rather than aligned by observed-frame fallback.

Generated videos, logs, recovery files, results, and reports remain below ignored `runs/` and are not committed.
