# FVM real-platform adaptive cliff search

`scripts/fvm_platform_cliff.py` performs a bounded, serial search using real 1920x1080 platform renditions as the PASS/FAIL oracle. It imports the existing 80/176 CRF30 failure, never uploads that case again, and allows at most one upload per new case and eight new uploads globally.

The search begins at 64/192 CRF30. A PASS opens a same-CRF level bisection; a FAIL triggers the prescribed 48/208 CRF30 and 80/176 CRF27 orthogonal diagnostics. Bisection is used only after an observed same-axis PASS/FAIL bracket. Uploads are serial and separated by at least 120 seconds. Review and rendition polling remain at least 60 seconds.

Without `--allow-upload`, the CLI only verifies and imports the known failure. A formal invocation is:

```powershell
python .\scripts\fvm_platform_cliff.py .\runs\fvm-platform-cliff `
  --input .\runs\fvm-codec-sweep\input\codec-sweep-32MiB.bin `
  --sweep-root .\runs\fvm-codec-sweep `
  --known-fail-run .\runs\fvm-platform-validation-20260812-120505 `
  --biliup "E:\Programs\bbup-app\binaries\biliup.exe" `
  --biliup-cookies "E:\Programs\bbup-app\cookies.json" --tid 171 `
  --allow-upload --max-new-uploads 8 --upload-cooldown 120 --poll-interval 60 --resume
```

`search_state.json` is atomically replaced after state changes. A case with a stored validation run/BVID is resumed and is never uploaded again. CAPTCHA, risk control, authentication failure, rate limiting, or submission rejection stop the complete search without bypass or upload retry.

Search artifacts, platform downloads, recovered files, and reports remain under ignored `runs/`. Observed cliffs are sample-specific and do not change production defaults.
