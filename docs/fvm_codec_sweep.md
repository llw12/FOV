# FVM codec efficiency sweep

This experiment measures how source H.264 CRF and symmetric binary grayscale levels affect FVM file-video size and recovery. It does not alter the production encoder, decoder, wire format, RS coding, RaptorQ behavior, packet schedule, geometry, or defaults.

The fixed matrix is four levels (`0/255`, `48/208`, `64/192`, `80/176`) by five CRFs (`18`, `21`, `24`, `27`, `30`). Every source uses 1920×1080 at 30 fps, 6 px cells, 28×RS(255,239), 6400-byte symbols, 8 MiB blocks, 3% RaptorQ repair, interleave window 4, x264 `slow`, and yuv420p. The decoder threshold remains 128.

Run locally:

```powershell
python .\scripts\fvm_codec_sweep.py .\runs\fvm-codec-sweep `
  --proxy-reference .\runs\fvm-500MiB-r03\fvm-500MiB-r03-platform-1080p.mp4 `
  --resume
```

The tool first runs a 1 MiB `64/192`, CRF 24 preflight. It then creates a deterministic 32 MiB high-entropy input and runs all selected cases independently. Each source and proxy is decoded through `fvm_video2file.decode()` using only the video and output directory; offline raw BER is a separate oracle diagnostic and never assists recovery.

The proxy is one frozen local x264 bitrate/VBV transcode calibrated from the observed average bitrate of the supplied reference rendition. **PROXY ≠ PLATFORM**: it is a repeatable stress proxy, not a model or guarantee of any platform encoder. All cases use the same proxy settings, and those settings must not be adjusted after the sweep begins.

Formal artifacts live below `runs/fvm-codec-sweep` and remain ignored by Git. Source MP4s and case diagnostics are retained. Proxy MP4s and recovered files are deleted by default after diagnostics; pass `--keep-proxy` or `--keep-recovered` when those artifacts are needed. A deleted proxy intentionally makes the strict `--resume` check rerun that case because a COMPLETE resume requires both videos to exist.

Summary outputs include JSON/CSV results, a recoverable Pareto frontier, the experiment manifest, reference observations, and `REPORT.md`. Pareto X is source MP4 bytes per input byte; Y is proxy RS-failed frames per observed frame. Only source- and proxy-SHA-exact cases enter the recoverable frontier.
