# FVM0-T temporal transition probe

FVM0-T is a controlled physical-channel probe, not FVM1 or a payload transport. Each block starts with an independent absolute random anchor; later frames flip exactly `floor(cells * ratio + 0.5)` cells. Schedules are PCG64 deterministic, balanced per repeat, and stored in the manifest.

```powershell
python scripts/fvm0_temporal_encode.py runs/fvm0-t-5px.mp4 --cell-size 5 --block-size 150 --ratios 0.1,0.2,0.3,0.4,0.5 --repeats 5 --warmup-blocks 1
python scripts/fvm0_temporal_decode.py runs/fvm0-t-5px.mp4 runs/fvm0-t-5px.manifest.json --output-dir runs/fvm0-t-local
python scripts/fvm0_temporal_analyze.py runs/fvm0-t-local
```

The 50% condition is a transition-density control, not equivalent to FVM0_RAW. A 150-frame block is an experiment setting, not a permanent platform GOP claim. Upload manually using an existing workflow; this tool never uploads to Bilibili or carries real payload coding.

## Transition-mask channel

For adjacent non-anchor frames within one block, the expected mask is `expected_previous XOR expected_current`; the observed mask is `actual_previous XOR actual_current`. Anchors reset both histories and are never compared with the previous block. The decoder records TP/FN/FP/TN, mask BER `(FN+FP)/N`, recall, precision, F1, missed-flip rate, false-flip rate, and transition-count bias. Direction is measured separately: a detected flip can be mask-correct while its observed `0->1`/`1->0` direction is wrong.

Ratio aggregates use total counts, never the mean of per-frame rates. The analyzer produces `fvm0_temporal_transition_mask_ber_vs_ratio.png`, `fvm0_temporal_transition_error_types_vs_ratio.png`, `fvm0_temporal_transition_precision_recall.png`, and `fvm0_temporal_transition_direction.png`. Older frame CSV files lack these fields; re-run the updated decoder into a new result directory before analysis.

To preserve the previous absolute experiment for comparison, decode an existing platform rendition into a new directory:

```powershell
python .\scripts\fvm0_temporal_decode.py .\runs\fvm0-t-5px-bilibili-1080.mp4 .\runs\fvm0-t-5px.manifest.json --output-dir .\runs\fvm0-t-5px-bilibili-transition
python .\scripts\fvm0_temporal_analyze.py .\runs\fvm0-t-5px-bilibili-transition --ffprobe-video .\runs\fvm0-t-5px-bilibili-1080.mp4
```

The default probe is 1920x1080 at 30 FPS with 5px cells (384x216 = 82,944 cells), ratios 10/20/30/40/50%, one warm-up block, and five balanced shuffled repeats: 3,900 frames total. Each block begins with an independent anchor. Ratio BER excludes warm-up and anchor frames; changed and unchanged BER use their respective total bit denominators. Luminance p1/p99 is a distribution diagnostic. Packet size and ffprobe-confirmed keyframe metadata are correlation diagnostics only. The constant-weight rate is `log2(C(N,m))` adjusted by FPS and `(block_size-1)/block_size`; it excludes headers, sync, CRC, FEC, metadata, and rank/unrank implementation.

Before manual platform upload, run local decode and analysis and require all main-ratio observed BER values to be zero. Disable platform watermarks, wait for the 1920x1080 rendition, download that rendition, decode it, then run:

```powershell
python .\scripts\fvm0_temporal_analyze.py RESULT_DIR --ffprobe-video PLATFORM_1080P.mp4
```
