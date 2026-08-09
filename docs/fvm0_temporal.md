# FVM0-T temporal transition probe

FVM0-T is a controlled physical-channel probe, not FVM1 or a payload transport. Each block starts with an independent absolute random anchor; later frames flip exactly `floor(cells * ratio + 0.5)` cells. Schedules are PCG64 deterministic, balanced per repeat, and stored in the manifest.

```powershell
python scripts/fvm0_temporal_encode.py runs/fvm0-t-5px.mp4 --cell-size 5 --block-size 150 --ratios 0.1,0.2,0.3,0.4,0.5 --repeats 5 --warmup-blocks 1
python scripts/fvm0_temporal_decode.py runs/fvm0-t-5px.mp4 runs/fvm0-t-5px.manifest.json --output-dir runs/fvm0-t-local
python scripts/fvm0_temporal_analyze.py runs/fvm0-t-local
```

The 50% condition is a transition-density control, not equivalent to FVM0_RAW. A 150-frame block is an experiment setting, not a permanent platform GOP claim. Upload manually using an existing workflow; this tool never uploads to Bilibili or carries real payload coding.
