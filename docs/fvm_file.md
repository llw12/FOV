# FVM FILE MODE v1

FVM FILE MODE is the first local file-to-video-to-file prototype. It connects the existing FOV1 file protocol and streaming RaptorQ recovery to the already validated 6 px matrix and RS physical layer:

`file → FOV1 META/SYMBOL + RaptorQ → FVM transport → RS(255,239) × 28 → byte-column interleave → 6 px matrix → H.264`

The decoder applies the inverse validation chain and accepts data only after RS decoding, transport CRC/header validation, FOV packet CRC validation, RaptorQ recovery, and final SHA-256 equality. A damaged physical frame that cannot pass those checks is an erasure; outer RaptorQ repair symbols can replace it.

## Fixed v1 profile

- 1920×1080, 30 fps, 6 px cells (320×180 bits)
- 7200 physical bytes per frame
- 28 RS(255,239) codewords: 6692 logical bytes, 7140 coded bytes
- byte-column interleave identical to `FVM0_RS_PROBE`
- 60 reserved bytes per frame
- default FOV symbol payload: 6400 bytes
- default RaptorQ repair ratio: 10%
- existing 8 MiB blocks and four-block temporal interleave

The transport logical frame uses magic `F6F1`, version 1, a uint32 big-endian diagnostic frame index, a uint16 packet length, up to 6677 packet bytes, deterministic SHAKE-256 padding, and a final big-endian CRC32 over the first 6688 bytes. The reserved physical bytes use SHAKE-256 with domain `FVM_FILE_RESERVED_V1|` and the embedded frame index.

Frame indices are diagnostic only. Gaps, duplicates, and out-of-order indices are recorded but never used as expected truth or to help correction. Gap counts include observable leading and internal gaps among successfully recovered indices; unknowable trailing loss is not guessed. FOV block/symbol identifiers provide unordered recovery and symbol deduplication.

## Commands

```powershell
python .\fvm_file2video.py INPUT_FILE OUTPUT.mp4
python .\fvm_video2file.py OUTPUT.mp4 OUTPUT_DIR
```

Encoder options are `--symbol-size 6400 --repair 0.10 --crf 15 --preset medium` by default. The decoder intentionally accepts only the video and output directory. It needs no manifest, original file, seed, or other sidecar. It writes `OUTPUT_DIR/.fvm/fvm_decode_results.json` with physical, transport, packet, RaptorQ, and final file diagnostics. The `.fvm/` directory is a reserved decoder-diagnostics namespace so diagnostic files cannot overwrite recovered user files. If an input filename is itself `.fvm`, the existing output collision policy selects a `recovered_...` filename.

Production diagnostics record RS failure positions in observed-video-frame coordinates, failed codeword counts and indices, neighboring successful transport indices and packet descriptors, and consecutive failure bursts. A transport index is inferred only when bracketing successful indices make the mapping unambiguous. This inference is diagnostic only and is never used for file recovery.

`FVM0_RS_PROBE` remains a separate channel experiment with expected truth and raw BER measurements. FVM FILE MODE is the real file transport and never uses probe truth.
