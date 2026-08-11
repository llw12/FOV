# FVM offline boundary diagnostics

This tool performs paired offline analysis of an encoder-produced source FVM video and a platform-transcoded rendition. It measures whether RS failures and correction burden align with the FOV source-to-repair symbol boundary, codeword lanes, packet scheduling, META placement, or codec-frame metadata.

```powershell
python .\scripts\fvm_boundary_diagnostics.py SOURCE.mp4 PLATFORM.mp4 OUTPUT_DIR --window-symbols 8
```

The source video is decoded through the real 6 px matrix, RS, transport, and FOV packet layers. Its successfully recovered transport indices and packets are the offline ground truth; the analyzer does not reconstruct an expected packet stream from formulas or require an original file, manifest, seed, or sidecar. Platform frames join source truth by recovered embedded transport index. An RS-failed platform frame is attributed only when the production-safe neighboring-index inference is unambiguous. Observed video frame indices are never used as a fallback packet mapping.

For a block with `K` source symbols, `boundary_offset = symbol_id - (K - 1)`. Offset 0 is the last source symbol, +1 is the first repair symbol, and negative offsets precede the boundary. The default report covers offsets -8 through +8 and compares them with source symbols whose absolute offset exceeds 16.

Outputs are `boundary_summary.json`, `boundary_offsets.csv`, `boundary_frames.csv`, `rs_failure_attribution.csv`, `codeword_boundary_summary.csv`, and `failure_observed_windows.csv`. Codec metadata comes from optional `ffprobe -show_frames`; core RS and boundary analysis remains available if ffprobe fails. A full paired-frame CSV is emitted only with `--full-frame-csv`.

For offline payload, RS-lane, physical-matrix, and raw-BER localization, see [FVM payload and physical-structure diagnostics](fvm_payload_structure_diagnostics.md).

This is an offline paired analyzer, not the production decoder. `fvm_video2file.py` continues to recover files without source video or expected packets. Associations from one platform rendition are diagnostic evidence, not universal codec or channel guarantees.
