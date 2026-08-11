# FVM payload and physical-structure diagnostics

`scripts/fvm_payload_structure_diagnostics.py` is an offline paired analyzer for a locally encoded FVM source MP4 and a platform-transcoded MP4. It measures RaptorQ payload structure, the canonical transport/RS/interleave/matrix representation, and pre-RS raw physical errors. It does not change or participate in production encoding, decoding, FEC, or recovery.

```powershell
python .\scripts\fvm_payload_structure_diagnostics.py SOURCE.mp4 PLATFORM.mp4 OUTPUT_DIR `
  --original-file ORIGINAL_FILE `
  --boundary-summary BOUNDARY_SUMMARY.json `
  --window-symbols 8
```

The source video is decoded through the real 6 px, RS, transport, and FOV packet path. Its recovered packet and embedded transport index are the offline packet truth. The tool then uses the deterministic production transforms to reconstruct the canonical pre-H.264 physical frame. Platform failures are mapped only by conservative transport-neighbor inference; observed frame indices are never a fallback join key.

For each block, `boundary_offset = symbol_id - (K - 1)`. The complete `[-8,+8]` window is retained. `NON_BOUNDARY_SOURCE` is a documented deterministic control sample: source symbols `0`, `K//4`, `K//2`, and `3K//4` per block, excluding `abs(boundary_offset) <= 16`.

When the original file is supplied, the analyzer seeks directly to each selected source chunk. It reports whether the observed source-video RaptorQ payload prefix matches the original bytes. Any remaining bytes are called the **observed payload suffix** until measurements establish their contents; this terminology does not assume universal RaptorQ padding semantics.

`bit_transition_density_1d` expands bytes using the production `BIT_ORDER` and divides adjacent unequal-bit pairs by all adjacent pairs. Canonical physical metrics use the 180×320 pre-render matrix. Raw physical BER is the XOR of that canonical matrix with the thresholded source or platform matrix and is distinct from RS decoder correction counts.

Outputs are `payload_structure_summary.json`, `target_frame_structure.csv`, `offset_structure_summary.csv`, `last_source_suffix_summary.csv`, `rs_codeword_structure.csv`, `rs_codeword_segment_map.csv`, `lane_physical_mapping.csv`, `physical_tile_structure.csv`, `raw_physical_error_frames.csv`, `raw_physical_error_codewords.csv`, `raw_physical_error_tiles.csv`, and `failure_structure_attribution.csv`. Payload bytes are omitted unless `--dump-target-bytes` is explicitly supplied.

This is a single-rendition causal-localization aid. Entropy, packet allocation, raw BER, codeword, lane, or tile associations do not by themselves prove a universal codec or protocol cause. Production `fvm_video2file.py` never depends on source video, original-file truth, expected physical frames, or this analyzer.
