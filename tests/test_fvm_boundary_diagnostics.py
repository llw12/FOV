from pathlib import Path

import pytest

from fvm_boundary_analysis import (PairedFrameRecord, PlatformFrame, SourceFrame,
                                   aggregate_codewords, aggregate_frame_metrics, classify_symbol,
                                   pair_frames, parse_ffprobe_frames, probe_codec_frames,
                                   spacing_analysis)
from fvm_diagnostics import PacketDiagnostic


def _source(transport: int, symbol: int, k: int = 1311, n: int = 1351) -> SourceFrame:
    fields = classify_symbol(symbol, k, n)
    return SourceFrame(
        observed_frame_index=transport,
        transport_index=transport,
        packet_type="SYMBOL",
        block_id=2,
        symbol_id=symbol,
        source_symbols=k,
        encoded_symbols=n,
        **fields,
    )


def _platform(
    observed: int,
    transport: int | None,
    *,
    failed: bool = False,
    confident: bool = True,
    corrections: tuple[int | None, ...] | None = None,
) -> PlatformFrame:
    values = corrections or tuple(0 for _ in range(28))
    return PlatformFrame(
        observed_frame_index=observed,
        transport_index=transport,
        inference_confident=confident,
        rs_failed=failed,
        codeword_successes=sum(value is not None for value in values),
        codeword_failures=sum(value is None for value in values),
        failed_codeword_indices=tuple(index for index, value in enumerate(values) if value is None),
        corrected_symbols=sum(value or 0 for value in values),
        max_corrections_per_codeword=max((value or 0 for value in values), default=0),
        corrections_per_codeword=values,
        packet_descriptor=None,
    )


def _paired(
    offset: int,
    corrected: int,
    maximum: int,
    *,
    failed: bool = False,
    corrections: tuple[int | None, ...] | None = None,
) -> PairedFrameRecord:
    values = corrections or tuple(0 for _ in range(28))
    return PairedFrameRecord(
        transport_index=100 + offset,
        source_observed_frame_index=100 + offset,
        platform_observed_frame_index=100 + offset,
        packet_type="SYMBOL",
        block_id=2,
        symbol_id=1310 + offset,
        source_symbols=1311,
        encoded_symbols=1351,
        classification="SOURCE" if offset <= 0 else "REPAIR",
        boundary_offset=offset,
        is_last_source=offset == 0,
        is_first_repair=offset == 1,
        platform_rs_failed=failed,
        codeword_failures=sum(value is None for value in values),
        failed_codeword_indices=tuple(index for index, value in enumerate(values) if value is None),
        corrected_symbols=corrected,
        max_corrections_per_codeword=maximum,
        corrections_per_codeword=values,
        transport_failed=False,
        packet_failed=False,
        mapping_confident=True,
        observed_alignment_match=True,
        frames_since_previous_meta=10,
        frames_until_next_meta=20,
    )


@pytest.mark.parametrize(
    "k,n,symbol,offset,classification",
    [
        (1311, 1351, 1302, -8, "SOURCE"),
        (1311, 1351, 1309, -1, "SOURCE"),
        (1311, 1351, 1310, 0, "SOURCE"),
        (1311, 1351, 1311, 1, "REPAIR"),
        (1311, 1351, 1318, 8, "REPAIR"),
        (656, 676, 655, 0, "SOURCE"),
        (656, 676, 656, 1, "REPAIR"),
    ],
)
def test_boundary_classification(k, n, symbol, offset, classification):
    result = classify_symbol(symbol, k, n)
    assert result["boundary_offset"] == offset
    assert result["classification"] == classification
    assert result["is_last_source"] is (offset == 0)
    assert result["is_first_repair"] is (offset == 1)


def test_confident_failure_maps_by_transport_to_last_source():
    source = {100: _source(100, 1309), 101: _source(101, 1310), 102: _source(102, 1311)}
    platform = [
        _platform(100, 100),
        _platform(101, 101, failed=True, confident=True),
        _platform(102, 102),
    ]
    paired, stats = pair_frames(source, platform)
    failure = paired[1]
    assert stats["failures_mapped"] == 1
    assert (failure.block_id, failure.symbol_id, failure.boundary_offset) == (2, 1310, 0)
    assert failure.is_last_source


def test_ambiguous_failure_never_falls_back_to_observed_index():
    source = {101: _source(101, 1310)}
    paired, stats = pair_frames(source, [_platform(101, None, failed=True, confident=False)])
    assert stats["failures_ambiguous"] == 1
    assert paired[0].classification == "UNKNOWN"
    assert paired[0].boundary_offset is None


def test_offset_aggregate_handles_partial_failed_frame_corrections():
    records = [
        _paired(0, 10, 6),
        _paired(0, 20, 7),
        _paired(0, 30, 8, failed=True, corrections=(None,) + tuple(1 for _ in range(27))),
    ]
    result = aggregate_frame_metrics(records)
    assert result["frame_count"] == 3
    assert result["rs_failure_rate"] == pytest.approx(1 / 3)
    assert result["mean_corrected_symbols"] == 20
    assert result["mean_corrected_symbols_rs_success_frames"] == 15
    assert result["median_corrected_symbols"] == 20
    assert result["p95_corrected_symbols"] == pytest.approx(29)
    assert result["frames_with_max_corrections_ge_7"] == 2
    assert result["frames_with_max_corrections_eq_8"] == 1


def test_codeword_aggregate_separates_failures_from_successful_corrections():
    normal = list(0 for _ in range(28))
    high = list(0 for _ in range(28))
    failed = list(0 for _ in range(28))
    normal[8], high[8], failed[8] = 6, 8, None
    rows = aggregate_codewords([
        _paired(0, 6, 6, corrections=tuple(normal)),
        _paired(0, 8, 8, corrections=tuple(high)),
        _paired(0, 0, 0, failed=True, corrections=tuple(failed)),
    ])
    codeword8 = rows[8]
    assert codeword8["failed_count"] == 1
    assert codeword8["mean_corrections"] == 7
    assert codeword8["count_corrections_ge_7"] == 1
    assert codeword8["count_corrections_eq_8"] == 1


def test_periodicity_uses_actual_source_transport_indices():
    boundaries = [_source(100, 1310), _source(200, 1310), _source(300, 1310)]
    for block_id, record in zip((0, 4, 8), boundaries):
        record.block_id = block_id
    meta = SourceFrame(150, 150, "META")
    result = spacing_analysis(boundaries, boundaries + [meta])
    assert result["spacing_histogram"] == {"100": 2}
    assert result["same_interleave_lane_spacing_histogram"] == {"100": 2}
    assert result["chronological_pairs"][0]["meta_frames_between"] == 1


def test_ffprobe_parser_preserves_order_and_tolerates_missing_fields():
    frames = parse_ffprobe_frames({"frames": [
        {"key_frame": 1, "best_effort_timestamp_time": "0.0", "pkt_size": "123", "pict_type": "I"},
        {"pict_type": "B"},
    ]})
    assert len(frames) == 2
    assert (frames[0].key_frame, frames[0].pkt_size, frames[0].pict_type) == (True, 123, "I")
    assert frames[1].pkt_size is None and frames[1].pict_type == "B"


def test_ffprobe_failure_is_nonfatal(monkeypatch, tmp_path: Path):
    monkeypatch.setattr("fvm_boundary_analysis.subprocess.run", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError()))
    frames, warning = probe_codec_frames(tmp_path / "video.mp4")
    assert frames == []
    assert warning and "ffprobe" in warning
