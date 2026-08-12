from __future__ import annotations

import json
from pathlib import Path

import pytest

from fvm_platform_validate import (UploadGuard, alignment_allows_positional, compare_proxy_platform,
                                   decode_summary, format_state, recovery_pass, redacted,
                                   parse_failed_blocks, select_1080p_format, validate_1080p_probe)


FORMATS = [
    {"format_id": "720", "width": 1280, "height": 720, "vcodec": "avc1.64001f", "tbr": 5000},
    {"format_id": "1080-hevc", "width": 1920, "height": 1080, "vcodec": "hev1.1.6", "tbr": 9000},
    {"format_id": "1080-avc", "width": 1920, "height": 1080, "vcodec": "avc1.640028", "tbr": 7000},
    {"format_id": "4k", "width": 3840, "height": 2160, "vcodec": "avc1.640033", "tbr": 20000},
]


def test_selects_1080_avc_before_other_codecs_and_resolutions() -> None:
    assert select_1080p_format(FORMATS)["format_id"] == "1080-avc"


def test_selects_1080_non_avc_without_720_fallback() -> None:
    selected = select_1080p_format(FORMATS[:2])
    assert selected["format_id"] == "1080-hevc"
    assert select_1080p_format(FORMATS[:1]) is None
    assert format_state(FORMATS[:1]) == ("WAITING_1080P", None)


def test_probe_gate_requires_1080p_30fps() -> None:
    validate_1080p_probe({"stream": {"width": 1920, "height": 1080}, "fps": 30.0}, "video")
    with pytest.raises(ValueError): validate_1080p_probe({"stream": {"width": 1280, "height": 720}, "fps": 30.0}, "video")
    with pytest.raises(ValueError): validate_1080p_probe({"stream": {"width": 1920, "height": 1080}, "fps": 25.0}, "video")


def test_production_decoder_result_parsing() -> None:
    payload = {"video": {"observed_frames": 10}, "transport": {"rs_frames_success": 9, "rs_frames_failed": 1,
        "transport_crc_failures": 2, "invalid_header": 3, "invalid_packet_length": 4,
        "duplicate_embedded_indices": 0, "out_of_order_count": 0},
        "rs": {"codewords_success": 279, "codewords_failed": 1, "corrected_symbols": 8},
        "packets": {"packet_crc_failed": 5, "valid_meta": 6, "post_decode_symbols": 7},
        "raptorq": {"blocks_total": 4, "blocks_decoded": 4, "source_symbols_received": 100, "repair_symbols_received": 2},
        "file": {"original_sha256": "a", "recovered_sha256": "a", "exact": True}}
    parsed = decode_summary(payload)
    assert parsed["rs_failed_frames"] == 1
    assert parsed["codewords_failed"] == 1
    assert parsed["repair_symbols_received"] == 2
    assert parsed["sha_exact"] is True


def test_sha_exact_is_primary_pass_gate() -> None:
    good = {"sha_exact": True, "recovered_sha256": "expected", "blocks_total": 4, "blocks_decoded": 4}
    assert recovery_pass(good, "expected")
    assert not recovery_pass({**good, "recovered_sha256": "wrong"}, "expected")
    assert not recovery_pass({**good, "blocks_decoded": 3}, "expected")


def test_failed_block_diagnostics_are_structured() -> None:
    error = "RaptorQ decode failed: block_id=0, K=1311, N=1351, received_unique=1265, received/K=0.96"
    assert parse_failed_blocks(error) == [{"block_id": 0, "required_k": 1311, "encoded_n": 1351,
                                           "received_unique": 1265, "symbol_deficit": 46}]


def test_one_upload_guard_never_retries_failure() -> None:
    guard = UploadGuard(); calls = []
    def failed_upload(): calls.append(1); raise RuntimeError("rejected")
    with pytest.raises(RuntimeError): guard.run(failed_upload)
    assert guard.attempt_count == 1 and guard.attempted
    with pytest.raises(RuntimeError, match="already attempted"): guard.run(lambda: calls.append(2))
    assert calls == [1]


def test_resume_guard_starts_attempted() -> None:
    guard = UploadGuard(attempted=True, attempt_count=1)
    with pytest.raises(RuntimeError, match="already attempted"):
        guard.run(lambda: pytest.fail("must not upload"))
    assert guard.attempt_count == 1


def test_credential_keys_are_not_persisted(tmp_path: Path) -> None:
    cleaned = redacted({"state": "PASS", "cookies": "secret", "nested": {"authorization": "secret", "value": 1}})
    text = json.dumps(cleaned)
    assert "secret" not in text and "cookie" not in text.lower()
    assert cleaned["nested"]["value"] == 1


@pytest.mark.parametrize("platform_ber,platform_rs,expected", [
    (5e-5, 0, "UNDER-STRESS"), (1.8e-5, 0, "ROUGHLY COMPARABLE"), (5e-6, 0, "OVER-STRESS"),
    (1.8e-5, 5, "UNDER-STRESS")])
def test_platform_proxy_observational_comparison(platform_ber: float, platform_rs: int, expected: str) -> None:
    result = compare_proxy_platform({"raw_ber": 1.8e-5, "rs_failed_frames": 0},
                                    {"raw_ber": platform_ber, "rs_failed_frames": platform_rs})
    assert result["verdict"] == expected


def test_positional_oracle_requires_trustworthy_alignment() -> None:
    source = {"frame_count": 5442}; platform = {"frame_count": 5442}
    diagnostics = {"duplicate_embedded_indices": 0, "out_of_order_count": 0}
    assert alignment_allows_positional(source, platform, diagnostics)
    assert not alignment_allows_positional(source, {"frame_count": 5441}, diagnostics)
    assert not alignment_allows_positional(source, platform, {**diagnostics, "duplicate_embedded_indices": 1})
    assert not alignment_allows_positional(source, platform, {**diagnostics, "out_of_order_count": 1})
    assert not alignment_allows_positional(source, platform, {**diagnostics, "embedded_frame_gaps": 1})
