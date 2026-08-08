from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from scripts.scan_params import ScanCase, build_cases, parse_case, parse_decoder_output, simulation_bitrate_kbps


def test_parse_case_defaults_and_explicit_values() -> None:
    assert parse_case("1920x1080:500") == ScanCase(1920, 1080, 500, 0.20, 30)
    assert parse_case("1280x720:280:0.3:25") == ScanCase(1280, 720, 280, 0.30, 25)


def test_parse_case_rejects_invalid_text() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        parse_case("1080p:500")


def test_parse_decoder_output() -> None:
    text = """
QR:
  decoded: 964
  failed: 19

Packets:
  valid META: 23
  CRC failed: 0
  post-decode symbols: 141
  decoded blocks: 1
"""
    assert parse_decoder_output(text) == {
        "qr_decoded": 964,
        "qr_failed": 19,
        "crc_failed": 0,
        "post_decode_symbols": 141,
    }


def test_targeted_matrix_and_repair_expansion() -> None:
    args = SimpleNamespace(case=None, repairs=[0.10, 0.20], preset="targeted", fps=30)
    cases = build_cases(args)
    assert ScanCase(1280, 720, 280, 0.20, 30) in cases
    assert ScanCase(1920, 1080, 500, 0.10, 30) in cases
    assert len(cases) == 14


def test_custom_cases_override_preset() -> None:
    custom = [ScanCase(1920, 1080, 500, 0.2), ScanCase(1920, 1080, 500, 0.2)]
    args = SimpleNamespace(case=custom, repairs=[0.1], preset="full", fps=30)
    assert build_cases(args) == [custom[0]]


def test_simulation_bitrate_uses_resolution_band() -> None:
    assert simulation_bitrate_kbps(ScanCase(1280, 720, 280, 0.2), 900, 1500) == 900
    assert simulation_bitrate_kbps(ScanCase(1920, 1080, 500, 0.2), 900, 1500) == 1500
