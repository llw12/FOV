from __future__ import annotations

import argparse

from scripts.bilibili_scan import (
    PLATFORM_TARGETED_BASE_CASES,
    PlatformScanResult,
    build_cases,
    make_title,
)
from scripts.scan_params import ScanCase


def namespace(**overrides):
    values = {
        "case": None,
        "preset": "platform",
        "repairs": [0.20],
        "fps": 30,
        "max_cases": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_platform_preset_is_small_real_channel_default() -> None:
    cases = build_cases(namespace())
    assert [(case.width, case.height, case.symbol_size) for case in cases] == (
        PLATFORM_TARGETED_BASE_CASES
    )
    assert all(case.repair == 0.20 for case in cases)


def test_repairs_cross_product_and_max_cases() -> None:
    cases = build_cases(namespace(repairs=[0.10, 0.20], max_cases=3))
    assert cases == [
        ScanCase(1920, 1080, 500, 0.10, 30),
        ScanCase(1920, 1080, 500, 0.20, 30),
        ScanCase(1920, 1080, 520, 0.10, 30),
    ]


def test_explicit_cases_override_preset() -> None:
    custom = [
        ScanCase(1920, 1080, 505, 0.15, 30),
        ScanCase(1280, 720, 280, 0.20, 30),
    ]
    assert build_cases(namespace(case=custom)) == custom


def test_title_contains_case_and_is_bilibili_length_safe() -> None:
    title = make_title(
        "FOV 参数扫描",
        ScanCase(1920, 1080, 520, 0.20, 30),
        "20260808-171500",
        2,
    )
    assert "1920x1080" in title
    assert "s520" in title
    assert "r20" in title
    assert len(title) <= 80


def test_platform_metrics() -> None:
    result = PlatformScanResult(
        case_id="1920x1080_s520_r20",
        width=1920,
        height=1080,
        fps=30,
        symbol_size=520,
        repair=0.20,
        input_bytes=399825,
        platform_qr_decoded=894,
        platform_qr_failed=52,
        platform_blocks=[
            {
                "block_id": 0,
                "received_source": 717,
                "received_repair": 52,
                "decoded_at_frame": 832,
            }
        ],
    )
    assert round(result.platform_erasure_rate or 0, 6) == round(52 / 946, 6)
    assert result.repair_symbols_received == 52
