from __future__ import annotations

import numpy as np
import pytest

import video2file


class _Result:
    text = "decoded"


@pytest.mark.parametrize(
    ("height", "width", "expected_size"),
    [
        (720, 1280, 720),
        (1080, 1920, 1080),
    ],
)
def test_decode_qr_uses_full_centered_short_edge(monkeypatch, height: int, width: int, expected_size: int) -> None:
    seen_shapes: list[tuple[int, ...]] = []

    def fake_read_barcodes(candidate):
        seen_shapes.append(candidate.shape)
        return [_Result()]

    monkeypatch.setattr(video2file.zxingcpp, "read_barcodes", fake_read_barcodes)
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    assert video2file.decode_qr(frame) == "decoded"
    assert seen_shapes == [(expected_size, expected_size, 3)]
