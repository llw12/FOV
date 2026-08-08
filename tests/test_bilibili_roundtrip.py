from __future__ import annotations

import hashlib
import hmac

from scripts.bilibili_roundtrip import (
    Credentials,
    build_signed_headers,
    classify_archive,
    compact_json,
)


def test_compact_json_matches_bilibili_md5_example() -> None:
    body = compact_json({"name": "test.mp4", "utype": "0"})
    assert body == b'{"name":"test.mp4","utype":"0"}'
    assert hashlib.md5(body).hexdigest() == "18323d990354c0c0d63340f0a67ce8e4"


def test_build_signed_headers_is_deterministic() -> None:
    credentials = Credentials("client", "secret", "token")
    body = compact_json({"hello": "world"})
    headers = build_signed_headers(
        credentials,
        body_bytes=body,
        timestamp=1_700_000_000,
        nonce="nonce-1",
    )
    canonical = "\n".join(
        [
            "x-bili-accesskeyid:client",
            f"x-bili-content-md5:{hashlib.md5(body).hexdigest()}",
            "x-bili-signature-method:HMAC-SHA256",
            "x-bili-signature-nonce:nonce-1",
            "x-bili-signature-version:2.0",
            "x-bili-timestamp:1700000000",
        ]
    )
    expected = hmac.new(b"secret", canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    assert headers["Authorization"] == expected
    assert headers["Access-Token"] == "token"
    assert headers["Content-Type"] == "application/json"


def test_classify_archive_open() -> None:
    state = classify_archive(
        {
            "data": {
                "addit_info": {"state": 0, "state_desc": "开放浏览", "reject_reason": ""},
                "video_info": {"share_url": "https://www.bilibili.com/video/BV1TEST"},
            }
        }
    )
    assert state.outcome == "open"
    assert state.share_url.endswith("BV1TEST")


def test_classify_archive_waiting() -> None:
    state = classify_archive(
        {"data": {"addit_info": {"state": -1, "state_desc": "审核中", "reject_reason": ""}}}
    )
    assert state.outcome == "waiting"


def test_classify_archive_failed() -> None:
    state = classify_archive(
        {
            "data": {
                "addit_info": {
                    "state": -2,
                    "state_desc": "已退回",
                    "reject_reason": "测试退回原因",
                }
            }
        }
    )
    assert state.outcome == "failed"
    assert state.reject_reason == "测试退回原因"
