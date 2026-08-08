from __future__ import annotations

import json
from pathlib import Path

from scripts.bilibili_roundtrip import (
    UploadIdentity,
    bvid_in_list,
    build_upload_command,
    export_biliup_cookies_to_netscape,
    find_bvid_by_title,
    parse_decoder_output,
    parse_list_entries,
    parse_upload_identity,
)


def test_parse_upload_identity_from_biliup_1_2_2_log() -> None:
    text = '''
2026-08-08 16:50:18 INFO ResponseData { code: 0, data: Some(Object {
"aid": Number(117059015018441), "bvid": String("BV1wFug64EZg")
}), message: "OK" }
APP接口投稿成功
'''
    assert parse_upload_identity(text) == UploadIdentity(
        bvid="BV1wFug64EZg", aid=117059015018441
    )


def test_parse_list_entries_and_lookup() -> None:
    text = '''
2026-08-08 16:51:32 INFO user: example
BV1wFug64EZg    FOV 自动回环测试 500B   开放浏览
BV1YEug6VE9Z    另一个测试                    开放浏览
'''
    assert parse_list_entries(text) == [
        ("BV1wFug64EZg", "FOV 自动回环测试 500B   开放浏览"),
        ("BV1YEug6VE9Z", "另一个测试                    开放浏览"),
    ]
    assert bvid_in_list(text, "BV1wFug64EZg")
    assert not bvid_in_list(text, "BV1DOESNOTEXIST")
    assert find_bvid_by_title(text, "FOV 自动回环测试 500B") == "BV1wFug64EZg"


def test_build_upload_command_matches_biliup_cli_shape(tmp_path: Path) -> None:
    cookies = tmp_path / "cookies.json"
    video = tmp_path / "source.mp4"
    command = build_upload_command(
        "biliup.exe",
        cookies,
        video,
        title="FOV 自动回环测试 500B",
        tid=171,
        tag="FOV,二维码,测试",
        desc="FOV 视频信道自动化实验",
        copyright_value=1,
        submit="app",
        limit=3,
        cover=None,
    )
    assert command[:5] == [
        "biliup.exe",
        "-u",
        str(cookies),
        "upload",
        str(video),
    ]
    assert command[command.index("--tid") + 1] == "171"
    assert command[command.index("--submit") + 1] == "app"
    assert command[command.index("--limit") + 1] == "3"


def test_export_biliup_cookies_to_netscape(tmp_path: Path) -> None:
    source = tmp_path / "cookies.json"
    target = tmp_path / "cookies.txt"
    source.write_text(
        json.dumps(
            {
                "cookie_info": {
                    "cookies": [
                        {"name": "SESSDATA", "value": "secret", "secure": True},
                        {
                            "name": "bili_jct",
                            "value": "csrf",
                            "domain": "bilibili.com",
                            "path": "/",
                            "expires": 1800000000,
                        },
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    assert export_biliup_cookies_to_netscape(source, target) == 2
    text = target.read_text(encoding="utf-8")
    assert text.startswith("# Netscape HTTP Cookie File\n")
    assert "\tSESSDATA\tsecret\n" in text
    assert ".bilibili.com\tTRUE\t/" in text
    assert "\tbili_jct\tcsrf\n" in text


def test_parse_decoder_output_with_repair_symbols() -> None:
    text = '''
Total frames: 946
Block 0:
  K: 769
  N: 923
  received unique: 769
  received source: 717
  received repair: 52
  received/K: 1.00
  decoded at frame: 832
  [OK]

QR:
  decoded: 894
  failed: 52
  failed frame indices (0-based): 203,205,211
  failed frame indices truncated: no

Packets:
  valid META: 23
  CRC failed: 0
  post-decode symbols: 102

Original SHA256: abc
Recovered SHA256: abc
[OK] File fully recovered
'''
    parsed = parse_decoder_output(text)
    assert parsed["total_frames"] == 946
    assert parsed["qr_decoded"] == 894
    assert parsed["qr_failed"] == 52
    assert parsed["failed_frame_indices"] == [203, 205, 211]
    assert parsed["crc_failed"] == 0
    assert parsed["blocks"] == [
        {
            "block_id": 0,
            "k": 769,
            "n": 923,
            "received_unique": 769,
            "received_source": 717,
            "received_repair": 52,
            "decoded_at_frame": 832,
        }
    ]
    assert parsed["sha256_match"] is True
    assert parsed["file_fully_recovered"] is True
