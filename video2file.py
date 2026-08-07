import os
import sys
import base64
import hashlib
import zlib

import cv2

def decode(video_file, output_dir="."):

    cap = cv2.VideoCapture(video_file)

    detector = cv2.QRCodeDetector()

    chunks = {}

    metadata = None

    frame_index = 0

    while True:

        ok, frame = cap.read()

        if not ok:
            break

        frame_index += 1

        text, points, _ = detector.detectAndDecode(
            frame
        )

        if not text:
            continue

        #
        # 元数据
        #
        if text.startswith("META|"):

            try:
                _, name64, size, sha256, chunk_size, total = \
                    text.split("|")

                filename = base64.b64decode(
                    name64
                ).decode("utf-8")

                metadata = {
                    "filename": filename,
                    "size": int(size),
                    "sha256": sha256,
                    "chunk_size": int(chunk_size),
                    "total": int(total)
                }

                print("META:", metadata)

            except Exception as e:
                print("META 解析失败:", e)

            continue

        #
        # 数据块
        #
        if text.startswith("DATA|"):

            try:

                _, seq, total, crc_hex, payload = \
                    text.split("|", 4)

                seq = int(seq)

                if seq in chunks:
                    continue

                data = base64.b64decode(
                    payload
                )

                expected_crc = int(
                    crc_hex,
                    16
                )

                actual_crc = (
                    zlib.crc32(data)
                    & 0xffffffff
                )

                if actual_crc != expected_crc:
                    print(
                        "CRC 错误:",
                        seq
                    )

                    continue

                chunks[seq] = data

                print(
                    f"\r已恢复 {len(chunks)} 块",
                    end="",
                    flush=True
                )

            except Exception:
                pass

    cap.release()

    print()

    if metadata is None:
        raise RuntimeError(
            "没有找到 META"
        )

    total = metadata["total"]

    print(
        f"需要 {total} 块，"
        f"找到 {len(chunks)} 块"
    )

    missing = [
        i
        for i in range(total)
        if i not in chunks
    ]

    if missing:

        print(
            "缺失数据块:",
            missing[:50]
        )

        raise RuntimeError(
            "数据不完整"
        )

    filename = os.path.basename(
        metadata["filename"]
    )

    output_file = os.path.join(
        output_dir,
        filename
    )

    with open(output_file, "wb") as f:

        for i in range(total):
            f.write(
                chunks[i]
            )

    #
    # SHA256 最终检查
    #
    h = hashlib.sha256()

    with open(output_file, "rb") as f:

        while True:

            data = f.read(
                1024 * 1024
            )

            if not data:
                break

            h.update(data)

    actual_hash = h.hexdigest()

    print(
        "原始 SHA256:",
        metadata["sha256"]
    )

    print(
        "恢复 SHA256:",
        actual_hash
    )

    if actual_hash == metadata["sha256"]:
        print("✅ 文件完整恢复")
        print("输出:", output_file)

    else:
        print("❌ SHA256 不一致")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print(
            "python video2file.py "
            "<video.mp4>"
        )

        sys.exit(1)

    decode(
        sys.argv[1]
    )