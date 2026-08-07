import os
import sys
import base64
import hashlib
import zlib

import cv2
import zxingcpp


# 中央裁剪区域。当前视频为 1280x720，二维码居中显示。
# 700 可以保留二维码周围的白色静区，同时减少整帧搜索开销。
ROI_SIZE = 700


def try_decode(detector, frame):
    """
    尝试多种方式识别当前帧中的二维码。

    顺序：
    1. OpenCV：中央 ROI 原图
    2. ZXing-C++：中央 ROI 原图
    3. ZXing-C++：灰度图
    4. ZXing-C++：固定阈值二值化
    5. ZXing-C++：Otsu 自适应二值化
    6. ZXing-C++：2 倍最近邻放大

    返回：
        str: 成功识别出的二维码文本
        "" : 识别失败
    """

    h, w = frame.shape[:2]

    crop_size = min(ROI_SIZE, h, w)

    x1 = max(0, (w - crop_size) // 2)
    y1 = max(0, (h - crop_size) // 2)

    roi = frame[
        y1:y1 + crop_size,
        x1:x1 + crop_size
    ]

    # ------------------------------------------------------------
    # 1. OpenCV
    # ------------------------------------------------------------
    try:
        text, _, _ = detector.detectAndDecode(roi)

        if text:
            return text
    except cv2.error:
        pass

    # ------------------------------------------------------------
    # 2. ZXing-C++：原图
    # ------------------------------------------------------------
    try:
        results = zxingcpp.read_barcodes(roi)

        if results:
            return results[0].text
    except Exception:
        pass

    # ------------------------------------------------------------
    # 3. 灰度图
    # ------------------------------------------------------------
    gray = cv2.cvtColor(
        roi,
        cv2.COLOR_BGR2GRAY
    )

    try:
        results = zxingcpp.read_barcodes(gray)

        if results:
            return results[0].text
    except Exception:
        pass

    # ------------------------------------------------------------
    # 4. 固定阈值二值化
    # ------------------------------------------------------------
    _, binary = cv2.threshold(
        gray,
        127,
        255,
        cv2.THRESH_BINARY
    )

    try:
        results = zxingcpp.read_barcodes(binary)

        if results:
            return results[0].text
    except Exception:
        pass

    # ------------------------------------------------------------
    # 5. Otsu 二值化
    # ------------------------------------------------------------
    _, otsu = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    try:
        results = zxingcpp.read_barcodes(otsu)

        if results:
            return results[0].text
    except Exception:
        pass

    # ------------------------------------------------------------
    # 6. 最近邻 2 倍放大
    # ------------------------------------------------------------
    enlarged = cv2.resize(
        binary,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_NEAREST
    )

    try:
        results = zxingcpp.read_barcodes(enlarged)

        if results:
            return results[0].text
    except Exception:
        pass

    return ""


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            h.update(data)

    return h.hexdigest()


def decode(video_file, output_dir="."):
    cap = cv2.VideoCapture(video_file)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频: {video_file}"
        )

    detector = cv2.QRCodeDetector()

    chunks = {}
    metadata = None

    frame_index = 0
    decoded_frames = 0
    invalid_frames = 0
    crc_errors = 0

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print("视频:", video_file)

    if total_frames > 0:
        print("总帧数:", total_frames)

    while True:
        ok, frame = cap.read()

        if not ok:
            break

        frame_index += 1

        text = try_decode(
            detector,
            frame
        )

        if not text:
            invalid_frames += 1
            continue

        decoded_frames += 1

        # --------------------------------------------------------
        # META
        # --------------------------------------------------------
        if text.startswith("META|"):
            try:
                _, name64, size, sha256, chunk_size, total = \
                    text.split("|")

                filename = base64.b64decode(
                    name64
                ).decode("utf-8")

                new_metadata = {
                    "filename": filename,
                    "size": int(size),
                    "sha256": sha256,
                    "chunk_size": int(chunk_size),
                    "total": int(total)
                }

                # 避免 META 连续重复 30 帧时刷屏
                if metadata != new_metadata:
                    metadata = new_metadata
                    print("META:", metadata)

            except Exception as e:
                print(
                    f"\nMETA 解析失败 "
                    f"(frame={frame_index}): {e}"
                )

            continue

        # --------------------------------------------------------
        # DATA
        # --------------------------------------------------------
        if text.startswith("DATA|"):
            try:
                _, seq, total, crc_hex, payload = \
                    text.split("|", 4)

                seq = int(seq)
                total = int(total)

                # REPEAT 会导致相同数据块连续出现多次
                if seq in chunks:
                    continue

                data = base64.b64decode(
                    payload,
                    validate=True
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
                    crc_errors += 1

                    print(
                        f"\nCRC 错误: "
                        f"seq={seq}, frame={frame_index}"
                    )

                    continue

                chunks[seq] = data

                expected_total = (
                    metadata["total"]
                    if metadata
                    else total
                )

                print(
                    f"\r已恢复 "
                    f"{len(chunks)}/{expected_total} 块 "
                    f"(frame {frame_index}/{total_frames or '?'})",
                    end="",
                    flush=True
                )

            except Exception:
                # 识别出文本但 DATA 内容不完整/损坏时忽略，
                # 后续重复帧仍有机会恢复。
                continue

    cap.release()

    print()

    if metadata is None:
        raise RuntimeError(
            "没有找到有效 META"
        )

    total = metadata["total"]

    print(
        f"需要 {total} 块，"
        f"找到 {len(chunks)} 块"
    )

    print(
        f"成功识别二维码帧: {decoded_frames}"
    )

    print(
        f"未识别二维码帧: {invalid_frames}"
    )

    print(
        f"CRC 错误帧: {crc_errors}"
    )

    missing = [
        i
        for i in range(total)
        if i not in chunks
    ]

    if missing:
        print(
            f"缺失 {len(missing)} 个数据块"
        )

        print(
            "前 100 个缺失块:",
            missing[:100]
        )

        raise RuntimeError(
            "数据不完整"
        )

    filename = os.path.basename(
        metadata["filename"]
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_file = os.path.join(
        output_dir,
        filename
    )

    # 避免覆盖原始测试文件。
    # 如果目标文件已存在，则输出为 recovered_文件名。
    if os.path.exists(output_file):
        output_file = os.path.join(
            output_dir,
            "recovered_" + filename
        )

    with open(output_file, "wb") as f:
        for i in range(total):
            f.write(
                chunks[i]
            )

    # 元数据里记录了原始大小，最后一个 chunk
    # 可能不足 CHUNK_SIZE，因此按原始大小截断。
    with open(output_file, "r+b") as f:
        f.truncate(
            metadata["size"]
        )

    actual_hash = sha256_file(
        output_file
    )

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

        raise RuntimeError(
            "文件已拼接，但 SHA256 校验失败"
        )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "用法:"
        )
        print(
            "  python video2file_zxing.py "
            "<video.mp4> [output_dir]"
        )

        sys.exit(1)

    video_file = sys.argv[1]

    output_dir = (
        sys.argv[2]
        if len(sys.argv) >= 3
        else "."
    )

    decode(
        video_file,
        output_dir
    )
