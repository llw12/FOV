import os
import sys
import math
import base64
import hashlib
import zlib
import subprocess

import cv2
import numpy as np
import qrcode
from qrcode.constants import ERROR_CORRECT_M


WIDTH = 1280
HEIGHT = 720
FPS = 30

# 每个二维码保存多少原始字节
CHUNK_SIZE = 200

# 每个二维码重复多少帧
REPEAT = 3


def sha256_file(path):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        while True:
            data = f.read(1024 * 1024)

            if not data:
                break

            h.update(data)

    return h.hexdigest()


def create_qr_frame(text):
    qr = qrcode.QRCode(
        version=None,
        error_correction=ERROR_CORRECT_M,
        box_size=8,
        border=4
    )

    qr.add_data(text)
    qr.make(fit=True)

    img = qr.make_image(
        fill_color="black",
        back_color="white"
    ).convert("RGB")

    qr_array = np.array(img)

    qr_array = cv2.cvtColor(
        qr_array,
        cv2.COLOR_RGB2BGR
    )

    frame = np.full(
        (HEIGHT, WIDTH, 3),
        255,
        dtype=np.uint8
    )

    h, w = qr_array.shape[:2]

    if h > HEIGHT - 40 or w > WIDTH - 40:
        raise RuntimeError(
            f"二维码过大: {w}x{h}，请减小 CHUNK_SIZE"
        )

    x = (WIDTH - w) // 2
    y = (HEIGHT - h) // 2

    frame[
        y:y + h,
        x:x + w
    ] = qr_array

    return frame


def encode(input_file, output_video):
    size = os.path.getsize(input_file)

    total = math.ceil(
        size / CHUNK_SIZE
    )

    file_hash = sha256_file(input_file)

    filename = os.path.basename(input_file)

    filename_b64 = base64.b64encode(
        filename.encode("utf-8")
    ).decode("ascii")

    print("文件:", filename)
    print("大小:", size)
    print("SHA256:", file_hash)
    print("数据块:", total)

    cmd = [
        "ffmpeg",
        "-y",

        "-f", "rawvideo",
        "-pix_fmt", "bgr24",

        "-s",
        f"{WIDTH}x{HEIGHT}",

        "-r", str(FPS),

        "-i", "-",

        "-an",

        "-c:v", "libx264",

        # 尽量减少第一次编码造成的损失
        "-crf", "15",

        "-preset", "medium",

        "-pix_fmt", "yuv420p",

        output_video
    ]

    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE
    )

    #
    # META 帧
    #
    meta = (
        f"META|"
        f"{filename_b64}|"
        f"{size}|"
        f"{file_hash}|"
        f"{CHUNK_SIZE}|"
        f"{total}"
    )

    meta_frame = create_qr_frame(meta)

    # 开头显示 1 秒元数据
    for _ in range(FPS):
        process.stdin.write(
            meta_frame.tobytes()
        )

    #
    # DATA 帧
    #
    with open(input_file, "rb") as f:

        seq = 0

        while True:
            chunk = f.read(CHUNK_SIZE)

            if not chunk:
                break

            crc = zlib.crc32(chunk) & 0xffffffff

            payload = base64.b64encode(
                chunk
            ).decode("ascii")

            text = (
                f"DATA|"
                f"{seq}|"
                f"{total}|"
                f"{crc:08x}|"
                f"{payload}"
            )

            frame = create_qr_frame(text)

            for _ in range(REPEAT):
                process.stdin.write(
                    frame.tobytes()
                )

            seq += 1

            if seq % 100 == 0:
                print(
                    f"\r{seq}/{total}",
                    end="",
                    flush=True
                )

    #
    # 结尾再放一次 META
    #
    for _ in range(FPS):
        process.stdin.write(
            meta_frame.tobytes()
        )

    process.stdin.close()
    process.wait()

    print()
    print("完成:", output_video)


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print(
            "python file2video.py "
            "<input_file> <output.mp4>"
        )

        sys.exit(1)

    encode(
        sys.argv[1],
        sys.argv[2]
    )