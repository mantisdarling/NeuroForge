"""Create a compact, deterministic MNIST fixture for the browser-only digit lab."""
from __future__ import annotations

import base64
import gzip
import struct
from pathlib import Path
from urllib.request import urlopen

BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
IMAGES_URL = BASE + "train-images-idx3-ubyte.gz"
LABELS_URL = BASE + "train-labels-idx1-ubyte.gz"
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "client" / "src" / "engine" / "mnistMini.ts"


def fetch(url: str) -> bytes:
    with urlopen(url, timeout=30) as response:
        return gzip.decompress(response.read())


def downsample(image: bytes) -> bytes:
    result = bytearray()
    for row in range(14):
        for column in range(14):
            values = [image[(row * 2 + dy) * 28 + (column * 2 + dx)] for dy in range(2) for dx in range(2)]
            result.append(round(sum(values) / 4))
    return bytes(result)


def main() -> None:
    image_blob, label_blob = fetch(IMAGES_URL), fetch(LABELS_URL)
    _, image_count, rows, columns = struct.unpack(">IIII", image_blob[:16])
    _, label_count = struct.unpack(">II", label_blob[:8])
    if image_count != label_count or (rows, columns) != (28, 28):
        raise ValueError("Unexpected MNIST IDX dimensions")

    found = {label: 0 for label in range(10)}
    examples: list[tuple[int, str]] = []
    for index in range(image_count):
        label = label_blob[8 + index]
        if found[label] >= 6:
            continue
        start = 16 + index * 784
        pixels = downsample(image_blob[start : start + 784])
        examples.append((label, base64.b64encode(pixels).decode("ascii")))
        found[label] += 1
        if all(count == 6 for count in found.values()):
            break
    if len(examples) != 60:
        raise ValueError("Could not select six examples for every digit")

    records = ",\n".join(f"  {{ label: {label}, pixels: \"{pixels}\" }}" for label, pixels in examples)
    OUTPUT.write_text(
        "// Generated from the public MNIST training set by scripts/prepare_mnist_subset.py.\n"
        "// 60 fixed 14×14 grayscale samples; no runtime network request is made.\n"
        "export type MnistMiniSample = { label: number; pixels: string };\n"
        "export const MNIST_MINI: MnistMiniSample[] = [\n" + records + "\n];\n"
        "export function decodeMnistMini(encoded: string): number[] {\n"
        "  const bytes = Uint8Array.from(atob(encoded), (character) => character.charCodeAt(0));\n"
        "  return Array.from(bytes, (value) => value / 255);\n"
        "}\n"
    )
    print(f"Wrote {OUTPUT} with {len(examples)} examples.")


if __name__ == "__main__":
    main()
