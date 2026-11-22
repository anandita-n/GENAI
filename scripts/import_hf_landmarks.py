"""
SignBridge AI - Import a public landmark dataset from Hugging Face.

Downloads sid220/asl-now-fingerspelling (MIT licensed, ~3 MB, 2,122 JSON files of
21 MediaPipe hand landmarks per sample, 26 letters) and converts every sample into
the project's 83-feature format. Output is written as data/recorded_HF_<LETTER>.json,
which scripts/train_model.py already picks up as real-world training data.

Usage:
    python scripts/import_hf_landmarks.py            # skips J and Z (motion letters)
    python scripts/import_hf_landmarks.py --all      # also import J and Z poses
"""

import argparse
import json
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from app.config import DATA_DIR
from app.landmark_extractor import LandmarkExtractor

REPO = "sid220/asl-now-fingerspelling"
API_URL = f"https://huggingface.co/api/datasets/{REPO}"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/{{path}}"
CACHE_DIR = DATA_DIR / "real_hf"
MOTION_LETTERS = {"J", "Z"}  # signed with movement; a single frame cannot capture them


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "signbridge-import"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return resp.read()


def list_files():
    info = json.loads(_get(API_URL))
    return [
        s["rfilename"] for s in info["siblings"]
        if "/" in s["rfilename"] and s["rfilename"].endswith(".json")
    ]


def download(path: str) -> Path:
    dest = CACHE_DIR / path
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_get(FILE_URL.format(path=path)))
    return dest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="also import J and Z")
    args = parser.parse_args()

    files = list_files()
    if not args.all:
        files = [f for f in files if f.split("/")[0] not in MOTION_LETTERS]
    print(f"Downloading {len(files)} files from {REPO} ...")

    with ThreadPoolExecutor(max_workers=16) as pool:
        local_paths = list(pool.map(download, files))

    by_letter = defaultdict(list)
    skipped = 0
    for rel, path in zip(files, local_paths):
        letter = rel.split("/")[0].upper()
        try:
            landmarks = json.loads(path.read_text())
            if len(landmarks) != 21:
                raise ValueError("expected 21 landmarks")
            feats = LandmarkExtractor.from_client_landmarks(landmarks)
            if feats.shape[0] != 83 or not np.all(np.isfinite(feats)):
                raise ValueError("bad feature vector")
            by_letter[letter].append(feats.tolist())
        except Exception as e:
            skipped += 1
            print(f"  skipped {rel}: {e}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for old in DATA_DIR.glob("recorded_HF_*.json"):
        old.unlink()
    for letter, feats in sorted(by_letter.items()):
        out = DATA_DIR / f"recorded_HF_{letter}.json"
        out.write_text(json.dumps({"sign": letter, "features": feats}))

    total = sum(len(v) for v in by_letter.values())
    print(f"Converted {total} samples across {len(by_letter)} letters ({skipped} skipped).")
    print("Per letter:", {k: len(v) for k, v in sorted(by_letter.items())})
    print("Now run: python scripts/train_model.py")


if __name__ == "__main__":
    sys.exit(main())
