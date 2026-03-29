"""
Transforms the raw data + samples.json into an ImageFolder-compatible directory structure:
    dataset/
        train/
            A/  B/  C/  ...
        test/
            A/  B/  C/  ...
"""

import json
import shutil
from pathlib import Path

SRC_ROOT = Path(__file__).parent
SAMPLES_JSON = SRC_ROOT / "samples.json"
DST_ROOT = SRC_ROOT / "dataset"


def main():
    print("Loading samples.json ...")
    with open(SAMPLES_JSON) as f:
        data = json.load(f)

    samples = data["samples"]
    print(f"Total samples: {len(samples)}")

    skipped = 0
    moved = 0

    for sample in samples:
        tags = sample.get("tags", [])
        split = None
        if "train" in tags:
            split = "train"
        elif "test" in tags:
            split = "test"
        else:
            skipped += 1
            continue

        label = sample["ground_truth"]["label"]
        src_path = SRC_ROOT / sample["filepath"]

        dst_dir = DST_ROOT / split / label
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)
        moved += 1

    print(f"Done. Moved: {moved}, Skipped (no tag): {skipped}")

    # Print class distribution
    for split in ("train", "test"):
        split_dir = DST_ROOT / split
        if not split_dir.exists():
            continue
        classes = sorted(split_dir.iterdir())
        total = sum(len(list(c.iterdir())) for c in classes)
        print(f"\n{split}: {len(classes)} classes, {total} images")
        for c in classes:
            print(f"  {c.name}: {len(list(c.iterdir()))}")


if __name__ == "__main__":
    main()
