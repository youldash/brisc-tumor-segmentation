"""Command-line inference: segment one or more MRI slices and save the overlays.

    python app/predict_cli.py app/examples/*.jpg --out outputs/predictions
    python app/predict_cli.py scan.png --model models/unet_resnet34_fp32.onnx --threshold 0.6

Prints one JSON line of measurements per image. Educational prototype — not a medical device.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from src.inference import OnnxSegmenter, render_heatmap, render_overlay  # noqa: E402


def run(images, segmenter, out_dir: Path, threshold, min_area):
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for path in images:
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(json.dumps({"image": str(path), "error": "cannot read image"}))
            continue
        result = segmenter.segment(image, threshold=threshold, min_area=min_area)
        panel = np.hstack([render_overlay(image, result), render_heatmap(image, result.prob)])
        target = out_dir / f"{Path(path).stem}_prediction.png"
        cv2.imwrite(str(target), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
        record = {"image": str(path), "output": str(target), **result.stats}
        print(json.dumps(record))
        results.append(record)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("images", nargs="+")
    p.add_argument("--model")
    p.add_argument("--out", default="outputs/predictions")
    p.add_argument("--threshold", type=float)
    p.add_argument("--min-area", type=int, default=0)
    args = p.parse_args()
    segmenter = OnnxSegmenter(args.model)
    run(args.images, segmenter, Path(args.out), args.threshold, args.min_area)


if __name__ == "__main__":
    main()
