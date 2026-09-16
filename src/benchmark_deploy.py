"""Benchmark the deployable ONNX models against the PyTorch reference.

For every model/provider combination: test Dice/IoU (native resolution, same protocol as
src.evaluate), healthy-slice false positives, latency (model only and end-to-end) and file size.

Usage (from the project root):
    python -m src.benchmark_deploy
    python -m src.benchmark_deploy --coreml          # also try Apple's Core ML execution provider
    python -m src.benchmark_deploy --limit 100       # quick run

Writes assets/deployment_benchmark.json and assets/deployment_benchmark.md.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.analysis import (component_areas, healthy_metrics, overall_metrics, remove_small_components,
                          to_markdown)
from src.data_index import load_pair
from src.inference import OnnxSegmenter, preprocess, to_rgb_gray
from src.metrics import confusion_counts, per_image_scores
from src.utils import resolve_path, save_json


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=["models/unet_resnet34_fp32.onnx", "models/unet_resnet34_int8.onnx"])
    p.add_argument("--reference-run", default="runs/unet_resnet34", help="PyTorch evaluation to compare with")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--limit", type=int, help="use only the first N test slices")
    p.add_argument("--coreml", action="store_true", help="also benchmark FP32 with CoreMLExecutionProvider")
    p.add_argument("--threads", type=int, help="ONNX Runtime intra-op threads (default: automatic)")
    p.add_argument("--out", default="assets")
    return p.parse_args()


def file_size_mb(path: Path) -> float:
    size = path.stat().st_size
    data = path.with_name(path.name + ".data")
    return (size + (data.stat().st_size if data.exists() else 0)) / 1e6


def score_split(seg: OnnxSegmenter, df: pd.DataFrame, root: Path, threshold: float):
    rows, model_s, total_s = [], 0.0, 0.0
    for _, r in df.iterrows():
        image, gt = load_pair(root, r)
        t0 = time.perf_counter()
        rgb = to_rgb_gray(image)
        x = preprocess(rgb, seg.size)
        t1 = time.perf_counter()
        out = seg.session.run(None, {seg.input_name: x})[0]
        t2 = time.perf_counter()
        prob = np.asarray(out, np.float32).reshape(seg.size, seg.size)
        prob = cv2.resize(prob, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
        pred = remove_small_components(prob > threshold, 0)
        t3 = time.perf_counter()
        model_s += t2 - t1
        total_s += t3 - t0
        c = confusion_counts(pred[None], gt[None].astype(bool))[0]
        s = per_image_scores(c[None])
        areas = component_areas(pred)
        rows.append({"stem": r["stem"], "tumor_type": r["tumor_type"], "dice": s["dice"][0], "iou": s["iou"][0],
                     "precision": s["precision"][0], "recall": s["recall"][0],
                     "tp": int(c[0]), "fp": int(c[1]), "fn": int(c[2]), "tn": int(c[3]),
                     "pred_area_frac": float(pred.mean()),
                     "largest_component_px": int(areas.max()) if len(areas) else 0})
    n = max(len(df), 1)
    return pd.DataFrame(rows), 1000 * model_s / n, 1000 * total_s / n


def benchmark(name: str, seg: OnnxSegmenter, size_mb: float, test_df, healthy_df, root, threshold,
              warmup: int = 5) -> tuple[dict, pd.DataFrame]:
    dummy = np.zeros((1, 3, seg.size, seg.size), np.float32)
    for _ in range(warmup):
        seg.session.run(None, {seg.input_name: dummy})
    test_res, model_ms, total_ms = score_split(seg, test_df, root, threshold)
    healthy_res, _, _ = score_split(seg, healthy_df, root, threshold) if len(healthy_df) else (pd.DataFrame(), 0, 0)
    o = overall_metrics(test_res)
    h = healthy_metrics(healthy_res) if len(healthy_res) else {}
    row = {"variant": name, "size_mb": size_mb, "test_dice": o["dice"], "test_dice_median": o["dice_median"],
           "test_iou": o["iou"], "precision": o["precision"], "recall": o["recall"],
           "healthy_fp_rate_ge_50px": h.get("fp_rate_largest_ge_50px", float("nan")),
           "model_ms_per_image": model_ms, "end_to_end_ms_per_image": total_ms,
           "by_tumor_type": {t: float(g["dice"].mean()) for t, g in test_res.groupby("tumor_type")}}
    return row, test_res


def main():
    args = parse_args()
    root = resolve_path(args.data_root)
    splits = resolve_path(args.splits_dir)
    test_df = pd.read_csv(splits / "test.csv")
    healthy_path = splits / "healthy_test.csv"
    healthy_df = pd.read_csv(healthy_path).fillna({"mask": ""}) if healthy_path.exists() else pd.DataFrame()
    if args.limit:
        test_df = test_df.head(args.limit)
        healthy_df = healthy_df.head(max(1, args.limit // 6)) if len(healthy_df) else healthy_df

    rows, per_image = [], {}
    ref_dir = resolve_path(args.reference_run) / "eval"
    if (ref_dir / "metrics.json").exists():
        ref = json.loads((ref_dir / "metrics.json").read_text())
        ref_test = pd.read_csv(ref_dir / "test_per_image.csv")
        ref_test = ref_test[ref_test["stem"].isin(test_df["stem"])]
        ref_h = pd.read_csv(ref_dir / "healthy_per_image.csv") if (ref_dir / "healthy_per_image.csv").exists() else pd.DataFrame()
        if len(ref_h):
            ref_h = ref_h[ref_h["stem"].isin(healthy_df["stem"])]
        o = overall_metrics(ref_test)
        best = resolve_path(args.reference_run) / "best.pt"
        rows.append({"variant": "PyTorch FP32 (MPS GPU, batch 16)",
                     "size_mb": best.stat().st_size / 1e6 if best.exists() else float("nan"),
                     "test_dice": o["dice"], "test_dice_median": o["dice_median"], "test_iou": o["iou"],
                     "precision": o["precision"], "recall": o["recall"],
                     "healthy_fp_rate_ge_50px": healthy_metrics(ref_h).get("fp_rate_largest_ge_50px", float("nan")) if len(ref_h) else float("nan"),
                     "model_ms_per_image": ref["speed"]["ms_per_image_forward"], "end_to_end_ms_per_image": float("nan"),
                     "by_tumor_type": {t: float(g["dice"].mean()) for t, g in ref_test.groupby("tumor_type")}})
        per_image["pytorch"] = ref_test

    for model in args.models:
        path = resolve_path(model)
        if not path.exists():
            print(f"Skipping {path} (not found)")
            continue
        precision = "INT8" if "int8" in path.name.lower() else "FP32"
        configs = [("CPUExecutionProvider", f"ONNX {precision} (CPU)")]
        if args.coreml and precision == "FP32":
            configs.append(("CoreMLExecutionProvider", f"ONNX {precision} (Core ML)"))
        for provider, name in configs:
            try:
                providers = [provider] + (["CPUExecutionProvider"] if provider != "CPUExecutionProvider" else [])
                seg = OnnxSegmenter(path, providers=providers, num_threads=args.threads)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {name}: {exc}")
                continue
            print(f"Benchmarking {name} on {len(test_df)} test + {len(healthy_df)} healthy slices ...")
            row, res = benchmark(name, seg, file_size_mb(path), test_df, healthy_df, root, args.threshold)
            rows.append(row)
            per_image[name] = res
            print(f"  Dice {row['test_dice']:.4f} | {row['model_ms_per_image']:.1f} ms model | "
                  f"{row['end_to_end_ms_per_image']:.1f} ms end-to-end")

    # agreement of each ONNX variant with the PyTorch reference, slice by slice
    if "pytorch" in per_image:
        ref = per_image["pytorch"].set_index("stem")["dice"]
        for row in rows[1:]:
            d = per_image[row["variant"]].set_index("stem")["dice"]
            row["mean_abs_dice_diff_vs_pytorch"] = float((d - ref.reindex(d.index)).abs().mean())

    out = resolve_path(args.out)
    save_json({"n_test": len(test_df), "n_healthy": len(healthy_df), "threshold": args.threshold,
               "results": rows}, out / "deployment_benchmark.json")
    def fmt(v, spec, pct=False):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        return f"{100 * v:.1f} %" if pct else format(v, spec)

    table = pd.DataFrame([{
        "Variant": r["variant"], "Size (MB)": fmt(r["size_mb"], ".1f"), "Test Dice": fmt(r["test_dice"], ".4f"),
        "IoU": fmt(r["test_iou"], ".4f"), "Healthy FP (≥50 px)": fmt(r["healthy_fp_rate_ge_50px"], "", pct=True),
        "Model ms/slice": fmt(r["model_ms_per_image"], ".1f"),
        "End-to-end ms/slice": fmt(r["end_to_end_ms_per_image"], ".1f"),
        "|ΔDice| vs PyTorch": fmt(r.get("mean_abs_dice_diff_vs_pytorch", float("nan")), ".4f"),
    } for r in rows])
    md = to_markdown(table)
    (out / "deployment_benchmark.md").write_text(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
