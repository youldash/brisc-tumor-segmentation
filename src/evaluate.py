"""Evaluate a trained run on the held-out test set.

Predictions are made at the training resolution (256 px) and the probability map is
upsampled back to the native image size, so metrics are computed against the original
expert masks. Also measures false positives on healthy slices and inference speed.

Usage (from the project root):
    python -m src.evaluate --run runs/unet_resnet34
    python -m src.evaluate --run runs/unet_resnet34 --tta --min-area 50   # improvement experiments

Outputs: runs/<run>/<out-name>/ with metrics.json, report.md, per-image CSVs and galleries.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402

from src.analysis import (add_size_buckets, component_areas, grouped_metrics, healthy_metrics,  # noqa: E402
                          metrics_table, overall_metrics, plot_gallery, remove_small_components,
                          to_markdown)
from src.data_index import load_pair  # noqa: E402
from src.metrics import confusion_counts, per_image_scores  # noqa: E402
from src.models import build_model  # noqa: E402
from src.transforms import get_eval_transforms  # noqa: E402
from src.utils import get_device, resolve_path, save_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory containing best.pt")
    p.add_argument("--tta", action="store_true", help="average with a horizontally flipped prediction")
    p.add_argument("--min-area", type=int, default=0, help="remove predicted blobs smaller than this (px)")
    p.add_argument("--threshold", type=float, help="override the probability threshold from the config")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--n-gallery", type=int, default=4)
    p.add_argument("--out-name", help="output sub-folder (default derived from the options)")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def load_model(run_dir: Path, device):
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_cfg = dict(cfg["model"], encoder_weights=None)  # weights come from the checkpoint
    model = build_model(model_cfg)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg, ckpt


class Predictor:
    """Batched inference returning probability maps at each image's native size."""

    def __init__(self, model, device, img_size: int, tta: bool = False):
        self.model, self.device, self.tta = model, device, tta
        self.transform = get_eval_transforms(img_size)
        self.forward_seconds, self.n_images = 0.0, 0

    def _sync(self):
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.no_grad()
    def __call__(self, images: list[np.ndarray]) -> list[np.ndarray]:
        x = torch.stack([
            torch.from_numpy(np.ascontiguousarray(self.transform(image=im)["image"].transpose(2, 0, 1)))
            for im in images
        ]).float().to(self.device)
        self._sync()
        t0 = time.perf_counter()
        probs = torch.sigmoid(self.model(x))
        if self.tta:
            flipped = torch.sigmoid(self.model(torch.flip(x, dims=[3])))
            probs = 0.5 * (probs + torch.flip(flipped, dims=[3]))
        self._sync()
        self.forward_seconds += time.perf_counter() - t0
        self.n_images += len(images)

        out = []
        for p, im in zip(probs, images):
            up = F.interpolate(p.unsqueeze(0), size=im.shape[:2], mode="bilinear", align_corners=False)
            out.append(up[0, 0].float().cpu().numpy())
        return out


def postprocess(prob: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    return remove_small_components(prob > threshold, min_area)


def run_split(predictor, df, root, batch_size, threshold, min_area, desc):
    rows = []
    for start in tqdm(range(0, len(df), batch_size), desc=desc):
        chunk = df.iloc[start:start + batch_size]
        pairs = [load_pair(root, r) for _, r in chunk.iterrows()]
        probs = predictor([im for im, _ in pairs])
        for (_, r), (_, gt), prob in zip(chunk.iterrows(), pairs, probs):
            pred = postprocess(prob, threshold, min_area)
            c = confusion_counts(pred[None], gt[None].astype(bool))[0]
            s = per_image_scores(c[None])
            areas = component_areas(pred)
            rows.append({
                "stem": r["stem"], "tumor_type": r["tumor_type"], "plane": r["plane"],
                "dice": s["dice"][0], "iou": s["iou"][0],
                "precision": s["precision"][0], "recall": s["recall"][0],
                "tp": int(c[0]), "fp": int(c[1]), "fn": int(c[2]), "tn": int(c[3]),
                "gt_area_frac": float(gt.mean()), "pred_area_frac": float(pred.mean()),
                "n_pred_components": int(len(areas)),
                "largest_component_px": int(areas.max()) if len(areas) else 0,
                "max_prob": float(prob.max()),
                "height": int(gt.shape[0]), "width": int(gt.shape[1]),
            })
    return pd.DataFrame(rows)


def gallery_items(predictor, df_rows, index_df, root, threshold, min_area):
    lookup = index_df.set_index("stem")
    items = []
    for _, r in df_rows.iterrows():
        image, gt = load_pair(root, lookup.loc[r["stem"]])
        prob = predictor([image])[0]
        items.append({
            "image": image, "gt": gt, "prob": prob, "pred": postprocess(prob, threshold, min_area),
            "caption": f"{r['tumor_type']} · {r['plane']}\nDice {r['dice']:.3f} · "
                       f"tumour {100 * r['gt_area_frac']:.2f}% of slice",
        })
    return items


def main():
    args = parse_args()
    run_dir = resolve_path(args.run)
    device = get_device(args.device)
    model, cfg, ckpt = load_model(run_dir, device)
    threshold = args.threshold if args.threshold is not None else float(cfg["train"]["threshold"])

    out_name = args.out_name or "eval" + ("_tta" if args.tta else "") + (
        f"_min{args.min_area}" if args.min_area else "") + (
        f"_thr{threshold:.2f}" if args.threshold is not None else "")
    out_dir = run_dir / out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    root = resolve_path(cfg["data"]["root"])
    splits_dir = resolve_path(cfg["data"]["splits_dir"])
    test_df = pd.read_csv(splits_dir / "test.csv")
    healthy_path = splits_dir / "healthy_test.csv"
    healthy_df = pd.read_csv(healthy_path).fillna({"mask": ""}) if healthy_path.exists() else pd.DataFrame()

    predictor = Predictor(model, device, int(cfg["data"]["img_size"]), tta=args.tta)
    warm = [load_pair(root, r)[0] for _, r in test_df.head(args.batch_size).iterrows()]
    predictor(warm)  # warm-up (kernel compilation) is excluded from timing
    predictor.forward_seconds, predictor.n_images = 0.0, 0

    print(f"Evaluating {cfg['run_name']} (best epoch {ckpt['epoch']}) on {device} | "
          f"threshold {threshold} | TTA {args.tta} | min-area {args.min_area}")
    test_res = run_split(predictor, test_df, root, args.batch_size, threshold, args.min_area, "test")
    test_res, size_edges = add_size_buckets(test_res)
    ms_per_image = 1000 * predictor.forward_seconds / max(predictor.n_images, 1)

    healthy_res = pd.DataFrame()
    if len(healthy_df):
        healthy_res = run_split(predictor, healthy_df, root, args.batch_size, threshold,
                                args.min_area, "healthy")

    metrics = {
        "run_name": cfg["run_name"],
        "model": cfg["model"],
        "best_epoch": int(ckpt["epoch"]),
        "settings": {"threshold": threshold, "tta": args.tta, "min_area": args.min_area,
                     "img_size": cfg["data"]["img_size"], "metrics_resolution": "native"},
        "overall": overall_metrics(test_res),
        "by_tumor_type": grouped_metrics(test_res, "tumor_type"),
        "by_plane": grouped_metrics(test_res, "plane"),
        "by_tumor_size": grouped_metrics(test_res, "size_bucket"),
        "size_bucket_edges_area_frac": size_edges,
        "healthy": healthy_metrics(healthy_res),
        "speed": {"device": str(device), "batch_size": args.batch_size,
                  "ms_per_image_forward": ms_per_image, "images_per_second": 1000 / max(ms_per_image, 1e-9)},
    }
    save_json(metrics, out_dir / "metrics.json")
    test_res.to_csv(out_dir / "test_per_image.csv", index=False)
    if len(healthy_res):
        healthy_res.to_csv(out_dir / "healthy_per_image.csv", index=False)

    # Galleries: best / worst test predictions, and the worst false positives on healthy slices
    n = args.n_gallery
    ranked = test_res.sort_values("dice")
    plot_gallery(gallery_items(predictor, ranked.tail(n).iloc[::-1], test_df, root, threshold, args.min_area),
                 f"{cfg['run_name']} — best test predictions", out_dir / "gallery_best.png")
    plot_gallery(gallery_items(predictor, ranked.head(n), test_df, root, threshold, args.min_area),
                 f"{cfg['run_name']} — worst test predictions (failure cases)", out_dir / "gallery_worst.png")
    near_median = test_res.iloc[(test_res["dice"] - test_res["dice"].median()).abs().argsort()[:n]]
    plot_gallery(gallery_items(predictor, near_median, test_df, root, threshold, args.min_area),
                 f"{cfg['run_name']} — typical predictions (near median Dice)", out_dir / "gallery_typical.png")
    if len(healthy_res) and (healthy_res["largest_component_px"] > 0).any():
        fp = healthy_res[healthy_res["largest_component_px"] > 0].sort_values(
            "largest_component_px", ascending=False).head(n)
        plot_gallery(gallery_items(predictor, fp, healthy_df, root, threshold, args.min_area),
                     f"{cfg['run_name']} — false positives on healthy slices", out_dir / "gallery_healthy_fp.png")

    # Human-readable report
    o, h = metrics["overall"], metrics["healthy"]
    lines = [
        f"# Evaluation — {cfg['run_name']}", "",
        f"Best epoch {ckpt['epoch']} · threshold {threshold} · TTA {args.tta} · min-area {args.min_area} px · "
        f"metrics at native resolution", "",
        "## Overall (test set)", "",
        to_markdown(pd.DataFrame([{"n": o["n"], "Dice (mean)": o["dice"], "Dice (median)": o["dice_median"],
                                   "IoU": o["iou"], "Precision": o["precision"], "Recall": o["recall"],
                                   "Global Dice": o["global_dice"]}])), "",
        "## By tumour type", "", to_markdown(metrics_table(metrics["by_tumor_type"], "Tumour type")), "",
        "## By imaging plane", "", to_markdown(metrics_table(metrics["by_plane"], "Plane")), "",
        "## By tumour size (tertiles of slice area)", "",
        to_markdown(metrics_table(metrics["by_tumor_size"], "Size")), "",
        "Bucket edges (fraction of slice): " + ", ".join(f"{e:.4f}" for e in size_edges), "",
    ]
    if h.get("n"):
        lines += ["## Healthy slices (false positives)", "",
                  f"{h['n']} slices without tumour. Share of slices with a predicted blob of at least "
                  f"1 px: {h['fp_rate_largest_ge_1px']:.1%} · ≥ 50 px: {h['fp_rate_largest_ge_50px']:.1%} · "
                  f"≥ 200 px: {h['fp_rate_largest_ge_200px']:.1%}", ""]
    lines += ["## Speed", "",
              f"{ms_per_image:.1f} ms / image (model forward, batch {args.batch_size}, {device})", ""]
    (out_dir / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
