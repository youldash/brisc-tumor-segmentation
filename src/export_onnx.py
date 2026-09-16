"""Export the best checkpoint of a run to ONNX (FP32) and verify it against PyTorch.

Usage (from the project root):
    python -m src.export_onnx --run runs/unet_resnet34

Writes models/<run>_fp32.onnx and models/model_card.json. The exported graph takes a
(1, 3, 256, 256) normalised image and returns a (1, 1, 256, 256) tumour probability map.
"""
from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import argparse  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.data_index import load_pair  # noqa: E402
from src.evaluate import Predictor, load_model  # noqa: E402
from src.inference import MODELS_DIR, OnnxSegmenter  # noqa: E402
from src.metrics import confusion_counts, summarize  # noqa: E402
from src.utils import resolve_path  # noqa: E402


class ProbabilityModel(torch.nn.Module):
    """Wraps the segmentation network so the exported graph outputs probabilities."""

    def __init__(self, net: torch.nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="runs/unet_resnet34")
    p.add_argument("--out", help="output .onnx path (default models/<run>_fp32.onnx)")
    p.add_argument("--opset", type=int, default=18)
    p.add_argument("--n-check", type=int, default=50, help="test images used for the parity check")
    return p.parse_args()


def export(model: torch.nn.Module, dummy: torch.Tensor, out: Path, opset: int) -> str:
    """Try the modern (dynamo) exporter first, then the legacy TorchScript exporter."""
    kwargs = dict(input_names=["image"], output_names=["probability"], opset_version=opset)
    attempts = [("dynamo", dict(dynamo=True, external_data=False)),
                ("dynamo", dict(dynamo=True)),
                ("torchscript", dict(dynamo=False, do_constant_folding=True))]
    errors = []
    for name, extra in attempts:
        try:
            torch.onnx.export(model, (dummy,), str(out), **kwargs, **extra)
            return name
        except TypeError as exc:  # keyword not supported by this torch version
            errors.append(f"{name}: {exc}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name}: {exc.__class__.__name__}: {exc}")
    raise RuntimeError("ONNX export failed:\n" + "\n".join(errors))


def main():
    args = parse_args()
    run_dir = resolve_path(args.run)
    cpu = torch.device("cpu")
    model, cfg, ckpt = load_model(run_dir, cpu)
    size = int(cfg["data"]["img_size"])
    wrapped = ProbabilityModel(model).eval()

    MODELS_DIR.mkdir(exist_ok=True)
    out = Path(args.out) if args.out else MODELS_DIR / f"{cfg['run_name']}_fp32.onnx"
    exporter = export(wrapped, torch.zeros(1, 3, size, size), out, args.opset)
    data_file = out.with_name(out.name + ".data")
    print(f"Exported with the {exporter} exporter -> {out} ({out.stat().st_size / 1e6:.1f} MB)"
          + (f" + external weights {data_file.name}" if data_file.exists() else ""))

    # ---- parity check: PyTorch (CPU) vs ONNX Runtime on real test slices
    root = resolve_path(cfg["data"]["root"])
    test_df = pd.read_csv(resolve_path(cfg["data"]["splits_dir"]) / "test.csv")
    sample = test_df.sample(n=min(args.n_check, len(test_df)), random_state=0)
    torch_pred = Predictor(model, cpu, size)
    onnx_seg = OnnxSegmenter(out)
    threshold = float(cfg["train"]["threshold"])
    max_diff, agree, dice_torch, dice_onnx = 0.0, [], [], []
    for _, row in sample.iterrows():
        image, gt = load_pair(root, row)
        p_t = torch_pred([image])[0]
        p_o = onnx_seg.predict_prob(image)
        max_diff = max(max_diff, float(np.abs(p_t - p_o).max()))
        m_t, m_o = p_t > threshold, p_o > threshold
        agree.append(summarize(confusion_counts(m_o[None], m_t[None]))["dice"])
        dice_torch.append(summarize(confusion_counts(m_t[None], gt[None].astype(bool)))["dice"])
        dice_onnx.append(summarize(confusion_counts(m_o[None], gt[None].astype(bool)))["dice"])
    parity = {"n_images": len(sample), "max_abs_prob_diff": max_diff,
              "mask_agreement_dice": float(np.mean(agree)),
              "dice_pytorch": float(np.mean(dice_torch)), "dice_onnx": float(np.mean(dice_onnx))}
    print("Parity check:", json.dumps(parity, indent=2))
    ok = max_diff < 1e-3 and parity["mask_agreement_dice"] > 0.999
    print("✅ ONNX output matches PyTorch" if ok else "⚠️  ONNX output differs from PyTorch — inspect before deploying")

    metrics_path = run_dir / "eval" / "metrics.json"
    card = {
        "model": cfg["model"], "source_run": cfg["run_name"], "best_epoch": int(ckpt["epoch"]),
        "input": {"name": "image", "shape": [1, 3, size, size], "dtype": "float32",
                  "preprocessing": "grayscale -> 3 channels, bilinear resize, /255, ImageNet mean/std"},
        "output": {"name": "probability", "shape": [1, 1, size, size], "meaning": "tumour probability per pixel"},
        "threshold": threshold, "opset": args.opset, "exporter": exporter,
        "fp32_file": out.name, "parity_check": parity,
        "test_metrics_pytorch": json.loads(metrics_path.read_text())["overall"] if metrics_path.exists() else None,
        "intended_use": "Educational prototype for SDAIA Academy. Not a medical device.",
    }
    (out.parent / "model_card.json").write_text(json.dumps(card, indent=2))
    print(f"Model card -> {out.parent / 'model_card.json'}")


if __name__ == "__main__":
    main()
