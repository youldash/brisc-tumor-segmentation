"""Draw the project diagrams used in the README.

    python scripts/make_diagrams.py

Writes assets/workflow_diagram.png, assets/inference_pipeline.png and assets/model_architecture.png.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ASSETS = Path(__file__).resolve().parents[1] / "assets"
LANES = {
    "data": ("#e8f1fb", "#2f6db5"),
    "model": ("#eaf6ec", "#2e8b57"),
    "eval": ("#fdf2e6", "#d9822b"),
    "deploy": ("#f3ecfa", "#7b4fb3"),
}


def box(ax, x, y, w, h, title, body, lane, fontsize=8.6):
    fill, edge = LANES[lane]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc="white", ec=edge, lw=1.6, zorder=3))
    ax.text(x + w / 2, y + h - 0.13, title, ha="center", va="top", fontsize=fontsize + 0.8,
            weight="bold", color=edge, zorder=4)
    ax.text(x + w / 2, y + h - 0.42, body, ha="center", va="top", fontsize=fontsize,
            color="#333333", zorder=4, linespacing=1.35)


def arrow(ax, p, q, color="#555555", style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=13, color=color, lw=1.4,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2, linestyle=ls))


def workflow():
    fig, ax = plt.subplots(figsize=(17, 9.4))
    ax.set_xlim(0, 17); ax.set_ylim(0.75, 10.05); ax.axis("off")
    lanes = [("data", 7.85, "1 · DATA"), ("model", 5.55, "2 · MODEL"),
             ("eval", 3.25, "3 · EVALUATION"), ("deploy", 0.95, "4 · DEPLOYMENT")]
    for lane, y, name in lanes:
        fill, edge = LANES[lane]
        ax.add_patch(FancyBboxPatch((0.15, y - 0.15), 16.7, 2.15, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=fill, ec="none", zorder=0))
        ax.text(0.45, y + 0.93, name, rotation=90, ha="center", va="center", fontsize=11, weight="bold", color=edge)

    W, H = 3.35, 1.85
    xs = [0.9, 4.85, 8.8, 12.75]
    y = 7.85
    box(ax, xs[0], y, W, H, "BRISC 2025 (Kaggle)",
        "6,000 T1-CE MRI slices, CC BY 4.0\nsegmentation: 3,933 train / 860 test\n(tumour slices only)\nhealthy: 1,067 / 140 (classification)", "data")
    box(ax, xs[1], y, W, H, "Indexing & splits",
        "src/prepare_data.py\npair images ↔ masks, parse type/plane\nstratified 85/15 train/val split\nleakage & integrity checks", "data")
    box(ax, xs[2], y, W, H, "Exploratory analysis",
        "class / plane balance\ntumour size & location priors\nmixed image sizes, JPEG-like masks\n→ Dice-based loss & metrics", "data")
    box(ax, xs[3], y, W, H, "Preprocessing",
        "gray → 3 channels, resize 256 px\nImageNet normalisation, mask ≥ 127\naugment (train only): flip, affine,\ncontrast, gamma, elastic", "data")

    y = 5.55
    box(ax, xs[0], y, W, H, "U-Net + ResNet34",
        "ImageNet-pretrained encoder\n(transfer learning), 24.4 M params\nsegmentation_models_pytorch", "model")
    box(ax, xs[1], y, W, H, "Training on Apple M4 (MPS)",
        "src/train.py · BCE + Dice loss\nAdamW, warm-up + cosine LR\nencoder LR × 0.3, batch 8\n40 epochs ≈ 1.9 h (≈ 170 s / epoch)", "model")
    box(ax, xs[2], y, W, H, "Model selection",
        "validation Dice each epoch\nbest checkpoint (epoch 35)\nearly stopping, resumable\nhistory.csv / summary.json", "model")
    box(ax, xs[3], y, W, H, "Validation checks",
        "shape / loss / metric unit tests\nhistory ↔ checkpoint consistency\nsynthetic smoke test", "model")

    y = 3.25
    box(ax, xs[0], y, W, H, "Test evaluation",
        "src/evaluate.py · 860 slices\nupsample to native resolution\nDice, IoU, precision, recall\nby type, plane, tumour size", "eval")
    box(ax, xs[1], y, W, H, "Healthy-slice analysis",
        "140 healthy test slices\nfalse-positive rate\ntumour-present ROC / AUC", "eval")
    box(ax, xs[2], y, W, H, "Improvement experiments",
        "min-area post-processing\nflip test-time augmentation\nthreshold tuned on validation", "eval")
    box(ax, xs[3], y, W, H, "Analysis & report",
        "notebook 02 · statistics, galleries\nsuccess / failure cases\nsrc/report.py → README", "eval")

    y = 0.95
    box(ax, xs[0], y, W, H, "ONNX export",
        "src/export_onnx.py\nsigmoid head, opset 18\nparity check vs PyTorch\nmodel_card.json", "deploy")
    box(ax, xs[1], y, W, H, "INT8 quantization",
        "src/quantize_onnx.py\nstatic QDQ, per-channel weights\ncalibrated on training slices", "deploy")
    box(ax, xs[2], y, W, H, "Deployment benchmark",
        "src/benchmark_deploy.py\naccuracy · latency · size\nFP32 vs INT8 (CPU / Core ML)", "deploy")
    box(ax, xs[3], y, W, H, "Application",
        "Gradio web demo (app/app.py)\nCLI batch tool (predict_cli.py)\nmask, overlay, heat-map, stats", "deploy")

    for row_y in (7.85, 5.55, 3.25, 0.95):
        for a, b in zip(xs[:-1], xs[1:]):
            arrow(ax, (a + W, row_y + H / 2), (b, row_y + H / 2))
    # lane transitions (right end → left start of next lane)
    for y_from, y_to in ((7.85, 5.55), (5.55, 3.25), (3.25, 0.95)):
        arrow(ax, (xs[3] + W / 2, y_from - 0.02), (xs[0] + W / 2, y_to + H + 0.02), color="#888888", rad=0.0, ls="--")
    fig.suptitle("Brain tumour segmentation on MRI (BRISC 2025) — project workflow", fontsize=15, weight="bold", y=0.985)
    fig.savefig(ASSETS / "workflow_diagram.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def inference_pipeline():
    fig, ax = plt.subplots(figsize=(17, 2.9))
    ax.set_xlim(0, 17.2); ax.set_ylim(0, 2.6); ax.axis("off")
    steps = [
        ("Input", "MRI slice\nany size, gray/RGB\n(JPEG / PNG)", "data"),
        ("Preprocess", "→ grayscale → 3 ch\nresize 256×256\nImageNet normalise", "data"),
        ("ONNX model", "U-Net + ResNet34\nINT8 / FP32\nONNX Runtime", "model"),
        ("Probability map", "sigmoid output\n256×256\nupsample to input size", "model"),
        ("Post-process", "threshold (0.5)\noptional min-area\nconnected components", "eval"),
        ("Outputs", "tumour mask & overlay\nprobability heat-map\narea %, bbox, detected?", "deploy"),
    ]
    w, gap = 2.4, 0.42
    for i, (t, b, lane) in enumerate(steps):
        x = 0.2 + i * (w + gap)
        box(ax, x, 0.75, w, 1.6, t, b, lane, fontsize=9)
        if i:
            arrow(ax, (x - gap, 1.55), (x, 1.55))
    ax.text(8.6, 0.35, "Runtime path shared by the Gradio app, the CLI and the deployment benchmark (src/inference.py) "
            "— no PyTorch required", ha="center", fontsize=9.5, style="italic", color="#444444")
    fig.suptitle("Inference pipeline (system diagram)", fontsize=13, weight="bold")
    fig.savefig(ASSETS / "inference_pipeline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(16, 7.2))
    ax.set_xlim(0, 18.9); ax.set_ylim(0.2, 7.6); ax.axis("off")
    rows = [6.3, 5.3, 4.3, 3.3, 2.3, 1.3]  # one row per spatial resolution
    enc = [("Input slice", "3 × 256 × 256"), ("Stem: conv 7×7, BN, ReLU", "64 × 128 × 128"),
           ("ResNet34 layer1 (3 blocks)", "64 × 64 × 64"), ("ResNet34 layer2 (4 blocks)", "128 × 32 × 32"),
           ("ResNet34 layer3 (6 blocks)", "256 × 16 × 16"), ("ResNet34 layer4 (3 blocks)", "512 × 8 × 8")]
    dec = {6.3: ("Decoder block 5", "16 × 256 × 256"), 5.3: ("Decoder block 4", "32 × 128 × 128"),
           4.3: ("Decoder block 3", "64 × 64 × 64"), 3.3: ("Decoder block 2", "128 × 32 × 32"),
           2.3: ("Decoder block 1", "256 × 16 × 16")}
    ew, eh = 3.6, 0.78
    ex = [0.5 + i * 0.3 for i in range(6)]
    dx = {y: 11.0 - i * 0.3 for i, y in enumerate(rows[:5])}
    for i, ((t, sub), y) in enumerate(zip(enc, rows)):
        box(ax, ex[i], y, ew, eh, t, sub, "data" if i == 0 else "model", fontsize=8.5)
        if i:
            arrow(ax, (ex[i] + 1.0, rows[i - 1]), (ex[i] + 1.0, y + eh))
    for y, (t, sub) in dec.items():
        box(ax, dx[y], y, ew, eh, t, sub, "eval", fontsize=8.5)
    for lower, upper in zip(rows[1:5], rows[:4]):  # decoder goes up: 16² -> 256²
        arrow(ax, (dx[lower] + ew - 1.0, lower + eh), (dx[upper] + ew - 1.0, upper))
    # bottleneck: layer4 -> decoder block 1
    arrow(ax, (ex[5] + ew, 1.3 + eh / 2), (dx[2.3] + 1.2, 2.3), rad=0.25)
    # skip connections at equal resolution
    for i, y in enumerate(rows[1:5], start=1):
        arrow(ax, (ex[i] + ew, y + eh / 2), (dx[y], y + eh / 2), color="#2f6db5", ls="--")
    ax.text(8.1, 5.3 + eh / 2 + 0.12, "skip connections (concatenate)", ha="center", color="#2f6db5", fontsize=9.5)
    box(ax, 15.2, 6.3, 3.5, eh, "Head: conv 3×3 + sigmoid", "1 × 256 × 256 probability", "deploy", fontsize=8.5)
    arrow(ax, (dx[6.3] + ew, 6.3 + eh / 2), (15.2, 6.3 + eh / 2))
    ax.text(2.6, 0.55, "Encoder: ResNet34, ImageNet-pretrained, LR × 0.3", ha="center", fontsize=10,
            weight="bold", color=LANES["model"][1])
    ax.text(12.0, 0.55, "Decoder block: 2× nearest upsample → concat skip → 2 × (conv 3×3, BN, ReLU)",
            ha="center", fontsize=10, weight="bold", color=LANES["eval"][1])
    fig.suptitle("U-Net with a ResNet34 encoder (segmentation_models_pytorch) — 24.4 M parameters",
                 fontsize=13, weight="bold")
    fig.savefig(ASSETS / "model_architecture.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ASSETS.mkdir(exist_ok=True)
    workflow()
    inference_pipeline()
    architecture()
    print("Diagrams written to", ASSETS)
