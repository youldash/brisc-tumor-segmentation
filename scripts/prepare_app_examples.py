"""Copy a few BRISC 2025 test slices into app/examples/ for the demo.

One typical slice per tumour type (closest to that type's median Dice, if an evaluation
exists), one hard glioma case and one healthy slice. BRISC 2025 is CC BY 4.0.

    python scripts/prepare_app_examples.py
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

DATA_ROOT = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
EVAL = ROOT / "runs" / "unet_resnet34" / "eval" / "test_per_image.csv"
OUT = ROOT / "app" / "examples"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(SPLITS / "test.csv")
    if EVAL.exists():
        test = test.merge(pd.read_csv(EVAL)[["stem", "dice"]], on="stem")
    picks = []
    for t, g in test.groupby("tumor_type"):
        if "dice" in g:
            row = g.iloc[(g["dice"] - g["dice"].median()).abs().argsort().iloc[0]]
        else:
            row = g.sample(n=1, random_state=0).iloc[0]
        picks.append((f"{t}_{row['plane']}", row["image"]))
    if "dice" in test:
        hard = test[test["tumor_type"] == "glioma"].sort_values("dice").iloc[len(test[test["tumor_type"] == "glioma"]) // 10]
        picks.append((f"glioma_{hard['plane']}_hard", hard["image"]))
    healthy = pd.read_csv(SPLITS / "healthy_test.csv")
    if len(healthy):
        row = healthy.sample(n=1, random_state=0).iloc[0]
        picks.append((f"healthy_{row['plane']}", row["image"]))

    for name, rel in picks:
        target = OUT / f"{name}.jpg"
        shutil.copy(DATA_ROOT / rel, target)
        print(f"{rel} -> {target.relative_to(ROOT)}")
    (OUT / "SOURCE.md").write_text(
        "# Example images\n\nSlices from the **BRISC 2025** test set "
        "(https://www.kaggle.com/datasets/briscdataset/brisc2025), licensed under CC BY 4.0. "
        "Used unmodified for demonstration.\n\n" + "\n".join(f"- `{n}.jpg` ← `{r}`" for n, r in picks) + "\n")


if __name__ == "__main__":
    main()
