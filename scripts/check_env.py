"""Verify the Python environment and benchmark training speed on this Mac.

    python scripts/check_env.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def version(module_name):
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:  # noqa: BLE001
        return f"MISSING ({exc.__class__.__name__})"


print(f"Python                : {sys.version.split()[0]}")
for name in ["torch", "torchvision", "segmentation_models_pytorch", "timm", "albumentations",
             "cv2", "sklearn", "onnx", "onnxruntime", "gradio"]:
    print(f"{name:22s}: {version(name)}")

import torch  # noqa: E402

from src.models import UNet, count_parameters  # noqa: E402

print(f"\nMPS built / available : {torch.backends.mps.is_built()} / {torch.backends.mps.is_available()}")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = UNet(base_channels=32).to(device)
opt = torch.optim.AdamW(model.parameters(), 1e-3)
x = torch.randn(8, 3, 256, 256, device=device)
y = (torch.rand(8, 1, 256, 256, device=device) > 0.9).float()
loss_fn = torch.nn.BCEWithLogitsLoss()

print(f"Benchmarking U-Net ({count_parameters(model) / 1e6:.1f} M params), batch 8 @ 256px on {device} ...")
for _ in range(3):  # warm-up (kernel compilation)
    opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
if device.type == "mps":
    torch.mps.synchronize()
steps = 10
t0 = time.time()
for _ in range(steps):
    opt.zero_grad(); loss_fn(model(x), y).backward(); opt.step()
if device.type == "mps":
    torch.mps.synchronize()
per_step = (time.time() - t0) / steps
print(f"≈ {per_step:.3f} s / step  ->  ≈ {8 / per_step:.0f} images/s")
print(f"Rough estimate for ~4,000 training images: ≈ {4000 / 8 * per_step / 60:.1f} min per epoch "
      "(compute only, excluding data loading)")
