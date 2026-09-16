"""Deployment inference: ONNX Runtime pipeline from an MRI slice to a tumour mask.

No PyTorch needed. Preprocessing matches training exactly: grayscale -> 3 channels,
bilinear resize to the model size, ImageNet normalisation. The ONNX model outputs a
probability map (sigmoid included at export), which is resized back to the input size.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from src.analysis import component_areas, contour_overlay, fill_overlay, remove_small_components

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32)

# A prediction counts as "tumour detected" if its largest blob has at least this many pixels.
DEFAULT_DETECTION_AREA = 50


def to_rgb_gray(image: np.ndarray) -> np.ndarray:
    """Any input (gray, RGB, RGBA) -> 3 identical channels, as seen during training."""
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)


def preprocess(image_rgb: np.ndarray, size: int) -> np.ndarray:
    """uint8 RGB (H, W, 3) -> float32 tensor (1, 3, size, size)."""
    x = cv2.resize(image_rgb, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def find_default_model() -> Path:
    for name in ("unet_resnet34_int8.onnx", "unet_resnet34_fp32.onnx"):
        if (MODELS_DIR / name).exists():
            return MODELS_DIR / name
    found = sorted(MODELS_DIR.glob("*.onnx"))
    if not found:
        raise FileNotFoundError("No ONNX model in models/. Run `python -m src.export_onnx --run runs/unet_resnet34`.")
    return found[0]


def load_model_card(model_path: Path) -> dict:
    card = Path(model_path).parent / "model_card.json"
    return json.loads(card.read_text()) if card.exists() else {}


@dataclass
class SegmentationResult:
    prob: np.ndarray
    mask: np.ndarray
    stats: dict = field(default_factory=dict)


class OnnxSegmenter:
    def __init__(self, model_path: str | Path | None = None, providers: list[str] | None = None,
                 session=None, num_threads: int | None = None):
        self.model_path = Path(model_path) if model_path else find_default_model()
        if session is None:
            import onnxruntime as ort

            options = ort.SessionOptions()
            if num_threads:
                options.intra_op_num_threads = num_threads
            session = ort.InferenceSession(str(self.model_path), options,
                                           providers=providers or ["CPUExecutionProvider"])
        self.session = session
        inp = session.get_inputs()[0]
        self.input_name = inp.name
        self.size = int(inp.shape[2]) if isinstance(inp.shape[2], int) else 256
        self.card = load_model_card(self.model_path)
        self.default_threshold = float(self.card.get("threshold", 0.5))

    def predict_prob(self, image: np.ndarray) -> np.ndarray:
        """Probability map at the input image's resolution."""
        rgb = to_rgb_gray(image)
        out = self.session.run(None, {self.input_name: preprocess(rgb, self.size)})[0]
        prob = np.asarray(out, np.float32).reshape(self.size, self.size)
        h, w = rgb.shape[:2]
        return np.clip(cv2.resize(prob, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0)

    def segment(self, image: np.ndarray, threshold: float | None = None, min_area: int = 0,
                detection_area: int = DEFAULT_DETECTION_AREA) -> SegmentationResult:
        threshold = self.default_threshold if threshold is None else float(threshold)
        prob = self.predict_prob(image)
        mask = remove_small_components(prob > threshold, int(min_area))
        return SegmentationResult(prob, mask, describe(prob, mask, detection_area))


def describe(prob: np.ndarray, mask: np.ndarray, detection_area: int = DEFAULT_DETECTION_AREA) -> dict:
    areas = component_areas(mask)
    stats = {
        "image_size": [int(mask.shape[1]), int(mask.shape[0])],
        "tumour_detected": bool(len(areas) and areas.max() >= detection_area),
        "tumour_area_px": int(mask.sum()),
        "tumour_area_pct": round(100 * float(mask.mean()), 3),
        "n_regions": int(len(areas)),
        "largest_region_px": int(areas.max()) if len(areas) else 0,
        "max_probability": round(float(prob.max()), 4),
        "mean_probability_in_mask": round(float(prob[mask].mean()), 4) if mask.any() else None,
        "bounding_box_xywh": None,
    }
    if mask.any():
        x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        stats["bounding_box_xywh"] = [int(x), int(y), int(w), int(h)]
    return stats


def render_overlay(image: np.ndarray, result: SegmentationResult) -> np.ndarray:
    rgb = to_rgb_gray(image)
    out = contour_overlay(fill_overlay(rgb, result.mask, (230, 50, 50), 0.35), result.mask, (255, 215, 0), 2)
    box = result.stats.get("bounding_box_xywh")
    if box:
        x, y, w, h = box
        cv2.rectangle(out, (x, y), (x + w, y + h), (80, 200, 255), 1)
    return out


def render_heatmap(image: np.ndarray, prob: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    rgb = to_rgb_gray(image)
    heat = cv2.applyColorMap((prob * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb, 1 - alpha, heat, alpha, 0)
