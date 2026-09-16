"""Gradio demo: upload a brain MRI slice and get the predicted tumour region.

    python app/app.py                       # uses models/unet_resnet34_int8.onnx (or FP32)
    python app/app.py --model models/unet_resnet34_fp32.onnx --share

Educational prototype — not a medical device.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gradio as gr  # noqa: E402

from src.inference import OnnxSegmenter, render_heatmap, render_overlay  # noqa: E402

DISCLAIMER = ("⚠️ **Educational prototype (SDAIA Academy final project). Not a medical device — "
              "do not use for diagnosis.**")


def format_stats(stats: dict, model_name: str, threshold: float, min_area: int) -> str:
    verdict = "🔴 **Tumour region detected**" if stats["tumour_detected"] else "🟢 **No tumour region detected**"
    lines = [
        verdict, "",
        "| Measure | Value |", "|---|---|",
        f"| Predicted tumour area | {stats['tumour_area_px']:,} px ({stats['tumour_area_pct']:.2f} % of slice) |",
        f"| Separate regions | {stats['n_regions']} (largest {stats['largest_region_px']:,} px) |",
        f"| Maximum probability | {stats['max_probability']:.3f} |",
        f"| Mean probability inside region | "
        f"{stats['mean_probability_in_mask']:.3f} |" if stats["mean_probability_in_mask"] is not None
        else "| Mean probability inside region | — |",
        f"| Bounding box (x, y, w, h) | {stats['bounding_box_xywh'] or '—'} |",
        f"| Image size | {stats['image_size'][0]} × {stats['image_size'][1]} |",
        "",
        f"Model `{model_name}` · threshold {threshold:.2f} · min region {min_area} px · "
        "a region counts as detected from 50 px.",
    ]
    return "\n".join(lines)


def build_demo(segmenter: OnnxSegmenter) -> gr.Blocks:
    def predict(image, threshold, min_area):
        if image is None:
            return None, None, "Upload an MRI slice or pick an example."
        result = segmenter.segment(image, threshold=threshold, min_area=int(min_area))
        return (render_overlay(image, result), render_heatmap(image, result.prob),
                format_stats(result.stats, segmenter.model_path.name, threshold, int(min_area)))

    examples = sorted((ROOT / "app" / "examples").glob("*.jpg"))
    with gr.Blocks(title="Brain Tumour Segmentation (BRISC 2025)") as demo:
        gr.Markdown("# 🧠 Brain Tumour Segmentation on MRI\n"
                    "U-Net with an ImageNet-pretrained ResNet34 encoder, trained on BRISC 2025 "
                    "(T1-weighted, contrast-enhanced slices). Runs locally with ONNX Runtime.\n\n" + DISCLAIMER)
        with gr.Row():
            with gr.Column():
                image = gr.Image(type="numpy", label="MRI slice (axial, coronal or sagittal)")
                threshold = gr.Slider(0.05, 0.95, value=segmenter.default_threshold, step=0.05,
                                      label="Probability threshold")
                min_area = gr.Slider(0, 2000, value=0, step=10, label="Remove regions smaller than (px)")
                button = gr.Button("Segment", variant="primary")
                if examples:
                    gr.Examples(examples=[[str(p)] for p in examples], inputs=[image],
                                label="Examples from the BRISC 2025 test set (CC BY 4.0)")
            with gr.Column():
                overlay = gr.Image(label="Predicted tumour region (red, yellow outline)")
                heatmap = gr.Image(label="Tumour probability map")
                stats = gr.Markdown()
        button.click(predict, inputs=[image, threshold, min_area], outputs=[overlay, heatmap, stats])
        image.change(predict, inputs=[image, threshold, min_area], outputs=[overlay, heatmap, stats])
    return demo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="ONNX model (default: models/unet_resnet34_int8.onnx, else FP32)")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--share", action="store_true", help="create a temporary public link")
    args = p.parse_args()
    segmenter = OnnxSegmenter(args.model)
    print(f"Loaded {segmenter.model_path}")
    build_demo(segmenter).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
