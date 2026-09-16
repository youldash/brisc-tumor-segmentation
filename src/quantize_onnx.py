"""Static INT8 quantization (ONNX Runtime, QDQ format) of the exported FP32 model.

Calibration uses slices from the TRAINING split only (never the test set).

Usage (from the project root):
    python -m src.quantize_onnx --model models/unet_resnet34_fp32.onnx
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import pandas as pd

from src.data_index import read_image_rgb
from src.inference import preprocess
from src.utils import resolve_path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/unet_resnet34_fp32.onnx")
    p.add_argument("--out", help="default: <model name with fp32 -> int8>")
    p.add_argument("--data-root", default="data/raw")
    p.add_argument("--splits-dir", default="data/splits")
    p.add_argument("--n-calib", type=int, default=120)
    p.add_argument("--method", choices=["minmax", "percentile"], default="minmax",
                   help="percentile keeps all activations in memory: use with --n-calib <= 20 on 16 GB")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def calibration_frames(splits_dir: Path, n: int, seed: int) -> pd.DataFrame:
    """Stratified sample of training slices (tumour types x planes) plus some healthy slices."""
    train = pd.read_csv(splits_dir / "train.csv")
    per_group = max(1, n // max(1, train.groupby(["tumor_type", "plane"]).ngroups))
    tumour = train.groupby(["tumor_type", "plane"], group_keys=False).apply(
        lambda g: g.sample(n=min(per_group, len(g)), random_state=seed))
    healthy_path = splits_dir / "healthy_train.csv"
    healthy = pd.read_csv(healthy_path).fillna({"mask": ""}) if healthy_path.exists() else pd.DataFrame()
    if len(healthy):
        healthy = healthy.sample(n=min(len(healthy), max(1, n // 10)), random_state=seed)
    return pd.concat([tumour, healthy], ignore_index=True)


def main():
    args = parse_args()
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)

    model_path = resolve_path(args.model)
    out = resolve_path(args.out) if args.out else model_path.with_name(model_path.name.replace("fp32", "int8"))
    if out == model_path:
        out = model_path.with_name(model_path.stem + "_int8.onnx")
    size = 256
    frames = calibration_frames(resolve_path(args.splits_dir), args.n_calib, args.seed)
    root = resolve_path(args.data_root)
    print(f"Calibrating on {len(frames)} training slices ({args.method})")

    class Reader(CalibrationDataReader):
        def __init__(self, input_name: str):
            self.input_name = input_name
            self.paths = list(frames["image"])
            self.i = 0

        def get_next(self):
            if self.i >= len(self.paths):
                return None
            x = preprocess(read_image_rgb(root / self.paths[self.i]), size)
            self.i += 1
            return {self.input_name: x}

        def rewind(self):
            self.i = 0

    import onnxruntime as ort

    input_name = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]).get_inputs()[0].name

    with tempfile.TemporaryDirectory() as tmp:
        prepared = Path(tmp) / "prepared.onnx"
        try:  # shape inference + graph optimisation recommended before quantization
            from onnxruntime.quantization.shape_inference import quant_pre_process

            quant_pre_process(str(model_path), str(prepared), skip_symbolic_shape=True)
            source = prepared
        except Exception as exc:  # noqa: BLE001
            print(f"Pre-processing skipped ({exc.__class__.__name__}: {exc}); quantizing the raw model")
            source = model_path

        method = {"minmax": CalibrationMethod.MinMax, "percentile": CalibrationMethod.Percentile}[args.method]
        # MinMax: merge activation ranges every few images so memory stays bounded on a 16 GB Mac
        extra = {"CalibMaxIntermediateOutputs": 4} if args.method == "minmax" else {}
        quantize_static(
            str(source), str(out), Reader(input_name),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            calibrate_method=method,
            extra_options=extra,
        )

    fp32_mb = model_path.stat().st_size / 1e6
    data_file = model_path.with_name(model_path.name + ".data")
    if data_file.exists():
        fp32_mb += data_file.stat().st_size / 1e6
    int8_mb = out.stat().st_size / 1e6
    print(f"INT8 model -> {out}  ({fp32_mb:.1f} MB -> {int8_mb:.1f} MB, {fp32_mb / int8_mb:.1f}x smaller)")

    card_path = model_path.parent / "model_card.json"
    if card_path.exists():
        card = json.loads(card_path.read_text())
        card["int8_file"] = out.name
        card["quantization"] = {"format": "QDQ", "weights": "int8 per-channel", "activations": "uint8",
                                "calibration": f"{args.method}, {len(frames)} training slices",
                                "size_mb": {"fp32": round(fp32_mb, 1), "int8": round(int8_mb, 1)}}
        card_path.write_text(json.dumps(card, indent=2))


if __name__ == "__main__":
    main()
