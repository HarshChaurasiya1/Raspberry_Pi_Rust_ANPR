#!/usr/bin/env python3
"""Export YOLOv8n plate detector model to ONNX and NCNN format.

Run on development machine (PC/Mac/Linux with PyTorch/Ultralytics), NOT on Raspberry Pi.

Workflow:
    1. Train or load YOLOv8n plate detector weights (.pt)
    2. Export to ONNX with baked-in NMS:
       yolo export model=yolov8n_plate.pt format=onnx imgsz=640 opset=12
    3. (Optional) Export with dynamic input size or different resolutions:
       yolo export model=yolov8n_plate.pt format=onnx imgsz=512
    4. Convert to NCNN format for Raspberry Pi deployment.

Usage:
    python tools/export_yolov8n.py --weights yolov8n_plate.pt --imgsz 640 --out-dir models/
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="Export YOLOv8n to ONNX / NCNN for edge ANPR")
    ap.add_argument("--weights", default="yolov8n.pt", help="Path to YOLOv8n weights (.pt)")
    ap.add_argument("--imgsz", type=int, default=640, help="Input image size (640, 512, 416)")
    ap.add_argument("--out-dir", default="models", help="Output directory")
    ap.add_argument("--half", action="store_true", help="Export FP16 ONNX")
    ap.add_argument("--int8", action="store_true", help="Quantize to INT8")
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        print("[NOTE] 'ultralytics' is not installed in this environment.")
        print("To export YOLOv8n on your training machine:")
        print("  pip install ultralytics onnx onnxslim")
        print(f"  python tools/export_yolov8n.py --weights {args.weights} --imgsz {args.imgsz}")
        return 0

    print(f"Loading YOLO model from {args.weights}...")
    model = YOLO(args.weights)

    out_name = f"yolov8n_{args.imgsz}.onnx"
    out_path = os.path.join(args.out_dir, out_name)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Exporting to ONNX at {args.imgsz}x{args.imgsz}...")
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        half=args.half,
        int8=args.int8,
        opset=12,
        simplify=True,
    )

    if exported and os.path.exists(exported):
        shutil.move(exported, out_path)
        print(f"[SUCCESS] Exported ONNX model to: {out_path}")
        print(f"  Model size: {os.path.getsize(out_path) / 1e6:.2f} MB")
    else:
        print(f"[WARNING] Export did not produce expected file at {exported}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
