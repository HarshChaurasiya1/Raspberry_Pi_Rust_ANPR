#!/usr/bin/env python3
"""Convert ONNX detector models to NCNN format (.param / .bin).

Workflow:
    1. Using pnnx (recommended for PyTorch/TorchScript/ONNX models):
       pnnx detector.onnx inputshape=[1,3,640,640]
       -> generates detector.ncnn.param and detector.ncnn.bin

    2. Using onnx2ncnn (classic NCNN tool):
       onnx2ncnn detector.onnx detector.ncnn.param detector.ncnn.bin
       ncnnoptimize detector.ncnn.param detector.ncnn.bin detector_opt.param detector_opt.bin 0

    3. For INT8 models:
       Note that ONNX INT8 QDQ models may not directly convert to NCNN INT8
       without using ncnn2table + ncnn2int8 calibration workflow.
       Converting FP32/FP16 models is recommended for NCNN.

Usage:
    python tools/convert_ncnn.py --onnx models/detector_int8.onnx --out-dir models/
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys


def find_tool(name: str) -> str | None:
    """Find executable in PATH or standard install locations."""
    path = shutil.which(name)
    if path:
        return path
    # Common locations
    for loc in [
        "/usr/local/bin",
        "/usr/bin",
        os.path.expanduser("~/.local/bin"),
    ]:
        candidate = os.path.join(loc, name)
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def convert_with_pnnx(onnx_path: str, out_dir: str, input_shape: str = "[1,3,640,640]") -> bool:
    """Convert using pnnx tool."""
    pnnx = find_tool("pnnx")
    if not pnnx:
        print("  [INFO] 'pnnx' executable not found in PATH.")
        return False

    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    cmd = [
        pnnx,
        onnx_path,
        f"inputshape={input_shape}",
    ]
    print(f"  Running: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, cwd=out_dir, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            print(f"  [SUCCESS] Converted with pnnx to {out_dir}")
            return True
        else:
            print(f"  [WARNING] pnnx failed (exit code {res.returncode}):\n{res.stderr}")
            return False
    except Exception as e:
        print(f"  [ERROR] Failed to run pnnx: {e}")
        return False


def convert_with_onnx2ncnn(onnx_path: str, out_dir: str) -> bool:
    """Convert using onnx2ncnn and ncnnoptimize."""
    onnx2ncnn = find_tool("onnx2ncnn")
    if not onnx2ncnn:
        print("  [INFO] 'onnx2ncnn' executable not found in PATH.")
        return False

    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    raw_param = os.path.join(out_dir, f"{stem}.raw.param")
    raw_bin = os.path.join(out_dir, f"{stem}.raw.bin")
    opt_param = os.path.join(out_dir, f"{stem}.ncnn.param")
    opt_bin = os.path.join(out_dir, f"{stem}.ncnn.bin")

    cmd1 = [onnx2ncnn, onnx_path, raw_param, raw_bin]
    print(f"  Running: {' '.join(cmd1)}")
    try:
        res = subprocess.run(cmd1, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"  [WARNING] onnx2ncnn failed (exit {res.returncode}): {res.stderr}")
            return False

        ncnnopt = find_tool("ncnnoptimize")
        if ncnnopt:
            cmd2 = [ncnnopt, raw_param, raw_bin, opt_param, opt_bin, "0"]
            print(f"  Running: {' '.join(cmd2)}")
            subprocess.run(cmd2, capture_output=True, text=True, check=False)
            if os.path.exists(raw_param): os.remove(raw_param)
            if os.path.exists(raw_bin): os.remove(raw_bin)
        else:
            shutil.move(raw_param, opt_param)
            shutil.move(raw_bin, opt_bin)

        print(f"  [SUCCESS] Converted to {opt_param} and {opt_bin}")
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to run onnx2ncnn: {e}")
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert ONNX models to NCNN")
    ap.add_argument("--onnx", default="models/detector_int8.onnx", help="Input ONNX model")
    ap.add_argument("--out-dir", default="models", help="Output directory")
    ap.add_argument("--shape", default="[1,3,640,640]", help="Input tensor shape")
    args = ap.parse_args()

    if not os.path.isfile(args.onnx):
        print(f"error: input file '{args.onnx}' not found", file=sys.stderr)
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Converting {args.onnx} -> NCNN (.param / .bin)...")

    # Try pnnx first, then onnx2ncnn
    if convert_with_pnnx(args.onnx, args.out_dir, args.shape):
        return 0
    if convert_with_onnx2ncnn(args.onnx, args.out_dir):
        return 0

    print("\n[NOTE] Neither 'pnnx' nor 'onnx2ncnn' is installed on this system.")
    print("To convert models on Raspberry Pi or Linux development machine:")
    print("  1. Install pnnx: pip install pnnx (or download prebuilt binaries from github.com/Tencent/ncnn/releases)")
    print(f"  2. Run: pnnx {args.onnx} inputshape={args.shape}")
    print(f"  3. Copy .ncnn.param and .ncnn.bin to {args.out_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
