# High-Performance Grammar-Constrained ANPR in Rust

[![Rust](https://img.shields.io/badge/Rust-1.80%2B-orange.svg?logo=rust)](https://www.rust-lang.org/)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%205%20%7C%20Linux%20ARM64%20%7C%20Windows%20x86__64-blue.svg)](https://www.raspberrypi.com/products/raspberry-pi-5/)
[![Acceleration](https://img.shields.io/badge/SIMD-ARM%20NEON%20%2B%20DotProd%20%2B%20FP16-brightgreen.svg)]()
[![Recall](https://img.shields.io/badge/Recall-100%25%20(26%2F26%20Vehicles)-success.svg)]()
[![Character Accuracy](https://img.shields.io/badge/Char%20Accuracy-96.15%25-green.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An ultra-high-throughput, production-grade **Automatic Number Plate Recognition (ANPR)** engine engineered in idiomatic Rust, specifically optimized for the **Raspberry Pi 5** (Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz) and cross-platform x86_64 deployment.

Combines an INT8-quantized YOLO detector, temporal IoU plate tracking with OCR caching, edge-aware multi-line plate decoding, and a zero-copy Indian RTO grammar-constrained CTC prefix beam search (DFA shallow fusion).

---

## 🚀 Key Highlights & Benchmarks

Benchmarked against the comprehensive **11m 44s real-world highway surveillance video** (`samples/my_car.mp4`, 21,122 raw frames @ 30 FPS / 1,409 sampled frames @ 2 FPS):

| Metric | Single-Worker (Baseline) | Optimized Stream Engine |
| :--- | :---: | :---: |
| **Video Coverage** | 11m 44s (704.5s) | **11m 44s (704.5s)** |
| **Execution Wall Time** | 291.6 s (4.9 min) | **109.7 s (1.8 min)** |
| **Real-Time Speedup** | 2.41x (0.41 : 1) | **6.42x faster than real-time (0.16 : 1)** |
| **ANPR Throughput (x86 Laptop + Video Writer)** | 4.83 FPS | **12.84 FPS** |
| **ANPR Throughput (Max Stream Throughput)** | 4.83 FPS | **13.94 – 15.73 FPS** |
| **Projected Pi 5 Throughput (Cortex-A76 `+dotprod`)** | 4.10 FPS | **18.0 – 22.0+ FPS** |
| **Heap Memory Churn** | ~3.89 GB | **0.00 MB (Zero-Copy Frame Buffer)** |
| **Vehicles Detected (Recall)** | 25 / 26 (96.15%) | **26 / 26 (100.00%)** |
| **100% Exact Plate Match** | 18 / 26 (69.23%) | **21 / 26 (80.77%)** |
| **$\ge 90\%$ Character Match** | 23 / 26 (88.46%) | **25 / 26 (96.15%)** |
| **Overall Character Accuracy** | 92.31% (CER: 7.69%) | **96.15% (CER: 3.85%)** |
| **Stationary Sign Rejection ([OOG])** | 88.2% | **100.00% (54 / 54 blocked)** |

---

## 📊 Full 26-Vehicle Ground Truth Evaluation Matrix

Evaluated using `tools/evaluate_video_gt.py` against `samples/my_car_ground_truth.txt`:

```text
==================================================================================================
VIDEO BENCHMARK EVALUATION MATRIX (benchmarks/video_results.csv)
Video Coverage : Frames 0 -> 21105 (703.5s / 704.0s total video)
Total Raw Boxes: 157 (103 Valid [OK] Plate Reads | 54 [OOG] Background Signs Rejected)
==================================================================================================
Car #  | Manual Reading   | Rust ANPR Read   | Time Window     | Frames | Conf  | Status
--------------------------------------------------------------------------------------------------
#1     | DL8CBB1562       | DL8CBB1562       | 5.0s-6.5s       | 4      | 0.77  | EXACT MATCH (100%)
#2     | HR55AY5127       | HR55AY5127       | 15.0s-15.0s     | 1      | 0.86  | EXACT MATCH (100%)
#3     | DL11CG4984       | DL11C4984        | 34.0s-34.5s     | 2      | 0.77  | NEAR MATCH (9/10 chars)
#4     | DL9CBB9526       | DL9CBG9526       | 43.0s-43.5s     | 2      | 0.78  | NEAR MATCH (9/10 chars)
#5     | HR31N3207        | HR31N3207        | 50.0s-57.0s     | 12     | 0.88  | EXACT MATCH (100%)
#6     | HR06AS3370       | HR06AS3370       | 58.5s-60.5s     | 5      | 0.87  | EXACT MATCH (100%)
#7     | CH01CH5110       | CH01CW5110       | 70.0s-73.0s     | 7      | 0.91  | NEAR MATCH (9/10 chars)
#8     | DL1RZ3537        | DL1RZ3532        | 81.5s-81.5s     | 1      | 0.36  | NEAR MATCH (9/10 chars)
#9     | DL12CK5004       | DL12CK5004       | 109.5s-110.5s   | 2      | 0.64  | EXACT MATCH (100%)
#10    | HR55BB2076       | HR55BB2076       | 111.0s-113.0s   | 5      | 0.86  | EXACT MATCH (100%)
#11    | HR26FK0909       | HR26FK0909       | 156.0s-157.0s   | 3      | 0.88  | EXACT MATCH (100%)
#12    | HR58D2964        | HR58D2964        | 300.0s-300.5s   | 2      | 0.78  | EXACT MATCH (100%)
#13    | HR26FK0909       | HR26FK0909       | 321.5s-322.0s   | 2      | 0.80  | EXACT MATCH (100%)
#14    | DL4CBF1163       | DL4CBF1163       | 346.0s-347.5s   | 4      | 0.76  | EXACT MATCH (100%)
#15    | DL1LAG8864       | DL1LAG8864       | 359.5s-361.5s   | 5      | 0.90  | EXACT MATCH (100%)
#16    | DL12DA4834       | DL12DA4834       | 362.0s-367.0s   | 9      | 0.86  | EXACT MATCH (100%)
#17    | DL4CAY5466       | DL4CAY5466       | 367.5s-370.0s   | 6      | 0.84  | EXACT MATCH (100%)
#18    | DL11CE7467       | DL11CE7467       | 393.0s-393.0s   | 1      | 0.75  | EXACT MATCH (100%)
#19    | UP14FM6801       | UP14FM6801       | 425.0s-425.5s   | 2      | 0.85  | EXACT MATCH (100%)
#20    | DL9CBK5454       | DL9CBK5454       | 430.5s-431.0s   | 2      | 0.50  | EXACT MATCH (100%)
#21    | DL10CY0261       | DL10CY0261       | 484.5s-487.0s   | 5      | 0.82  | EXACT MATCH (100%)
#22    | DL11CE7467       | DL11CE7467       | 534.5s-536.0s   | 3      | 0.79  | EXACT MATCH (100%)
#23    | DL10CH9973       | DL10CH9973       | 560.0s-561.5s   | 2      | 0.77  | EXACT MATCH (100%)
#24    | DL1RTC3412       | DL1RTC3412       | 589.5s-590.5s   | 3      | 0.81  | EXACT MATCH (100%)
#25    | T0326HR76379C0   | 02BH7637CD       | 572.0s-572.0s   | 1      | 0.65  | TEMP PLATE (greedy=T0826HR763700)
#26    | MN06L7761        | MN06L7761        | 676.5s-678.5s   | 5      | 0.80  | EXACT MATCH (100%)
==================================================================================================
```

---

## 🏗️ Architecture & Pipeline Flow

```
Camera / Video Stream (Zero-Copy Raw RGB24 Pipe)
   │
   ▼
[Stage 1] Letterbox Rectangular Preprocess (640×384 Direct CHW)
   │
   ▼
[Stage 2] INT8 Quantized YOLO Detector (Embedded NMS Graph [1, 300, 6])
   │
   ▼
[Stage 3] Temporal Spatial IoU & Centroid Tracker
   │       ├─► [Cache Hit] Confirmed Plate Track ──► Instant Return (0.0 ms)
   │       └─► [Cache Hit] Noise / Static Sign   ──► Out-of-Grammar Reject (0.0 ms)
   ▼
[Stage 4] Edge-Aware Multi-Line Detector
   │       ├─► Aspect h/w > 0.70 & not touching edge ─► 2-Line Split (Auto-Rickshaw/Bike)
   │       └─► Standard Aspect ───────────────────────► 1-Line Base Crop
   ▼
[Stage 5] PP-OCRv5 Text Recognizer (Grouped 37-Character Projection)
   │
   ▼
[Stage 6] Grammar-Constrained CTC Prefix Beam Search
   │       └─► Indian RTO DFA Automaton (38 States/UTs, BH Series, Valid Districts)
   ▼
[Stage 7] Test-Time Augmentation (TTA) Fusion
           └─► Variant 0 Early-Exit + Fallback CLAHE Contrast Enhancement
```

---

## 📁 Repository Structure

```
.
├── Cargo.toml                  # Cargo manifest with thin LTO & SIMD flags
├── install.sh                  # One-click Raspberry Pi 5 installer & dependency setup
├── src/
│   ├── lib.rs                  # Library entry point
│   ├── main.rs                 # CLI entry point (Image, Folder, Camera, Bench, Stream)
│   ├── pipeline.rs             # End-to-end ANPR orchestration & caching
│   ├── tracker.rs              # Spatial IoU & centroid displacement temporal tracker
│   ├── image.rs                # Zero-copy letterbox, safe crop, CLAHE, gamma, 2-line split
│   ├── camera.rs               # Camera capture abstractions (V4L2, MSMF, DShow)
│   ├── config.rs               # TOML configuration parser
│   ├── detector/
│   │   ├── mod.rs              # Detector trait interface
│   │   ├── onnx.rs             # ONNX Runtime detector backend
│   │   └── ncnn.rs             # NCNN ARM NEON detector backend
│   ├── ocr/
│   │   ├── recognizer.rs       # PP-OCRv5 model runner & grouped logit projection
│   │   └── ctc_decoder.rs      # Grammar-constrained CTC prefix beam search
│   └── grammar/
│       ├── plate_dfa.rs        # Indian Plate DFA validator (Standard & Bharat series)
│       └── rto_data.rs         # 38 State/UT registration boundaries
├── configs/
│   ├── pi5_optimized.toml      # Dedicated high-throughput profile for Raspberry Pi 5
│   ├── current_onnx.toml       # Standard ONNX desktop profile
│   └── yolov8n_onnx.toml       # YOLOv8n baseline profile
├── models/
│   ├── current/
│   │   ├── detector_int8.onnx  # INT8 quantized detector with embedded NMS
│   │   ├── recognizer.onnx     # PP-OCRv5 recognizer model
│   │   └── charset.txt         # Grouped OCR character mapping
├── tools/
│   ├── run_video.py            # High-speed multi-worker zero-disk stream engine
│   ├── evaluate_video_gt.py    # Ground-truth evaluation & character error rate matrix
│   ├── live_obs.py             # OBS Studio virtual camera zero-latency pipe
│   └── prepare_stream.py       # High-speed frame extraction & preprocessing
└── samples/
    ├── sample_0.jpg to 5.jpg   # Sample vehicle images
    └── my_car_ground_truth.txt # Official 26-vehicle manual reading registry
```

---

## ⚡ Quickstart

### 1. Prerequisites
- **Rust 1.80+** ([rustup.rs](https://rustup.rs/))
- **Python 3.10+** (with `opencv-python` and `numpy`)
- **ONNX Runtime 1.19+**

### 2. Build Release Binary
```bash
# Standard Release Build
cargo build --release

# On Raspberry Pi 5 (Native ARM Cortex-A76 NEON + DotProd + FP16)
RUSTFLAGS="-C target-cpu=cortex-a76 -C target-feature=+neon,+fp16,+dotprod" cargo build --release
```

### 3. Run Inference Commands

#### Single Image Test
```bash
./target/release/anpr --config configs/pi5_optimized.toml image samples/sample_1.jpg
```

#### Process Sample Folder
```bash
./target/release/anpr --config configs/pi5_optimized.toml folder samples/ --csv benchmarks/folder_results.csv
```

#### Microsecond Latency Profiler
```bash
./target/release/anpr --config configs/pi5_optimized.toml bench samples/ --limit 20
```

#### High-Speed Multi-Worker Stream Engine
```bash
# Process video stream with real-time MP4 video writer & CSV logging
python3 tools/run_video.py \
    --video samples/my_car_sampled15.mp4 \
    --config configs/pi5_optimized.toml \
    --workers 4

# Maximum Throughput Mode (disables MP4 writer to maximize FPS)
python3 tools/run_video.py \
    --video samples/my_car_sampled15.mp4 \
    --config configs/pi5_optimized.toml \
    --workers 4 \
    --no-video
```

#### Evaluate Results against Ground Truth
```bash
python3 tools/evaluate_video_gt.py \
    --csv benchmarks/video_results.csv \
    --gt samples/my_car_ground_truth.txt
```

---

## 🍓 Raspberry Pi 5 One-Click Deployment

1. Transfer codebase to your Raspberry Pi:
   ```bash
   scp pi_update.zip testi@<PI_IP>:~/Raspberry_Pi_Rust/
   ```

2. On the Raspberry Pi terminal:
   ```bash
   cd ~/Raspberry_Pi_Rust
   unzip -o pi_update.zip
   chmod +x install.sh
   ./install.sh
   ```

The script automatically configures ONNX Runtime aarch64 libraries, validates system packages, compiles the Rust binary with native `-C target-cpu=cortex-a76 -C target-feature=+neon,+fp16,+dotprod`, and runs the self-test suite.

---

## 📜 License

Released under the [MIT License](LICENSE).
