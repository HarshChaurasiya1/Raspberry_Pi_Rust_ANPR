# 🚗 Complete Beginner's Guide to the Rust ANPR System

Welcome! This guide explains **every folder, every file, and every step** of how this Automatic Number Plate Recognition (ANPR) project works — written in plain, everyday English with zero technical jargon assumed.

---

## 1. What Does This Project Actually Do? (The Big Picture)

Imagine a human security guard standing at a parking gate watching cars drive by:
1. **Eyes (Camera / Video)**: The guard looks at the whole street scene.
2. **Spotting the Plate (Detector)**: Out of the whole car, trees, and road, the guard zooms in on just the small rectangular license plate.
3. **Putting on Glasses (Image Enhancement)**: If it is dark, blurry, or sunny, the guard squints or adjusts the light to see the letters clearly.
4. **Reading the Letters (OCR Recognizer)**: The guard reads the characters from left to right.
5. **Brain Check (Indian RTO Grammar Rulebook)**: If a smudged plate looks like `"FDL8CBB1562"`, the guard's brain knows: *"Wait, no Indian state starts with FD! That must be `DL` (Delhi), RTO `8`, series `CBB`, number `1562`!"*

**Our software does those exact 5 steps automatically in about 0.12 seconds!**

---

## 2. Why Did We Build It in "Rust"?

Originally, you had this project written in **Python** (`D:\Raspberry_Pi\Raspberry_Pi`).
- **Python** is like speaking through a live translator: it is easy to write, but the computer has to translate every line while running, which uses extra memory (**380 MB RAM**) and runs slower on a small mini-computer like a **Raspberry Pi 5**.
- **Rust** (`D:\Raspberry_Pi_Rust`) is a modern programming language that compiles directly into a native machine binary (`anpr.exe`). It needs **no translator**, uses almost half the memory (**222 MB RAM**), starts **7.5x faster**, and processes the plate-reading math **up to 4x faster**.

---

## 3. Tour of Every Folder in `D:\Raspberry_Pi_Rust`

When you look at the left sidebar of VS Code, here is what every single folder and file is for:

```text
D:\Raspberry_Pi_Rust\
│
├── 📁 configs/          --> "Settings switches" (which model & settings to use)
├── 📁 models/           --> "The AI Brains" (trained neural networks)
├── 📁 samples/          --> "Test Photos & Videos" (your car images & videos)
├── 📁 src/              --> "The Rust Source Code" (the actual program logic)
├── 📁 target/           --> "The Compiled App" (contains target/release/anpr.exe)
├── 📁 tests/            --> "Automated Quality Checks" (10 self-tests)
├── 📁 tools/            --> "Helper Scripts" (video runner & accuracy calculator)
├── 📁 benchmarks/       --> "Results & Reports" (output videos, CSV tables, speed reports)
│
├── 📄 Cargo.toml        --> Project recipe (lists which Rust libraries we use)
├── 📄 Cargo.lock        --> Exact locked versions of those libraries
├── 📄 build.rs          --> Tiny setup helper used when compiling
├── 📄 install.sh        --> 1-click installer script for Raspberry Pi (Linux)
└── 📄 README.md         --> Quick summary manual
```

---

## 4. Deep Dive: Inside Each Folder & File

### 📁 Folder 1: `models/` (The AI Brains)
AI models are files that have already been trained on thousands of photos so they know what license plates and letters look like.
- **`models/current/detector_int8.onnx` (10 MB)**:
  - **Job**: Looks at a full photo of a car and finds the `[x1, y1, x2, y2]` rectangle coordinates of the number plate.
  - **What `INT8` means**: Normally, AI models store numbers as big 32-bit decimals (`FP32`). `INT8` shrinks them to 8-bit whole numbers so the file is **4x smaller** and runs much faster on a CPU.
- **`models/current/recognizer.onnx` (7.8 MB)**:
  - **Job**: Takes the cropped picture of the number plate and predicts what letter or digit (`A-Z`, `0-9`) is at each position from left to right.
- **`models/current/charset.txt`**:
  - **Job**: The alphabet list used by `recognizer.onnx` so it knows Class #1 is `0`, Class #2 is `1`, etc.
- **`models/current/detector_ncnn.param` & `.bin`**:
  - **Job**: The exact same plate detector converted into **NCNN format** (a special ultra-fast format made by Tencent specifically for mobile/Raspberry Pi ARM processors).
- **`models/yolov8n/`**:
  - **Job**: Contains a second detector architecture (`YOLOv8n`) used to compare speed and FPS against the `current` detector.

---

### 📁 Folder 2: `configs/` (The Control Panel)
Instead of editing code when you want to change a setting, you edit these simple `.toml` text files:
1. **`current_onnx.toml`**: Uses your trained `current` detector with the **ONNX Runtime** engine *(Default)*.
2. **`current_ncnn.toml`**: Uses your trained `current` detector with the **NCNN** engine.
3. **`yolov8n_onnx.toml`**: Uses the YOLOv8n detector with **ONNX Runtime**.
4. **`yolov8n_ncnn.toml`**: Uses the YOLOv8n detector with **NCNN**.
5. **`anpr.service`**: A Linux background service file so the Raspberry Pi starts the program automatically when powered on.

---

### 📁 Folder 3: `src/` (The Heart of the Rust Code)
This is where all the Rust code lives. Let's look at every file inside `src/`:

#### 1. `src/main.rs` — **The Front Door (CLI Commands)**
When you type `anpr.exe image ...` or `anpr.exe folder ...`, `main.rs` reads your command, loads the config, runs the pipeline, and prints the results on your screen.

#### 2. `src/config.rs` — **The Settings Reader**
Reads the `.toml` file from `configs/` (such as how many CPU threads to use, confidence threshold `0.30`, and model file paths).

#### 3. `src/image.rs` — **The Photo Editor & Enhancer**
Contains 5 critical image-processing tools written in pure fast Rust:
- **`letterbox()`**: Resizes a rectangular car photo (like `1920x1080`) into a square (`320x320` or `640x640`) by adding gray padding bars on top/bottom so the car doesn't look squished.
- **`safe_crop()`**: Cuts out just the number plate from the car photo with a 5% safety border around the edges.
- **2-Line Motorcycle Plate Splitter (`is_two_line` / `split_two_line`)**:
  - Some plates (like scooters, motorcycles, or commercial trucks) have **2 rows of text** (e.g., `DL8C` on top and `BB1562` on bottom).
  - If the cropped plate is tall (`height / width > 0.7`), our code slices it horizontally into a **Top Strip** and **Bottom Strip** and reads both!
- **`apply_clahe()` (Contrast Boost)**: Fixes shadows or glare on dirty plates by balancing local brightness.
- **`apply_gamma_lut()` (Night Vision Boost)**: Brightens dark night-time plates using a pre-calculated lookup table.

#### 4. `src/detector/` — **The Plate Finder**
- **`mod.rs`**: Defines what a `Detection` looks like (a bounding box `[x1, y1, x2, y2]` and a confidence score from `0.0` to `1.0`).
- **`onnx.rs`**: Runs `detector_int8.onnx` using Microsoft's ONNX Runtime and applies **NMS (Non-Maximum Suppression)** — which removes duplicate overlapping boxes on the same plate.
- **`ncnn.rs`**: Runs the NCNN `.param` and `.bin` detector models.

#### 5. `src/ocr/` — **The Character Reader & Brain Decoder**
- **`recognizer.rs`**: Feeds the cropped plate into `recognizer.onnx`. Instead of just spitting out one final guess, it outputs a **probability grid (Lattice)** — e.g., *"Position 1 is 85% `D` and 15% `0`; Position 2 is 92% `L`..."*.
- **`ctc_decoder.rs`**: Takes those probabilities from all 3 image variants (Raw, CLAHE, Gamma-brightened) and combines them with the Indian RTO Grammar rulebook (`plate_dfa.rs`) using **Prefix Beam Search** to find the highest-scoring legal Indian number plate.

#### 6. `src/grammar/` — **The Indian RTO Law & Rulebook**
This is the smartest part of your system:
- **`rto_data.rs`**: Contains a complete database of **all 38 official Indian State & Union Territory codes** (`DL` for Delhi, `HR` for Haryana, `MH` for Maharashtra, `KA` for Karnataka, `UP`, `TN`, `GJ`, etc.) plus the maximum valid RTO district number for each state, plus the **Bharat (`BH`) series** format (`22BH1234AA`).
- **`plate_dfa.rs`**: Builds a **DFA (Deterministic Finite Automaton)** — think of it like a strict railway track for Indian plates:
  ```text
  [State: 2 Letters] --> [RTO District: 1-2 Digits] --> [Series: 1-3 Letters] --> [Number: 1-4 Digits]
       "DL"           -->           "8"              -->         "CBB"         -->       "1562"
  ```
  - If a shop signboard says `"SCE"` or `"COFFEE"`, it cannot follow this railway track, so the DFA flags it as **`[OOG]` (Out-Of-Grammar / Not a real plate)**!
  - If motion blur makes a plate look like `"FDL8CBB1562"`, the DFA knows `"FD"` is not an Indian state, drops the `"F"`, and locks onto **`[OK ] DL8CBB1562`**!

#### 7. `src/pipeline.rs` — **The Conductor**
Connects all the stages together in order (`Detect -> Crop -> Enhance -> OCR -> Grammar Decode`) and measures the exact time in microseconds taken by every single stage.

#### 8. `src/camera.rs` — **Live Camera Connector**
Connects to a USB webcam or Raspberry Pi CSI camera for live gate monitoring.

#### 9. `src/benchmark/` — **The Stopwatch & Report Writer**
- **`latency.rs`**: Calculates average, median (`p50`), `p95`, `p99`, min, and max speed.
- **`metrics.rs`**: Measures how much RAM (MB) and CPU (%) the program is using.
- **`report.rs`**: Saves the CSV tables and Markdown reports into `benchmarks/`.

---

### 📁 Folder 4: `tools/` (Helper Python Scripts)
- **`tools/run_video.py`**: The script you just ran! It opens any `.mp4` video, resizes 4K frames to HD so it runs fast, calls `target/release/anpr.exe` on the frames, draws the green plate text onto the video, and saves `benchmarks/annotated_output.mp4` and `benchmarks/video_results.csv`.
- **`tools/calculate_metrics.py`**: Compares predictions against `ground_truth.csv` to calculate Precision, Recall, F1-score, and Character Error Rate (CER).
- **`tools/export_yolov8n.py` & `convert_ncnn.py`**: Utilities for exporting YOLO models to ONNX and NCNN formats.

---

### 📁 Folder 5: `benchmarks/` (Your Outputs & Results)
Every time you run a test or video, your outputs land here:
- **`annotated_output.mp4`**: Your video with frame numbers and detected plates drawn on top.
- **`video_results.csv`**: Spreadsheet of every plate detected in your video.
- **`four_way_comparison.csv` & `python_vs_rust.csv`**: Speed comparison tables.
- **`reports/`**: Full engineering reports (`final_report.md`, `python_vs_rust.md`, `detector_comparison.md`, `baseline_report.md`).

---

## 5. Step-by-Step: What Happened When You Ran `my_car.mp4`?

Here is the exact life cycle of **Frame 180** from your video `samples\my_car.mp4`:

```text
1. [Video Frame 180] (3840x2160 4K image at 6.0 seconds into your video)
       │
       ▼
2. [Auto-Resize & Letterbox] (`src/image.rs`)
   Resized to 1280x720, then padded into a 640x640 square for the AI detector.
       │
       ▼
3. [Plate Detector] (`src/detector/onnx.rs` -> `detector_int8.onnx`)
   Found a license plate box on the car with 71% confidence (det=0.71)!
       │
       ▼
4. [Safe Crop & 1-Line Check] (`src/image.rs`)
   Cropped out the rectangular plate. Checked height/width ratio (< 0.7 -> 1-line plate).
       │
       ▼
5. [Triple Image Enhancement] (`src/image.rs`)
   Created 3 versions of the crop:
   • Variant 1: Original Raw Crop
   • Variant 2: CLAHE (Contrast enhanced)
   • Variant 3: Gamma 2.5 (Shadow brightened)
       │
       ▼
6. [OCR Character Lattice] (`src/ocr/recognizer.rs` -> `recognizer.onnx`)
   Scanned all 3 variants and generated character probabilities across the plate.
       │
       ▼
7. [Indian RTO Grammar Beam Search] (`src/ocr/ctc_decoder.rs` + `src/grammar/plate_dfa.rs`)
   Checked candidates against Indian RTO rules:
   • "DL" -> Valid State (Delhi)
   • "8"  -> Valid Delhi RTO District (01 to 18 allowed)
   • "CBB"-> Valid Vehicle Series
   • "1562"-> Valid 4-Digit Registration Number
       │
       ▼
8. [Final Validated Output]
   Printed: [OK ] DL8CBB1562  det=0.71  greedy=DL8CBB1562  1-line (124 ms)
   Wrote to: benchmarks/video_results.csv & benchmarks/annotated_output.mp4
```

---

## 6. Quick Glossary of Terms You See in the Terminal

| Term in Terminal | What It Means in Plain English |
| :--- | :--- |
| **`[OK ]`** | **Valid Indian Plate!** The text passed all Indian RTO state, district, and number rules. |
| **`[OOG]`** | **Out Of Grammar.** The detector saw some text (like a shop sign `"SCE"` or a blurry bumper sticker), and the RTO rulebook rejected it because it's not a legal Indian plate format. |
| **`det=0.71`** | **Detection Confidence (71%).** How sure the detector AI is that the box is a number plate. |
| **`greedy=DL8CBB1562`** | **Raw OCR Guess.** What the OCR read *before* applying the Indian RTO Grammar rulebook. Compare `greedy=FDL8CBB11562` vs final `DL8CB1562` on Frame 195 — that shows the grammar rulebook fixing OCR mistakes! |
| **`1-line` / `2-line`** | Whether it is a normal wide car plate (`1-line`) or a tall square motorcycle/auto plate (`2-line`). |
| **`(124 ms)`** | **124 milliseconds (0.124 seconds).** Total time taken to detect, enhance, read, and grammar-check that plate. |
