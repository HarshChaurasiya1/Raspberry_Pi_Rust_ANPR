#!/usr/bin/env python3
"""Real-Time 1:1 OBS Virtual Camera & Live Video ANPR Runner.

Features:
1. Connects directly to OBS Virtual Camera (--camera 0) or a video file (--video samples/my_car.mp4).
2. Strict 1:1 Real-Time Clock:
   - A 10-minute or 11m 44s video runs in exact 1:1 real time (1.00x wall-clock speed).
   - Never builds up frame lag: capture runs at 30 FPS while the persistent Rust ANPR
     engine (`target/release/anpr.exe stream`) continuously evaluates the newest frame in memory.
3. Live Visual HUD Window + VS Code Terminal Output + CSV Logging (`benchmarks/obs_live_results.csv`).

Usage:
  # 1. Live from OBS Virtual Camera (Default):
  python tools/live_obs.py

  # 2. Direct 1:1 Real-Time playback from a video file:
  python tools/live_obs.py --video samples/my_car.mp4
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import threading
import time

import cv2
import numpy as np


class RustStreamWorker:
    """Persistent zero-disk-I/O pipe to `target/release/anpr.exe stream`."""

    def __init__(self, anpr_bin: str, config: str, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self.cmd = [
            anpr_bin,
            "--config", config,
            "stream",
            "--width", str(width),
            "--height", str(height),
        ]
        self.proc = subprocess.Popen(
            self.cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )
        self.lock = threading.Lock()
        self.latest_frame_bgr = None
        self.latest_frame_info = (0, 0.0)  # (frame_idx, timestamp_s)
        self.new_frame_event = threading.Event()

        self.latest_plates = []
        self.last_valid_plate = None
        self.last_valid_time = 0.0
        self.anpr_fps = 0.0
        self.anpr_ms = 0.0
        self.processed_frames = 0
        self.running = True
        self.csv_rows = []
        self.recent_log = []  # last 5 valid plates

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def submit_frame(self, frame_bgr: np.ndarray, frame_idx: int, timestamp_s: float) -> None:
        with self.lock:
            self.latest_frame_bgr = frame_bgr
            self.latest_frame_info = (frame_idx, timestamp_s)
        self.new_frame_event.set()

    def _worker_loop(self) -> None:
        while self.running:
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()

            with self.lock:
                frame_bgr = self.latest_frame_bgr
                f_idx, ts_s = self.latest_frame_info

            if frame_bgr is None or self.proc.stdin is None or self.proc.stdout is None:
                continue

            if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
                frame_bgr = cv2.resize(frame_bgr, (self.width, self.height), interpolation=cv2.INTER_NEAREST)

            rgb_bytes = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).tobytes()
            t0 = time.perf_counter()
            try:
                self.proc.stdin.write(rgb_bytes)
                self.proc.stdin.flush()
                line = self.proc.stdout.readline()
                if not line:
                    break
                data = json.loads(line.decode("utf-8", errors="replace"))
            except Exception:
                break

            dt = time.perf_counter() - t0
            plates = data.get("plates", [])
            ms = float(data.get("ms", dt * 1000.0))

            with self.lock:
                self.processed_frames += 1
                inst_fps = 1.0 / max(dt, 1e-6)
                self.anpr_fps = inst_fps if self.anpr_fps == 0.0 else (0.8 * self.anpr_fps + 0.2 * inst_fps)
                self.anpr_ms = ms
                self.latest_plates = plates

                for p in plates:
                    is_valid = bool(p.get("valid", False))
                    text = p.get("text", "")
                    det_score = float(p.get("det_score", 0.0))
                    greedy = p.get("greedy", "")
                    read_ms = float(p.get("read_ms", 0.0))
                    two_line = 1 if p.get("two_line", False) else 0

                    self.csv_rows.append((
                        f"frame_{f_idx:06d}",
                        f"{ts_s:.2f}",
                        text,
                        1 if is_valid else 0,
                        greedy,
                        f"{det_score:.3f}",
                        two_line,
                    ))

                    if is_valid and text:
                        self.last_valid_plate = (text, det_score, p.get("box", [0, 0, 0, 0]))
                        self.last_valid_time = time.perf_counter()
                        if not self.recent_log or self.recent_log[-1][0] != text:
                            self.recent_log.append((text, det_score, ts_s))
                            if len(self.recent_log) > 6:
                                self.recent_log.pop(0)
                        print(
                            f"  [LIVE {ts_s:6.1f}s | Frame {f_idx:6d}] [OK ] {text:<12} "
                            f"det={det_score:.2f} greedy={greedy:<12} ({read_ms:.0f} ms | {self.anpr_fps:.1f} FPS)",
                            flush=True,
                        )

    def close(self) -> None:
        self.running = False
        try:
            if self.proc.stdin:
                self.proc.stdin.close()
            self.proc.terminate()
        except Exception:
            pass


def open_capture(args: argparse.Namespace):
    if args.video:
        if not os.path.exists(args.video):
            raise FileNotFoundError(f"Video file not found: {args.video}")
        cap = cv2.VideoCapture(args.video)
        source_name = f"Video File ({args.video})"
        is_live_camera = False
    else:
        if os.name == "nt":
            cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)
            if not cap.isOpened() or not cap.read()[0]:
                cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(args.camera)
        else:
            cap = cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        source_name = f"OBS Virtual Camera / Live Device #{args.camera}"
        is_live_camera = True

    return cap, source_name, is_live_camera


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-Time 1:1 OBS Virtual Camera & Live Video ANPR Runner")
    ap.add_argument("--camera", type=int, default=0, help="Camera index (default: 0 = OBS Virtual Camera)")
    ap.add_argument("--video", default="", help="Optional path to video file to run in strict 1:1 real-time mode")
    ap.add_argument("--config", default="configs/current_onnx.toml", help="TOML config path")
    ap.add_argument("--out-csv", default="benchmarks/obs_live_results.csv", help="CSV output path")
    ap.add_argument("--max-seconds", type=float, default=0.0, help="Stop after N seconds (0 = run until end or 'Q' pressed)")
    ap.add_argument("--headless", action="store_true", help="Run without opening a GUI preview window")
    args = ap.parse_args()

    bin_candidates = ["target/release/anpr.exe", "target/release/anpr"]
    anpr_bin = next((b for b in bin_candidates if os.path.exists(b)), None)
    if not anpr_bin:
        print("Building release binary with 'cargo build --release'...", flush=True)
        subprocess.run(["cargo", "build", "--release"], check=True)
        anpr_bin = next((b for b in bin_candidates if os.path.exists(b)), None)

    cap, source_name, is_live_camera = open_capture(args)
    if not cap.isOpened():
        print(f"Error: Could not open {source_name}. Make sure OBS 'Start Virtual Camera' is clicked!", file=sys.stderr)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1.0 or fps > 120.0:
        fps = 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    proc_w, proc_h = 1280, 720

    print("=" * 76, flush=True)
    print(f"RUST ANPR LIVE 1:1 STREAMING ENGINE", flush=True)
    print("=" * 76, flush=True)
    print(f"Input Source : {source_name} ({orig_w}x{orig_h} @ {fps:.1f} FPS)", flush=True)
    print(f"Stream Pipe  : {proc_w}x{proc_h} RGB24 -> {anpr_bin} stream (Zero Disk I/O)", flush=True)
    print(f"Timing Mode  : Strict 1:1 Real-Time (1.00x wall-clock ratio)", flush=True)
    print(f"Output CSV   : {args.out_csv}", flush=True)
    if not args.headless:
        print("Controls     : Press 'Q' or 'ESC' in the video window (or Ctrl+C here) to stop", flush=True)
    print("-" * 76, flush=True)

    worker = RustStreamWorker(anpr_bin, args.config, proc_w, proc_h)
    t_start = time.perf_counter()
    frame_idx = 0

    try:
        while True:
            elapsed = time.perf_counter() - t_start
            if args.max_seconds > 0 and elapsed >= args.max_seconds:
                print(f"\nReached --max-seconds {args.max_seconds:.1f}s limit.", flush=True)
                break

            if not is_live_camera:
                # Strict 1:1 Real-Time Video Pacing:
                # Keep video frame index locked to wall-clock elapsed time (11 min video == 11 min runtime)
                target_frame = int(elapsed * fps)
                while frame_idx < target_frame:
                    if not cap.grab():
                        break
                    frame_idx += 1
                if frame_idx > target_frame:
                    sleep_s = (frame_idx - target_frame) / fps
                    if sleep_s > 0.002:
                        time.sleep(min(sleep_s, 0.03))

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            ts_s = elapsed if is_live_camera else (frame_idx / fps)

            if frame.shape[1] != proc_w or frame.shape[0] != proc_h:
                frame_hd = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_NEAREST)
            else:
                frame_hd = frame

            worker.submit_frame(frame_hd, frame_idx, ts_s)

            if not args.headless:
                display = frame_hd.copy()
                with worker.lock:
                    plates = list(worker.latest_plates)
                    last_valid = worker.last_valid_plate
                    last_valid_age = time.perf_counter() - worker.last_valid_time
                    anpr_fps = worker.anpr_fps
                    anpr_ms = worker.anpr_ms
                    recent_log = list(worker.recent_log)

                # Draw active bounding boxes
                for p in plates:
                    box = p.get("box", [0, 0, 0, 0])
                    is_valid = bool(p.get("valid", False))
                    text = p.get("text", "")
                    score = float(p.get("det_score", 0.0))
                    x1, y1, x2, y2 = [int(v) for v in box]
                    color = (0, 255, 0) if is_valid else (0, 165, 255)
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 3)
                    label = f"{text} ({score:.2f})" if is_valid else f"[OOG] {text}"
                    cv2.rectangle(display, (x1, max(0, y1 - 32)), (x1 + max(180, len(label) * 14), y1), color, -1)
                    cv2.putText(display, label, (x1 + 6, max(22, y1 - 8)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                # Draw top HUD bar
                cv2.rectangle(display, (10, 10), (720, 105), (15, 15, 15), -1)
                cv2.rectangle(display, (10, 10), (720, 105), (80, 80, 80), 1)
                cv2.putText(
                    display,
                    f"1:1 LIVE | Time: {elapsed:6.1f}s | ANPR: {anpr_fps:4.1f} FPS ({anpr_ms:4.0f} ms)",
                    (22, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                )

                if last_valid and last_valid_age < 4.0:
                    v_text, v_score, _ = last_valid
                    cv2.putText(
                        display,
                        f"DETECTED PLATE: {v_text}  [VALID RTO | conf={v_score:.2f}]",
                        (22, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.82,
                        (0, 255, 0),
                        2,
                    )
                else:
                    cv2.putText(
                        display,
                        "DETECTED PLATE: Scanning live stream...",
                        (22, 85),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.75,
                        (180, 180, 180),
                        2,
                    )

                # Draw recent valid plates log on top-right
                if recent_log:
                    rx = proc_w - 340
                    ry = 10
                    rh = 40 + len(recent_log) * 30
                    cv2.rectangle(display, (rx, ry), (proc_w - 10, ry + rh), (15, 15, 15), -1)
                    cv2.putText(display, "RECENT VALID PLATES:", (rx + 12, ry + 28),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    for i, (r_txt, r_sc, r_ts) in enumerate(reversed(recent_log)):
                        cv2.putText(
                            display,
                            f"{r_txt:<11} ({r_ts:5.1f}s)",
                            (rx + 12, ry + 58 + i * 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                        )

                try:
                    cv2.imshow("Rust ANPR Live - OBS Virtual Camera / 1:1 Stream (Press Q to Quit)", display)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), ord("Q"), 27):
                        print("\nStopped by user.", flush=True)
                        break
                except cv2.error:
                    args.headless = True
                    print("\n[INFO] No graphical desktop display detected. Running in high-speed terminal streaming mode.", flush=True)
            else:
                time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nInterrupted by user (Ctrl+C).", flush=True)
    finally:
        total_wall = time.perf_counter() - t_start
        cap.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        worker.close()

        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["image", "timestamp_s", "plate", "valid", "greedy", "det_score", "two_line"])
            for r in worker.csv_rows:
                w.writerow(r)

        print("-" * 76, flush=True)
        print(f"Wall-Clock Duration  : {total_wall:.2f}s (1:1 Real-Time Ratio)", flush=True)
        print(f"ANPR Frames Evaluated: {worker.processed_frames} frames (avg {worker.anpr_fps:.2f} FPS)", flush=True)
        print(f"Valid Plates Logged  : {len(worker.recent_log)} distinct recent plates -> {args.out_csv}", flush=True)
        print("=" * 76, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
