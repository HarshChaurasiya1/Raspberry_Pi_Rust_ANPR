#!/usr/bin/env bash
# Automated Setup & Deployment Script for Raspberry Pi 5 (Debian 12 Bookworm 64-bit)
# Native Rust High-Performance ANPR Engine
set -euo pipefail

ARCH=$(uname -m)
IS_PI=0
if [[ -f /proc/device-tree/model ]] && grep -qi "raspberry pi" /proc/device-tree/model 2>/dev/null; then
  IS_PI=1
  echo "==> Detected Hardware: $(tr -d '\0' < /proc/device-tree/model)"
else
  echo "==> Architecture: $ARCH (Generic Linux / Dev environment)"
fi

echo "==> 1. Installing System Dependencies..."
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential \
    cmake \
    pkg-config \
    libssl-dev \
    libv4l-dev \
    v4l-utils \
    git \
    ffmpeg \
    python3-opencv

if [ ! -f /usr/local/lib/libonnxruntime.so ] && [ ! -f /usr/lib/libonnxruntime.so ]; then
  echo "==> Installing ONNX Runtime shared library for aarch64 / ARM64..."
  curl -fsSL https://github.com/microsoft/onnxruntime/releases/download/v1.19.2/onnxruntime-linux-aarch64-1.19.2.tgz -o /tmp/ort.tgz
  tar -xzf /tmp/ort.tgz -C /tmp/
  sudo cp -d /tmp/onnxruntime-linux-aarch64-1.19.2/lib/libonnxruntime.so* /usr/local/lib/
  sudo ldconfig
  rm -rf /tmp/ort.tgz /tmp/onnxruntime-linux-aarch64-1.19.2
fi

if [[ $IS_PI -eq 1 ]]; then
  echo "==> Adding user to 'video' group for camera access..."
  sudo usermod -a -G video "$USER" || true
  sudo apt-get install -y -qq rpicam-apps || true

  echo "==> Tuning Pi 5 CPU Governor & Memory Parameters..."
  # Lock all 4 Cortex-A76 cores to 2.4 GHz performance mode (eliminates frequency scaling latency)
  echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor > /dev/null 2>&1 || true
  # With 8GB RAM, prioritize RAM caching and prevent SD-card swap thrashing
  sudo sysctl -w vm.swappiness=10 > /dev/null 2>&1 || true
  sudo sysctl -w vm.vfs_cache_pressure=50 > /dev/null 2>&1 || true
fi

echo "==> 2. Checking Rust Toolchain..."
if ! command -v cargo >/dev/null 2>&1; then
  echo "    Rust not found. Installing via rustup..."
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
  # shellcheck disable=SC1091
  source "$HOME/.cargo/env"
else
  echo "    Rust is already installed: $(rustc --version)"
fi

# Ensure cargo is in PATH for current script execution
export PATH="$HOME/.cargo/bin:$PATH"

echo "==> 3. Building Optimized Release Binary for Cortex-A76..."
if [[ $IS_PI -eq 1 || "$ARCH" == "aarch64" ]]; then
  RUSTFLAGS="-C target-cpu=cortex-a76 -C target-feature=+neon,+fp16,+dotprod" cargo build --release
else
  cargo build --release
fi

echo "==> 4. Running Unit & Integration Tests..."
cargo test

echo "==> 5. Self-Test Inference on sample_1.jpg..."
./target/release/anpr image samples/sample_1.jpg

echo "==> 6. Platform Hardware Telemetry..."
if command -v vcgencmd >/dev/null 2>&1; then
  echo "    Throttle Status : $(vcgencmd get_throttled)   (0x0 = normal, no throttling)"
  echo "    SoC Temperature : $(vcgencmd measure_temp)"
fi

cat <<'EOF'
================================================================================
DEPLOYMENT & VERIFICATION COMPLETE!

Quick Verification Commands:
  1. Test Single Image:
     ./target/release/anpr image samples/sample_1.jpg

  2. Run Latency Benchmark:
     ./target/release/anpr bench samples/

  3. Run 4-Way Comparison Matrix:
     ./target/release/anpr compare --samples samples/

  4. Live Camera Gate Mode (USB / CSI):
     ./target/release/anpr camera --source auto --log /var/log/anpr_plates.csv

To install as an autostart background daemon (systemd):
  sudo cp configs/anpr.service /etc/systemd/system/
  sudo systemctl daemon-reload
  sudo systemctl enable --now anpr.service
================================================================================
EOF
