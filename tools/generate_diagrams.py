import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

os.makedirs("D:/Raspberry_Pi_Rust/docs", exist_ok=True)

# =============================================================================
# FIGURE 1: CLEAN, MODERN SYSTEM ARCHITECTURE DIAGRAM
# =============================================================================
fig, ax = plt.subplots(figsize=(15, 8.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# Clean background
fig.patch.set_facecolor('#0F172A')  # Dark modern tech background
ax.set_facecolor('#0F172A')

# Header
ax.text(50, 95.5, 'Raspberry Pi 5 ANPR System Architecture', 
        ha='center', va='center', fontsize=20, fontweight='bold', color='#F8FAFC', fontfamily='sans-serif')
ax.text(50, 91.8, 'High-Throughput Native Rust Pipeline (14.6 - 16.2 FPS) with Temporal Gating & Indian RTO Grammar', 
        ha='center', va='center', fontsize=11, color='#94A3B8', fontfamily='sans-serif')

def draw_block(ax, x, y, w, h, bg_color, border_color, title, subtitle, points, badge=""):
    # Main Box
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.8,rounding_size=1.5',
                                facecolor=bg_color, edgecolor=border_color, linewidth=2.0, zorder=2)
    ax.add_patch(box)
    
    # Title
    ax.text(x + 2.5, y + h - 4.2, title, fontsize=12, fontweight='bold', color='#F8FAFC', zorder=4)
    if subtitle:
        ax.text(x + 2.5, y + h - 7.5, subtitle, fontsize=8.8, fontweight='semibold', color=border_color, zorder=4)
    
    # Badge if present
    if badge:
        b_box = patches.FancyBboxPatch((x + w - 21, y + h - 5.5), 19, 4.2, boxstyle='round,pad=0.3,rounding_size=0.8',
                                       facecolor=border_color, edgecolor='none', zorder=4)
        ax.add_patch(b_box)
        ax.text(x + w - 11.5, y + h - 3.4, badge, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#0F172A', zorder=5)

    # Bullet Points
    start_y = y + h - 11.5
    for i, pt in enumerate(points):
        ax.text(x + 3.0, start_y - (i * 3.8), f"- {pt}", fontsize=8.5, color='#CBD5E1', zorder=4)

# 1. INPUT LAYER
draw_block(ax, 3, 50, 20, 36, '#1E293B', '#38BDF8', 
           '1. Video Capture', 'Hardware MJPG 30 FPS',
           ['USB Cam / CSI / Video file',
            'V4L2 FourCC = MJPG codec',
            'Zero-copy RGB24 buffer',
            'No USB 2.0 5 FPS clamp',
            'Multi-process pipe feeder'],
           badge='30 FPS IN')

# 2. SCHEDULER & GATING
draw_block(ax, 26, 50, 22, 36, '#1E293B', '#F59E0B',
           '2. Temporal Gating', 'Run YOLO Every 4th Frame',
           ['Keyframe (k % 4 == 0): YOLO',
            'Skipped (k % 4 != 0): Kalman',
            'Empty frame fast exit (1.8 ms)',
            'Saves 70% CPU cycles',
            '4 Rust workers in parallel'],
           badge='3.5x BOOST')

# 3. DETECTION & NOISE FILTER
draw_block(ax, 51, 50, 22, 36, '#1E293B', '#EC4899',
           '3. Detection & Filter', 'INT8 YOLO + Noise Cache',
           ['YOLO INT8 (ONNX Runtime)',
            'Embedded NMS graph',
            '32px spatial noise grid',
            'Blocks road signs at 0.0 ms',
            '100% false signs rejected'],
           badge='100% FILTER')

# 4. OCR & GRAMMAR
draw_block(ax, 76, 50, 21, 36, '#1E293B', '#10B981',
           '4. OCR & Grammar', 'PP-OCRv5 + Indian DFA',
           ['PP-OCRv5 recognizer',
            'Max 2 OCR attempts per car',
            '2-line auto-split for bikes',
            'Indian RTO DFA beam search',
            'DL, HR, UP state validation'],
           badge='96.2% ACC')

# ARROWS BETWEEN TOP BLOCKS
def draw_arrow(ax, x1, y1, x2, y2, color='#64748B'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor=color, edgecolor=color, width=2.5, headwidth=8, headlength=8, shrink=0.05),
                zorder=6)

draw_arrow(ax, 23, 68, 26, 68, '#38BDF8')
draw_arrow(ax, 48, 68, 51, 68, '#F59E0B')
draw_arrow(ax, 73, 68, 76, 68, '#EC4899')

# 5. HARDWARE PLATFORM (BOTTOM LEFT)
hw_box = patches.FancyBboxPatch((3, 10), 45, 34, boxstyle='round,pad=0.8,rounding_size=1.5',
                                facecolor='#1E293B', edgecolor='#818CF8', linewidth=2.0, zorder=2)
ax.add_patch(hw_box)
ax.text(5.5, 39.5, 'Raspberry Pi 5 Hardware & Telemetry', fontsize=12, fontweight='bold', color='#F8FAFC', zorder=4)
ax.text(5.5, 36.5, 'Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz | 8 GB LPDDR4X', fontsize=8.8, fontweight='semibold', color='#818CF8', zorder=4)

hw_points = [
    'Vectorized via ARM NEON, FP16, and INT8 Dot-Product (+dotprod)',
    '4 Independent Rust Worker Engines (zero lock contention / no GIL)',
    'Active Cooler Keeps Temperature Stable at 56.4°C (Peak: 61.1°C)',
    'Sustained 2,400 MHz Clock Speed (Zero Thermal Throttling)',
    'Memory Footprint: 1,049 MB (Only 13.0% of 8 GB RAM used)'
]
for i, pt in enumerate(hw_points):
    ax.text(6.0, 31.5 - (i * 4.2), f"- {pt}", fontsize=8.5, color='#CBD5E1', zorder=4)

# 6. OUTPUT & METRICS (BOTTOM RIGHT)
out_box = patches.FancyBboxPatch((51, 10), 46, 34, boxstyle='round,pad=0.8,rounding_size=1.5',
                                 facecolor='#1E293B', edgecolor='#34D399', linewidth=2.0, zorder=2)
ax.add_patch(out_box)
ax.text(53.5, 39.5, 'Production Output & Verified Benchmark Results', fontsize=12, fontweight='bold', color='#F8FAFC', zorder=4)
ax.text(53.5, 36.5, 'Evaluated on 11m 44s Highway Video (26/26 Vehicles Detected)', fontsize=8.8, fontweight='semibold', color='#34D399', zorder=4)

out_points = [
    'Throughput: 14.63 FPS (Offline Video) | 16.20 FPS (Live Camera Feed)',
    'Recall: 100% (26 of 26 target vehicles correctly detected & logged)',
    'Accuracy: 96.15% character match (Character Error Rate = 3.85%)',
    'False Positive Block: 54 of 54 stationary road signs rejected at 0.0 ms',
    'Real-Time Live HUD: Rolling FPS, Plate List, CPU %, SoC Temp, Bounding Boxes'
]
for i, pt in enumerate(out_points):
    ax.text(54.0, 31.5 - (i * 4.2), f"- {pt}", fontsize=8.5, color='#CBD5E1', zorder=4)

# Connector from Step 4 down to Output
draw_arrow(ax, 86, 50, 74, 44, '#34D399')
# Connector from HW to Step 2
draw_arrow(ax, 26, 44, 37, 50, '#818CF8')

plt.tight_layout()
plt.savefig('D:/Raspberry_Pi_Rust/docs/system_architecture.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print("High-definition system_architecture.png created successfully.")

# =============================================================================
# FIGURE 2: CLEAN, MODERN PERFORMANCE BENCHMARK CHART
# =============================================================================
fig = plt.figure(figsize=(14, 8.5), dpi=300)
fig.patch.set_facecolor('#0F172A')

# Title
plt.suptitle('Raspberry Pi 5 Optimization Benchmarks & Hardware Telemetry', 
             fontsize=17, fontweight='bold', color='#F8FAFC', y=0.97)

# 1. Throughput (FPS)
ax1 = plt.subplot(2, 2, 1)
ax1.set_facecolor('#1E293B')
stages = ['Baseline\n(1 Worker)', 'Phase 1\n(Gate=2)', 'Phase 2\n(Gate=4 + Cache)', 'Phase 3\n(Live Peak)']
fps = [4.59, 8.32, 14.63, 16.20]
bar_colors = ['#EF4444', '#F59E0B', '#10B981', '#38BDF8']
bars = ax1.bar(stages, fps, color=bar_colors, width=0.52, edgecolor='#334155', linewidth=1.5)
ax1.set_ylabel('Speed (Frames Per Second)', fontsize=10, fontweight='bold', color='#F8FAFC')
ax1.set_title('Processing Speed (FPS) on Raspberry Pi 5', fontsize=11.5, fontweight='bold', color='#F8FAFC', pad=8)
ax1.tick_params(colors='#94A3B8')
ax1.grid(axis='y', linestyle='--', alpha=0.25, color='#64748B')
ax1.set_ylim(0, 20)
for bar in bars:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., h + 0.45, f'{h:.1f} FPS', 
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#F8FAFC')

# 2. Frame Latency Breakdown (ms)
ax2 = plt.subplot(2, 2, 2)
ax2.set_facecolor('#1E293B')
det = [125.0, 62.5, 31.2, 26.0]
ocr = [75.0, 45.0, 30.0, 28.0]
pipe = [18.0, 12.7, 7.1, 7.7]
idx = np.arange(len(stages))
w = 0.52

b1 = ax2.bar(idx, det, w, label='YOLO Detection', color='#818CF8', edgecolor='#334155', linewidth=1.2)
b2 = ax2.bar(idx, ocr, w, bottom=det, label='OCR & Grammar Check', color='#EC4899', edgecolor='#334155', linewidth=1.2)
bot3 = np.array(det) + np.array(ocr)
b3 = ax2.bar(idx, pipe, w, bottom=bot3, label='Tracking & IPC', color='#34D399', edgecolor='#334155', linewidth=1.2)

ax2.set_xticks(idx)
ax2.set_xticklabels(stages, fontsize=8.5, color='#94A3B8')
ax2.set_ylabel('Latency per Frame (ms)', fontsize=10, fontweight='bold', color='#F8FAFC')
ax2.set_title('Per-Frame Latency Breakdown (ms)', fontsize=11.5, fontweight='bold', color='#F8FAFC', pad=8)
ax2.tick_params(colors='#94A3B8')
ax2.grid(axis='y', linestyle='--', alpha=0.25, color='#64748B')
leg = ax2.legend(loc='upper right', facecolor='#0F172A', edgecolor='#334155', fontsize=8)
for text in leg.get_texts():
    text.set_color('#F8FAFC')

# 3. Hardware Resource Usage
ax3 = plt.subplot(2, 2, 3)
ax3.set_facecolor('#1E293B')
metrics = ['CPU Usage\n(4 Cores)', 'RAM Usage\n(of 8 GB)', 'SoC Temp\n(Sustained)', 'Peak Temp\n(Max Load)']
vals = [96.9, 13.0, 56.4, 61.1]
units = ['96.9%', '1,049 MB', '56.4°C', '61.1°C']
colors_m = ['#38BDF8', '#818CF8', '#10B981', '#F59E0B']
bars3 = ax3.bar(metrics, vals, color=colors_m, width=0.52, edgecolor='#334155', linewidth=1.5)
ax3.set_ylabel('Percentage / Value', fontsize=10, fontweight='bold', color='#F8FAFC')
ax3.set_title('Hardware Telemetry During 14.6+ FPS Run', fontsize=11.5, fontweight='bold', color='#F8FAFC', pad=8)
ax3.tick_params(colors='#94A3B8')
ax3.grid(axis='y', linestyle='--', alpha=0.25, color='#64748B')
ax3.set_ylim(0, 115)
for i, bar in enumerate(bars3):
    h = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., h + 2.5, units[i], 
             ha='center', va='bottom', fontsize=9.5, fontweight='bold', color='#F8FAFC')

# 4. Accuracy & False Positive Filtration
ax4 = plt.subplot(2, 2, 4)
ax4.set_facecolor('#1E293B')
acc_cats = ['Vehicle Recall\n(26 Cars)', 'Character\nAccuracy', 'Static Sign\nRejection', 'Zero Heap\nChurn']
acc_vals = [100.0, 96.15, 100.0, 100.0]
acc_labels = ['26 / 26 (100%)', '96.15% (3.8% CER)', '54 / 54 (100%)', '0.0 MB churn']
bars4 = ax4.bar(acc_cats, acc_vals, color='#10B981', width=0.52, edgecolor='#334155', linewidth=1.5)
ax4.set_ylabel('Success Rate (%)', fontsize=10, fontweight='bold', color='#F8FAFC')
ax4.set_title('Detection Recall & Noise Filtration', fontsize=11.5, fontweight='bold', color='#F8FAFC', pad=8)
ax4.tick_params(colors='#94A3B8')
ax4.grid(axis='y', linestyle='--', alpha=0.25, color='#64748B')
ax4.set_ylim(0, 120)
for i, bar in enumerate(bars4):
    h = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., h + 2.5, acc_labels[i], 
             ha='center', va='bottom', fontsize=9.0, fontweight='bold', color='#F8FAFC')

plt.tight_layout()
plt.savefig('D:/Raspberry_Pi_Rust/docs/performance_benchmark.png', dpi=300, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print("High-definition performance_benchmark.png created successfully.")
