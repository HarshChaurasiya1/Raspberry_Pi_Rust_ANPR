import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageDraw, ImageFont
import numpy as np

os.makedirs("D:/Raspberry_Pi_Rust/docs", exist_ok=True)
output_path = "D:/Raspberry_Pi_Rust/docs/guard_security_dashboard.png"

# Setup Figure (1920x1080 px at 120 DPI -> 16 x 9 inches)
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor('#0B1120') # Deep rich dark navy slate

ax = plt.subplot(1, 1, 1)
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis('off')

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
def draw_rounded_rect(ax, x, y, w, h, bg_color, border_color='#334155', lw=1.5, radius=1.0, zorder=2):
    box = patches.FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0.2,rounding_size={radius}',
                                facecolor=bg_color, edgecolor=border_color, linewidth=lw, zorder=zorder)
    ax.add_patch(box)
    return box

# -----------------------------------------------------------------------------
# 1. TOP HEADER NAVIGATION BAR
# -----------------------------------------------------------------------------
draw_rounded_rect(ax, 1, 91.5, 98, 7.5, '#1E293B', '#38BDF8', lw=1.8, radius=0.8)

# Logo / University Title
ax.text(2.5, 96.2, 'DELHI TECHNOLOGICAL UNIVERSITY', fontsize=13, fontweight='bold', color='#38BDF8', fontfamily='sans-serif')
ax.text(2.5, 93.2, 'CAMPUS SECURITY OPERATIONS CENTER (SOC) — MAIN GATE KIOSK [TERMINAL #01]', fontsize=9, fontweight='semibold', color='#94A3B8')

# Center Live Badge
draw_rounded_rect(ax, 52, 92.8, 14, 4.8, '#064E3B', '#10B981', lw=1.2, radius=0.6)
ax.text(59, 95.2, '● LIVE STREAMING', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#34D399')

# Hardware Telemetry Pill (Right)
draw_rounded_rect(ax, 67.5, 92.8, 30.5, 4.8, '#0F172A', '#64748B', lw=1.0, radius=0.6)
ax.text(69, 95.2, 'Raspberry Pi 5  |  56.4°C  |  16.2 FPS  |  4x A76 @ 2.4 GHz', va='center', fontsize=8.5, fontweight='bold', color='#E2E8F0')

# -----------------------------------------------------------------------------
# 2. MAIN LEFT PANEL: LIVE CAMERA FEED (LANE 01 - ENTRY GATE)
# -----------------------------------------------------------------------------
draw_rounded_rect(ax, 1, 14, 58, 76, '#0F172A', '#334155', lw=1.5, radius=1.0)

# Feed Header
ax.text(2.8, 87.2, 'LANE 01: MAIN GATE INWARD (ENTRY CAMERA)', fontsize=11, fontweight='bold', color='#F8FAFC')
ax.text(32, 87.2, '[ 1080p @ 30 FPS - V4L2 MJPG ]', fontsize=8.5, fontweight='semibold', color='#38BDF8')

# Camera Status Indicator
draw_rounded_rect(ax, 50.5, 86.0, 7.5, 2.8, '#DC2626', '#EF4444', lw=0.8, radius=0.4)
ax.text(54.25, 87.4, '● REC', ha='center', va='center', fontsize=8, fontweight='bold', color='#FFFFFF')

# Camera Canvas Area (Simulated Live CCTV View)
c_box = draw_rounded_rect(ax, 2.5, 19.5, 55, 65.5, '#020617', '#1E293B', lw=1.0, radius=0.8)

# Load real image if available or draw perspective roadway
frame_loaded = False
sample_img_path = "D:/Raspberry_Pi_Rust/benchmarks/frame_68.png"
if not os.path.exists(sample_img_path):
    sample_img_path = "D:/Raspberry_Pi_Rust/samples/sample_1.jpg"

if os.path.exists(sample_img_path):
    try:
        img = Image.open(sample_img_path)
        # Crop & display in bbox coordinates: [2.5, 57.5, 19.5, 85]
        ax.imshow(img, extent=[2.5, 57.5, 19.5, 85.0], aspect='auto', zorder=3)
        frame_loaded = True
    except Exception as e:
        print("Image load error:", e)

if not frame_loaded:
    # Roadway representation
    road = patches.Polygon([[12, 19.5], [48, 19.5], [38, 70], [22, 70]], closed=True, facecolor='#1E293B', edgecolor='#475569', zorder=3)
    ax.add_patch(road)
    car_body = patches.FancyBboxPatch((20, 28), 20, 32, boxstyle='round,pad=1.0,rounding_size=3.0',
                                      facecolor='#334155', edgecolor='#64748B', linewidth=1.5, zorder=4)
    ax.add_patch(car_body)

# OVERLAYS ON CAMERA VIEW (Target Bounding Boxes & HUD)
# Dark translucent overlay at top of video
v_header = patches.Rectangle((2.5, 80.5), 55, 4.5, facecolor='#020617', alpha=0.75, zorder=5)
ax.add_patch(v_header)
ax.text(4.0, 82.7, 'CAM_01_DTU_MAIN_ENTRY | EXPOSURE: AUTO | IR-CUT: OFF | DAY MODE', fontsize=8, color='#94A3B8', zorder=6)
ax.text(55.5, 82.7, '16-NOV-2026 10:42:18', ha='right', fontsize=8, fontweight='bold', color='#F8FAFC', zorder=6)

# Vehicle Detection Box (Cyan/Green)
veh_box = patches.Rectangle((14, 25), 32, 48, linewidth=2.0, edgecolor='#06B6D4', facecolor='none', linestyle='--', zorder=6)
ax.add_patch(veh_box)
# Vehicle Tag
v_tag = patches.Rectangle((14, 73), 16, 3.2, facecolor='#06B6D4', edgecolor='none', zorder=7)
ax.add_patch(v_tag)
ax.text(14.8, 74.6, 'Vehicle [Track #104] 97.4%', fontsize=7.5, fontweight='bold', color='#020617', zorder=8)

# Target ANPR License Plate Box (High-Visibility Solid Neon Green)
plate_box = patches.Rectangle((21.5, 30.5), 17, 8.5, linewidth=2.8, edgecolor='#10B981', facecolor='#064E3B', alpha=0.25, zorder=8)
ax.add_patch(plate_box)

# Corner Reticle Brackets
for (cx, cy, dx, dy) in [(21.5, 30.5, 2, 2), (38.5, 30.5, -2, 2), (21.5, 39.0, 2, -2), (38.5, 39.0, -2, -2)]:
    ax.plot([cx, cx + dx], [cy, cy], color='#34D399', linewidth=3, zorder=9)
    ax.plot([cx, cx], [cy, cy + dy], color='#34D399', linewidth=3, zorder=9)

# Plate Text Tag Above Box
p_tag = patches.FancyBboxPatch((21.5, 39.5), 17, 4.0, boxstyle='round,pad=0.2,rounding_size=0.4',
                               facecolor='#10B981', edgecolor='none', zorder=9)
ax.add_patch(p_tag)
ax.text(30.0, 41.5, 'DL 8C BB 1562 (Conf: 94.8%)', ha='center', va='center', fontsize=8.5, fontweight='bold', color='#020617', zorder=10)

# Bottom Video Telemetry Banner
v_footer = patches.Rectangle((2.5, 19.5), 55, 4.8, facecolor='#020617', alpha=0.85, zorder=5)
ax.add_patch(v_footer)
ax.text(4.0, 21.9, 'LATENCY: 61.8 ms  |  ROLLING FPS: 16.2  |  TRACKER: ACTIVE  |  RELAY #1 [BOOM BARRIER]: OPENING', 
        fontsize=8, fontweight='bold', color='#34D399', zorder=6)

# -----------------------------------------------------------------------------
# 3. TOP RIGHT PANEL: INSTANT VEHICLE VERIFICATION CARD
# -----------------------------------------------------------------------------
draw_rounded_rect(ax, 60.5, 52.0, 38.5, 38.0, '#1E293B', '#334155', lw=1.5, radius=1.0)
ax.text(62.2, 87.2, 'LATEST VERIFICATION & ACCESS STATUS', fontsize=11, fontweight='bold', color='#F8FAFC')

# Indian HSRP License Plate Render (Authentic Look)
hsrp_bg = patches.FancyBboxPatch((62.5, 75.5), 34.5, 8.5, boxstyle='round,pad=0.3,rounding_size=0.6',
                                facecolor='#FFFFFF', edgecolor='#000000', linewidth=2.5, zorder=3)
ax.add_patch(hsrp_bg)
# Blue IND strip
ind_strip = patches.FancyBboxPatch((62.5, 75.5), 4.5, 8.5, boxstyle='round,pad=0.1,rounding_size=0.4',
                                  facecolor='#1D4ED8', edgecolor='none', zorder=4)
ax.add_patch(ind_strip)
ax.text(64.75, 80.8, 'IND', ha='center', va='center', fontsize=8, fontweight='bold', color='#FFFFFF', zorder=5)
# Ashoka Chakra simulated dot
ax.plot(64.75, 78.0, marker='o', markersize=4, color='#F59E0B', zorder=5)

# High-Contrast Plate Characters
ax.text(82.0, 79.7, 'DL 8C BB 1562', ha='center', va='center', fontsize=19, fontweight='heavy', color='#000000', fontfamily='monospace', zorder=5)

# Access Granted Big Banner
draw_rounded_rect(ax, 62.5, 68.2, 34.5, 5.5, '#064E3B', '#10B981', lw=1.5, radius=0.6)
ax.text(79.75, 70.9, '✓ ACCESS GRANTED — BOOM BARRIER OPEN', ha='center', va='center', fontsize=10.5, fontweight='bold', color='#34D399')

# Details Grid
details = [
    ('Owner Name', 'Prof. Alok Kumar', 'Dept', 'Computer Science & Engg.'),
    ('Role / Category', 'FACULTY / PERMANENT STAFF', 'Vehicle', 'Honda City (White)'),
    ('Gate Access', 'DTU ALL GATES (24x7)', 'FASTag ID', 'FT-8830194 (Verified)'),
    ('RTO Grammar', 'Delhi North (DL-8C) [PASS]', 'Match Score', '100% Exact Match')
]

for row_idx, (k1, v1, k2, v2) in enumerate(details):
    curr_y = 63.8 - (row_idx * 3.4)
    ax.text(63.0, curr_y, f'{k1}:', fontsize=8, color='#94A3B8')
    ax.text(71.0, curr_y, v1, fontsize=8, fontweight='bold', color='#F8FAFC')
    
    ax.text(81.0, curr_y, f'{k2}:', fontsize=8, color='#94A3B8')
    ax.text(87.5, curr_y, v2, fontsize=8, fontweight='bold', color='#38BDF8' if 'DL-8C' in v2 or 'White' in v2 else '#F8FAFC')

# -----------------------------------------------------------------------------
# 4. MIDDLE RIGHT PANEL: RECENT VEHICLE TRANSIT LOG TABLE
# -----------------------------------------------------------------------------
draw_rounded_rect(ax, 60.5, 14.0, 38.5, 36.5, '#1E293B', '#334155', lw=1.5, radius=1.0)
ax.text(62.2, 47.7, 'RECENT TRANSIT LOG (DTU MAIN GATE)', fontsize=11, fontweight='bold', color='#F8FAFC')
ax.text(90.5, 47.7, 'AUTO-SYNC [ON]', fontsize=8, fontweight='bold', color='#10B981')

# Table Header
t_hdr = patches.Rectangle((62.0, 42.5), 35.5, 3.2, facecolor='#0F172A', edgecolor='#334155', linewidth=0.8, zorder=3)
ax.add_patch(t_hdr)
ax.text(63.0, 44.1, 'TIME', fontsize=7.5, fontweight='bold', color='#94A3B8', zorder=4)
ax.text(70.0, 44.1, 'PLATE NUMBER', fontsize=7.5, fontweight='bold', color='#94A3B8', zorder=4)
ax.text(81.5, 44.1, 'CATEGORY', fontsize=7.5, fontweight='bold', color='#94A3B8', zorder=4)
ax.text(91.0, 44.1, 'ACTION', fontsize=7.5, fontweight='bold', color='#94A3B8', zorder=4)

# Table Rows
rows_data = [
    ('10:42:15', 'DL 8C BB 1562', 'Faculty (CSE)', 'GRANTED', '#10B981', '#064E3B'),
    ('10:40:50', 'HR 26 FK 0909', 'Student (M.Tech)', 'GRANTED', '#10B981', '#064E3B'),
    ('10:38:12', 'UP 14 FM 6801', 'Visitor / Courier', 'PASS REQD', '#F59E0B', '#78350F'),
    ('10:35:40', 'DL 10 CH 9973', 'Blacklisted / Unknown', 'ALERT / STOP', '#EF4444', '#7F1D1D'),
    ('10:31:05', 'DL 4C BF 1163', 'Staff / Admin', 'GRANTED', '#10B981', '#064E3B'),
    ('10:28:22', 'HR 55 AY 5127', 'Faculty (ECE)', 'GRANTED', '#10B981', '#064E3B'),
]

for r_i, (t_val, p_val, c_val, act_val, text_col, bg_col) in enumerate(rows_data):
    ry = 38.8 - (r_i * 4.0)
    # Row strip
    strip = patches.Rectangle((62.0, ry - 0.8), 35.5, 3.4, facecolor='#0F172A' if r_i % 2 == 1 else '#1E293B', 
                              edgecolor='none', zorder=3)
    ax.add_patch(strip)
    ax.text(63.0, ry + 0.8, t_val, fontsize=7.8, color='#CBD5E1', zorder=4)
    ax.text(70.0, ry + 0.8, p_val, fontsize=8.2, fontweight='bold', color='#F8FAFC', fontfamily='monospace', zorder=4)
    ax.text(81.5, ry + 0.8, c_val, fontsize=7.5, color='#94A3B8', zorder=4)
    
    # Status Badge
    s_badge = patches.FancyBboxPatch((90.0, ry), 7.0, 2.4, boxstyle='round,pad=0.1,rounding_size=0.4',
                                    facecolor=bg_col, edgecolor=text_col, linewidth=0.8, zorder=4)
    ax.add_patch(s_badge)
    ax.text(93.5, ry + 1.2, act_val, ha='center', va='center', fontsize=6.8, fontweight='bold', color=text_col, zorder=5)

# -----------------------------------------------------------------------------
# 5. BOTTOM CONTROL BAR: GUARD ACTION BUTTONS & HARDWARE HEALTH
# -----------------------------------------------------------------------------
draw_rounded_rect(ax, 1, 1.5, 98, 11.0, '#1E293B', '#334155', lw=1.5, radius=0.8)

# Guard Quick Buttons (Left to Middle)
def draw_btn(ax, x, y, w, h, bg, border, text, subtext=''):
    btn = patches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.2,rounding_size=0.6',
                                facecolor=bg, edgecolor=border, linewidth=1.5, zorder=3)
    ax.add_patch(btn)
    if subtext:
        ax.text(x + w/2., y + h/2. + 1.0, text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#FFFFFF', zorder=4)
        ax.text(x + w/2., y + h/2. - 1.2, subtext, ha='center', va='center', fontsize=6.8, color='#E2E8F0', zorder=4)
    else:
        ax.text(x + w/2., y + h/2., text, ha='center', va='center', fontsize=8.5, fontweight='bold', color='#FFFFFF', zorder=4)

draw_btn(ax, 2.5, 3.2, 12.5, 7.6, '#059669', '#34D399', 'OPEN BARRIER', '[ MANUAL OVERRIDE ]')
draw_btn(ax, 16.0, 3.2, 12.5, 7.6, '#DC2626', '#F87171', 'EMERGENCY STOP', '[ LOCK BOOM GATE ]')
draw_btn(ax, 29.5, 3.2, 12.5, 7.6, '#D97706', '#FBBF24', 'VISITOR PASS', '[ QUICK PRINT SLIP ]')
draw_btn(ax, 43.0, 3.2, 12.5, 7.6, '#475569', '#94A3B8', 'RESET TRACKER', '[ CLEAR 0.0 ms CACHE ]')

# Hardware Health Box (Right)
draw_rounded_rect(ax, 57.0, 2.8, 41.0, 8.4, '#0F172A', '#38BDF8', lw=1.2, radius=0.6)
ax.text(58.5, 9.0, 'EDGE COMPUTING HEALTH (RASPBERRY PI 5 - 8GB)', fontsize=8.5, fontweight='bold', color='#38BDF8')

ax.text(58.5, 6.8, '• CPU Load: 96.9% (4x Cortex-A76 @ 2.4 GHz)', fontsize=7.8, color='#E2E8F0')
ax.text(58.5, 4.6, '• RAM Usage: 1,049 MB / 8 GB (13.0% Used)', fontsize=7.8, color='#E2E8F0')

ax.text(78.5, 6.8, '• SoC Temp: 56.4°C (Active Cooler - Safe)', fontsize=7.8, color='#34D399')
ax.text(78.5, 4.6, '• Engine: Rust Native 16.2 FPS (Zero Heap Churn)', fontsize=7.8, color='#34D399')

plt.tight_layout()
plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print("Guard Security Dashboard PNG generated successfully at:", output_path)
