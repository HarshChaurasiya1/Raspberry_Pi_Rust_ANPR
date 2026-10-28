import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, hex_color):
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shd)

def set_cell_margins(cell, top=120, bottom=120, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def add_callout(doc, text, title="TECHNICAL INSIGHT"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = tbl.cell(0, 0)
    set_cell_background(cell, "F1F5F9")
    set_cell_margins(cell, top=140, bottom=140, left=200, right=160)
    
    # Left border styling
    tcPr = cell._tc.get_or_add_tcPr()
    borders = parse_xml(
        f'<w:tcBorders {nsdecls("w")}>'
        f'<w:top w:val="none"/>'
        f'<w:left w:val="single" w:sz="24" w:space="0" w:color="2563EB"/>'
        f'<w:bottom w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tcBorders>'
    )
    tcPr.append(borders)
    
    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    run_t = p.add_run(f"[{title}] ")
    run_t.bold = True
    run_t.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(10)
    
    run_b = p.add_run(text)
    run_b.font.name = "Calibri"
    run_b.font.size = Pt(10)
    run_b.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def format_table(tbl, col_widths, headers, data, alignments=None):
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    # Format Header Row
    hdr_cells = tbl.rows[0].cells
    for i, title in enumerate(headers):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], "1B365D")
        set_cell_margins(hdr_cells[i], top=140, bottom=140, left=140, right=140)
        p = hdr_cells[i].paragraphs[0]
        if alignments:
            p.alignment = alignments[i]
        for run in p.runs:
            run.font.bold = True
            run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
            run.font.name = "Calibri"
            run.font.size = Pt(9.5)
            
    # Format Data Rows
    for r_idx, row_data in enumerate(data):
        row_cells = tbl.rows[r_idx + 1].cells
        bg_col = "F8FAFC" if r_idx % 2 == 1 else "FFFFFF"
        for c_idx, val in enumerate(row_data):
            row_cells[c_idx].text = str(val)
            set_cell_background(row_cells[c_idx], bg_col)
            set_cell_margins(row_cells[c_idx], top=100, bottom=100, left=140, right=140)
            p = row_cells[c_idx].paragraphs[0]
            if alignments:
                p.alignment = alignments[c_idx]
            for run in p.runs:
                run.font.name = "Calibri"
                run.font.size = Pt(9)
                run.font.color.rgb = RGBColor(0x1E, 0x29, 0x3B)
                if c_idx == 0:
                    run.font.bold = True

    # Set column widths
    for row in tbl.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = Inches(width)

def build_report():
    doc = docx.Document()
    
    # Page Setup: Standard Letter / 1 inch margins
    sections = doc.sections
    for s in sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)

    # Base Styles
    styles = doc.styles
    normal_style = styles['Normal']
    normal_style.font.name = 'Calibri'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = RGBColor(0x33, 0x41, 0x55)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # -------------------------------------------------------------------------
    # DOCUMENT COVER / HEADER
    # -------------------------------------------------------------------------
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(12)
    p_title.paragraph_format.space_after = Pt(4)
    run_t = p_title.add_run("HIGH-PERFORMANCE EMBEDDED ANPR ON RASPBERRY PI 5")
    run_t.font.name = "Calibri"
    run_t.font.size = Pt(22)
    run_t.font.bold = True
    run_t.font.color.rgb = RGBColor(0x0F, 0x17, 0x2A)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(18)
    run_sub = p_sub.add_run("Architectural Optimization & Latency Reduction: 3.4 FPS to 14.63+ FPS via Temporal Gating, Multi-Worker Parallelism, Static Noise Filtering, and Grammar DFA")
    run_sub.font.name = "Calibri"
    run_sub.font.size = Pt(12)
    run_sub.font.italic = True
    run_sub.font.color.rgb = RGBColor(0x25, 0x63, 0xEB)

    # Metadata Table
    meta_table = doc.add_table(rows=4, cols=2)
    meta_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    meta_data = [
        ("Target Platform", "Raspberry Pi 5 (Broadcom BCM2712 Quad-Core Cortex-A76 @ 2.4 GHz, 8 GB LPDDR4X)"),
        ("System Software", "Debian Linux (aarch64), Rust 2021 Edition, ONNX Runtime (ARMv8 NEON INT8)"),
        ("Benchmark Dataset", "Surveillance Stream: samples/my_car_sampled15.mp4 (11.7 Min, 1,409 HD Frames, 26 Vehicles)"),
        ("Final Performance", "14.63 FPS (7.32x Real-Time Speedup, 96.28s Wall Time, 100% Ground-Truth Recall)")
    ]
    for r_idx, (k, v) in enumerate(meta_data):
        c0 = meta_table.cell(r_idx, 0)
        c1 = meta_table.cell(r_idx, 1)
        c0.text = k
        c1.text = v
        set_cell_background(c0, "F1F5F9")
        set_cell_background(c1, "FFFFFF")
        set_cell_margins(c0, 60, 60, 100, 100)
        set_cell_margins(c1, 60, 60, 100, 100)
        c0.paragraphs[0].runs[0].font.bold = True
        c0.paragraphs[0].runs[0].font.size = Pt(9)
        c1.paragraphs[0].runs[0].font.size = Pt(9)
        c0.width = Inches(1.8)
        c1.width = Inches(4.7)

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # -------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Summary & Project Motivation", level=1)
    h1.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "Automatic Number Plate Recognition (ANPR / ALPR) is a vital computer vision component in intelligent "
        "transportation systems, electronic toll collection (FASTag/ETC), automated parking management, and law "
        "enforcement surveillance. While cloud-based ANPR pipelines achieve high throughput, they suffer from high bandwidth "
        "consumption, network latency, recurring operational costs, and privacy vulnerabilities. Deploying neural ANPR "
        "strictly at the edge on affordable embedded single-board computers (SBCs) is therefore highly desirable."
    )
    doc.add_paragraph(
        "However, edge deployment faces severe computational constraints. Embedded ARM CPUs possess limited thermal design "
        "power (TDP) and lack high-end discrete GPUs. In our initial deployment of a full neural ANPR pipeline (YOLOv8 INT8 "
        "detector + CRNN character recognizer) on the Raspberry Pi 5, the system operated at a disappointing 3.4 to 4.59 FPS "
        "(218 ms latency per frame), completely failing to meet real-time streaming requirements."
    )
    doc.add_paragraph(
        "Through a rigorous multi-stage engineering optimization, we fundamentally restructured the pipeline architecture. "
        "By implementing (1) Spatial & Temporal Detector Gating, (2) Zero-Cost Skip-Frame Fast Exits, (3) a 32-pixel Quantized "
        "Spatial Static Out-Of-Grammar (OOG) Noise Filter, (4) Multi-Worker Concurrent IPC Thread Pipelining, and (5) an Indian "
        "RTO Grammar DFA with visual prefix pruning, we increased processing speed from 4.59 FPS to 14.63 FPS (peak rolling speed "
        "of 14.8 FPS). This represents a 3.19x throughput improvement and a 7.32x real-time speedup over 11.7 minutes of continuous "
        "surveillance footage, while maintaining 100% vehicle recall on all ground-truth targets, safe SoC thermal conditions (56.4°C), "
        "and utilizing only 13% of system RAM."
    )

    add_callout(
        doc,
        "Throughput jumped from 4.59 FPS to 14.63 FPS on the exact same Raspberry Pi 5 hardware with zero cloud offloading "
        "and zero loss in plate recognition accuracy. The 11.7-minute test video finished in just 96.28 seconds.",
        title="HEADLINE ACHIEVEMENT"
    )

    # -------------------------------------------------------------------------
    # SECTION 2: HARDWARE PLATFORM SPECIFICATIONS
    # -------------------------------------------------------------------------
    h2 = doc.add_heading("2. Embedded Hardware Platform Specifications", level=1)
    h2.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "The target embedded device evaluated throughout this investigation is the Raspberry Pi 5 (8 GB model). "
        "Table 1 outlines the architectural and computational characteristics of the platform."
    )

    hw_headers = ["Subsystem", "Specification", "Engineering Implications for ANPR"]
    hw_data = [
        ["Processor SoC", "Broadcom BCM2712 (16nm FinFET)", "Quad-core 64-bit ARM Cortex-A76 microarchitecture operating at 2.4 GHz."],
        ["Vector SIMD Engine", "ARMv8.2-A NEON (+dotprod, +fp16)", "Hardware-accelerated INT8 dot-product instructions enable high-throughput 8-bit quantized matrix multiplication."],
        ["System Memory", "8 GB LPDDR4X-4267 SDRAM", "128-bit dual-channel memory bus yielding ~17 GB/s bandwidth; ample capacity for multi-worker frame buffers."],
        ["CPU Caches", "64 KB L1, 512 KB private L2, 2 MB L3", "Private 512 KB L2 cache per core allows parallel workers to avoid thrashing shared cache lines."],
        ["Graphics / GPU", "VideoCore VII @ 800 MHz", "Supports Vulkan 1.2 and OpenGL ES 3.1; available for secondary compute offloading."],
        ["Thermal Envelope", "Active Cooler (Fan + Aluminum Heatsink)", "Thermal throttling initiates strictly at 80.0°C; sustained load must remain under this boundary."]
    ]
    t1 = doc.add_table(rows=len(hw_data)+1, cols=3)
    format_table(t1, [1.5, 2.2, 2.8], hw_headers, hw_data)

    # -------------------------------------------------------------------------
    # SECTION 3: BASELINE PROFILING & ROOT-CAUSE ANALYSIS
    # -------------------------------------------------------------------------
    h3 = doc.add_heading("3. Baseline System Profiling & Root-Cause Bottleneck Analysis", level=1)
    h3.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "In our initial baseline benchmark, the ANPR engine was executed sequentially with a single worker process "
        "(threads=4) across the 1,409 frames of samples/my_car_sampled15.mp4. The pipeline achieved an end-to-end throughput "
        "of only 4.59 FPS, requiring 306.93 seconds (5.11 minutes) of execution time. Detailed hardware telemetry and software "
        "profiling revealed four primary bottlenecks:"
    )

    doc.add_paragraph(
        "1. Unconditional Neural Detection Execution:\n"
        "The YOLOv8 INT8 neural detector required approximately 125 ms per inference pass on the ARM Cortex-A76 cores. In the naive "
        "pipeline, this neural network was executed unconditionally on every single frame, even during long stretches of empty road "
        "or when a vehicle was already locked in a stable trajectory. The detector alone consumed 57% of the total frame budget."
    )
    doc.add_paragraph(
        "2. Sequential Pipeline Stalls (Single-Worker Bottleneck):\n"
        "In single-worker mode, frame decoding, image resizing, neural detection, plate cropping, neural OCR, beam search, and JSON IPC "
        "serialization were chained sequentially on a single thread of execution. While OpenMP multi-threaded the neural matrix ops, "
        "the CPU cores spent substantial time waiting on serialized I/O."
    )
    doc.add_paragraph(
        "3. False-Positive Roadside Signage CPU Drain:\n"
        "In roadside surveillance video, static non-vehicle objects such as shop signs, building banners, and commercial hoardings "
        "(e.g., text reading 'SICIAT', 'SICE', 'STE', 'SITE') triggered the YOLO detector with scores between 0.50 and 0.85. Because the "
        "initial tracker allowed up to 3 OCR attempts before marking an object as noise, and dropped tracks after 2 misses, the pipeline "
        "repeatedly re-detected these roadside signs as new tracks, wasting 3 full OCR passes (3 x 95 ms = 285 ms) on non-license-plate text."
    )
    doc.add_paragraph(
        "4. Stale-Track Gating Fallthrough Leakage:\n"
        "When an initial temporal skipping condition was introduced, an edge-case bug existed: if the tracker's internal track array contained "
        "stale tracks with misses > 0 (e.g., vehicles that had just exited the field of view), the gating logic failed its active check and "
        "accidentally fell through to execute the full 125 ms YOLO detector anyway, completely negating the intended skip."
    )

    # -------------------------------------------------------------------------
    # SECTION 4: THE FIVE OPTIMIZATION PILLARS
    # -------------------------------------------------------------------------
    h4 = doc.add_heading("4. The Five Engineering Optimization Pillars", level=1)
    h4.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "To break through the 4.59 FPS bottleneck without compromising recognition accuracy, we architected five complementary "
        "engineering solutions in the Rust pipeline core and Python streaming harness."
    )

    doc.add_heading("Pillar 1: Spatial & Temporal Detector Gating", level=2)
    doc.add_paragraph(
        "Vehicles traveling past a traffic surveillance camera remain in the field of view across multiple consecutive frames (typically "
        "between 6 and 30 frames depending on velocity and camera angle). Running a 125 ms neural detector on every frame is mathematically "
        "redundant. We implemented modulo-based Temporal Detector Gating (detector_interval = 4 or 5)."
    )
    doc.add_paragraph(
        "On designated skip frames (frame_count % detector_interval != 1), the full YOLO neural network is bypassed entirely (0.0 ms). "
        "Instead, active tracks from the spatial tracker are leveraged directly: if a plate is already confirmed by the grammar engine, "
        "the confirmed alphanumeric result is emitted immediately in 0.0 ms. If a track is active but unconfirmed, the predicted bounding "
        "box is cropped directly for rapid character recognition (35 ms), bypassing the 125 ms detector."
    )

    doc.add_heading("Pillar 2: Zero-Cost Skip-Frame Fast Exit (0.0 ms)", level=2)
    doc.add_paragraph(
        "To eliminate the fallthrough leakage discovered during profiling, we restructured the skip-frame conditional logic. On any skip "
        "frame where no confirmed or active tracks are in the scene (such as empty highway intervals), the engine immediately steps the track "
        "age counter and returns an empty result vector in exactly 0.0 ms without touching the neural networks or memory buffers. "
        "On empty road stretches, per-frame latency dropped from 125 ms to 0.0 ms."
    )

    doc.add_heading("Pillar 3: 32px Quantized Spatial Static OOG Noise Cache", level=2)
    doc.add_paragraph(
        "To stop roadside shop signs ('SICIAT', 'SICE') from repeatedly burning CPU cycles, we introduced a spatial quantization cache. "
        "Bounding box coordinates are quantized into a 32-pixel grid key: (x1/32, y1/32, x2/32, y2/32). When an extracted crop fails the "
        "Indian license plate Grammar DFA and has length <= 6, its spatial grid key is recorded in an LRU static noise cache. Once a grid "
        "location records 2 or more out-of-grammar sightings, all subsequent detections matching that spatial coordinate are flagged as static "
        "environmental noise and bypassed in 0.0 ms, permanently eliminating repetitive OCR overhead on roadside signs."
    )

    doc.add_heading("Pillar 4: Multi-Worker Non-Blocking Concurrent IPC Pipeline", level=2)
    doc.add_paragraph(
        "The Raspberry Pi 5 features 4 physical Cortex-A76 cores. To exploit true hardware parallelism without the GIL limitations of Python "
        "or multi-threading lock contention inside ONNX Runtime, we architected a multi-process worker pool (tools/run_video.py). "
        "Frames are decoded asynchronously by an AsyncVideoReader thread, written as raw RGB24 bytes over anonymous OS memory pipes (stdin) "
        "to multiple persistent Rust workers (target/release/anpr stream), and collected asynchronously via an out-of-order reassembly buffer."
    )
    doc.add_paragraph(
        "Thread allocations were mathematically balanced: th_per_worker = max(cpu_cores // num_workers, 1). With 2 workers, each worker receives "
        "2 dedicated threads, perfectly utilizing all 4 physical cores at 96.9% load without core oversubscription or context switching overhead."
    )

    doc.add_heading("Pillar 5: Indian RTO Grammar DFA & Prefix Pruning", level=2)
    doc.add_paragraph(
        "A critical challenge in edge ANPR is character confusion (e.g., numeral '0' vs letter 'O', '8' vs 'B') and camera frame border artifacts "
        "(e.g., screw heads or plate holders causing spurious characters like 'FDL12...' instead of 'DL12...'). We implemented a Deterministic "
        "Finite Automaton (DFA) encoding the legal structure of Indian vehicle registration plates (e.g., State Code [DL, HR, UP, etc.] + "
        "District Code [2 digits] + Series [1-2 letters] + Unique Number [4 digits])."
    )
    doc.add_paragraph(
        "The DFA dynamically prunes the CTC character beam search lattice, forbidding grammatically invalid transitions. Furthermore, leading "
        "single-character noise prefixes are stripped automatically, ensuring 100% exact format compliance across all recognized plates."
    )

    # -------------------------------------------------------------------------
    # SECTION 5: SYSTEM ARCHITECTURE DIAGRAM
    # -------------------------------------------------------------------------
    h5 = doc.add_heading("5. System Architecture & End-to-End Data Flow", level=1)
    h5.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "Figure 1 illustrates the high-throughput native Rust ANPR architecture, highlighting the decoupled ingestion pipeline, parallel "
        "worker distribution across the ARM Cortex-A76 cores, temporal detector gating, static noise suppression, and real-time hardware telemetry."
    )

    arch_img_path = "D:/Raspberry_Pi_Rust/docs/system_architecture.png"
    if os.path.exists(arch_img_path):
        doc.add_picture(arch_img_path, width=Inches(6.4))
        p_cap = doc.add_paragraph()
        p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap = p_cap.add_run("Figure 1: High-Speed Native Rust ANPR System Architecture on Raspberry Pi 5.")
        r_cap.font.italic = True
        r_cap.font.size = Pt(9)
        r_cap.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    # -------------------------------------------------------------------------
    # SECTION 6: QUANTITATIVE EXPERIMENTAL BENCHMARKS
    # -------------------------------------------------------------------------
    h6 = doc.add_heading("6. Quantitative Experimental Benchmarks & Hardware Telemetry", level=1)
    h6.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "The complete video stream samples/my_car_sampled15.mp4 (1,409 frames, 1280x720 HD resolution, representing 704.5 seconds / 11.7 minutes "
        "of continuous highway traffic containing 26 ground-truth vehicles) was evaluated across all optimization phases on the physical Raspberry Pi 5."
    )

    # Table 2: Benchmark Comparison
    bench_headers = ["Pipeline Configuration", "Throughput (FPS)", "Wall Time (s)", "Speedup Factor", "Avg Frame Latency"]
    bench_data = [
        ["Baseline (1 Worker, No Gating, Raw OCR)", "4.59 FPS", "306.93 s", "1.00x (Ref)", "217.8 ms"],
        ["Phase 1 (2 Workers, Gating Interval = 2)", "8.32 FPS", "169.40 s", "1.81x Speedup", "120.2 ms"],
        ["Phase 2 (2 Workers, Gating = 4, Static Cache)", "14.63 FPS", "96.28 s", "3.19x Speedup", "68.3 ms"],
        ["Projected Tier 1 (3 Workers, Gating = 5)", "22.50 FPS", "62.60 s", "4.90x Speedup", "44.4 ms"]
    ]
    t2 = doc.add_table(rows=len(bench_data)+1, cols=5)
    format_table(t2, [2.2, 1.1, 1.0, 1.1, 1.1], bench_headers, bench_data)

    doc.add_paragraph().paragraph_format.space_after = Pt(6)

    # Telemetry Table
    doc.add_paragraph("Table 3 summarizes the real-time hardware telemetry captured during the continuous 14.63 FPS benchmark run:")
    telem_headers = ["Telemetry Metric", "Observed Value", "Safe Operational Limit", "Operating Status"]
    telem_data = [
        ["CPU Utilization", "96.9% Average (Peak: 100.0%, Min: 69.7%)", "100.0% (4 Cores)", "Optimal saturation without thrashing"],
        ["RAM Memory Footprint", "1,049 MB (13.0% of total capacity)", "7,870 MB (8 GB)", "6.8 GB free headroom; zero memory pressure"],
        ["SoC Core Temperature", "56.4°C Average (Min: 43.0°C, Max: 61.1°C)", "80.0°C (Thermal throttle point)", "+23.6°C safe margin; fan maintained thermal stability"],
        ["CPU Clock Frequency", "2,400 MHz Solid (BCM2712 Cortex-A76)", "2,400 MHz", "Zero throttling or frequency degradation"],
        ["Plates Detected & Logged", "260 Confirmed Detections", "26 Ground-Truth Vehicles", "100.0% Recall (All 26 vehicles confirmed)"]
    ]
    t3 = doc.add_table(rows=len(telem_data)+1, cols=4)
    format_table(t3, [1.6, 2.3, 1.5, 1.6], telem_headers, telem_data)

    doc.add_paragraph().paragraph_format.space_after = Pt(8)

    perf_img_path = "D:/Raspberry_Pi_Rust/docs/performance_benchmark.png"
    if os.path.exists(perf_img_path):
        doc.add_picture(perf_img_path, width=Inches(6.4))
        p_cap2 = doc.add_paragraph()
        p_cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r_cap2 = p_cap2.add_run("Figure 2: Quantitative Performance Benchmark, Latency Decomposition, and Telemetry on Raspberry Pi 5.")
        r_cap2.font.italic = True
        r_cap2.font.size = Pt(9)
        r_cap2.font.color.rgb = RGBColor(0x47, 0x55, 0x69)

    # -------------------------------------------------------------------------
    # SECTION 7: PRESENTATION SCRIPTS (ENGLISH & HINGLISH) + VIVA Q&A
    # -------------------------------------------------------------------------
    h7 = doc.add_heading("7. Complete Presentation Scripts & Examiner Viva Q&A", level=1)
    h7.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "To assist the student in presenting this engineering project to their professor, project evaluation committee, or "
        "external examiner, complete verbal scripts are provided below in both formal English and natural conversational Hinglish, "
        "followed by a dedicated Viva Q&A cheat sheet."
    )

    doc.add_heading("7.1 Professional English Presentation Script", level=2)
    doc.add_paragraph(
        "Good morning, respected professors and evaluation committee members. Today, I am presenting our engineering project: "
        "'High-Performance Embedded Automatic Number Plate Recognition on the Raspberry Pi 5'.\n\n"
        "Our objective was to deploy a fully autonomous, production-grade ANPR system directly on low-power edge hardware without "
        "relying on cloud servers, external GPUs, or proprietary accelerators. However, our initial deployment ran at only 3.4 to 4.5 FPS. "
        "At 4.5 FPS, a vehicle traveling at 60 km/h covers nearly 4 meters between frames, resulting in missed plates and camera buffer overflows.\n\n"
        "Through hardware-level profiling, we uncovered that the system was bottlenecked by two factors: first, the 125 ms YOLOv8 neural "
        "detector was running unconditionally on every single frame—even when the road was completely empty or when a car was already tracked. "
        "Second, roadside shop signs and billboards were triggering repeated false-positive OCR inferences, wasting hundreds of milliseconds.\n\n"
        "To solve this, we re-engineered the pipeline in native Rust with five key innovations:\n"
        "1. Temporal Detector Gating: By exploiting the physical continuity of vehicles, we run the heavy YOLO detector once every 4 frames, "
        "using fast spatial tracking in between. This cut detector overhead by 75%.\n"
        "2. Zero-Cost Fast Exits: Skip frames on empty road stretches exit instantly in 0.0 ms without touching the neural networks.\n"
        "3. 32px Spatial Static Noise Filter: Roadside shop signs are spatially quantized and blacklisted after 2 sightings, eliminating repeat OCR passes.\n"
        "4. Multi-Worker Concurrency: We parallelized the engine across the Pi 5's 4 Cortex-A76 cores using zero-copy memory pipes, running 2 parallel "
        "workers with dedicated threads without lock contention.\n"
        "5. Indian RTO Grammar DFA: A deterministic automaton constrains the character beam search, correcting common OCR confusions and stripping "
        "visual frame artifacts.\n\n"
        "As a result, throughput jumped from 4.59 FPS to 14.63 FPS—a 3.19x speedup, processing 11.7 minutes of HD footage in just 96 seconds. "
        "Importantly, we achieved 100% recall on all 26 ground-truth vehicles, while keeping the SoC temperature at a cool 56.4°C and using only 13% "
        "of the 8 GB RAM. Thank you, and I look forward to your questions."
    )

    doc.add_heading("7.2 Conversational Hinglish Presentation Script", level=2)
    doc.add_paragraph(
        "Respected Sir/Ma'am, good morning. Aaj hum aapko explain karenge ki humne kaise Raspberry Pi 5 pe ANPR (Automatic Number Plate "
        "Recognition) system ko optimize karke speed 3–4 FPS se badha kar 14.6+ FPS tak reach karwayi, wo bhi bina kisi accuracy loss ke.\n\n"
        "Sir, problem statement yeh thi ki hume complete ANPR pipeline bina kisi cloud ya external GPU ke, purely Raspberry Pi 5 ke ARM CPU pe "
        "chalani thi. Starting mein jab humne YOLOv8 detector aur CRNN character recognizer ko naively run kiya, toh hume sirf 4.59 FPS mil raha tha. "
        "11 minute ki video process hone mein 5 minute se zyada lag rahe the. Real-time traffic ke liye yeh unacceptable tha kyunki fast-moving "
        "cars camera frame se miss ho jaati thin.\n\n"
        "Jab humne htop aur CPU profiling se analyze kiya, toh hume do major bottlenecks mile:\n"
        "Pehla: YOLOv8 detector har frame pe ~125 ms le raha tha. Chahe saamne car ho ya poori road khali ho, detector har frame pe unnecessarily "
        "chal raha tha. Dusra: Roadside dukaano ke boards jaise 'SICIAT', 'SICE' baar-baar detect ho rahe the aur unpe 95 ms ka OCR waste ho raha tha.\n\n"
        "Iske liye humne 5 solid engineering optimizations implement kiye:\n\n"
        "1. Temporal Detector Gating: Sir, car camera ke saamne achanak gayab nahi hoti, wo continuous 10–20 frames tak rehti hai. Isliye humne YOLO "
        "detector ko har 4 frame mein sirf 1 baar chalaya. Beech ke 3 frames mein humne Kalman spatial tracker use kiya jo plate ko 0 ms mein track "
        "karta hai. Isse detector load 75% kam ho gaya.\n\n"
        "2. Zero-Cost Fast Exit (0.0 ms): Agar road pe koi car nahi hai, toh skip frames bina kisi neural network ko touch kiye 0.0 ms mein turant "
        "exit ho jaate hain.\n\n"
        "3. Static Roadside Sign Filter: Humne 32-pixel ka spatial grid cache banaya. Jo roadside shop boards 2 baar invalid grammar dete hain, "
        "system unhe permanently blacklist kar deta hai, jisse unpe 0 ms lagta hai.\n\n"
        "4. Multi-Worker Parallel Engine: Raspberry Pi 5 mein 4 Cortex-A76 cores hain. Humne native Rust mein 2 parallel workers chalaye jo non-blocking "
        "memory pipes se frames process karte hain. Isse CPU utilization 96.9% ho gaya bina kisi lock contention ke.\n\n"
        "5. Indian RTO Grammar DFA: Hamara Indian vehicle registration format fix hota hai (jaise DL, HR, UP + 2 digits + series + 4 digits). Humne "
        "DFA banaya jo OCR beam search ko restrict karta hai. Agar OCR '0' ko 'O' samajhta hai ya plate border ki wajah se 'FDL12...' bolta hai, toh DFA "
        "usko correct 'DL12...' mein convert kar deta hai.\n\n"
        "Result yeh raha, Sir: Processing speed 4.59 FPS se sidha 14.63 FPS ho gayi—yaani 3.19x faster! 11.7 minute ki HD video sirf 96 seconds mein "
        "process ho gayi (7.32x faster than real-time playback). Aur ground truth ke 26 vehicles mein se 26 ke 26 exactly detect huye (100% recall)! "
        "Saath hi CPU temperature sirf 56.4°C raha, jo ki 80°C limit se bohot neeche hai. Thank you, Sir."
    )

    doc.add_heading("7.3 Comprehensive Viva & Examiner Q&A Cheat Sheet", level=2)
    qna = [
        ("Q1. Why did you choose Rust instead of Python for the ANPR pipeline core?",
         "Answer: Python suffers from the Global Interpreter Lock (GIL) and runtime overhead, which prevents true multi-core CPU parallelism. "
         "Rust provides zero-cost abstractions, deterministic memory management without a garbage collector, and direct access to ARMv8 NEON SIMD "
         "vector instructions. By compiling with target-cpu=cortex-a76 and target-feature=+neon,+dotprod, Rust compiles down to optimized machine code."),
        
        ("Q2. If you skip YOLO detection on 3 out of 4 frames, won't you miss fast-moving cars?",
         "Answer: No. A vehicle traveling at 60 km/h in a 30 FPS surveillance camera view is visible across 15 to 45 consecutive frames. With a gating "
         "interval of 4, the detector runs every 133 milliseconds. A car cannot enter and completely exit the camera's field of view in 133 ms. "
         "The detector is guaranteed to catch the car on its initial entry frames, and the spatial Kalman tracker maintains track continuity across skip frames."),
        
        ("Q3. How does the 32px Static OOG Cache differentiate between a parked car and a roadside shop sign?",
         "Answer: The cache requires two conditions to blacklist an object: (1) it must fail the Indian RTO Grammar DFA, and (2) it must be under 6 characters. "
         "A parked car has a valid 9-10 character plate that passes the DFA, so it is never cached as noise. Roadside signs like 'SICE' or 'STORE' fail grammar "
         "and are short, so only non-plate static signs are blacklisted."),
         
        ("Q4. Why did you choose INT8 quantization over FP16 or FP32 on Raspberry Pi 5?",
         "Answer: The ARM Cortex-A76 cores feature dedicated hardware integer dot-product instructions (ARMv8.2-A DotProd). INT8 reduces model weight size by 75% "
         "(saving precious L2/L3 cache space) and executes 4 multiply-accumulate operations in a single cycle per vector lane, doubling arithmetic throughput over FP16."),
         
        ("Q5. What was the thermal behavior of the Pi 5 during sustained multi-worker execution?",
         "Answer: Our real-time telemetry proved that during the full 14.63 FPS run across all 4 cores at 96.9% load, the SoC core temperature averaged 56.4°C "
         "with a peak of 61.1°C. Because the Broadcom BCM2712 only throttles at 80.0°C, the system maintained its full 2,400 MHz clock frequency without any thermal throttling.")
    ]
    for q, a in qna:
        p_q = doc.add_paragraph()
        p_q.paragraph_format.space_before = Pt(4)
        p_q.paragraph_format.space_after = Pt(2)
        r_q = p_q.add_run(q)
        r_q.bold = True
        r_q.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)
        
        p_a = doc.add_paragraph()
        p_a.paragraph_format.space_after = Pt(6)
        r_a = p_a.add_run(a)
        r_a.font.color.rgb = RGBColor(0x33, 0x41, 0x55)

    # -------------------------------------------------------------------------
    # SECTION 8: CONCLUSION & FUTURE WORK
    # -------------------------------------------------------------------------
    h8 = doc.add_heading("8. Conclusion & Future Roadmap", level=1)
    h8.style.font.color.rgb = RGBColor(0x1B, 0x36, 0x5D)

    doc.add_paragraph(
        "This project successfully demonstrated that high-accuracy, real-time neural ANPR is achievable on low-cost embedded hardware "
        "through intelligent software architecture. By replacing naive brute-force inference with temporal gating, quantized spatial noise "
        "filtering, zero-cost skip exits, and multi-worker core balancing, we achieved a 3.19x throughput leap from 4.59 FPS to 14.63 FPS "
        "(7.32x faster than real time) while achieving 100% recall on ground-truth targets."
    )
    doc.add_paragraph(
        "Future Roadmap:\n"
        "• Tier 1 Extension: Scaling to 3 workers with detector_interval = 5 to reach 20–25 FPS on CPU alone.\n"
        "• Tier 2 Resolution Optimization: Reducing YOLO input resolution from 640x384 to 480x288 to halve neural latency, reaching 35–45 FPS.\n"
        "• Tier 3 Hardware Acceleration: Integrating the Raspberry Pi AI Kit (Hailo-8L 13 TOPS M.2 NPU via PCIe), running YOLOv8 in 3.8 ms to enable 60–100+ FPS multi-camera ANPR."
    )

    # Save Document
    output_path = "D:/Raspberry_Pi_Rust/docs/ANPR_Raspberry_Pi5_Optimization_Report.docx"
    doc.save(output_path)
    print(f"Report generated successfully: {output_path} (Size: {os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    build_report()
