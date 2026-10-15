//! Benchmark report generation in CSV and Markdown formats.

use std::fs::File;
use std::io::Write;
use std::path::Path;
use anyhow::Result;
use crate::benchmark::latency::StageTiming;

pub struct BenchmarkSummary {
    pub configuration_name: String,
    pub detector_name: String,
    pub backend_name: String,
    pub model_size_mb: f64,
    pub detector_ms: f64,
    pub ocr_ms: f64,
    pub decoder_ms: f64,
    pub total_ms: f64,
    pub fps: f64,
    pub recall_pct: f32,
    pub exact_match_pct: f32,
    pub cpu_pct: f32,
    pub ram_mb: f64,
    pub temp_c: String,
}

pub fn write_pipeline_latency_csv<P: AsRef<Path>>(
    path: P,
    timings: &[StageTiming],
    config_name: &str,
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "configuration,camera_capture_ms,frame_prepare_ms,detector_preprocess_ms,detector_inference_ms,detector_postprocess_ms,crop_ms,plate_preprocess_ms,ocr_preprocess_ms,ocr_inference_ms,ctc_decode_ms,grammar_decode_ms,final_postprocess_ms,total_pipeline_ms,fps")?;

    for t in timings {
        let fps = if t.total_pipeline_ms > 0.0 { 1000.0 / t.total_pipeline_ms } else { 0.0 };
        writeln!(
            file,
            "{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
            config_name,
            t.camera_capture_ms,
            t.frame_prepare_ms,
            t.detector_preprocess_ms,
            t.detector_inference_ms,
            t.detector_postprocess_ms,
            t.crop_ms,
            t.plate_preprocess_ms,
            t.ocr_preprocess_ms,
            t.ocr_inference_ms,
            t.ctc_decode_ms,
            t.grammar_decode_ms,
            t.final_postprocess_ms,
            t.total_pipeline_ms,
            fps,
        )?;
    }
    Ok(())
}

pub fn write_four_way_comparison_csv<P: AsRef<Path>>(
    path: P,
    summaries: &[BenchmarkSummary],
) -> Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "Configuration,Detector,Backend,Model Size MB,Detector ms,OCR ms,Decoder ms,Total ms,FPS,Recall %,OCR Exact %,CPU %,RAM MB,Temp")?;

    for s in summaries {
        writeln!(
            file,
            "{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.1},{:.1},{:.1},{:.1},{}",
            s.configuration_name,
            s.detector_name,
            s.backend_name,
            s.model_size_mb,
            s.detector_ms,
            s.ocr_ms,
            s.decoder_ms,
            s.total_ms,
            s.fps,
            s.recall_pct,
            s.exact_match_pct,
            s.cpu_pct,
            s.ram_mb,
            s.temp_c,
        )?;
    }
    Ok(())
}
