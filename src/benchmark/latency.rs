//! Microsecond per-stage latency measurement and statistics calculation.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageTiming {
    pub camera_capture_ms: f64,
    pub frame_prepare_ms: f64,
    pub detector_preprocess_ms: f64,
    pub detector_inference_ms: f64,
    pub detector_postprocess_ms: f64,
    pub crop_ms: f64,
    pub plate_preprocess_ms: f64,
    pub ocr_preprocess_ms: f64,
    pub ocr_inference_ms: f64,
    pub ctc_decode_ms: f64,
    pub grammar_decode_ms: f64,
    pub final_postprocess_ms: f64,
    pub total_pipeline_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStats {
    pub count: usize,
    pub min_ms: f64,
    pub avg_ms: f64,
    pub median_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

impl LatencyStats {
    pub fn compute(mut values: Vec<f64>) -> Self {
        if values.is_empty() {
            return Self {
                count: 0,
                min_ms: 0.0,
                avg_ms: 0.0,
                median_ms: 0.0,
                p95_ms: 0.0,
                p99_ms: 0.0,
                max_ms: 0.0,
            };
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let count = values.len();
        let sum: f64 = values.iter().sum();
        let avg = sum / count as f64;
        let min = values[0];
        let max = values[count - 1];

        let median = if count % 2 == 0 {
            (values[count / 2 - 1] + values[count / 2]) / 2.0
        } else {
            values[count / 2]
        };

        let p95_idx = ((count as f64 * 0.95).round() as usize).min(count - 1);
        let p99_idx = ((count as f64 * 0.99).round() as usize).min(count - 1);

        Self {
            count,
            min_ms: min,
            avg_ms: avg,
            median_ms: median,
            p95_ms: values[p95_idx],
            p99_ms: values[p99_idx],
            max_ms: max,
        }
    }
}
