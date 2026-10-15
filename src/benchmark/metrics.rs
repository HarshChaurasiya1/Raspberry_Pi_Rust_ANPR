//! Telemetry, RAM, CPU, and Accuracy Metrics.

use sysinfo::{System, Pid};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage_pct: f32,
    pub ram_rss_mb: f64,
    pub temp_c: Option<f32>,
}

pub fn collect_system_metrics() -> SystemMetrics {
    let mut sys = System::new_all();
    sys.refresh_all();

    let pid = Pid::from_u32(std::process::id());
    let mut ram_mb = 0.0;
    let mut cpu_pct = 0.0;

    if let Some(proc_) = sys.process(pid) {
        ram_mb = proc_.memory() as f64 / (1024.0 * 1024.0);
        cpu_pct = proc_.cpu_usage();
    }

    SystemMetrics {
        cpu_usage_pct: cpu_pct,
        ram_rss_mb: ram_mb,
        temp_c: None,
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AccuracyMetrics {
    pub total_samples: usize,
    pub total_ground_truth_plates: usize,
    pub detected_plates: usize,
    pub true_positives: usize,
    pub false_positives: usize,
    pub missed_plates: usize,
    pub recall_pct: f32,
    pub precision_pct: f32,
    pub exact_match_count: usize,
    pub exact_match_pct: f32,
    pub grammar_valid_count: usize,
    pub grammar_valid_pct: f32,
}
