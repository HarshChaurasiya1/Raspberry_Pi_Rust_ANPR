//! Runtime configuration loaded from TOML files or environment variables.

use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};
use anyhow::{Context, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    #[serde(default = "default_detector_model")]
    pub model_path: String,
    #[serde(default = "default_empty_string")]
    pub param_path: String,
    #[serde(default = "default_empty_string")]
    pub bin_path: String,
    #[serde(default = "default_detector_backend")]
    pub backend: String, // "onnx" or "ncnn"
    #[serde(default = "default_input_size")]
    pub input_size: u32,
    #[serde(default = "default_detection_conf")]
    pub confidence_threshold: f32,
    #[serde(default = "default_nms_threshold")]
    pub nms_threshold: f32,
    #[serde(default = "default_max_plates")]
    pub max_plates: usize,
    #[serde(default = "default_crop_padding")]
    pub crop_padding: u32,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default = "default_false")]
    pub allow_spinning: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            model_path: default_detector_model(),
            param_path: String::new(),
            bin_path: String::new(),
            backend: default_detector_backend(),
            input_size: default_input_size(),
            confidence_threshold: default_detection_conf(),
            nms_threshold: default_nms_threshold(),
            max_plates: default_max_plates(),
            crop_padding: default_crop_padding(),
            threads: default_threads(),
            allow_spinning: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecognizerConfig {
    #[serde(default = "default_recognizer_model")]
    pub model_path: String,
    #[serde(default = "default_charset_path")]
    pub charset_path: String,
    #[serde(default = "default_rec_height")]
    pub input_height: u32,
    #[serde(default = "default_rec_width")]
    pub input_width: u32,
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default = "default_false")]
    pub allow_spinning: bool,
    #[serde(default = "default_false")]
    pub grayscale: bool,
}

impl Default for RecognizerConfig {
    fn default() -> Self {
        Self {
            model_path: default_recognizer_model(),
            charset_path: default_charset_path(),
            input_height: default_rec_height(),
            input_width: default_rec_width(),
            threads: default_threads(),
            allow_spinning: false,
            grayscale: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecodingConfig {
    #[serde(default = "default_lambda_grammar")]
    pub lambda_grammar: f32,
    #[serde(default = "default_beam_width")]
    pub beam_width: usize,
    #[serde(default = "default_nbest")]
    pub nbest: usize,
    #[serde(default = "default_num_variants")]
    pub num_variants: usize,
    #[serde(default = "default_false")]
    pub strict_grammar: bool,
}

impl Default for DecodingConfig {
    fn default() -> Self {
        Self {
            lambda_grammar: default_lambda_grammar(),
            beam_width: default_beam_width(),
            nbest: default_nbest(),
            num_variants: default_num_variants(),
            strict_grammar: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeConfig {
    #[serde(default = "default_threads")]
    pub threads: usize,
    #[serde(default = "default_false")]
    pub allow_spinning: bool,
    #[serde(default = "default_frame_skip")]
    pub frame_skip: usize,
    #[serde(default = "default_cam_res")]
    pub camera_resolution: (u32, u32),
    #[serde(default = "default_true")]
    pub tracking: bool,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        Self {
            threads: default_threads(),
            allow_spinning: false,
            frame_skip: default_frame_skip(),
            camera_resolution: default_cam_res(),
            tracking: true,
        }
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub detector: DetectorConfig,
    #[serde(default)]
    pub recognizer: RecognizerConfig,
    #[serde(default)]
    pub decoding: DecodingConfig,
    #[serde(default)]
    pub runtime: RuntimeConfig,
}

// Default helper functions
fn default_detector_model() -> String { "models/current/detector_int8.onnx".to_string() }
fn default_recognizer_model() -> String { "models/current/recognizer.onnx".to_string() }
fn default_charset_path() -> String { "models/current/charset.txt".to_string() }
fn default_empty_string() -> String { String::new() }
fn default_detector_backend() -> String { "onnx".to_string() }
fn default_input_size() -> u32 { 640 }
fn default_detection_conf() -> f32 { 0.25 }
fn default_nms_threshold() -> f32 { 0.45 }
fn default_max_plates() -> usize { 5 }
fn default_crop_padding() -> u32 { 4 }
fn default_threads() -> usize { 3 }
fn default_false() -> bool { false }
fn default_true() -> bool { true }
fn default_rec_height() -> u32 { 48 }
fn default_rec_width() -> u32 { 320 }
fn default_lambda_grammar() -> f32 { 1.0 }
fn default_beam_width() -> usize { 8 }
fn default_nbest() -> usize { 8 }
fn default_num_variants() -> usize { 3 }
fn default_frame_skip() -> usize { 1 }
fn default_cam_res() -> (u32, u32) { (1280, 720) }

impl Config {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file: {}", path.as_ref().display()))?;
        let mut cfg: Config = toml::from_str(&content)
            .with_context(|| format!("Failed to parse TOML from: {}", path.as_ref().display()))?;
        cfg.apply_env_overrides();
        Ok(cfg)
    }

    pub fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("ANPR_THREADS") {
            if let Ok(n) = v.parse::<usize>() {
                self.detector.threads = n;
                self.recognizer.threads = n;
                self.runtime.threads = n;
            }
        }
        if let Ok(v) = std::env::var("ANPR_CONF") {
            if let Ok(f) = v.parse::<f32>() {
                self.detector.confidence_threshold = f;
            }
        }
        if let Ok(v) = std::env::var("ANPR_LAMBDA") {
            if let Ok(f) = v.parse::<f32>() {
                self.decoding.lambda_grammar = f;
            }
        }
        if let Ok(v) = std::env::var("ANPR_VARIANTS") {
            if let Ok(n) = v.parse::<usize>() {
                self.decoding.num_variants = n;
            }
        }
        if let Ok(v) = std::env::var("ANPR_BEAM") {
            if let Ok(n) = v.parse::<usize>() {
                self.decoding.beam_width = n;
            }
        }
    }
}
