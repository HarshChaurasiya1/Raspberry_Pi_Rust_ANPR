//! Tests for configuration loading and validation.

use raspberry_pi_rust::config::Config;

#[test]
fn test_default_config() {
    let cfg = Config::default();
    assert_eq!(cfg.detector.input_size, 640);
    assert_eq!(cfg.detector.confidence_threshold, 0.25);
    assert_eq!(cfg.decoding.beam_width, 8);
    assert_eq!(cfg.decoding.num_variants, 3);
    assert_eq!(cfg.detector.threads, 3);
}

#[test]
fn test_load_current_onnx_config() {
    let cfg = Config::load("configs/current_onnx.toml").expect("Failed to load current_onnx.toml");
    assert_eq!(cfg.detector.backend, "onnx");
    assert_eq!(cfg.detector.model_path, "models/current/detector_int8.onnx");
}

#[test]
fn test_load_current_ncnn_config() {
    let cfg = Config::load("configs/current_ncnn.toml").expect("Failed to load current_ncnn.toml");
    assert_eq!(cfg.detector.backend, "ncnn");
}
