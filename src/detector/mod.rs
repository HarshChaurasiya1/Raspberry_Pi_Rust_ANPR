//! Plate Detector interface and implementations for ONNX Runtime and NCNN backends.

pub mod onnx;
pub mod ncnn;

use anyhow::Result;
use image::RgbImage;
use crate::config::DetectorConfig;

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
    pub score: f32,
    pub class_id: usize,
}

impl Detection {
    pub fn new(x1: u32, y1: u32, x2: u32, y2: u32, score: f32, class_id: usize) -> Self {
        Self { x1, y1, x2, y2, score, class_id }
    }

    pub fn box_tuple(&self) -> (u32, u32, u32, u32) {
        (self.x1, self.y1, self.x2, self.y2)
    }

    pub fn crop(&self, image: &RgbImage, pad: u32) -> RgbImage {
        crate::image::safe_crop(image, self.x1, self.y1, self.x2, self.y2, pad)
    }
}

pub trait Detector: Send + Sync {
    fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>>;
    fn name(&self) -> &str;
}

pub fn create_detector(config: &DetectorConfig) -> Result<Box<dyn Detector>> {
    match config.backend.to_lowercase().as_str() {
        "onnx" => {
            let det = onnx::OnnxDetector::new(
                &config.model_path,
                config.confidence_threshold,
                config.nms_threshold,
                config.input_size,
                config.threads,
                config.allow_spinning,
            )?;
            Ok(Box::new(det))
        }
        "ncnn" => {
            let det = ncnn::NcnnDetector::new(
                &config.param_path,
                &config.bin_path,
                config.confidence_threshold,
                config.nms_threshold,
                config.input_size,
                config.threads,
            )?;
            Ok(Box::new(det))
        }
        other => anyhow::bail!("Unsupported detector backend: {}", other),
    }
}
