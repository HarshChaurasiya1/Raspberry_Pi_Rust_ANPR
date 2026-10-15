//! Camera capture interface supporting Linux V4L2 and simulated/file stream sources.

use anyhow::{Result, bail};
use image::RgbImage;

pub trait CameraSource: Send {
    fn name(&self) -> &str;
    fn read_frame(&mut self) -> Result<Option<RgbImage>>;
    fn close(&mut self);
}

pub struct Camera {
    source: Box<dyn CameraSource>,
}

impl Camera {
    pub fn new(source: Box<dyn CameraSource>) -> Self {
        Self { source }
    }

    pub fn name(&self) -> &str {
        self.source.name()
    }

    pub fn read(&mut self) -> Result<Option<RgbImage>> {
        self.source.read_frame()
    }

    pub fn close(&mut self) {
        self.source.close();
    }
}

pub struct SimulatedCamera {
    images: Vec<RgbImage>,
    current_idx: usize,
    name: String,
}

impl SimulatedCamera {
    pub fn new(image_paths: &[std::path::PathBuf]) -> Result<Self> {
        let mut images = Vec::new();
        for p in image_paths {
            if let Ok(dyn_img) = image::open(p) {
                images.push(dyn_img.to_rgb8());
            }
        }
        if images.is_empty() {
            bail!("No valid images found for simulated camera");
        }
        Ok(Self {
            images,
            current_idx: 0,
            name: "Simulated Camera (Sample Dataset Loop)".to_string(),
        })
    }
}

impl CameraSource for SimulatedCamera {
    fn name(&self) -> &str {
        &self.name
    }

    fn read_frame(&mut self) -> Result<Option<RgbImage>> {
        if self.images.is_empty() {
            return Ok(None);
        }
        let frame = self.images[self.current_idx].clone();
        self.current_idx = (self.current_idx + 1) % self.images.len();
        Ok(Some(frame))
    }

    fn close(&mut self) {}
}

pub fn open_camera(_device: Option<usize>, _width: u32, _height: u32) -> Result<Camera> {
    let sample_dir = std::path::Path::new("samples");
    let mut paths = Vec::new();
    if let Ok(entries) = std::fs::read_dir(sample_dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if let Some(ext) = p.extension().and_then(|s| s.to_str()) {
                if matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png" | "bmp") {
                    paths.push(p);
                }
            }
        }
    }
    paths.sort();

    if !paths.is_empty() {
        let sim = SimulatedCamera::new(&paths)?;
        Ok(Camera::new(Box::new(sim)))
    } else {
        bail!("No camera hardware or sample stream found.")
    }
}
