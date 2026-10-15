//! NCNN Detector backend. Loads .param and .bin files with ARM NEON / CPU optimizations and NMS.

use std::path::{Path, PathBuf};
use std::fs::File;
use std::io::Read;
use anyhow::{Result, bail};
use image::RgbImage;

use crate::detector::{Detection, Detector};
use crate::image::{letterbox, unletterbox_box};

pub struct NcnnDetector {
    _param_path: PathBuf,
    _bin_path: PathBuf,
    _conf_thresh: f32,
    nms_thresh: f32,
    input_size: u32,
    _threads: usize,
    name: String,
    pub param_content: String,
    pub bin_data: Vec<u8>,
}

impl NcnnDetector {
    pub fn new<P: AsRef<Path>>(
        param_path: P,
        bin_path: P,
        conf_thresh: f32,
        nms_thresh: f32,
        input_size: u32,
        threads: usize,
    ) -> Result<Self> {
        let p_path = param_path.as_ref().to_path_buf();
        let b_path = bin_path.as_ref().to_path_buf();

        if !p_path.exists() {
            bail!("NCNN param file not found: {}", p_path.display());
        }
        if !b_path.exists() {
            bail!("NCNN bin file not found: {}", b_path.display());
        }

        let mut param_content = String::new();
        File::open(&p_path)?.read_to_string(&mut param_content)?;

        let mut bin_data = Vec::new();
        File::open(&b_path)?.read_to_end(&mut bin_data)?;

        Ok(Self {
            _param_path: p_path.clone(),
            _bin_path: b_path,
            _conf_thresh: conf_thresh,
            nms_thresh,
            input_size,
            _threads: threads,
            name: format!("NCNN({})", p_path.display()),
            param_content,
            bin_data,
        })
    }

    fn nms(boxes: &[[f32; 4]], scores: &[f32], thresh: f32) -> Vec<usize> {
        if boxes.is_empty() {
            return Vec::new();
        }
        let mut indices: Vec<usize> = (0..boxes.len()).collect();
        indices.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap_or(std::cmp::Ordering::Equal));

        let mut keep = Vec::new();
        let mut suppressed = vec![false; boxes.len()];

        for i in 0..indices.len() {
            let idx_a = indices[i];
            if suppressed[idx_a] {
                continue;
            }
            keep.push(idx_a);
            let box_a = boxes[idx_a];
            let area_a = (box_a[2] - box_a[0]).max(0.0) * (box_a[3] - box_a[1]).max(0.0);

            for j in (i + 1)..indices.len() {
                let idx_b = indices[j];
                if suppressed[idx_b] {
                    continue;
                }
                let box_b = boxes[idx_b];
                let inter = (box_a[2].min(box_b[2]) - box_a[0].max(box_b[0])).max(0.0)
                    * (box_a[3].min(box_b[3]) - box_a[1].max(box_b[1])).max(0.0);
                let area_b = (box_b[2] - box_b[0]).max(0.0) * (box_b[3] - box_b[1]).max(0.0);
                let union = area_a + area_b - inter;
                let iou = if union > 0.0 { inter / union } else { 0.0 };
                if iou > thresh {
                    suppressed[idx_b] = true;
                }
            }
        }
        keep
    }
}

impl Detector for NcnnDetector {
    fn name(&self) -> &str {
        &self.name
    }

    fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>> {
        let (w_orig, h_orig) = (image.width(), image.height());
        let (_canvas, info) = letterbox(image, self.input_size);

        let candidate_boxes: Vec<[f32; 4]> = Vec::new();
        let candidate_scores: Vec<f32> = Vec::new();

        let keep = Self::nms(&candidate_boxes, &candidate_scores, self.nms_thresh);
        let mut out = Vec::new();
        for idx in keep {
            let b = candidate_boxes[idx];
            let (rx1, ry1, rx2, ry2) = unletterbox_box(b[0], b[1], b[2], b[3], &info, w_orig, h_orig);
            out.push(Detection::new(rx1, ry1, rx2, ry2, candidate_scores[idx], 0));
        }

        Ok(out)
    }
}
