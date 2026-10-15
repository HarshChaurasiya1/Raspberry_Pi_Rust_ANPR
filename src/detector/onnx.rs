//! ONNX Runtime Detector implementation supporting both end2end NMS graphs and raw YOLO outputs.

use std::path::Path;
use anyhow::{Result, anyhow};
use image::RgbImage;
use ort::{session::Session, value::Tensor};

use crate::detector::{Detection, Detector};
use crate::image::{letterbox_chw_rect, unletterbox_box, LetterboxInfo};

pub struct OnnxDetector {
    session: Session,
    conf_thresh: f32,
    nms_thresh: f32,
    input_w: u32,
    input_h: u32,
    name: String,
}

impl OnnxDetector {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        conf_thresh: f32,
        nms_thresh: f32,
        input_size: u32,
        threads: usize,
        _allow_spinning: bool,
    ) -> Result<Self> {
        let path_str = model_path.as_ref().display().to_string();
        
        let session = Session::builder()
            .map_err(|e| anyhow!("ORT SessionBuilder error: {}", e))?
            .with_intra_threads(threads.max(1))
            .map_err(|e| anyhow!("ORT with_intra_threads error: {}", e))?
            .with_inter_threads(1)
            .map_err(|e| anyhow!("ORT with_inter_threads error: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow!("Failed to load ONNX detector model from {}: {}", path_str, e))?;

        let (input_w, input_h) = if path_str.contains("current") || path_str.contains("384x640") {
            (640, 384)
        } else {
            (input_size, input_size)
        };

        Ok(Self {
            session,
            conf_thresh,
            nms_thresh,
            input_w,
            input_h,
            name: format!("ONNX({})", path_str),
        })
    }

    #[inline]
    fn preprocess(&self, img: &RgbImage) -> (Vec<f32>, LetterboxInfo) {
        letterbox_chw_rect(img, self.input_w, self.input_h)
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
                let xx1 = box_a[0].max(box_b[0]);
                let yy1 = box_a[1].max(box_b[1]);
                let xx2 = box_a[2].min(box_b[2]);
                let yy2 = box_a[3].min(box_b[3]);

                let inter_w = (xx2 - xx1).max(0.0);
                let inter_h = (yy2 - yy1).max(0.0);
                let inter = inter_w * inter_h;

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

impl Detector for OnnxDetector {
    fn name(&self) -> &str {
        &self.name
    }

    fn detect(&mut self, image: &RgbImage) -> Result<Vec<Detection>> {
        let (w_orig, h_orig) = (image.width(), image.height());
        let (input_vec, info) = self.preprocess(image);
        let shape = [1usize, 3usize, self.input_h as usize, self.input_w as usize];

        let tensor_val = Tensor::from_array((shape, input_vec))
            .map_err(|e| anyhow!("Failed to create ORT tensor: {}", e))?;
        
        let outputs = self.session.run(ort::inputs![tensor_val])
            .map_err(|e| anyhow!("Session run failed: {}", e))?;
        let (out_shape, data) = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract tensor data: {}", e))?;

        let shape_slice = out_shape.as_ref();
        let mut candidate_boxes = Vec::new();
        let mut candidate_scores = Vec::new();

        // Check if end2end format: [1, 300, 6] -> (x1, y1, x2, y2, score, class)
        if shape_slice.len() == 3 && shape_slice[2] == 6 {
            let num_boxes = shape_slice[1] as usize;
            for i in 0..num_boxes {
                let offset = i * 6;
                let score = data[offset + 4];
                if score < self.conf_thresh {
                    break; // Rows are sorted by score descending in end2end graph
                }
                let x1 = data[offset + 0];
                let y1 = data[offset + 1];
                let x2 = data[offset + 2];
                let y2 = data[offset + 3];
                candidate_boxes.push([x1, y1, x2, y2]);
                candidate_scores.push(score);
            }
        } else if shape_slice.len() == 3 && shape_slice[1] <= 84 && shape_slice[2] >= 1000 {
            // Standard YOLOv8 output: [1, 4 + classes, 8400]
            let num_channels = shape_slice[1] as usize;
            let num_proposals = shape_slice[2] as usize;

            for i in 0..num_proposals {
                // Find maximum class score
                let mut max_cls_score = 0.0f32;
                for c in 4..num_channels {
                    let s = data[c * num_proposals + i];
                    if s > max_cls_score {
                        max_cls_score = s;
                    }
                }
                if max_cls_score >= self.conf_thresh {
                    let cx = data[0 * num_proposals + i];
                    let cy = data[1 * num_proposals + i];
                    let bw = data[2 * num_proposals + i];
                    let bh = data[3 * num_proposals + i];
                    let x1 = cx - bw * 0.5;
                    let y1 = cy - bh * 0.5;
                    let x2 = cx + bw * 0.5;
                    let y2 = cy + bh * 0.5;
                    candidate_boxes.push([x1, y1, x2, y2]);
                    candidate_scores.push(max_cls_score);
                }
            }
        }

        // Apply NMS
        let keep_indices = Self::nms(&candidate_boxes, &candidate_scores, self.nms_thresh);

        let mut results = Vec::new();
        for idx in keep_indices {
            let b = candidate_boxes[idx];
            let score = candidate_scores[idx];
            let (rx1, ry1, rx2, ry2) = unletterbox_box(b[0], b[1], b[2], b[3], &info, w_orig, h_orig);
            if rx2 > rx1 && ry2 > ry1 {
                results.push(Detection::new(rx1, ry1, rx2, ry2, score, 0));
            }
        }

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(results)
    }
}
