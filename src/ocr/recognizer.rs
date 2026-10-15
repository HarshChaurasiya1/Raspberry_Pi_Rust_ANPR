//! PP-OCRv5 Text Recognizer model execution returning full CTC posteriors [T, 37].

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;
use anyhow::{Context, Result, anyhow};
use image::RgbImage;
use ndarray::Array2;
use ort::{session::Session, value::Tensor};

pub const PLATE_CHARSET: &[char] = &[
    '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K',
    'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V',
    'W', 'X', 'Y', 'Z'
];

#[inline(always)]
pub fn char_to_plate_idx(c: char) -> Option<usize> {
    let b = c as u32;
    if (b'0' as u32..=b'9' as u32).contains(&b) {
        Some((b - b'0' as u32 + 1) as usize)
    } else if (b'A' as u32..=b'Z' as u32).contains(&b) {
        Some((b - b'A' as u32 + 11) as usize)
    } else if b == b'-' as u32 {
        Some(0)
    } else {
        None
    }
}

pub struct PlateRecognizer {
    session: Session,
    groups: Vec<Vec<usize>>,
    active_indices: Vec<usize>,
    pub height: u32,
    pub width: u32,
    pub grayscale: bool,
}

impl PlateRecognizer {
    pub fn new<P: AsRef<Path>>(
        model_path: P,
        charset_path: P,
        height: u32,
        width: u32,
        threads: usize,
        _allow_spinning: bool,
        grayscale: bool,
    ) -> Result<Self> {
        let path_str = model_path.as_ref().display().to_string();
        let charset_str = charset_path.as_ref().display().to_string();

        let session = Session::builder()
            .map_err(|e| anyhow!("ORT SessionBuilder error: {}", e))?
            .with_intra_threads(threads.max(1))
            .map_err(|e| anyhow!("ORT with_intra_threads error: {}", e))?
            .with_inter_threads(1)
            .map_err(|e| anyhow!("ORT with_inter_threads error: {}", e))?
            .commit_from_file(&model_path)
            .map_err(|e| anyhow!("Failed to load ONNX recognizer model from {}: {}", path_str, e))?;

        let file = File::open(&charset_path)
            .with_context(|| format!("Failed to read charset file: {}", charset_str))?;
        let reader = BufReader::new(file);
        let mut engine_chars = Vec::new();
        for line in reader.lines() {
            engine_chars.push(line?);
        }

        let mut buckets: Vec<Vec<usize>> = vec![Vec::new(); PLATE_CHARSET.len()];
        buckets[0].push(0); // CTC blank -> blank

        for (engine_idx, ch_str) in engine_chars.iter().enumerate() {
            let actual_idx = engine_idx + 1;
            let upper_s = ch_str.to_uppercase();
            if let Some(first_ch) = upper_s.chars().next() {
                if let Some(target_idx) = PLATE_CHARSET.iter().position(|&c| c == first_ch) {
                    buckets[target_idx].push(actual_idx);
                }
            }
        }

        let mut active_indices: Vec<usize> = buckets.iter().flatten().copied().collect();
        active_indices.sort_unstable();
        active_indices.dedup();

        Ok(Self {
            session,
            groups: buckets,
            active_indices,
            height,
            width,
            grayscale,
        })
    }

    pub fn preprocess(&self, crop: &RgbImage) -> (Vec<f32>, [usize; 4]) {
        let (w, h) = (crop.width(), crop.height());
        let target_w = ((self.height as f32 * (w as f32 / h.max(1) as f32)).ceil() as u32)
            .clamp(8, self.width);

        let resized = image::imageops::resize(crop, target_w, self.height, image::imageops::FilterType::Nearest);
        let h_sz = self.height as usize;
        let w_sz = self.width as usize;
        let plane = h_sz * w_sz;
        let mut vec = vec![0.0f32; 3 * plane];

        for y in 0..self.height {
            let yi = y as usize * w_sz;
            for x in 0..target_w {
                let p = resized.get_pixel(x, y);
                let idx = yi + x as usize;
                if self.grayscale {
                    let lum = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) / 255.0;
                    let norm = (lum - 0.5) * 2.0;
                    vec[idx] = norm;
                    vec[plane + idx] = norm;
                    vec[2 * plane + idx] = norm;
                } else {
                    vec[idx] = (p[0] as f32 / 255.0 - 0.5) * 2.0;
                    vec[plane + idx] = (p[1] as f32 / 255.0 - 0.5) * 2.0;
                    vec[2 * plane + idx] = (p[2] as f32 / 255.0 - 0.5) * 2.0;
                }
            }
        }

        (vec, [1, 3, h_sz, w_sz])
    }

    pub fn lattice(&mut self, crop: &RgbImage) -> Result<Array2<f32>> {
        let (input_vec, shape) = self.preprocess(crop);
        let tensor_val = Tensor::from_array((shape, input_vec))
            .map_err(|e| anyhow!("Failed to create ORT tensor: {}", e))?;
        
        let outputs = self.session.run(ort::inputs![tensor_val])
            .map_err(|e| anyhow!("Recognizer session run failed: {}", e))?;
        let (out_shape, data) = outputs[0].try_extract_tensor::<f32>()
            .map_err(|e| anyhow!("Failed to extract recognizer output tensor: {}", e))?;
        let shape_slice = out_shape.as_ref();

        // Logits shape: [1, T, C]
        let (t_steps, num_classes) = if shape_slice.len() == 3 {
            (shape_slice[1] as usize, shape_slice[2] as usize)
        } else {
            (shape_slice[0] as usize, shape_slice[1] as usize)
        };

        let mut plate_log_probs = Array2::<f32>::zeros((t_steps, PLATE_CHARSET.len()));

        for t in 0..t_steps {
            let offset = t * num_classes;
            let step_logits = &data[offset..offset + num_classes];

            // Only find max over active plate character indices (~73 indices instead of 6625)
            let mut max_l = f32::NEG_INFINITY;
            for &idx in &self.active_indices {
                if idx < num_classes {
                    let v = step_logits[idx];
                    if v > max_l { max_l = v; }
                }
            }

            // Grouped projection directly to 37 plate characters (global denominator cancels out!)
            let mut plate_step_probs = [0.0f32; 37];
            let mut total_plate_prob = 0.0f32;
            for (k, group) in self.groups.iter().enumerate() {
                let mut p_k = 0.0f32;
                for &idx in group {
                    if idx < num_classes {
                        p_k += (step_logits[idx] - max_l).exp();
                    }
                }
                plate_step_probs[k] = p_k;
                total_plate_prob += p_k;
            }

            let log_denom = total_plate_prob.max(1e-12).ln();
            for k in 0..PLATE_CHARSET.len() {
                let pk = plate_step_probs[k];
                plate_log_probs[[t, k]] = if pk > 1e-12 {
                    pk.ln() - log_denom
                } else {
                    -27.631021
                };
            }
        }

        Ok(plate_log_probs)
    }
}
