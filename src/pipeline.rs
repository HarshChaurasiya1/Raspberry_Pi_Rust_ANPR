//! End-to-end ANPR Pipeline: Detect -> Safe Crop -> Enhance Variants -> Recognizer Lattice -> CTC GCD Search -> Merge.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use anyhow::Result;
use image::RgbImage;
use ndarray::{s, Array2};

use crate::config::Config;
use crate::detector::{create_detector, Detector};
use crate::grammar::{GrammarConfig, PlateDFA};
use crate::image::{apply_clahe, apply_gamma, is_two_line, split_two_line, upscale_base};
use crate::ocr::{
    ctc_gcd_beam_search, ctc_greedy, merge_variant_nbest,
    PlateRecognizer, Hypothesis,
};
use crate::tracker::PlateTracker;

pub type SharedOogCache = Arc<RwLock<HashMap<(u32, u32, u32, u32), (String, usize)>>>;

#[derive(Debug, Clone)]
pub struct PlateResult {
    pub text: String,
    pub score: f32,
    pub detection_score: f32,
    pub box_tuple: (u32, u32, u32, u32),
    pub greedy_text: String,
    pub valid: bool,
    pub two_line: bool,
    pub timings_ms: HashMap<String, f64>,
    pub track_id: Option<usize>,
}

pub struct ANPRPipeline {
    pub cfg: Config,
    pub detector: Box<dyn Detector>,
    pub recognizer: PlateRecognizer,
    pub dfa: PlateDFA,
    static_oog_cache: SharedOogCache,
    pub tracker: PlateTracker,
}

impl ANPRPipeline {
    pub fn new(config: Option<Config>) -> Result<Self> {
        Self::new_with_cache(config, Arc::new(RwLock::new(HashMap::new())))
    }

    pub fn new_with_cache(config: Option<Config>, static_oog_cache: SharedOogCache) -> Result<Self> {
        let cfg = config.unwrap_or_default();
        let detector = create_detector(&cfg.detector)?;
        let recognizer = PlateRecognizer::new(
            &cfg.recognizer.model_path,
            &cfg.recognizer.charset_path,
            cfg.recognizer.input_height,
            cfg.recognizer.input_width,
            cfg.recognizer.threads,
            cfg.recognizer.allow_spinning,
            cfg.recognizer.grayscale,
        )?;
        let dfa = PlateDFA::new(Some(GrammarConfig {
            strict_modern: cfg.decoding.strict_grammar,
            ..Default::default()
        }));

        Ok(Self {
            cfg,
            detector,
            recognizer,
            dfa,
            static_oog_cache,
            tracker: PlateTracker::default(),
        })
    }

    fn eval_single_variant(&mut self, variant: &RgbImage, two_line: bool, nbest_lists: &mut Vec<Vec<Hypothesis>>) -> Result<String> {
        let lattice = self.recognizer.lattice(variant)?;
        let greedy = ctc_greedy(&lattice, 0);
        let hyps = ctc_gcd_beam_search(
            &lattice,
            0,
            Some(&self.dfa),
            self.cfg.decoding.lambda_grammar,
            self.cfg.decoding.beam_width,
            self.cfg.decoding.nbest,
        );
        let has_valid = hyps.iter().any(|h| self.dfa.accepts(&h.text));
        nbest_lists.push(hyps);

        // Only run expensive 2-line split if crop looks like a real multi-character plate
        // and single-line beam search did not already find a valid grammar hypothesis
        if two_line && greedy.len() >= 4 && !has_valid {
            let (top, bottom) = split_two_line(variant);
            let top_lattice = self.recognizer.lattice(&top)?;
            let bot_lattice = self.recognizer.lattice(&bottom)?;

            let (t_top, c_top) = (top_lattice.shape()[0], top_lattice.shape()[1]);
            let (t_bot, _) = (bot_lattice.shape()[0], bot_lattice.shape()[1]);
            let mut joint = Array2::<f32>::zeros((t_top + t_bot, c_top));
            joint.slice_mut(s![0..t_top, ..]).assign(&top_lattice);
            joint.slice_mut(s![t_top..t_top + t_bot, ..]).assign(&bot_lattice);

            let hyps_joint = ctc_gcd_beam_search(
                &joint,
                0,
                Some(&self.dfa),
                self.cfg.decoding.lambda_grammar,
                self.cfg.decoding.beam_width,
                self.cfg.decoding.nbest,
            );
            nbest_lists.push(hyps_joint);
        }

        Ok(greedy)
    }

    pub fn decode_crop(&mut self, crop: &RgbImage) -> Result<PlateResult> {
        self.decode_crop_ext(crop, is_two_line(crop))
    }

    pub fn decode_crop_ext(&mut self, crop: &RgbImage, two_line: bool) -> Result<PlateResult> {
        let mut nbest_lists: Vec<Vec<Hypothesis>> = Vec::new();

        // Step 1: Run Variant 0 (Raw upscaled crop) first
        let base = upscale_base(crop);
        let greedy_text = self.eval_single_variant(&base, two_line, &mut nbest_lists)?;

        // Fast Early Exit 1: Short non-plate background sign (e.g. "SICI", "SCE", "B")
        let has_valid_v0 = nbest_lists
            .iter()
            .flatten()
            .any(|h| self.dfa.accepts(&h.text));

        if !has_valid_v0 && greedy_text.len() <= 5 {
            return Ok(PlateResult {
                text: greedy_text.clone(),
                score: f32::NEG_INFINITY,
                detection_score: 0.0,
                box_tuple: (0, 0, 0, 0),
                greedy_text,
                valid: false,
                two_line,
                timings_ms: HashMap::new(),
                track_id: None,
            });
        }

        // Fast Early Exit 2: Variant 0 already found a clean 9-10 char valid plate
        if has_valid_v0 && greedy_text.len() >= 9 {
            let merged = merge_variant_nbest(&nbest_lists, Some(&self.dfa), self.cfg.decoding.lambda_grammar);
            if let Some(best) = merged.first() {
                if best.text.len() >= 9 && self.dfa.accepts(&best.text) && (self.dfa.accepts(&greedy_text) || best.score > -2.5) {
                    return Ok(PlateResult {
                        text: best.text.clone(),
                        score: best.score,
                        detection_score: 0.0,
                        box_tuple: (0, 0, 0, 0),
                        greedy_text,
                        valid: true,
                        two_line,
                        timings_ms: HashMap::new(),
                        track_id: None,
                    });
                }
            }
        }

        // Step 2: Only generate & run CLAHE and Gamma variants when Variant 0 was ambiguous
        let num_variants = self.cfg.decoding.num_variants;
        if num_variants >= 2 {
            let clahe = apply_clahe(&base);
            let _ = self.eval_single_variant(&clahe, two_line, &mut nbest_lists)?;
        }
        if num_variants >= 3 {
            let gamma = apply_gamma(&base);
            let _ = self.eval_single_variant(&gamma, two_line, &mut nbest_lists)?;
        }

        let merged = merge_variant_nbest(&nbest_lists, Some(&self.dfa), self.cfg.decoding.lambda_grammar);
        if let Some(best) = merged.first() {
            return Ok(PlateResult {
                text: best.text.clone(),
                score: best.score,
                detection_score: 0.0,
                box_tuple: (0, 0, 0, 0),
                greedy_text,
                valid: self.dfa.accepts(&best.text),
                two_line,
                timings_ms: HashMap::new(),
                track_id: None,
            });
        }

        let is_valid = self.dfa.accepts(&greedy_text);
        Ok(PlateResult {
            text: greedy_text.clone(),
            score: f32::NEG_INFINITY,
            detection_score: 0.0,
            box_tuple: (0, 0, 0, 0),
            greedy_text,
            valid: is_valid,
            two_line,
            timings_ms: HashMap::new(),
            track_id: None,
        })
    }

    pub fn process_tracked(&mut self, image: &RgbImage) -> Result<Vec<PlateResult>> {
        let t0 = Instant::now();
        let detections = self.detector.detect(image)?;
        let t_det = t0.elapsed().as_secs_f64() * 1000.0;

        let max_plates = self.cfg.detector.max_plates.min(detections.len());
        let active_dets = &detections[..max_plates];

        let det_boxes: Vec<(u32, u32, u32, u32)> = active_dets.iter().map(|d| d.box_tuple()).collect();
        let matches = self.tracker.match_detections(&det_boxes);
        let mut results = Vec::with_capacity(active_dets.len());
        let mut matched_track_indices = Vec::new();

        for (d_idx, maybe_t_idx) in matches {
            let det = &active_dets[d_idx];
            let box_tuple = det_boxes[d_idx];

            if let Some(t_idx) = maybe_t_idx {
                matched_track_indices.push(t_idx);
                {
                    let track = &mut self.tracker.tracks[t_idx];
                    let prev_score = track.detection_score;
                    track.bbox = box_tuple;
                    track.detection_score = det.score;

                    if track.confirmed {
                        results.push(crate::tracker::PlateTracker::build_cached_result(track, det.score, t_det));
                        continue;
                    }
                    if track.is_noise && det.score < prev_score + 0.20 {
                        let mut r = crate::tracker::PlateTracker::build_cached_result(track, det.score, t_det);
                        r.valid = false;
                        results.push(r);
                        continue;
                    }
                }

                let crop = det.crop(image, self.cfg.detector.crop_padding);
                if crop.width() == 0 || crop.height() == 0 {
                    continue;
                }

                let is_edge = box_tuple.0 <= 8 || box_tuple.2 >= image.width().saturating_sub(8);
                let two_line = !is_edge && crate::image::is_two_line(&crop);

                let t1 = Instant::now();
                let mut res = self.decode_crop_ext(&crop, two_line)?;
                let t_rec = t1.elapsed().as_secs_f64() * 1000.0;

                res.detection_score = det.score;
                res.box_tuple = box_tuple;
                res.timings_ms.insert("detect".to_string(), t_det);
                res.timings_ms.insert("read".to_string(), t_rec);
                res.track_id = Some(self.tracker.tracks[t_idx].id);

                self.tracker.update_track_ocr(t_idx, &res);
                results.push(res);
            } else {
                let crop = det.crop(image, self.cfg.detector.crop_padding);
                if crop.width() == 0 || crop.height() == 0 {
                    continue;
                }

                let is_edge = box_tuple.0 <= 8 || box_tuple.2 >= image.width().saturating_sub(8);
                let two_line = !is_edge && crate::image::is_two_line(&crop);

                let t1 = Instant::now();
                let mut res = self.decode_crop_ext(&crop, two_line)?;
                let t_rec = t1.elapsed().as_secs_f64() * 1000.0;

                res.detection_score = det.score;
                res.box_tuple = box_tuple;
                res.timings_ms.insert("detect".to_string(), t_det);
                res.timings_ms.insert("read".to_string(), t_rec);

                let t_idx = self.tracker.create_track(box_tuple, det.score, Some(&res));
                matched_track_indices.push(t_idx);
                res.track_id = Some(self.tracker.tracks[t_idx].id);
                results.push(res);
            }
        }

        self.tracker.step_age(&matched_track_indices);
        Ok(results)
    }

    pub fn process(&mut self, image: &RgbImage) -> Result<Vec<PlateResult>> {
        if self.cfg.runtime.tracking {
            return self.process_tracked(image);
        }

        let t0 = Instant::now();
        let detections = self.detector.detect(image)?;
        let t_det = t0.elapsed().as_secs_f64() * 1000.0;

        let mut results = Vec::new();
        let max_plates = self.cfg.detector.max_plates.min(detections.len());

        for det in &detections[..max_plates] {
            let (x1, y1, x2, y2) = det.box_tuple();
            // Quantize bounding box coordinates to 32px grid to detect stationary background signs (e.g. "SICI")
            let grid_key = (x1 / 32, y1 / 32, x2 / 32, y2 / 32);
            let cached_hit = self
                .static_oog_cache
                .read()
                .ok()
                .and_then(|map| map.get(&grid_key).cloned());
            if let Some((cached_text, hits)) = cached_hit {
                if hits >= 2 {
                    let mut timings = HashMap::new();
                    timings.insert("detect".to_string(), t_det);
                    timings.insert("read".to_string(), 0.0);
                    results.push(PlateResult {
                        text: cached_text.clone(),
                        score: f32::NEG_INFINITY,
                        detection_score: det.score,
                        box_tuple: (x1, y1, x2, y2),
                        greedy_text: cached_text,
                        valid: false,
                        two_line: false,
                        timings_ms: timings,
                        track_id: None,
                    });
                    continue;
                }
            }

            let crop = det.crop(image, self.cfg.detector.crop_padding);
            if crop.width() == 0 || crop.height() == 0 {
                continue;
            }

            let t1 = Instant::now();
            let mut res = self.decode_crop(&crop)?;
            let t_rec = t1.elapsed().as_secs_f64() * 1000.0;

            if !res.valid && res.text.len() <= 6 {
                if let Ok(mut map) = self.static_oog_cache.write() {
                    let entry = map.entry(grid_key).or_insert_with(|| (res.text.clone(), 0));
                    entry.1 += 1;
                }
            } else if res.valid {
                if let Ok(mut map) = self.static_oog_cache.write() {
                    map.remove(&grid_key);
                }
            }

            res.detection_score = det.score;
            res.box_tuple = (x1, y1, x2, y2);
            res.timings_ms.insert("detect".to_string(), t_det);
            res.timings_ms.insert("read".to_string(), t_rec);
            res.track_id = None;
            results.push(res);
        }

        Ok(results)
    }
}
