//! High-performance image operations: letterboxing, safe crops, CLAHE, gamma correction, and 2-line splitting.

use image::{RgbImage, Rgb, imageops::{resize, FilterType}};
use std::cmp::{max, min};

pub const TWO_LINE_ASPECT: f32 = 0.70;

#[derive(Debug, Clone, Copy)]
pub struct LetterboxInfo {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub target_w: u32,
    pub target_h: u32,
}

/// Letterbox resize preserving aspect ratio, padding with color 114 (YOLO format).
pub fn letterbox(img: &RgbImage, target_size: u32) -> (RgbImage, LetterboxInfo) {
    let (w, h) = (img.width(), img.height());
    let scale = (target_size as f32 / h as f32).min(target_size as f32 / w as f32);
    let nw = (w as f32 * scale).round() as u32;
    let nh = (h as f32 * scale).round() as u32;

    let resized = resize(img, nw, nh, FilterType::Nearest);
    let mut canvas = RgbImage::from_pixel(target_size, target_size, Rgb([114, 114, 114]));

    let top = (target_size - nh) / 2;
    let left = (target_size - nw) / 2;

    image::imageops::overlay(&mut canvas, &resized, left as i64, top as i64);

    let info = LetterboxInfo {
        scale,
        pad_x: left as f32,
        pad_y: top as f32,
        target_w: target_size,
        target_h: target_size,
    };

    (canvas, info)
}

/// Fused zero-allocation letterbox + CHW float32 normalization directly into a contiguous buffer.
pub fn letterbox_chw(img: &RgbImage, target_size: u32) -> (Vec<f32>, LetterboxInfo) {
    letterbox_chw_rect(img, target_size, target_size)
}

/// Fused zero-allocation rectangular letterbox + CHW float32 normalization directly into a contiguous buffer.
pub fn letterbox_chw_rect(img: &RgbImage, target_w: u32, target_h: u32) -> (Vec<f32>, LetterboxInfo) {
    let (w, h) = (img.width(), img.height());
    let scale = (target_h as f32 / h as f32).min(target_w as f32 / w as f32);
    let nw = ((w as f32 * scale).round() as u32).clamp(1, target_w);
    let nh = ((h as f32 * scale).round() as u32).clamp(1, target_h);

    let top = (target_h - nh) / 2;
    let left = (target_w - nw) / 2;
    let tw = target_w as usize;
    let th = target_h as usize;
    let plane = tw * th;
    let pad_val = 114.0f32 / 255.0;
    let mut vec = vec![pad_val; 3 * plane];

    let raw = img.as_raw();
    let stride = (w as usize) * 3;
    let inv_255 = 1.0f32 / 255.0;

    for dy in 0..nh {
        let sy = ((dy * h) / nh).min(h - 1) as usize;
        let row_offset = sy * stride;
        let out_y = (top + dy) as usize * tw;
        for dx in 0..nw {
            let sx = ((dx * w) / nw).min(w - 1) as usize;
            let px = row_offset + sx * 3;
            let out_idx = out_y + (left + dx) as usize;
            unsafe {
                vec[out_idx] = (*raw.get_unchecked(px) as f32) * inv_255;
                vec[plane + out_idx] = (*raw.get_unchecked(px + 1) as f32) * inv_255;
                vec[2 * plane + out_idx] = (*raw.get_unchecked(px + 2) as f32) * inv_255;
            }
        }
    }

    let info = LetterboxInfo {
        scale,
        pad_x: left as f32,
        pad_y: top as f32,
        target_w,
        target_h,
    };

    (vec, info)
}

/// Map letterboxed bounding box back to original image coordinates.
pub fn unletterbox_box(
    x1: f32, y1: f32, x2: f32, y2: f32,
    info: &LetterboxInfo,
    orig_w: u32, orig_h: u32,
) -> (u32, u32, u32, u32) {
    let bx1 = ((x1 - info.pad_x) / info.scale).round() as i32;
    let by1 = ((y1 - info.pad_y) / info.scale).round() as i32;
    let bx2 = ((x2 - info.pad_x) / info.scale).round() as i32;
    let by2 = ((y2 - info.pad_y) / info.scale).round() as i32;

    let rx1 = bx1.max(0).min(orig_w as i32) as u32;
    let ry1 = by1.max(0).min(orig_h as i32) as u32;
    let rx2 = bx2.max(0).min(orig_w as i32) as u32;
    let ry2 = by2.max(0).min(orig_h as i32) as u32;

    (rx1, ry1, rx2, ry2)
}

/// Safely crop a rectangular region from image with boundary clamping and padding.
pub fn safe_crop(img: &RgbImage, x1: u32, y1: u32, x2: u32, y2: u32, pad: u32) -> RgbImage {
    let w = img.width();
    let h = img.height();

    let cx1 = x1.saturating_sub(pad);
    let cy1 = y1.saturating_sub(pad);
    let cx2 = (x2 + pad).min(w);
    let cy2 = (y2 + pad).min(h);

    if cx2 <= cx1 || cy2 <= cy1 {
        return RgbImage::new(1, 1);
    }

    let cw = cx2 - cx1;
    let ch = cy2 - cy1;

    let sub = image::imageops::crop_imm(img, cx1, cy1, cw, ch);
    sub.to_image()
}

/// Fast CLAHE-like adaptive contrast enhancement on RGB image.
pub fn apply_clahe(img: &RgbImage) -> RgbImage {
    let (w, h) = (img.width(), img.height());
    if w == 0 || h == 0 {
        return img.clone();
    }

    let mut out = img.clone();
    let grid_x = 8;
    let grid_y = 8;
    let tile_w = max(1, w / grid_x);
    let tile_h = max(1, h / grid_y);
    let clip_limit = 3.0f32;

    for ty in 0..grid_y {
        for tx in 0..grid_x {
            let start_x = tx * tile_w;
            let start_y = ty * tile_h;
            let end_x = if tx == grid_x - 1 { w } else { (tx + 1) * tile_w };
            let end_y = if ty == grid_y - 1 { h } else { (ty + 1) * tile_h };
            let tile_pixels = ((end_x - start_x) * (end_y - start_y)) as f32;

            if tile_pixels < 1.0 {
                continue;
            }

            let mut hist = [0u32; 256];
            for y in start_y..end_y {
                for x in start_x..end_x {
                    let p = img.get_pixel(x, y);
                    let lum = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as usize;
                    hist[lum.min(255)] += 1;
                }
            }

            let clip = (clip_limit * tile_pixels / 256.0).round() as u32;
            let mut clipped_excess = 0u32;
            for count in hist.iter_mut() {
                if *count > clip {
                    clipped_excess += *count - clip;
                    *count = clip;
                }
            }
            let redist = clipped_excess / 256;
            for count in hist.iter_mut() {
                *count += redist;
            }

            let mut cdf = [0u32; 256];
            let mut sum = 0u32;
            for i in 0..256 {
                sum += hist[i];
                cdf[i] = sum;
            }

            let cdf_min = cdf.iter().copied().find(|&v| v > 0).unwrap_or(0) as f32;
            let denom = (tile_pixels - cdf_min).max(1.0);

            for y in start_y..end_y {
                for x in start_x..end_x {
                    let p = img.get_pixel(x, y);
                    let old_lum = (0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32) as usize;
                    let new_lum = (((cdf[old_lum.min(255)] as f32 - cdf_min) / denom) * 255.0).clamp(0.0, 255.0);
                    let factor = if old_lum > 0 { new_lum / old_lum as f32 } else { 1.0 };

                    let r = (p[0] as f32 * factor).clamp(0.0, 255.0) as u8;
                    let g = (p[1] as f32 * factor).clamp(0.0, 255.0) as u8;
                    let b = (p[2] as f32 * factor).clamp(0.0, 255.0) as u8;
                    out.put_pixel(x, y, Rgb([r, g, b]));
                }
            }
        }
    }
    out
}

/// Gamma LUT (inv_gamma = 1.0 / 0.4 = 2.5) for night / low-light brightening.
pub fn apply_gamma(img: &RgbImage) -> RgbImage {
    let mut table = [0u8; 256];
    let inv_gamma = 1.0 / 0.4f64;
    for i in 0..256 {
        let norm = (i as f64) / 255.0;
        let val = norm.powf(inv_gamma) * 255.0;
        table[i] = val.clamp(0.0, 255.0) as u8;
    }

    let mut out = img.clone();
    for pixel in out.pixels_mut() {
        pixel[0] = table[pixel[0] as usize];
        pixel[1] = table[pixel[1] as usize];
        pixel[2] = table[pixel[2] as usize];
    }
    out
}

/// Upscale crop height to at least 48px if smaller.
pub fn upscale_base(crop: &RgbImage) -> RgbImage {
    let h = crop.height();
    if h < 48 {
        let scale = 48.0 / (h.max(1) as f32);
        let nw = ((crop.width() as f32) * scale).round() as u32;
        resize(crop, nw, 48, FilterType::Triangle)
    } else {
        crop.clone()
    }
}

/// Generate enhancement variants: Raw (upscaled >=48px), CLAHE, and Gamma LUT.
pub fn enhance_variants(crop: &RgbImage) -> Vec<RgbImage> {
    let base = upscale_base(crop);
    let clahe = apply_clahe(&base);
    let gamma = apply_gamma(&base);
    vec![base, clahe, gamma]
}

/// Check if plate crop is a 2-line motorcycle plate (aspect h/w > 0.7).
pub fn is_two_line(crop: &RgbImage) -> bool {
    let (w, h) = (crop.width(), crop.height());
    w > 0 && (h as f32 / w as f32) > TWO_LINE_ASPECT
}

/// Split stacked 2-line plate into overlapping top and bottom strips.
pub fn split_two_line(crop: &RgbImage) -> (RgbImage, RgbImage) {
    let (w, h) = (crop.width(), crop.height());
    let mid = h / 2;
    let overlap = max((0.25 * h as f32) as u32, 8);

    let top_h = min(h, mid + overlap);
    let bot_start = mid.saturating_sub(overlap);
    let bot_h = h - bot_start;

    let top = image::imageops::crop_imm(crop, 0, 0, w, top_h).to_image();
    let bottom = image::imageops::crop_imm(crop, 0, bot_start, w, bot_h).to_image();

    (top, bottom)
}
