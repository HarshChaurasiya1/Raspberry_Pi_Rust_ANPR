//! Tests for image cropping, boundary clamping, and 2-line detection.

use image::{RgbImage, Rgb};
use raspberry_pi_rust::image::{safe_crop, is_two_line, split_two_line};

#[test]
fn test_safe_crop_clamping() {
    let img = RgbImage::from_pixel(100, 100, Rgb([255, 0, 0]));
    
    // Crop within bounds
    let c1 = safe_crop(&img, 10, 10, 50, 50, 4);
    assert_eq!(c1.width(), 48); // (50+4) - (10-4) = 54 - 6 = 48
    assert_eq!(c1.height(), 48);

    // Crop extending outside bounds
    let c2 = safe_crop(&img, 0, 0, 20, 20, 10);
    assert_eq!(c2.width(), 30); // (20+10).min(100) - 0 = 30
}

#[test]
fn test_two_line_aspect_detection() {
    let wide_plate = RgbImage::new(200, 50); // Aspect h/w = 0.25 (1-line car plate)
    assert!(!is_two_line(&wide_plate));

    let tall_plate = RgbImage::new(100, 80); // Aspect h/w = 0.80 (2-line bike plate)
    assert!(is_two_line(&tall_plate));
}

#[test]
fn test_two_line_split() {
    let tall_plate = RgbImage::new(100, 80);
    let (top, bot) = split_two_line(&tall_plate);
    assert!(top.height() > 40);
    assert!(bot.height() > 40);
}
