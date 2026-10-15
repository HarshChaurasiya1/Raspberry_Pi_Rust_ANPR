//! Tests for CTC Decoder and hypothesis fusion.

use ndarray::Array2;
use raspberry_pi_rust::ocr::{ctc_greedy, PLATE_CHARSET};

#[test]
fn test_ctc_greedy_decoding() {
    // 3 timesteps, 37 classes
    let mut lp = Array2::<f32>::from_elem((3, PLATE_CHARSET.len()), -10.0);
    // Timestep 0 -> 'H' (index 17)
    let h_idx = PLATE_CHARSET.iter().position(|&c| c == 'H').unwrap();
    lp[[0, h_idx]] = 0.0;

    // Timestep 1 -> blank (index 0)
    lp[[1, 0]] = 0.0;

    // Timestep 2 -> 'R' (index 27)
    let r_idx = PLATE_CHARSET.iter().position(|&c| c == 'R').unwrap();
    lp[[2, r_idx]] = 0.0;

    let text = ctc_greedy(&lp, 0);
    assert_eq!(text, "HR");
}
