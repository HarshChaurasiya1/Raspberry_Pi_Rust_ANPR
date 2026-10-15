//! Weighted deterministic finite automaton of the Indian license-plate registration grammar.

use std::collections::{HashMap, HashSet};
use crate::grammar::rto_data::{get_rto_max_district, is_valid_state_code, get_valid_first_letters};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum DfaState {
    S0,
    S1(char),
    Code(String),
    Dist(String, u32, usize), // (code, value, ndigits)
    Ser(usize),               // (k_letters)
    Num(usize, bool),         // (k_digits, non_zero)
    B1,
    B2,
    B3,
    B4,
    Bn(usize, bool),          // (k_digits, non_zero)
    Bl(usize),                // (k_letters)
}

#[derive(Debug, Clone)]
pub struct GrammarConfig {
    pub w_bh: f32,
    pub w_rto_invalid: f32,
    pub w_serlen: HashMap<usize, f32>,
    pub w_numlen: HashMap<usize, f32>,
    pub strict_modern: bool,
}

impl Default for GrammarConfig {
    fn default() -> Self {
        let mut w_serlen = HashMap::new();
        w_serlen.insert(1, -0.8);
        w_serlen.insert(2, 0.0);
        w_serlen.insert(3, -0.6);

        let mut w_numlen = HashMap::new();
        w_numlen.insert(1, -1.5);
        w_numlen.insert(2, -1.0);
        w_numlen.insert(3, -0.5);
        w_numlen.insert(4, 0.0);

        Self {
            w_bh: -4.0,
            w_rto_invalid: -2.5,
            w_serlen,
            w_numlen,
            strict_modern: false,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PlateDFA {
    pub cfg: GrammarConfig,
    first_letters: HashSet<char>,
}

impl PlateDFA {
    pub fn new(config: Option<GrammarConfig>) -> Self {
        Self {
            cfg: config.unwrap_or_default(),
            first_letters: get_valid_first_letters(),
        }
    }

    pub fn start(&self) -> DfaState {
        DfaState::S0
    }

    fn w_rto(&self, code: &str, value: u32) -> f32 {
        let max_d = get_rto_max_district(code);
        if value >= 1 && value <= max_d {
            0.0
        } else {
            self.cfg.w_rto_invalid
        }
    }

    pub fn step(&self, state: &DfaState, ch: char) -> Option<(DfaState, f32)> {
        let is_letter = ch.is_ascii_uppercase();
        let is_digit = ch.is_ascii_digit();
        if !is_letter && !is_digit {
            return None;
        }

        match state {
            DfaState::S0 => {
                if is_letter && self.first_letters.contains(&ch) {
                    Some((DfaState::S1(ch), 0.0))
                } else if is_digit && !self.cfg.strict_modern {
                    Some((DfaState::B1, self.cfg.w_bh))
                } else {
                    None
                }
            }
            DfaState::S1(c) => {
                if is_letter {
                    let mut code = String::with_capacity(2);
                    code.push(*c);
                    code.push(ch);
                    if is_valid_state_code(&code) {
                        Some((DfaState::Code(code), 0.0))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            DfaState::Code(code) => {
                if is_digit {
                    let val = ch.to_digit(10).unwrap();
                    Some((DfaState::Dist(code.clone(), val, 1), 0.0))
                } else {
                    None
                }
            }
            DfaState::Dist(code, value, ndig) => {
                if is_digit && *ndig == 1 {
                    let v2 = value * 10 + ch.to_digit(10).unwrap();
                    if v2 >= 1 {
                        Some((DfaState::Dist(code.clone(), v2, 2), 0.0))
                    } else {
                        None
                    }
                } else if is_letter && *value >= 1 {
                    let w = self.w_rto(code, *value);
                    Some((DfaState::Ser(1), w))
                } else {
                    None
                }
            }
            DfaState::Ser(k) => {
                if is_letter && *k < 3 {
                    Some((DfaState::Ser(k + 1), 0.0))
                } else if is_digit {
                    let nz = ch != '0';
                    let w = self.cfg.w_serlen.get(k).copied().unwrap_or(0.0);
                    Some((DfaState::Num(1, nz), w))
                } else {
                    None
                }
            }
            DfaState::Num(k, nz) => {
                if is_digit && *k < 4 {
                    Some((DfaState::Num(k + 1, *nz || ch != '0'), 0.0))
                } else {
                    None
                }
            }
            // Bharat (BH) Series
            DfaState::B1 => {
                if is_digit {
                    Some((DfaState::B2, 0.0))
                } else {
                    None
                }
            }
            DfaState::B2 => {
                if ch == 'B' {
                    Some((DfaState::B3, 0.0))
                } else {
                    None
                }
            }
            DfaState::B3 => {
                if ch == 'H' {
                    Some((DfaState::B4, 0.0))
                } else {
                    None
                }
            }
            DfaState::B4 => {
                if is_digit {
                    Some((DfaState::Bn(1, ch != '0'), 0.0))
                } else {
                    None
                }
            }
            DfaState::Bn(k, nz) => {
                if is_digit && *k < 4 {
                    Some((DfaState::Bn(k + 1, *nz || ch != '0'), 0.0))
                } else if is_letter && *k == 4 && *nz {
                    Some((DfaState::Bl(1), 0.0))
                } else {
                    None
                }
            }
            DfaState::Bl(k) => {
                if is_letter && *k == 1 {
                    Some((DfaState::Bl(2), 0.0))
                } else {
                    None
                }
            }
        }
    }

    pub fn allowed(&self, state: &DfaState) -> HashMap<char, (DfaState, f32)> {
        let mut out = HashMap::with_capacity(36);
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".chars() {
            if let Some(nxt) = self.step(state, ch) {
                out.insert(ch, nxt);
            }
        }
        out
    }

    pub fn final_weight(&self, state: &DfaState) -> Option<f32> {
        match state {
            DfaState::Num(k, nz) => {
                if self.cfg.strict_modern && *k != 4 {
                    None
                } else if *nz {
                    self.cfg.w_numlen.get(k).copied()
                } else {
                    None
                }
            }
            DfaState::Bl(_) => {
                if self.cfg.strict_modern {
                    None
                } else {
                    Some(0.0)
                }
            }
            _ => None,
        }
    }

    pub fn score(&self, text: &str) -> (bool, f32) {
        let mut state = self.start();
        let mut total = 0.0f32;
        let clean: String = text.chars().filter(|c| !c.is_whitespace()).collect();
        for ch in clean.to_ascii_uppercase().chars() {
            match self.step(&state, ch) {
                Some((nxt_st, w)) => {
                    state = nxt_st;
                    total += w;
                }
                None => return (false, f32::NEG_INFINITY),
            }
        }
        match self.final_weight(&state) {
            Some(fw) => (true, total + fw),
            None => (false, f32::NEG_INFINITY),
        }
    }

    pub fn accepts(&self, text: &str) -> bool {
        self.score(text).0
    }
}
