//! Grammar-constrained CTC prefix beam search, greedy decoding, and N-best fusion.

use std::collections::HashMap;
use ndarray::Array2;
use crate::grammar::{PlateDFA, DfaState};
use crate::ocr::recognizer::PLATE_CHARSET;

pub const NEG_INF: f32 = -1e30;

#[inline(always)]
pub fn logaddexp(a: f32, b: f32) -> f32 {
    if a <= NEG_INF { return b; }
    if b <= NEG_INF { return a; }
    let (max_val, min_val) = if a > b { (a, b) } else { (b, a) };
    max_val + (min_val - max_val).exp().ln_1p()
}

#[derive(Debug, Clone)]
pub struct Hypothesis {
    pub text: String,
    pub score: f32,
    pub am_score: f32,
    pub g_score: f32,
}

#[derive(Debug, Clone)]
struct BeamEntry {
    p_b: f32,
    p_nb: f32,
    g: f32,
    state: Option<DfaState>,
}

impl BeamEntry {
    fn new(p_b: f32, p_nb: f32, g: f32, state: Option<DfaState>) -> Self {
        Self { p_b, p_nb, g, state }
    }

    #[inline(always)]
    fn total_am(&self) -> f32 {
        logaddexp(self.p_b, self.p_nb)
    }
}

pub fn ctc_gcd_beam_search(
    log_probs: &Array2<f32>,
    blank: usize,
    dfa: Option<&PlateDFA>,
    lambda_g: f32,
    beam_width: usize,
    nbest: usize,
) -> Vec<Hypothesis> {
    let t_steps = log_probs.shape()[0];
    let num_chars = log_probs.shape()[1];

    let start_state = dfa.map(|d| d.start());
    let mut beams: HashMap<String, BeamEntry> = HashMap::new();
    beams.insert(String::new(), BeamEntry::new(0.0, NEG_INF, 0.0, start_state));

    for t in 0..t_steps {
        let mut nxt: HashMap<String, BeamEntry> = HashMap::new();

        for (prefix, b) in beams.iter() {
            let total = b.total_am();
            if total <= NEG_INF {
                continue;
            }

            // 1. Extend with blank: prefix unchanged
            let lp_blank = log_probs[[t, blank]];
            let entry = nxt.entry(prefix.clone()).or_insert_with(|| BeamEntry::new(NEG_INF, NEG_INF, b.g, b.state.clone()));
            entry.p_b = logaddexp(entry.p_b, total + lp_blank);

            // 2. Repeat last character: prefix unchanged
            if let Some(last_ch) = prefix.chars().last() {
                if let Some(last_idx) = PLATE_CHARSET.iter().position(|&c| c == last_ch) {
                    let lp_repeat = log_probs[[t, last_idx]];
                    let entry = nxt.entry(prefix.clone()).or_insert_with(|| BeamEntry::new(NEG_INF, NEG_INF, b.g, b.state.clone()));
                    entry.p_nb = logaddexp(entry.p_nb, b.p_nb + lp_repeat);
                }
            }

            // 3. Extend with a new character
            if let Some(d) = dfa {
                if let Some(ref st) = b.state {
                    for (&ch, (nxt_st, w)) in d.allowed(st).iter() {
                        if let Some(idx) = PLATE_CHARSET.iter().position(|&c| c == ch) {
                            let mut new_prefix = prefix.clone();
                            new_prefix.push(ch);

                            let lp_ch = log_probs[[t, idx]];
                            let contrib = if prefix.ends_with(ch) {
                                b.p_b + lp_ch
                            } else {
                                total + lp_ch
                            };

                            if contrib > NEG_INF {
                                let entry = nxt.entry(new_prefix).or_insert_with(|| {
                                    BeamEntry::new(NEG_INF, NEG_INF, b.g + w, Some(nxt_st.clone()))
                                });
                                entry.p_nb = logaddexp(entry.p_nb, contrib);
                            }
                        }
                    }
                }
            } else {
                for idx in 0..num_chars {
                    if idx == blank { continue; }
                    let ch = PLATE_CHARSET[idx];
                    let mut new_prefix = prefix.clone();
                    new_prefix.push(ch);

                    let lp_ch = log_probs[[t, idx]];
                    let contrib = if prefix.ends_with(ch) {
                        b.p_b + lp_ch
                    } else {
                        total + lp_ch
                    };

                    if contrib > NEG_INF {
                        let entry = nxt.entry(new_prefix).or_insert_with(|| {
                            BeamEntry::new(NEG_INF, NEG_INF, 0.0, None)
                        });
                        entry.p_nb = logaddexp(entry.p_nb, contrib);
                    }
                }
            }
        }

        // Prune to beam_width
        let mut entries: Vec<(String, BeamEntry)> = nxt.into_iter().collect();
        entries.sort_by(|a, b| {
            let score_a = a.1.total_am() + lambda_g * a.1.g;
            let score_b = b.1.total_am() + lambda_g * b.1.g;
            score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
        });

        beams = entries.into_iter().take(beam_width).collect();
    }

    // Finalize
    let mut results = Vec::new();
    for (prefix, b) in beams {
        if prefix.is_empty() { continue; }
        let am = b.total_am();
        if am <= NEG_INF { continue; }

        let g_total = if let Some(d) = dfa {
            if let Some(ref st) = b.state {
                match d.final_weight(st) {
                    Some(fw) => b.g + fw,
                    None => continue, // Reject non-accepting states
                }
            } else {
                continue;
            }
        } else {
            0.0
        };

        results.push(Hypothesis {
            text: prefix,
            score: am + lambda_g * g_total,
            am_score: am,
            g_score: g_total,
        });
    }

    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results.into_iter().take(nbest).collect()
}

pub fn ctc_greedy(log_probs: &Array2<f32>, blank: usize) -> String {
    let t_steps = log_probs.shape()[0];
    let mut out = String::new();
    let mut prev = usize::MAX;

    for t in 0..t_steps {
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for c in 0..PLATE_CHARSET.len() {
            let v = log_probs[[t, c]];
            if v > best_val {
                best_val = v;
                best_idx = c;
            }
        }

        if best_idx != prev && best_idx != blank {
            out.push(PLATE_CHARSET[best_idx]);
        }
        prev = best_idx;
    }

    out
}

pub fn merge_variant_nbest(
    nbest_lists: &[Vec<Hypothesis>],
    dfa: Option<&PlateDFA>,
    lambda_g: f32,
) -> Vec<Hypothesis> {
    let mut pool: HashMap<String, (f32, f32)> = HashMap::new(); // text -> (am_score, g_score)

    for lst in nbest_lists {
        for h in lst {
            if let Some(entry) = pool.get_mut(&h.text) {
                entry.0 = logaddexp(entry.0, h.am_score);
            } else {
                let g = if let Some(d) = dfa {
                    let (ok, score) = d.score(&h.text);
                    if !ok { continue; }
                    score
                } else {
                    h.g_score
                };
                pool.insert(h.text.clone(), (h.am_score, g));
            }
        }
    }

    let mut merged: Vec<Hypothesis> = pool.into_iter().map(|(text, (am, g))| {
        Hypothesis {
            text,
            score: am + lambda_g * g,
            am_score: am,
            g_score: g,
        }
    }).collect();

    merged.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    merged
}
