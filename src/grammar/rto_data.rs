//! Verified Indian RTO registration data for the plate grammar.

use std::collections::HashSet;

/// Two-letter state/UT codes valid as the first two plate characters.
pub const STATE_CODES: &[&str] = &[
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DN", "DL",
    "GA", "GJ", "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD",
    "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "OR", "PB", "PY",
    "RJ", "SK", "TN", "TS", "TG", "TR", "UK", "UA", "UP", "WB",
];

/// Get maximum allocated district number for a given state code.
pub fn get_rto_max_district(code: &str) -> u32 {
    match code {
        "AN" => 3,
        "AP" => 40,
        "AR" => 22,
        "AS" => 34,
        "BR" => 78,
        "CG" => 30,
        "CH" => 4,
        "DD" => 4,
        "DN" => 9,
        "DL" => 16,
        "GA" => 12,
        "GJ" => 39,
        "HP" => 99,
        "HR" => 99,
        "JH" => 24,
        "JK" => 22,
        "KA" => 72,
        "KL" => 99,
        "LA" => 4,
        "LD" => 2,
        "MH" => 56,
        "ML" => 10,
        "MN" => 8,
        "MP" => 75,
        "MZ" => 11,
        "NL" => 10,
        "OD" => 35,
        "OR" => 35,
        "PB" => 99,
        "PY" => 8,
        "RJ" => 59,
        "SK" => 8,
        "TN" => 99,
        "TS" => 38,
        "TG" => 38,
        "TR" => 8,
        "UK" => 20,
        "UA" => 20,
        "UP" => 96,
        "WB" => 98,
        _ => 99,
    }
}

pub fn is_valid_state_code(code: &str) -> bool {
    STATE_CODES.contains(&code)
}

pub fn get_valid_first_letters() -> HashSet<char> {
    STATE_CODES.iter().map(|c| c.chars().next().unwrap()).collect()
}
