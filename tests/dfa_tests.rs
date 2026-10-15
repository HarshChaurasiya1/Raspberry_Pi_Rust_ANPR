//! Tests for Indian Number Plate Grammar and DFA.

use raspberry_pi_rust::grammar::PlateDFA;

#[test]
fn test_standard_plate_acceptance() {
    let dfa = PlateDFA::new(None);

    // Standard valid plates
    assert!(dfa.accepts("HR10U2288"));
    assert!(dfa.accepts("DL01AA1111"));
    assert!(dfa.accepts("MH12DE1433"));
    assert!(dfa.accepts("KA05MJ5005"));
    assert!(dfa.accepts("TN01AB1234"));
    assert!(dfa.accepts("AP39EA9999"));
    assert!(dfa.accepts("TG09AB1234"));
}

#[test]
fn test_bharat_series_acceptance() {
    let dfa = PlateDFA::new(None);
    assert!(dfa.accepts("22BH1234AA"));
    assert!(dfa.accepts("23BH9999Z"));
}

#[test]
fn test_invalid_plates_rejection() {
    let dfa = PlateDFA::new(None);
    assert!(!dfa.accepts("ZZ01AA1111")); // Invalid state code ZZ
    assert!(!dfa.accepts("HR00U2288"));   // District 0 is invalid
    assert!(!dfa.accepts("12345"));       // Plain numbers
    assert!(!dfa.accepts("ABCDEFG"));     // Plain letters
}
