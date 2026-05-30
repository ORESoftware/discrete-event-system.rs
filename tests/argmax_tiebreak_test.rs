//! TypeScript source: `src/des/test/argmax-tiebreak-test.ts`
//! Rust target: `tests/argmax_tiebreak_test.rs`

use discrete_event_system_rs::des::general::des_base::argmax::{
    all_argmax_ties_default, arg_max_with_tie_break_default, choose_random_tied,
    scan_argmax_tie_break_default,
};
use discrete_event_system_rs::des::general::prng::Mulberry32;

fn rng(seed: u32) -> impl FnMut() -> f64 {
    let mut rng = Mulberry32::new(seed);
    move || rng.next_f64()
}

#[test]
fn argmax_empty_and_singleton_match_typescript_surface() {
    assert_eq!(arg_max_with_tie_break_default(&[], &mut rng(1)), None);
    assert_eq!(
        arg_max_with_tie_break_default(&[42.0], &mut rng(1)),
        Some(0)
    );
}

#[test]
fn argmax_unique_winner_is_always_returned() {
    let mut next = rng(1);
    for _ in 0..200 {
        assert_eq!(
            arg_max_with_tie_break_default(&[1.0, 2.0, 3.0, 2.0, 1.0], &mut next),
            Some(2)
        );
    }
}

#[test]
fn argmax_uniformly_breaks_five_way_ties() {
    let mut counts = [0usize; 5];
    let trials = 5_000usize;
    let mut next = rng(42);
    for _ in 0..trials {
        let index = arg_max_with_tie_break_default(&[7.0, 7.0, 7.0, 7.0, 7.0], &mut next).unwrap();
        counts[index] += 1;
    }
    let expected = trials as f64 / 5.0;
    let sigma = (trials as f64 * (1.0 / 5.0) * (4.0 / 5.0)).sqrt();
    assert!(
        counts
            .iter()
            .all(|count| (*count as f64 - expected).abs() <= 4.0 * sigma),
        "counts={counts:?}, expected={expected}, sigma={sigma}"
    );
    assert!(counts.iter().all(|count| *count > 0));
}

#[test]
fn argmax_eps_tolerance_treats_near_equal_values_as_tied() {
    let mut counts = [0usize; 3];
    for trial in 0..1_000u32 {
        let mut next = rng(trial + 1);
        let index =
            arg_max_with_tie_break_default(&[1.0, 1.0 + 1e-15, 1.0 - 1e-15], &mut next).unwrap();
        counts[index] += 1;
    }
    assert!(counts.iter().all(|count| *count > 100), "counts={counts:?}");
}

#[test]
fn scan_argmax_excludes_non_finite_scores() {
    let mut next = rng(1);
    let index = scan_argmax_tie_break_default(
        4,
        |action| {
            if action == 1 {
                f64::NEG_INFINITY
            } else {
                7.0
            }
        },
        &mut next,
    );
    assert_ne!(index, Some(1));
    assert_eq!(
        scan_argmax_tie_break_default(3, |_| f64::NEG_INFINITY, &mut rng(1)),
        None
    );
}

#[test]
fn scan_argmax_uniformly_breaks_four_way_ties() {
    let mut counts = [0usize; 4];
    for trial in 0..2_000u32 {
        let mut next = rng(trial * 31 + 17);
        let index = scan_argmax_tie_break_default(4, |_| 1.0, &mut next).unwrap();
        counts[index] += 1;
    }
    assert!(
        counts.iter().all(|count| *count > 350 && *count < 650),
        "counts={counts:?}"
    );
}

#[test]
fn all_argmax_ties_and_choose_random_tied_match_typescript() {
    assert_eq!(
        all_argmax_ties_default(&[1.0, 3.0, 3.0, 2.0, 3.0]),
        vec![1, 2, 4]
    );
    assert_eq!(all_argmax_ties_default(&[5.0]), vec![0]);
    assert!(all_argmax_ties_default(&[]).is_empty());
    assert_eq!(choose_random_tied::<i32>(&[], &mut rng(1)), None);
    assert_eq!(choose_random_tied(&[42], &mut rng(1)), Some(42));
}
