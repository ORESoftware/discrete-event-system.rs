//! Port of `src/des/general/des-base/argmax.ts`.
//!
//! Random-tie-breaking argmax for value-based decision making. The TS `rng: ()
//! => number` becomes an injected `&mut impl RandomSource` (reservoir tie-break
//! stays reproducible with a seeded source). The `-1` "no winner" sentinel
//! becomes `Option<usize>`.

use crate::des::shared::capabilities::RandomSource;

/// Default float-comparison epsilon. Two scores within ±eps are tied.
pub const ARGMAX_EPS_DEFAULT: f64 = 1e-12;

/// Index of the maximum value, breaking ties uniformly at random. `None` if
/// `values` is empty. Two scores within `eps` are treated as tied.
pub fn arg_max_with_tie_break(values: &[f64], rng: &mut impl RandomSource, eps: f64) -> Option<usize> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(0);
    }
    let mut best = values[0];
    let mut tie_count = 1.0;
    let mut winner = 0;
    for (i, &v) in values.iter().enumerate().skip(1) {
        if v > best + eps {
            best = v;
            winner = i;
            tie_count = 1.0;
        } else if v >= best - eps {
            tie_count += 1.0;
            if rng.next_float() * tie_count < 1.0 {
                winner = i;
            }
        }
    }
    Some(winner)
}

/// Same as [`arg_max_with_tie_break`] but scores are produced lazily by
/// `score(a)`. Non-finite scores are skipped; `None` if no action is finite.
pub fn scan_arg_max_tie_break(
    n: usize,
    score: impl Fn(usize) -> f64,
    rng: &mut impl RandomSource,
    eps: f64,
) -> Option<usize> {
    if n == 0 {
        return None;
    }
    let mut best = f64::NEG_INFINITY;
    let mut tie_count = 0.0;
    let mut winner: Option<usize> = None;
    for a in 0..n {
        let v = score(a);
        if !v.is_finite() {
            continue;
        }
        if winner.is_none() || v > best + eps {
            best = v;
            winner = Some(a);
            tie_count = 1.0;
        } else if v >= best - eps {
            tie_count += 1.0;
            if rng.next_float() * tie_count < 1.0 {
                winner = Some(a);
            }
        }
    }
    winner
}

/// Pick one uniformly at random from an already-tied candidate set. `None` if
/// empty.
pub fn choose_random_tied<'a, T>(candidates: &'a [T], rng: &mut impl RandomSource) -> Option<&'a T> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        return Some(&candidates[0]);
    }
    let idx = (rng.next_float() * candidates.len() as f64).floor() as usize;
    candidates.get(idx.min(candidates.len() - 1))
}

/// All indices tied for the maximum (within `eps`).
pub fn all_arg_max_ties(values: &[f64], eps: f64) -> Vec<usize> {
    let n = values.len();
    if n == 0 {
        return Vec::new();
    }
    let mut best = values[0];
    for &v in values.iter().skip(1) {
        if v > best {
            best = v;
        }
    }
    (0..n).filter(|&i| values[i] >= best - eps).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::SeededRandom;

    #[test]
    fn clear_winner() {
        let mut r = SeededRandom::new(1);
        assert_eq!(arg_max_with_tie_break(&[1.0, 5.0, 2.0], &mut r, ARGMAX_EPS_DEFAULT), Some(1));
        assert_eq!(arg_max_with_tie_break(&[], &mut r, ARGMAX_EPS_DEFAULT), None);
    }

    #[test]
    fn ties_pick_within_set() {
        let mut r = SeededRandom::new(7);
        for _ in 0..50 {
            let w = arg_max_with_tie_break(&[3.0, 3.0, 3.0], &mut r, ARGMAX_EPS_DEFAULT).unwrap();
            assert!(w < 3);
        }
        assert_eq!(all_arg_max_ties(&[3.0, 3.0, 1.0], ARGMAX_EPS_DEFAULT), vec![0, 1]);
    }

    #[test]
    fn scan_skips_non_finite() {
        let mut r = SeededRandom::new(3);
        let w = scan_arg_max_tie_break(3, |a| if a == 2 { 9.0 } else { f64::NEG_INFINITY }, &mut r, ARGMAX_EPS_DEFAULT);
        assert_eq!(w, Some(2));
    }
}
