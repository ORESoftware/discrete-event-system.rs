//! TypeScript source: `src/des/general/des-base/argmax.ts`
//! Rust target: `src/des/general/des_base/argmax.rs`
//!
//! Porting note: these helpers remain pure functions. Rust callers inject the
//! RNG as a closure, which keeps deterministic tie-breaking explicit.

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/des-base/argmax.ts",
    "src/des/general/des_base/argmax.rs",
    &[
        "RUST MIGRATION: `ARGMAX_EPS_DEFAULT` is a `pub const`.",
        "RUST MIGRATION: Empty TS `-1`/`undefined` results are represented as `None`.",
        "RUST MIGRATION: Random tie-breaking accepts injected `FnMut() -> f64` RNG closures.",
    ],
    &[
        "ARGMAX_EPS_DEFAULT",
        "all_argmax_ties",
        "arg_max_with_tie_break",
        "choose_random_tied",
        "scan_argmax_tie_break",
    ],
);

pub const ARGMAX_EPS_DEFAULT: f64 = 1e-12;

pub fn arg_max_with_tie_break(
    values: &[f64],
    rng: &mut impl FnMut() -> f64,
    eps: f64,
) -> Option<usize> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    if n == 1 {
        return Some(0);
    }

    let mut best = values[0];
    let mut tie_count = 1usize;
    let mut winner = 0usize;

    for (index, value) in values.iter().copied().enumerate().skip(1) {
        if value > best + eps {
            best = value;
            winner = index;
            tie_count = 1;
        } else if value >= best - eps {
            tie_count += 1;
            if rng() * (tie_count as f64) < 1.0 {
                winner = index;
            }
        }
    }

    Some(winner)
}

pub fn arg_max_with_tie_break_default(
    values: &[f64],
    rng: &mut impl FnMut() -> f64,
) -> Option<usize> {
    arg_max_with_tie_break(values, rng, ARGMAX_EPS_DEFAULT)
}

pub fn scan_argmax_tie_break(
    n: usize,
    mut score: impl FnMut(usize) -> f64,
    rng: &mut impl FnMut() -> f64,
    eps: f64,
) -> Option<usize> {
    if n == 0 {
        return None;
    }

    let mut best = f64::NEG_INFINITY;
    let mut tie_count = 0usize;
    let mut winner = None;

    for action in 0..n {
        let value = score(action);
        if !value.is_finite() {
            continue;
        }
        if winner.is_none() || value > best + eps {
            best = value;
            winner = Some(action);
            tie_count = 1;
        } else if value >= best - eps {
            tie_count += 1;
            if rng() * (tie_count as f64) < 1.0 {
                winner = Some(action);
            }
        }
    }

    winner
}

pub fn scan_argmax_tie_break_default(
    n: usize,
    score: impl FnMut(usize) -> f64,
    rng: &mut impl FnMut() -> f64,
) -> Option<usize> {
    scan_argmax_tie_break(n, score, rng, ARGMAX_EPS_DEFAULT)
}

pub fn choose_random_tied<T: Copy>(candidates: &[T], rng: &mut impl FnMut() -> f64) -> Option<T> {
    match candidates.len() {
        0 => None,
        1 => Some(candidates[0]),
        len => {
            let index = (rng() * len as f64).floor() as usize;
            Some(candidates[index.min(len - 1)])
        }
    }
}

pub fn all_argmax_ties(values: &[f64], eps: f64) -> Vec<usize> {
    if values.is_empty() {
        return Vec::new();
    }
    let mut best = values[0];
    for value in values.iter().copied().skip(1) {
        if value > best {
            best = value;
        }
    }
    values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| (*value >= best - eps).then_some(index))
        .collect()
}

pub fn all_argmax_ties_default(values: &[f64]) -> Vec<usize> {
    all_argmax_ties(values, ARGMAX_EPS_DEFAULT)
}
