//! Port of `src/des/general/prng.ts` — seedable mulberry32 PRNG.
//!
//! In TS this monkey-patched the global `Math.random`; in Rust that whole hack
//! is replaced by the RNG capability port. `mulberry32` IS `SeededRandom`
//! (re-exported from `shared::capabilities`); `with_seed` runs a closure with a
//! freshly-seeded `SeededRandom` injected, rather than swapping a global.

pub use crate::des::shared::capabilities::{RandomSource, SeededRandom};

/// Construct a seeded RNG (the `mulberry32(seed)` factory).
pub fn mulberry32(seed: u32) -> SeededRandom {
    SeededRandom::new(seed)
}

/// Run `f` with a freshly seeded `SeededRandom` injected. Replaces the TS
/// `withSeed`, which swapped the global `Math.random`.
pub fn with_seed<T>(seed: u32, f: impl FnOnce(&mut SeededRandom) -> T) -> T {
    let mut rng = SeededRandom::new(seed);
    f(&mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reproducible() {
        let a: Vec<f64> = with_seed(99, |r| (0..5).map(|_| r.next_float()).collect());
        let b: Vec<f64> = with_seed(99, |r| (0..5).map(|_| r.next_float()).collect());
        assert_eq!(a, b);
    }
}
