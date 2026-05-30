//! TypeScript source: `src/des/general/prng.ts`
//! Rust target: `src/des/general/prng.rs`
//!
//! Porting note: the TypeScript module exposes `mulberry32(seed)` and a
//! `withSeed` helper that temporarily replaces `Math.random`. Rust should keep
//! the deterministic generator, but pass it explicitly instead of relying on a
//! process-global random source.

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/prng.ts",
    "src/des/general/prng.rs",
    &[
        "RUST MIGRATION: `mulberry32(seed)` is `Mulberry32::new(seed)` plus `next_f64()`.",
        "RUST MIGRATION: Avoid a `withSeed` global RNG swap; inject `Mulberry32` or a closure into callers.",
    ],
    &["Mulberry32", "mulberry32_sequence", "with_seed"],
);

#[derive(Debug, Clone)]
pub struct Mulberry32 {
    state: u32,
}

impl Mulberry32 {
    pub fn new(seed: u32) -> Self {
        Self { state: seed }
    }

    pub fn next_u32(&mut self) -> u32 {
        self.state = self.state.wrapping_add(0x6D2B_79F5);
        let mut t = (self.state ^ (self.state >> 15)).wrapping_mul(1 | self.state);
        t = t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t)) ^ t;
        t ^ (t >> 14)
    }

    pub fn next_f64(&mut self) -> f64 {
        f64::from(self.next_u32()) / 4_294_967_296.0
    }
}

pub fn mulberry32_sequence(seed: u32, count: usize) -> Vec<f64> {
    let mut rng = Mulberry32::new(seed);
    (0..count).map(|_| rng.next_f64()).collect()
}

pub fn with_seed<T>(seed: u32, mut f: impl FnMut(&mut Mulberry32) -> T) -> T {
    let mut rng = Mulberry32::new(seed);
    f(&mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mulberry32_is_deterministic() {
        assert_eq!(mulberry32_sequence(7, 5), mulberry32_sequence(7, 5));
    }

    #[test]
    fn mulberry32_matches_typescript_first_values() {
        let values = mulberry32_sequence(1, 3);
        assert!((values[0] - 0.6270739405881613).abs() < 1e-15);
        assert!((values[1] - 0.002735721180215478).abs() < 1e-15);
        assert!((values[2] - 0.5274470399599522).abs() < 1e-15);
    }
}
