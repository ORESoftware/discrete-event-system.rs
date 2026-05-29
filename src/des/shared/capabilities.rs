//! Port of `src/des/shared/capabilities.ts`.
//!
//! Capability ports for non-deterministic dependencies (randomness, clock).
//! Dependencies are passed in rather than reached for as globals, so behaviour
//! is reproducible and testable.
//!
//!   interface RandomSource  ->  trait RandomSource
//!   interface Clock         ->  trait Clock
//!
//! The deterministic `SeededRandom` (mulberry32) is implemented natively so the
//! foundation has zero external dependencies. `SystemRandom`/`SystemClock` wrap
//! the platform; in this port they default the seed/clock from the OS so the
//! crate stays dependency-free (swap in the `rand`/`std::time` ecosystem later
//! behind the same traits if desired).

use std::time::{SystemTime, UNIX_EPOCH};

/// Source of pseudo-randomness. `next_float()` returns `[0, 1)`.
pub trait RandomSource {
    /// Uniform float in `[0, 1)`.
    fn next_float(&mut self) -> f64;

    /// Uniform integer in `[min, max)`.
    fn next_int(&mut self, min: i64, max: i64) -> i64 {
        min + (self.next_float() * (max - min) as f64).floor() as i64
    }

    /// Standard-normal sample (mean 0, variance 1), via Box–Muller.
    fn next_gaussian(&mut self) -> f64 {
        let mut u = 0.0;
        let mut v = 0.0;
        while u == 0.0 {
            u = self.next_float();
        }
        while v == 0.0 {
            v = self.next_float();
        }
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

/// Source of the current time, in milliseconds since the Unix epoch.
pub trait Clock {
    fn now_ms(&self) -> u128;
}

/// Seeded, reproducible RNG (mulberry32). Deterministic given a seed.
#[derive(Clone, Debug)]
pub struct SeededRandom {
    state: u32,
}

impl SeededRandom {
    pub fn new(seed: u32) -> Self {
        SeededRandom { state: seed }
    }
}

impl RandomSource for SeededRandom {
    fn next_float(&mut self) -> f64 {
        // mulberry32
        self.state = self.state.wrapping_add(0x6d2b_79f5);
        let mut t = self.state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        ((t ^ (t >> 14)) as f64) / 4_294_967_296.0
    }
}

/// Default platform RNG. Seeds a `SeededRandom` from the wall clock so the
/// foundation needs no external crate; replace with `rand` behind the trait if
/// a CSPRNG is required.
pub struct SystemRandom {
    inner: SeededRandom,
}

impl Default for SystemRandom {
    fn default() -> Self {
        let seed = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u32)
            .unwrap_or(0x1234_5678);
        SystemRandom {
            inner: SeededRandom::new(seed),
        }
    }
}

impl SystemRandom {
    pub fn new() -> Self {
        Self::default()
    }
}

impl RandomSource for SystemRandom {
    fn next_float(&mut self) -> f64 {
        self.inner.next_float()
    }
}

/// Wall-clock clock.
#[derive(Clone, Copy, Debug, Default)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now_ms(&self) -> u128 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeded_is_deterministic() {
        let mut a = SeededRandom::new(42);
        let mut b = SeededRandom::new(42);
        for _ in 0..100 {
            assert_eq!(a.next_float(), b.next_float());
        }
    }

    #[test]
    fn floats_in_unit_interval() {
        let mut r = SeededRandom::new(7);
        for _ in 0..1000 {
            let f = r.next_float();
            assert!((0.0..1.0).contains(&f));
        }
    }

    #[test]
    fn next_int_in_range() {
        let mut r = SeededRandom::new(123);
        for _ in 0..1000 {
            let n = r.next_int(3, 9);
            assert!((3..9).contains(&n));
        }
    }
}
