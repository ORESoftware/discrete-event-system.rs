//! Port of `src/des/general/time-accrued.ts` — the global simulation clock
//! (accrued time + step size), accumulated in EXACT decimal.
//!
//! ## Why `Decimal` and not `f64`
//!
//! This is the engine's master clock: every tick adds `step_size` to a running
//! total, for as many ticks as the simulation runs (often millions). In plain
//! `f64`, adding `0.05` a million times drifts to `49999.99999…` (~1e-6 off);
//! the clock is exactly the COMPOUND-ACCUMULATION case that the precision
//! policy reserves for [`Decimal`] (see `des::shared::precision`). The TS source
//! used `mathjs.BigNumber` here for the same reason and coerced to `Number` on
//! read-out — so we store [`Decimal`] and expose [`SimClock::now_f64`].
//!
//! ## Shape change vs. the TS source
//!
//! The TypeScript module held a module-level mutable singleton (`timeAccrued`)
//! mutated by free setters. Rust has no ergonomic global mutable, and the
//! migration header calls for the `Clock` capability pattern instead: an owned
//! [`SimClock`] threaded through the simulation. The free functions
//! (`getStepSize`/`setStepSize`/`bumpTimeAccruedByTimeStep`/`getTimeAccrued`)
//! become methods (`step_size`/`set_step_size`/`bump`/`now`).

use crate::des::shared::precision::{bgn_int, to_f64, Decimal};

/// The simulation clock: an accrued-time total plus the current step size, both
/// exact base-10 [`Decimal`]s so the running total never drifts.
#[derive(Clone, Debug)]
pub struct SimClock {
    current_time: Decimal,
    step_size_millis: Decimal,
}

impl Default for SimClock {
    fn default() -> Self {
        // TS defaults: currentTime = bgn(0), stepSizeMillis = bgn(10).
        SimClock {
            current_time: Decimal::ZERO,
            step_size_millis: bgn_int(10),
        }
    }
}

impl SimClock {
    /// A fresh clock at t=0 with the default 10ms step (matches the TS singleton).
    pub fn new() -> Self {
        Self::default()
    }

    /// Current step size (`getStepSize`).
    pub fn step_size(&self) -> Decimal {
        self.step_size_millis
    }

    /// Set the step size (`setStepSize`).
    pub fn set_step_size(&mut self, v: Decimal) {
        self.step_size_millis = v;
    }

    /// Advance the clock by `time_step` (`bumpTimeAccruedByTimeStep`). Exact
    /// decimal addition — no accumulation drift.
    pub fn bump(&mut self, time_step: Decimal) {
        self.current_time += time_step;
    }

    /// Advance the clock by exactly one `step_size` tick.
    pub fn tick(&mut self) {
        self.current_time += self.step_size_millis;
    }

    /// Accrued time (`getTimeAccrued`).
    pub fn now(&self) -> Decimal {
        self.current_time
    }

    /// Accrued time coerced to `f64` for read-out / interop, mirroring the TS
    /// `Number(currentTime.toString())` per-tick coercion.
    pub fn now_f64(&self) -> f64 {
        to_f64(self.current_time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::precision::bgn;

    #[test]
    fn defaults_match_ts_singleton() {
        let c = SimClock::new();
        assert_eq!(c.now(), Decimal::ZERO);
        assert_eq!(c.step_size(), bgn_int(10));
        assert_eq!(c.now_f64(), 0.0);
    }

    #[test]
    fn bump_accumulates_exactly() {
        let mut c = SimClock::new();
        c.set_step_size(bgn(0.05));
        for _ in 0..1_000_000 {
            c.bump(c.step_size());
        }
        // Exact — the whole reason this clock is Decimal, not f64.
        assert_eq!(c.now(), bgn(50000.0));
        assert_eq!(c.now_f64(), 50000.0);
    }

    #[test]
    fn tick_uses_current_step_size() {
        let mut c = SimClock::new();
        c.set_step_size(bgn(0.1));
        c.tick();
        c.tick();
        c.tick();
        assert_eq!(c.now(), bgn(0.3)); // 0.1+0.1+0.1 == 0.3 exactly in Decimal
        assert_eq!(c.now_f64(), 0.3);
    }
}
