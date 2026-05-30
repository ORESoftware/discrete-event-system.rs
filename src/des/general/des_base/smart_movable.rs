//! Port of `src/des/general/des-base/smart-movable.ts`.
//!
//! `SmartMovable` — a token that is ALSO a run-loop participant: it moves
//! through the graph yet advances itself each tick. Only `run_time_step` is
//! abstract; everything else is a provided default backed by an `active` flag.
//!
//! ## Rust shape (faithful translation of the TS abstract class)
//!
//! The TS class `implements Token, IterativeDESParticipant`. Neither of those
//! has a trait in the ported `des-base`:
//!
//!   * **`Token`** — the ported `station.rs` defines NO `Token` trait; tokens
//!     are plain `'static` payloads carried as `Rc<dyn Any>`. So the `Token`
//!     supertrait bound collapses to "any `'static` type". (FLAGGED.)
//!   * **`IterativeDESParticipant`** — `runner.rs` collapsed the TS
//!     participant interface (`DESRunLoopEntity`) into the [`DESStation`] trait;
//!     its participant type is `StationRef = Rc<RefCell<dyn DESStation>>`. There
//!     is no standalone participant trait to be a supertrait of. (FLAGGED.)
//!
//! Because neither supertrait exists, this file follows the migration header's
//! intent literally as a **hook trait + core struct + provided methods**:
//!
//!   * [`SmartMovableCore`] holds the shared fields (`id`, `active`).
//!   * [`SmartMovable`] is the contract: the single required hook
//!     `run_time_step` (the only `abstract` method in TS) plus
//!     `core`/`core_mut` accessors, and provided defaults
//!     (`activate`/`deactivate`/`is_active`/`has_work`/`assert_preconditions`/
//!     `on_finalize`/`num_validators`/`run_validation`) backed by the `active`
//!     flag.
//!
//! INTEGRATION FLAG: to feed a `SmartMovable` into the ported
//! [`run_iterative_des`](super::runner::run_iterative_des) (which takes
//! `Vec<StationRef>`), the concrete movable must additionally `impl DESStation`
//! (carrying a `StationCore`) and be wrapped in `Rc<RefCell<…>>`, because the
//! Rust runner's participant type is `dyn DESStation` rather than the TS's
//! structural `IterativeDESParticipant`. The test below drives the movable
//! directly via its `SmartMovable` methods, matching the TS surface 1:1.

use super::validation::ValidationCheck;

/// Shared state for every [`SmartMovable`] (the fields of the TS abstract
/// class: a readonly `id` and the mutable `active` flag).
#[derive(Clone, Debug)]
pub struct SmartMovableCore {
    pub id: String,
    pub active: bool,
}

impl SmartMovableCore {
    /// `active` starts `false` (matching `protected active = false`).
    pub fn new(id: impl Into<String>) -> Self {
        SmartMovableCore { id: id.into(), active: false }
    }
}

/// A token that advances itself on every run-loop tick while it is active.
///
/// The single required method is [`SmartMovable::run_time_step`] (the only
/// `abstract` member in the TS class). All other methods are provided defaults
/// over the [`SmartMovableCore`] `active` flag and may be left as-is.
pub trait SmartMovable {
    /// Borrow the shared movable state.
    fn core(&self) -> &SmartMovableCore;
    /// Mutably borrow the shared movable state.
    fn core_mut(&mut self) -> &mut SmartMovableCore;

    /// Single self-advancing step (the abstract hook).
    fn run_time_step(&mut self);

    /// This movable's id.
    fn id(&self) -> &str {
        &self.core().id
    }

    /// Mark the movable active (it will report `has_work`).
    fn activate(&mut self) {
        self.core_mut().active = true;
    }

    /// Mark the movable inactive.
    fn deactivate(&mut self) {
        self.core_mut().active = false;
    }

    /// Whether the movable is currently active.
    fn is_active(&self) -> bool {
        self.core().active
    }

    /// Pre-run guard. Default no-op (the TS `assertPreconditions(): void {}`).
    fn assert_preconditions(&mut self) {}

    /// Has work iff active.
    fn has_work(&self) -> bool {
        self.is_active()
    }

    /// Called once after the loop terminates. Default no-op.
    fn on_finalize(&mut self) {}

    /// Number of validators. Default `0` (the TS `numValidators(): 0`).
    fn num_validators(&self) -> usize {
        0
    }

    /// Run validators. Default empty (the TS `runValidation(): []`).
    fn run_validation(&self) -> Vec<ValidationCheck> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 1-D point mass that drifts by `velocity` each tick. It deactivates
    /// itself once it reaches/exceeds `goal`, exercising both the position
    /// update and the `active`-flag-driven `has_work`.
    struct Walker {
        core: SmartMovableCore,
        position: f64,
        velocity: f64,
        goal: f64,
        ticks: usize,
    }

    impl Walker {
        fn new(id: &str, velocity: f64, goal: f64) -> Self {
            Walker { core: SmartMovableCore::new(id), position: 0.0, velocity, goal, ticks: 0 }
        }
    }

    impl SmartMovable for Walker {
        fn core(&self) -> &SmartMovableCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut SmartMovableCore {
            &mut self.core
        }
        fn run_time_step(&mut self) {
            if !self.is_active() {
                return;
            }
            self.position += self.velocity;
            self.ticks += 1;
            if self.position >= self.goal {
                self.deactivate();
            }
        }
    }

    #[test]
    fn inactive_until_activated() {
        let mut w = Walker::new("w", 1.0, 3.0);
        assert!(!w.is_active());
        assert!(!w.has_work());
        // A run_time_step while inactive is a no-op.
        w.run_time_step();
        assert_eq!(w.position, 0.0);
        assert_eq!(w.ticks, 0);
        // No validators / default guards.
        assert_eq!(w.num_validators(), 0);
        assert!(w.run_validation().is_empty());
        assert_eq!(w.id(), "w");
    }

    #[test]
    fn advances_position_each_tick_until_goal() {
        let mut w = Walker::new("w", 1.0, 3.0);
        w.activate();
        assert!(w.has_work());

        let mut steps = 0;
        while w.has_work() {
            w.run_time_step();
            steps += 1;
            assert!(steps <= 10, "walker should terminate");
        }

        assert_eq!(steps, 3);
        assert_eq!(w.ticks, 3);
        assert!((w.position - 3.0).abs() < 1e-12);
        // Reaching the goal deactivated it.
        assert!(!w.is_active());
        assert!(!w.has_work());
    }

    #[test]
    fn deactivate_halts_motion() {
        let mut w = Walker::new("w", 2.0, 100.0);
        w.activate();
        w.run_time_step();
        assert!((w.position - 2.0).abs() < 1e-12);
        w.deactivate();
        assert!(!w.has_work());
        w.run_time_step();
        // Position unchanged after deactivation.
        assert!((w.position - 2.0).abs() < 1e-12);
        assert_eq!(w.ticks, 1);
    }
}
