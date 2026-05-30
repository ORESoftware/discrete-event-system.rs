//! Port of `src/des/general/des-base/runner.ts`.
//!
//! The iterative DES runner: drives every run-loop participant's
//! [`DESStation::run_time_step`] in (optionally shuffled) order each tick until
//! the system goes quiescent, a stop predicate fires, or a tick cap is reached.
//!
//! ## Rust shape (faithful translation of the TS module)
//!
//! * `type IterativeDESParticipant = DESRunLoopEntity` — the TS participant
//!   interface is a structural subset of `DESStation` (optional
//!   `assertPreconditions`/`hasWork`/`onFinalize`/`numValidators`/
//!   `runValidation`). The ported `station.rs` does **not** define a separate
//!   `DESRunLoopEntity` trait; every one of those methods already exists on the
//!   [`DESStation`] trait (as provided defaults), so the participant type here
//!   is simply [`StationRef`] (`Rc<RefCell<dyn DESStation>>`). See the dep flag
//!   in the module docs of the migration notes.
//! * `interface IterativeRunOptions` → [`IterativeRunOptions`] with boxed
//!   `FnMut` callback fields; `Default` gives the TS option defaults
//!   (`shuffle = true`, `run_validators = true`, no tick cap).
//! * `interface IterativeRunSummary` → [`IterativeRunSummary`]; the
//!   `'done' | 'maxticks' | 'stop-when'` union → [`RunReason`].
//! * `type DESResultStation<R>` → trait [`DESResultStation`] (`DESStation` + a
//!   `result(..) -> R` method).
//! * `rng: () => number` defaulting to `Math.random` → an injected
//!   `Box<dyn FnMut() -> f64>`. Per the repo's "no ambient impurity" rule we do
//!   NOT call a global RNG: when no `rng` is supplied we fall back to a
//!   **deterministic** seeded mulberry32 (a behavioural divergence from the TS
//!   `Math.random` default — flagged).
//! * `seen = new Set<participant>()` (identity-keyed) → a first-seen-ordered
//!   `Vec<StationRef>` deduplicated by station id (`HashSet<String>`), per the
//!   migration header's suggestion.

use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use super::station::{run_station_validation, DESStation, StationRef};
use super::validation::ValidationCheck;

/// A run-loop participant. The TS `IterativeDESParticipant = DESRunLoopEntity`;
/// here every participant is a shared-handle [`DESStation`].
pub type IterativeDESParticipant = StationRef;

/// Why the run loop terminated (`'done' | 'maxticks' | 'stop-when'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunReason {
    /// No participant had work left.
    Done,
    /// The `max_ticks` cap was reached before quiescence.
    MaxTicks,
    /// The `stop_when` predicate returned `true`.
    StopWhen,
}

impl RunReason {
    /// The TS string spelling of this reason.
    pub fn as_str(self) -> &'static str {
        match self {
            RunReason::Done => "done",
            RunReason::MaxTicks => "maxticks",
            RunReason::StopWhen => "stop-when",
        }
    }
}

/// Options controlling a [`run_iterative_des`] run.
///
/// Mirrors the TS `IterativeRunOptions`. Callback fields are owned boxed
/// `FnMut`s; absent callbacks are `None`. Construct via [`Default`] and override
/// fields, e.g. `IterativeRunOptions { shuffle: false, ..Default::default() }`.
pub struct IterativeRunOptions {
    /// Maximum simulation ticks. `None` = run until quiescent (TS `Infinity`).
    pub max_ticks: Option<usize>,
    /// Stop predicate run BEFORE each tick; returning `true` terminates.
    pub stop_when: Option<Box<dyn FnMut(usize, &[StationRef]) -> bool>>,
    /// Optional dynamic roster (for participants entering/leaving mid-run).
    pub get_run_loop_entities: Option<Box<dyn FnMut() -> Vec<StationRef>>>,
    /// RNG for tick-order shuffling; `None` uses a deterministic default.
    pub rng: Option<Box<dyn FnMut() -> f64>>,
    /// Whether to randomise station-execution order each tick (default `true`).
    pub shuffle: bool,
    /// Optional per-tick callback (instrumentation, animation, …).
    pub on_tick: Option<Box<dyn FnMut(usize, &[StationRef])>>,
    /// Run validators on every station after the loop terminates (default
    /// `true`).
    pub run_validators: bool,
}

impl Default for IterativeRunOptions {
    fn default() -> Self {
        IterativeRunOptions {
            max_ticks: None,
            stop_when: None,
            get_run_loop_entities: None,
            rng: None,
            shuffle: true,
            on_tick: None,
            run_validators: true,
        }
    }
}

/// Outcome of a [`run_iterative_des`] run.
#[derive(Clone, Debug, Default)]
pub struct IterativeRunSummary {
    /// Number of completed ticks.
    pub ticks: usize,
    /// Why the loop terminated.
    pub reason: Option<RunReason>,
    /// Aggregated validator output — `Some` iff `run_validators` was on AND at
    /// least one station had validators registered. First-seen run-loop order.
    pub validation: Option<Vec<ValidationCheck>>,
    /// `Some(true)` iff every entry in `validation` passed.
    pub validation_ok: Option<bool>,
}

/// A station that, once run, can be reduced to a result value of type `R`.
/// (`type DESResultStation<R> = DESStation & { result(validation?) : R }`.)
pub trait DESResultStation<R>: DESStation {
    /// Reduce the terminal station state (plus the run's validation output)
    /// into a result value.
    fn result(&self, validation: &[ValidationCheck]) -> R;
}

/// Deterministic seeded RNG used when the caller supplies no `rng`.
///
/// A mulberry32 generator. NOTE: the TS default was `Math.random` (ambient,
/// non-deterministic); per the repo's capability-injection rule we substitute a
/// fixed-seed deterministic stream here. Pass `rng: Some(..)` for full control.
fn default_rng() -> Box<dyn FnMut() -> f64> {
    let mut state: u32 = 0x9E37_79B9;
    Box::new(move || {
        state = state.wrapping_add(0x6D2B_79F5);
        let mut t = state;
        t = (t ^ (t >> 15)).wrapping_mul(t | 1);
        t ^= t.wrapping_add((t ^ (t >> 7)).wrapping_mul(t | 61));
        let r = (t ^ (t >> 14)) as f64;
        r / 4_294_967_296.0
    })
}

/// Mutate `arr` in place via Fisher-Yates with the given `[0,1)` RNG.
fn shuffle_in_place<T>(arr: &mut [T], rng: &mut dyn FnMut() -> f64) {
    if arr.len() < 2 {
        return;
    }
    let mut i = arr.len() - 1;
    while i > 0 {
        let mut j = (rng() * ((i + 1) as f64)).floor() as usize;
        if j > i {
            j = i; // guard against an out-of-range RNG returning >= 1.0
        }
        arr.swap(i, j);
        i -= 1;
    }
}

/// Run the simulation until a terminal condition fires.
///
/// Termination order (checked at the top of every tick):
///   1. `stop_when(tick)` returns `true`  → [`RunReason::StopWhen`]
///   2. `tick >= max_ticks`               → [`RunReason::MaxTicks`]
///   3. no station has work               → [`RunReason::Done`]
///
/// Each tick: compute "any participant has work?", shuffle the order (if
/// enabled), tick every participant, then fire `on_tick`.
pub fn run_iterative_des(stations: Vec<StationRef>, opts: IterativeRunOptions) -> IterativeRunSummary {
    let IterativeRunOptions {
        max_ticks,
        mut stop_when,
        mut get_run_loop_entities,
        rng,
        shuffle,
        mut on_tick,
        run_validators: want_validate,
    } = opts;
    let mut rng: Box<dyn FnMut() -> f64> = rng.unwrap_or_else(default_rng);

    // `seen`: first-seen-ordered roster, deduplicated by station id.
    let mut seen: Vec<StationRef> = Vec::new();
    let mut seen_ids: HashSet<String> = HashSet::new();

    let mut current_entities = || -> Vec<StationRef> {
        let entities = match &mut get_run_loop_entities {
            Some(f) => f(),
            None => stations.clone(),
        };
        for s in &entities {
            let id = s.borrow().id().to_string();
            if seen_ids.insert(id) {
                seen.push(s.clone());
            }
        }
        entities
    };

    for s in &current_entities() {
        s.borrow_mut().assert_preconditions();
    }

    let mut tick: usize = 0;
    let reason: RunReason;
    loop {
        let entities = current_entities();
        if let Some(f) = stop_when.as_mut() {
            if f(tick, &entities) {
                reason = RunReason::StopWhen;
                break;
            }
        }
        if let Some(mt) = max_ticks {
            if tick >= mt {
                eprintln!(
                    "[run_iterative_des] hit max_ticks={mt} before the system went quiescent ({} participants still active) — increase max_ticks or check for a non-terminating model.",
                    entities.len()
                );
                reason = RunReason::MaxTicks;
                break;
            }
        }
        let any_work = entities.iter().any(|s| s.borrow().has_work());
        if !any_work {
            reason = RunReason::Done;
            break;
        }
        let mut order = entities.clone();
        if shuffle {
            shuffle_in_place(&mut order, rng.as_mut());
        }
        for s in &order {
            s.borrow_mut().run_time_step();
        }
        if let Some(f) = on_tick.as_mut() {
            f(tick, &entities);
        }
        tick += 1;
    }

    // Release the closure's borrows of `seen` before the finalize/validate loops.
    drop(current_entities);

    for s in &seen {
        s.borrow_mut().on_finalize();
    }

    let mut summary = IterativeRunSummary {
        ticks: tick,
        reason: Some(reason),
        validation: None,
        validation_ok: None,
    };

    if want_validate {
        let mut all_checks: Vec<ValidationCheck> = Vec::new();
        for s in &seen {
            let st = s.borrow();
            if st.num_validators() == 0 {
                continue;
            }
            all_checks.extend(run_station_validation(&*st));
        }
        if !all_checks.is_empty() {
            let ok = all_checks.iter().all(|c| c.passed);
            if !ok {
                let failed: Vec<&str> = all_checks.iter().filter(|c| !c.passed).map(|c| c.name.as_str()).collect();
                eprintln!(
                    "[run_iterative_des] {}/{} validators FAILED after {tick} ticks: {}",
                    failed.len(),
                    all_checks.len(),
                    failed.join(", ")
                );
            }
            summary.validation = Some(all_checks);
            summary.validation_ok = Some(ok);
        }
    }

    summary
}

/// Run a single result-bearing station and reduce it to its result value.
pub fn run_result_station<R, S>(station: Rc<RefCell<S>>, opts: IterativeRunOptions) -> R
where
    S: DESResultStation<R> + 'static,
{
    let participant: StationRef = station.clone();
    let summary = run_iterative_des(vec![participant], opts);
    let checks = summary.validation.unwrap_or_default();
    let result = station.borrow().result(&checks);
    result
}

/// The failed checks in `summary.validation` (empty if none / absent).
pub fn failed_validation_checks(summary: &IterativeRunSummary) -> Vec<ValidationCheck> {
    summary
        .validation
        .as_ref()
        .map(|v| v.iter().filter(|c| !c.passed).cloned().collect())
        .unwrap_or_default()
}

/// Comma-joined names of the failed validation checks.
pub fn validation_failure_names(summary: &IterativeRunSummary) -> String {
    failed_validation_checks(summary).iter().map(|c| c.name.clone()).collect::<Vec<_>>().join(", ")
}

/// Return `Err` if any validation check failed (TS threw an `Error`).
///
/// A validation failure is a recoverable, expected runtime outcome (not an
/// invariant bug), so per the repo's error convention this returns
/// `Result<(), String>` rather than `panic!`-ing.
pub fn assert_no_validation_failures(summary: &IterativeRunSummary, model_name: &str) -> Result<(), String> {
    let names = validation_failure_names(summary);
    if !names.is_empty() {
        let n = failed_validation_checks(summary).len();
        eprintln!("[{model_name}] post-run validation failed ({n} checks): {names}");
        return Err(format!("{model_name} validation failed: {names}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::des_base::station::StationCore;
    use crate::des::general::des_base::validation::{FnValidator, Validator};
    use std::any::Any;

    /// A station with `remaining` units of self-generated work; it ticks down
    /// each step and reports `has_work()` from its own counter (not an inbox).
    struct Countdown {
        core: StationCore,
        remaining: usize,
        ticks: usize,
    }

    impl Countdown {
        fn new(id: &str, remaining: usize) -> Self {
            Countdown { core: StationCore::new(id), remaining, ticks: 0 }
        }
    }

    impl DESStation for Countdown {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn has_work(&self) -> bool {
            self.remaining > 0
        }
        fn run_time_step(&mut self) {
            if self.remaining > 0 {
                self.remaining -= 1;
                self.ticks += 1;
            }
        }
    }

    impl DESResultStation<usize> for Countdown {
        fn result(&self, _validation: &[ValidationCheck]) -> usize {
            self.ticks
        }
    }

    /// A station that always reports work — used to exercise the tick cap.
    struct Forever {
        core: StationCore,
    }

    impl DESStation for Forever {
        fn core(&self) -> &StationCore {
            &self.core
        }
        fn core_mut(&mut self) -> &mut StationCore {
            &mut self.core
        }
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn has_work(&self) -> bool {
            true
        }
        fn run_time_step(&mut self) {}
    }

    #[test]
    fn drives_to_quiescence() {
        let s = Rc::new(RefCell::new(Countdown::new("cd", 4)));
        let summary = run_iterative_des(
            vec![s.clone() as StationRef],
            IterativeRunOptions { shuffle: false, ..Default::default() },
        );
        assert_eq!(summary.reason, Some(RunReason::Done));
        assert_eq!(summary.ticks, 4);
        assert_eq!(s.borrow().ticks, 4);
        assert_eq!(s.borrow().remaining, 0);
        // No validators registered -> no validation block.
        assert!(summary.validation.is_none());
    }

    #[test]
    fn respects_max_ticks_and_stop_when() {
        let forever = Rc::new(RefCell::new(Forever { core: StationCore::new("inf") }));
        let summary = run_iterative_des(
            vec![forever.clone() as StationRef],
            IterativeRunOptions { max_ticks: Some(5), shuffle: false, ..Default::default() },
        );
        assert_eq!(summary.reason, Some(RunReason::MaxTicks));
        assert_eq!(summary.ticks, 5);

        // stop_when fires before max_ticks.
        let forever2 = Rc::new(RefCell::new(Forever { core: StationCore::new("inf2") }));
        let summary = run_iterative_des(
            vec![forever2 as StationRef],
            IterativeRunOptions {
                max_ticks: Some(100),
                shuffle: false,
                stop_when: Some(Box::new(|tick, _| tick >= 3)),
                ..Default::default()
            },
        );
        assert_eq!(summary.reason, Some(RunReason::StopWhen));
        assert_eq!(summary.ticks, 3);
    }

    #[test]
    fn aggregates_validation_and_result_station() {
        let s = Rc::new(RefCell::new(Countdown::new("cd", 3)));
        {
            let v: Box<dyn Validator<dyn DESStation>> =
                FnValidator::new("drained", |st: &dyn DESStation| {
                    let cd = st.as_any().downcast_ref::<Countdown>().unwrap();
                    vec![ValidationCheck {
                        name: "drained".to_string(),
                        passed: cd.remaining == 0,
                        ..Default::default()
                    }]
                })
                .boxed();
            s.borrow_mut().add_validator(v);
        }

        let ticks = run_result_station(s.clone(), IterativeRunOptions { shuffle: false, ..Default::default() });
        assert_eq!(ticks, 3);

        // Re-run via the raw runner to inspect the validation summary.
        let s2 = Rc::new(RefCell::new(Countdown::new("cd2", 2)));
        {
            let v: Box<dyn Validator<dyn DESStation>> =
                FnValidator::new("drained", |st: &dyn DESStation| {
                    let cd = st.as_any().downcast_ref::<Countdown>().unwrap();
                    vec![ValidationCheck {
                        name: "drained".to_string(),
                        passed: cd.remaining == 0,
                        ..Default::default()
                    }]
                })
                .boxed();
            s2.borrow_mut().add_validator(v);
        }
        let summary = run_iterative_des(vec![s2 as StationRef], IterativeRunOptions { shuffle: false, ..Default::default() });
        assert_eq!(summary.validation_ok, Some(true));
        assert_eq!(summary.validation.as_ref().map(|v| v.len()), Some(1));
        assert!(assert_no_validation_failures(&summary, "test-model").is_ok());
    }
}
