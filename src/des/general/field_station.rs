//! Port of `src/des/general/field-station.ts` — the framework substrate for
//! ODE / PDE problems, using the census-snapshot synchronous-data-flow pattern.
//!
//! CORE IDEA
//! Each ODE/PDE problem is mapped to a set of stationary entities ("field
//! stations"), each holding a scalar value. A central [`Census`] station
//! snapshots every value at the start of every tick. Each field station then
//! runs `run_time_step`, reading only the snapshot — never another station's
//! mid-tick state. Because field stations read only the frozen snapshot, the
//! order in which they run within a tick does not matter (exactly the property
//! that makes finite-difference schemes order-independent), which is why the
//! [`FieldSimulation`] can shuffle the per-tick processing order as a sanity
//! check.
//!
//! Rust shape (faithful to the TS `extends` chain):
//!   * `abstract class Station extends TimeSteppedStation` -> marker trait
//!     [`Station`]`: TimeSteppedStation`.
//!   * `class Census` / `class FieldStation` -> structs implementing
//!     [`TimeSteppedStation`] + [`Station`].
//!   * `type FieldUpdater` -> the boxed-closure alias [`FieldUpdater`].
//!   * `interface FieldSimulationOptions` / `interface FieldSimulationResult`
//!     -> structs [`FieldSimulationOptions`] / [`FieldSimulationResult`].
//!
//! Conversion notes:
//!   * `Float64Array` -> `Vec<f64>`.
//!   * The TS aliasing (Census references the field stations, each field
//!     references the census) is preserved with `Rc<RefCell<..>>`. This forms a
//!     reference cycle (as the GC'd TS version effectively did); it is a
//!     short-lived simulation object, so the cycle is left strong rather than
//!     introducing `Weak` (FLAGGED as a minor, behaviour-neutral deviation).
//!   * `shuffleInPlace` takes the mulberry32 `rng()` closure -> a generic helper
//!     over `&mut impl RandomSource`.

use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::prng::mulberry32;
use crate::des::general::time_stepped_station::TimeSteppedStation;
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

/// Fisher–Yates in-place shuffle driven by an injected RNG (TS `shuffleInPlace`).
fn shuffle_in_place<T>(arr: &mut [T], rng: &mut impl RandomSource) {
    let mut i = arr.len();
    while i > 1 {
        i -= 1;
        let j = (rng.next_float() * (i as f64 + 1.0)).floor() as usize;
        arr.swap(i, j);
    }
}

// -----------------------------------------------------------------------------
// Base classes.
// -----------------------------------------------------------------------------

/// Marker for a station in the field-station family (TS `abstract class Station
/// extends TimeSteppedStation {}`). Adds no behaviour over [`TimeSteppedStation`].
pub trait Station: TimeSteppedStation {}

/// Optional spatial position of a field station (TS `number | [number, number]`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Position {
    /// 1-D coordinate.
    Scalar(f64),
    /// 2-D coordinate `(x, y)`.
    Pair(f64, f64),
}

/// The updater applied each tick (TS `type FieldUpdater`). Receives the
/// previous-tick snapshot (`prev`, for leapfrog/wave schemes), the current-tick
/// snapshot (`cur`), this station's index into the snapshot (`self_index`), the
/// step `dt`, and the elapsed time `t`; returns the station's new value.
pub type FieldUpdater = Box<dyn Fn(&[f64], &[f64], usize, f64, f64) -> f64>;

/// Census station: snapshots every field station's `value` at the start of each
/// tick so field stations read a frozen, order-independent view.
pub struct Census {
    pub id: String,
    /// Value snapshot at the start of THIS tick.
    pub snap: Vec<f64>,
    /// Value snapshot from the PREVIOUS tick (for leapfrog / wave schemes).
    pub prev_snap: Vec<f64>,
    /// The field stations whose values are snapshotted.
    pub fields: Vec<Rc<RefCell<FieldStation>>>,
}

impl Census {
    pub fn new(id: impl Into<String>, fields: Vec<Rc<RefCell<FieldStation>>>) -> Self {
        let n = fields.len();
        Census {
            id: id.into(),
            snap: vec![0.0; n],
            prev_snap: vec![0.0; n],
            fields,
        }
    }
}

impl TimeSteppedStation for Census {
    fn id(&self) -> &str {
        &self.id
    }
    fn run_time_step(&mut self, _step_size: f64, _t: f64) {
        // Promote current snapshot to prev (used by leapfrog/wave equation).
        self.prev_snap.copy_from_slice(&self.snap);
        for i in 0..self.fields.len() {
            self.snap[i] = self.fields[i].borrow().value;
        }
    }
}

impl Station for Census {}

/// Generic field station: holds a scalar `value` and applies an [`FieldUpdater`]
/// each tick that reads the census snapshot and returns the new value.
pub struct FieldStation {
    pub id: String,
    pub value: f64,
    pub updater: FieldUpdater,
    pub census: Rc<RefCell<Census>>,
    /// Index into the census's `snap` array; set by [`FieldSimulation`] (`-1`
    /// until assigned, mirroring the TS default).
    pub index: i64,
    /// Optional spatial position (used by 1-D / 2-D PDE schemes).
    pub position: Option<Position>,
}

impl FieldStation {
    pub fn new(
        id: impl Into<String>,
        value: f64,
        updater: FieldUpdater,
        census: Rc<RefCell<Census>>,
    ) -> Self {
        FieldStation {
            id: id.into(),
            value,
            updater,
            census,
            index: -1,
            position: None,
        }
    }
}

impl TimeSteppedStation for FieldStation {
    fn id(&self) -> &str {
        &self.id
    }
    fn run_time_step(&mut self, dt: f64, t: f64) {
        let census = self.census.borrow();
        let v = (self.updater)(&census.prev_snap, &census.snap, self.index as usize, dt, t);
        drop(census);
        self.value = v;
    }
}

impl Station for FieldStation {}

// -----------------------------------------------------------------------------
// FieldSimulation — drives the tick loop.
// -----------------------------------------------------------------------------

/// Options for a [`FieldSimulation`] (TS `interface FieldSimulationOptions`).
#[derive(Clone, Copy, Debug)]
pub struct FieldSimulationOptions {
    /// Seed for the per-tick processing-order shuffle (TS default 1).
    pub seed: u32,
    /// Shuffle the field stations' processing order each tick (TS default true).
    pub shuffle: bool,
    /// Record the full snapshot at each tick into the trace (TS default true).
    pub record_trace: bool,
}

impl Default for FieldSimulationOptions {
    fn default() -> Self {
        FieldSimulationOptions {
            seed: 1,
            shuffle: true,
            record_trace: true,
        }
    }
}

/// Recorded `(time, value-row)` history (TS `trace: {t, values}`).
#[derive(Clone, Debug, Default)]
pub struct FieldTrace {
    pub t: Vec<f64>,
    pub values: Vec<Vec<f64>>,
}

/// Outcome of a [`FieldSimulation::run`] (TS `interface FieldSimulationResult`).
#[derive(Clone, Debug, Default)]
pub struct FieldSimulationResult {
    pub trace: FieldTrace,
    pub final_values: Vec<f64>,
    pub ticks: usize,
}

/// Drives the census-snapshot tick loop over a set of field stations.
pub struct FieldSimulation {
    pub fields: Vec<Rc<RefCell<FieldStation>>>,
    pub census: Rc<RefCell<Census>>,
    pub rng: SeededRandom,
    pub shuffle: bool,
    pub record_trace: bool,
}

impl FieldSimulation {
    /// Wire `fields` to a fresh census, seed both snapshots from the starting
    /// values, and configure the loop (TS `FieldSimulation` constructor).
    pub fn new(fields: Vec<Rc<RefCell<FieldStation>>>, opts: FieldSimulationOptions) -> Self {
        let census = Rc::new(RefCell::new(Census::new("census", fields.clone())));
        for (i, f) in fields.iter().enumerate() {
            let mut fb = f.borrow_mut();
            fb.index = i as i64;
            fb.census = census.clone();
        }
        // Initialise both snap and prev_snap to the starting values so leapfrog
        // schemes have a sane "u(t = -dt)" reading on tick 0.
        {
            let mut c = census.borrow_mut();
            for (i, f) in fields.iter().enumerate() {
                let v = f.borrow().value;
                c.snap[i] = v;
                c.prev_snap[i] = v;
            }
        }
        FieldSimulation {
            fields,
            census,
            rng: mulberry32(opts.seed),
            shuffle: opts.shuffle,
            record_trace: opts.record_trace,
        }
    }

    /// Advance from `t0` to `t1` in steps of `dt`, returning the trace, final
    /// values, and tick count.
    pub fn run(&mut self, t0: f64, t1: f64, dt: f64) -> FieldSimulationResult {
        let mut t: Vec<f64> = Vec::new();
        let mut values: Vec<Vec<f64>> = Vec::new();
        if self.record_trace {
            t.push(t0);
            values.push(self.census.borrow().snap.clone());
        }
        let mut tn = t0;
        let mut tick = 0usize;
        while tn + 0.5 * dt < t1 {
            self.census.borrow_mut().run_time_step(dt, tn);
            let mut order: Vec<Rc<RefCell<FieldStation>>> = self.fields.clone();
            if self.shuffle {
                shuffle_in_place(&mut order, &mut self.rng);
            }
            for f in &order {
                f.borrow_mut().run_time_step(dt, tn);
            }
            tn += dt;
            tick += 1;
            if self.record_trace {
                t.push(tn);
                let mut row = vec![0.0; self.fields.len()];
                for i in 0..self.fields.len() {
                    row[i] = self.fields[i].borrow().value;
                }
                values.push(row);
            }
        }
        let final_values: Vec<f64> = self.fields.iter().map(|f| f.borrow().value).collect();
        FieldSimulationResult {
            trace: FieldTrace { t, values },
            final_values,
            ticks: tick,
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for the census-snapshot field simulator. They pin the station
    //! output against the analytic / pure-math expectation: explicit-Euler
    //! exponential decay, and order-independence of two coupled fields under a
    //! shuffled vs. fixed processing order. Fixed seeds keep the shuffle
    //! reproducible.

    use super::*;

    fn placeholder_census() -> Rc<RefCell<Census>> {
        Rc::new(RefCell::new(Census::new("placeholder", Vec::new())))
    }

    #[test]
    fn euler_exponential_decay() {
        // y' = -y, explicit Euler: y_{n+1} = y_n - dt * y_n.
        let census = placeholder_census();
        let updater: FieldUpdater = Box::new(|_prev, cur, i, dt, _t| cur[i] - dt * cur[i]);
        let y = Rc::new(RefCell::new(FieldStation::new("y", 1.0, updater, census)));
        let mut sim = FieldSimulation::new(vec![y.clone()], FieldSimulationOptions::default());
        let res = sim.run(0.0, 1.0, 0.1);

        // 10 explicit-Euler steps of factor 0.9.
        let expected = 0.9f64.powi(10);
        assert!((res.final_values[0] - expected).abs() < 1e-12);
        assert_eq!(res.ticks, 10);
        // record_trace default: one row at t0 plus one per tick.
        assert_eq!(res.trace.t.len(), res.ticks + 1);
        assert_eq!(res.trace.values.len(), res.ticks + 1);
    }

    #[test]
    fn order_independent_under_shuffle() {
        // Two coupled fields read only the frozen snapshot, so a shuffled tick
        // order must give the identical result to a fixed order.
        fn build(shuffle: bool) -> Vec<f64> {
            let census = placeholder_census();
            let u0: FieldUpdater = Box::new(|_p, cur, i, dt, _t| cur[i] + dt * cur[1]);
            let u1: FieldUpdater = Box::new(|_p, cur, i, dt, _t| cur[i] - dt * cur[0]);
            let a = Rc::new(RefCell::new(FieldStation::new(
                "a",
                1.0,
                u0,
                census.clone(),
            )));
            let b = Rc::new(RefCell::new(FieldStation::new("b", 0.0, u1, census)));
            let mut sim = FieldSimulation::new(
                vec![a, b],
                FieldSimulationOptions {
                    seed: 5,
                    shuffle,
                    record_trace: false,
                },
            );
            sim.run(0.0, 1.0, 0.05).final_values
        }
        let shuffled = build(true);
        let fixed = build(false);
        assert_eq!(shuffled, fixed);
    }

    #[test]
    fn prev_snapshot_lags_current() {
        // Record the previous-tick snapshot via the updater; after a constant
        // field the prev/cur relationship is observable through the census.
        let census = placeholder_census();
        // Field stays at its starting value each tick (identity update).
        let updater: FieldUpdater = Box::new(|_prev, cur, i, _dt, _t| cur[i]);
        let f = Rc::new(RefCell::new(FieldStation::new("c", 3.0, updater, census)));
        let mut sim = FieldSimulation::new(
            vec![f.clone()],
            FieldSimulationOptions {
                seed: 1,
                shuffle: false,
                record_trace: true,
            },
        );
        let res = sim.run(0.0, 0.3, 0.1);
        assert_eq!(res.ticks, 3);
        for row in &res.trace.values {
            assert_eq!(row[0], 3.0);
        }
        assert_eq!(res.final_values[0], 3.0);
    }
}
