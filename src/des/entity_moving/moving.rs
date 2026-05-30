//! Canonical use path: `crate::des::entity_moving::moving::*`
//!
//! Port of `src/des/entity-moving/moving.ts` — the "tokens" (customers / jobs)
//! that flow through the queueing network.
//!
//! The TS inheritance chain `Entity -> AbstractMovingEntity -> BasicMovingEntity
//! -> ProcessableMovingEntity` is flattened: shared timing/visit state lives in
//! [`MovingCore`]; shared behaviour lives in the object-safe [`MovingEntity`]
//! trait (a sub-trait of [`Entity`]); concrete tokens compose a core and `impl`
//! both traits.
//!
//! Wall-clock bookkeeping (`Date.now()` in `init`/`doFinish`) uses
//! `SystemTime`/`UNIX_EPOCH` millis — this is real-time accounting, distinct from
//! the simulation clock (`time_accrued::get_time_accrued`). The auto-increment
//! `static nextMovingId` is an `AtomicU64`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::des::general::general::get_short_uuid;
use crate::des::general::time_accrued::get_time_accrued;
use crate::des::r#abstract::interfaces::EntityGraphData;
use crate::des::r#abstract::r#abstract::{Entity, EntityCore};
use crate::des::shared::linked_queue::{is_void, LinkedQueue};
use crate::des::shared::precision::{bgn_int, to_f64, Decimal};

/// `static nextMovingId` — process-wide monotonic id source.
static NEXT_MOVING_ID: AtomicU64 = AtomicU64::new(0);

fn now_ms_i64() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

/// Round a decimal to 5 dp and coerce to `f64`, matching TS `Number(x.toFixed(5))`.
fn round5(d: Decimal) -> f64 {
    to_f64(d.round_dp(5))
}

/// `stationsVisited` value (`{ count }`).
#[derive(Clone, Copy, Debug, Default)]
pub struct VisitCount {
    pub count: u64,
}

/// Erased return of `getValue()` (TS returned `{id, value}` or `{q}`).
///
/// PORT NOTE: the TS `getValue(): V` was generic `any`. The union of the concrete
/// shapes is captured here; each field is optional and populated per token type.
#[derive(Clone, Debug, Default)]
pub struct MovingValue {
    pub id: Option<String>,
    pub value: Option<f64>,
    pub q: Option<f64>,
}

/// Shared field-bag for every moving entity (the data half of
/// `abstract class AbstractMovingEntity`).
pub struct MovingCore {
    pub entity: EntityCore,
    pub moving_id: u64,
    pub moving_uuid: String,
    pub stations_visited_count: u64,
    pub total_wait_time: Decimal,
    pub total_in_process_time: Decimal,
    pub time_in_system: Decimal,
    pub has_exited_system: bool,
    pub real_time_in_system: i64,
    pub out_queue_wait_time: Decimal,
    pub stations_visited: HashMap<String, VisitCount>,
    /// Wall-clock millis (set in `init`); `-1` until started.
    pub start_time: i64,
    /// Wall-clock millis (set in `doFinish`); `-1` until finished.
    pub end_time: i64,
}

impl MovingCore {
    pub fn new(id: String) -> Self {
        MovingCore {
            entity: EntityCore::new(id.clone()),
            moving_id: NEXT_MOVING_ID.fetch_add(1, Ordering::Relaxed),
            moving_uuid: id,
            stations_visited_count: 0,
            total_wait_time: Decimal::ZERO,
            total_in_process_time: Decimal::ZERO,
            time_in_system: Decimal::ZERO,
            has_exited_system: false,
            real_time_in_system: -1,
            out_queue_wait_time: Decimal::ZERO,
            stations_visited: HashMap::new(),
            start_time: -1,
            end_time: -1,
        }
    }
}

/// `abstract class AbstractMovingEntity` — behaviour shared by every token.
/// Object-safe; the TS methods returning `this` return `()`.
pub trait MovingEntity: Entity {
    fn moving_core(&self) -> &MovingCore;
    fn moving_core_mut(&mut self) -> &mut MovingCore;

    /// `abstract getValue(): V`.
    fn get_value(&self) -> MovingValue;

    /// `abstract runFinish()`.
    fn run_finish(&mut self);

    // ── defaults ──────────────────────────────────────────────────────────

    /// `init()` — stamp the wall-clock start time (TS returned `this`).
    fn init(&mut self) {
        self.moving_core_mut().start_time = now_ms_i64();
    }

    /// `bumpTimeInSystem(stepSize)` (TS returned `this`).
    fn bump_time_in_system(&mut self, step_size: Decimal) {
        self.moving_core_mut().time_in_system += step_size;
    }

    /// `doFinish()` — exit the system, recording real elapsed time.
    fn do_finish(&mut self) {
        let end = now_ms_i64();
        {
            let core = self.moving_core_mut();
            core.end_time = end;
            core.real_time_in_system = end - core.start_time;
            core.has_exited_system = true;
        }
        self.run_finish();
    }

    fn start_time(&self) -> i64 {
        self.moving_core().start_time
    }
    fn set_start_time(&mut self, value: i64) {
        self.moving_core_mut().start_time = value;
    }
    fn end_time(&self) -> i64 {
        self.moving_core().end_time
    }
    fn set_end_time(&mut self, value: i64) {
        self.moving_core_mut().end_time = value;
    }

    /// `addVisitedStation(name)`.
    fn add_visited_station(&mut self, name: &str) {
        let entry = self
            .moving_core_mut()
            .stations_visited
            .entry(name.to_string())
            .or_insert(VisitCount { count: 0 });
        entry.count += 1;
    }
}

/// `interface ProcessingTimeValue` — per-station timing record.
///
/// All fields are decimals carrying a `-1` sentinel until set (the TS used a
/// `number` `-1` sentinel that was later overwritten with a `BigNumber`).
#[derive(Clone, Copy, Debug)]
pub struct ProcessingTimeValue {
    pub time_in_input_queue: Decimal,
    pub time_in_process_queue: Decimal,
    pub time_in_output_queue: Decimal,
    pub start_time_in_input_queue: Decimal,
    pub start_time_in_process_queue: Decimal,
    pub start_time_in_output_queue: Decimal,
}

// =============================================================================
// BasicMovingEntity  (TS abstract class, here a concrete token)
// =============================================================================

/// `class BasicMovingEntity` — a plain token (id generated from a short UUID).
pub struct BasicMovingEntity {
    pub core: MovingCore,
}

impl Default for BasicMovingEntity {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicMovingEntity {
    pub fn new() -> Self {
        BasicMovingEntity {
            core: MovingCore::new(get_short_uuid()),
        }
    }
}

impl Entity for BasicMovingEntity {
    fn core(&self) -> &EntityCore {
        &self.core.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        // Object.assign(getWithComputedProperties(), getSerializableData())
        EntityGraphData::default()
            .with("timeInSystem", round5(self.core.time_in_system))
            .with("hasExitedSystem", if self.core.has_exited_system { 1.0 } else { 0.0 })
            .with("totalInProcessTime", round5(self.core.total_in_process_time))
            .with("stationsVisitedCount", self.core.stations_visited_count as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        // TS: `throw new Error('not yet implemented.')`
        panic!("not yet implemented.");
    }
}

impl MovingEntity for BasicMovingEntity {
    fn moving_core(&self) -> &MovingCore {
        &self.core
    }
    fn moving_core_mut(&mut self) -> &mut MovingCore {
        &mut self.core
    }
    fn get_value(&self) -> MovingValue {
        MovingValue {
            id: Some(self.core.entity.id.clone()),
            value: None,
            q: None,
        }
    }
    fn run_finish(&mut self) {
        // TS body is commented out (noop).
    }
}

// =============================================================================
// ProcessableMovingEntity
// =============================================================================

/// `class ProcessableMovingEntity` — a token that records per-station processing
/// times in a keyed [`LinkedQueue`].
pub struct ProcessableMovingEntity {
    pub core: MovingCore,
    pub processing_time_by_station: LinkedQueue<String, ProcessingTimeValue>,
}

impl Default for ProcessableMovingEntity {
    fn default() -> Self {
        Self::new()
    }
}

impl ProcessableMovingEntity {
    pub fn new() -> Self {
        ProcessableMovingEntity {
            core: MovingCore::new(get_short_uuid()),
            processing_time_by_station: LinkedQueue::new(),
        }
    }

    /// `startNewStation(stationId)` — reset and begin timing at a station.
    pub fn start_new_station(&mut self, station_id: &str) {
        let key = station_id.to_string();
        self.processing_time_by_station.remove(&key);

        let current_time = get_time_accrued();
        self.processing_time_by_station.enqueue_keyed(
            key,
            ProcessingTimeValue {
                time_in_input_queue: bgn_int(-1),
                time_in_process_queue: bgn_int(-1),
                time_in_output_queue: bgn_int(-1),
                start_time_in_input_queue: current_time,
                start_time_in_process_queue: bgn_int(-1),
                start_time_in_output_queue: bgn_int(-1),
            },
        );
    }

    pub fn set_start_time_in_process_queue(&mut self, station_id: &str) {
        let key = station_id.to_string();
        {
            let got = self.processing_time_by_station.get(&key);
            if is_void(&got) {
                panic!("missing value: {station_id}");
            }
        }
        let current_time = get_time_accrued();
        let z = self.processing_time_by_station.get_mut(&key).unwrap();
        if z.start_time_in_process_queue != bgn_int(-1) {
            panic!("value should be 0.");
        }
        z.start_time_in_process_queue = current_time;
    }

    pub fn set_start_time_in_output_queue(&mut self, station_id: &str) {
        let key = station_id.to_string();
        {
            let got = self.processing_time_by_station.get(&key);
            if is_void(&got) {
                panic!("missing value: {station_id}");
            }
        }
        let current_time = get_time_accrued();
        let z = self.processing_time_by_station.get_mut(&key).unwrap();
        if z.start_time_in_output_queue != bgn_int(-1) {
            panic!("value should be 0.");
        }
        z.start_time_in_output_queue = current_time;
    }

    pub fn set_time_in_input_queue(&mut self, station_id: &str) -> Decimal {
        let key = station_id.to_string();
        {
            let got = self.processing_time_by_station.get(&key);
            if is_void(&got) {
                panic!("missing value: {station_id}");
            }
        }
        let current_time = get_time_accrued();
        let z = self.processing_time_by_station.get_mut(&key).unwrap();
        if z.time_in_input_queue != bgn_int(-1) {
            panic!("value should be 0.");
        }
        z.time_in_input_queue = current_time - z.start_time_in_input_queue;
        z.time_in_input_queue
    }

    pub fn set_time_in_processing_queue(&mut self, station_id: &str) -> Decimal {
        let key = station_id.to_string();
        {
            let got = self.processing_time_by_station.get(&key);
            if is_void(&got) {
                panic!("missing value: {station_id}");
            }
        }
        let current_time = get_time_accrued();
        let z = self.processing_time_by_station.get_mut(&key).unwrap();
        if z.time_in_process_queue != bgn_int(-1) {
            panic!("value should be 0.");
        }
        z.time_in_process_queue = current_time - z.start_time_in_process_queue;
        z.time_in_process_queue
    }

    /// PORT NOTE: faithfully preserves the TS guard `z.timeInOutputQueue !== 0`
    /// (the other setters guard against `-1`; this one inconsistently checks `0`).
    pub fn set_time_in_output_queue(&mut self, station_id: &str) -> Decimal {
        let key = station_id.to_string();
        {
            let got = self.processing_time_by_station.get(&key);
            if is_void(&got) {
                panic!("missing value: {station_id}");
            }
        }
        let current_time = get_time_accrued();
        let z = self.processing_time_by_station.get_mut(&key).unwrap();
        if z.time_in_output_queue != Decimal::ZERO {
            panic!("value should be 0.");
        }
        z.time_in_output_queue = current_time - z.start_time_in_output_queue;
        z.time_in_output_queue
    }

    /// `bumpTotalWaitTime` (TS returned `this`).
    pub fn bump_total_wait_time(&mut self, in_millis: Decimal) {
        self.core.total_wait_time += in_millis;
    }

    /// `bumpOutQueueWaitTime` (TS returned `this`).
    pub fn bump_out_queue_wait_time(&mut self, in_millis: Decimal) {
        self.core.out_queue_wait_time += in_millis;
    }

    /// `bumpTotalProcessingTime` (TS returned `this`).
    pub fn bump_total_processing_time(&mut self, in_millis: Decimal) {
        self.core.total_in_process_time += in_millis;
    }
}

impl Entity for ProcessableMovingEntity {
    fn core(&self) -> &EntityCore {
        &self.core.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("timeInSystem", round5(self.core.time_in_system))
            .with("hasExitedSystem", if self.core.has_exited_system { 1.0 } else { 0.0 })
            .with("totalInProcessTime", round5(self.core.total_in_process_time))
            .with("stationsVisitedCount", self.core.stations_visited_count as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        panic!("not yet implemented.");
    }
}

impl MovingEntity for ProcessableMovingEntity {
    fn moving_core(&self) -> &MovingCore {
        &self.core
    }
    fn moving_core_mut(&mut self) -> &mut MovingCore {
        &mut self.core
    }
    fn get_value(&self) -> MovingValue {
        MovingValue {
            id: Some(self.core.entity.id.clone()),
            value: None,
            q: None,
        }
    }
    fn run_finish(&mut self) {}
}

// =============================================================================
// BasicQuantityMovingEntity
// =============================================================================

/// `class BasicQuantityMovingEntity` — a token carrying a numeric quantity.
pub struct BasicQuantityMovingEntity {
    pub core: MovingCore,
    pub value: f64,
}

impl BasicQuantityMovingEntity {
    pub fn new(q: f64) -> Self {
        BasicQuantityMovingEntity {
            core: MovingCore::new(get_short_uuid()),
            value: q,
        }
    }
}

impl Entity for BasicQuantityMovingEntity {
    fn core(&self) -> &EntityCore {
        &self.core.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("timeInSystem", round5(self.core.time_in_system))
            .with("hasExitedSystem", if self.core.has_exited_system { 1.0 } else { 0.0 })
            .with("totalInProcessTime", round5(self.core.total_in_process_time))
            .with("stationsVisitedCount", self.core.stations_visited_count as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        panic!("not yet implemented.");
    }
}

impl MovingEntity for BasicQuantityMovingEntity {
    fn moving_core(&self) -> &MovingCore {
        &self.core
    }
    fn moving_core_mut(&mut self) -> &mut MovingCore {
        &mut self.core
    }
    fn get_value(&self) -> MovingValue {
        MovingValue {
            id: None,
            value: None,
            q: Some(self.value),
        }
    }
    fn run_finish(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::general::time_accrued::{reset_global_clock, set_step_size};
    use crate::des::shared::precision::bgn;

    #[test]
    fn moving_ids_are_unique_and_monotonic() {
        let a = BasicMovingEntity::new();
        let b = BasicMovingEntity::new();
        assert!(b.core.moving_id > a.core.moving_id);
    }

    #[test]
    fn init_and_visited_stations() {
        let mut m = BasicMovingEntity::new();
        m.init();
        assert!(m.start_time() >= 0);
        m.add_visited_station("s1");
        m.add_visited_station("s1");
        m.add_visited_station("s2");
        assert_eq!(m.core.stations_visited.get("s1").unwrap().count, 2);
        assert_eq!(m.core.stations_visited.get("s2").unwrap().count, 1);
    }

    #[test]
    fn bump_time_in_system_accumulates() {
        let mut m = BasicMovingEntity::new();
        m.bump_time_in_system(bgn(0.1));
        m.bump_time_in_system(bgn(0.2));
        assert_eq!(m.core.time_in_system, bgn(0.3));
    }

    #[test]
    fn processable_station_timing() {
        reset_global_clock();
        set_step_size(bgn(0.0));
        let mut p = ProcessableMovingEntity::new();
        p.start_new_station("stationA");
        // start time in process queue is the (zero) accrued clock; first set succeeds.
        p.set_start_time_in_process_queue("stationA");
        let got = p.processing_time_by_station.get(&"stationA".to_string());
        assert!(got.is_some());
        p.bump_total_processing_time(bgn(1.5));
        assert_eq!(p.core.total_in_process_time, bgn(1.5));
    }

    #[test]
    fn quantity_entity_get_value() {
        let q = BasicQuantityMovingEntity::new(42.0);
        assert_eq!(q.get_value().q, Some(42.0));
    }

    #[test]
    #[should_panic(expected = "missing value")]
    fn set_time_missing_station_panics() {
        let mut p = ProcessableMovingEntity::new();
        p.set_time_in_input_queue("nope");
    }
}
