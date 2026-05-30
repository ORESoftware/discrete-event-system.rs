//! Port of `src/des/general/factory-floor-track3t.ts` — module
//! `des::general::factory_floor_track3t`.
//!
//! A warehouse / factory-floor comparison model grounded in archived Track3t
//! product claims (continuous indoor material tracking, high ID/location
//! accuracy, cloud analytics, fewer production/shipping errors). It contrasts a
//! conventional WMS floor against a Track3t-enabled floor under an identical
//! QMDP/POMDP routing planner.
//!
//! Framework mapping:
//!   * `WarehouseSource` emits movable `WarehousePallet` jobs.
//!   * `WarehouseStation` / `WarehouseSink` are stationary floor entities.
//!   * `WarehouseForklift` is a `SmartMovable` driven by a POMDP/QMDP planner.
//!
//! Decision model: the hidden state is `(forklift position, pallet true
//! location, carrying flag)`; an action drives to a stationary entity; QMDP
//! value iteration solves the fully-observable relaxation, and noisy sensor
//! observations update a belief over hidden pallet locations.
//!
//! ## Conversion notes (per the TS "RUST MIGRATION" header)
//!
//!   * `WarehouseStationKind` / observation & action kind unions become enums.
//!   * `mulberry32(seed)` becomes the injected [`RandomSource`] (`SeededRandom`).
//!   * JS `Map` / `Set` over ids become `HashMap` / `HashSet`.
//!   * The TS station/movable inheritance becomes trait + composition:
//!     `WarehouseStation` embeds a [`StationCore`] and implements [`DESStation`];
//!     `WarehouseSource` / `WarehouseSink` compose a `WarehouseStation`;
//!     `WarehouseForklift` embeds a [`SmartMovableCore`] and implements
//!     [`SmartMovable`].
//!   * `WarehousePallet` is shared mutable state (queued, carried, collected)
//!     so it flows as `Rc<RefCell<WarehousePallet>>`, mirroring the TS object
//!     reference semantics.
//!   * The `WarehousePOMDPModel` callbacks (`encode/decode/nextState`) become
//!     methods delegating to free functions; the `POMDPSpec` transition /
//!     observation / reward callbacks become boxed closures over cloned layout /
//!     scenario data (they cannot borrow the owning model).
//!   * `Preconditions.*` throw in TS; here the `require` helper turns a failed
//!     guard into a `panic!`, and the explicit `throw new Error(...)` structural
//!     failures become `panic!`.
//!
//! FLAGGED divergence: [`build_warehouse_floor`]'s returned `stations` map holds
//! independent `WarehouseStation` instances (rather than aliasing the same
//! source/sink objects as the TS `Map` does). The simulation never reads that
//! map, so the difference is unobservable.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use serde::Serialize;

use crate::des::general::belief::DiscreteBelief;
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::smart_movable::{SmartMovable, SmartMovableCore};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::pomdp::{belief_update, POMDPSpec};
use crate::des::general::prng::mulberry32;
use crate::des::shared::capabilities::RandomSource;

/// Panic on a failed precondition (the TS guards `throw`).
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// Archive grounding
// =============================================================================

/// One archived Track3t source note.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ArchiveNote {
    pub label: &'static str,
    pub url: &'static str,
    pub model_use: &'static str,
}

/// The `TRACK3T_ARCHIVE_GROUNDING` provenance list.
pub fn track3t_archive_grounding() -> Vec<ArchiveNote> {
    vec![
        ArchiveNote {
            label: "Track3t archived home, 2018-03-31",
            url: "https://web.archive.org/web/20180331125107/https://www.track3t.com/",
            model_use: "Motivates material-flow visibility, lower transit time, fewer bottlenecks, and fewer production/shipping errors.",
        },
        ArchiveNote {
            label: "Track3t archived solution page, 2018-08-15",
            url: "https://web.archive.org/web/20180815170011/https://www.track3t.com/the-solution",
            model_use: "Motivates high location accuracy, high ID accuracy, continuous sensing, dashboards, forensics, and predictive analytics.",
        },
        ArchiveNote {
            label: "Track3t archived about page, 2018-08-15",
            url: "https://web.archive.org/web/20180815174909/https://www.track3t.com/about-us/",
            model_use: "Motivates RFID/wireless/cloud architecture and the move beyond dock-gate-only RFID observations.",
        },
    ]
}

// =============================================================================
// Data shapes
// =============================================================================

/// `'source' | 'storage' | 'aisle' | 'sink'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum WarehouseStationKind {
    Source,
    Storage,
    Aisle,
    Sink,
}

impl WarehouseStationKind {
    pub fn as_str(self) -> &'static str {
        match self {
            WarehouseStationKind::Source => "source",
            WarehouseStationKind::Storage => "storage",
            WarehouseStationKind::Aisle => "aisle",
            WarehouseStationKind::Sink => "sink",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct StationDefinition {
    pub id: String,
    pub label: String,
    pub kind: WarehouseStationKind,
    pub x: f64,
    pub y: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub can_hold_pallet: Option<bool>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WarehouseLayout {
    pub stations: Vec<StationDefinition>,
    pub source_station_id: String,
    pub sink_station_ids: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_meters: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub route_edges: Option<Vec<(String, String)>>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WarehouseScenarioConfig {
    pub id: String,
    pub label: String,
    pub location_accuracy: f64,
    pub id_accuracy: f64,
    pub initial_misplacement_probability: f64,
    pub placement_error_probability: f64,
    pub forklift_speed_meters_per_minute: f64,
    pub route_inflation: f64,
    pub handling_minutes: f64,
    pub confirmation_delay_minutes: f64,
    pub search_penalty_minutes: f64,
    pub rework_penalty_minutes: f64,
    pub delivery_reward: f64,
    pub wrong_delivery_penalty: f64,
    pub discount: f64,
    pub qmdp_tol: f64,
    pub qmdp_max_iter: usize,
    pub due_minutes: f64,
    pub sensor_refresh_seconds: f64,
}

#[derive(Clone, Debug, Default)]
pub struct WarehouseSimulationOptions {
    pub jobs: Option<usize>,
    pub seed: Option<u32>,
    pub max_steps_per_job: Option<usize>,
    pub layout: Option<WarehouseLayout>,
    pub record_trace: Option<bool>,
    pub destination_plan: Option<Vec<String>>,
}

/// Action `kind` discriminator (only `'go-to'` exists).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarehouseActionKind {
    GoTo,
}

#[derive(Clone, Debug)]
pub struct WarehouseAction {
    pub kind: WarehouseActionKind,
    pub target: usize,
    pub label: String,
}

/// Observation `kind` (`'location' | 'carrying' | 'complete'`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarehouseObsKind {
    Location,
    Carrying,
    Complete,
}

#[derive(Clone, Debug)]
pub struct WarehouseObservation {
    pub kind: WarehouseObsKind,
    pub station: Option<usize>,
    pub label: String,
}

/// Decoded hidden state. `forklift`/`pallet` are `-1` in the terminal state.
#[derive(Clone, Copy, Debug)]
pub struct WarehouseDecisionState {
    pub forklift: i64,
    pub pallet: i64,
    pub carrying: bool,
    pub terminal: bool,
}

/// Trace event kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum WarehouseEvent {
    SearchMiss,
    Pickup,
    MoveLoaded,
    Delivered,
    DeliveryError,
    Failed,
}

impl WarehouseEvent {
    pub fn as_str(self) -> &'static str {
        match self {
            WarehouseEvent::SearchMiss => "search-miss",
            WarehouseEvent::Pickup => "pickup",
            WarehouseEvent::MoveLoaded => "move-loaded",
            WarehouseEvent::Delivered => "delivered",
            WarehouseEvent::DeliveryError => "delivery-error",
            WarehouseEvent::Failed => "failed",
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WarehouseStepTrace {
    pub scenario_id: String,
    pub job_id: String,
    pub job_index: usize,
    pub step: usize,
    pub time_start: f64,
    pub time_end: f64,
    pub action: String,
    pub action_target: String,
    pub observation: String,
    pub event: WarehouseEvent,
    pub destination: String,
    pub forklift_before: String,
    pub forklift_after: String,
    pub pallet_before: String,
    pub pallet_after: String,
    pub carrying_before: bool,
    pub carrying_after: bool,
    pub belief_entropy: f64,
    pub belief_by_station: Vec<f64>,
    pub cumulative_delivered: usize,
    pub cumulative_errors: usize,
    pub cumulative_search_misses: usize,
    pub cycle_time_so_far: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WarehouseJobSummary {
    pub job_id: String,
    pub destination: String,
    pub completed: bool,
    pub shipping_error: bool,
    pub cycle_time: f64,
    pub steps: usize,
    pub search_misses: usize,
    pub on_time: bool,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WarehouseMetrics {
    pub jobs_created: usize,
    pub completed_jobs: usize,
    pub failed_jobs: usize,
    pub shipping_errors: usize,
    pub shipping_error_rate: f64,
    pub total_time: f64,
    pub mean_cycle_time: f64,
    pub throughput_per_hour: f64,
    pub on_time_rate: f64,
    pub mean_steps_per_job: f64,
    pub mean_search_misses_per_job: f64,
    pub mean_belief_entropy: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WarehouseScenarioResult {
    pub scenario: WarehouseScenarioConfig,
    pub layout: WarehouseLayout,
    pub metrics: WarehouseMetrics,
    pub jobs: Vec<WarehouseJobSummary>,
    pub trace: Vec<WarehouseStepTrace>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComparisonDeltas {
    pub mean_cycle_time_reduction_pct: f64,
    pub throughput_lift_pct: f64,
    pub search_miss_reduction_pct: f64,
    pub error_reduction_pct: f64,
    pub entropy_reduction_pct: f64,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WarehouseComparisonResult {
    pub layout: WarehouseLayout,
    pub baseline: WarehouseScenarioResult,
    pub track3t: WarehouseScenarioResult,
    pub deltas: ComparisonDeltas,
    pub source_notes: Vec<ArchiveNote>,
}

// =============================================================================
// Movable load + floor entities
// =============================================================================

#[derive(Clone, Debug)]
pub struct WarehousePallet {
    pub id: String,
    pub destination_id: String,
    pub location_id: String,
    pub created_at: f64,
}

/// A stationary floor entity (the TS `WarehouseStation extends DESStation`).
pub struct WarehouseStation {
    core: StationCore,
    pub def: StationDefinition,
    pub queue: Vec<Rc<RefCell<WarehousePallet>>>,
}

impl WarehouseStation {
    fn new(def: StationDefinition) -> Self {
        let id = def.id.clone();
        WarehouseStation {
            core: StationCore::new(id),
            def,
            queue: Vec::new(),
        }
    }

    fn receive(&mut self, pallet: Rc<RefCell<WarehousePallet>>) {
        pallet.borrow_mut().location_id = self.def.id.clone();
        self.queue.push(pallet);
    }

    fn remove(&mut self, pallet: &Rc<RefCell<WarehousePallet>>) -> bool {
        let target_id = pallet.borrow().id.clone();
        if let Some(idx) = self.queue.iter().position(|p| p.borrow().id == target_id) {
            self.queue.remove(idx);
            true
        } else {
            false
        }
    }
}

impl DESStation for WarehouseStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn run_time_step(&mut self) {}
}

/// A pallet source (the TS `WarehouseSource extends WarehouseStation`).
pub struct WarehouseSource {
    pub station: WarehouseStation,
}

impl WarehouseSource {
    fn new(def: StationDefinition) -> Self {
        WarehouseSource {
            station: WarehouseStation::new(def),
        }
    }

    fn emit_pallet(
        &mut self,
        id: String,
        destination_id: String,
        created_at: f64,
    ) -> Rc<RefCell<WarehousePallet>> {
        let pallet = Rc::new(RefCell::new(WarehousePallet {
            id,
            destination_id,
            location_id: self.station.def.id.clone(),
            created_at,
        }));
        self.station.receive(pallet.clone());
        pallet
    }
}

/// A collected pallet record.
#[derive(Clone, Debug)]
pub struct CollectedPallet {
    pub pallet: Rc<RefCell<WarehousePallet>>,
    pub time: f64,
    pub correct: bool,
}

/// A delivery sink (the TS `WarehouseSink extends WarehouseStation`).
pub struct WarehouseSink {
    pub station: WarehouseStation,
    pub collected: Vec<CollectedPallet>,
}

impl WarehouseSink {
    fn new(def: StationDefinition) -> Self {
        WarehouseSink {
            station: WarehouseStation::new(def),
            collected: Vec::new(),
        }
    }

    fn collect(&mut self, pallet: Rc<RefCell<WarehousePallet>>, time: f64, correct: bool) {
        pallet.borrow_mut().location_id = self.station.def.id.clone();
        self.collected.push(CollectedPallet {
            pallet,
            time,
            correct,
        });
    }
}

/// A planner-driven forklift (the TS `WarehouseForklift extends SmartMovable`).
pub struct WarehouseForklift {
    core: SmartMovableCore,
    pub station_id: String,
    pub carrying: Option<Rc<RefCell<WarehousePallet>>>,
}

impl WarehouseForklift {
    fn new(id: String, station_id: String) -> Self {
        WarehouseForklift {
            core: SmartMovableCore::new(id),
            station_id,
            carrying: None,
        }
    }

    fn move_to(&mut self, station_id: &str) {
        self.station_id = station_id.to_string();
        if let Some(p) = &self.carrying {
            p.borrow_mut().location_id = station_id.to_string();
        }
    }

    fn pickup(&mut self, pallet: Rc<RefCell<WarehousePallet>>) {
        pallet.borrow_mut().location_id = self.station_id.clone();
        self.carrying = Some(pallet);
    }

    fn drop_pallet(&mut self) -> Option<Rc<RefCell<WarehousePallet>>> {
        self.carrying.take()
    }
}

impl SmartMovable for WarehouseForklift {
    fn core(&self) -> &SmartMovableCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut SmartMovableCore {
        &mut self.core
    }
    fn run_time_step(&mut self) {}
}

/// The constructed floor (the TS `buildWarehouseFloor` return shape).
pub struct WarehouseFloor {
    pub source: WarehouseSource,
    pub sinks: HashMap<String, WarehouseSink>,
    pub stations: HashMap<String, WarehouseStation>,
}

// =============================================================================
// Default layout + scenarios
// =============================================================================

pub fn default_warehouse_layout() -> WarehouseLayout {
    let mut stations: Vec<StationDefinition> = vec![
        StationDefinition {
            id: "receiving".into(),
            label: "Recv".into(),
            kind: WarehouseStationKind::Source,
            x: 0.0,
            y: 3.0,
            can_hold_pallet: Some(true),
        },
        StationDefinition {
            id: "staging".into(),
            label: "Stage".into(),
            kind: WarehouseStationKind::Storage,
            x: 2.0,
            y: 3.0,
            can_hold_pallet: Some(true),
        },
    ];
    let mut route_edges: Vec<(String, String)> = vec![("receiving".into(), "staging".into())];
    let row_names = ["a", "b", "c", "d"];
    for (r, row) in row_names.iter().enumerate() {
        let y = (r * 2) as f64;
        let mut prev_id = "staging".to_string();
        for c in 1..=3 {
            let id = format!("reserve-{row}{c}");
            stations.push(StationDefinition {
                id: id.clone(),
                label: format!("{}{}", row.to_uppercase(), c),
                kind: WarehouseStationKind::Storage,
                x: 4.0 + (c - 1) as f64 * 2.0,
                y,
                can_hold_pallet: Some(true),
            });
            route_edges.push((prev_id.clone(), id.clone()));
            prev_id = id;
        }
    }
    stations.push(StationDefinition {
        id: "aisle-main".into(),
        label: "Aisle".into(),
        kind: WarehouseStationKind::Aisle,
        x: 10.0,
        y: 3.0,
        can_hold_pallet: Some(true),
    });
    stations.push(StationDefinition {
        id: "line-a".into(),
        label: "Line A".into(),
        kind: WarehouseStationKind::Sink,
        x: 12.0,
        y: 0.0,
        can_hold_pallet: None,
    });
    stations.push(StationDefinition {
        id: "line-b".into(),
        label: "Line B".into(),
        kind: WarehouseStationKind::Sink,
        x: 12.0,
        y: 2.0,
        can_hold_pallet: None,
    });
    stations.push(StationDefinition {
        id: "line-c".into(),
        label: "Line C".into(),
        kind: WarehouseStationKind::Sink,
        x: 12.0,
        y: 4.0,
        can_hold_pallet: None,
    });
    stations.push(StationDefinition {
        id: "shipping".into(),
        label: "Ship".into(),
        kind: WarehouseStationKind::Sink,
        x: 12.0,
        y: 6.0,
        can_hold_pallet: None,
    });
    for row in row_names {
        route_edges.push((format!("reserve-{row}3"), "aisle-main".into()));
    }
    route_edges.push(("aisle-main".into(), "line-a".into()));
    route_edges.push(("aisle-main".into(), "line-b".into()));
    route_edges.push(("aisle-main".into(), "line-c".into()));
    route_edges.push(("aisle-main".into(), "shipping".into()));
    WarehouseLayout {
        source_station_id: "receiving".into(),
        sink_station_ids: vec![
            "line-a".into(),
            "line-b".into(),
            "line-c".into(),
            "shipping".into(),
        ],
        grid_meters: Some(12.0),
        stations,
        route_edges: Some(route_edges),
    }
}

pub fn baseline_warehouse_scenario() -> WarehouseScenarioConfig {
    WarehouseScenarioConfig {
        id: "baseline".into(),
        label: "Conventional WMS / manual lookup".into(),
        location_accuracy: 0.64,
        id_accuracy: 0.94,
        initial_misplacement_probability: 0.24,
        placement_error_probability: 0.12,
        forklift_speed_meters_per_minute: 72.0,
        route_inflation: 1.26,
        handling_minutes: 1.8,
        confirmation_delay_minutes: 2.2,
        search_penalty_minutes: 6.5,
        rework_penalty_minutes: 18.0,
        delivery_reward: 110.0,
        wrong_delivery_penalty: 65.0,
        discount: 0.96,
        qmdp_tol: 1e-6,
        qmdp_max_iter: 1200,
        due_minutes: 22.0,
        sensor_refresh_seconds: 900.0,
    }
}

pub fn track3t_warehouse_scenario() -> WarehouseScenarioConfig {
    WarehouseScenarioConfig {
        id: "track3t".into(),
        label: "Track3t-enabled floor".into(),
        location_accuracy: 0.985,
        id_accuracy: 0.999,
        initial_misplacement_probability: 0.055,
        placement_error_probability: 0.02,
        forklift_speed_meters_per_minute: 78.0,
        route_inflation: 0.94,
        handling_minutes: 1.5,
        confirmation_delay_minutes: 0.25,
        search_penalty_minutes: 0.9,
        rework_penalty_minutes: 18.0,
        delivery_reward: 110.0,
        wrong_delivery_penalty: 65.0,
        discount: 0.96,
        qmdp_tol: 1e-6,
        qmdp_max_iter: 1200,
        due_minutes: 22.0,
        sensor_refresh_seconds: 0.5,
    }
}

// =============================================================================
// POMDP model
// =============================================================================

/// The QMDP/POMDP routing model for one destination.
pub struct WarehousePOMDPModel {
    pub layout: WarehouseLayout,
    pub scenario: WarehouseScenarioConfig,
    pub destination_index: usize,
    pub states: Vec<usize>,
    pub actions: Vec<WarehouseAction>,
    pub observations: Vec<WarehouseObservation>,
    pub terminal_state: usize,
    pub n: usize,
    pub spec: POMDPSpec<usize, WarehouseAction, WarehouseObservation>,
}

impl WarehousePOMDPModel {
    pub fn encode_state(&self, forklift: usize, pallet: usize, carrying: bool) -> usize {
        encode_state_raw(forklift, pallet, carrying, self.n)
    }
    pub fn decode_state(&self, state_id: usize) -> WarehouseDecisionState {
        decode_state_raw(state_id, self.n, self.terminal_state)
    }
    pub fn next_state(&self, state_id: usize, action_idx: usize) -> usize {
        next_state_raw(
            state_id,
            action_idx,
            &self.actions,
            self.destination_index,
            self.n,
            self.terminal_state,
        )
    }
    pub fn observation_index_for_location(&self, station_idx: usize) -> usize {
        station_idx
    }
}

fn encode_state_raw(forklift: usize, pallet: usize, carrying: bool, n: usize) -> usize {
    require(Preconditions::integer_in_range(
        "WarehousePOMDP",
        "forklift",
        forklift as f64,
        0.0,
        (n - 1) as f64,
    ));
    require(Preconditions::integer_in_range(
        "WarehousePOMDP",
        "pallet",
        pallet as f64,
        0.0,
        (n - 1) as f64,
    ));
    (forklift * n + pallet) * 2 + if carrying { 1 } else { 0 }
}

fn decode_state_raw(state_id: usize, n: usize, terminal_state: usize) -> WarehouseDecisionState {
    if state_id == terminal_state {
        return WarehouseDecisionState {
            forklift: -1,
            pallet: -1,
            carrying: false,
            terminal: true,
        };
    }
    let carrying = state_id % 2 == 1;
    let mut rest = state_id / 2;
    let pallet = (rest % n) as i64;
    rest /= n;
    let forklift = rest as i64;
    WarehouseDecisionState {
        forklift,
        pallet,
        carrying,
        terminal: false,
    }
}

fn next_state_raw(
    state_id: usize,
    action_idx: usize,
    actions: &[WarehouseAction],
    destination_index: usize,
    n: usize,
    terminal_state: usize,
) -> usize {
    if state_id == terminal_state {
        return terminal_state;
    }
    let s = decode_state_raw(state_id, n, terminal_state);
    let target = actions[action_idx].target;
    if s.carrying {
        if target == destination_index {
            return terminal_state;
        }
        return encode_state_raw(target, target, true, n);
    }
    if target as i64 == s.pallet {
        return encode_state_raw(target, target, true, n);
    }
    encode_state_raw(target, s.pallet as usize, false, n)
}

pub fn build_warehouse_pomdp(
    layout: &WarehouseLayout,
    scenario: &WarehouseScenarioConfig,
    destination_index: usize,
) -> WarehousePOMDPModel {
    validate_layout(layout);
    validate_scenario(scenario);
    require(Preconditions::integer_in_range(
        "WarehousePOMDP",
        "destinationIndex",
        destination_index as f64,
        0.0,
        (layout.stations.len() - 1) as f64,
    ));
    let n = layout.stations.len();
    let terminal_state = n * n * 2;
    let states: Vec<usize> = (0..=terminal_state).collect();
    let actions: Vec<WarehouseAction> = layout
        .stations
        .iter()
        .enumerate()
        .map(|(target, s)| WarehouseAction {
            kind: WarehouseActionKind::GoTo,
            target,
            label: format!("go to {}", s.label),
        })
        .collect();
    let mut observations: Vec<WarehouseObservation> = layout
        .stations
        .iter()
        .enumerate()
        .map(|(station, s)| WarehouseObservation {
            kind: WarehouseObsKind::Location,
            station: Some(station),
            label: format!("sensor says {}", s.label),
        })
        .collect();
    observations.push(WarehouseObservation {
        kind: WarehouseObsKind::Carrying,
        station: None,
        label: "forklift carrying pallet".into(),
    });
    observations.push(WarehouseObservation {
        kind: WarehouseObsKind::Complete,
        station: None,
        label: "delivery complete".into(),
    });

    let carrying_obs_idx = n;
    let complete_obs_idx = n + 1;
    let num_states = states.len();
    let observations_len = observations.len();

    let actions_t = actions.clone();
    let transition: Box<dyn Fn(usize, usize) -> Vec<f64>> = Box::new(move |s_idx, a_idx| {
        let mut row = vec![0.0; num_states];
        row[next_state_raw(
            s_idx,
            a_idx,
            &actions_t,
            destination_index,
            n,
            terminal_state,
        )] = 1.0;
        row
    });

    let scenario_o = scenario.clone();
    let observation: Box<dyn Fn(usize, usize) -> Vec<f64>> = Box::new(move |s_next_idx, _a_idx| {
        let mut row = vec![0.0; observations_len];
        if s_next_idx == terminal_state {
            row[complete_obs_idx] = 1.0;
            return row;
        }
        let s = decode_state_raw(s_next_idx, n, terminal_state);
        if s.carrying {
            row[carrying_obs_idx] = 1.0;
            return row;
        }
        let wrong_mass = (1.0 - scenario_o.location_accuracy) / ((n - 1).max(1) as f64);
        for (i, slot) in row.iter_mut().enumerate().take(n) {
            *slot = if i as i64 == s.pallet {
                scenario_o.location_accuracy
            } else {
                wrong_mass
            };
        }
        row
    });

    let layout_r = layout.clone();
    let scenario_r = scenario.clone();
    let actions_r = actions.clone();
    let reward: Box<dyn Fn(usize, usize) -> f64> = Box::new(move |s_idx, a_idx| {
        if s_idx == terminal_state {
            return 0.0;
        }
        let s = decode_state_raw(s_idx, n, terminal_state);
        let target = actions_r[a_idx].target;
        let travel = travel_minutes(&layout_r, &scenario_r, s.forklift as usize, target);
        let confirm = scenario_r.confirmation_delay_minutes;
        if s.carrying && target == destination_index {
            return scenario_r.delivery_reward
                - travel
                - scenario_r.handling_minutes
                - confirm
                - scenario_r.placement_error_probability * scenario_r.wrong_delivery_penalty;
        }
        if s.carrying {
            return -travel - 0.25 * scenario_r.handling_minutes;
        }
        if target as i64 == s.pallet {
            return -travel - scenario_r.handling_minutes - confirm;
        }
        -travel - scenario_r.search_penalty_minutes - confirm
    });

    let spec: POMDPSpec<usize, WarehouseAction, WarehouseObservation> = POMDPSpec {
        states: states.clone(),
        actions: actions.clone(),
        observations: observations.clone(),
        transition,
        observation,
        reward,
        discount: scenario.discount,
        initial_belief: None,
        is_terminal: Some(Box::new(move |s_idx| s_idx == terminal_state)),
    };

    WarehousePOMDPModel {
        layout: layout.clone(),
        scenario: scenario.clone(),
        destination_index,
        states,
        actions,
        observations,
        terminal_state,
        n,
        spec,
    }
}

/// QMDP solver: value iteration on the fully-observable relaxation.
pub struct WarehouseQMDPSolver {
    pub q: Vec<Vec<f64>>,
    pub iterations: usize,
    pub final_delta: f64,
}

impl WarehouseQMDPSolver {
    pub fn new(model: &WarehousePOMDPModel) -> Self {
        let k = model.states.len();
        let a_count = model.actions.len();
        let gamma = model.scenario.discount;
        let max_iter = model.scenario.qmdp_max_iter;
        let tol = model.scenario.qmdp_tol;
        let mut v = vec![0.0; k];
        let mut iterations = 0;
        let mut final_delta = f64::INFINITY;
        for iter in 0..max_iter {
            let mut next = vec![0.0; k];
            let mut delta = 0.0;
            for s in 0..k {
                if s == model.terminal_state {
                    continue;
                }
                let mut best = f64::NEG_INFINITY;
                for a in 0..a_count {
                    let sp = model.next_state(s, a);
                    let q = (model.spec.reward)(s, a) + gamma * v[sp];
                    if q > best {
                        best = q;
                    }
                }
                next[s] = best;
                let d = (next[s] - v[s]).abs();
                if d > delta {
                    delta = d;
                }
            }
            v = next;
            iterations = iter + 1;
            final_delta = delta;
            if delta <= tol {
                break;
            }
        }

        let mut q = vec![vec![0.0; a_count]; k];
        for s in 0..k {
            if s == model.terminal_state {
                continue;
            }
            for a in 0..a_count {
                let sp = model.next_state(s, a);
                q[s][a] = (model.spec.reward)(s, a) + gamma * v[sp];
            }
        }
        WarehouseQMDPSolver {
            q,
            iterations,
            final_delta,
        }
    }

    pub fn act(&self, belief: &DiscreteBelief<usize>, rng: &mut dyn RandomSource) -> usize {
        let a_count = self.q.first().map_or(0, |r| r.len());
        let mut best_a = 0;
        let mut best_q = f64::NEG_INFINITY;
        let mut ties = 0;
        for a in 0..a_count {
            let mut q = 0.0;
            for s in 0..belief.weights.len() {
                let w = belief.weights[s];
                if w != 0.0 {
                    q += w * self.q[s][a];
                }
            }
            if q > best_q + 1e-12 {
                best_a = a;
                best_q = q;
                ties = 1;
            } else if (q - best_q).abs() <= 1e-12 {
                ties += 1;
                if rng.next_float() < 1.0 / ties as f64 {
                    best_a = a;
                }
            }
        }
        best_a
    }
}

/// A built+solved plan for one destination.
pub struct WarehousePlan {
    pub model: WarehousePOMDPModel,
    pub solver: WarehouseQMDPSolver,
}

/// Caches one [`WarehousePlan`] per destination index.
pub struct WarehousePlanner {
    layout: WarehouseLayout,
    scenario: WarehouseScenarioConfig,
    cache: HashMap<usize, Rc<WarehousePlan>>,
}

/// The result of [`WarehousePlanner::choose_action`].
pub struct ChooseActionResult {
    pub plan: Rc<WarehousePlan>,
    pub action_index: usize,
    pub action: WarehouseAction,
}

impl WarehousePlanner {
    pub fn new(layout: WarehouseLayout, scenario: WarehouseScenarioConfig) -> Self {
        WarehousePlanner {
            layout,
            scenario,
            cache: HashMap::new(),
        }
    }

    pub fn for_destination(&mut self, destination_index: usize) -> Rc<WarehousePlan> {
        if let Some(cached) = self.cache.get(&destination_index) {
            return cached.clone();
        }
        let model = build_warehouse_pomdp(&self.layout, &self.scenario, destination_index);
        let solver = WarehouseQMDPSolver::new(&model);
        let plan = Rc::new(WarehousePlan { model, solver });
        self.cache.insert(destination_index, plan.clone());
        plan
    }

    pub fn choose_action(
        &mut self,
        destination_index: usize,
        belief: &DiscreteBelief<usize>,
        rng: &mut dyn RandomSource,
    ) -> ChooseActionResult {
        let plan = self.for_destination(destination_index);
        let action_index = plan.solver.act(belief, rng);
        let action = plan.model.actions[action_index].clone();
        ChooseActionResult {
            plan,
            action_index,
            action,
        }
    }
}

// =============================================================================
// Simulation
// =============================================================================

pub fn simulate_warehouse_scenario(
    scenario: WarehouseScenarioConfig,
    opts: WarehouseSimulationOptions,
) -> WarehouseScenarioResult {
    let layout = opts.layout.clone().unwrap_or_else(default_warehouse_layout);
    validate_layout(&layout);
    validate_scenario(&scenario);
    let jobs = opts.jobs.unwrap_or(120);
    let seed = opts.seed.unwrap_or(7);
    let max_steps_per_job = opts.max_steps_per_job.unwrap_or(24);
    require(Preconditions::integer_in_range(
        "WarehouseSimulation",
        "jobs",
        jobs as f64,
        1.0,
        10000.0,
    ));
    require(Preconditions::integer_in_range(
        "WarehouseSimulation",
        "maxStepsPerJob",
        max_steps_per_job as f64,
        1.0,
        200.0,
    ));

    let mut floor = build_warehouse_floor(&layout);
    let mut rng = mulberry32(seed);
    let mut planner = WarehousePlanner::new(layout.clone(), scenario.clone());
    let mut forklift = WarehouseForklift::new(
        format!("{}-forklift-1", scenario.id),
        layout.source_station_id.clone(),
    );
    forklift.activate();
    let station_to_index = station_index_map(&layout);
    let source_index = *station_to_index
        .get(&layout.source_station_id)
        .expect("source index present");
    let sink_plan = opts
        .destination_plan
        .clone()
        .unwrap_or_else(|| make_destination_plan(&layout, jobs, &mut rng));
    let record_trace = opts.record_trace.unwrap_or(true);

    let mut now = 0.0_f64;
    let mut completed_jobs = 0usize;
    let mut failed_jobs = 0usize;
    let mut shipping_errors = 0usize;
    let mut cumulative_search_misses = 0usize;
    let mut entropy_sum = 0.0_f64;
    let mut entropy_count = 0usize;
    let mut job_summaries: Vec<WarehouseJobSummary> = Vec::new();
    let mut trace: Vec<WarehouseStepTrace> = Vec::new();

    for j in 0..jobs {
        let destination_id = sink_plan[j % sink_plan.len()].clone();
        let destination_index = *station_to_index
            .get(&destination_id)
            .unwrap_or_else(|| panic!("unknown destination in plan: {destination_id}"));
        let job_id = format!("{}-p{}", scenario.id, j + 1);
        let pallet = floor
            .source
            .emit_pallet(job_id.clone(), destination_id.clone(), now);
        let initial_pallet_index = sample_initial_pallet_location(&layout, &scenario, &mut rng);
        pallet.borrow_mut().location_id = layout.stations[initial_pallet_index].id.clone();
        let job_start = now;
        let initial_forklift_index = station_to_index
            .get(&forklift.station_id)
            .copied()
            .unwrap_or(source_index);

        let plan = planner.for_destination(destination_index);
        let model = &plan.model;
        let mut actual_state =
            model.encode_state(initial_forklift_index, initial_pallet_index, false);
        let mut observed_location =
            sample_location_observation(&layout, &scenario, initial_pallet_index, &mut rng);
        let mut belief = initial_warehouse_belief(
            &layout,
            &scenario,
            model,
            initial_forklift_index,
            observed_location,
        );
        let mut search_misses = 0usize;
        let mut completed = false;
        let mut shipping_error = false;
        let mut steps_taken = 0usize;

        for step in 0..max_steps_per_job {
            steps_taken = step + 1;
            let before = model.decode_state(actual_state);
            let before_station_id = if before.terminal {
                forklift.station_id.clone()
            } else {
                layout.stations[before.forklift as usize].id.clone()
            };
            let pallet_before_id = if before.terminal {
                destination_id.clone()
            } else {
                layout.stations[before.pallet as usize].id.clone()
            };

            let car = planner.choose_action(destination_index, &belief, &mut rng);
            let action_index = car.action_index;
            let action = car.action.clone();
            let target = action.target;
            let next_state_id = model.next_state(actual_state, action_index);
            let after = model.decode_state(next_state_id);
            let duration =
                action_duration_minutes(&layout, &scenario, &before, target, destination_index);
            let time_start = now;
            now += duration;

            let event: WarehouseEvent;
            if before.carrying && target == destination_index {
                completed = true;
                completed_jobs += 1;
                shipping_error = rng.next_float() > scenario.id_accuracy
                    || rng.next_float() < scenario.placement_error_probability;
                if shipping_error {
                    shipping_errors += 1;
                    now += scenario.rework_penalty_minutes;
                    event = WarehouseEvent::DeliveryError;
                } else {
                    event = WarehouseEvent::Delivered;
                }
                let delivered = forklift.drop_pallet().unwrap_or_else(|| pallet.clone());
                if let Some(sink) = floor.sinks.get_mut(&destination_id) {
                    sink.collect(delivered, now, !shipping_error);
                }
                forklift.move_to(&destination_id);
            } else if before.carrying {
                event = WarehouseEvent::MoveLoaded;
                forklift.move_to(&layout.stations[target].id);
            } else if target as i64 == before.pallet {
                event = WarehouseEvent::Pickup;
                forklift.move_to(&layout.stations[target].id);
                forklift.pickup(pallet.clone());
            } else {
                event = WarehouseEvent::SearchMiss;
                search_misses += 1;
                cumulative_search_misses += 1;
                forklift.move_to(&layout.stations[target].id);
            }

            let obs_dist = (model.spec.observation)(next_state_id, action_index);
            let obs_idx = sample_index(&obs_dist, &mut rng);
            observed_location = observation_to_location(model, obs_idx, observed_location);
            belief = belief_update(&model.spec, &belief, action_index, obs_idx);
            let entropy = belief.entropy();
            entropy_sum += entropy;
            entropy_count += 1;

            if record_trace {
                let after_station_id = if after.terminal {
                    destination_id.clone()
                } else {
                    layout.stations[after.forklift as usize].id.clone()
                };
                let pallet_after_id = if after.terminal {
                    destination_id.clone()
                } else {
                    layout.stations[after.pallet as usize].id.clone()
                };
                trace.push(WarehouseStepTrace {
                    scenario_id: scenario.id.clone(),
                    job_id: job_id.clone(),
                    job_index: j,
                    step,
                    time_start,
                    time_end: now,
                    action: action.label.clone(),
                    action_target: layout.stations[target].id.clone(),
                    observation: model.observations[obs_idx].label.clone(),
                    event,
                    destination: destination_id.clone(),
                    forklift_before: before_station_id,
                    forklift_after: after_station_id,
                    pallet_before: pallet_before_id,
                    pallet_after: pallet_after_id,
                    carrying_before: before.carrying,
                    carrying_after: if after.terminal {
                        false
                    } else {
                        after.carrying
                    },
                    belief_entropy: entropy,
                    belief_by_station: belief_by_station(model, &belief),
                    cumulative_delivered: completed_jobs,
                    cumulative_errors: shipping_errors,
                    cumulative_search_misses,
                    cycle_time_so_far: now - job_start,
                });
            }

            actual_state = next_state_id;
            if completed {
                break;
            }
        }

        if !completed {
            failed_jobs += 1;
            now += scenario.rework_penalty_minutes;
            trace.push(WarehouseStepTrace {
                scenario_id: scenario.id.clone(),
                job_id: job_id.clone(),
                job_index: j,
                step: steps_taken,
                time_start: now,
                time_end: now,
                action: "manual escalation".into(),
                action_target: forklift.station_id.clone(),
                observation: "job failed before delivery".into(),
                event: WarehouseEvent::Failed,
                destination: destination_id.clone(),
                forklift_before: forklift.station_id.clone(),
                forklift_after: forklift.station_id.clone(),
                pallet_before: pallet.borrow().location_id.clone(),
                pallet_after: pallet.borrow().location_id.clone(),
                carrying_before: forklift.carrying.is_some(),
                carrying_after: forklift.carrying.is_some(),
                belief_entropy: belief.entropy(),
                belief_by_station: belief_by_station(model, &belief),
                cumulative_delivered: completed_jobs,
                cumulative_errors: shipping_errors,
                cumulative_search_misses,
                cycle_time_so_far: now - job_start,
            });
            forklift.carrying = None;
        }

        let cycle_time = now - job_start;
        job_summaries.push(WarehouseJobSummary {
            job_id: job_id.clone(),
            destination: destination_id.clone(),
            completed,
            shipping_error,
            cycle_time,
            steps: steps_taken,
            search_misses,
            on_time: completed && cycle_time <= scenario.due_minutes,
        });
    }

    let total_cycle: f64 = job_summaries.iter().map(|j| j.cycle_time).sum();
    let completed_list: Vec<&WarehouseJobSummary> =
        job_summaries.iter().filter(|j| j.completed).collect();
    let metrics = WarehouseMetrics {
        jobs_created: jobs,
        completed_jobs,
        failed_jobs,
        shipping_errors,
        shipping_error_rate: if completed_jobs > 0 {
            shipping_errors as f64 / completed_jobs as f64
        } else {
            0.0
        },
        total_time: now,
        mean_cycle_time: if !job_summaries.is_empty() {
            total_cycle / job_summaries.len() as f64
        } else {
            0.0
        },
        throughput_per_hour: if now > 0.0 {
            completed_jobs as f64 / now * 60.0
        } else {
            0.0
        },
        on_time_rate: if !completed_list.is_empty() {
            completed_list.iter().filter(|j| j.on_time).count() as f64 / completed_list.len() as f64
        } else {
            0.0
        },
        mean_steps_per_job: mean(
            &job_summaries
                .iter()
                .map(|j| j.steps as f64)
                .collect::<Vec<_>>(),
        ),
        mean_search_misses_per_job: mean(
            &job_summaries
                .iter()
                .map(|j| j.search_misses as f64)
                .collect::<Vec<_>>(),
        ),
        mean_belief_entropy: if entropy_count > 0 {
            entropy_sum / entropy_count as f64
        } else {
            0.0
        },
    };

    WarehouseScenarioResult {
        scenario,
        layout,
        metrics,
        jobs: job_summaries,
        trace,
    }
}

pub fn run_warehouse_comparison(opts: WarehouseSimulationOptions) -> WarehouseComparisonResult {
    let layout = opts.layout.clone().unwrap_or_else(default_warehouse_layout);
    let seed = opts.seed.unwrap_or(7);
    let jobs = opts.jobs.unwrap_or(120);
    let destination_plan = opts.destination_plan.clone().unwrap_or_else(|| {
        make_destination_plan(&layout, jobs, &mut mulberry32(seed.wrapping_add(404)))
    });
    let baseline = simulate_warehouse_scenario(
        baseline_warehouse_scenario(),
        WarehouseSimulationOptions {
            jobs: Some(jobs),
            seed: Some(seed),
            max_steps_per_job: opts.max_steps_per_job,
            layout: Some(layout.clone()),
            record_trace: opts.record_trace,
            destination_plan: Some(destination_plan.clone()),
        },
    );
    let track3t = simulate_warehouse_scenario(
        track3t_warehouse_scenario(),
        WarehouseSimulationOptions {
            jobs: Some(jobs),
            seed: Some(seed),
            max_steps_per_job: opts.max_steps_per_job,
            layout: Some(layout.clone()),
            record_trace: opts.record_trace,
            destination_plan: Some(destination_plan.clone()),
        },
    );
    let deltas = ComparisonDeltas {
        mean_cycle_time_reduction_pct: pct_reduction(
            baseline.metrics.mean_cycle_time,
            track3t.metrics.mean_cycle_time,
        ),
        throughput_lift_pct: pct_lift(
            baseline.metrics.throughput_per_hour,
            track3t.metrics.throughput_per_hour,
        ),
        search_miss_reduction_pct: pct_reduction(
            baseline.metrics.mean_search_misses_per_job,
            track3t.metrics.mean_search_misses_per_job,
        ),
        error_reduction_pct: pct_reduction(
            baseline.metrics.shipping_error_rate,
            track3t.metrics.shipping_error_rate,
        ),
        entropy_reduction_pct: pct_reduction(
            baseline.metrics.mean_belief_entropy,
            track3t.metrics.mean_belief_entropy,
        ),
    };
    WarehouseComparisonResult {
        layout,
        baseline,
        track3t,
        deltas,
        source_notes: track3t_archive_grounding(),
    }
}

pub fn summarize_warehouse_comparison(result: &WarehouseComparisonResult) -> String {
    let rows: Vec<Vec<String>> = vec![
        vec!["metric".into(), "baseline".into(), "track3t".into()],
        vec![
            "completed".into(),
            result.baseline.metrics.completed_jobs.to_string(),
            result.track3t.metrics.completed_jobs.to_string(),
        ],
        vec![
            "mean cycle min".into(),
            fmt(result.baseline.metrics.mean_cycle_time),
            fmt(result.track3t.metrics.mean_cycle_time),
        ],
        vec![
            "throughput jobs/hr".into(),
            fmt(result.baseline.metrics.throughput_per_hour),
            fmt(result.track3t.metrics.throughput_per_hour),
        ],
        vec![
            "search misses/job".into(),
            fmt(result.baseline.metrics.mean_search_misses_per_job),
            fmt(result.track3t.metrics.mean_search_misses_per_job),
        ],
        vec![
            "shipping error rate".into(),
            pct(result.baseline.metrics.shipping_error_rate),
            pct(result.track3t.metrics.shipping_error_rate),
        ],
        vec![
            "on-time rate".into(),
            pct(result.baseline.metrics.on_time_rate),
            pct(result.track3t.metrics.on_time_rate),
        ],
        vec![
            "mean belief entropy".into(),
            fmt(result.baseline.metrics.mean_belief_entropy),
            fmt(result.track3t.metrics.mean_belief_entropy),
        ],
    ];
    let widths: Vec<usize> = (0..rows[0].len())
        .map(|i| rows.iter().map(|r| r[i].len()).max().unwrap_or(0))
        .collect();
    rows.iter()
        .enumerate()
        .map(|(idx, r)| {
            let line = r
                .iter()
                .enumerate()
                .map(|(i, c)| format!("{:width$}", c, width = widths[i]))
                .collect::<Vec<_>>()
                .join("  ");
            if idx == 0 {
                let sep = widths
                    .iter()
                    .map(|w| "-".repeat(*w))
                    .collect::<Vec<_>>()
                    .join("  ");
                format!("{line}\n{sep}")
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

// =============================================================================
// Floor + belief construction
// =============================================================================

pub fn build_warehouse_floor(layout: &WarehouseLayout) -> WarehouseFloor {
    validate_layout(layout);
    let mut stations: HashMap<String, WarehouseStation> = HashMap::new();
    let mut sinks: HashMap<String, WarehouseSink> = HashMap::new();
    let mut source: Option<WarehouseSource> = None;
    for def in &layout.stations {
        if def.id == layout.source_station_id {
            source = Some(WarehouseSource::new(def.clone()));
            stations.insert(def.id.clone(), WarehouseStation::new(def.clone()));
        } else if layout.sink_station_ids.contains(&def.id) {
            sinks.insert(def.id.clone(), WarehouseSink::new(def.clone()));
            stations.insert(def.id.clone(), WarehouseStation::new(def.clone()));
        } else {
            stations.insert(def.id.clone(), WarehouseStation::new(def.clone()));
        }
    }
    let source =
        source.unwrap_or_else(|| panic!("source station not found: {}", layout.source_station_id));
    WarehouseFloor {
        source,
        sinks,
        stations,
    }
}

pub fn initial_warehouse_belief(
    layout: &WarehouseLayout,
    scenario: &WarehouseScenarioConfig,
    model: &WarehousePOMDPModel,
    forklift_index: usize,
    observed_location_index: usize,
) -> DiscreteBelief<usize> {
    let n = layout.stations.len();
    let prior = initial_location_prior(layout, scenario);
    let mut location_posterior = vec![0.0; n];
    for loc in 0..n {
        location_posterior[loc] = prior[loc]
            * location_observation_probability(n, scenario, loc, observed_location_index);
    }
    normalize_in_place(&mut location_posterior);
    let mut weights = vec![0.0; model.states.len()];
    for (loc, &p) in location_posterior.iter().enumerate() {
        weights[model.encode_state(forklift_index, loc, false)] = p;
    }
    DiscreteBelief::new(model.states.clone(), Some(&weights))
}

pub fn belief_by_station(model: &WarehousePOMDPModel, belief: &DiscreteBelief<usize>) -> Vec<f64> {
    let n = model.layout.stations.len();
    let mut out = vec![0.0; n];
    for i in 0..belief.weights.len() {
        let w = belief.weights[i];
        if w == 0.0 || i == model.terminal_state {
            continue;
        }
        let s = model.decode_state(i);
        if s.terminal {
            continue;
        }
        out[s.pallet as usize] += w;
    }
    out
}

pub fn travel_minutes(
    layout: &WarehouseLayout,
    scenario: &WarehouseScenarioConfig,
    from_station_idx: usize,
    to_station_idx: usize,
) -> f64 {
    if from_station_idx == to_station_idx {
        return 0.0;
    }
    let meters = manhattan_distance(layout, from_station_idx, to_station_idx)
        * layout.grid_meters.unwrap_or(12.0);
    meters / scenario.forklift_speed_meters_per_minute * scenario.route_inflation
}

fn action_duration_minutes(
    layout: &WarehouseLayout,
    scenario: &WarehouseScenarioConfig,
    before: &WarehouseDecisionState,
    target: usize,
    destination_index: usize,
) -> f64 {
    if before.terminal {
        return 0.0;
    }
    let travel = travel_minutes(layout, scenario, before.forklift as usize, target);
    if before.carrying && target == destination_index {
        return travel + scenario.handling_minutes + scenario.confirmation_delay_minutes;
    }
    if before.carrying {
        return travel;
    }
    if target as i64 == before.pallet {
        return travel + scenario.handling_minutes + scenario.confirmation_delay_minutes;
    }
    travel + scenario.search_penalty_minutes + scenario.confirmation_delay_minutes
}

fn sample_initial_pallet_location(
    layout: &WarehouseLayout,
    scenario: &WarehouseScenarioConfig,
    rng: &mut dyn RandomSource,
) -> usize {
    let station_to_index = station_index_map(layout);
    let source_index = *station_to_index
        .get(&layout.source_station_id)
        .expect("source index present");
    if rng.next_float() >= scenario.initial_misplacement_probability {
        return source_index;
    }
    let candidates: Vec<usize> = pallet_candidate_indexes(layout)
        .into_iter()
        .filter(|&i| i != source_index)
        .collect();
    let idx = (rng.next_float() * candidates.len() as f64).floor() as usize;
    candidates.get(idx).copied().unwrap_or(source_index)
}

fn sample_location_observation(
    layout: &WarehouseLayout,
    scenario: &WarehouseScenarioConfig,
    true_location_index: usize,
    rng: &mut dyn RandomSource,
) -> usize {
    let n = layout.stations.len();
    if rng.next_float() < scenario.location_accuracy {
        return true_location_index;
    }
    let mut obs = (rng.next_float() * (n - 1) as f64).floor() as usize;
    if obs >= true_location_index {
        obs += 1;
    }
    obs
}

fn observation_to_location(model: &WarehousePOMDPModel, obs_idx: usize, fallback: usize) -> usize {
    let obs = &model.observations[obs_idx];
    if obs.kind == WarehouseObsKind::Location {
        if let Some(station) = obs.station {
            return station;
        }
    }
    fallback
}

fn initial_location_prior(
    layout: &WarehouseLayout,
    scenario: &WarehouseScenarioConfig,
) -> Vec<f64> {
    let n = layout.stations.len();
    let station_to_index = station_index_map(layout);
    let source_index = *station_to_index
        .get(&layout.source_station_id)
        .expect("source index present");
    let candidates = pallet_candidate_indexes(layout);
    let mut prior = vec![0.0; n];
    prior[source_index] = 1.0 - scenario.initial_misplacement_probability;
    let others: Vec<usize> = candidates
        .into_iter()
        .filter(|&i| i != source_index)
        .collect();
    let share = scenario.initial_misplacement_probability / (others.len().max(1) as f64);
    for i in others {
        prior[i] = share;
    }
    normalize_in_place(&mut prior);
    prior
}

fn location_observation_probability(
    num_stations: usize,
    scenario: &WarehouseScenarioConfig,
    true_location_index: usize,
    observed_location_index: usize,
) -> f64 {
    if true_location_index == observed_location_index {
        scenario.location_accuracy
    } else {
        (1.0 - scenario.location_accuracy) / ((num_stations - 1).max(1) as f64)
    }
}

fn make_destination_plan(
    layout: &WarehouseLayout,
    jobs: usize,
    rng: &mut dyn RandomSource,
) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(jobs);
    for _ in 0..jobs {
        let idx = (rng.next_float() * layout.sink_station_ids.len() as f64).floor() as usize;
        out.push(layout.sink_station_ids[idx].clone());
    }
    out
}

fn pallet_candidate_indexes(layout: &WarehouseLayout) -> Vec<usize> {
    layout
        .stations
        .iter()
        .enumerate()
        .filter(|(_, s)| s.kind != WarehouseStationKind::Sink && s.can_hold_pallet != Some(false))
        .map(|(i, _)| i)
        .collect()
}

fn manhattan_distance(layout: &WarehouseLayout, a_idx: usize, b_idx: usize) -> f64 {
    let a = &layout.stations[a_idx];
    let b = &layout.stations[b_idx];
    (a.x - b.x).abs() + (a.y - b.y).abs()
}

fn station_index_map(layout: &WarehouseLayout) -> HashMap<String, usize> {
    layout
        .stations
        .iter()
        .enumerate()
        .map(|(i, s)| (s.id.clone(), i))
        .collect()
}

// =============================================================================
// Validation + small numeric helpers
// =============================================================================

fn validate_layout(layout: &WarehouseLayout) {
    require(Preconditions::non_empty(
        "WarehouseLayout",
        "stations",
        &layout.stations,
    ));
    let mut ids: HashSet<String> = HashSet::new();
    for (i, s) in layout.stations.iter().enumerate() {
        if ids.contains(&s.id) {
            panic!("WarehouseLayout: duplicate station id {}", s.id);
        }
        ids.insert(s.id.clone());
        require(Preconditions::finite(
            "WarehouseLayout",
            &format!("stations[{i}].x"),
            s.x,
        ));
        require(Preconditions::finite(
            "WarehouseLayout",
            &format!("stations[{i}].y"),
            s.y,
        ));
    }
    if !ids.contains(&layout.source_station_id) {
        panic!(
            "WarehouseLayout: source missing: {}",
            layout.source_station_id
        );
    }
    for id in &layout.sink_station_ids {
        if !ids.contains(id) {
            panic!("WarehouseLayout: sink missing: {id}");
        }
    }
    if let Some(grid) = layout.grid_meters {
        require(Preconditions::positive(
            "WarehouseLayout",
            "gridMeters",
            grid,
        ));
    }
}

fn validate_scenario(s: &WarehouseScenarioConfig) {
    let model = format!("WarehouseScenario({})", s.id);
    require(Preconditions::in_range(
        &model,
        "locationAccuracy",
        s.location_accuracy,
        0.5,
        1.0,
    ));
    require(Preconditions::in_range(
        &model,
        "idAccuracy",
        s.id_accuracy,
        0.0,
        1.0,
    ));
    require(Preconditions::in_range(
        &model,
        "initialMisplacementProbability",
        s.initial_misplacement_probability,
        0.0,
        1.0,
    ));
    require(Preconditions::in_range(
        &model,
        "placementErrorProbability",
        s.placement_error_probability,
        0.0,
        1.0,
    ));
    require(Preconditions::positive(
        &model,
        "forkliftSpeedMetersPerMinute",
        s.forklift_speed_meters_per_minute,
    ));
    require(Preconditions::positive(
        &model,
        "routeInflation",
        s.route_inflation,
    ));
    require(Preconditions::non_negative(
        &model,
        "handlingMinutes",
        s.handling_minutes,
    ));
    require(Preconditions::non_negative(
        &model,
        "confirmationDelayMinutes",
        s.confirmation_delay_minutes,
    ));
    require(Preconditions::non_negative(
        &model,
        "searchPenaltyMinutes",
        s.search_penalty_minutes,
    ));
    require(Preconditions::non_negative(
        &model,
        "reworkPenaltyMinutes",
        s.rework_penalty_minutes,
    ));
    require(Preconditions::positive(
        &model,
        "deliveryReward",
        s.delivery_reward,
    ));
    require(Preconditions::non_negative(
        &model,
        "wrongDeliveryPenalty",
        s.wrong_delivery_penalty,
    ));
    require(Preconditions::in_range(
        &model, "discount", s.discount, 0.0, 1.0,
    ));
    require(Preconditions::positive(&model, "qmdpTol", s.qmdp_tol));
    require(Preconditions::integer_in_range(
        &model,
        "qmdpMaxIter",
        s.qmdp_max_iter as f64,
        1.0,
        100000.0,
    ));
    require(Preconditions::positive(&model, "dueMinutes", s.due_minutes));
    require(Preconditions::positive(
        &model,
        "sensorRefreshSeconds",
        s.sensor_refresh_seconds,
    ));
}

fn sample_index(probabilities: &[f64], rng: &mut dyn RandomSource) -> usize {
    let u = rng.next_float();
    let mut acc = 0.0;
    for (i, &p) in probabilities.iter().enumerate() {
        acc += p;
        if u <= acc {
            return i;
        }
    }
    probabilities.len() - 1
}

fn normalize_in_place(xs: &mut [f64]) {
    let total: f64 = xs.iter().sum();
    if !total.is_finite() || total <= 0.0 {
        let u = 1.0 / xs.len() as f64;
        for v in xs.iter_mut() {
            *v = u;
        }
        return;
    }
    for v in xs.iter_mut() {
        *v /= total;
    }
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        0.0
    } else {
        xs.iter().sum::<f64>() / xs.len() as f64
    }
}

fn pct_reduction(a: f64, b: f64) -> f64 {
    if a.abs() < 1e-12 {
        return if b < a { 100.0 } else { 0.0 };
    }
    (a - b) / a.abs() * 100.0
}

fn pct_lift(a: f64, b: f64) -> f64 {
    if a.abs() < 1e-12 {
        return if b > a { 100.0 } else { 0.0 };
    }
    (b - a) / a.abs() * 100.0
}

fn fmt(x: f64) -> String {
    if x.is_finite() {
        format!("{x:.3}")
    } else {
        "n/a".to_string()
    }
}

fn pct(x: f64) -> String {
    if x.is_finite() {
        format!("{:.1}%", x * 100.0)
    } else {
        "n/a".to_string()
    }
}

#[cfg(test)]
mod tests {
    //! Fast structural / kernel tests. The full QMDP value iteration and the
    //! 120-job comparison are exercised by the integration suite to keep unit
    //! tests cheap.

    use super::*;

    #[test]
    fn default_layout_shape() {
        let layout = default_warehouse_layout();
        // 2 fixed + 4 rows x 3 reserves + aisle + 4 sinks = 19 stations.
        assert_eq!(layout.stations.len(), 19);
        assert_eq!(layout.source_station_id, "receiving");
        assert_eq!(
            layout.sink_station_ids,
            vec!["line-a", "line-b", "line-c", "shipping"]
        );
        validate_layout(&layout);
    }

    #[test]
    fn pomdp_model_dimensions_and_encoding() {
        let layout = default_warehouse_layout();
        let scenario = baseline_warehouse_scenario();
        let model = build_warehouse_pomdp(&layout, &scenario, 0);
        let n = layout.stations.len();
        assert_eq!(model.n, n);
        assert_eq!(model.terminal_state, n * n * 2);
        assert_eq!(model.states.len(), n * n * 2 + 1);
        assert_eq!(model.actions.len(), n);
        assert_eq!(model.observations.len(), n + 2);

        // encode/decode round-trips for a non-terminal state.
        let sid = model.encode_state(3, 5, true);
        let s = model.decode_state(sid);
        assert!(!s.terminal);
        assert_eq!(s.forklift, 3);
        assert_eq!(s.pallet, 5);
        assert!(s.carrying);

        // the terminal state decodes as terminal.
        assert!(model.decode_state(model.terminal_state).terminal);
    }

    #[test]
    fn next_state_pickup_move_and_delivery() {
        let layout = default_warehouse_layout();
        let scenario = baseline_warehouse_scenario();
        let destination = 4usize;
        let model = build_warehouse_pomdp(&layout, &scenario, destination);

        // Not carrying, drive onto the pallet -> pick it up (carrying at target).
        let not_carrying = model.encode_state(0, 6, false);
        let picked = model.next_state(not_carrying, 6);
        let s = model.decode_state(picked);
        assert!(s.carrying);
        assert_eq!(s.forklift, 6);
        assert_eq!(s.pallet, 6);

        // Carrying, drive to the destination -> terminal.
        let carrying = model.encode_state(2, 2, true);
        assert_eq!(
            model.next_state(carrying, destination),
            model.terminal_state
        );
    }

    #[test]
    fn travel_minutes_zero_when_in_place() {
        let layout = default_warehouse_layout();
        let scenario = baseline_warehouse_scenario();
        assert_eq!(travel_minutes(&layout, &scenario, 3, 3), 0.0);
        assert!(travel_minutes(&layout, &scenario, 0, 1) > 0.0);
    }

    #[test]
    fn pct_helpers() {
        assert!((pct_reduction(10.0, 4.0) - 60.0).abs() < 1e-9);
        assert!((pct_lift(10.0, 13.0) - 30.0).abs() < 1e-9);
        assert_eq!(pct_reduction(0.0, 0.0), 0.0);
        assert_eq!(pct_lift(0.0, 1.0), 100.0);
    }
}
