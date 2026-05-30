//! Port of `src/des/general/network-mutex.ts` — a network mutex as DES stations
//! plus child tokens.
//!
//! Model:
//!   Source -> Station A (lock-aware worker) -> Station C (sink)
//!                 | request/release child tokens
//!                 v
//!             Station B (lock service)
//!
//! The real item stays in Station A's internal FIFO until a child
//! LockRequestToken is granted by Station B. A then processes the item while the
//! lock is held, sends a LockReleaseToken, and only then emits the item onward.
//! This gives "request spawning request" semantics without hidden global events.
//!
//! ## Rust shape (faithful translation)
//!
//!   * The string-literal state unions `MutexWorkState` / `MutexChildState` →
//!     the [`MutexWorkState`] / [`MutexChildState`] enums.
//!   * `interface MutexWorkItem extends StatefulToken<S>` (and the child token
//!     interfaces) → structs that EMBED a [`StatefulToken<S>`] field (Rust has no
//!     interface inheritance); state transitions mutate that embedded token.
//!   * Tokens travel as `Rc<dyn Any>`; the drain/mutate/re-emit flow clones the
//!     owned token data on drain, mutates the owned copy, and emits a fresh `Rc`
//!     carrying the accumulated mutations (the TS shared one object reference).
//!   * `class NetworkMutexWorkerStation extends CompositeDESStation` → a struct
//!     that OWNS a [`CompositeDESStation`] plus typed handles to its substations.
//!   * The shared `events` array (passed by reference to every station in TS) →
//!     an `Rc<RefCell<Vec<NetworkMutexTraceEvent>>>` clone shared by handle.
//!   * The `*_CHANNEL` consts → `&'static str`; the `new Set(...).size === len`
//!     dedupe validator → a `HashSet` cardinality compare.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashSet;
use std::rc::Rc;

use crate::des::general::des_base::composite_station::CompositeDESStation;
use crate::des::general::des_base::preconditions::Preconditions;
use crate::des::general::des_base::runner::{
    failed_validation_checks, run_iterative_des, IterativeRunOptions, RunReason,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::stateful_token::{
    make_stateful_token, spawn_stateful_child_token, transition_token, MakeStatefulTokenOpts,
    SpawnStatefulChildTokenOpts, StatefulToken, TransitionTokenOpts,
};
use crate::des::general::des_base::validation::intrinsic_check;

pub const MUTEX_WORK_CHANNEL: &str = "work";
pub const MUTEX_DONE_CHANNEL: &str = "done";
pub const MUTEX_REQUEST_CHANNEL: &str = "lock-request";
pub const MUTEX_GRANT_CHANNEL: &str = "lock-grant";
pub const MUTEX_RELEASE_CHANNEL: &str = "lock-release";
const MUTEX_LOCKED_WORK_CHANNEL: &str = "locked-work";

/// Lifecycle states of a unit of work.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutexWorkState {
    Created,
    Queued,
    WaitingLock,
    LockGranted,
    Processing,
    Releasing,
    Completed,
}

/// Lifecycle states of a child (request/grant/release) token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MutexChildState {
    Spawned,
    Queued,
    Granted,
    Released,
    Accepted,
    Invalid,
}

/// Trace-event kinds (the TS `NetworkMutexTraceEvent['event']` union).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkMutexEventKind {
    WorkArrived,
    RequestSpawned,
    RequestQueued,
    GrantScheduled,
    GrantReceived,
    ProcessingStarted,
    ProcessingFinished,
    ReleaseSpawned,
    ReleaseAccepted,
    WorkCompleted,
    InvalidRelease,
}

impl NetworkMutexEventKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NetworkMutexEventKind::WorkArrived => "work-arrived",
            NetworkMutexEventKind::RequestSpawned => "request-spawned",
            NetworkMutexEventKind::RequestQueued => "request-queued",
            NetworkMutexEventKind::GrantScheduled => "grant-scheduled",
            NetworkMutexEventKind::GrantReceived => "grant-received",
            NetworkMutexEventKind::ProcessingStarted => "processing-started",
            NetworkMutexEventKind::ProcessingFinished => "processing-finished",
            NetworkMutexEventKind::ReleaseSpawned => "release-spawned",
            NetworkMutexEventKind::ReleaseAccepted => "release-accepted",
            NetworkMutexEventKind::WorkCompleted => "work-completed",
            NetworkMutexEventKind::InvalidRelease => "invalid-release",
        }
    }
}

/// Lock bookkeeping attached to a work item (the TS inline `lock?` object).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MutexLockInfo {
    pub request_id: String,
    pub requested_tick: usize,
    pub granted_tick: Option<usize>,
    pub processing_started_tick: Option<usize>,
    pub processing_finished_tick: Option<usize>,
    pub released_tick: Option<usize>,
}

/// A unit of work flowing through the mutex network (`MutexWorkItem`).
#[derive(Clone, Debug)]
pub struct MutexWorkItem {
    /// Embedded lineage/state-machine token (the TS `extends StatefulToken<S>`).
    pub token: StatefulToken<MutexWorkState>,
    pub item_id: String,
    pub created_tick: usize,
    pub lock: Option<MutexLockInfo>,
}

/// A lock request child token (`LockRequestToken`).
#[derive(Clone, Debug)]
pub struct LockRequestToken {
    pub token: StatefulToken<MutexChildState>,
    pub parent_item_id: String,
    pub token_id: String,
    pub owner_id: String,
    pub created_tick: usize,
}

/// A lock grant child token (`LockGrantToken`).
#[derive(Clone, Debug)]
pub struct LockGrantToken {
    pub token: StatefulToken<MutexChildState>,
    pub parent_item_id: String,
    pub token_id: String,
    pub owner_id: String,
    pub created_tick: usize,
    pub granted_tick: usize,
    pub service_request_queued_tick: usize,
}

/// A lock release child token (`LockReleaseToken`).
#[derive(Clone, Debug)]
pub struct LockReleaseToken {
    pub token: StatefulToken<MutexChildState>,
    pub parent_item_id: String,
    pub token_id: String,
    pub owner_id: String,
    pub created_tick: usize,
    pub released_tick: usize,
}

/// A single trace-log entry.
#[derive(Clone, Debug)]
pub struct NetworkMutexTraceEvent {
    pub tick: usize,
    pub station_id: String,
    pub event: NetworkMutexEventKind,
    pub item_id: Option<String>,
    pub child_token_id: Option<String>,
    pub detail: Option<String>,
}

/// Source spec for the work generator.
#[derive(Clone, Copy, Debug)]
pub struct MutexSourceSpec {
    pub count: usize,
    pub interarrival_ticks: usize,
    pub first_arrival_tick: Option<usize>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NetworkMutexLockServiceOpts {
    /// Ticks from lock-service decision to grant token arrival at Station A.
    pub grant_delay_ticks: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
pub struct NetworkMutexWorkerOpts {
    pub processing_ticks: usize,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NetworkMutexSimulationOpts {
    pub source: Option<MutexSourceSpec>,
    pub worker: Option<NetworkMutexWorkerOpts>,
    pub lock: Option<NetworkMutexLockServiceOpts>,
    pub max_ticks: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct NetworkMutexLockStats {
    pub grant_count: usize,
    pub release_count: usize,
    pub invalid_release_count: usize,
    pub final_holder_item_id: Option<String>,
    pub waiting_requests: usize,
    pub max_wait_queue: usize,
    pub mean_service_queue_wait_ticks: f64,
    pub lock_held_ticks: usize,
    pub utilization: f64,
}

#[derive(Clone, Debug, Default)]
pub struct NetworkMutexWorkerStats {
    pub arrived: usize,
    pub completed: usize,
    pub final_queue: usize,
    pub max_queue: usize,
    pub mean_queue_wait_ticks: f64,
    pub mean_lock_wait_ticks: f64,
    pub mean_time_in_system_ticks: f64,
    pub child_requests_spawned: usize,
    pub child_releases_spawned: usize,
}

#[derive(Clone, Debug, Default)]
pub struct NetworkMutexSimulationResult {
    pub generated: usize,
    pub completed: usize,
    pub total_ticks: usize,
    pub worker: NetworkMutexWorkerStats,
    pub lock: NetworkMutexLockStats,
    pub completed_items: Vec<MutexWorkItem>,
    pub trace: Vec<NetworkMutexTraceEvent>,
    pub invariant_violations: Vec<String>,
}

// ── Internal carrier structs ────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct QueuedWork {
    item: MutexWorkItem,
    enqueued_tick: usize,
}

#[derive(Clone, Debug)]
struct ActiveWork {
    item: MutexWorkItem,
    grant: LockGrantToken,
    remaining_ticks: i64,
}

#[derive(Clone, Debug)]
struct LockedWorkToken {
    item: MutexWorkItem,
    grant: LockGrantToken,
}

#[derive(Clone, Debug)]
struct PendingGrant {
    grant: LockGrantToken,
    deliver_at_tick: usize,
}

#[derive(Clone, Debug)]
struct CurrentHolder {
    request: LockRequestToken,
    acquired_tick: usize,
}

/// Shared, append-only trace log.
type TraceLog = Rc<RefCell<Vec<NetworkMutexTraceEvent>>>;

#[allow(clippy::too_many_arguments)]
fn trace(
    log: &TraceLog,
    tick: usize,
    station_id: &str,
    event: NetworkMutexEventKind,
    item_id: Option<String>,
    child_token_id: Option<String>,
    detail: Option<String>,
) {
    log.borrow_mut().push(NetworkMutexTraceEvent {
        tick,
        station_id: station_id.to_string(),
        event,
        item_id,
        child_token_id,
        detail,
    });
}

/// `{...(lock ?? base), <field overrides>}` merge: keep existing fields, with a
/// base `request_id`/`requested_tick` fallback when no prior lock exists.
fn lock_with(
    existing: Option<MutexLockInfo>,
    base_request_id: &str,
    base_requested_tick: usize,
) -> MutexLockInfo {
    existing.unwrap_or(MutexLockInfo {
        request_id: base_request_id.to_string(),
        requested_tick: base_requested_tick,
        ..Default::default()
    })
}

// -----------------------------------------------------------------------------
// SOURCE
// -----------------------------------------------------------------------------

pub struct MutexWorkSourceStation {
    core: StationCore,
    count: usize,
    interarrival_ticks: usize,
    first_arrival_tick: usize,
    tick: usize,
    emitted: usize,
}

impl MutexWorkSourceStation {
    pub fn new(id: &str, spec: MutexSourceSpec) -> Self {
        MutexWorkSourceStation {
            core: StationCore::new(id),
            count: spec.count,
            interarrival_ticks: spec.interarrival_ticks,
            first_arrival_tick: spec.first_arrival_tick.unwrap_or(0),
            tick: 0,
            emitted: 0,
        }
    }

    pub fn emitted_count(&self) -> usize {
        self.emitted
    }
}

impl DESStation for MutexWorkSourceStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        let id = self.core.id.clone();
        Preconditions::integer_in_range("network-mutex", &format!("{id}.count"), self.count as f64, 0.0, 1e9)
            .expect("count");
        Preconditions::integer_in_range(
            "network-mutex",
            &format!("{id}.interarrivalTicks"),
            self.interarrival_ticks as f64,
            1.0,
            1e9,
        )
        .expect("interarrivalTicks");
        Preconditions::integer_in_range(
            "network-mutex",
            &format!("{id}.firstArrivalTick"),
            self.first_arrival_tick as f64,
            0.0,
            1e9,
        )
        .expect("firstArrivalTick");
    }
    fn has_work(&self) -> bool {
        self.emitted < self.count
    }
    fn run_time_step(&mut self) {
        let next_arrival = self.first_arrival_tick + self.emitted * self.interarrival_ticks;
        if self.emitted < self.count && self.tick >= next_arrival {
            let item_id = format!("item-{}", self.emitted + 1);
            let base = make_stateful_token::<MutexWorkState>(MakeStatefulTokenOpts {
                kind: "mutex-work".to_string(),
                token_id: format!("work:{item_id}"),
                initial_state: MutexWorkState::Created,
                tick: self.tick as f64,
                station_id: self.core.id.clone(),
                event: None,
                detail: None,
            });
            let item = MutexWorkItem { token: base, item_id, created_tick: self.tick, lock: None };
            self.emitted += 1;
            self.core.emit(Rc::new(item), MUTEX_WORK_CHANNEL);
        }
        self.tick += 1;
    }
}

// -----------------------------------------------------------------------------
// LOCK SERVICE (Station B)
// -----------------------------------------------------------------------------

pub struct NetworkMutexLockServiceStation {
    core: StationCore,
    grant_delay_ticks: usize,
    wait_queue: Vec<LockRequestToken>,
    pending_grants: Vec<PendingGrant>,
    holder: Option<CurrentHolder>,
    tick: usize,
    max_wait_queue: usize,
    total_service_queue_wait_ticks: usize,
    grants: usize,
    releases: usize,
    invalid_releases: usize,
    lock_held_ticks: usize,
    events: TraceLog,
}

impl NetworkMutexLockServiceStation {
    pub fn new(id: &str, opts: NetworkMutexLockServiceOpts, events: TraceLog) -> Self {
        let mut st = NetworkMutexLockServiceStation {
            core: StationCore::new(id),
            grant_delay_ticks: opts.grant_delay_ticks.unwrap_or(2),
            wait_queue: Vec::new(),
            pending_grants: Vec::new(),
            holder: None,
            tick: 0,
            max_wait_queue: 0,
            total_service_queue_wait_ticks: 0,
            grants: 0,
            releases: 0,
            invalid_releases: 0,
            lock_held_ticks: 0,
            events,
        };
        let v = intrinsic_check::<dyn DESStation>(
            "network-mutex.lock.single-holder",
            |s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<NetworkMutexLockServiceStation>().unwrap();
                st.pending_grant_count() <= 1 || st.current_holder().is_some()
            },
            Some("at most one outstanding grant for the current holder".to_string()),
            Some(Box::new(|s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<NetworkMutexLockServiceStation>().unwrap();
                format!(
                    "holder={}, pendingGrants={}",
                    st.current_holder().unwrap_or_else(|| "none".to_string()),
                    st.pending_grant_count()
                )
            })),
            Some("network-mutex".to_string()),
            None,
        );
        st.add_validator(v.boxed());
        st
    }

    pub fn stats(&self, total_ticks: usize) -> NetworkMutexLockStats {
        NetworkMutexLockStats {
            grant_count: self.grants,
            release_count: self.releases,
            invalid_release_count: self.invalid_releases,
            final_holder_item_id: self.holder.as_ref().map(|h| h.request.parent_item_id.clone()),
            waiting_requests: self.wait_queue.len(),
            max_wait_queue: self.max_wait_queue,
            mean_service_queue_wait_ticks: self.total_service_queue_wait_ticks as f64
                / (self.grants.max(1)) as f64,
            lock_held_ticks: self.lock_held_ticks,
            utilization: self.lock_held_ticks as f64 / (total_ticks.max(1)) as f64,
        }
    }

    pub fn current_holder(&self) -> Option<String> {
        self.holder.as_ref().map(|h| h.request.parent_item_id.clone())
    }

    pub fn pending_grant_count(&self) -> usize {
        self.pending_grants.len()
    }
}

impl DESStation for NetworkMutexLockServiceStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        Preconditions::integer_in_range(
            "network-mutex",
            &format!("{}.grantDelayTicks", self.core.id),
            self.grant_delay_ticks as f64,
            0.0,
            1e9,
        )
        .expect("grantDelayTicks");
    }
    fn has_work(&self) -> bool {
        self.core.has_work() || !self.wait_queue.is_empty() || !self.pending_grants.is_empty()
    }
    fn run_time_step(&mut self) {
        let id = self.core.id.clone();
        // Releases.
        for release in self.core.drain::<LockReleaseToken>(MUTEX_RELEASE_CHANNEL) {
            let mut release = (*release).clone();
            let matches_holder = self
                .holder
                .as_ref()
                .map(|h| release.token_id == h.request.token_id)
                .unwrap_or(false);
            if matches_holder {
                transition_token(
                    &mut release.token,
                    MutexChildState::Accepted,
                    TransitionTokenOpts {
                        tick: self.tick as f64,
                        station_id: id.clone(),
                        event: NetworkMutexEventKind::ReleaseAccepted.as_str().to_string(),
                        detail: None,
                    },
                );
                let acquired = self.holder.as_ref().unwrap().acquired_tick;
                self.lock_held_ticks += self.tick.saturating_sub(acquired);
                self.holder = None;
                self.releases += 1;
                trace(
                    &self.events,
                    self.tick,
                    &id,
                    NetworkMutexEventKind::ReleaseAccepted,
                    Some(release.parent_item_id.clone()),
                    Some(release.token_id.clone()),
                    None,
                );
            } else {
                transition_token(
                    &mut release.token,
                    MutexChildState::Invalid,
                    TransitionTokenOpts {
                        tick: self.tick as f64,
                        station_id: id.clone(),
                        event: NetworkMutexEventKind::InvalidRelease.as_str().to_string(),
                        detail: None,
                    },
                );
                self.invalid_releases += 1;
                trace(
                    &self.events,
                    self.tick,
                    &id,
                    NetworkMutexEventKind::InvalidRelease,
                    Some(release.parent_item_id.clone()),
                    Some(release.token_id.clone()),
                    None,
                );
            }
        }

        // Requests.
        for req in self.core.drain::<LockRequestToken>(MUTEX_REQUEST_CHANNEL) {
            let mut req = (*req).clone();
            transition_token(
                &mut req.token,
                MutexChildState::Queued,
                TransitionTokenOpts {
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: NetworkMutexEventKind::RequestQueued.as_str().to_string(),
                    detail: None,
                },
            );
            let parent_item_id = req.parent_item_id.clone();
            let token_id = req.token_id.clone();
            self.wait_queue.push(req);
            self.max_wait_queue = self.max_wait_queue.max(self.wait_queue.len());
            trace(
                &self.events,
                self.tick,
                &id,
                NetworkMutexEventKind::RequestQueued,
                Some(parent_item_id),
                Some(token_id),
                None,
            );
        }

        // Grant the lock to the head of the queue if free.
        if self.holder.is_none() && !self.wait_queue.is_empty() {
            let req = self.wait_queue.remove(0);
            self.total_service_queue_wait_ticks += self.tick.saturating_sub(req.created_tick);
            let grant_token = spawn_stateful_child_token::<MutexChildState>(
                &req.token.lineage,
                SpawnStatefulChildTokenOpts {
                    kind: "lock-grant".to_string(),
                    token_id: format!("{}:grant", req.token_id),
                    initial_state: MutexChildState::Granted,
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: Some(NetworkMutexEventKind::GrantScheduled.as_str().to_string()),
                    detail: None,
                },
            );
            let grant = LockGrantToken {
                token: grant_token,
                parent_item_id: req.parent_item_id.clone(),
                token_id: req.token_id.clone(),
                owner_id: req.owner_id.clone(),
                created_tick: self.tick,
                granted_tick: self.tick,
                service_request_queued_tick: req.created_tick,
            };
            let parent_item_id = req.parent_item_id.clone();
            let token_id = req.token_id.clone();
            self.holder = Some(CurrentHolder { request: req, acquired_tick: self.tick });
            self.pending_grants.push(PendingGrant {
                grant,
                deliver_at_tick: self.tick + self.grant_delay_ticks,
            });
            self.grants += 1;
            trace(
                &self.events,
                self.tick,
                &id,
                NetworkMutexEventKind::GrantScheduled,
                Some(parent_item_id),
                Some(token_id),
                Some(format!("deliverAt={}", self.tick + self.grant_delay_ticks)),
            );
        }

        // Deliver due grants.
        let mut due: Vec<PendingGrant> = Vec::new();
        let mut keep: Vec<PendingGrant> = Vec::new();
        for pg in std::mem::take(&mut self.pending_grants) {
            if pg.deliver_at_tick <= self.tick {
                due.push(pg);
            } else {
                keep.push(pg);
            }
        }
        self.pending_grants = keep;
        for pg in due {
            self.core.emit(Rc::new(pg.grant), MUTEX_GRANT_CHANNEL);
        }
        self.tick += 1;
    }
}

// -----------------------------------------------------------------------------
// QUEUE SUBSTATION (inside the worker)
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct MutexQueueSubstationStats {
    arrived: usize,
    final_queue: usize,
    max_queue: usize,
    mean_queue_wait_ticks: f64,
    mean_lock_wait_ticks: f64,
    child_requests_spawned: usize,
}

struct MutexQueueSubstation {
    core: StationCore,
    owner_id: String,
    queue: Vec<QueuedWork>,
    outstanding: Option<LockRequestToken>,
    tick: usize,
    max_queue: usize,
    queue_area: usize,
    arrived: usize,
    child_requests: usize,
    total_queue_wait_ticks: usize,
    total_lock_wait_ticks: usize,
    events: TraceLog,
}

impl MutexQueueSubstation {
    fn new(id: &str, owner_id: &str, events: TraceLog) -> Self {
        MutexQueueSubstation {
            core: StationCore::new(id),
            owner_id: owner_id.to_string(),
            queue: Vec::new(),
            outstanding: None,
            tick: 0,
            max_queue: 0,
            queue_area: 0,
            arrived: 0,
            child_requests: 0,
            total_queue_wait_ticks: 0,
            total_lock_wait_ticks: 0,
            events,
        }
    }

    fn stats(&self, completed: usize) -> MutexQueueSubstationStats {
        MutexQueueSubstationStats {
            arrived: self.arrived,
            final_queue: self.queue.len() + usize::from(self.outstanding.is_some()),
            max_queue: self.max_queue,
            mean_queue_wait_ticks: self.total_queue_wait_ticks as f64 / (completed.max(1)) as f64,
            mean_lock_wait_ticks: self.total_lock_wait_ticks as f64 / (completed.max(1)) as f64,
            child_requests_spawned: self.child_requests,
        }
    }
}

impl DESStation for MutexQueueSubstation {
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
        self.core.has_work() || !self.queue.is_empty() || self.outstanding.is_some()
    }
    fn run_time_step(&mut self) {
        let id = self.core.id.clone();
        // Incoming work.
        for item in self.core.drain::<MutexWorkItem>(MUTEX_WORK_CHANNEL) {
            let mut item = (*item).clone();
            transition_token(
                &mut item.token,
                MutexWorkState::Queued,
                TransitionTokenOpts {
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: NetworkMutexEventKind::WorkArrived.as_str().to_string(),
                    detail: None,
                },
            );
            let item_id = item.item_id.clone();
            self.queue.push(QueuedWork { item, enqueued_tick: self.tick });
            self.arrived += 1;
            self.max_queue = self.max_queue.max(self.queue.len());
            trace(&self.events, self.tick, &id, NetworkMutexEventKind::WorkArrived, Some(item_id), None, None);
        }

        // Incoming grants.
        for grant in self.core.drain::<LockGrantToken>(MUTEX_GRANT_CHANNEL) {
            let matches = match &self.outstanding {
                Some(o) => grant.token_id == o.token_id && !self.queue.is_empty(),
                None => false,
            };
            if !matches {
                continue;
            }
            let outstanding = self.outstanding.take().unwrap();
            let mut queued = self.queue.remove(0);
            transition_token(
                &mut queued.item.token,
                MutexWorkState::LockGranted,
                TransitionTokenOpts {
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: NetworkMutexEventKind::GrantReceived.as_str().to_string(),
                    detail: None,
                },
            );
            let mut lock = lock_with(queued.item.lock.take(), &grant.token_id, outstanding.created_tick);
            lock.granted_tick = Some(self.tick);
            queued.item.lock = Some(lock);
            self.total_queue_wait_ticks += self.tick.saturating_sub(queued.enqueued_tick);
            self.total_lock_wait_ticks += self.tick.saturating_sub(outstanding.created_tick);
            let item_id = queued.item.item_id.clone();
            let grant_token_id = grant.token_id.clone();
            self.core.emit(
                Rc::new(LockedWorkToken { item: queued.item, grant: (*grant).clone() }),
                MUTEX_LOCKED_WORK_CHANNEL,
            );
            trace(
                &self.events,
                self.tick,
                &id,
                NetworkMutexEventKind::GrantReceived,
                Some(item_id),
                Some(grant_token_id),
                None,
            );
        }

        // Spawn a lock request for the head item if none is outstanding.
        if self.outstanding.is_none() && !self.queue.is_empty() {
            let owner_id = self.owner_id.clone();
            let head_item_id = self.queue[0].item.item_id.clone();
            let token_id = format!("{owner_id}:{head_item_id}:lock");
            let req_token = spawn_stateful_child_token::<MutexChildState>(
                &self.queue[0].item.token.lineage,
                SpawnStatefulChildTokenOpts {
                    kind: "lock-request".to_string(),
                    token_id: token_id.clone(),
                    initial_state: MutexChildState::Spawned,
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: Some(NetworkMutexEventKind::RequestSpawned.as_str().to_string()),
                    detail: None,
                },
            );
            let req = LockRequestToken {
                token: req_token,
                parent_item_id: head_item_id.clone(),
                token_id: token_id.clone(),
                owner_id,
                created_tick: self.tick,
            };
            transition_token(
                &mut self.queue[0].item.token,
                MutexWorkState::WaitingLock,
                TransitionTokenOpts {
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: NetworkMutexEventKind::RequestSpawned.as_str().to_string(),
                    detail: None,
                },
            );
            self.queue[0].item.lock = Some(MutexLockInfo {
                request_id: req.token_id.clone(),
                requested_tick: req.created_tick,
                ..Default::default()
            });
            self.outstanding = Some(req.clone());
            self.child_requests += 1;
            self.core.emit(Rc::new(req), MUTEX_REQUEST_CHANNEL);
            trace(
                &self.events,
                self.tick,
                &id,
                NetworkMutexEventKind::RequestSpawned,
                Some(head_item_id),
                Some(token_id),
                None,
            );
        }

        self.queue_area += self.queue.len();
        self.max_queue = self.max_queue.max(self.queue.len());
        self.tick += 1;
    }
}

// -----------------------------------------------------------------------------
// PROCESSOR SUBSTATION (inside the worker)
// -----------------------------------------------------------------------------

#[derive(Clone, Debug, Default)]
struct MutexProcessorStats {
    completed: usize,
    final_queue: usize,
    mean_time_in_system_ticks: f64,
    child_releases_spawned: usize,
}

struct MutexProcessorSubstation {
    core: StationCore,
    owner_id: String,
    processing_ticks: i64,
    ready: Vec<LockedWorkToken>,
    active: Option<ActiveWork>,
    tick: usize,
    completed: usize,
    child_releases: usize,
    total_time_in_system_ticks: usize,
    completed_items: Vec<MutexWorkItem>,
    events: TraceLog,
}

impl MutexProcessorSubstation {
    fn new(id: &str, owner_id: &str, processing_ticks: usize, events: TraceLog) -> Self {
        MutexProcessorSubstation {
            core: StationCore::new(id),
            owner_id: owner_id.to_string(),
            processing_ticks: processing_ticks as i64,
            ready: Vec::new(),
            active: None,
            tick: 0,
            completed: 0,
            child_releases: 0,
            total_time_in_system_ticks: 0,
            completed_items: Vec::new(),
            events,
        }
    }

    fn stats(&self) -> MutexProcessorStats {
        MutexProcessorStats {
            completed: self.completed,
            final_queue: self.ready.len() + usize::from(self.active.is_some()),
            mean_time_in_system_ticks: self.total_time_in_system_ticks as f64 / (self.completed.max(1)) as f64,
            child_releases_spawned: self.child_releases,
        }
    }

    fn completed_items_view(&self) -> &[MutexWorkItem] {
        &self.completed_items
    }
}

impl DESStation for MutexProcessorSubstation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        Preconditions::integer_in_range(
            "network-mutex",
            &format!("{}.processingTicks", self.core.id),
            self.processing_ticks as f64,
            1.0,
            1e9,
        )
        .expect("processingTicks");
    }
    fn has_work(&self) -> bool {
        self.core.has_work() || !self.ready.is_empty() || self.active.is_some()
    }
    fn run_time_step(&mut self) {
        let id = self.core.id.clone();
        let drained = self.core.drain::<LockedWorkToken>(MUTEX_LOCKED_WORK_CHANNEL);
        self.ready.extend(drained.into_iter().map(|rc| (*rc).clone()));

        if self.active.is_none() && !self.ready.is_empty() {
            let mut next = self.ready.remove(0);
            transition_token(
                &mut next.item.token,
                MutexWorkState::Processing,
                TransitionTokenOpts {
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: NetworkMutexEventKind::ProcessingStarted.as_str().to_string(),
                    detail: None,
                },
            );
            let mut lock = lock_with(
                next.item.lock.take(),
                &next.grant.token_id,
                next.grant.service_request_queued_tick,
            );
            lock.processing_started_tick = Some(self.tick);
            next.item.lock = Some(lock);
            let item_id = next.item.item_id.clone();
            let token_id = next.grant.token_id.clone();
            self.active = Some(ActiveWork { item: next.item, grant: next.grant, remaining_ticks: self.processing_ticks });
            trace(
                &self.events,
                self.tick,
                &id,
                NetworkMutexEventKind::ProcessingStarted,
                Some(item_id),
                Some(token_id),
                None,
            );
        }

        let finished = match self.active.as_mut() {
            Some(active) => {
                active.remaining_ticks -= 1;
                active.remaining_ticks <= 0
            }
            None => false,
        };
        if finished {
            let active = self.active.take().unwrap();
            let mut item = active.item;
            let token_id = active.grant.token_id.clone();
            transition_token(
                &mut item.token,
                MutexWorkState::Releasing,
                TransitionTokenOpts {
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: NetworkMutexEventKind::ProcessingFinished.as_str().to_string(),
                    detail: None,
                },
            );
            let mut lock = lock_with(item.lock.take(), &token_id, active.grant.service_request_queued_tick);
            lock.processing_finished_tick = Some(self.tick);
            lock.released_tick = Some(self.tick);
            item.lock = Some(lock);
            let release_token = spawn_stateful_child_token::<MutexChildState>(
                &item.token.lineage,
                SpawnStatefulChildTokenOpts {
                    kind: "lock-release".to_string(),
                    token_id: format!("{token_id}:release"),
                    initial_state: MutexChildState::Released,
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: Some(NetworkMutexEventKind::ReleaseSpawned.as_str().to_string()),
                    detail: None,
                },
            );
            let release = LockReleaseToken {
                token: release_token,
                parent_item_id: item.item_id.clone(),
                token_id: token_id.clone(),
                owner_id: self.owner_id.clone(),
                created_tick: self.tick,
                released_tick: self.tick,
            };
            self.child_releases += 1;
            self.core.emit(Rc::new(release), MUTEX_RELEASE_CHANNEL);
            transition_token(
                &mut item.token,
                MutexWorkState::Completed,
                TransitionTokenOpts {
                    tick: self.tick as f64,
                    station_id: id.clone(),
                    event: NetworkMutexEventKind::WorkCompleted.as_str().to_string(),
                    detail: None,
                },
            );
            let item_id = item.item_id.clone();
            self.core.emit(Rc::new(item.clone()), MUTEX_DONE_CHANNEL);
            self.completed += 1;
            self.total_time_in_system_ticks += (self.tick + 1).saturating_sub(item.created_tick);
            self.completed_items.push(item);
            trace(
                &self.events,
                self.tick,
                &id,
                NetworkMutexEventKind::ProcessingFinished,
                Some(item_id.clone()),
                Some(token_id.clone()),
                None,
            );
            trace(
                &self.events,
                self.tick,
                &id,
                NetworkMutexEventKind::ReleaseSpawned,
                Some(item_id.clone()),
                Some(token_id.clone()),
                None,
            );
            trace(
                &self.events,
                self.tick,
                &id,
                NetworkMutexEventKind::WorkCompleted,
                Some(item_id),
                Some(token_id),
                None,
            );
        }
        self.tick += 1;
    }
}

// -----------------------------------------------------------------------------
// WORKER (Station A) — composite of queue + processor substations
// -----------------------------------------------------------------------------

pub struct NetworkMutexWorkerStation {
    composite: CompositeDESStation,
    queue_station: Rc<RefCell<MutexQueueSubstation>>,
    processor_station: Rc<RefCell<MutexProcessorSubstation>>,
}

impl NetworkMutexWorkerStation {
    pub fn new(id: &str, opts: NetworkMutexWorkerOpts, events: TraceLog) -> Self {
        let mut composite = CompositeDESStation::new(id);
        let queue_station = composite.add_substation(Rc::new(RefCell::new(MutexQueueSubstation::new(
            &format!("{id}:queue"),
            id,
            events.clone(),
        ))));
        let processor_station = composite.add_substation(Rc::new(RefCell::new(MutexProcessorSubstation::new(
            &format!("{id}:processor"),
            id,
            opts.processing_ticks,
            events,
        ))));
        composite.expose_input(MUTEX_WORK_CHANNEL, queue_station.clone() as StationRef, MUTEX_WORK_CHANNEL);
        composite.expose_input(MUTEX_GRANT_CHANNEL, queue_station.clone() as StationRef, MUTEX_GRANT_CHANNEL);
        queue_station.borrow_mut().core_mut().pipe(
            processor_station.clone() as StationRef,
            MUTEX_LOCKED_WORK_CHANNEL,
            MUTEX_LOCKED_WORK_CHANNEL,
        );
        composite.expose_output(queue_station.clone() as StationRef, MUTEX_REQUEST_CHANNEL, MUTEX_REQUEST_CHANNEL);
        composite.expose_output(processor_station.clone() as StationRef, MUTEX_RELEASE_CHANNEL, MUTEX_RELEASE_CHANNEL);
        composite.expose_output(processor_station.clone() as StationRef, MUTEX_DONE_CHANNEL, MUTEX_DONE_CHANNEL);

        let mut worker = NetworkMutexWorkerStation { composite, queue_station, processor_station };
        // Worker-level validator: completed item ids unique. Registered via the
        // DEFAULT add_validator so it lands in `core().validators` (which the
        // runner's `run_station_validation` reads) rather than the composite's
        // `own_validators`.
        let v = intrinsic_check::<dyn DESStation>(
            "network-mutex.worker.no-duplicate-completions",
            |s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<NetworkMutexWorkerStation>().unwrap();
                let ids = st.completed_item_ids();
                ids.iter().cloned().collect::<HashSet<_>>().len() == ids.len()
            },
            Some("completed item ids unique".to_string()),
            Some(Box::new(|s: &dyn DESStation| {
                let st = s.as_any().downcast_ref::<NetworkMutexWorkerStation>().unwrap();
                format!("completed={}", st.completed_item_ids().join(","))
            })),
            Some("network-mutex".to_string()),
            None,
        );
        worker.add_validator(v.boxed());
        worker
    }

    pub fn stats(&self) -> NetworkMutexWorkerStats {
        let processor = self.processor_station.borrow().stats();
        let queue = self.queue_station.borrow().stats(processor.completed);
        NetworkMutexWorkerStats {
            arrived: queue.arrived,
            completed: processor.completed,
            final_queue: queue.final_queue + processor.final_queue,
            max_queue: queue.max_queue,
            mean_queue_wait_ticks: queue.mean_queue_wait_ticks,
            mean_lock_wait_ticks: queue.mean_lock_wait_ticks,
            mean_time_in_system_ticks: processor.mean_time_in_system_ticks,
            child_requests_spawned: queue.child_requests_spawned,
            child_releases_spawned: processor.child_releases_spawned,
        }
    }

    pub fn completed_items_view(&self) -> Vec<MutexWorkItem> {
        self.processor_station.borrow().completed_items_view().to_vec()
    }

    fn completed_item_ids(&self) -> Vec<String> {
        self.processor_station.borrow().completed_items_view().iter().map(|x| x.item_id.clone()).collect()
    }
}

impl DESStation for NetworkMutexWorkerStation {
    fn core(&self) -> &StationCore {
        self.composite.core()
    }
    fn core_mut(&mut self) -> &mut StationCore {
        self.composite.core_mut()
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn assert_preconditions(&mut self) {
        self.composite.assert_preconditions();
    }
    fn has_work(&self) -> bool {
        self.composite.has_work()
    }
    fn run_time_step(&mut self) {
        self.composite.run_time_step();
    }
    fn on_finalize(&mut self) {
        self.composite.on_finalize();
    }
}

// -----------------------------------------------------------------------------
// SINK (Station C)
// -----------------------------------------------------------------------------

pub struct MutexCompletionSinkStation {
    core: StationCore,
    pub completed: Vec<MutexWorkItem>,
}

impl MutexCompletionSinkStation {
    pub fn new(id: &str) -> Self {
        MutexCompletionSinkStation { core: StationCore::new(id), completed: Vec::new() }
    }
}

impl DESStation for MutexCompletionSinkStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {
        let drained = self.core.drain::<MutexWorkItem>(MUTEX_DONE_CHANNEL);
        self.completed.extend(drained.into_iter().map(|rc| (*rc).clone()));
    }
}

// -----------------------------------------------------------------------------
// BUILD + RUN
// -----------------------------------------------------------------------------

/// The wired-up stations of a mutex network, with the shared trace log.
pub struct NetworkMutexStations {
    pub source: Rc<RefCell<MutexWorkSourceStation>>,
    pub worker: Rc<RefCell<NetworkMutexWorkerStation>>,
    pub lock: Rc<RefCell<NetworkMutexLockServiceStation>>,
    pub sink: Rc<RefCell<MutexCompletionSinkStation>>,
    pub events: TraceLog,
}

/// Construct and wire the four stations of a mutex network.
pub fn build_network_mutex_stations(opts: NetworkMutexSimulationOpts) -> NetworkMutexStations {
    let events: TraceLog = Rc::new(RefCell::new(Vec::new()));
    let source = Rc::new(RefCell::new(MutexWorkSourceStation::new(
        "source",
        opts.source.unwrap_or(MutexSourceSpec { count: 8, interarrival_ticks: 1, first_arrival_tick: None }),
    )));
    let worker = Rc::new(RefCell::new(NetworkMutexWorkerStation::new(
        "station-A",
        opts.worker.unwrap_or(NetworkMutexWorkerOpts { processing_ticks: 4 }),
        events.clone(),
    )));
    let lock = Rc::new(RefCell::new(NetworkMutexLockServiceStation::new(
        "station-B-lock",
        opts.lock.unwrap_or(NetworkMutexLockServiceOpts { grant_delay_ticks: Some(2) }),
        events.clone(),
    )));
    let sink = Rc::new(RefCell::new(MutexCompletionSinkStation::new("station-C")));

    source.borrow_mut().core_mut().pipe(worker.clone() as StationRef, MUTEX_WORK_CHANNEL, MUTEX_WORK_CHANNEL);
    worker.borrow_mut().core_mut().pipe(lock.clone() as StationRef, MUTEX_REQUEST_CHANNEL, MUTEX_REQUEST_CHANNEL);
    worker.borrow_mut().core_mut().pipe(lock.clone() as StationRef, MUTEX_RELEASE_CHANNEL, MUTEX_RELEASE_CHANNEL);
    lock.borrow_mut().core_mut().pipe(worker.clone() as StationRef, MUTEX_GRANT_CHANNEL, MUTEX_GRANT_CHANNEL);
    worker.borrow_mut().core_mut().pipe(sink.clone() as StationRef, MUTEX_DONE_CHANNEL, MUTEX_DONE_CHANNEL);

    NetworkMutexStations { source, worker, lock, sink, events }
}

/// Run the mutex network to quiescence and reduce it to summary statistics.
pub fn run_network_mutex_simulation(opts: NetworkMutexSimulationOpts) -> NetworkMutexSimulationResult {
    let max_ticks = opts.max_ticks.unwrap_or(10_000);
    let stations = build_network_mutex_stations(opts);
    let NetworkMutexStations { source, worker, lock, sink, events } = stations;

    let participants: Vec<StationRef> = vec![source.clone(), worker.clone(), lock.clone(), sink.clone()];
    let summary = run_iterative_des(
        participants,
        IterativeRunOptions { shuffle: false, max_ticks: Some(max_ticks), ..Default::default() },
    );

    let mut invariant_violations: Vec<String> = Vec::new();
    if summary.reason == Some(RunReason::MaxTicks) {
        invariant_violations.push(format!("network mutex reached maxTicks={max_ticks}"));
    }
    let generated = source.borrow().emitted_count();
    let completed = sink.borrow().completed.len();
    if completed != generated {
        invariant_violations.push(format!("completed {completed} != generated {generated}"));
    }
    let lock_stats = lock.borrow().stats(summary.ticks);
    if lock_stats.invalid_release_count > 0 {
        invariant_violations.push(format!("invalid releases: {}", lock_stats.invalid_release_count));
    }
    for c in failed_validation_checks(&summary) {
        let detail = c.details.or(c.observed).unwrap_or_else(|| "failed".to_string());
        invariant_violations.push(format!("{}: {}", c.name, detail));
    }

    let worker_stats = worker.borrow().stats();
    let completed_items = sink.borrow().completed.clone();
    let trace = events.borrow().clone();

    NetworkMutexSimulationResult {
        generated,
        completed,
        total_ticks: summary.ticks,
        worker: worker_stats,
        lock: lock_stats,
        completed_items,
        trace,
        invariant_violations,
    }
}

#[cfg(test)]
mod tests {
    //! Network-mutex tests: the lock service must grant exclusive access (every
    //! item completes, no invalid releases) and the run must conserve items
    //! (completed == generated) with the single-holder invariant intact.

    use super::*;

    #[test]
    fn grants_exclusive_access_and_completes_all() {
        let res = run_network_mutex_simulation(NetworkMutexSimulationOpts {
            source: Some(MutexSourceSpec { count: 5, interarrival_ticks: 1, first_arrival_tick: None }),
            worker: Some(NetworkMutexWorkerOpts { processing_ticks: 3 }),
            lock: Some(NetworkMutexLockServiceOpts { grant_delay_ticks: Some(2) }),
            max_ticks: Some(1000),
        });
        assert_eq!(res.generated, 5);
        assert_eq!(res.completed, 5);
        // Exclusive access ⇒ no invalid releases and one grant/release per item.
        assert_eq!(res.lock.invalid_release_count, 0);
        assert_eq!(res.lock.grant_count, 5);
        assert_eq!(res.lock.release_count, 5);
        assert!(res.invariant_violations.is_empty(), "violations: {:?}", res.invariant_violations);
    }

    #[test]
    fn completed_item_ids_are_unique() {
        let res = run_network_mutex_simulation(NetworkMutexSimulationOpts {
            source: Some(MutexSourceSpec { count: 6, interarrival_ticks: 2, first_arrival_tick: None }),
            ..Default::default()
        });
        let ids: HashSet<_> = res.completed_items.iter().map(|i| i.item_id.clone()).collect();
        assert_eq!(ids.len(), res.completed_items.len());
        // Every completed item ended in the terminal `Completed` state.
        for item in &res.completed_items {
            assert_eq!(item.token.current_state, Some(MutexWorkState::Completed));
        }
    }

    #[test]
    fn empty_source_completes_immediately() {
        let res = run_network_mutex_simulation(NetworkMutexSimulationOpts {
            source: Some(MutexSourceSpec { count: 0, interarrival_ticks: 1, first_arrival_tick: None }),
            ..Default::default()
        });
        assert_eq!(res.generated, 0);
        assert_eq!(res.completed, 0);
        assert!(res.invariant_violations.is_empty());
    }
}
