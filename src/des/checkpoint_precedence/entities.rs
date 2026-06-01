//! Canonical use path: `crate::des::checkpoint_precedence::entities::*`
//!
//! The three station/token types for the checkpoint-precedence demo:
//!
//! * [`LabeledToken`] — a movable with a *caller-chosen* UUID (so constraints can
//!   reference it by name) carrying a numeric payload.
//! * [`OrderedTokenSource`] — emits a fixed list of `LabeledToken`s (one per
//!   tick), registering each token's stamp + constraints into the shared
//!   [`PrecedenceLedger`].
//! * [`CheckpointGate`] — the enforcer station: it parks arriving tokens in a
//!   **balanced BST** keyed by `seq` ([`BTreeMap`]) and, each tick, releases the
//!   lowest-`seq` token whose predecessor constraints are all satisfied, recording
//!   the clearance. This is the token-level ordering brain.
//! * [`RecordingSink`] — records each absorbed token's payload, so the produced
//!   order can be asserted exactly.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::{BTreeMap, VecDeque};
use std::rc::Rc;

use crate::des::checkpoint_precedence::ledger::{PrecedenceLedger, Requirement, TokenSpec};
use crate::des::entity_moving::moving::{MovingCore, MovingEntity, MovingValue};
use crate::des::entity_queue::queue::{build_in_conn, build_out_conn};
use crate::des::entity_sink::sink::{AbstractSinkEntity, SinkCore, SinkKind};
use crate::des::entity_source::source::SourceCore;
use crate::des::r#abstract::interfaces::{
    EntityGraphData, HasInput, HasInternalQueue, HasManyInputConnections, HasManyOutputConnections,
    HasOutput,
};
use crate::des::r#abstract::r#abstract::{BidirectionalCore, Entity, EntityConnection, EntityCore};
use crate::des::shared::precision::Decimal;

// =============================================================================
// LabeledToken — a movable whose UUID we choose.
// =============================================================================

/// A token whose `moving_uuid` is a caller-chosen label (e.g. `"T1"`), so other
/// tokens can declare constraints that reference it.
pub struct LabeledToken {
    pub core: MovingCore,
    pub value: f64,
}

impl LabeledToken {
    pub fn new(uuid: &str, value: f64) -> Self {
        LabeledToken {
            core: MovingCore::new(uuid.to_string()),
            value,
        }
    }
}

impl Entity for LabeledToken {
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
        EntityGraphData::default().with("value", self.value)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        // Tokens are carried, not stepped by the node loop; keep it inert.
        self.core.entity.time_step_count += 1;
    }
}

impl MovingEntity for LabeledToken {
    fn moving_core(&self) -> &MovingCore {
        &self.core
    }
    fn moving_core_mut(&mut self) -> &mut MovingCore {
        &mut self.core
    }
    fn get_value(&self) -> MovingValue {
        MovingValue {
            id: Some(self.core.entity.id.clone()),
            value: Some(self.value),
            q: Some(self.value),
        }
    }
    fn run_finish(&mut self) {}
}

// =============================================================================
// OrderedTokenSource — emits a prepared list, one token per tick.
// =============================================================================

/// A `(uuid, payload, requirements)` triple staged for emission.
pub struct PreparedToken {
    pub uuid: String,
    pub payload: f64,
    pub requirements: Vec<Requirement>,
}

impl PreparedToken {
    pub fn new(uuid: &str, payload: f64, requirements: Vec<Requirement>) -> Self {
        PreparedToken {
            uuid: uuid.to_string(),
            payload,
            requirements,
        }
    }
}

/// Output-only source. At construction it stamps each prepared token with a
/// monotonic `seq` and registers it in the shared ledger (so the whole precedence
/// graph is known and can be validated before a single tick runs). Each tick it
/// mints and emits the next token.
pub struct OrderedTokenSource {
    pub core: SourceCore,
    ledger: Rc<RefCell<PrecedenceLedger>>,
    pending: VecDeque<(u64, PreparedToken)>,
    emitted_order: Vec<String>,
}

impl OrderedTokenSource {
    pub fn new(
        id: String,
        ledger: Rc<RefCell<PrecedenceLedger>>,
        tokens: Vec<PreparedToken>,
    ) -> Self {
        let mut pending = VecDeque::new();
        {
            let mut led = ledger.borrow_mut();
            for (i, t) in tokens.into_iter().enumerate() {
                let seq = (i as u64) + 1;
                led.register(TokenSpec {
                    uuid: t.uuid.clone(),
                    seq,
                    payload: t.payload,
                    requirements: t.requirements.clone(),
                });
                pending.push_back((seq, t));
            }
        }
        OrderedTokenSource {
            core: SourceCore::new(id),
            ledger,
            pending,
            emitted_order: Vec::new(),
        }
    }

    pub fn emitted_order(&self) -> &[String] {
        &self.emitted_order
    }

    pub fn is_drained(&self) -> bool {
        self.pending.is_empty()
    }
}

impl Entity for OrderedTokenSource {
    fn core(&self) -> &EntityCore {
        &self.core.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        false
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default().with("emittedCount", self.emitted_order.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.core.entity.time_step_count += 1;
        let (_seq, prepared) = match self.pending.pop_front() {
            Some(x) => x,
            None => return,
        };

        let mut tok = LabeledToken::new(&prepared.uuid, prepared.payload);
        tok.init();
        let m: Rc<RefCell<dyn MovingEntity>> = Rc::new(RefCell::new(tok));
        self.emitted_order.push(prepared.uuid.clone());

        let conns = self.core.get_out_connections();
        let mut accepted = false;
        for conn in &conns {
            if let Some(t) = conn.borrow().get_target() {
                if t.borrow_mut().accept_item(m.clone()) {
                    t.borrow_mut().take_item(m.clone());
                    accepted = true;
                    break;
                }
            }
        }
        if !accepted {
            eprintln!(
                "[ordered-source:{}] no downstream accepted token {}",
                self.core.entity.id, prepared.uuid
            );
        }
    }
}

impl HasOutput for OrderedTokenSource {
    fn id(&self) -> String {
        self.core.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        Some(self.core.add_out_connection_to(target))
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_targets(&mut self) {}
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl HasManyOutputConnections for OrderedTokenSource {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.core.get_out_connections()
    }
}

// =============================================================================
// CheckpointGate — the BST-backed waiting room that enforces token order.
// =============================================================================

/// A bidirectional station that holds tokens until their precedence constraints
/// are met, then releases them downstream in deterministic order.
///
/// Waiting tokens live in a `BTreeMap<seq, token>` — a balanced BST whose ordered
/// iteration yields the lowest-`seq` token first, giving the deterministic
/// tie-break among simultaneously-eligible tokens in `O(log n)` per insert/remove.
pub struct CheckpointGate {
    pub bi: BidirectionalCore,
    waiting: BTreeMap<u64, Rc<RefCell<dyn MovingEntity>>>,
    ledger: Rc<RefCell<PrecedenceLedger>>,
    released_order: Vec<String>,
}

impl CheckpointGate {
    pub fn new(id: String, ledger: Rc<RefCell<PrecedenceLedger>>) -> Self {
        CheckpointGate {
            bi: BidirectionalCore::new(id),
            waiting: BTreeMap::new(),
            ledger,
            released_order: Vec::new(),
        }
    }

    /// UUIDs in the order they were released through this checkpoint.
    pub fn released_order(&self) -> &[String] {
        &self.released_order
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }

    fn forward(&self, tok: Rc<RefCell<dyn MovingEntity>>) {
        let conns = self.bi.get_out_connections();
        let mut accepted = false;
        for conn in &conns {
            if let Some(t) = conn.borrow().get_target() {
                if t.borrow_mut().accept_item(tok.clone()) {
                    t.borrow_mut().take_item(tok.clone());
                    accepted = true;
                    break;
                }
            }
        }
        if !accepted {
            eprintln!(
                "[gate:{}] no downstream accepted a released token",
                self.bi.entity.id
            );
        }
    }
}

impl Entity for CheckpointGate {
    fn core(&self) -> &EntityCore {
        &self.bi.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.bi.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        true
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default()
            .with("waiting", self.waiting.len() as f64)
            .with("released", self.released_order.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.bi.entity.time_step_count += 1;
        let checkpoint = self.bi.entity.id.clone();

        // Release every currently-eligible token, lowest `seq` first. Releasing a
        // token may unblock a lower-`seq` successor, so we re-scan after each
        // release until nothing is eligible.
        loop {
            let next_seq = {
                let ledger = self.ledger.borrow();
                self.waiting.iter().find_map(|(seq, tok)| {
                    let uuid = tok.borrow().moving_core().moving_uuid.clone();
                    if ledger.requirements_satisfied(&uuid, &checkpoint) {
                        Some(*seq)
                    } else {
                        None
                    }
                })
            };

            let seq = match next_seq {
                Some(s) => s,
                None => break,
            };

            let tok = self
                .waiting
                .remove(&seq)
                .expect("eligible seq must be present");
            let uuid = tok.borrow().moving_core().moving_uuid.clone();
            tok.borrow_mut().add_visited_station(&checkpoint);
            self.ledger.borrow_mut().mark_cleared(&checkpoint, &uuid);
            self.released_order.push(uuid);
            self.forward(tok);
        }
    }
}

impl HasInternalQueue for CheckpointGate {
    fn max_queue_size(&self) -> usize {
        usize::MAX
    }
    fn is_full(&self) -> bool {
        false
    }
    fn is_empty(&self) -> bool {
        self.waiting.is_empty()
    }
}

impl HasInput for CheckpointGate {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        true
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        let uuid = m.borrow().moving_core().moving_uuid.clone();
        let seq = self.ledger.borrow().seq_of(&uuid).unwrap_or_else(|| {
            panic!(
                "token '{uuid}' reached gate '{}' but was never registered in the precedence ledger",
                self.bi.entity.id
            )
        });
        self.waiting.insert(seq, m);
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_sources(&mut self) {
        self.bi.notify_sources();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
    fn add_in_connection(
        &mut self,
        _source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        let conn = build_in_conn();
        self.bi.add_in_connection(conn.clone());
        Some(conn)
    }
}

impl HasManyInputConnections for CheckpointGate {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_in_connections()
    }
}

impl HasOutput for CheckpointGate {
    fn id(&self) -> String {
        self.bi.entity.id.clone()
    }
    fn add_out_connection(
        &mut self,
        target: Rc<RefCell<dyn HasInput>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        let conn = build_out_conn(target);
        self.bi.add_out_connection(conn.clone());
        Some(conn)
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        true
    }
    fn notify_targets(&mut self) {
        self.bi.notify_targets();
    }
    fn do_setup_after_output_conn(&mut self) -> bool {
        true
    }
}

impl HasManyOutputConnections for CheckpointGate {
    fn get_out_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.bi.get_out_connections()
    }
}

// =============================================================================
// RecordingSink — records each absorbed payload.
// =============================================================================

/// A sink that appends each absorbed token's payload to `recorded`, so the
/// produced order can be asserted exactly.
pub struct RecordingSink {
    pub core: SinkCore,
    pub kind: SinkKind,
    pub recorded: Vec<f64>,
}

impl RecordingSink {
    pub fn new(id: String) -> Self {
        RecordingSink {
            core: SinkCore::new(id),
            kind: SinkKind::Sink,
            recorded: Vec::new(),
        }
    }
}

impl AbstractSinkEntity for RecordingSink {}

impl Entity for RecordingSink {
    fn core(&self) -> &EntityCore {
        &self.core.entity
    }
    fn core_mut(&mut self) -> &mut EntityCore {
        &mut self.core.entity
    }
    fn do_validation(&mut self) {}
    fn do_validation_before_run(&mut self) -> bool {
        false
    }
    fn get_graph_data(&self) -> EntityGraphData {
        EntityGraphData::default().with("recordedCount", self.recorded.len() as f64)
    }
    fn run_time_step(&mut self, _step_size: Decimal) {
        self.core.entity.time_step_count += 1;
    }
}

impl HasInput for RecordingSink {
    fn id(&self) -> String {
        self.core.entity.id.clone()
    }
    fn accept_item(&mut self, _m: Rc<RefCell<dyn MovingEntity>>) -> bool {
        true
    }
    fn take_item(&mut self, m: Rc<RefCell<dyn MovingEntity>>) {
        let q = m.borrow().get_value().q;
        if let Some(v) = q {
            self.recorded.push(v);
        }
        m.borrow_mut().do_finish();
    }
    fn do_setup_after_input_conn(&mut self) -> bool {
        false
    }
    fn notify_sources(&mut self) {}
    fn do_setup_after_output_conn(&mut self) -> bool {
        false
    }
    fn add_in_connection(
        &mut self,
        source: Rc<RefCell<dyn HasManyOutputConnections>>,
    ) -> Option<Rc<RefCell<EntityConnection>>> {
        Some(self.core.add_in_connection_from(source))
    }
}

impl HasManyInputConnections for RecordingSink {
    fn get_in_connections(&self) -> Vec<Rc<RefCell<EntityConnection>>> {
        self.core.get_in_connections()
    }
}
