//! Port of `src/des/general/control-systems/observability-controllability.ts` —
//! a general evaluator for the two structural properties of dynamical systems:
//! CONTROLLABILITY (can an input drive the state anywhere?) and OBSERVABILITY
//! (can the output reveal the full internal state?).
//!
//! Three lenses:
//!   1. Linear state-space (StateSpaceModel): controllable iff the Kalman
//!      controllability matrix has full rank n; observable iff the observability
//!      matrix has full rank n.
//!   2. MDP (MarkovDecisionProcess): the analog of controllability is
//!      reachability — the transitive closure of the controlled transition graph
//!      being complete (strongly connected).
//!   3. POMDP (PartiallyObservableProcess): the analog of observability is state
//!      distinguishability, answered by partition refinement — split states on
//!      their observation distribution, then iteratively split blocks whose
//!      members transition under some action to different blocks until stable.
//!
//! Everything is expressed as types with methods (LinAlg / MatrixRank do the
//! numeric linear algebra). The DES pipeline wires sources to
//! `PureTransformEntity` evaluators to a sink so verdicts flow as tokens on
//! named channels. `throw` invariants become `panic!`; matrices are `f64`.
#![allow(dead_code)]

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use super::linear_algebra::{LinAlg, Matrix, MatrixRank};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::station::{DESStation, StationCore};
use crate::des::general::des_base::transform_entity::{
    OutputChannel, PureTransformEntity, TransformContext, TransformEntity, TransformEntityCore,
    TransformEntityOptions, TransformResult,
};

/// `Preconditions` return `Result`; an invariant violation is fatal -> `panic!`.
fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

// =============================================================================
// 1. LINEAR STATE-SPACE MODEL
// =============================================================================

/// Specification for a linear time-invariant model (TS `interface StateSpaceSpec`).
#[derive(Clone, Debug)]
pub struct StateSpaceSpec {
    /// state matrix A (n×n)
    pub a: Matrix,
    /// input matrix B (n×m)
    pub b: Matrix,
    /// output matrix C (p×n)
    pub c: Matrix,
    /// feedthrough D (p×m). `None` -> zero.
    pub d: Option<Matrix>,
}

/// A linear time-invariant model with the Kalman controllability/observability
/// tests as methods.
#[derive(Clone, Debug)]
pub struct StateSpaceModel {
    pub a: Matrix,
    pub b: Matrix,
    pub c: Matrix,
    pub d: Matrix,
}

impl StateSpaceModel {
    pub fn new(spec: StateSpaceSpec) -> Self {
        let cls = "StateSpaceModel";
        require(Preconditions::square_matrix(cls, "A", &spec.a));
        let n = spec.a.len();
        require(Preconditions::rectangular_matrix(cls, "B", &spec.b));
        require(Preconditions::length_eq(cls, "B", &spec.b, n));
        require(Preconditions::rectangular_matrix(cls, "C", &spec.c));
        require(Preconditions::length_eq(cls, "C[0]", &spec.c[0], n));
        let d = match &spec.d {
            Some(d) => LinAlg::copy(d),
            None => LinAlg::zeros(spec.c.len(), spec.b[0].len()),
        };
        StateSpaceModel {
            a: LinAlg::copy(&spec.a),
            b: LinAlg::copy(&spec.b),
            c: LinAlg::copy(&spec.c),
            d,
        }
    }

    /// State dimension n.
    pub fn state_dim(&self) -> usize {
        self.a.len()
    }

    /// Input dimension m.
    pub fn input_dim(&self) -> usize {
        LinAlg::cols(&self.b)
    }

    /// Output dimension p.
    pub fn output_dim(&self) -> usize {
        self.c.len()
    }

    /// Controllability matrix [ B  AB  A²B … Aⁿ⁻¹B ]  (n × n·m).
    pub fn controllability_matrix(&self) -> Matrix {
        let n = self.state_dim();
        let mut blocks: Vec<Matrix> = Vec::new();
        let mut a_power_b = self.b.clone(); // A⁰B = B
        blocks.push(a_power_b.clone());
        for _k in 1..n {
            a_power_b = LinAlg::mat_mul(&self.a, &a_power_b);
            blocks.push(a_power_b.clone());
        }
        LinAlg::hstack(&blocks)
    }

    /// Observability matrix [ C; CA; CA²; …; CAⁿ⁻¹ ]  (n·p × n).
    pub fn observability_matrix(&self) -> Matrix {
        let n = self.state_dim();
        let mut blocks: Vec<Matrix> = Vec::new();
        let mut c_a_power = self.c.clone(); // C·A⁰ = C
        blocks.push(c_a_power.clone());
        for _k in 1..n {
            c_a_power = LinAlg::mat_mul(&c_a_power, &self.a);
            blocks.push(c_a_power.clone());
        }
        LinAlg::vstack(&blocks)
    }

    pub fn controllability_rank(&self) -> usize {
        LinAlg::rank(&self.controllability_matrix(), None)
    }

    pub fn observability_rank(&self) -> usize {
        LinAlg::rank(&self.observability_matrix(), None)
    }

    pub fn is_controllable(&self) -> bool {
        MatrixRank::new(&self.controllability_matrix(), None).is_full_rank(self.state_dim())
    }

    pub fn is_observable(&self) -> bool {
        MatrixRank::new(&self.observability_matrix(), None).is_full_rank(self.state_dim())
    }
}

// =============================================================================
// 2. MARKOV DECISION PROCESS — reachability ("controllability")
// =============================================================================

/// `interface MdpSpec`.
#[derive(Clone, Debug)]
pub struct MdpSpec {
    /// number of states S
    pub num_states: usize,
    /// number of actions A
    pub num_actions: usize,
    /// transition[a][s][s'] = P(s' | s, a). Each transition[a][s] is a pmf.
    pub transition: Vec<Vec<Vec<f64>>>,
}

/// A finite MDP whose structural-controllability test is reachability of the
/// controlled transition graph (transitive closure).
#[derive(Clone, Debug)]
pub struct MarkovDecisionProcess {
    pub num_states: usize,
    pub num_actions: usize,
    pub transition: Vec<Vec<Vec<f64>>>,
}

impl MarkovDecisionProcess {
    pub fn new(spec: MdpSpec) -> Self {
        let cls = "MarkovDecisionProcess";
        require(Preconditions::integer_in_range(cls, "numStates", spec.num_states as f64, 1.0, 100_000.0));
        require(Preconditions::integer_in_range(cls, "numActions", spec.num_actions as f64, 1.0, 100_000.0));
        require(Preconditions::length_eq(cls, "transition", &spec.transition, spec.num_actions));
        for a in 0..spec.num_actions {
            require(Preconditions::length_eq(cls, &format!("transition[{a}]"), &spec.transition[a], spec.num_states));
            for s in 0..spec.num_states {
                require(Preconditions::probability_vector(
                    cls,
                    &format!("transition[{a}][{s}]"),
                    &spec.transition[a][s],
                    1e-6,
                ));
            }
        }
        MarkovDecisionProcess {
            num_states: spec.num_states,
            num_actions: spec.num_actions,
            transition: spec.transition,
        }
    }

    /// One-step adjacency: edge s → s' iff some action gives positive probability.
    pub fn one_step_adjacency(&self) -> Vec<Vec<bool>> {
        let n = self.num_states;
        let mut adj = vec![vec![false; n]; n];
        for a in 0..self.num_actions {
            for s in 0..n {
                for t in 0..n {
                    if self.transition[a][s][t] > 1e-12 {
                        adj[s][t] = true;
                    }
                }
            }
        }
        adj
    }

    /// Reachability closure (Floyd–Warshall transitive closure; diagonal true).
    pub fn reachability_closure(&self) -> Vec<Vec<bool>> {
        let n = self.num_states;
        let mut reach = self.one_step_adjacency();
        for s in 0..n {
            reach[s][s] = true;
        }
        for k in 0..n {
            for i in 0..n {
                if !reach[i][k] {
                    continue;
                }
                for j in 0..n {
                    if reach[k][j] {
                        reach[i][j] = true;
                    }
                }
            }
        }
        reach
    }

    /// Structurally controllable iff every state is reachable from every state.
    pub fn is_structurally_controllable(&self) -> bool {
        let reach = self.reachability_closure();
        for i in 0..self.num_states {
            for j in 0..self.num_states {
                if !reach[i][j] {
                    return false;
                }
            }
        }
        true
    }

    /// Count of reachable ordered (s, t) pairs — a controllability "degree".
    pub fn reachable_pair_count(&self) -> usize {
        let reach = self.reachability_closure();
        let mut c = 0;
        for row in &reach {
            for &v in row {
                if v {
                    c += 1;
                }
            }
        }
        c
    }
}

// =============================================================================
// 3. PARTIALLY OBSERVABLE PROCESS — distinguishability ("observability")
// =============================================================================

/// `interface PomdpSpec extends MdpSpec` — flattened (no interface inheritance).
#[derive(Clone, Debug)]
pub struct PomdpSpec {
    pub num_states: usize,
    pub num_actions: usize,
    pub transition: Vec<Vec<Vec<f64>>>,
    /// number of distinct observations O
    pub num_observations: usize,
    /// observation[s][o] = P(o | s). Each observation[s] is a pmf.
    pub observation: Vec<Vec<f64>>,
}

/// A finite POMDP whose structural-observability test is whether the observation
/// process can eventually distinguish every pair of states.
#[derive(Clone, Debug)]
pub struct PartiallyObservableProcess {
    pub mdp: MarkovDecisionProcess,
    pub num_observations: usize,
    pub observation: Vec<Vec<f64>>,
}

impl PartiallyObservableProcess {
    pub fn new(spec: PomdpSpec) -> Self {
        let mdp = MarkovDecisionProcess::new(MdpSpec {
            num_states: spec.num_states,
            num_actions: spec.num_actions,
            transition: spec.transition,
        });
        let cls = "PartiallyObservableProcess";
        require(Preconditions::integer_in_range(cls, "numObservations", spec.num_observations as f64, 1.0, 100_000.0));
        require(Preconditions::length_eq(cls, "observation", &spec.observation, spec.num_states));
        for s in 0..spec.num_states {
            require(Preconditions::probability_vector(
                cls,
                &format!("observation[{s}]"),
                &spec.observation[s],
                1e-6,
            ));
        }
        PartiallyObservableProcess {
            mdp,
            num_observations: spec.num_observations,
            observation: spec.observation,
        }
    }

    /// Partition-refinement labels (TS default tol = 1e-9): states sharing a
    /// label are not (yet) distinguishable.
    pub fn distinguishability_classes(&self) -> Vec<usize> {
        let tol = 1e-9;
        let n = self.mdp.num_states;
        let init_sigs: Vec<String> = (0..n).map(|s| Self::quantise(&self.observation[s], tol)).collect();
        let mut labels = Self::label_by_signature(&init_sigs);
        for _iter in 0..(n + 1) {
            let mut signatures: Vec<String> = Vec::with_capacity(n);
            for s in 0..n {
                // Signature = own label + per-action distribution over current labels.
                let mut parts: Vec<String> = vec![labels[s].to_string()];
                for a in 0..self.mdp.num_actions {
                    let mut block_mass = vec![0.0; n];
                    for t in 0..n {
                        block_mass[labels[t]] += self.mdp.transition[a][s][t];
                    }
                    parts.push(Self::quantise(&block_mass, tol));
                }
                signatures.push(parts.join("|"));
            }
            let next = Self::label_by_signature(&signatures);
            if Self::same_partition(&labels, &next) {
                break;
            }
            labels = next;
        }
        labels
    }

    /// Structurally observable iff refinement yields all-singleton classes.
    pub fn is_structurally_observable(&self) -> bool {
        let labels = self.distinguishability_classes();
        labels.iter().cloned().collect::<HashSet<usize>>().len() == self.mdp.num_states
    }

    /// Pairs (s, t), s < t, that remain indistinguishable (perceptual aliasing).
    pub fn indistinguishable_pairs(&self) -> Vec<(usize, usize)> {
        let labels = self.distinguishability_classes();
        let mut out: Vec<(usize, usize)> = Vec::new();
        for s in 0..labels.len() {
            for t in (s + 1)..labels.len() {
                if labels[s] == labels[t] {
                    out.push((s, t));
                }
            }
        }
        out
    }

    /// Number of distinguishability classes (full-rank analog: equals S).
    pub fn class_count(&self) -> usize {
        self.distinguishability_classes().iter().cloned().collect::<HashSet<usize>>().len()
    }

    fn quantise(v: &[f64], tol: f64) -> String {
        let digits = (0.0_f64).max((-(tol.log10())).round()) as usize;
        v.iter()
            .map(|x| format!("{:.*}", digits, x))
            .collect::<Vec<String>>()
            .join(",")
    }

    fn label_by_signature(signatures: &[String]) -> Vec<usize> {
        let mut map: HashMap<String, usize> = HashMap::new();
        let mut labels = vec![0usize; signatures.len()];
        for i in 0..signatures.len() {
            let id = match map.get(&signatures[i]) {
                Some(&id) => id,
                None => {
                    let id = map.len();
                    map.insert(signatures[i].clone(), id);
                    id
                }
            };
            labels[i] = id;
        }
        labels
    }

    fn same_partition(a: &[usize], b: &[usize]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        // Compare induced partitions (label values themselves may renumber).
        let mut map_ab: HashMap<usize, usize> = HashMap::new();
        let mut map_ba: HashMap<usize, usize> = HashMap::new();
        for i in 0..a.len() {
            match map_ab.get(&a[i]) {
                Some(&ea) => {
                    if ea != b[i] {
                        return false;
                    }
                }
                None => {
                    map_ab.insert(a[i], b[i]);
                }
            }
            match map_ba.get(&b[i]) {
                Some(&eb) => {
                    if eb != a[i] {
                        return false;
                    }
                }
                None => {
                    map_ba.insert(b[i], a[i]);
                }
            }
        }
        true
    }
}

// =============================================================================
// DES PIPELINE
// =============================================================================

/// Channel names for the obs/ctrl pipeline (TS static consts).
pub struct ObsCtrlChannels;

impl ObsCtrlChannels {
    pub const MODEL_LTI: &'static str = "model-lti";
    pub const MODEL_MDP: &'static str = "model-mdp";
    pub const MODEL_POMDP: &'static str = "model-pomdp";
    pub const RESULT: &'static str = "evaluation";
}

#[derive(Clone, Debug)]
pub struct StateSpaceToken {
    pub label: String,
    pub model: StateSpaceModel,
}

impl StateSpaceToken {
    pub fn new(label: String, model: StateSpaceModel) -> Self {
        StateSpaceToken { label, model }
    }
}

#[derive(Clone, Debug)]
pub struct MdpToken {
    pub label: String,
    pub mdp: MarkovDecisionProcess,
}

impl MdpToken {
    pub fn new(label: String, mdp: MarkovDecisionProcess) -> Self {
        MdpToken { label, mdp }
    }
}

#[derive(Clone, Debug)]
pub struct PomdpToken {
    pub label: String,
    pub pomdp: PartiallyObservableProcess,
}

impl PomdpToken {
    pub fn new(label: String, pomdp: PartiallyObservableProcess) -> Self {
        PomdpToken { label, pomdp }
    }
}

/// `type EvaluationKind = 'controllability' | 'observability' | ...`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EvaluationKind {
    Controllability,
    Observability,
    MdpControllability,
    PomdpObservability,
}

/// A single structural verdict produced by an evaluator station.
#[derive(Clone, Debug)]
pub struct EvaluationToken {
    pub label: String,
    pub kind: EvaluationKind,
    /// measured rank / class-count / reachable-degree
    pub measure: f64,
    /// target value for a "full"/positive verdict
    pub target: f64,
    pub full: bool,
    pub detail: String,
}

impl EvaluationToken {
    pub fn new(
        label: String,
        kind: EvaluationKind,
        measure: f64,
        target: f64,
        full: bool,
        detail: String,
    ) -> Self {
        EvaluationToken { label, kind, measure, target, full, detail }
    }
}

/// Emits a fixed list of linear state-space models once.
pub struct StateSpaceSourceStation {
    core: StationCore,
    models: Vec<StateSpaceToken>,
    emitted: bool,
}

impl StateSpaceSourceStation {
    pub fn new(id: &str, models: Vec<StateSpaceToken>) -> Self {
        StateSpaceSourceStation { core: StationCore::new(id), models, emitted: false }
    }
}

impl DESStation for StateSpaceSourceStation {
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
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let models = self.models.clone();
        for m in models {
            self.core.emit(Rc::new(m), ObsCtrlChannels::MODEL_LTI);
        }
        self.emitted = true;
    }
}

/// Emits a fixed list of MDPs once.
pub struct MdpSourceStation {
    core: StationCore,
    models: Vec<MdpToken>,
    emitted: bool,
}

impl MdpSourceStation {
    pub fn new(id: &str, models: Vec<MdpToken>) -> Self {
        MdpSourceStation { core: StationCore::new(id), models, emitted: false }
    }
}

impl DESStation for MdpSourceStation {
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
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let models = self.models.clone();
        for m in models {
            self.core.emit(Rc::new(m), ObsCtrlChannels::MODEL_MDP);
        }
        self.emitted = true;
    }
}

/// Emits a fixed list of POMDPs once.
pub struct PomdpSourceStation {
    core: StationCore,
    models: Vec<PomdpToken>,
    emitted: bool,
}

impl PomdpSourceStation {
    pub fn new(id: &str, models: Vec<PomdpToken>) -> Self {
        PomdpSourceStation { core: StationCore::new(id), models, emitted: false }
    }
}

impl DESStation for PomdpSourceStation {
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
        !self.emitted
    }
    fn run_time_step(&mut self) {
        if self.emitted {
            return;
        }
        let models = self.models.clone();
        for m in models {
            self.core.emit(Rc::new(m), ObsCtrlChannels::MODEL_POMDP);
        }
        self.emitted = true;
    }
}

/// Kalman controllability test as a zero-backlog transform.
pub struct ControllabilityEvaluatorStation {
    tcore: TransformEntityCore<StateSpaceToken, EvaluationToken>,
}

impl ControllabilityEvaluatorStation {
    pub fn new(id: &str) -> Self {
        ControllabilityEvaluatorStation {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![ObsCtrlChannels::MODEL_LTI.to_string()],
                    output_channel: OutputChannel::Fixed(ObsCtrlChannels::RESULT.to_string()),
                    ..Default::default()
                },
            ),
        }
    }
}

impl TransformEntity<StateSpaceToken, EvaluationToken> for ControllabilityEvaluatorStation {
    fn tcore(&self) -> &TransformEntityCore<StateSpaceToken, EvaluationToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<StateSpaceToken, EvaluationToken> {
        &mut self.tcore
    }
}

impl PureTransformEntity<StateSpaceToken, EvaluationToken> for ControllabilityEvaluatorStation {
    fn transform(
        &mut self,
        token: &StateSpaceToken,
        _ctx: &mut TransformContext<EvaluationToken>,
    ) -> TransformResult<EvaluationToken> {
        let n = token.model.state_dim();
        let rank = token.model.controllability_rank();
        TransformResult::One(EvaluationToken::new(
            token.label.clone(),
            EvaluationKind::Controllability,
            rank as f64,
            n as f64,
            rank == n,
            format!("rank C = {rank} / n = {n}"),
        ))
    }
}

impl DESStation for ControllabilityEvaluatorStation {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

/// Kalman observability test as a zero-backlog transform.
pub struct ObservabilityEvaluatorStation {
    tcore: TransformEntityCore<StateSpaceToken, EvaluationToken>,
}

impl ObservabilityEvaluatorStation {
    pub fn new(id: &str) -> Self {
        ObservabilityEvaluatorStation {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![ObsCtrlChannels::MODEL_LTI.to_string()],
                    output_channel: OutputChannel::Fixed(ObsCtrlChannels::RESULT.to_string()),
                    ..Default::default()
                },
            ),
        }
    }
}

impl TransformEntity<StateSpaceToken, EvaluationToken> for ObservabilityEvaluatorStation {
    fn tcore(&self) -> &TransformEntityCore<StateSpaceToken, EvaluationToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<StateSpaceToken, EvaluationToken> {
        &mut self.tcore
    }
}

impl PureTransformEntity<StateSpaceToken, EvaluationToken> for ObservabilityEvaluatorStation {
    fn transform(
        &mut self,
        token: &StateSpaceToken,
        _ctx: &mut TransformContext<EvaluationToken>,
    ) -> TransformResult<EvaluationToken> {
        let n = token.model.state_dim();
        let rank = token.model.observability_rank();
        TransformResult::One(EvaluationToken::new(
            token.label.clone(),
            EvaluationKind::Observability,
            rank as f64,
            n as f64,
            rank == n,
            format!("rank O = {rank} / n = {n}"),
        ))
    }
}

impl DESStation for ObservabilityEvaluatorStation {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

/// MDP reachability ("controllability") test as a zero-backlog transform.
pub struct MdpControllabilityEvaluatorStation {
    tcore: TransformEntityCore<MdpToken, EvaluationToken>,
}

impl MdpControllabilityEvaluatorStation {
    pub fn new(id: &str) -> Self {
        MdpControllabilityEvaluatorStation {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![ObsCtrlChannels::MODEL_MDP.to_string()],
                    output_channel: OutputChannel::Fixed(ObsCtrlChannels::RESULT.to_string()),
                    ..Default::default()
                },
            ),
        }
    }
}

impl TransformEntity<MdpToken, EvaluationToken> for MdpControllabilityEvaluatorStation {
    fn tcore(&self) -> &TransformEntityCore<MdpToken, EvaluationToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<MdpToken, EvaluationToken> {
        &mut self.tcore
    }
}

impl PureTransformEntity<MdpToken, EvaluationToken> for MdpControllabilityEvaluatorStation {
    fn transform(
        &mut self,
        token: &MdpToken,
        _ctx: &mut TransformContext<EvaluationToken>,
    ) -> TransformResult<EvaluationToken> {
        let s = token.mdp.num_states;
        let target = s * s;
        let reachable = token.mdp.reachable_pair_count();
        TransformResult::One(EvaluationToken::new(
            token.label.clone(),
            EvaluationKind::MdpControllability,
            reachable as f64,
            target as f64,
            token.mdp.is_structurally_controllable(),
            format!("reachable ordered pairs = {reachable} / S^2 = {target}"),
        ))
    }
}

impl DESStation for MdpControllabilityEvaluatorStation {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

/// POMDP distinguishability ("observability") test as a zero-backlog transform.
pub struct PomdpObservabilityEvaluatorStation {
    tcore: TransformEntityCore<PomdpToken, EvaluationToken>,
}

impl PomdpObservabilityEvaluatorStation {
    pub fn new(id: &str) -> Self {
        PomdpObservabilityEvaluatorStation {
            tcore: TransformEntityCore::new(
                id,
                TransformEntityOptions {
                    input_channels: vec![ObsCtrlChannels::MODEL_POMDP.to_string()],
                    output_channel: OutputChannel::Fixed(ObsCtrlChannels::RESULT.to_string()),
                    ..Default::default()
                },
            ),
        }
    }
}

impl TransformEntity<PomdpToken, EvaluationToken> for PomdpObservabilityEvaluatorStation {
    fn tcore(&self) -> &TransformEntityCore<PomdpToken, EvaluationToken> {
        &self.tcore
    }
    fn tcore_mut(&mut self) -> &mut TransformEntityCore<PomdpToken, EvaluationToken> {
        &mut self.tcore
    }
}

impl PureTransformEntity<PomdpToken, EvaluationToken> for PomdpObservabilityEvaluatorStation {
    fn transform(
        &mut self,
        token: &PomdpToken,
        _ctx: &mut TransformContext<EvaluationToken>,
    ) -> TransformResult<EvaluationToken> {
        let s = token.pomdp.mdp.num_states;
        let classes = token.pomdp.class_count();
        TransformResult::One(EvaluationToken::new(
            token.label.clone(),
            EvaluationKind::PomdpObservability,
            classes as f64,
            s as f64,
            token.pomdp.is_structurally_observable(),
            format!("distinguishability classes = {classes} / S = {s}"),
        ))
    }
}

impl DESStation for PomdpObservabilityEvaluatorStation {
    fn core(&self) -> &StationCore {
        &self.tcore.station
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.tcore.station
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn run_time_step(&mut self) {}
    fn has_work(&self) -> bool {
        false
    }
    fn assert_preconditions(&mut self) {
        self.assert_transform_preconditions();
    }
}

/// Collects evaluation verdicts.
pub struct EvaluationSinkStation {
    core: StationCore,
    pub results: Vec<Rc<EvaluationToken>>,
}

impl EvaluationSinkStation {
    pub fn new(id: &str) -> Self {
        EvaluationSinkStation { core: StationCore::new(id), results: Vec::new() }
    }

    /// Verdicts for one label, in arrival order.
    pub fn for_label(&self, label: &str) -> Vec<Rc<EvaluationToken>> {
        self.results.iter().filter(|r| r.label == label).cloned().collect()
    }
}

impl DESStation for EvaluationSinkStation {
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
        self.core.inbox_size(ObsCtrlChannels::RESULT) > 0
    }
    fn run_time_step(&mut self) {
        let drained = self.core.drain::<EvaluationToken>(ObsCtrlChannels::RESULT);
        self.results.extend(drained);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    use crate::des::general::des_base::station::StationRef;

    /// Double integrator: A=[[0,1],[0,0]], B=[[0],[1]], C=[[1,0]] is both
    /// controllable and observable (rank 2).
    fn double_integrator() -> StateSpaceModel {
        StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
            b: vec![vec![0.0], vec![1.0]],
            c: vec![vec![1.0, 0.0]],
            d: None,
        })
    }

    #[test]
    fn lti_controllability_and_observability() {
        let m = double_integrator();
        assert_eq!(m.state_dim(), 2);
        assert_eq!(m.controllability_rank(), 2);
        assert_eq!(m.observability_rank(), 2);
        assert!(m.is_controllable());
        assert!(m.is_observable());
    }

    #[test]
    fn uncontrollable_when_input_misses_a_mode() {
        // Two decoupled modes but B only drives the first -> rank 1.
        let m = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![1.0, 0.0], vec![0.0, 2.0]],
            b: vec![vec![1.0], vec![0.0]],
            c: vec![vec![1.0, 1.0]],
            d: None,
        });
        assert_eq!(m.controllability_rank(), 1);
        assert!(!m.is_controllable());
    }

    #[test]
    fn mdp_reachability() {
        // 2 states, 1 action, fully mixing -> strongly connected.
        let mdp = MarkovDecisionProcess::new(MdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
        });
        assert!(mdp.is_structurally_controllable());
        assert_eq!(mdp.reachable_pair_count(), 4);

        // Absorbing state 1 cannot reach state 0 -> not controllable.
        let mdp2 = MarkovDecisionProcess::new(MdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.0, 1.0]]],
        });
        assert!(!mdp2.is_structurally_controllable());
    }

    #[test]
    fn pomdp_distinguishability() {
        // Distinct, deterministic observations per state -> fully observable.
        let pomdp = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
        });
        assert!(pomdp.is_structurally_observable());
        assert_eq!(pomdp.class_count(), 2);

        // Identical observations -> states aliased, not observable.
        let pomdp2 = PartiallyObservableProcess::new(PomdpSpec {
            num_states: 2,
            num_actions: 1,
            transition: vec![vec![vec![0.5, 0.5], vec![0.5, 0.5]]],
            num_observations: 2,
            observation: vec![vec![0.5, 0.5], vec![0.5, 0.5]],
        });
        assert!(!pomdp2.is_structurally_observable());
        assert_eq!(pomdp2.indistinguishable_pairs(), vec![(0, 1)]);
    }

    struct EvalCollect {
        core: StationCore,
        got: Vec<Rc<EvaluationToken>>,
    }
    impl DESStation for EvalCollect {
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
            let drained = self.core.drain::<EvaluationToken>(ObsCtrlChannels::RESULT);
            self.got.extend(drained);
        }
    }

    #[test]
    fn evaluator_station_emits_verdict() {
        let sink = Rc::new(RefCell::new(EvalCollect {
            core: StationCore::new("sink"),
            got: Vec::new(),
        }));
        let mut ev = ControllabilityEvaluatorStation::new("ctrl-eval");
        ev.tcore_mut().station.pipe(
            sink.clone() as StationRef,
            ObsCtrlChannels::RESULT,
            ObsCtrlChannels::RESULT,
        );
        ev.take(
            Rc::new(StateSpaceToken::new("dbl".to_string(), double_integrator())),
            ObsCtrlChannels::MODEL_LTI,
        );
        sink.borrow_mut().run_time_step();
        let got = &sink.borrow().got;
        assert_eq!(got.len(), 1);
        assert!(got[0].full);
        assert_eq!(got[0].kind, EvaluationKind::Controllability);
    }
}
