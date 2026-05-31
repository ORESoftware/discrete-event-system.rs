//! Dual / shadow simulation evaluation of controllability & observability.
//!
//! `observability_controllability` answers the binary STRUCTURAL question from
//! known `A`/`B`/`C` matrices (Kalman rank). `empirical_control` measures the
//! quantitative DEGREE — but it too starts from the matrices, rolling out a
//! `DiscreteLinearSystem` surrogate.
//!
//! This module closes the remaining gap: given an ACTUAL running simulation —
//! treated as a black box you can reset, drive, and observe — it spins up
//! *shadow copies* of that simulation, perturbs them, and recovers the
//! controllability/observability Gramians purely from the responses. No
//! matrices required, so the very same probe quantifies a nonlinear plant.
//!
//! Two complementary, well-known constructions (Lall–Marsden–Glavaški 2002):
//!
//!   * EMPIRICAL CONTROLLABILITY GRAMIAN — fire a one-step input impulse on each
//!     channel from the operating point and accumulate the squared state
//!     deviation it produces. For a linear plant this reproduces
//!     `W_c = Σ_k Ad^k Bd Bdᵀ (Adᵀ)^k` exactly.
//!   * EMPIRICAL OBSERVABILITY GRAMIAN — nudge each initial-state direction
//!     (central difference) with the nominal input held and accumulate the
//!     output sensitivity. For a linear plant this reproduces
//!     `W_o = Σ_k (Adᵀ)^k Cᵀ C Ad^k` exactly.
//!
//! The eigen-structure of those Gramians (reusing [`empirical_control`]'s
//! [`GramianDegree`]) gives the quantitative verdict: smallest eigenvalue =
//! hardest-to-drive / hardest-to-see direction; condition number = anisotropy;
//! numeric rank = the structural verdict, but now robust and graded.
//!
//! Because the probe is a black box it can be wrapped around the real
//! [`DcMotorShadowPlant`] (the back-EMF DC motor integrated with RK4), and the
//! result cross-checked against the analytic Gramian of its `state_space()`.
//!
//! The optional NESTED layer abstracts the same shadow plant into a coarse
//! finite MDP (controllability ⇒ regime reachability) and POMDP
//! (observability ⇒ sensor distinguishability) so the structural questions can
//! be re-asked through the decision-process lens.
#![allow(dead_code)]

use serde::Serialize;

use super::dc_motor::{DcMotorDynamics, DcMotorParams};
use super::empirical_control::{
    ControllabilityGramian, DiscreteLinearSystem, GramianDegree, MdpControllabilityDegree,
    MonteCarloDistinguishability, ObservabilityGramian, RandomPolicyOpts,
};
use super::linear_algebra::{LinAlg, Matrix, Vector};
use super::numerical_solvers::{FixedStepIntegrator, RungeKutta4Integrator};
use super::observability_controllability::{
    MarkovDecisionProcess, MdpSpec, PartiallyObservableProcess, PomdpSpec, StateSpaceModel,
    StateSpaceSpec,
};

// =============================================================================
// THE SHADOW PLANT — any simulation reduced to a black-box probe.
// =============================================================================

/// A recorded shadow trajectory: state and output at every grid point, index 0
/// being the initial condition.
#[derive(Clone, Debug)]
pub struct ShadowTrajectory {
    /// `states[k]` for `k = 0..=steps` (index 0 = `x0`).
    pub states: Vec<Vector>,
    /// `outputs[k] = h(states[k])`.
    pub outputs: Vec<Vector>,
}

/// A simulation expressed as a black box the shadow evaluator can drive. A type
/// implements four primitives — dimensions, operating point, a one-step advance,
/// and an output map — and gets the multi-step [`ShadowPlant::rollout`] for free.
///
/// The contract is deliberately matrix-free: the evaluator never inspects the
/// dynamics, it only RUNS them, so the same code path quantifies linear and
/// nonlinear plants alike.
pub trait ShadowPlant {
    fn state_dim(&self) -> usize;
    fn input_dim(&self) -> usize;
    fn output_dim(&self) -> usize;

    /// Operating point `x*` the perturbations are taken around.
    fn nominal_state(&self) -> Vector;
    /// Nominal input `u*` held during the probes.
    fn nominal_input(&self) -> Vector;

    /// Advance the state one discrete step of size `dt` under input `u`.
    fn step(&self, x: &[f64], u: &[f64], dt: f64) -> Vector;
    /// Output (sensor) map `y = h(x)`.
    fn output(&self, x: &[f64]) -> Vector;

    /// Roll forward under `inputs` (one entry per step). Returns the trajectory
    /// of length `inputs.len() + 1`, including the initial point.
    fn rollout(&self, x0: &[f64], inputs: &[Vector], dt: f64) -> ShadowTrajectory {
        let mut states = Vec::with_capacity(inputs.len() + 1);
        let mut outputs = Vec::with_capacity(inputs.len() + 1);
        let mut x = x0.to_vec();
        outputs.push(self.output(&x));
        states.push(x.clone());
        for u in inputs {
            x = self.step(&x, u, dt);
            outputs.push(self.output(&x));
            states.push(x.clone());
        }
        ShadowTrajectory { states, outputs }
    }
}

// =============================================================================
// EMPIRICAL GRAMIANS — recovered from shadow simulations alone.
// =============================================================================

/// Empirical controllability Gramian. Fires a one-step input impulse `eps·e_j`
/// on each channel `j` (everything else held at the nominal input), measures the
/// resulting state deviation from the nominal trajectory, and accumulates
/// `W_c = (1/eps²) Σ_j Σ_{k=1}^{H} δx_k δx_kᵀ`.
///
/// For a linear plant discretised the same way as the analytic side this equals
/// `Σ_{k=0}^{H-1} Ad^k Bd Bdᵀ (Adᵀ)^k` to floating-point precision.
pub fn empirical_controllability_gramian(
    plant: &dyn ShadowPlant,
    horizon: usize,
    dt: f64,
    eps: f64,
) -> Matrix {
    assert!(horizon >= 1, "shadow controllability: horizon must be >= 1");
    assert!(eps > 0.0, "shadow controllability: eps must be > 0");
    let n = plant.state_dim();
    let m = plant.input_dim();
    let x0 = plant.nominal_state();
    let u_nom = plant.nominal_input();

    let nominal_inputs: Vec<Vector> = vec![u_nom.clone(); horizon];
    let nominal = plant.rollout(&x0, &nominal_inputs, dt);

    let mut w = LinAlg::zeros(n, n);
    let inv = 1.0 / (eps * eps);
    for j in 0..m {
        let mut inputs = nominal_inputs.clone();
        inputs[0][j] += eps; // impulse on channel j, first step only
        let traj = plant.rollout(&x0, &inputs, dt);
        for k in 1..=horizon {
            let mut dx = vec![0.0; n];
            for i in 0..n {
                dx[i] = traj.states[k][i] - nominal.states[k][i];
            }
            for a in 0..n {
                if dx[a] == 0.0 {
                    continue;
                }
                for b in 0..n {
                    w[a][b] += dx[a] * dx[b] * inv;
                }
            }
        }
    }
    w
}

/// Empirical observability Gramian. Perturbs each initial-state direction by
/// `±eps·e_i` with the nominal input held, forms the central-difference output
/// sensitivity `s^i_k = (y_k^+ − y_k^-) / (2 eps)`, and accumulates
/// `W_o[i][j] = Σ_{k=0}^{H-1} s^i_k · s^j_k`.
///
/// For a linear plant this equals `Σ_{k=0}^{H-1} (Adᵀ)^k Cᵀ C Ad^k`.
pub fn empirical_observability_gramian(
    plant: &dyn ShadowPlant,
    horizon: usize,
    dt: f64,
    eps: f64,
) -> Matrix {
    assert!(horizon >= 1, "shadow observability: horizon must be >= 1");
    assert!(eps > 0.0, "shadow observability: eps must be > 0");
    let n = plant.state_dim();
    let p = plant.output_dim();
    let x0 = plant.nominal_state();
    let u_nom = plant.nominal_input();
    let inputs: Vec<Vector> = vec![u_nom.clone(); horizon];

    // sens[i][k] = output sensitivity (R^p) to a unit nudge in state-direction i.
    let mut sens: Vec<Vec<Vector>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut xp = x0.clone();
        let mut xm = x0.clone();
        xp[i] += eps;
        xm[i] -= eps;
        let tp = plant.rollout(&xp, &inputs, dt);
        let tm = plant.rollout(&xm, &inputs, dt);
        let mut s_i: Vec<Vector> = Vec::with_capacity(horizon);
        for k in 0..horizon {
            let mut s = vec![0.0; p];
            for q in 0..p {
                s[q] = (tp.outputs[k][q] - tm.outputs[k][q]) / (2.0 * eps);
            }
            s_i.push(s);
        }
        sens.push(s_i);
    }

    let mut w = LinAlg::zeros(n, n);
    for a in 0..n {
        for b in a..n {
            let mut acc = 0.0;
            for k in 0..horizon {
                for q in 0..p {
                    acc += sens[a][k][q] * sens[b][k][q];
                }
            }
            w[a][b] = acc;
            w[b][a] = acc; // symmetric
        }
    }
    w
}

// =============================================================================
// QUANTIFIED REPORT.
// =============================================================================

/// Eigen-summary of a symmetric PSD Gramian, serialisable for downstream JSON /
/// HTML rendering. Wraps [`GramianDegree`] and adds a numeric-rank verdict.
#[derive(Clone, Debug, Serialize)]
pub struct GramianSummary {
    pub dim: usize,
    /// Eigenvalues, ascending (λ_min … λ_max).
    pub eigenvalues: Vector,
    pub min: f64,
    pub max: f64,
    pub trace: f64,
    /// Σ ln λ — log-volume of the reachable / observable ellipsoid.
    pub log_volume: f64,
    /// λ_max / λ_min (∞ when a direction collapses).
    pub condition_number: f64,
    /// Eigenvalues above a relative threshold — the robust structural rank.
    pub numeric_rank: usize,
    /// Unit eigenvector of λ_min: the hardest-to-drive / hardest-to-see axis.
    pub weakest_direction: Vector,
    /// Unit eigenvector of λ_max: the easiest axis.
    pub strongest_direction: Vector,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gramian: Option<Matrix>,
}

impl GramianSummary {
    pub fn from_gramian(w: &Matrix, keep_matrix: bool) -> Self {
        let degree = GramianDegree::new(w.clone());
        let eigenvalues = degree.eigenvalues();
        let dim = eigenvalues.len();
        let max = degree.max();
        let min = degree.min();
        let rank_tol = (max * 1e-9).max(1e-12);
        let numeric_rank = eigenvalues.iter().filter(|&&l| l > rank_tol).count();
        let trace: f64 = eigenvalues.iter().sum();
        let log_volume: f64 = eigenvalues.iter().map(|&l| l.max(1e-300).ln()).sum();
        GramianSummary {
            dim,
            eigenvalues,
            min,
            max,
            trace,
            log_volume,
            condition_number: degree.condition_number(),
            numeric_rank,
            weakest_direction: degree.weakest_direction(),
            strongest_direction: degree.strongest_direction(),
            gramian: if keep_matrix { Some(w.clone()) } else { None },
        }
    }

    /// Full structural rank ⇒ controllable / observable.
    pub fn full_rank(&self) -> bool {
        self.numeric_rank == self.dim
    }
}

/// Options for [`evaluate_shadow`].
#[derive(Clone, Debug)]
pub struct ShadowEvalOpts {
    /// Number of discrete steps the probes run.
    pub horizon: usize,
    /// Step size `dt` handed to the plant.
    pub dt: f64,
    /// Perturbation magnitude (small ⇒ linearised around the operating point).
    pub epsilon: f64,
    /// Keep the dense Gramian matrices in the report (for rendering).
    pub keep_matrices: bool,
}

impl Default for ShadowEvalOpts {
    fn default() -> Self {
        ShadowEvalOpts {
            horizon: 80,
            dt: 0.02,
            epsilon: 1e-3,
            keep_matrices: true,
        }
    }
}

/// The full dual/shadow assessment of one simulation.
#[derive(Clone, Debug, Serialize)]
pub struct ShadowReport {
    pub label: String,
    pub state_dim: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    pub horizon: usize,
    pub dt: f64,
    pub epsilon: f64,
    /// Controllability Gramian recovered purely from shadow simulations.
    pub empirical_controllability: GramianSummary,
    /// Observability Gramian recovered purely from shadow simulations.
    pub empirical_observability: GramianSummary,
    /// Analytic Gramian (from known matrices), when available, for validation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analytic_controllability: Option<GramianSummary>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub analytic_observability: Option<GramianSummary>,
    pub controllable: bool,
    pub observable: bool,
    /// Worst relative eigenvalue gap between the shadow and analytic Gramians —
    /// a small number validates that the black-box probe matched the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cross_check_rel_error: Option<f64>,
}

impl ShadowReport {
    /// Human-readable multi-line summary (for stdout / `<pre>` blocks).
    pub fn summary_lines(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!(
            "── {} (n={}, m={}, p={}; H={}, dt={}, ε={:.0e}) ──\n",
            self.label,
            self.state_dim,
            self.input_dim,
            self.output_dim,
            self.horizon,
            self.dt,
            self.epsilon
        ));
        s.push_str(&format!(
            "  CONTROLLABILITY  rank {}/{}  → {}\n",
            self.empirical_controllability.numeric_rank,
            self.state_dim,
            if self.controllable {
                "controllable"
            } else {
                "NOT controllable"
            }
        ));
        s.push_str(&format!(
            "    W_c λ ∈ [{:.3e}, {:.3e}]  cond {}\n",
            self.empirical_controllability.min,
            self.empirical_controllability.max,
            fmt_cond(self.empirical_controllability.condition_number)
        ));
        s.push_str(&format!(
            "  OBSERVABILITY    rank {}/{}  → {}\n",
            self.empirical_observability.numeric_rank,
            self.state_dim,
            if self.observable {
                "observable"
            } else {
                "NOT observable"
            }
        ));
        s.push_str(&format!(
            "    W_o λ ∈ [{:.3e}, {:.3e}]  cond {}\n",
            self.empirical_observability.min,
            self.empirical_observability.max,
            fmt_cond(self.empirical_observability.condition_number)
        ));
        if let Some(rel) = self.cross_check_rel_error {
            s.push_str(&format!(
                "  shadow-vs-analytic eigenvalue agreement: rel.err {:.2e}\n",
                rel
            ));
        }
        s
    }
}

fn fmt_cond(c: f64) -> String {
    if c.is_finite() {
        format!("{c:.1e}")
    } else {
        "∞".to_string()
    }
}

/// Worst index-wise relative eigenvalue gap (eigenvalues are ascending in both).
fn rel_eig_error(empirical: &[f64], analytic: &[f64]) -> f64 {
    let scale = analytic
        .iter()
        .cloned()
        .fold(0.0_f64, |m, v| m.max(v.abs()))
        .max(1e-300);
    let mut err = 0.0_f64;
    for i in 0..empirical.len().min(analytic.len()) {
        err = err.max((empirical[i] - analytic[i]).abs() / scale);
    }
    err
}

/// Run the dual/shadow evaluation on any [`ShadowPlant`]: probe the empirical
/// controllability and observability Gramians and quantify them.
pub fn evaluate_shadow(
    label: &str,
    plant: &dyn ShadowPlant,
    opts: &ShadowEvalOpts,
) -> ShadowReport {
    let wc = empirical_controllability_gramian(plant, opts.horizon, opts.dt, opts.epsilon);
    let wo = empirical_observability_gramian(plant, opts.horizon, opts.dt, opts.epsilon);
    let cs = GramianSummary::from_gramian(&wc, opts.keep_matrices);
    let os = GramianSummary::from_gramian(&wo, opts.keep_matrices);
    let controllable = cs.full_rank();
    let observable = os.full_rank();
    ShadowReport {
        label: label.to_string(),
        state_dim: plant.state_dim(),
        input_dim: plant.input_dim(),
        output_dim: plant.output_dim(),
        horizon: opts.horizon,
        dt: opts.dt,
        epsilon: opts.epsilon,
        empirical_controllability: cs,
        empirical_observability: os,
        analytic_controllability: None,
        analytic_observability: None,
        controllable,
        observable,
        cross_check_rel_error: None,
    }
}

/// Attach the analytic Gramians of a known LTI model and the eigenvalue
/// agreement against the shadow estimate (the "dual" cross-validation).
pub fn attach_lti_cross_check(
    report: &mut ShadowReport,
    model: &StateSpaceModel,
    keep_matrices: bool,
) {
    let sys = DiscreteLinearSystem::from_continuous(model, report.dt);
    let wc = ControllabilityGramian::new(&sys, report.horizon);
    let wo = ObservabilityGramian::new(&sys, report.horizon);
    let ac = GramianSummary::from_gramian(wc.matrix(), keep_matrices);
    let ao = GramianSummary::from_gramian(wo.matrix(), keep_matrices);
    let rel = rel_eig_error(
        &report.empirical_controllability.eigenvalues,
        &ac.eigenvalues,
    )
    .max(rel_eig_error(
        &report.empirical_observability.eigenvalues,
        &ao.eigenvalues,
    ));
    report.analytic_controllability = Some(ac);
    report.analytic_observability = Some(ao);
    report.cross_check_rel_error = Some(rel);
}

// =============================================================================
// CONCRETE SHADOW PLANTS.
// =============================================================================

/// A linear time-invariant plant integrated with forward Euler — deliberately
/// matched to [`DiscreteLinearSystem::from_continuous`] so the shadow Gramians
/// reproduce the analytic ones to floating-point precision (the validation
/// anchor for the empirical method).
#[derive(Clone, Debug)]
pub struct LtiPlant {
    a: Matrix,
    b: Matrix,
    c: Matrix,
    x_star: Vector,
    u_star: Vector,
}

impl LtiPlant {
    pub fn new(model: &StateSpaceModel) -> Self {
        let n = model.state_dim();
        let m = model.input_dim();
        LtiPlant {
            a: model.a.clone(),
            b: model.b.clone(),
            c: model.c.clone(),
            x_star: vec![0.0; n],
            u_star: vec![0.0; m],
        }
    }
}

impl ShadowPlant for LtiPlant {
    fn state_dim(&self) -> usize {
        self.a.len()
    }
    fn input_dim(&self) -> usize {
        LinAlg::cols(&self.b)
    }
    fn output_dim(&self) -> usize {
        self.c.len()
    }
    fn nominal_state(&self) -> Vector {
        self.x_star.clone()
    }
    fn nominal_input(&self) -> Vector {
        self.u_star.clone()
    }
    fn step(&self, x: &[f64], u: &[f64], dt: f64) -> Vector {
        let ax = LinAlg::mat_vec(&self.a, x);
        let bu = LinAlg::mat_vec(&self.b, u);
        (0..x.len()).map(|i| x[i] + dt * (ax[i] + bu[i])).collect()
    }
    fn output(&self, x: &[f64]) -> Vector {
        LinAlg::mat_vec(&self.c, x)
    }
}

/// The real back-EMF DC motor as a shadow plant: state `[i, ω]`, input armature
/// voltage `V`, output rotor speed `ω`, advanced with the SAME RK4 step the
/// `DcMotorPlantStation` uses. The shadow evaluator probes this exactly as it
/// runs in simulation — no linearisation assumed.
#[derive(Clone, Debug)]
pub struct DcMotorShadowPlant {
    params: DcMotorParams,
    load_torque: f64,
}

impl DcMotorShadowPlant {
    pub fn new(params: DcMotorParams) -> Self {
        DcMotorShadowPlant {
            params,
            load_torque: 0.0,
        }
    }

    pub fn with_load(params: DcMotorParams, load_torque: f64) -> Self {
        DcMotorShadowPlant {
            params,
            load_torque,
        }
    }

    /// The continuous-time `(A, B, C, D)` model the motor exposes — used to
    /// cross-check the shadow estimate against the analytic Gramian.
    pub fn state_space_model(&self) -> StateSpaceModel {
        let dynamics = DcMotorDynamics::new(self.params.clone());
        let ss = dynamics.state_space();
        StateSpaceModel::new(StateSpaceSpec {
            a: ss.a,
            b: ss.b,
            c: ss.c,
            d: Some(ss.d),
        })
    }
}

impl ShadowPlant for DcMotorShadowPlant {
    fn state_dim(&self) -> usize {
        2
    }
    fn input_dim(&self) -> usize {
        1
    }
    fn output_dim(&self) -> usize {
        1
    }
    fn nominal_state(&self) -> Vector {
        vec![0.0, 0.0]
    }
    fn nominal_input(&self) -> Vector {
        vec![0.0]
    }
    fn step(&self, x: &[f64], u: &[f64], dt: f64) -> Vector {
        let mut dynamics = DcMotorDynamics::new(self.params.clone());
        dynamics.set_inputs(u[0], self.load_torque);
        RungeKutta4Integrator::new().step(&dynamics, 0.0, x, dt)
    }
    fn output(&self, x: &[f64]) -> Vector {
        vec![x[1]] // rotor speed ω
    }
}

// =============================================================================
// NESTED MDP / POMDP ABSTRACTION — controllability/observability re-asked
// through the decision-process lens.
// =============================================================================

/// Quantise a scalar into a regime index from ascending interior `edges`
/// (`edges.len() + 1` regimes).
pub fn quantize(value: f64, edges: &[f64]) -> usize {
    let mut idx = 0usize;
    for &e in edges {
        if value >= e {
            idx += 1;
        } else {
            break;
        }
    }
    idx
}

/// Build a controlled regime-MDP by shadow-simulating one macro-step per
/// (regime, action). The observed coordinate `obs_index` of the output defines
/// the regime; `rep_states[s]` is the representative initial state for regime
/// `s`; each `actions[a]` is a constant input applied for `macro_steps` steps.
/// The landing regime gives a deterministic (one-hot) transition row.
pub fn build_regime_mdp(
    plant: &dyn ShadowPlant,
    obs_index: usize,
    edges: &[f64],
    rep_states: &[Vector],
    actions: &[Vector],
    macro_steps: usize,
    dt: f64,
) -> MdpSpec {
    let num_states = edges.len() + 1;
    assert_eq!(
        rep_states.len(),
        num_states,
        "build_regime_mdp: need one representative state per regime"
    );
    assert!(macro_steps >= 1, "build_regime_mdp: macro_steps >= 1");
    let num_actions = actions.len();
    let mut transition = vec![vec![vec![0.0; num_states]; num_states]; num_actions];
    for (a, action) in actions.iter().enumerate() {
        let inputs = vec![action.clone(); macro_steps];
        for s in 0..num_states {
            let traj = plant.rollout(&rep_states[s], &inputs, dt);
            let y_final = traj.outputs[macro_steps][obs_index];
            let s_next = quantize(y_final, edges).min(num_states - 1);
            transition[a][s][s_next] = 1.0;
        }
    }
    MdpSpec {
        num_states,
        num_actions,
        transition,
    }
}

/// Build a STOCHASTIC regime-MDP by shadow-simulating a *spread* of starting
/// states within each regime over a short macro-step. Where [`build_regime_mdp`]
/// collapses each (regime, action) to a single deterministic landing,
/// `regime_samples[s]` supplies several representative initial states spanning
/// regime `s`; the empirical landing distribution becomes the transition pmf.
///
/// A short `macro_steps` (so the regime is not fully reset) yields genuinely
/// history-dependent, banded transitions — the setting in which a noisy sensor
/// actually limits observability.
pub fn build_regime_mdp_sampled(
    plant: &dyn ShadowPlant,
    obs_index: usize,
    edges: &[f64],
    regime_samples: &[Vec<Vector>],
    actions: &[Vector],
    macro_steps: usize,
    dt: f64,
) -> MdpSpec {
    let num_states = edges.len() + 1;
    assert_eq!(
        regime_samples.len(),
        num_states,
        "build_regime_mdp_sampled: need a sample set per regime"
    );
    assert!(
        macro_steps >= 1,
        "build_regime_mdp_sampled: macro_steps >= 1"
    );
    let num_actions = actions.len();
    let mut transition = vec![vec![vec![0.0; num_states]; num_states]; num_actions];
    for (a, action) in actions.iter().enumerate() {
        let inputs = vec![action.clone(); macro_steps];
        for s in 0..num_states {
            let samples = &regime_samples[s];
            let mut counts = vec![0.0; num_states];
            for x0 in samples {
                let traj = plant.rollout(x0, &inputs, dt);
                let y_final = traj.outputs[macro_steps][obs_index];
                counts[quantize(y_final, edges).min(num_states - 1)] += 1.0;
            }
            let total: f64 = counts.iter().sum();
            if total > 0.0 {
                for t in 0..num_states {
                    transition[a][s][t] = counts[t] / total;
                }
            } else {
                transition[a][s][s] = 1.0;
            }
        }
    }
    MdpSpec {
        num_states,
        num_actions,
        transition,
    }
}

/// Synthesise a noisy sensor (POMDP) over the regimes of `mdp`: one observation
/// per regime, read correctly with probability `confidence`, the rest spread
/// over the immediate neighbour regimes (rows renormalised).
pub fn synth_sensor_pomdp(mdp: &MdpSpec, confidence: f64) -> PomdpSpec {
    let s = mdp.num_states;
    let mut observation = vec![vec![0.0; s]; s];
    for st in 0..s {
        let mut neighbours: Vec<usize> = Vec::new();
        if st > 0 {
            neighbours.push(st - 1);
        }
        if st + 1 < s {
            neighbours.push(st + 1);
        }
        observation[st][st] = confidence;
        let leak = if neighbours.is_empty() {
            0.0
        } else {
            (1.0 - confidence) / neighbours.len() as f64
        };
        for &nb in &neighbours {
            observation[st][nb] += leak;
        }
        // Renormalise (handles the single-regime / clamped edge cases).
        let z: f64 = observation[st].iter().sum();
        if z > 0.0 {
            for o in 0..s {
                observation[st][o] /= z;
            }
        }
    }
    PomdpSpec {
        num_states: s,
        num_actions: mdp.num_actions,
        transition: mdp.transition.clone(),
        num_observations: s,
        observation,
    }
}

/// The decision-process view of controllability (MDP reachability) and
/// observability (POMDP belief distinguishability).
#[derive(Clone, Debug, Serialize)]
pub struct NestedAssessment {
    pub label: String,
    pub num_regimes: usize,
    pub num_actions: usize,
    // ── controllability ⇒ reachability ──
    pub mdp_structurally_controllable: bool,
    pub reachable_pairs: usize,
    pub reachable_fraction: f64,
    pub per_target_reach_degree: Vector,
    // ── observability ⇒ distinguishability ──
    pub sensor_confidence: f64,
    pub pomdp_structurally_observable: bool,
    pub distinguishability_min: f64,
    pub distinguishability_max: f64,
    pub belief_hit_probability: Vector,
}

/// Assess a regime MDP through both decision-process lenses, synthesising a
/// noisy sensor of the given `sensor_confidence` for the observability side.
pub fn assess_nested(label: &str, mdp_spec: &MdpSpec, sensor_confidence: f64) -> NestedAssessment {
    let mdp = MarkovDecisionProcess::new(mdp_spec.clone());
    let reachable = mdp.reachable_pair_count();
    let total = mdp_spec.num_states * mdp_spec.num_states;
    let reach_opts = RandomPolicyOpts {
        episodes: Some(300),
        ..Default::default()
    };
    let per_target = MdpControllabilityDegree::new(&mdp).per_target_degree(&reach_opts);

    let pomdp = PartiallyObservableProcess::new(synth_sensor_pomdp(mdp_spec, sensor_confidence));
    let structurally_observable = pomdp.is_structurally_observable();
    let dist = MonteCarloDistinguishability::new(&pomdp).run(&RandomPolicyOpts {
        episodes: Some(300),
        ..Default::default()
    });

    NestedAssessment {
        label: label.to_string(),
        num_regimes: mdp_spec.num_states,
        num_actions: mdp_spec.num_actions,
        mdp_structurally_controllable: mdp.is_structurally_controllable(),
        reachable_pairs: reachable,
        reachable_fraction: reachable as f64 / total as f64,
        per_target_reach_degree: per_target,
        sensor_confidence,
        pomdp_structurally_observable: structurally_observable,
        distinguishability_min: dist.min_degree,
        distinguishability_max: dist.max_degree,
        belief_hit_probability: dist.hit_probability,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn double_integrator() -> StateSpaceModel {
        StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![0.0, 1.0], vec![0.0, 0.0]],
            b: vec![vec![0.0], vec![1.0]],
            c: vec![vec![1.0, 0.0]],
            d: None,
        })
    }

    fn decoupled_unreachable() -> StateSpaceModel {
        // Two stable modes, but B drives only mode 0 and C sees only mode 0.
        StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![-1.0, 0.0], vec![0.0, -2.0]],
            b: vec![vec![1.0], vec![0.0]],
            c: vec![vec![1.0, 0.0]],
            d: None,
        })
    }

    fn motor_params() -> DcMotorParams {
        DcMotorParams {
            resistance: 2.0,
            inductance: 0.5,
            back_emf_constant: 0.1,
            torque_constant: 0.1,
            inertia: 0.02,
            friction: 0.002,
        }
    }

    #[test]
    fn shadow_gramians_match_analytic_for_lti() {
        let model = double_integrator();
        let plant = LtiPlant::new(&model);
        let opts = ShadowEvalOpts {
            horizon: 40,
            dt: 0.02,
            epsilon: 1e-3,
            keep_matrices: false,
        };
        let mut report = evaluate_shadow("double integrator", &plant, &opts);
        attach_lti_cross_check(&mut report, &model, false);
        assert!(report.controllable, "double integrator is controllable");
        assert!(report.observable, "double integrator is observable");
        // Euler shadow vs Euler analytic: agreement to floating-point precision.
        let rel = report.cross_check_rel_error.unwrap();
        assert!(rel < 1e-6, "shadow/analytic eigenvalue rel.err = {rel}");
    }

    #[test]
    fn shadow_detects_uncontrollable_unobservable() {
        let model = decoupled_unreachable();
        let plant = LtiPlant::new(&model);
        let report = evaluate_shadow("decoupled", &plant, &ShadowEvalOpts::default());
        assert!(!report.controllable, "mode 1 is undrivable");
        assert!(!report.observable, "mode 1 is invisible");
        assert_eq!(report.empirical_controllability.numeric_rank, 1);
        assert_eq!(report.empirical_observability.numeric_rank, 1);
    }

    #[test]
    fn dc_motor_shadow_is_controllable_and_observable() {
        let plant = DcMotorShadowPlant::new(motor_params());
        let opts = ShadowEvalOpts {
            horizon: 60,
            dt: 0.01,
            epsilon: 1e-4,
            keep_matrices: false,
        };
        let mut report = evaluate_shadow("dc motor", &plant, &opts);
        attach_lti_cross_check(&mut report, &plant.state_space_model(), false);
        assert!(report.controllable, "back-EMF motor is controllable");
        assert!(report.observable, "speed sensor makes the motor observable");
        // RK4 shadow vs Euler analytic differ only by discretisation order.
        let rel = report.cross_check_rel_error.unwrap();
        assert!(
            rel.is_finite() && rel < 0.25,
            "motor cross-check rel.err = {rel}"
        );
    }

    #[test]
    fn quantize_partitions_by_edges() {
        let edges = [3.0, 9.0];
        assert_eq!(quantize(0.0, &edges), 0);
        assert_eq!(quantize(2.999, &edges), 0);
        assert_eq!(quantize(3.0, &edges), 1);
        assert_eq!(quantize(8.5, &edges), 1);
        assert_eq!(quantize(9.0, &edges), 2);
        assert_eq!(quantize(100.0, &edges), 2);
    }

    #[test]
    fn regime_mdp_from_integrator_is_reachable() {
        // 1-state integrator x' = u, output = x. Actions push to fixed levels.
        let model = StateSpaceModel::new(StateSpaceSpec {
            a: vec![vec![0.0]],
            b: vec![vec![1.0]],
            c: vec![vec![1.0]],
            d: None,
        });
        let plant = LtiPlant::new(&model);
        // 3 regimes split at x = 1 and x = 3, centred at x = 0, 2, 4.
        let edges = [1.0, 3.0];
        let rep_states = vec![vec![0.0], vec![2.0], vec![4.0]];
        // Each macro-step moves dt*macro_steps*u = 0.2*u; ±10 ⇒ ±2 = one regime
        // step (so brake/hold/drive can reach every neighbouring regime).
        let actions = vec![vec![-10.0], vec![0.0], vec![10.0]];
        let mdp_spec = build_regime_mdp(&plant, 0, &edges, &rep_states, &actions, 10, 0.02);
        let nested = assess_nested("integrator regimes", &mdp_spec, 0.85);
        assert!(
            nested.mdp_structurally_controllable,
            "drive/hold/brake should reach every regime"
        );
        assert_eq!(nested.reachable_fraction, 1.0);
    }

    #[test]
    fn nested_sensor_distinguishability_improves_with_confidence() {
        // Ring MDP: 0->1->2->0 (strongly connected, controllable).
        let mdp_spec = MdpSpec {
            num_states: 3,
            num_actions: 1,
            transition: vec![vec![
                vec![0.0, 1.0, 0.0],
                vec![0.0, 0.0, 1.0],
                vec![1.0, 0.0, 0.0],
            ]],
        };
        let sharp = assess_nested("ring sharp", &mdp_spec, 0.95);
        let blurry = assess_nested("ring blurry", &mdp_spec, 0.40);
        assert!(sharp.mdp_structurally_controllable);
        assert!(
            sharp.distinguishability_min >= blurry.distinguishability_min - 1e-9,
            "sharper sensor should not distinguish worse: {} vs {}",
            sharp.distinguishability_min,
            blurry.distinguishability_min
        );
    }
}
