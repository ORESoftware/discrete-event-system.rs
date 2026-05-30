//! Port of `src/des/main-court-mdp.ts`.
//!
//! USACC case-flow MDP simulated inside the framework: cases (moving entities)
//! flow through four stage stations, each applying an interchangeable `Policy`
//! to the case's `CaseState`, then performing the stochastic transition
//! (advance / close / accept / exhaust).
//!
//! ## Rust shape
//!   * The MDP itself (`CaseState`, `encode`/`decode`, `outcomes`,
//!     `sample_initial_state`, terminal constants) is reused from
//!     `crate::des::mdp::usacc_mdp`; value iteration from
//!     `crate::des::mdp::value_iteration`.
//!   * The TS `Policy` interface → [`Policy`] trait with one struct impl each
//!     (`AlwaysEscalatePolicy`, `RejectAllPolicy`, `NaiveThresholdPolicy`,
//!     `OptimalPolicy`).
//!   * The `CaseSource` / `StageStation` / `TerminalSink`
//!     `BufferedTimeSteppedStation`s are realised as index-keyed inboxes inside
//!     [`run_court_sim`] (cases route by stage number), faithfully reproducing
//!     the per-tick routing without the `Rc<RefCell<dyn …>>` graph. See
//!     `crate::des::general::time_stepped_station` for the base traits.
//!   * `fisherYatesShuffle` (cosmetic stage order, drives RNG draws) and
//!     `mulberry32`/`withSeed` are reused from `crate::des::general::{general,
//!     prng}`.

#![allow(dead_code)]

use crate::des::general::general::fisher_yates_shuffle;
use crate::des::general::prng::{mulberry32, with_seed};
use crate::des::mdp::usacc_mdp::{
    decode, encode, outcomes, sample_initial_state, CaseState, ACCEPTED, ACTIONS, CLOSED,
    EXHAUSTED, N_ACTIONS, STAGES,
};
use crate::des::mdp::value_iteration::{value_iteration, VIOptions, VIResult};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

fn action_index(name: &str) -> usize {
    ACTIONS
        .iter()
        .position(|&a| a == name)
        .expect("known action")
}

// -----------------------------------------------------------------------------
// Policy trait + implementations.
// -----------------------------------------------------------------------------

/// A decision rule mapping a case state to an action index.
pub trait Policy {
    fn name(&self) -> &str;
    fn pick(&self, s: &CaseState, state_id: usize) -> usize;
}

pub struct AlwaysEscalatePolicy;
impl Policy for AlwaysEscalatePolicy {
    fn name(&self) -> &str {
        "always-escalate"
    }
    fn pick(&self, _s: &CaseState, _id: usize) -> usize {
        action_index("escalate_to_next_stage")
    }
}

pub struct RejectAllPolicy;
impl Policy for RejectAllPolicy {
    fn name(&self) -> &str {
        "reject-all"
    }
    fn pick(&self, _s: &CaseState, _id: usize) -> usize {
        action_index("reject_or_close")
    }
}

/// Hand-tuned heuristic comparison policy.
pub struct NaiveThresholdPolicy;
impl Policy for NaiveThresholdPolicy {
    fn name(&self) -> &str {
        "naive-threshold"
    }
    fn pick(&self, s: &CaseState, _id: usize) -> usize {
        if s.funding == 0 {
            return action_index("release_escrow");
        }
        if s.manipulation >= 2 {
            return action_index("hold_for_audit");
        }
        if s.conflict == 1 {
            return action_index("assign_reviewers");
        }
        if s.evidence == 0 {
            return action_index("request_more_evidence");
        }
        if s.corroboration == 0 {
            return action_index("verify_identity");
        }
        let score = s.evidence as f64 + s.corroboration as f64
            - s.manipulation as f64
            - 1.5 * s.conflict as f64;
        if score - 1.5 * 0.0 >= 2.0 {
            // NOTE: matches TS `evidence + corroboration − manipulation − 1.5·conflict ≥ 2`.
            return action_index("escalate_to_next_stage");
        }
        action_index("reject_or_close")
    }
}

/// Looks up π* from value iteration.
pub struct OptimalPolicy {
    pub action: Vec<i32>,
}
impl Policy for OptimalPolicy {
    fn name(&self) -> &str {
        "optimal"
    }
    fn pick(&self, _s: &CaseState, state_id: usize) -> usize {
        self.action[state_id].max(0) as usize
    }
}

// -----------------------------------------------------------------------------
// Moving entity.
// -----------------------------------------------------------------------------

#[derive(Clone)]
struct Case {
    id: usize,
    state: CaseState,
    history: Vec<(usize, usize, f64)>, // (stateId, action, reward)
    total_reward: f64,
    arrival_time: i64,
    exit_time: i64,
    terminal: i64,
    steps: usize,
}

impl Case {
    fn new(id: usize, state: CaseState, t: i64) -> Self {
        Case {
            id,
            state,
            history: Vec::new(),
            total_reward: 0.0,
            arrival_time: t,
            exit_time: -1,
            terminal: -1,
            steps: 0,
        }
    }
}

// -----------------------------------------------------------------------------
// Public API.
// -----------------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
pub struct CourtMDPConfig {
    pub total_cases: usize,
    pub arrivals_per_tick: usize,
    pub max_ticks: usize,
    pub seed: u32,
}

#[derive(Clone, Debug)]
pub struct CourtAggregates {
    pub n: usize,
    pub n_accepted: usize,
    pub n_closed: usize,
    pub n_exhausted: usize,
    pub n_timed_out: usize,
    pub mean_reward: f64,
    pub mean_steps: f64,
    pub p95_steps: usize,
    pub fraction_accepted: f64,
    pub fraction_closed: f64,
    pub fraction_exhausted: f64,
}

#[derive(Clone, Debug)]
pub struct CourtMDPResult {
    pub policy: String,
    pub config: CourtMDPConfig,
    pub aggregates: CourtAggregates,
}

/// Sample one stochastic outcome index for `(state_id, action)`.
fn step_case(c: &mut Case, policy: &dyn Policy, stage_rng: &mut SeededRandom) -> i64 {
    let state_id = encode(&c.state);
    let action = policy.pick(&c.state, state_id);
    let ol = outcomes(state_id, action);
    let r = stage_rng.next_float();
    let mut cum = 0.0;
    let mut chosen = ol[0];
    for o in &ol {
        cum += o.prob;
        if r <= cum {
            chosen = *o;
            break;
        }
    }
    c.history.push((state_id, action, chosen.reward));
    c.total_reward += chosen.reward;
    c.steps += 1;
    chosen.next_state as i64
}

/// Run the case-flow simulation under a single policy.
pub fn run_court_sim(cfg: CourtMDPConfig, policy: &dyn Policy) -> CourtMDPResult {
    with_seed(cfg.seed, |shuffle_rng| {
        let mut stage_rng = mulberry32(cfg.seed ^ 0xCAFE);
        let mut stage_inboxes: Vec<Vec<Case>> = vec![Vec::new(); 4];
        let mut accepted: Vec<Case> = Vec::new();
        let mut closed: Vec<Case> = Vec::new();
        let mut exhausted: Vec<Case> = Vec::new();

        let mut source_idx = 0usize;
        let mut t: i64 = 0;
        while (t as usize) < cfg.max_ticks {
            // Source: emit arrivals into stage 0.
            if source_idx < cfg.total_cases {
                let mut rng = mulberry32(cfg.seed.wrapping_add(source_idx as u32));
                let mut k = 0;
                while k < cfg.arrivals_per_tick && source_idx < cfg.total_cases {
                    let init = sample_initial_state(&mut rng);
                    stage_inboxes[0].push(Case::new(source_idx, init, t));
                    k += 1;
                    source_idx += 1;
                }
            }

            // Process stages in shuffled order.
            let mut order: Vec<usize> = (0..4).collect();
            fisher_yates_shuffle(&mut order, shuffle_rng);
            for &si in &order {
                let todo = std::mem::take(&mut stage_inboxes[si]);
                for mut c in todo {
                    if c.state.stage as usize != si {
                        panic!("StageStation stage{si} got case at stage {}", c.state.stage);
                    }
                    let next_state = step_case(&mut c, policy, &mut stage_rng);
                    if next_state == ACCEPTED as i64 {
                        c.terminal = ACCEPTED as i64;
                        c.exit_time = t;
                        accepted.push(c);
                    } else if next_state == CLOSED as i64 {
                        c.terminal = CLOSED as i64;
                        c.exit_time = t;
                        closed.push(c);
                    } else if next_state == EXHAUSTED as i64 {
                        c.terminal = EXHAUSTED as i64;
                        c.exit_time = t;
                        exhausted.push(c);
                    } else {
                        let s_next = decode(next_state as usize).expect("decodable next state");
                        c.state = s_next;
                        let dest = s_next.stage as usize;
                        stage_inboxes[dest].push(c);
                    }
                }
            }
            t += 1;
            let collected = accepted.len() + closed.len() + exhausted.len();
            if collected == cfg.total_cases {
                break;
            }
        }

        let mut all_finished: Vec<&Case> = Vec::new();
        all_finished.extend(accepted.iter());
        all_finished.extend(closed.iter());
        all_finished.extend(exhausted.iter());
        let n_timed_out = cfg.total_cases - all_finished.len();

        let rewards: Vec<f64> = all_finished.iter().map(|c| c.total_reward).collect();
        let steps: Vec<usize> = all_finished.iter().map(|c| c.steps).collect();
        let mut sorted_steps = steps.clone();
        sorted_steps.sort_unstable();
        let p95 = if sorted_steps.is_empty() {
            0
        } else {
            sorted_steps[(0.95 * (sorted_steps.len() as f64 - 1.0)).floor() as usize]
        };
        let mean = |xs: &[f64]| xs.iter().sum::<f64>() / (xs.len().max(1) as f64);
        let mean_u = |xs: &[usize]| xs.iter().sum::<usize>() as f64 / (xs.len().max(1) as f64);

        CourtMDPResult {
            policy: policy.name().to_string(),
            config: cfg,
            aggregates: CourtAggregates {
                n: cfg.total_cases,
                n_accepted: accepted.len(),
                n_closed: closed.len(),
                n_exhausted: exhausted.len(),
                n_timed_out,
                mean_reward: mean(&rewards),
                mean_steps: mean_u(&steps),
                p95_steps: p95,
                fraction_accepted: accepted.len() as f64 / cfg.total_cases as f64,
                fraction_closed: closed.len() as f64 / cfg.total_cases as f64,
                fraction_exhausted: exhausted.len() as f64 / cfg.total_cases as f64,
            },
        }
    })
}

fn dump_vi() -> VIResult {
    let vi = value_iteration(VIOptions {
        gamma: 0.95,
        tol: 1e-10,
        max_iter: 5000,
    });
    println!(
        "# value iteration: {} sweeps, max|ΔV| = {:.3e}",
        vi.iterations, vi.final_delta
    );
    vi
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Entry point (TS top-level `main`).
pub fn run() {
    let n = env_usize("CASES", 5000);
    let seed = std::env::var("SEED")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(42u32);
    let arrivals_per_tick = env_usize("ARRIVALS_PER_TICK", 5);
    let max_ticks = env_usize("MAX_TICKS", 10000);

    println!("# USACC MDP simulation");
    println!("#   {n} cases, {arrivals_per_tick} arrivals/tick, maxTicks={max_ticks}, seed={seed}");

    let vi = dump_vi();
    let optimal = OptimalPolicy {
        action: vi.policy.clone(),
    };

    let policies: Vec<Box<dyn Policy>> = vec![
        Box::new(RejectAllPolicy),
        Box::new(AlwaysEscalatePolicy),
        Box::new(NaiveThresholdPolicy),
        Box::new(optimal),
    ];

    let cfg = CourtMDPConfig {
        total_cases: n,
        arrivals_per_tick,
        max_ticks,
        seed,
    };

    let mut results_json: Vec<serde_json::Value> = Vec::new();
    for p in &policies {
        let r = run_court_sim(cfg, p.as_ref());
        let a = &r.aggregates;
        println!();
        println!("# policy = {}", p.name());
        println!(
            "#   meanReward = {:.2}    meanSteps = {:.2}    p95Steps = {}",
            a.mean_reward, a.mean_steps, a.p95_steps
        );
        println!(
            "#   accepted = {:.1}%    closed = {:.1}%    exhausted = {:.1}%",
            a.fraction_accepted * 100.0,
            a.fraction_closed * 100.0,
            a.fraction_exhausted * 100.0
        );
        if a.n_timed_out > 0 {
            println!(
                "#   WARNING: {} cases timed out (raise MAX_TICKS)",
                a.n_timed_out
            );
        }
        results_json.push(serde_json::json!({
            "policy": r.policy,
            "aggregates": {
                "n": a.n,
                "nAccepted": a.n_accepted,
                "nClosed": a.n_closed,
                "nExhausted": a.n_exhausted,
                "nTimedOut": a.n_timed_out,
                "meanReward": a.mean_reward,
                "meanSteps": a.mean_steps,
                "p95Steps": a.p95_steps,
                "fractionAccepted": a.fraction_accepted,
                "fractionClosed": a.fraction_closed,
                "fractionExhausted": a.fraction_exhausted,
            },
        }));
    }

    // Dump V*, π*, and per-policy aggregates for the validator.
    let out = serde_json::json!({
        "config": {
            "totalCases": cfg.total_cases,
            "arrivalsPerTick": cfg.arrivals_per_tick,
            "maxTicks": cfg.max_ticks,
            "seed": cfg.seed,
        },
        "vi": {
            "gamma": vi.gamma,
            "iterations": vi.iterations,
            "finalDelta": vi.final_delta,
            "V": vi.v,
            "policy": vi.policy,
        },
        "results": results_json,
    });
    let _ = std::fs::create_dir_all("out");
    let out_path = "out/court-mdp-framework.json";
    std::fs::write(
        out_path,
        serde_json::to_string(&out).expect("serialize court-mdp result"),
    )
    .expect("write court-mdp-framework.json");
    println!();
    println!("# wrote {out_path}");

    // Optimal action distribution by stage (from π*).
    println!();
    println!("# Optimal action distribution by stage (from π*):");
    for stage in 0..4 {
        let mut counts = [0usize; N_ACTIONS];
        let mut total = 0;
        for s in 0..864usize {
            if let Some(cs) = decode(s) {
                if cs.stage as usize == stage {
                    let a = vi.policy[s];
                    if a >= 0 {
                        counts[a as usize] += 1;
                    }
                    total += 1;
                }
            }
        }
        let parts: Vec<String> = counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c > 0)
            .map(|(i, &c)| format!("{}={}", &ACTIONS[i][..4], c))
            .collect();
        println!(
            "#   {:<4} ({} states): {}",
            STAGES[stage],
            total,
            parts.join(" ")
        );
    }
}
