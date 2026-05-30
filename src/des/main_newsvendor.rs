//! Port of `src/des/main-newsvendor.ts`.
//!
//! The classic single-period newsvendor problem: finds q* three ways — closed
//! form critical fractile, brute search over the demand PMF, and a 1-step MDP
//! value iteration — then simulates the day-by-day policy in the DES framework.
//!
//! Conversion notes:
//!   - day-by-day demand sampling routes through the seeded `SeededRandom`.
//!   - closed-form / VI are pure fns; `NewsvendorStation` is a `DESStation`.
//!   - `async main` → [`run`]; `JSON.stringify` → a hand-built JSON string
//!     (no `serde_json` dependency in this crate).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::prng::{mulberry32, with_seed};
use crate::des::general::value_iteration::{value_iteration, MDPSpec, Outcome, VIOptions};
use crate::des::shared::capabilities::{RandomSource, SeededRandom};

// -----------------------------------------------------------------------------
// Demand distribution (discrete on {0, 1, …, dMax}).
// -----------------------------------------------------------------------------

/// Discrete demand distribution: `pmf[k] = P(D = k)`.
#[derive(Clone, Debug)]
pub struct DemandDist {
    pub pmf: Vec<f64>,
}

pub fn demand_poisson_pmf(lambda: f64, d_max: usize) -> DemandDist {
    let mut pmf = vec![0.0; d_max + 1];
    let mut p = (-lambda).exp();
    pmf[0] = p;
    for k in 1..=d_max {
        p = p * lambda / k as f64;
        pmf[k] = p;
    }
    let total: f64 = pmf.iter().sum();
    pmf[d_max] += 1.0 - total;
    DemandDist { pmf }
}

#[allow(dead_code)]
pub fn demand_uniform_pmf(lo: usize, hi: usize, d_max: usize) -> DemandDist {
    let mut pmf = vec![0.0; d_max + 1];
    let w = (hi - lo + 1) as f64;
    let mut k = lo;
    while k <= hi && k <= d_max {
        pmf[k] = 1.0 / w;
        k += 1;
    }
    DemandDist { pmf }
}

pub fn cdf_from_pmf(d: &DemandDist) -> Vec<f64> {
    let mut cdf = vec![0.0; d.pmf.len()];
    let mut acc = 0.0;
    for k in 0..d.pmf.len() {
        acc += d.pmf[k];
        cdf[k] = acc;
    }
    cdf
}

pub fn sample_demand(d: &DemandDist, rng: &mut dyn RandomSource) -> usize {
    let u = rng.next_float();
    let mut acc = 0.0;
    for k in 0..d.pmf.len() {
        acc += d.pmf[k];
        if u <= acc {
            return k;
        }
    }
    d.pmf.len() - 1
}

// -----------------------------------------------------------------------------
// Newsvendor cost / profit functions.
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct NewsvendorParams {
    pub unit_cost: f64,
    pub unit_price: f64,
    pub unit_salvage: f64,
    pub demand: DemandDist,
    pub q_max: usize,
}

pub fn profit(q: usize, d: usize, p: &NewsvendorParams) -> f64 {
    let sold = q.min(d) as f64;
    let leftover = if q > d { (q - d) as f64 } else { 0.0 };
    p.unit_price * sold + p.unit_salvage * leftover - p.unit_cost * q as f64
}

/// Exact expected profit for order quantity `q` under the discrete demand PMF.
pub fn expected_profit(q: usize, p: &NewsvendorParams) -> f64 {
    let mut e = 0.0;
    for d in 0..p.demand.pmf.len() {
        e += p.demand.pmf[d] * profit(q, d, p);
    }
    e
}

// -----------------------------------------------------------------------------
// (a) ANALYTICAL: critical-fractile q*.
// -----------------------------------------------------------------------------

pub struct AnalyticalResult {
    pub q_star: usize,
    pub critical_ratio: f64,
}

pub fn analytical_optimal_q(p: &NewsvendorParams) -> AnalyticalResult {
    let cu = p.unit_price - p.unit_cost;
    let co = p.unit_cost - p.unit_salvage;
    if cu <= 0.0 {
        return AnalyticalResult {
            q_star: 0,
            critical_ratio: 0.0,
        };
    }
    let cr = cu / (cu + co);
    let cdf = cdf_from_pmf(&p.demand);
    for k in 0..cdf.len() {
        if cdf[k] >= cr {
            return AnalyticalResult {
                q_star: k,
                critical_ratio: cr,
            };
        }
    }
    AnalyticalResult {
        q_star: cdf.len() - 1,
        critical_ratio: cr,
    }
}

// -----------------------------------------------------------------------------
// (b) BRUTE-SEARCH: argmax_q E[profit(q)].
// -----------------------------------------------------------------------------

pub struct BruteResult {
    pub q_star: usize,
    pub profile_ep: Vec<f64>,
}

pub fn brute_search_optimal_q(p: &NewsvendorParams) -> BruteResult {
    let mut profile = vec![0.0; p.q_max + 1];
    let mut best = f64::NEG_INFINITY;
    let mut best_q = 0usize;
    for q in 0..=p.q_max {
        profile[q] = expected_profit(q, p);
        if profile[q] > best {
            best = profile[q];
            best_q = q;
        }
    }
    BruteResult {
        q_star: best_q,
        profile_ep: profile,
    }
}

// -----------------------------------------------------------------------------
// (c) MDP VALUE ITERATION: 1-step MDP with γ = 0.
// -----------------------------------------------------------------------------

pub fn newsvendor_mdp_spec(p: &NewsvendorParams) -> MDPSpec {
    let q_max = p.q_max;
    let p_outcomes = p.clone();
    MDPSpec {
        num_states: 2,
        num_actions: Box::new(move |s| if s == 0 { q_max + 1 } else { 0 }),
        outcomes: Box::new(move |s, a| {
            if s != 0 {
                return vec![];
            }
            vec![Outcome {
                prob: 1.0,
                reward: expected_profit(a, &p_outcomes),
                next_state: 1,
            }]
        }),
        is_terminal: Some(Box::new(|s| s == 1)),
        terminal_reward: None,
        state_label: Some(Box::new(|s| ["morning", "end-of-day"][s].to_string())),
        action_label: Some(Box::new(|a| format!("q={}", a))),
    }
}

pub struct MdpResult {
    pub q_star: i32,
    pub v0: f64,
    pub iterations: usize,
}

pub fn mdp_optimal_q(p: &NewsvendorParams) -> MdpResult {
    let spec = newsvendor_mdp_spec(p);
    let result = value_iteration(
        spec,
        VIOptions {
            gamma: 0.0,
            tol: 1e-12,
            ..Default::default()
        },
    );
    MdpResult {
        q_star: result.policy[0],
        v0: result.v[0],
        iterations: result.iterations,
    }
}

// -----------------------------------------------------------------------------
// Framework simulation.
// -----------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct HistoryEntry {
    pub day: usize,
    pub q: usize,
    pub demand: usize,
    pub profit: f64,
    pub sold: usize,
    pub leftover: usize,
}

struct NewsvendorStation {
    core: StationCore,
    rng: SeededRandom,
    params: NewsvendorParams,
    q: usize,
    total_days: usize,
    total_profit: f64,
    days_simulated: usize,
    unmet_demand: f64,
    total_leftover: f64,
    history: Vec<HistoryEntry>,
}

impl NewsvendorStation {
    fn new(params: NewsvendorParams, q: usize, days: usize, seed: u32) -> Self {
        NewsvendorStation {
            core: StationCore::new("newsvendor-station"),
            rng: mulberry32(seed),
            params,
            q,
            total_days: days,
            total_profit: 0.0,
            days_simulated: 0,
            unmet_demand: 0.0,
            total_leftover: 0.0,
            history: Vec::new(),
        }
    }
}

impl DESStation for NewsvendorStation {
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
        self.days_simulated < self.total_days
    }
    fn run_time_step(&mut self) {
        if !self.has_work() {
            return;
        }
        let day = self.days_simulated;
        let d = sample_demand(&self.params.demand, &mut self.rng);
        let sold = self.q.min(d);
        let leftover = if self.q > d { self.q - d } else { 0 };
        let lost = if d > self.q { d - self.q } else { 0 };
        let pi = profit(self.q, d, &self.params);
        self.total_profit += pi;
        self.unmet_demand += lost as f64;
        self.total_leftover += leftover as f64;
        self.days_simulated += 1;
        self.history.push(HistoryEntry {
            day,
            q: self.q,
            demand: d,
            profit: pi,
            sold,
            leftover,
        });
    }
}

pub struct SimResult {
    pub mean_profit: f64,
    pub avg_leftover: f64,
    pub avg_unmet: f64,
    pub history: Vec<HistoryEntry>,
}

pub fn simulate(params: &NewsvendorParams, q: usize, days: usize, seed: u32) -> SimResult {
    with_seed(seed, |_rng| {
        let sta = Rc::new(RefCell::new(NewsvendorStation::new(
            params.clone(),
            q,
            days,
            seed,
        )));
        let sta_ref: StationRef = sta.clone();
        run_iterative_des(
            vec![sta_ref],
            IterativeRunOptions {
                shuffle: false,
                max_ticks: Some(days + 2),
                run_validators: false,
                ..Default::default()
            },
        );
        let b = sta.borrow();
        SimResult {
            mean_profit: b.total_profit / b.days_simulated as f64,
            avg_leftover: b.total_leftover / b.days_simulated as f64,
            avg_unmet: b.unmet_demand / b.days_simulated as f64,
            history: b.history.clone(),
        }
    })
}

// -----------------------------------------------------------------------------
// JSON helpers (no serde_json dependency).
// -----------------------------------------------------------------------------

/// `JSON.stringify` of a finite number: integers without trailing `.0`.
fn jn(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let lambda = env_f64("LAMBDA", 50.0);
    let d_max = env_usize("D_MAX", (lambda * 2.5).ceil() as usize);
    let params = NewsvendorParams {
        unit_cost: env_f64("UNIT_COST", 0.50),
        unit_price: env_f64("UNIT_PRICE", 1.00),
        unit_salvage: env_f64("UNIT_SALVAGE", 0.10),
        demand: demand_poisson_pmf(lambda, d_max),
        q_max: env_usize("Q_MAX", (lambda * 2.5).ceil() as usize),
    };
    let days = env_usize("DAYS", 1000);
    let seed = env_usize("SEED", 1) as u32;

    println!(
        "# Newsvendor: c={}, p={}, s={}",
        jn(params.unit_cost),
        jn(params.unit_price),
        jn(params.unit_salvage)
    );
    println!(
        "#   demand = Poisson(λ={}), truncated at {};  qMax={}",
        jn(lambda),
        d_max,
        params.q_max
    );
    println!(
        "#   underage cost c_u = p−c = {:.3}",
        params.unit_price - params.unit_cost
    );
    println!(
        "#   overage  cost c_o = c−s = {:.3}",
        params.unit_cost - params.unit_salvage
    );

    // (a) Analytical critical-fractile.
    let a = analytical_optimal_q(&params);
    println!();
    println!("(a) Analytical critical-fractile");
    println!(
        "    critical ratio = c_u / (c_u + c_o) = {:.4}",
        a.critical_ratio
    );
    println!("    q* = inf{{q : P(D ≤ q) ≥ CR}} = {}", a.q_star);
    println!(
        "    E[profit(q*)] = {:.4}",
        expected_profit(a.q_star, &params)
    );

    // (b) Brute search over E[profit(q)].
    let b = brute_search_optimal_q(&params);
    println!();
    println!("(b) Brute search over q ∈ [0, {}]", params.q_max);
    println!("    q*  = {}", b.q_star);
    println!("    E[profit(q*)] = {:.4}", b.profile_ep[b.q_star]);

    // (c) MDP value iteration.
    let c = mdp_optimal_q(&params);
    println!();
    println!("(c) MDP value iteration (1-step, γ=0)");
    println!("    q*       = {}", c.q_star);
    println!("    V(state=morning) = {:.4}", c.v0);
    println!("    iterations = {}", c.iterations);

    // Simulate at q* for sanity check.
    let sim = simulate(&params, a.q_star, days, seed);
    println!();
    println!("(sim) {}-day simulation at q = q* = {}", days, a.q_star);
    println!(
        "    mean profit/day  = {:.4}   (analytical {:.4})",
        sim.mean_profit,
        expected_profit(a.q_star, &params)
    );
    println!("    avg leftover/day = {:.2}", sim.avg_leftover);
    println!("    avg unmet/day    = {:.2}", sim.avg_unmet);

    let out_dir = std::path::Path::new("out");
    let _ = std::fs::create_dir_all(out_dir);
    let out_path = out_dir.join("newsvendor.json");

    let profile_json = b
        .profile_ep
        .iter()
        .map(|v| jn(*v))
        .collect::<Vec<_>>()
        .join(",");
    let history_json = sim
        .history
        .iter()
        .map(|h| {
            format!(
                "{{\"day\":{},\"q\":{},\"demand\":{},\"profit\":{},\"sold\":{},\"leftover\":{}}}",
                h.day,
                h.q,
                h.demand,
                jn(h.profit),
                h.sold,
                h.leftover
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        concat!(
            "{{\"params\":{{\"unitCost\":{},\"unitPrice\":{},\"unitSalvage\":{},\"qMax\":{},",
            "\"demandLambda\":{},\"dMax\":{}}},\"days\":{},\"seed\":{},",
            "\"analytical\":{{\"qStar\":{},\"criticalRatio\":{}}},",
            "\"bruteSearch\":{{\"qStar\":{},\"profileEP\":[{}]}},",
            "\"mdp\":{{\"qStar\":{},\"V0\":{},\"iterations\":{}}},",
            "\"simulation\":{{\"meanProfit\":{},\"avgLeftover\":{},\"avgUnmet\":{},",
            "\"history\":[{}]}}}}"
        ),
        jn(params.unit_cost),
        jn(params.unit_price),
        jn(params.unit_salvage),
        params.q_max,
        jn(lambda),
        d_max,
        days,
        seed,
        a.q_star,
        jn(a.critical_ratio),
        b.q_star,
        profile_json,
        c.q_star,
        jn(c.v0),
        c.iterations,
        jn(sim.mean_profit),
        jn(sim.avg_leftover),
        jn(sim.avg_unmet),
        history_json
    );
    let _ = std::fs::write(&out_path, json);
    println!("# wrote {}", out_path.display());
}
