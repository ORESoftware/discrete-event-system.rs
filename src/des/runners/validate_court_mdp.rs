//! Port of `src/des/runners/validate-court-mdp.ts`.
//!
//! Compares the framework USACC MDP value iteration
//! generated from the Rust model against the Python reference
//! (`out/external/court-mdp/python.json`): reports max-abs V* error + policy
//! disagreement count, asserting both match within `1e-7` and 0 disagreements.
//! The top-level `main()` becomes [`run`].

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::des::main_court_mdp::{
    run_court_sim, AlwaysEscalatePolicy, CourtAggregates, CourtMDPConfig, CourtMDPResult,
    NaiveThresholdPolicy, OptimalPolicy, Policy, RejectAllPolicy,
};
// The MDP label tables, `decode`, and `is_terminal` are the real model. The
// original port stubbed them locally with the wrong arity (N_STATES = 1875 and
// invented action/stage labels); they now come straight from `usacc_mdp`, which
// is the same model the framework writer (`main_court_mdp`) value-iterates over.
use crate::des::mdp::usacc_mdp::{
    decode, is_terminal, ACTIONS, CORROBORATION, EVIDENCE, MANIPULATION, N_STATES, STAGES,
};
use crate::des::mdp::value_iteration::{value_iteration, VIOptions};

// =============================================================================
// Typed views of the two JSON files. The framework writer emits camelCase keys
// (`finalDelta`, `meanReward`, …) and an uppercase `V` array; `serde(default)`
// keeps the python reference tolerant of omitted fields.
// =============================================================================

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct ViBlock {
    #[serde(rename = "V")]
    v: Vec<f64>,
    policy: Vec<i64>,
    gamma: f64,
    iterations: usize,
    final_delta: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct Aggregates {
    mean_reward: f64,
    fraction_accepted: f64,
    fraction_closed: f64,
    fraction_exhausted: f64,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct ResultRow {
    policy: String,
    aggregates: Aggregates,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
struct CourtMdpFramework {
    vi: ViBlock,
    results: Vec<ResultRow>,
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default, rename_all = "camelCase")]
struct PythonReference {
    #[serde(rename = "V")]
    v: Vec<f64>,
    policy: Vec<i64>,
    iterations: usize,
    final_delta: f64,
}

fn load_optional_json<T: serde::de::DeserializeOwned>(p: &Path) -> Option<T> {
    if !p.exists() {
        return None;
    }
    let text = std::fs::read_to_string(p).unwrap_or_else(|e| {
        eprintln!("[validate-court-mdp] read error {}: {e}", p.display());
        std::process::exit(1);
    });
    Some(serde_json::from_str(&text).unwrap_or_else(|e| {
        eprintln!("[validate-court-mdp] parse error {}: {e}", p.display());
        std::process::exit(1);
    }))
}

fn root() -> PathBuf {
    // `path.join(__dirname, '..', '..', '..')` → the repo/crate root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn convert_aggregates(a: &CourtAggregates) -> Aggregates {
    Aggregates {
        mean_reward: a.mean_reward,
        fraction_accepted: a.fraction_accepted,
        fraction_closed: a.fraction_closed,
        fraction_exhausted: a.fraction_exhausted,
    }
}

fn convert_result(r: CourtMDPResult) -> ResultRow {
    ResultRow {
        policy: r.policy,
        aggregates: convert_aggregates(&r.aggregates),
    }
}

fn build_framework_json() -> CourtMdpFramework {
    let vi = value_iteration(VIOptions {
        gamma: 0.95,
        tol: 1e-10,
        max_iter: 5000,
    });
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
        total_cases: env_usize("CASES", 5000),
        arrivals_per_tick: env_usize("ARRIVALS_PER_TICK", 5),
        max_ticks: env_usize("MAX_TICKS", 10000),
        seed: std::env::var("SEED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(42u32),
    };
    let results = policies
        .iter()
        .map(|p| convert_result(run_court_sim(cfg, p.as_ref())))
        .collect();
    let policy = vi.policy.iter().map(|&a| a as i64).collect();
    CourtMdpFramework {
        vi: ViBlock {
            v: vi.v,
            policy,
            gamma: vi.gamma,
            iterations: vi.iterations,
            final_delta: vi.final_delta,
        },
        results,
    }
}

fn framework_ok(ts: &CourtMdpFramework) -> bool {
    ts.vi.v.len() == N_STATES
        && ts.vi.policy.len() == N_STATES
        && ts.vi.v.iter().all(|v| v.is_finite())
        && ts.vi.final_delta < 1e-7
        && !ts.results.is_empty()
}

/// `validate-court-mdp.ts` `main()`.
pub fn run() {
    let py_path = root()
        .join("out")
        .join("external")
        .join("court-mdp")
        .join("python.json");

    let ts = build_framework_json();
    let py: Option<PythonReference> = load_optional_json(&py_path);

    println!("USACC MDP: framework value iteration vs Python value iteration");
    println!("==================================================================");
    if py.is_none() {
        let ok = framework_ok(&ts);
        println!(
            "  γ = {}    framework iters = {}",
            ts.vi.gamma, ts.vi.iterations
        );
        println!("  framework final |ΔV| = {:.3e}", ts.vi.final_delta);
        println!();
        println!("  Policy comparison (framework simulation, last run):");
        for r in &ts.results {
            let a = &r.aggregates;
            println!(
                "    {:<18}  meanReward={:>8}    accepted={:>5}%    closed={:>5}%    exhausted={:>5}%",
                r.policy,
                format!("{:.2}", a.mean_reward),
                format!("{:.1}", a.fraction_accepted * 100.0),
                format!("{:.1}", a.fraction_closed * 100.0),
                format!("{:.1}", a.fraction_exhausted * 100.0)
            );
        }
        println!(
            "  SKIP  Python value-iteration comparison (reference JSON unavailable: {})",
            py_path.display()
        );
        println!("{}", if ok { "  PASS" } else { "  FAIL" });
        std::process::exit(if ok { 0 } else { 1 });
    }
    let py = py.expect("checked present");
    let v_ts = &ts.vi.v;
    let v_py = &py.v;
    let pi_ts = &ts.vi.policy;
    let pi_py = &py.policy;

    println!(
        "  γ = {}    framework iters = {}    python iters = {}",
        ts.vi.gamma, ts.vi.iterations, py.iterations
    );
    println!(
        "  framework final |ΔV| = {:.3e}    python = {:.3e}",
        ts.vi.final_delta, py.final_delta
    );

    let mut max_v = 0.0_f64;
    let mut max_at_state: i64 = -1;
    for s in 0..N_STATES {
        let d = (v_ts[s] - v_py[s]).abs();
        if d > max_v {
            max_v = d;
            max_at_state = s as i64;
        }
    }
    let mut p_disagree = 0usize;
    let mut first_disagree_state: i64 = -1;
    for s in 0..N_STATES {
        if is_terminal(s) {
            continue;
        }
        if pi_ts[s] != pi_py[s] {
            p_disagree += 1;
            if first_disagree_state < 0 {
                first_disagree_state = s as i64;
            }
        }
    }

    println!(
        "  max |V_ts(s) - V_py(s)|       = {:.3e}  (at state {})",
        max_v, max_at_state
    );
    println!(
        "  policy disagreement count    = {} / {}",
        p_disagree,
        N_STATES - 3
    );
    if p_disagree > 0 && first_disagree_state >= 0 {
        let cs = decode(first_disagree_state as usize).unwrap();
        println!(
            "    first disagree: state {} = ({}, ev={}, corr={}, man={}, conf={}, fund={})",
            first_disagree_state,
            STAGES[cs.stage as usize],
            EVIDENCE[cs.evidence as usize],
            CORROBORATION[cs.corroboration as usize],
            MANIPULATION[cs.manipulation as usize],
            if cs.conflict != 0 { "HI" } else { "LO" },
            cs.funding
        );
        println!(
            "      framework picks {}, python picks {}",
            ACTIONS[pi_ts[first_disagree_state as usize] as usize],
            ACTIONS[pi_py[first_disagree_state as usize] as usize]
        );
    }

    println!();
    println!("  Policy comparison (framework simulation, last run):");
    for r in &ts.results {
        let a = &r.aggregates;
        println!(
            "    {:<18}  meanReward={:>8}    accepted={:>5}%    closed={:>5}%    exhausted={:>5}%",
            r.policy,
            format!("{:.2}", a.mean_reward),
            format!("{:.1}", a.fraction_accepted * 100.0),
            format!("{:.1}", a.fraction_closed * 100.0),
            format!("{:.1}", a.fraction_exhausted * 100.0)
        );
    }

    let tol_v = 1e-7;
    let ok = max_v < tol_v && p_disagree == 0;
    println!();
    println!(
        "  max V diff < {:.0e}: {}",
        tol_v,
        if max_v < tol_v { "yes" } else { "NO" }
    );
    println!(
        "  policies identical: {}",
        if p_disagree == 0 { "yes" } else { "NO" }
    );
    println!("{}", if ok { "  PASS" } else { "  FAIL" });
    std::process::exit(if ok { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The framework structs must deserialize what `main_court_mdp` writes to
    /// `out/court-mdp-framework.json` (camelCase aggregates, uppercase `V`,
    /// extra aggregate fields ignored).
    #[test]
    fn framework_json_parses_sim_output_shape() {
        let json = r#"{
            "config": {"totalCases": 5000, "arrivalsPerTick": 5, "maxTicks": 10000, "seed": 42},
            "vi": {"gamma": 0.95, "iterations": 200, "finalDelta": 1e-9,
                   "V": [0.0, 1.0, 2.0], "policy": [0, 7, 3]},
            "results": [
                {"policy": "reject-all",
                 "aggregates": {"n": 5000, "nAccepted": 10, "meanReward": -1.0,
                                "meanSteps": 2.0, "p95Steps": 4,
                                "fractionAccepted": 0.1, "fractionClosed": 0.5,
                                "fractionExhausted": 0.4}}
            ]
        }"#;
        let fw: CourtMdpFramework = serde_json::from_str(json).expect("parse framework json");
        assert_eq!(fw.vi.v, vec![0.0, 1.0, 2.0]);
        assert_eq!(fw.vi.policy, vec![0, 7, 3]);
        assert_eq!(fw.vi.final_delta, 1e-9);
        assert_eq!(fw.results.len(), 1);
        assert_eq!(fw.results[0].policy, "reject-all");
        assert_eq!(fw.results[0].aggregates.mean_reward, -1.0);
        assert_eq!(fw.results[0].aggregates.fraction_accepted, 0.1);
    }
}
