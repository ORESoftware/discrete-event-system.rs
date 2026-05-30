//! Port of `src/des/runners/validate-court-mdp.ts`.
//!
//! Compares the framework USACC MDP value iteration
//! (`out/court-mdp-framework.json`) against the Python reference
//! (`out/external/court-mdp/python.json`): reports max-abs V* error + policy
//! disagreement count, asserting both match within `1e-7` and 0 disagreements.
//! The top-level `main()` becomes [`run`].
//!
//! PORT NOTES (cross-module deps to wire later):
//!   * JSON loading is stubbed — the crate has no `serde`/`serde_json` dependency
//!     yet. The `load_*` helpers faithfully reproduce the missing-file `exit(1)`
//!     and, on the happy path, document the `serde_json::from_str` call to wire.
//!   * The MDP label tables (`STAGES`/`EVIDENCE`/…), `is_terminal`, and `decode`
//!     are local stubs; reuse `crate::des::mdp::usacc_mdp::*` once
//!     `runners/mod.rs` declares this module (cannot create `mod.rs` here).

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::{Path, PathBuf};

// =============================================================================
// Stubbed cross-module deps — PORT NOTE: reuse `crate::des::mdp::usacc_mdp`.
// =============================================================================

/// Number of MDP states (stub). PORT NOTE: `usacc_mdp::N_STATES`.
const N_STATES: usize = 1875;

/// Action labels (stub). PORT NOTE: `usacc_mdp::ACTIONS`.
const ACTIONS: &[&str] = &["dismiss", "investigate", "charge", "settle", "trial"];
const STAGES: &[&str] = &["intake", "discovery", "pretrial", "trial"];
const EVIDENCE: &[&str] = &["none", "weak", "moderate", "strong"];
const CORROBORATION: &[&str] = &["none", "partial", "full"];
const MANIPULATION: &[&str] = &["none", "suspected", "confirmed"];

/// A decoded composite court state. PORT NOTE: `usacc_mdp::CourtState`.
#[derive(Clone, Copy, Debug)]
struct CourtState {
    stage: usize,
    evidence: usize,
    corroboration: usize,
    manipulation: usize,
    conflict: bool,
    funding: i64,
}

/// PORT NOTE: `usacc_mdp::is_terminal`.
fn is_terminal(s: usize) -> bool {
    s >= N_STATES.saturating_sub(3)
}

/// PORT NOTE: `usacc_mdp::decode`.
fn decode(s: usize) -> Option<CourtState> {
    if s >= N_STATES {
        return None;
    }
    Some(CourtState {
        stage: 0,
        evidence: 0,
        corroboration: 0,
        manipulation: 0,
        conflict: false,
        funding: 0,
    })
}

// =============================================================================
// Typed views of the two JSON files (PORT NOTE: `#[derive(Deserialize)]`).
// =============================================================================

#[derive(Clone, Debug, Default)]
struct ViBlock {
    v: Vec<f64>,
    policy: Vec<i64>,
    gamma: f64,
    iterations: usize,
    final_delta: f64,
}

#[derive(Clone, Debug, Default)]
struct Aggregates {
    mean_reward: f64,
    fraction_accepted: f64,
    fraction_closed: f64,
    fraction_exhausted: f64,
}

#[derive(Clone, Debug, Default)]
struct ResultRow {
    policy: String,
    aggregates: Aggregates,
}

#[derive(Clone, Debug, Default)]
struct CourtMdpFramework {
    vi: ViBlock,
    results: Vec<ResultRow>,
}

#[derive(Clone, Debug, Default)]
struct PythonReference {
    v: Vec<f64>,
    policy: Vec<i64>,
    iterations: usize,
    final_delta: f64,
}

/// `loadJson` — faithful missing-file `exit(1)`; the parse step is stubbed.
///
/// PORT NOTE: once `serde_json` is a dependency, replace the diverging tail with
/// `serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()`.
fn load_json<T>(p: &Path) -> T {
    if !p.exists() {
        eprintln!("[validate-court-mdp] missing {}", p.display());
        std::process::exit(1);
    }
    eprintln!(
        "[validate-court-mdp] PORT NOTE: JSON parsing not wired (needs serde_json): {}",
        p.display()
    );
    std::process::exit(1);
}

fn root() -> PathBuf {
    // `path.join(__dirname, '..', '..', '..')` → the repo/crate root.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `validate-court-mdp.ts` `main()`.
pub fn run() {
    let ts_path = root().join("out").join("court-mdp-framework.json");
    let py_path = root()
        .join("out")
        .join("external")
        .join("court-mdp")
        .join("python.json");

    let ts: CourtMdpFramework = load_json(&ts_path);
    let py: PythonReference = load_json(&py_path);

    let v_ts = &ts.vi.v;
    let v_py = &py.v;
    let pi_ts = &ts.vi.policy;
    let pi_py = &py.policy;

    println!("USACC MDP: framework value iteration vs Python value iteration");
    println!("==================================================================");
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
            STAGES[cs.stage],
            EVIDENCE[cs.evidence],
            CORROBORATION[cs.corroboration],
            MANIPULATION[cs.manipulation],
            if cs.conflict { "HI" } else { "LO" },
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
