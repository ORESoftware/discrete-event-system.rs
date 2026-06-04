//! Port of `src/des/runners/validate-newsvendor.ts`.
//!
//! Validates the newsvendor and multi-period inventory MDP: critical-fractile ≡
//! brute search ≡ value iteration; γ→0 reduction; (s,S) vs base-stock structure;
//! simulation ≈ Bellman value; and native shadow references with an optional
//! Python (scipy/numpy) sidecar cross-check.
//! Driver → [`run`].
//!
//! PORT NOTES:
//!   * `crate::des::main_newsvendor::{analytical_optimal_q, brute_search_optimal_q,
//!     demand_poisson_pmf, demand_uniform_pmf, expected_profit, mdp_optimal_q,
//!     NewsvendorParams, simulate}`.
//!   * `crate::des::main_inventory_mdp::{detect_policy_structure, inventory_mdp_spec,
//!     InventoryParams, simulate_inventory_mdp}` (present as `main_inventory_mdp.rs`).
//!   * `crate::des::general::value_iteration::{value_iteration, q_value,
//!     VIOptions, MDPSpec}`
//!     (present).
//!   * Optional Python reference via `std::process::Command` is only attempted
//!     when `NEWSVENDOR_PY` is explicitly set. The default path stays Rust-only
//!     and validates against native reference projections.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;
use std::process::{Child, Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use crate::des::general::value_iteration::{q_value, value_iteration, MDPSpec, VIOptions};
use crate::des::main_inventory_mdp::{
    demand_poisson_pmf as inventory_demand_poisson_pmf, detect_policy_structure,
    inventory_mdp_spec, simulate_inventory_mdp, DemandDist as InventoryDemandDist, InventoryParams,
    PolicyKind,
};
use crate::des::main_newsvendor::{
    analytical_optimal_q, brute_search_optimal_q,
    demand_poisson_pmf as newsvendor_demand_poisson_pmf,
    demand_uniform_pmf as newsvendor_demand_uniform_pmf, expected_profit, mdp_optimal_q,
    DemandDist as NewsvendorDemandDist, NewsvendorParams,
};
use crate::des::observability::logger::{parse_json, JsonValue};

fn inventory_demand_from_newsvendor(demand: &NewsvendorDemandDist) -> InventoryDemandDist {
    InventoryDemandDist {
        pmf: demand.pmf.clone(),
    }
}

fn policy_kind_label(kind: &PolicyKind) -> &'static str {
    match kind {
        PolicyKind::BaseStock => "base-stock",
        PolicyKind::SS => "s-S",
        PolicyKind::Irregular => "irregular",
    }
}

fn policy_as_i64(policy: &[i32]) -> Vec<i64> {
    policy
        .iter()
        .map(|&value| i64::from(value.max(0)))
        .collect()
}

fn policy_as_usize(policy: &[i32]) -> Vec<usize> {
    policy
        .iter()
        .map(|&value| usize::try_from(value.max(0)).unwrap_or(0))
        .collect()
}

fn direct_expected_profit(q: usize, p: &NewsvendorParams) -> f64 {
    p.demand
        .pmf
        .iter()
        .enumerate()
        .map(|(d, &pr)| {
            let sold = q.min(d) as f64;
            let leftover = q.saturating_sub(d) as f64;
            pr * (p.unit_price * sold + p.unit_salvage * leftover - p.unit_cost * q as f64)
        })
        .sum()
}

fn max_bellman_residual(spec: &MDPSpec, v: &[f64], gamma: f64) -> f64 {
    let mut max_residual = 0.0_f64;
    for s in 0..spec.num_states {
        let terminal = spec.is_terminal.as_ref().map(|f| f(s)).unwrap_or(false);
        if terminal {
            continue;
        }
        let n_actions = (spec.num_actions)(s);
        if n_actions == 0 {
            continue;
        }
        let mut best = f64::NEG_INFINITY;
        for a in 0..n_actions {
            best = best.max(q_value(spec, v, s, a, gamma));
        }
        max_residual = max_residual.max((v[s] - best).abs());
    }
    max_residual
}

// Optional Python reference.
struct PyNewsvendor {
    q_star: i64,
    expected_profit_at_qstar: f64,
}
struct PyInventory {
    v_at_zero: f64,
    policy_first_20: Vec<i64>,
}
struct PyJson {
    newsvendor: PyNewsvendor,
    inventory_mdp: PyInventory,
}

fn get_any<'a>(value: &'a JsonValue, names: &[&str]) -> Option<&'a JsonValue> {
    names.iter().find_map(|name| value.get(name))
}

fn number_field(value: &JsonValue, names: &[&str], label: &str) -> Result<f64, String> {
    get_any(value, names)
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("missing numeric field {label}"))
}

fn i64_field(value: &JsonValue, names: &[&str], label: &str) -> Result<i64, String> {
    let n = number_field(value, names, label)?;
    if n.is_finite() && n.fract() == 0.0 {
        Ok(n as i64)
    } else {
        Err(format!("{label} must be an integer"))
    }
}

fn i64_array_field(value: &JsonValue, names: &[&str], label: &str) -> Result<Vec<i64>, String> {
    get_any(value, names)
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("missing array field {label}"))?
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let Some(n) = v.as_f64() else {
                return Err(format!("{label}[{i}] must be numeric"));
            };
            if n.is_finite() && n.fract() == 0.0 {
                Ok(n as i64)
            } else {
                Err(format!("{label}[{i}] must be an integer"))
            }
        })
        .collect()
}

fn parse_python_reference(text: &str) -> Result<PyJson, String> {
    let line = text
        .lines()
        .rev()
        .find(|line| !line.trim().is_empty())
        .ok_or_else(|| "Python reference produced no JSON output".to_string())?;
    let root = parse_json(line.trim())?;
    let newsvendor =
        get_any(&root, &["newsvendor"]).ok_or_else(|| "missing newsvendor result".to_string())?;
    let inventory = get_any(&root, &["inventoryMdp", "inventory_mdp"])
        .ok_or_else(|| "missing inventory_mdp result".to_string())?;
    Ok(PyJson {
        newsvendor: PyNewsvendor {
            q_star: i64_field(newsvendor, &["qStar", "q_star"], "newsvendor.qStar")?,
            expected_profit_at_qstar: number_field(
                newsvendor,
                &["expectedProfitAtQstar", "expected_profit_at_qstar"],
                "newsvendor.expectedProfitAtQstar",
            )?,
        },
        inventory_mdp: PyInventory {
            v_at_zero: number_field(inventory, &["vAtZero", "v_at_zero"], "inventoryMdp.vAtZero")?,
            policy_first_20: i64_array_field(
                inventory,
                &["policyFirst20", "policy_first_20"],
                "inventoryMdp.policyFirst20",
            )?,
        },
    })
}

fn newsvendor_python_timeout_ms() -> u64 {
    std::env::var("NEWSVENDOR_PY_TIMEOUT_MS")
        .ok()
        .or_else(|| std::env::var("PYTHON_REFERENCE_TIMEOUT_MS").ok())
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(120_000)
}

fn wait_for_newsvendor_python_output(
    mut child: Child,
    timeout_ms: u64,
) -> Result<(Output, bool), String> {
    let timeout = Duration::from_millis(timeout_ms);
    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => {
                return child
                    .wait_with_output()
                    .map(|output| (output, false))
                    .map_err(|err| format!("failed to wait for Python reference: {err}"));
            }
            Ok(None) => {}
            Err(err) => return Err(format!("failed to poll Python reference: {err}")),
        }

        if started.elapsed() >= timeout {
            let _ = child.kill();
            return child
                .wait_with_output()
                .map(|output| (output, true))
                .map_err(|err| format!("failed to collect timed-out Python reference: {err}"));
        }

        thread::sleep(Duration::from_millis(2));
    }
}

fn run_python(args: &[&str]) -> Result<Option<PyJson>, String> {
    let Some(python) = std::env::var("NEWSVENDOR_PY").ok() else {
        return Ok(None);
    };
    let script: PathBuf = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("external-references")
        .join("newsvendor")
        .join("newsvendor.py");
    let mut cmd = Command::new(python);
    cmd.arg(&script);
    for a in args {
        cmd.arg(a);
    }
    let timeout_ms = newsvendor_python_timeout_ms();
    let child = cmd
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to start Python reference: {e}"))?;
    match wait_for_newsvendor_python_output(child, timeout_ms) {
        Ok((out, timed_out)) if out.status.success() && !timed_out => {
            let stdout = String::from_utf8_lossy(&out.stdout);
            parse_python_reference(&stdout).map(Some)
        }
        Ok((mut out, timed_out)) => {
            if timed_out {
                let timeout_message = format!("Python reference timed out after {timeout_ms}ms");
                if out.stderr.is_empty() {
                    out.stderr = timeout_message.into_bytes();
                } else {
                    let mut stderr = String::from_utf8_lossy(&out.stderr).to_string();
                    stderr.push_str("; ");
                    stderr.push_str(&timeout_message);
                    out.stderr = stderr.into_bytes();
                }
            }
            Err(format!(
                "Python reference exited with status {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr).trim()
            ))
        }
        Err(e) => Err(format!("failed to run Python reference: {e}")),
    }
}

// =============================================================================
// Driver.
// =============================================================================

struct Checker {
    pass: u32,
    fail: u32,
}

impl Checker {
    fn new() -> Self {
        Checker { pass: 0, fail: 0 }
    }
    fn check(&mut self, label: &str, ok: bool, detail: &str) {
        if ok {
            self.pass += 1;
            println!("  PASS    {}", label);
        } else {
            self.fail += 1;
            let extra = if detail.is_empty() {
                String::new()
            } else {
                format!("  ({})", detail)
            };
            println!("  FAIL    {}{}", label, extra);
        }
    }
}

fn approx(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol
}

/// `validate-newsvendor.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    println!("\nStudy 1  Newsvendor: critical-fractile ≡ brute-search ≡ MDP");
    println!("==========================================================================");

    let scenarios: Vec<(&str, NewsvendorParams)> = vec![
        (
            "classic Poisson, λ=50, p/c/s = 1.0/0.5/0.1",
            NewsvendorParams {
                unit_cost: 0.5,
                unit_price: 1.0,
                unit_salvage: 0.1,
                demand: newsvendor_demand_poisson_pmf(50.0, 125),
                q_max: 125,
            },
        ),
        (
            "high margin, low salvage, λ=20",
            NewsvendorParams {
                unit_cost: 0.3,
                unit_price: 2.0,
                unit_salvage: 0.0,
                demand: newsvendor_demand_poisson_pmf(20.0, 60),
                q_max: 60,
            },
        ),
        (
            "low margin, high salvage, λ=100",
            NewsvendorParams {
                unit_cost: 0.9,
                unit_price: 1.0,
                unit_salvage: 0.7,
                demand: newsvendor_demand_poisson_pmf(100.0, 200),
                q_max: 200,
            },
        ),
        (
            "uniform demand U[10, 30]",
            NewsvendorParams {
                unit_cost: 0.5,
                unit_price: 1.0,
                unit_salvage: 0.1,
                demand: newsvendor_demand_uniform_pmf(10, 30, 40),
                q_max: 40,
            },
        ),
    ];

    for (name, params) in &scenarios {
        println!("\n  {}", name);
        let a = analytical_optimal_q(params);
        let b = brute_search_optimal_q(params);
        let m = mdp_optimal_q(params);
        println!(
            "    analytical q*={} (CR={:.4}),  brute q*={},  MDP q*={}",
            a.q_star, a.critical_ratio, b.q_star, m.q_star
        );
        println!(
            "    E[profit(q*)] analytical={:.4}  brute={:.4}  MDP V={:.4}",
            expected_profit(a.q_star, params),
            b.profile_ep[b.q_star],
            m.v0
        );
        c.check("analytical q* ≡ brute q*", a.q_star == b.q_star, "");
        c.check(
            "analytical q* ≡ MDP q*",
            usize::try_from(m.q_star).ok() == Some(a.q_star),
            "",
        );
        c.check(
            "E[profit] analytical ≡ brute",
            approx(
                expected_profit(a.q_star, params),
                b.profile_ep[b.q_star],
                1e-9,
            ),
            "",
        );
        c.check(
            "E[profit] analytical ≡ MDP V",
            approx(expected_profit(a.q_star, params), m.v0, 1e-9),
            "",
        );
    }

    println!("\nStudy 2  Multi-period MDP at γ=0 reduces to newsvendor");
    println!("==========================================================================");
    println!("  With γ=0 and \"salvage at end of day\" by setting unitCost = (effective);");
    println!("  the multi-period MDP at state x=0 should pick the newsvendor q*.");
    {
        let np = &scenarios[0].1;
        let ip = InventoryParams {
            x_max: np.q_max,
            a_max: np.q_max,
            demand: inventory_demand_from_newsvendor(&np.demand),
            unit_cost: np.unit_cost,
            fixed_cost: 0.0,
            unit_price: np.unit_price,
            hold_cost: -np.unit_salvage,
            lost_cost: 0.0,
            gamma: 0.0,
        };
        let spec = inventory_mdp_spec(&ip);
        let result = value_iteration(
            spec,
            VIOptions {
                gamma: 0.0,
                tol: 1e-12,
                ..Default::default()
            },
        );
        let policy_at_zero = result.policy[0];
        let newsvendor_q_star = analytical_optimal_q(np).q_star;
        println!(
            "    multi-period MDP π(0) = {}    newsvendor q* = {}",
            policy_at_zero, newsvendor_q_star
        );
        c.check(
            "γ=0 multi-period MDP π(0) = newsvendor q*",
            usize::try_from(policy_at_zero).ok() == Some(newsvendor_q_star),
            "",
        );
    }

    println!("\nStudy 3  Optimal policy structure: base-stock vs (s, S)");
    println!("==========================================================================");

    let inv_base = InventoryParams {
        x_max: 50,
        a_max: 50,
        demand: inventory_demand_poisson_pmf(20.0, 50),
        unit_cost: 1.0,
        fixed_cost: 0.0,
        unit_price: 2.0,
        hold_cost: 0.1,
        lost_cost: 0.5,
        gamma: 0.95,
    };

    {
        let mut params = inv_base.clone();
        params.fixed_cost = 0.0;
        let spec = inventory_mdp_spec(&params);
        let result = value_iteration(
            spec,
            VIOptions {
                gamma: params.gamma,
                tol: 1e-9,
                ..Default::default()
            },
        );
        let policy = policy_as_i64(&result.policy);
        let st = detect_policy_structure(&policy);
        println!(
            "  fixedCost = 0:  structure={}  S*={}  s*={}",
            policy_kind_label(&st.kind),
            st.s_level,
            st.reorder_point
        );
        c.check(
            "fixedCost=0 ⇒ base-stock policy",
            matches!(st.kind, PolicyKind::BaseStock),
            "",
        );
        c.check("base-stock S* > 0", st.s_level > 0, "");
        c.check(
            "base-stock S* ≤ xMax",
            st.s_level <= params.x_max as i64,
            "",
        );
    }

    {
        let mut params = inv_base.clone();
        params.fixed_cost = 10.0;
        let spec = inventory_mdp_spec(&params);
        let result = value_iteration(
            spec,
            VIOptions {
                gamma: params.gamma,
                tol: 1e-9,
                ..Default::default()
            },
        );
        let policy = policy_as_i64(&result.policy);
        let st = detect_policy_structure(&policy);
        println!(
            "  fixedCost = 10: structure={}  S*={}  s*={}",
            policy_kind_label(&st.kind),
            st.s_level,
            st.reorder_point
        );
        c.check(
            "fixedCost>0 ⇒ (s, S) policy",
            matches!(st.kind, PolicyKind::SS),
            "",
        );
        c.check(
            "s* < S* − 1 (gap due to setup cost)",
            st.reorder_point < st.s_level - 1,
            &format!("s={} S={}", st.reorder_point, st.s_level),
        );
    }

    {
        let ks = [0.0_f64, 1.0, 5.0, 10.0, 25.0, 50.0];
        let mut gaps: Vec<i64> = Vec::new();
        println!("\n  Sweep over fixedCost K:");
        println!("    K       S*    s*    S − s    structure");
        for k in ks {
            let mut p = inv_base.clone();
            p.fixed_cost = k;
            let spec = inventory_mdp_spec(&p);
            let r = value_iteration(
                spec,
                VIOptions {
                    gamma: p.gamma,
                    tol: 1e-9,
                    ..Default::default()
                },
            );
            let policy = policy_as_i64(&r.policy);
            let st = detect_policy_structure(&policy);
            let gap = st.s_level - st.reorder_point;
            gaps.push(gap);
            println!(
                "    {:>2}      {:>3}   {:>3}     {:>3}      {}",
                k as i64,
                st.s_level,
                st.reorder_point,
                gap,
                policy_kind_label(&st.kind)
            );
        }
        let mut monotonic = true;
        for i in 1..gaps.len() {
            if gaps[i] < gaps[i - 1] - 1 {
                monotonic = false;
                break;
            }
        }
        c.check("S − s gap is (weakly) increasing in K", monotonic, "");
    }

    println!("\nStudy 4  Simulation matches Bellman value");
    println!("==========================================================================");
    {
        let mut params = inv_base.clone();
        params.fixed_cost = 0.0;
        params.gamma = 0.95;
        let spec = inventory_mdp_spec(&params);
        let result = value_iteration(
            spec,
            VIOptions {
                gamma: params.gamma,
                tol: 1e-9,
                ..Default::default()
            },
        );
        let policy = policy_as_usize(&result.policy);
        let days = 50000usize;
        let sim = simulate_inventory_mdp(&params, &policy, days, 42, 0);
        let expected_avg = result.v[0] * (1.0 - params.gamma);
        println!(
            "    V(0) = {:.3},  V(0)·(1−γ) = {:.3}",
            result.v[0], expected_avg
        );
        println!(
            "    simulated mean reward over {} days = {:.3}",
            days, sim.mean_reward
        );
        let tol = 0.05 * expected_avg.abs();
        c.check(
            "simulation mean ≈ V(0)·(1−γ) within 5%",
            (sim.mean_reward - expected_avg).abs() < tol,
            &format!("sim={:.3} expected={:.3}", sim.mean_reward, expected_avg),
        );
    }

    println!("\nStudy 5  Native reference validation with optional Python sidecar");
    println!("==========================================================================");
    {
        let params = &scenarios[0].1;
        let analytical = analytical_optimal_q(params);
        let brute = brute_search_optimal_q(params);
        let ep_library = expected_profit(analytical.q_star, params);
        let ep_direct = direct_expected_profit(analytical.q_star, params);
        println!(
            "  newsvendor native: analytical q*={} brute q*={} EP(lib)={:.4} EP(direct)={:.4}",
            analytical.q_star, brute.q_star, ep_library, ep_direct
        );
        c.check(
            "newsvendor analytical q* matches brute-force reference",
            analytical.q_star == brute.q_star,
            "",
        );
        c.check(
            "newsvendor expected_profit matches direct PMF projection",
            approx(ep_library, ep_direct, 1e-12),
            &format!("|diff|={:.2e}", (ep_library - ep_direct).abs()),
        );

        let py = run_python(&["--lambda", "50", "--c", "0.5", "--p", "1.0", "--s", "0.1"]);
        match py {
            Ok(None) => println!(
                "  SKIP    optional Python newsvendor sidecar not configured (set NEWSVENDOR_PY)"
            ),
            Err(e) => c.check(
                "optional Python newsvendor reference runs and emits JSON",
                false,
                &e,
            ),
            Ok(Some(py)) => {
                println!(
                    "  newsvendor sidecar: Rust q*={} EP={:.4};  Py q*={} EP={:.4}",
                    analytical.q_star,
                    ep_library,
                    py.newsvendor.q_star,
                    py.newsvendor.expected_profit_at_qstar
                );
                c.check(
                    "newsvendor q* matches optional Python sidecar",
                    i64::try_from(analytical.q_star).ok() == Some(py.newsvendor.q_star),
                    "",
                );
                c.check(
                    "newsvendor E[profit] matches optional Python sidecar within 1e-6",
                    approx(ep_library, py.newsvendor.expected_profit_at_qstar, 1e-6),
                    &format!(
                        "|diff|={:.2e}",
                        (ep_library - py.newsvendor.expected_profit_at_qstar).abs()
                    ),
                );
            }
        }
    }

    {
        let params = InventoryParams {
            x_max: 50,
            a_max: 50,
            demand: inventory_demand_poisson_pmf(20.0, 51),
            unit_cost: 1.0,
            fixed_cost: 10.0,
            unit_price: 2.0,
            hold_cost: 0.1,
            lost_cost: 0.5,
            gamma: 0.95,
        };
        let spec = inventory_mdp_spec(&params);
        let ts_result = value_iteration(
            spec,
            VIOptions {
                gamma: params.gamma,
                tol: 1e-9,
                ..Default::default()
            },
        );
        let ts_policy = policy_as_i64(&ts_result.policy);
        let residual_spec = inventory_mdp_spec(&params);
        let residual = max_bellman_residual(&residual_spec, &ts_result.v, params.gamma);
        let shadow = value_iteration(
            inventory_mdp_spec(&params),
            VIOptions {
                gamma: params.gamma,
                tol: 1e-11,
                random_tie_break: false,
                ..Default::default()
            },
        );
        let shadow_policy = policy_as_i64(&shadow.policy);
        let structure = detect_policy_structure(&ts_policy);
        println!(
            "  inventory native: V(0)={:.4} residual={:.2e} policy={}({}, {})",
            ts_result.v[0],
            residual,
            policy_kind_label(&structure.kind),
            structure.reorder_point,
            structure.s_level
        );
        c.check(
            "multi-period Bellman residual is below 1e-6",
            residual <= 1e-6,
            &format!("residual={residual:.2e}"),
        );
        c.check(
            "multi-period strict re-solve keeps V(0) stable",
            approx(ts_result.v[0], shadow.v[0], 1e-4),
            &format!("|diff|={:.2e}", (ts_result.v[0] - shadow.v[0]).abs()),
        );
        c.check(
            "multi-period strict re-solve keeps policy[0..19] stable",
            ts_policy.iter().take(20).eq(shadow_policy.iter().take(20)),
            "",
        );
        let py = run_python(&[
            "--multi", "--lambda", "20", "--c", "1.0", "--K", "10", "--p", "2.0", "--h", "0.1",
            "--L", "0.5", "--gamma", "0.95", "--x-max", "50", "--a-max", "50",
        ]);
        match py {
            Ok(None) => println!("  SKIP    optional Python inventory sidecar not configured"),
            Err(e) => c.check(
                "optional Python inventory reference runs and emits JSON",
                false,
                &e,
            ),
            Ok(Some(py)) => {
                let ts_v0 = ts_result.v[0];
                let py_v0 = py.inventory_mdp.v_at_zero;
                println!(
                    "  multi-period sidecar: Rust V(0)={:.4},  Py V(0)={:.4}",
                    ts_v0, py_v0
                );
                let ts_head: Vec<String> =
                    ts_policy.iter().take(20).map(|v| v.to_string()).collect();
                let py_head: Vec<String> = py
                    .inventory_mdp
                    .policy_first_20
                    .iter()
                    .take(20)
                    .map(|v| v.to_string())
                    .collect();
                println!("  Rust policy[0..19] = [{}]", ts_head.join(", "));
                println!("  Py policy[0..19] = [{}]", py_head.join(", "));
                c.check(
                    "multi-period V(0) matches optional Python sidecar within 1e-3",
                    approx(ts_v0, py_v0, 1e-3),
                    &format!("|diff|={:.2e}", (ts_v0 - py_v0).abs()),
                );
                let mut policy_match = true;
                for x in 0..20 {
                    if ts_policy.get(x).copied().unwrap_or(0)
                        != py
                            .inventory_mdp
                            .policy_first_20
                            .get(x)
                            .copied()
                            .unwrap_or(0)
                    {
                        policy_match = false;
                        break;
                    }
                }
                c.check(
                    "multi-period policy[0..19] matches optional Python sidecar",
                    policy_match,
                    "",
                );
            }
        }
    }

    println!("\nsummary: {} pass, {} fail", c.pass, c.fail);
    std::process::exit(if c.fail == 0 { 0 } else { 1 });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newsvendor_python_sidecar_wait_enforces_timeout() {
        let child = Command::new("sleep")
            .arg("1")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .expect("spawn sleep");

        let (output, timed_out) =
            wait_for_newsvendor_python_output(child, 10).expect("timeout output");
        let stderr = String::from_utf8_lossy(&output.stderr);

        assert!(timed_out);
        assert!(!output.status.success());
        assert!(stderr.is_empty());
    }
}
