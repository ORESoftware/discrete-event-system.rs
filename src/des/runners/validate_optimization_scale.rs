//! Scale-envelope checks for native optimization solvers against external
//! open-source engines.
//!
//! This is not a performance shootout. It is a deterministic guardrail that
//! answers the practical parity question: as generated LP/MIP instances grow
//! beyond the tiny feature tests, do the native Rust solvers still agree with
//! installed reference engines, and where do timings start to separate?

#![allow(dead_code)]

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::des::general::ip_mip_des::{
    build_binary_knapsack_ip, solve_ipmip_with_des, ConcreteLpRelaxationAlgorithm, IPMIPProblem,
    IPMIPSolveOptions, IPMIPStatus, LpRelaxationAlgorithm,
};
use crate::des::general::lp::{
    solve_lp_external, solve_lp_internal, ExternalSolverOptions, InternalSimplexOptions, LPProblem,
    LPStatus, Sense,
};

#[derive(Clone, Debug)]
struct CheckRow {
    name: String,
    passed: bool,
    detail: String,
}

#[derive(Clone, Debug, Serialize)]
struct ScaleRow {
    family: String,
    size: usize,
    constraints: usize,
    native_solver: String,
    external_solver: String,
    native_status: String,
    external_status: String,
    native_objective: f64,
    external_objective: f64,
    objective_abs_diff: f64,
    native_ms: f64,
    external_ms: f64,
    native_nodes: Option<usize>,
    native_lp_solves: Option<usize>,
}

#[derive(Clone, Debug, Serialize)]
struct ScaleReport {
    generated_at_unix_ms: u128,
    lp_sizes: Vec<usize>,
    mip_sizes: Vec<usize>,
    rows: Vec<ScaleRow>,
}

#[derive(Debug, Deserialize)]
struct LinearCliReference {
    status: String,
    solver: String,
    objective: Option<f64>,
    message: String,
}

struct Driver {
    root: PathBuf,
    checks: Vec<CheckRow>,
    rows: Vec<ScaleRow>,
}

impl Driver {
    fn new() -> Self {
        let root = std::env::var("REPO_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
        Driver {
            root,
            checks: Vec::new(),
            rows: Vec::new(),
        }
    }

    fn check(&mut self, name: impl Into<String>, passed: bool, detail: impl Into<String>) {
        let name = name.into();
        let detail = detail.into();
        println!(
            "  {}  {}{}",
            if passed { "PASS" } else { "FAIL" },
            name,
            if detail.is_empty() {
                String::new()
            } else {
                format!(" - {detail}")
            }
        );
        self.checks.push(CheckRow {
            name,
            passed,
            detail,
        });
    }

    fn close(&mut self, name: impl Into<String>, a: f64, b: f64, tol: f64) {
        let diff = (a - b).abs();
        self.check(
            name,
            diff <= tol,
            format!("native={a:.10} external={b:.10} diff={diff:.3e} tol={tol:.1e}"),
        );
    }

    fn run_lp_case(&mut self, n: usize, method: &str) {
        let m = (n / 2).max(2);
        let problem = build_resource_lp(n, m);
        let native_t0 = Instant::now();
        let native = solve_lp_internal(
            &problem,
            &InternalSimplexOptions {
                max_iter: Some(20_000),
                tol: Some(1e-8),
            },
        );
        let native_ms = native_t0.elapsed().as_secs_f64() * 1000.0;
        let external = solve_lp_external(
            &problem,
            &ExternalSolverOptions {
                method: Some(method.to_string()),
                ..Default::default()
            },
        );
        let case = format!("LP n={n} method={method}");
        self.check(
            format!("{case} statuses optimal"),
            native.status == LPStatus::Optimal && external.status == LPStatus::Optimal,
            format!(
                "native={} external={} solver={}",
                native.status.as_str(),
                external.status.as_str(),
                external.solver
            ),
        );
        self.close(
            format!("{case} objective"),
            native.objective,
            external.objective,
            1e-7,
        );
        self.rows.push(ScaleRow {
            family: "lp-resource".to_string(),
            size: n,
            constraints: m,
            native_solver: native.solver,
            external_solver: external.solver,
            native_status: native.status.as_str().to_string(),
            external_status: external.status.as_str().to_string(),
            native_objective: native.objective,
            external_objective: external.objective,
            objective_abs_diff: (native.objective - external.objective).abs(),
            native_ms,
            external_ms: external.elapsed_ms,
            native_nodes: None,
            native_lp_solves: None,
        });
    }

    fn run_mip_case(&mut self, n: usize, solver: &str) {
        let problem = build_scale_knapsack(n);
        let native_t0 = Instant::now();
        let native = solve_ipmip_with_des(
            problem.clone(),
            IPMIPSolveOptions {
                lp_algorithm: Some(LpRelaxationAlgorithm::Concrete(
                    ConcreteLpRelaxationAlgorithm::InternalSimplex,
                )),
                max_cut_rounds: Some(1),
                max_nodes: Some(25_000),
                max_ticks: Some(250_000),
                ..Default::default()
            },
        );
        let native_ms = native_t0.elapsed().as_secs_f64() * 1000.0;
        let external_t0 = Instant::now();
        let external = self.run_linear_cli_reference("mip", solver, &mip_json(&problem));
        let external_ms = external_t0.elapsed().as_secs_f64() * 1000.0;
        if external.status == "unavailable" {
            println!(
                "  SKIP  MIP knapsack n={n} solver={solver}: {}",
                external.message
            );
            return;
        }
        let external_objective = external.objective.unwrap_or(f64::NAN);
        let case = format!("MIP knapsack n={n} solver={solver}");
        self.check(
            format!("{case} statuses optimal"),
            native.status == IPMIPStatus::Optimal && external.status == "optimal",
            format!(
                "native={} external={} solver={} message={}",
                native.status.as_str(),
                external.status,
                external.solver,
                external.message
            ),
        );
        self.close(
            format!("{case} objective"),
            native.z,
            external_objective,
            1e-7,
        );
        self.rows.push(ScaleRow {
            family: "mip-binary-knapsack".to_string(),
            size: n,
            constraints: problem.a.len(),
            native_solver: native.solver_kind.to_string(),
            external_solver: external.solver,
            native_status: native.status.as_str().to_string(),
            external_status: external.status,
            native_objective: native.z,
            external_objective,
            objective_abs_diff: (native.z - external_objective).abs(),
            native_ms,
            external_ms,
            native_nodes: Some(native.nodes_explored),
            native_lp_solves: Some(native.lp_solves),
        });
    }

    fn run_linear_cli_reference(
        &self,
        kind: &str,
        solver: &str,
        stdin_json: &str,
    ) -> LinearCliReference {
        use std::io::Write;

        let python = std::env::var("PYTHON")
            .or_else(|_| std::env::var("PYTHON_BIN"))
            .unwrap_or_else(|_| "python3".to_string());
        let script = self.root.join("scripts").join("linear_cli_reference.py");
        let mut child = Command::new(&python)
            .arg(&script)
            .arg("--kind")
            .arg(kind)
            .arg("--solver")
            .arg(solver)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap_or_else(|e| panic!("failed to start linear_cli_reference.py: {e}"));
        if let Some(mut stdin) = child.stdin.take() {
            stdin
                .write_all(stdin_json.as_bytes())
                .expect("write linear CLI stdin");
        }
        let output = child
            .wait_with_output()
            .expect("wait for linear CLI reference");
        if !output.status.success() {
            panic!(
                "linear_cli_reference.py failed: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        serde_json::from_slice(&output.stdout).expect("parse linear CLI reference JSON")
    }

    fn write_report(&self, lp_sizes: &[usize], mip_sizes: &[usize]) {
        let out_dir = self
            .root
            .join("out")
            .join("external")
            .join("optimization-scale");
        std::fs::create_dir_all(&out_dir).expect("create optimization-scale output dir");
        let report = ScaleReport {
            generated_at_unix_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system time before epoch")
                .as_millis(),
            lp_sizes: lp_sizes.to_vec(),
            mip_sizes: mip_sizes.to_vec(),
            rows: self.rows.clone(),
        };
        let path = out_dir.join("scale-report.json");
        let json = serde_json::to_string_pretty(&report).expect("serialize scale report");
        std::fs::write(&path, format!("{json}\n")).expect("write scale report");
        println!("\nWrote scale report: {}", path.display());
    }

    fn finish(self) {
        let failed: Vec<_> = self.checks.iter().filter(|c| !c.passed).collect();
        println!(
            "\nvalidate-optimization-scale: {}/{} checks passed.",
            self.checks.len() - failed.len(),
            self.checks.len()
        );
        if !failed.is_empty() {
            eprintln!("FAILED:");
            for row in failed {
                eprintln!("  - {}: {}", row.name, row.detail);
            }
            std::process::exit(1);
        }
    }
}

fn build_resource_lp(n: usize, m: usize) -> LPProblem {
    let ub_value = 6.0;
    let mut a_ub = Vec::with_capacity(m);
    let mut b_ub = Vec::with_capacity(m);
    for i in 0..m {
        let mut row = Vec::with_capacity(n);
        for j in 0..n {
            let coef = if (i + 2 * j) % 5 == 0 {
                0.0
            } else {
                1.0 + ((7 * i + 11 * j + 3) % 13) as f64 / 3.0
            };
            row.push(coef);
        }
        let full_use: f64 = row.iter().map(|a| a * ub_value).sum();
        b_ub.push(full_use * (0.28 + 0.02 * (i % 4) as f64));
        a_ub.push(row);
    }
    LPProblem {
        sense: Sense::Max,
        c: (0..n)
            .map(|j| 1.0 + ((17 * j + 5) % 19) as f64 / 4.0)
            .collect(),
        a_ub: Some(a_ub),
        b_ub: Some(b_ub),
        a_eq: None,
        b_eq: None,
        lb: Some(vec![Some(0.0); n]),
        ub: Some(vec![Some(ub_value); n]),
        var_names: Some((0..n).map(|j| format!("x{j}")).collect()),
        con_names: Some((0..m).map(|i| format!("resource_{i}")).collect()),
    }
}

fn build_scale_knapsack(n: usize) -> IPMIPProblem {
    let values: Vec<f64> = (0..n)
        .map(|j| 8.0 + ((37 * j + 13) % 41) as f64 + (j % 3) as f64 * 0.25)
        .collect();
    let weights: Vec<f64> = (0..n).map(|j| 3.0 + ((19 * j + 7) % 23) as f64).collect();
    let capacity = weights.iter().sum::<f64>() * 0.42;
    build_binary_knapsack_ip(values, weights, capacity)
}

fn mip_json(problem: &IPMIPProblem) -> String {
    serde_json::json!({
        "sense": problem.sense.as_str(),
        "c": &problem.c,
        "a": &problem.a,
        "b": &problem.b,
        "integer_vars": &problem.integer_vars,
        "ub": &problem.ub,
        "var_names": &problem.var_names,
        "con_names": &problem.con_names,
    })
    .to_string()
}

fn parse_size_list(env_name: &str, defaults: &[usize]) -> Vec<usize> {
    let Ok(raw) = std::env::var(env_name) else {
        return defaults.to_vec();
    };
    let values: Vec<usize> = raw
        .split(',')
        .filter_map(|part| part.trim().parse::<usize>().ok())
        .filter(|&n| n > 0)
        .collect();
    if values.is_empty() {
        defaults.to_vec()
    } else {
        values
    }
}

fn parse_solver_list(env_name: &str, defaults: &[&str]) -> Vec<String> {
    let Ok(raw) = std::env::var(env_name) else {
        return defaults.iter().map(|s| (*s).to_string()).collect();
    };
    let values: Vec<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    if values.is_empty() {
        defaults.iter().map(|s| (*s).to_string()).collect()
    } else {
        values
    }
}

pub fn run() {
    println!("Optimization scale envelope: native solvers vs external engines");
    println!("===============================================================");

    let lp_sizes = parse_size_list("SCALE_LP_SIZES", &[8, 16, 24]);
    let mip_sizes = parse_size_list("SCALE_MIP_SIZES", &[8, 12, 16]);
    let lp_methods = parse_solver_list("SCALE_LP_METHODS", &["highs", "glop"]);
    let mip_solvers = parse_solver_list("SCALE_MIP_SOLVERS", &["highs", "cbc"]);

    let mut driver = Driver::new();

    println!("\n-- LP resource family --");
    for &n in &lp_sizes {
        for method in &lp_methods {
            driver.run_lp_case(n, method);
        }
    }

    println!("\n-- MIP binary-knapsack family --");
    for &n in &mip_sizes {
        for solver in &mip_solvers {
            driver.run_mip_case(n, solver);
        }
    }

    driver.write_report(&lp_sizes, &mip_sizes);
    driver.finish();
}
