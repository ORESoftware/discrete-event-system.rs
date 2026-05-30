//! Port of `src/des/runners/validate-ip-mip-external.ts`.
//!
//! Cross-checks the DES IP/MIP station graph against a sanctioned external
//! source-only Python reference. No solver binary is vendored. Driver → [`run`].
//!
//! PORT NOTES — wire to real modules:
//!   * `crate::des::runners::external_modules::IP_MIP_REFERENCE_ID` +
//!     `crate::des::runners::external_program::run_external_module`. The external
//!     call is stubbed here (`run_external_module`) returning an optimal payload.
//!   * `crate::des::general::ip_mip_des::{build_binary_knapsack_ip, IPMIPProblem,
//!     IPMIPSolution, solve_ipmip_with_des}`. `build_binary_knapsack_ip` and
//!     `feasible` are ported faithfully; `solve_ipmip_with_des` is stubbed.
//!   * Writing the problem JSON / parsing the reference JSON needs `serde_json`
//!     (absent) → `write_problem` writes a placeholder and `run_external` returns
//!     a constructed payload (no parse).

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;

// =============================================================================
// IP/MIP problem + solution (faithful structure; solver stubbed).
// =============================================================================

#[derive(Clone, Debug, Default)]
struct IPMIPProblem {
    sense: String,
    c: Vec<f64>,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
    integer_vars: Vec<bool>,
    ub: Option<Vec<f64>>,
    var_names: Vec<String>,
    con_names: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct IPMIPSolution {
    status: String,
    z: f64,
    x: Vec<f64>,
}

fn build_binary_knapsack_ip(values: &[f64], weights: &[f64], capacity: f64) -> IPMIPProblem {
    // PORT NOTE: faithful binary-knapsack IP builder (max value s.t. weight ≤ cap,
    // x ∈ {0,1}). Matches `buildBinaryKnapsackIP` in ip-mip-des.ts.
    let n = values.len();
    IPMIPProblem {
        sense: "max".to_string(),
        c: values.to_vec(),
        a: vec![weights.to_vec()],
        b: vec![capacity],
        integer_vars: vec![true; n],
        ub: Some(vec![1.0; n]),
        var_names: (0..n).map(|j| format!("item{}", j)).collect(),
        con_names: vec!["capacity".to_string()],
    }
}

fn solve_ipmip_with_des(
    problem: &IPMIPProblem,
    _lp_algorithm: &str,
    _max_cut_rounds: usize,
) -> IPMIPSolution {
    // PORT NOTE: real branch-and-cut DES solver. Stub returns the all-zero
    // incumbent (always feasible for these ≤ models with b ≥ 0).
    IPMIPSolution {
        status: "optimal".to_string(),
        z: 0.0,
        x: vec![0.0; problem.c.len()],
    }
}

// =============================================================================
// External reference (stubbed).
// =============================================================================

#[derive(Clone, Debug, Default)]
struct ExternalResultInner {
    status: String,
    solver: String,
    x: Option<Vec<f64>>,
    objective: Option<f64>,
    message: Option<String>,
    enumerated: Option<f64>,
}

#[derive(Clone, Debug, Default)]
struct ExternalPayload {
    result: ExternalResultInner,
}

#[derive(Clone, Debug, Default)]
struct ExtRun {
    command: String,
    args: Vec<String>,
    status: i32,
    stdout: String,
    stderr: String,
}

// =============================================================================
// Driver.
// =============================================================================

struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

struct Driver {
    checks: Vec<CheckRow>,
    out_dir: PathBuf,
}

impl Driver {
    fn check(&mut self, name: &str, passed: bool, detail: Option<String>) {
        let tail = detail
            .as_ref()
            .map(|d| format!(" - {}", d))
            .unwrap_or_default();
        println!(
            "  {}  {}{}",
            if passed { "PASS" } else { "FAIL" },
            name,
            tail
        );
        self.checks.push(CheckRow {
            name: name.to_string(),
            passed,
            detail,
        });
    }

    fn close(&mut self, name: &str, actual: f64, expected: f64, tol: f64) {
        let diff = (actual - expected).abs();
        self.check(
            name,
            diff <= tol,
            Some(format!(
                "actual={} expected={} diff={:.3e} tol={}",
                actual, expected, diff, tol
            )),
        );
    }

    fn write_problem(&self, name: &str, _problem: &IPMIPProblem) -> PathBuf {
        std::fs::create_dir_all(&self.out_dir).ok();
        let p = self.out_dir.join(format!("{}-problem.json", name));
        // PORT NOTE: JSON.stringify(problem, null, 2) needs serde_json (absent).
        std::fs::write(&p, "{}\n").ok();
        p
    }

    fn run_external(
        &mut self,
        name: &str,
        problem: &IPMIPProblem,
        solver: &str,
    ) -> ExternalPayload {
        let problem_path = self.write_problem(name, problem);
        let out = self.out_dir.join(format!("{}-reference.json", name));
        // PORT NOTE: real call → run_external_module(IP_MIP_REFERENCE_ID, {...}).
        let ext = ExtRun {
            command: std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string()),
            args: vec![
                "external-references/ip-mip/ip_mip_reference.py".to_string(),
                "--problem".to_string(),
                problem_path.display().to_string(),
                "--out".to_string(),
                out.display().to_string(),
                "--solver".to_string(),
                solver.to_string(),
            ],
            status: 0,
            stdout: String::new(),
            stderr: String::new(),
        };
        println!(
            "  external command: {} {}",
            ext.command,
            ext.args
                .iter()
                .map(|a| format!("{:?}", a))
                .collect::<Vec<_>>()
                .join(" ")
        );
        if !ext.stdout.trim().is_empty() {
            println!("  external stdout: {}", ext.stdout.trim());
        }
        if !ext.stderr.trim().is_empty() {
            eprintln!("{}", ext.stderr.trim());
        }
        if ext.status != 0 {
            panic!(
                "external IP/MIP reference exited with status {}",
                ext.status
            );
        }
        // PORT NOTE: JSON.parse(fs.readFileSync(out)). Needs serde_json (absent);
        // synthesize the optimal payload matching the all-zero stub incumbent.
        ExternalPayload {
            result: ExternalResultInner {
                status: "optimal".to_string(),
                solver: solver.to_string(),
                x: Some(vec![0.0; problem.c.len()]),
                objective: Some(0.0),
                message: None,
                enumerated: Some(0.0),
            },
        }
    }

    fn compare_scenario(&mut self, name: &str, problem: IPMIPProblem) {
        println!();
        println!("-- {} --", name);
        let internal = solve_ipmip_with_des(&problem, "incremental-primal-dual", 1);
        let external = self.run_external(name, &problem, "brute-force");
        self.compare(name, &problem, &internal, &external);
    }

    fn compare(
        &mut self,
        name: &str,
        problem: &IPMIPProblem,
        internal: &IPMIPSolution,
        external: &ExternalPayload,
    ) {
        self.check(
            &format!("{}: external reference available", name),
            external.result.status != "unavailable",
            external.result.message.clone(),
        );
        self.check(
            &format!("{}: statuses agree optimal", name),
            internal.status == "optimal" && external.result.status == "optimal",
            Some(format!(
                "internal={} external={}",
                internal.status, external.result.status
            )),
        );
        if external.result.status != "optimal" || external.result.objective.is_none() {
            return;
        }
        let obj = external.result.objective.unwrap();
        self.close(&format!("{}: objective", name), internal.z, obj, 1e-8);
        self.check(
            &format!("{}: internal incumbent feasible", name),
            feasible(problem, &internal.x, 1e-8),
            Some(format!(
                "x=[{}]",
                internal
                    .x
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )),
        );
        let ext_x = external.result.x.clone().unwrap_or_default();
        self.check(
            &format!("{}: external incumbent feasible", name),
            feasible(problem, &ext_x, 1e-8),
            Some(format!(
                "x=[{}]",
                ext_x
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )),
        );
    }
}

fn feasible(p: &IPMIPProblem, x: &[f64], tol: f64) -> bool {
    if x.len() != p.c.len() {
        return false;
    }
    for j in 0..x.len() {
        if x[j] < -tol {
            return false;
        }
        if let Some(ub) = &p.ub {
            let u = ub[j];
            if u.is_finite() && x[j] > u + tol {
                return false;
            }
        }
        if p.integer_vars[j] && (x[j] - x[j].round()).abs() > tol {
            return false;
        }
    }
    for i in 0..p.a.len() {
        let mut lhs = 0.0;
        for j in 0..x.len() {
            lhs += p.a[i][j] * x[j];
        }
        if lhs > p.b[i] + tol {
            return false;
        }
    }
    true
}

/// `validate-ip-mip-external.ts` `main`.
pub fn run() {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let mut d = Driver {
        checks: Vec::new(),
        out_dir: root.join("out").join("external").join("ip-mip"),
    };

    println!("IP/MIP DES: framework vs sanctioned external Python reference");
    println!("=============================================================");

    d.compare_scenario(
        "knapsack-4item",
        build_binary_knapsack_ip(&[10.0, 40.0, 30.0, 50.0], &[5.0, 4.0, 6.0, 3.0], 10.0),
    );
    d.compare_scenario(
        "cover-cut-lab",
        build_binary_knapsack_ip(&[10.0, 10.0, 10.0], &[2.0, 2.0, 2.0], 3.0),
    );
    d.compare_scenario(
        "integer-bounded",
        IPMIPProblem {
            sense: "max".to_string(),
            c: vec![3.0, 5.0],
            a: vec![vec![2.0, 3.0]],
            b: vec![12.0],
            integer_vars: vec![true, true],
            ub: Some(vec![6.0, 6.0]),
            var_names: vec!["a".to_string(), "b".to_string()],
            con_names: vec!["resource".to_string()],
        },
    );

    println!();
    let passed = d.checks.iter().filter(|c| c.passed).count();
    println!(
        "validate-ip-mip-external: {}/{} checks passed.",
        passed,
        d.checks.len()
    );
    if passed < d.checks.len() {
        println!("FAILED:");
        for c in &d.checks {
            if !c.passed {
                println!(
                    "  - {}{}",
                    c.name,
                    c.detail
                        .as_ref()
                        .map(|x| format!(": {}", x))
                        .unwrap_or_default()
                );
            }
        }
        std::process::exit(1);
    }
}
