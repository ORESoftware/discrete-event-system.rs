//! Port of `src/des/runners/validate-incremental-lp.ts`.
//!
//! Validates the warm-startable incremental LP against the static
//! `solve_lp_internal` solver after every modification step (add/remove
//! constraint, change objective, add/remove variable). Top-level driver → [`run`].
//!
//! PORT NOTES (stubbed cross-module deps):
//!   * `crate::des::general::incremental_lp::{IncrementalLp, LpEvent}` and
//!     `crate::des::general::lp::{solve_lp_internal, LpProblem}`.
//!   * The stub `IncrementalLp` only tracks the LP *shape* (so the static and
//!     incremental variable counts stay aligned); both solvers return the
//!     all-zero point until the real simplex is wired.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

// =============================================================================
// Stubbed LP layer.
// =============================================================================

#[derive(Clone, Debug, Default)]
struct LpProblem {
    sense: &'static str,
    c: Vec<f64>,
    a_ub: Vec<Vec<f64>>,
    b_ub: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct LpResult {
    status: String,
    x: Vec<f64>,
    objective: f64,
}

fn solve_lp_internal(lp: &LpProblem) -> LpResult {
    LpResult {
        status: "optimal".to_string(),
        x: vec![0.0; lp.c.len()],
        objective: 0.0,
    }
}

/// `interface State` from the validator.
#[derive(Clone, Debug)]
struct State {
    sense: &'static str,
    c: Vec<f64>,
    a: Vec<Vec<f64>>,
    b: Vec<f64>,
}

/// `type LPEvent` discriminated union → enum (the `tick` field is metadata,
/// dropped here).
#[derive(Clone, Debug)]
enum LpEvent {
    AddConstraint { coefs: Vec<f64>, rhs: f64 },
    RemoveConstraint { index: usize },
    ChangeObjective { new_c: Vec<f64> },
    AddVariable { column: Vec<f64>, c_new: f64 },
    RemoveVariable { struct_index: usize },
}

/// PORT NOTE: `incremental_lp::IncrementalLp`. The stub mirrors the LP shape so
/// the variable count stays in sync with the static reference; solving is a no-op
/// (both sides return zeros until the real simplex is wired).
struct IncrementalLp {
    state: State,
}

impl IncrementalLp {
    fn new(init: State) -> Self {
        IncrementalLp { state: init }
    }

    fn solve_to_optimum(&mut self) {
        // PORT NOTE: dual/primal warm-started simplex restart.
    }

    fn apply_event(&mut self, ev: LpEvent) {
        match ev {
            LpEvent::AddConstraint { coefs, rhs } => {
                self.state.a.push(coefs);
                self.state.b.push(rhs);
            }
            LpEvent::RemoveConstraint { index } => {
                self.state.a.remove(index);
                self.state.b.remove(index);
            }
            LpEvent::ChangeObjective { new_c } => {
                self.state.c = new_c;
            }
            LpEvent::AddVariable { column, c_new } => {
                for (i, row) in self.state.a.iter_mut().enumerate() {
                    row.push(column[i]);
                }
                self.state.c.push(c_new);
            }
            LpEvent::RemoveVariable { struct_index } => {
                for row in self.state.a.iter_mut() {
                    row.remove(struct_index);
                }
                self.state.c.remove(struct_index);
            }
        }
    }

    fn get_x(&self) -> Vec<f64> {
        vec![0.0; self.state.c.len()]
    }

    fn get_z(&self) -> f64 {
        0.0
    }
}

// =============================================================================
// Driver helpers.
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
        let tail = if detail.is_empty() {
            String::new()
        } else {
            format!("  — {}", detail)
        };
        println!("{}  {}{}", if ok { "  PASS" } else { "  FAIL" }, label, tail);
        if ok {
            self.pass += 1;
        } else {
            self.fail += 1;
        }
    }
    fn close(&mut self, label: &str, a: f64, b: f64) {
        self.check(label, (a - b).abs() <= 1e-7, &format!("|{} − {}| = {:.2e}", a, b, (a - b).abs()));
    }
    fn array_close(&mut self, label: &str, a: &[f64], b: &[f64]) {
        if a.len() != b.len() {
            self.check(label, false, &format!("lengths {} vs {}", a.len(), b.len()));
            return;
        }
        let mut max_d = 0.0_f64;
        for i in 0..a.len() {
            max_d = f64::max(max_d, (a[i] - b[i]).abs());
        }
        self.check(label, max_d <= 1e-7, &format!("max|Δ|={:.2e}", max_d));
    }
}

fn solve_static(s: &State) -> (Vec<f64>, f64, String) {
    let lp = LpProblem {
        sense: s.sense,
        c: s.c.clone(),
        a_ub: s.a.iter().cloned().collect(),
        b_ub: s.b.clone(),
    };
    let sol = solve_lp_internal(&lp);
    (sol.x, sol.objective, sol.status)
}

fn st(sense: &'static str, c: &[f64], a: &[&[f64]], b: &[f64]) -> State {
    State {
        sense,
        c: c.to_vec(),
        a: a.iter().map(|r| r.to_vec()).collect(),
        b: b.to_vec(),
    }
}

/// `validate-incremental-lp.ts` top-level driver.
pub fn run() {
    let mut c = Checker::new();

    // Study 1 — Baseline 2D LP, no modifications.
    println!("\nStudy 1 — Baseline 2D LP, no modifications");
    {
        let init = st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]);
        let mut inc = IncrementalLp::new(init.clone());
        inc.solve_to_optimum();
        let stat = solve_static(&init);
        c.array_close("baseline x matches static", &inc.get_x(), &stat.0);
        c.close("baseline z matches static", inc.get_z(), stat.1);
    }

    // Study 2 — Add constraint after solving.
    println!("\nStudy 2 — Add constraint after solving (dual simplex restart)");
    {
        let init = st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]);
        let mut inc = IncrementalLp::new(init.clone());
        inc.solve_to_optimum();
        inc.apply_event(LpEvent::AddConstraint { coefs: vec![1.0, 0.0], rhs: 30.0 });
        inc.solve_to_optimum();
        let stat = solve_static(&st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0], &[1.0, 0.0]], &[100.0, 90.0, 30.0]));
        c.array_close("post-add-constraint x  matches static", &inc.get_x(), &stat.0);
        c.close("post-add-constraint z  matches static", inc.get_z(), stat.1);
    }

    // Study 3 — Remove a binding constraint.
    println!("\nStudy 3 — Remove a binding constraint");
    {
        let init = st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]);
        let mut inc = IncrementalLp::new(init.clone());
        inc.solve_to_optimum();
        inc.apply_event(LpEvent::RemoveConstraint { index: 0 });
        inc.solve_to_optimum();
        let stat = solve_static(&st("max", &[3.0, 5.0], &[&[1.0, 3.0]], &[90.0]));
        c.array_close("post-remove-constraint x matches static", &inc.get_x(), &stat.0);
        c.close("post-remove-constraint z matches static", inc.get_z(), stat.1);
    }

    // Study 4 — Change objective.
    println!("\nStudy 4 — Change objective (primal simplex restart)");
    {
        let init = st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]);
        let mut inc = IncrementalLp::new(init.clone());
        inc.solve_to_optimum();
        inc.apply_event(LpEvent::ChangeObjective { new_c: vec![5.0, 3.0] });
        inc.solve_to_optimum();
        let stat = solve_static(&st("max", &[5.0, 3.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]));
        c.array_close("post-change-objective x matches static", &inc.get_x(), &stat.0);
        c.close("post-change-objective z matches static", inc.get_z(), stat.1);
    }

    // Study 5 — Add a variable mid-run.
    println!("\nStudy 5 — Add a variable mid-run");
    {
        let init = st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]);
        let mut inc = IncrementalLp::new(init.clone());
        inc.solve_to_optimum();
        inc.apply_event(LpEvent::AddVariable { column: vec![1.0, 1.0], c_new: 7.0 });
        inc.solve_to_optimum();
        let stat = solve_static(&st("max", &[3.0, 5.0, 7.0], &[&[2.0, 1.0, 1.0], &[1.0, 3.0, 1.0]], &[100.0, 90.0]));
        c.array_close("post-add-variable x matches static", &inc.get_x(), &stat.0);
        c.close("post-add-variable z matches static", inc.get_z(), stat.1);
    }

    // Study 6 — Remove a variable mid-run.
    println!("\nStudy 6 — Remove a variable mid-run");
    {
        let init = st("max", &[3.0, 5.0, 7.0], &[&[2.0, 1.0, 1.0], &[1.0, 3.0, 1.0]], &[100.0, 90.0]);
        let mut inc = IncrementalLp::new(init.clone());
        inc.solve_to_optimum();
        inc.apply_event(LpEvent::RemoveVariable { struct_index: 2 });
        inc.solve_to_optimum();
        let stat = solve_static(&st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]));
        c.array_close("post-remove-variable x matches static", &inc.get_x(), &stat.0);
        c.close("post-remove-variable z matches static", inc.get_z(), stat.1);
    }

    // Study 7 — Sequence of all 5 modifications.
    println!("\nStudy 7 — Sequence of all 5 modifications, validating each step");
    {
        let mut inc = IncrementalLp::new(st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]));
        let base = st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]);
        inc.solve_to_optimum();
        c.array_close("S7.0 initial x", &inc.get_x(), &solve_static(&base).0);

        inc.apply_event(LpEvent::AddConstraint { coefs: vec![1.0, 0.0], rhs: 30.0 });
        inc.solve_to_optimum();
        let mut s = solve_static(&st("max", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0], &[1.0, 0.0]], &[100.0, 90.0, 30.0]));
        c.array_close("S7.a after add-constraint x", &inc.get_x(), &s.0);
        c.close("S7.a z", inc.get_z(), s.1);

        inc.apply_event(LpEvent::ChangeObjective { new_c: vec![5.0, 3.0] });
        inc.solve_to_optimum();
        s = solve_static(&st("max", &[5.0, 3.0], &[&[2.0, 1.0], &[1.0, 3.0], &[1.0, 0.0]], &[100.0, 90.0, 30.0]));
        c.array_close("S7.b after change-objective x", &inc.get_x(), &s.0);
        c.close("S7.b z", inc.get_z(), s.1);

        inc.apply_event(LpEvent::RemoveConstraint { index: 0 });
        inc.solve_to_optimum();
        s = solve_static(&st("max", &[5.0, 3.0], &[&[1.0, 3.0], &[1.0, 0.0]], &[90.0, 30.0]));
        c.array_close("S7.c after remove-constraint x", &inc.get_x(), &s.0);
        c.close("S7.c z", inc.get_z(), s.1);

        inc.apply_event(LpEvent::AddVariable { column: vec![1.0, 0.0], c_new: 4.0 });
        inc.solve_to_optimum();
        s = solve_static(&st("max", &[5.0, 3.0, 4.0], &[&[1.0, 3.0, 1.0], &[1.0, 0.0, 0.0]], &[90.0, 30.0]));
        c.array_close("S7.d after add-variable x", &inc.get_x(), &s.0);
        c.close("S7.d z", inc.get_z(), s.1);

        inc.apply_event(LpEvent::RemoveVariable { struct_index: 1 });
        inc.solve_to_optimum();
        s = solve_static(&st("max", &[5.0, 4.0], &[&[1.0, 1.0], &[1.0, 0.0]], &[90.0, 30.0]));
        c.array_close("S7.e after remove-variable x", &inc.get_x(), &s.0);
        c.close("S7.e z", inc.get_z(), s.1);
    }

    // Study 8 — Random 3-variable LP, randomised modification stream.
    println!("\nStudy 8 — Random 3-variable LP, randomised modification stream");
    {
        // mulberry32 (`rng(seed)`).
        let mut s_state: u32 = 1234;
        let mut rng = move || {
            s_state = s_state.wrapping_add(0x6D2B_79F5);
            let mut t = (s_state ^ (s_state >> 15)).wrapping_mul(1 | s_state);
            t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
            ((t ^ (t >> 14)) as f64) / 4294967296.0
        };
        let base_n = 3usize;
        let base_m = 3usize;
        let c0: Vec<f64> = (0..base_n).map(|_| 1.0 + (rng() * 9.0).floor()).collect();
        let a0: Vec<Vec<f64>> = (0..base_m)
            .map(|_| (0..base_n).map(|_| 1.0 + (rng() * 5.0).floor()).collect())
            .collect();
        let b0: Vec<f64> = (0..base_m).map(|_| 30.0 + (rng() * 50.0).floor()).collect();
        let mut state = State { sense: "max", c: c0.clone(), a: a0.clone(), b: b0.clone() };
        let mut inc = IncrementalLp::new(State { sense: "max", c: c0.clone(), a: a0.clone(), b: b0.clone() });
        inc.solve_to_optimum();
        let mut sstat = solve_static(&state);
        c.array_close("S8.0 initial x matches static", &inc.get_x(), &sstat.0);

        // 1: add a constraint x1 + x2 + x3 ≤ 50.
        state.a.push(vec![1.0, 1.0, 1.0]);
        state.b.push(50.0);
        inc.apply_event(LpEvent::AddConstraint { coefs: vec![1.0, 1.0, 1.0], rhs: 50.0 });
        inc.solve_to_optimum();
        sstat = solve_static(&state);
        c.array_close("S8.1 after add x1+x2+x3≤50", &inc.get_x(), &sstat.0);
        c.close("S8.1 z", inc.get_z(), sstat.1);

        // 2: change objective.
        state.c = vec![10.0, 7.0, 4.0];
        inc.apply_event(LpEvent::ChangeObjective { new_c: state.c.clone() });
        inc.solve_to_optimum();
        sstat = solve_static(&state);
        c.array_close("S8.2 after change obj", &inc.get_x(), &sstat.0);
        c.close("S8.2 z", inc.get_z(), sstat.1);

        // 3: remove constraint 1.
        state.a.remove(1);
        state.b.remove(1);
        inc.apply_event(LpEvent::RemoveConstraint { index: 1 });
        inc.solve_to_optimum();
        sstat = solve_static(&state);
        c.array_close("S8.3 after remove constraint 1", &inc.get_x(), &sstat.0);
        c.close("S8.3 z", inc.get_z(), sstat.1);

        // 4: add another constraint.
        state.a.push(vec![2.0, 0.0, 1.0]);
        state.b.push(40.0);
        inc.apply_event(LpEvent::AddConstraint { coefs: vec![2.0, 0.0, 1.0], rhs: 40.0 });
        inc.solve_to_optimum();
        sstat = solve_static(&state);
        c.array_close("S8.4 after add constraint", &inc.get_x(), &sstat.0);
        c.close("S8.4 z", inc.get_z(), sstat.1);

        // 5: add a 4th variable with column [1, 1, 1] and c = 6.
        for row in state.a.iter_mut() {
            row.push(1.0);
        }
        state.c.push(6.0);
        let column: Vec<f64> = state.a.iter().map(|r| r[r.len() - 1]).collect();
        inc.apply_event(LpEvent::AddVariable { column, c_new: 6.0 });
        inc.solve_to_optimum();
        sstat = solve_static(&state);
        c.array_close("S8.5 after add variable", &inc.get_x(), &sstat.0);
        c.close("S8.5 z", inc.get_z(), sstat.1);

        // 6: change obj again.
        state.c = vec![3.0, 12.0, 5.0, 8.0];
        inc.apply_event(LpEvent::ChangeObjective { new_c: state.c.clone() });
        inc.solve_to_optimum();
        sstat = solve_static(&state);
        c.array_close("S8.6 after change obj #2", &inc.get_x(), &sstat.0);
        c.close("S8.6 z", inc.get_z(), sstat.1);

        // 7: remove variable 0.
        for row in state.a.iter_mut() {
            row.remove(0);
        }
        state.c.remove(0);
        inc.apply_event(LpEvent::RemoveVariable { struct_index: 0 });
        inc.solve_to_optimum();
        sstat = solve_static(&state);
        c.array_close("S8.7 after remove variable 0", &inc.get_x(), &sstat.0);
        c.close("S8.7 z", inc.get_z(), sstat.1);

        // 8: change obj final.
        state.c = (0..state.c.len()).map(|i| (i + 1) as f64).collect();
        inc.apply_event(LpEvent::ChangeObjective { new_c: state.c.clone() });
        inc.solve_to_optimum();
        sstat = solve_static(&state);
        c.array_close("S8.8 final state x", &inc.get_x(), &sstat.0);
        c.close("S8.8 final state z", inc.get_z(), sstat.1);
    }

    // Study 9 — min-LP (sense flip).
    println!("\nStudy 9 — min-LP (sense flip)");
    {
        let init = st("min", &[3.0, 5.0], &[&[2.0, 1.0], &[1.0, 3.0]], &[100.0, 90.0]);
        let mut inc = IncrementalLp::new(init.clone());
        inc.solve_to_optimum();
        c.close("min-LP at origin: z = 0", inc.get_z(), 0.0);
        c.array_close("min-LP at origin: x = 0", &inc.get_x(), &[0.0, 0.0]);
    }

    println!("\n{} checks: {} passed, {} failed", c.pass + c.fail, c.pass, c.fail);
    if c.fail > 0 {
        std::process::exit(1);
    }
}
