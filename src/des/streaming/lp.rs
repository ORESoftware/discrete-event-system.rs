//! Streaming linear program — the live-edit case.
//!
//! Wraps [`IncrementalLP`](crate::des::general::incremental_lp::IncrementalLP),
//! which keeps a warm tableau across edits. A stream of commands mutates the
//! live problem — add/remove a variable, add/remove a constraint, change the
//! objective weights — and after each edit the solver re-pivots to optimality
//! (warm-started, not from scratch) and emits the updated solution. This is the
//! canonical "LP is a solver fed by a stream, not a discrete-event simulation"
//! shape.
//!
//! Standard form only (inherited from `IncrementalLP`): `A·x ≤ b`, `x ≥ 0`,
//! `b ≥ 0`. No equalities / free variables / Phase-I.
//!
//! Additive: this module only *calls* `IncrementalLP`'s public API.

use serde_json::{json, Value};

use crate::des::general::incremental_lp::{
    IncrementalLP, IncrementalLPInit, PivotEvent, PivotMode, Sense, SolverStatus,
};

use super::{
    bool_at, error_frame, f64_at, op_of, str_at, usize_at, vec_f64, vec_str, vec_vec_f64,
    SolverKind, StreamContract, StreamEvent, StreamOp, StreamingModel,
};

/// A live, warm-started LP driven by a JSONL command stream.
pub struct StreamingLp {
    lp: Option<IncrementalLP>,
    /// Re-solve to optimality automatically after each problem edit.
    auto_solve: bool,
    /// Safety cap on pivots per re-solve.
    max_pivots: usize,
    /// Emit one `pivot` frame per simplex pivot (off by default — solution-only).
    emit_pivots: bool,
}

impl Default for StreamingLp {
    fn default() -> Self {
        StreamingLp {
            lp: None,
            auto_solve: true,
            max_pivots: 10_000,
            emit_pivots: false,
        }
    }
}

impl StreamingLp {
    pub fn new() -> Self {
        Self::default()
    }

    fn status_str(status: SolverStatus) -> &'static str {
        match status {
            SolverStatus::Primal => "primal",
            SolverStatus::Dual => "dual",
            SolverStatus::Optimal => "optimal",
            SolverStatus::Infeasible => "infeasible",
            SolverStatus::Unbounded => "unbounded",
        }
    }

    fn pivot_mode_str(mode: PivotMode) -> &'static str {
        match mode {
            PivotMode::Primal => "primal",
            PivotMode::Dual => "dual",
            PivotMode::Optimal => "optimal",
            PivotMode::Infeasible => "infeasible",
            PivotMode::Unbounded => "unbounded",
            PivotMode::Idle => "idle",
        }
    }

    fn pivot_frame(event: &PivotEvent) -> Value {
        json!({
            "event": "pivot",
            "tick": event.tick,
            "mode": Self::pivot_mode_str(event.mode),
            "entering": event.entering,
            "leaving": event.leaving,
            "enteringName": event.entering_name,
            "leavingName": event.leaving_name,
        })
    }

    fn solution_frame(lp: &IncrementalLP) -> Value {
        let x = lp.get_x();
        let vars: serde_json::Map<String, Value> = lp
            .var_names
            .iter()
            .zip(x.iter())
            .map(|(name, value)| (name.clone(), json!(value)))
            .collect();
        json!({
            "event": "solution",
            "status": Self::status_str(lp.status),
            "objective": lp.get_z(),
            "x": x,
            "vars": Value::Object(vars),
            "tick": lp.tick,
            "numStruct": lp.num_struct,
            "numConstraints": lp.con_names.len(),
        })
    }

    /// Pivot to optimality (warm-started) and emit pivot frames (optional) plus
    /// the resulting solution frame.
    fn resolve(&mut self) -> Vec<Value> {
        let mut frames = Vec::new();
        let max_pivots = self.max_pivots;
        let emit_pivots = self.emit_pivots;
        if let Some(lp) = self.lp.as_mut() {
            let pivots = lp.solve_to_optimum(max_pivots);
            if emit_pivots {
                for pivot in &pivots {
                    frames.push(Self::pivot_frame(pivot));
                }
            }
            frames.push(Self::solution_frame(lp));
        }
        frames
    }

    fn maybe_resolve(&mut self) -> Vec<Value> {
        if self.auto_solve {
            self.resolve()
        } else {
            Vec::new()
        }
    }

    fn n_vars(&self) -> usize {
        self.lp.as_ref().map(|lp| lp.num_struct).unwrap_or(0)
    }

    fn n_cons(&self) -> usize {
        self.lp.as_ref().map(|lp| lp.con_names.len()).unwrap_or(0)
    }

    fn init(&mut self, command: &Value) -> Vec<Value> {
        let sense = match str_at(command, "sense").as_deref() {
            Some("min") | Some("Min") => Sense::Min,
            _ => Sense::Max,
        };
        let c = vec_f64(command, "c").unwrap_or_default();
        let a = vec_vec_f64(command, "a").unwrap_or_default();
        let b = vec_f64(command, "b").unwrap_or_default();

        if c.is_empty() {
            return vec![error_frame("`c` (objective) must be a non-empty array")];
        }
        if a.len() != b.len() {
            return vec![error_frame(format!(
                "`a` has {} rows but `b` has {} entries",
                a.len(),
                b.len()
            ))];
        }
        if b.iter().any(|&v| v < 0.0) {
            return vec![error_frame(
                "warm-start requires non-negative `b` (standard form A·x ≤ b, x ≥ 0)",
            )];
        }
        if a.iter().any(|row| row.len() != c.len()) {
            return vec![error_frame("each row of `a` must have length == len(c)")];
        }

        self.auto_solve = bool_at(command, "autoSolve", true);
        self.emit_pivots = bool_at(command, "emitPivots", false);
        if let Some(cap) = usize_at(command, "maxPivots") {
            if cap > 0 {
                self.max_pivots = cap;
            }
        }

        let init = IncrementalLPInit {
            sense,
            c,
            a,
            b,
            var_names: vec_str(command, "varNames"),
            con_names: vec_str(command, "conNames"),
        };
        self.lp = Some(IncrementalLP::new(init));

        let mut frames = vec![json!({
            "event": "initialized",
            "numStruct": self.n_vars(),
            "numConstraints": self.n_cons(),
        })];
        frames.extend(self.maybe_resolve());
        frames
    }
}

impl StreamingModel for StreamingLp {
    fn kind(&self) -> SolverKind {
        SolverKind::IterativeSolver
    }

    fn contract(&self) -> StreamContract {
        StreamContract::new(
            "streaming-lp",
            SolverKind::IterativeSolver,
            "Warm-started incremental linear program. Edits to the live problem \
             (add/remove variable, add/remove constraint, change objective) arrive \
             as a stream; the simplex re-pivots from the warm tableau and emits the \
             updated solution. Standard form A·x ≤ b, x ≥ 0, b ≥ 0.",
            vec![
                StreamOp::new(
                    "init",
                    "Define the initial LP. Optional: varNames, conNames, sense \
                     (max|min), autoSolve, emitPivots, maxPivots.",
                    json!({"op":"init","sense":"max","c":[3,2],"a":[[1,1],[1,3]],"b":[4,6]}),
                ),
                StreamOp::new(
                    "add_constraint",
                    "Add a ≤ row: coefs·x ≤ rhs (coefs length == #vars).",
                    json!({"op":"add_constraint","coefs":[1,0],"rhs":2,"name":"x1cap"}),
                ),
                StreamOp::new(
                    "remove_constraint",
                    "Remove the constraint at `index`.",
                    json!({"op":"remove_constraint","index":0}),
                ),
                StreamOp::new(
                    "add_variable",
                    "Add a structural variable with objective coef `c` and a column \
                     of length #constraints.",
                    json!({"op":"add_variable","c":5,"column":[1,2],"name":"x3"}),
                ),
                StreamOp::new(
                    "remove_variable",
                    "Remove the structural variable at `index`.",
                    json!({"op":"remove_variable","index":1}),
                ),
                StreamOp::new(
                    "change_objective",
                    "Replace the objective weight vector `c` (length == #vars).",
                    json!({"op":"change_objective","c":[1,5]}),
                ),
                StreamOp::new(
                    "step",
                    "Perform a single simplex pivot.",
                    json!({"op":"step"}),
                ),
                StreamOp::new(
                    "solve",
                    "Pivot to optimality from the current tableau.",
                    json!({"op":"solve"}),
                ),
                StreamOp::new(
                    "solution",
                    "Emit the current solution without pivoting.",
                    json!({"op":"solution"}),
                ),
            ],
            vec![
                StreamEvent::new("initialized", "Problem created; reports sizes."),
                StreamEvent::new("applied", "An edit was applied to the live problem."),
                StreamEvent::new("pivot", "One simplex pivot (when emitPivots / step)."),
                StreamEvent::new(
                    "solution",
                    "Current solution: status, objective, x, vars map.",
                ),
                StreamEvent::new("error", "A command could not be applied."),
            ],
        )
    }

    fn apply(&mut self, command: &Value) -> Vec<Value> {
        let op = op_of(command);
        if op == "init" {
            return self.init(command);
        }
        if self.lp.is_none() {
            return vec![error_frame(
                "no LP initialized; send {\"op\":\"init\", ...} first",
            )];
        }

        match op {
            "add_constraint" => {
                let coefs = vec_f64(command, "coefs").unwrap_or_default();
                let rhs = f64_at(command, "rhs", 0.0);
                if coefs.len() != self.n_vars() {
                    return vec![error_frame(
                        "`coefs` length must equal the number of variables",
                    )];
                }
                let name = str_at(command, "name");
                self.lp
                    .as_mut()
                    .unwrap()
                    .apply_add_constraint(&coefs, rhs, name);
                let mut frames = vec![json!({"event":"applied","op":"add_constraint","rhs":rhs})];
                frames.extend(self.maybe_resolve());
                frames
            }
            "remove_constraint" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_cons() {
                    return vec![error_frame("`index` out of range for constraints")];
                }
                self.lp.as_mut().unwrap().apply_remove_constraint(index);
                let mut frames =
                    vec![json!({"event":"applied","op":"remove_constraint","index":index})];
                frames.extend(self.maybe_resolve());
                frames
            }
            "add_variable" => {
                let column = vec_f64(command, "column").unwrap_or_default();
                let c_new = f64_at(command, "c", 0.0);
                if column.len() != self.n_cons() {
                    return vec![error_frame(
                        "`column` length must equal the number of constraints",
                    )];
                }
                let name = str_at(command, "name");
                self.lp
                    .as_mut()
                    .unwrap()
                    .apply_add_variable(&column, c_new, name);
                let mut frames = vec![json!({"event":"applied","op":"add_variable","c":c_new})];
                frames.extend(self.maybe_resolve());
                frames
            }
            "remove_variable" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_vars() {
                    return vec![error_frame("`index` out of range for variables")];
                }
                self.lp.as_mut().unwrap().apply_remove_variable(index);
                let mut frames =
                    vec![json!({"event":"applied","op":"remove_variable","index":index})];
                frames.extend(self.maybe_resolve());
                frames
            }
            "change_objective" => {
                let new_c = vec_f64(command, "c").unwrap_or_default();
                if new_c.len() != self.n_vars() {
                    return vec![error_frame("`c` length must equal the number of variables")];
                }
                self.lp.as_mut().unwrap().apply_change_objective(&new_c);
                let mut frames = vec![json!({"event":"applied","op":"change_objective"})];
                frames.extend(self.maybe_resolve());
                frames
            }
            "step" => {
                let pivot = self.lp.as_mut().unwrap().step();
                let mut frames = vec![Self::pivot_frame(&pivot)];
                frames.push(Self::solution_frame(self.lp.as_ref().unwrap()));
                frames
            }
            "solve" => self.resolve(),
            "solution" | "snapshot" => {
                vec![Self::solution_frame(self.lp.as_ref().unwrap())]
            }
            other => vec![error_frame(format!(
                "unknown op `{other}` for streaming-lp"
            ))],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::streaming::drive;

    fn last_solution(frames: &[Value]) -> &Value {
        frames
            .iter()
            .rev()
            .find(|f| f["event"] == json!("solution"))
            .expect("a solution frame")
    }

    #[test]
    fn solves_initial_lp() {
        // max 3x1 + 2x2  s.t. x1 + x2 <= 4, x1 + 3x2 <= 6, x >= 0.
        // Optimum at (4, 0) -> 12.
        let mut m = StreamingLp::new();
        let frames = drive(
            &mut m,
            &[json!({"op":"init","sense":"max","c":[3,2],"a":[[1,1],[1,3]],"b":[4,6]})],
        );
        let sol = last_solution(&frames);
        assert_eq!(sol["status"], json!("optimal"));
        assert!((sol["objective"].as_f64().unwrap() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn streamed_constraint_edit_changes_optimum() {
        let mut m = StreamingLp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","sense":"max","c":[3,2],"a":[[1,1],[1,3]],"b":[4,6]}),
                // Stream a new cap x1 <= 2; optimum must drop below 12.
                json!({"op":"add_constraint","coefs":[1,0],"rhs":2}),
            ],
        );
        let sol = last_solution(&frames);
        assert_eq!(sol["status"], json!("optimal"));
        let obj = sol["objective"].as_f64().unwrap();
        // x1=2, x2=4/3 -> 3*2 + 2*(4/3) = 8.6667.
        assert!((obj - 8.66667).abs() < 1e-3, "objective was {obj}");
    }

    #[test]
    fn reweighting_objective_restreams_solution() {
        let mut m = StreamingLp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","sense":"max","c":[3,2],"a":[[1,1],[1,3]],"b":[4,6]}),
                // Now strongly prefer x2.
                json!({"op":"change_objective","c":[1,5]}),
            ],
        );
        let sol = last_solution(&frames);
        assert_eq!(sol["status"], json!("optimal"));
        // With c = [1, 5] the optimum shifts toward x2 = 2 (x1+3x2<=6), obj = 1*0+5*2 = 10.
        let obj = sol["objective"].as_f64().unwrap();
        assert!((obj - 10.0).abs() < 1e-6, "objective was {obj}");
    }

    #[test]
    fn edit_before_init_errors() {
        let mut m = StreamingLp::new();
        let frames = drive(
            &mut m,
            &[json!({"op":"add_constraint","coefs":[1],"rhs":1})],
        );
        assert_eq!(frames[0]["event"], json!("error"));
    }
}
