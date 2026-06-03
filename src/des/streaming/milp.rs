//! Streaming mixed-integer linear program — the build-then-solve case.
//!
//! Wraps [`solve_milp`](crate::des::general::milp_bnb::solve_milp). The problem
//! is assembled and edited by a command stream (add a constraint, add an
//! integer/continuous variable, re-set the objective, flip a variable's
//! integrality); on a `solve` command the branch-and-bound runs and its node
//! trace is streamed out one frame per explored node, followed by the final
//! solution. Unlike the LP streamer there is no warm start — each `solve`
//! re-runs B&B on the current problem.
//!
//! Standard form (inherited from `milp_bnb`): `A·x ≤ b`, `x ≥ 0`, `b ≥ 0`,
//! `x_j ∈ ℤ` for the flagged columns.
//!
//! Additive: only the public `solve_milp` API is called. `solve_milp` validates
//! its input and may panic on a malformed problem, so each solve is run under
//! `catch_unwind` and reported as an `error` frame instead.

use std::panic::{catch_unwind, AssertUnwindSafe};

use serde_json::{json, Value};

use crate::des::general::milp_bnb::{
    solve_milp, BranchType, LpStatus, MILPProblem, MILPSolution, MILPSolveOptions, MILPStatus,
    NodeEvent, PrunedReason, Sense,
};

use super::{
    bool_at, error_frame, f64_at, op_of, str_at, usize_at, vec_f64, vec_str, vec_vec_f64,
    ModelStreamKind, SolverKind, StreamContract, StreamEvent, StreamOp, StreamingModel,
};

/// A MILP assembled by a JSONL command stream and solved on demand.
pub struct StreamingMilp {
    problem: MILPProblem,
    initialized: bool,
    /// Emit one `node` frame per explored B&B node on solve.
    emit_nodes: bool,
}

/// Integer-programming streams use the same branch-and-bound engine as MILP.
pub type StreamingMip = StreamingMilp;

/// Pure IP is a MILP with all decision variables marked integer.
pub type StreamingIp = StreamingMilp;

impl Default for StreamingMilp {
    fn default() -> Self {
        StreamingMilp {
            problem: MILPProblem {
                sense: Sense::Max,
                c: Vec::new(),
                a: Vec::new(),
                b: Vec::new(),
                integer_vars: Vec::new(),
                ub: None,
                var_names: None,
                con_names: None,
            },
            initialized: false,
            emit_nodes: false,
        }
    }
}

impl StreamingMilp {
    pub fn new() -> Self {
        Self::default()
    }

    fn n_vars(&self) -> usize {
        self.problem.c.len()
    }

    fn n_cons(&self) -> usize {
        self.problem.a.len()
    }

    fn status_str(status: MILPStatus) -> &'static str {
        match status {
            MILPStatus::Optimal => "optimal",
            MILPStatus::Infeasible => "infeasible",
            MILPStatus::Unbounded => "unbounded",
            MILPStatus::IterLimit => "iter-limit",
            MILPStatus::MaxNodes => "max-nodes",
        }
    }

    fn lp_status_str(status: LpStatus) -> &'static str {
        match status {
            LpStatus::Optimal => "optimal",
            LpStatus::Infeasible => "infeasible",
            LpStatus::Unbounded => "unbounded",
            LpStatus::IterLimit => "iter-limit",
        }
    }

    fn branch_type_str(branch: BranchType) -> &'static str {
        match branch {
            BranchType::Le => "le",
            BranchType::Ge => "ge",
        }
    }

    fn pruned_reason_str(reason: PrunedReason) -> &'static str {
        match reason {
            PrunedReason::Infeasible => "infeasible",
            PrunedReason::Unbounded => "unbounded",
            PrunedReason::Bound => "bound",
            PrunedReason::IntegerFeasible => "integer-feasible",
            PrunedReason::IterLimit => "iter-limit",
        }
    }

    fn node_frame(event: &NodeEvent) -> Value {
        json!({
            "event": "node",
            "nodeId": event.node_id,
            "parentId": event.parent_id,
            "depth": event.depth,
            "branchVar": event.branch_var,
            "branchType": event.branch_type.map(Self::branch_type_str),
            "branchValue": event.branch_value,
            "lpStatus": Self::lp_status_str(event.lp_status),
            "lpZ": event.lp_z,
            "fractional": event.fractional,
            "pruned": event.pruned,
            "prunedReason": event.pruned_reason.map(Self::pruned_reason_str),
            "incumbentUpdated": event.incumbent_updated,
        })
    }

    fn solution_frame(solution: &MILPSolution) -> Value {
        json!({
            "event": "solution",
            "status": Self::status_str(solution.status),
            "z": solution.z,
            "x": solution.x,
            "bestBound": solution.best_bound,
            "gap": solution.gap,
            "nodesExplored": solution.nodes_explored,
            "totalPivots": solution.total_pivots,
        })
    }

    fn parse_bool_vec(command: &Value, key: &str) -> Option<Vec<bool>> {
        command.get(key)?.as_array().map(|items| {
            items
                .iter()
                .map(|x| x.as_bool().unwrap_or(false))
                .collect::<Vec<bool>>()
        })
    }

    fn snapshot_frame(&self) -> Value {
        json!({
            "event": "model",
            "numVars": self.n_vars(),
            "numConstraints": self.n_cons(),
            "integerVars": self.problem.integer_vars,
            "hasUpperBounds": self.problem.ub.is_some(),
        })
    }

    fn ensure_ub(&mut self) -> &mut Vec<f64> {
        let n = self.n_vars();
        self.problem
            .ub
            .get_or_insert_with(|| vec![f64::INFINITY; n])
    }

    fn solve(&mut self, command: &Value) -> Vec<Value> {
        if !self.initialized {
            return vec![error_frame(
                "no MILP initialized; send {\"op\":\"init\", ...} first",
            )];
        }
        let opts = MILPSolveOptions {
            max_nodes: usize_at(command, "maxNodes"),
            lp_max_iters: usize_at(command, "lpMaxIters"),
            int_tol: command.get("intTol").and_then(Value::as_f64),
            branch_rule: None,
            verbose: Some(false),
            initial_incumbent_z: None,
            branch_seed: None,
            lp_pivot_rule: None,
        };
        let problem = self.problem.clone();
        let result = catch_unwind(AssertUnwindSafe(|| solve_milp(&problem, opts)));
        let solution = match result {
            Ok(solution) => solution,
            Err(_) => {
                return vec![error_frame(
                    "solve failed: problem is malformed (check dimensions and b ≥ 0)",
                )]
            }
        };
        let mut frames = Vec::new();
        if self.emit_nodes {
            for event in &solution.trace {
                frames.push(Self::node_frame(event));
            }
        }
        frames.push(json!({
            "event": "trace",
            "nodes": solution.trace.len(),
        }));
        frames.push(Self::solution_frame(&solution));
        frames
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
            return vec![error_frame("`a` rows must match `b` length")];
        }
        if b.iter().any(|&v| v < 0.0) {
            return vec![error_frame(
                "`b` must be non-negative (standard form A·x ≤ b, x ≥ 0)",
            )];
        }
        if a.iter().any(|row| row.len() != c.len()) {
            return vec![error_frame("each row of `a` must have length == len(c)")];
        }

        let mut integer_vars = Self::parse_bool_vec(command, "integerVars").unwrap_or_default();
        integer_vars.resize(c.len(), false);

        self.emit_nodes = bool_at(command, "emitNodes", false);
        let ub = vec_f64(command, "ub");
        if ub.as_ref().is_some_and(|ub| ub.len() != c.len()) {
            return vec![error_frame("`ub` length must equal len(c) when provided")];
        }

        self.problem = MILPProblem {
            sense,
            c,
            a,
            b,
            integer_vars,
            ub,
            var_names: vec_str(command, "varNames"),
            con_names: vec_str(command, "conNames"),
        };
        self.initialized = true;

        vec![json!({
            "event": "initialized",
            "numVars": self.n_vars(),
            "numConstraints": self.n_cons(),
            "integerVars": self.problem.integer_vars,
        })]
    }
}

impl StreamingModel for StreamingMilp {
    fn kind(&self) -> SolverKind {
        SolverKind::IterativeSolver
    }

    fn contract(&self) -> StreamContract {
        StreamContract::new_for_stream_kind(
            "streaming-milp",
            SolverKind::IterativeSolver,
            ModelStreamKind::Mip,
            "Mixed-integer linear program assembled by a command stream and \
             solved on demand by branch-and-bound. On `solve`, the node trace is \
             streamed (when emitNodes), then the final solution. Standard form \
             A·x ≤ b, x ≥ 0, b ≥ 0; flagged columns are integer.",
            vec![
                StreamOp::new(
                    "init",
                    "Define the MILP. integerVars is a bool array aligned with c. \
                     Optional: ub, varNames, conNames, sense (max|min), emitNodes.",
                    json!({"op":"init","sense":"max","c":[1],"a":[[2]],"b":[3],"integerVars":[true]}),
                ),
                StreamOp::new(
                    "add_constraint",
                    "Add a ≤ row: coefs·x ≤ rhs (coefs length == #vars).",
                    json!({"op":"add_constraint","coefs":[1,1],"rhs":3}),
                ),
                StreamOp::new(
                    "set_constraint",
                    "Modify an existing row by index. Optional fields: coefs, rhs, name. Alias: modify_constraint.",
                    json!({"op":"set_constraint","index":0,"coefs":[2,1],"rhs":5}),
                ),
                StreamOp::new(
                    "remove_constraint",
                    "Remove a constraint row by index.",
                    json!({"op":"remove_constraint","index":0}),
                ),
                StreamOp::new(
                    "set_coefficient",
                    "Modify one matrix coefficient A[row][col].",
                    json!({"op":"set_coefficient","row":0,"col":1,"value":2.5}),
                ),
                StreamOp::new(
                    "set_rhs",
                    "Modify one constraint right-hand side.",
                    json!({"op":"set_rhs","index":0,"rhs":4}),
                ),
                StreamOp::new(
                    "add_variable",
                    "Add a variable: objective coef `c`, integrality `integer`, and \
                     a `column` of length #constraints.",
                    json!({"op":"add_variable","c":2,"integer":true,"column":[1]}),
                ),
                StreamOp::new(
                    "set_variable",
                    "Modify one variable by index. Optional fields: c, integer, ub, column, name.",
                    json!({"op":"set_variable","index":0,"c":4,"integer":true,"ub":10}),
                ),
                StreamOp::new(
                    "remove_variable",
                    "Remove a variable column by index.",
                    json!({"op":"remove_variable","index":1}),
                ),
                StreamOp::new(
                    "set_objective",
                    "Replace the objective vector `c` (length == #vars).",
                    json!({"op":"set_objective","c":[3,2]}),
                ),
                StreamOp::new(
                    "set_integer",
                    "Set integrality of the variable at `index`.",
                    json!({"op":"set_integer","index":0,"integer":false}),
                ),
                StreamOp::new(
                    "set_upper_bound",
                    "Set a finite or infinite upper bound for one variable. Alias: set_ub.",
                    json!({"op":"set_upper_bound","index":0,"ub":1}),
                ),
                StreamOp::new(
                    "snapshot",
                    "Emit current model dimensions without solving.",
                    json!({"op":"snapshot"}),
                ),
                StreamOp::new(
                    "solve",
                    "Run branch-and-bound. Optional: maxNodes, lpMaxIters, intTol.",
                    json!({"op":"solve","maxNodes":10000}),
                ),
            ],
            vec![
                StreamEvent::new("initialized", "Problem created; reports sizes."),
                StreamEvent::new("applied", "An edit was applied to the problem."),
                StreamEvent::new("node", "One explored B&B node (when emitNodes)."),
                StreamEvent::new("trace", "Node count for the completed solve."),
                StreamEvent::new("solution", "Final MILP solution: status, z, x, gap."),
                StreamEvent::new("error", "A command could not be applied."),
            ],
        )
    }

    fn apply(&mut self, command: &Value) -> Vec<Value> {
        let op = op_of(command);
        match op {
            "init" => return self.init(command),
            "solve" => return self.solve(command),
            "snapshot" => {
                if self.initialized {
                    return vec![self.snapshot_frame()];
                }
            }
            _ => {}
        }
        if !self.initialized {
            return vec![error_frame(
                "no MILP initialized; send {\"op\":\"init\", ...} first",
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
                if rhs < 0.0 {
                    return vec![error_frame("`rhs` must be non-negative (b ≥ 0)")];
                }
                self.problem.a.push(coefs);
                self.problem.b.push(rhs);
                if let Some(names) = self.problem.con_names.as_mut() {
                    names.push(
                        str_at(command, "name").unwrap_or_else(|| format!("c{}", names.len() + 1)),
                    );
                }
                vec![
                    json!({"event":"applied","op":"add_constraint","numConstraints": self.n_cons()}),
                ]
            }
            "set_constraint" | "modify_constraint" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_cons() {
                    return vec![error_frame("`index` out of range for constraints")];
                }
                if let Some(coefs) = vec_f64(command, "coefs") {
                    if coefs.len() != self.n_vars() {
                        return vec![error_frame(
                            "`coefs` length must equal the number of variables",
                        )];
                    }
                    self.problem.a[index] = coefs;
                }
                if command.get("rhs").is_some() {
                    let rhs = f64_at(command, "rhs", self.problem.b[index]);
                    if rhs < 0.0 {
                        return vec![error_frame("`rhs` must be non-negative (b ≥ 0)")];
                    }
                    self.problem.b[index] = rhs;
                }
                if let Some(name) = str_at(command, "name") {
                    if let Some(names) = self.problem.con_names.as_mut() {
                        names[index] = name;
                    }
                }
                vec![json!({"event":"applied","op":"set_constraint","index":index})]
            }
            "remove_constraint" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_cons() {
                    return vec![error_frame("`index` out of range for constraints")];
                }
                self.problem.a.remove(index);
                self.problem.b.remove(index);
                if let Some(names) = self.problem.con_names.as_mut() {
                    names.remove(index);
                }
                vec![
                    json!({"event":"applied","op":"remove_constraint","numConstraints": self.n_cons()}),
                ]
            }
            "set_rhs" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_cons() {
                    return vec![error_frame("`index` out of range for constraints")];
                }
                let rhs = f64_at(command, "rhs", self.problem.b[index]);
                if rhs < 0.0 {
                    return vec![error_frame("`rhs` must be non-negative (b ≥ 0)")];
                }
                self.problem.b[index] = rhs;
                vec![json!({"event":"applied","op":"set_rhs","index":index,"rhs":rhs})]
            }
            "set_coefficient" => {
                let row = match usize_at(command, "row") {
                    Some(i) => i,
                    None => return vec![error_frame("`row` (usize) is required")],
                };
                let col = match usize_at(command, "col") {
                    Some(i) => i,
                    None => return vec![error_frame("`col` (usize) is required")],
                };
                if row >= self.n_cons() {
                    return vec![error_frame("`row` out of range for constraints")];
                }
                if col >= self.n_vars() {
                    return vec![error_frame("`col` out of range for variables")];
                }
                let value = f64_at(command, "value", self.problem.a[row][col]);
                self.problem.a[row][col] = value;
                vec![
                    json!({"event":"applied","op":"set_coefficient","row":row,"col":col,"value":value}),
                ]
            }
            "add_variable" => {
                let column = vec_f64(command, "column").unwrap_or_default();
                if column.len() != self.n_cons() {
                    return vec![error_frame(
                        "`column` length must equal the number of constraints",
                    )];
                }
                let coef = f64_at(command, "c", 0.0);
                let integer = bool_at(command, "integer", false);
                self.problem.c.push(coef);
                self.problem.integer_vars.push(integer);
                for (row, value) in self.problem.a.iter_mut().zip(column.iter()) {
                    row.push(*value);
                }
                if self.problem.ub.is_some() || command.get("ub").is_some() {
                    let upper = f64_at(command, "ub", f64::INFINITY);
                    self.ensure_ub().push(upper);
                }
                if let Some(names) = self.problem.var_names.as_mut() {
                    names.push(
                        str_at(command, "name").unwrap_or_else(|| format!("x{}", names.len() + 1)),
                    );
                }
                vec![json!({"event":"applied","op":"add_variable","numVars": self.n_vars()})]
            }
            "set_variable" | "modify_variable" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_vars() {
                    return vec![error_frame("`index` out of range for variables")];
                }
                if command.get("c").is_some() {
                    self.problem.c[index] = f64_at(command, "c", self.problem.c[index]);
                }
                if command.get("integer").is_some() {
                    self.problem.integer_vars[index] = bool_at(command, "integer", true);
                }
                if command.get("ub").is_some() {
                    let upper = f64_at(command, "ub", f64::INFINITY);
                    self.ensure_ub()[index] = upper;
                }
                if let Some(column) = vec_f64(command, "column") {
                    if column.len() != self.n_cons() {
                        return vec![error_frame(
                            "`column` length must equal the number of constraints",
                        )];
                    }
                    for (row, value) in self.problem.a.iter_mut().zip(column.iter()) {
                        row[index] = *value;
                    }
                }
                if let Some(name) = str_at(command, "name") {
                    if let Some(names) = self.problem.var_names.as_mut() {
                        names[index] = name;
                    }
                }
                vec![json!({"event":"applied","op":"set_variable","index":index})]
            }
            "remove_variable" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_vars() {
                    return vec![error_frame("`index` out of range for variables")];
                }
                if self.n_vars() == 1 {
                    return vec![error_frame("cannot remove the last variable")];
                }
                self.problem.c.remove(index);
                self.problem.integer_vars.remove(index);
                for row in &mut self.problem.a {
                    row.remove(index);
                }
                if let Some(ub) = self.problem.ub.as_mut() {
                    ub.remove(index);
                }
                if let Some(names) = self.problem.var_names.as_mut() {
                    names.remove(index);
                }
                vec![json!({"event":"applied","op":"remove_variable","numVars": self.n_vars()})]
            }
            "set_objective" => {
                let new_c = vec_f64(command, "c").unwrap_or_default();
                if new_c.len() != self.n_vars() {
                    return vec![error_frame("`c` length must equal the number of variables")];
                }
                self.problem.c = new_c;
                vec![json!({"event":"applied","op":"set_objective"})]
            }
            "set_sense" => {
                self.problem.sense = match str_at(command, "sense").as_deref() {
                    Some("min") | Some("Min") => Sense::Min,
                    _ => Sense::Max,
                };
                let sense = if self.problem.sense == Sense::Min {
                    "min"
                } else {
                    "max"
                };
                vec![json!({"event":"applied","op":"set_sense","sense": sense})]
            }
            "set_integer" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_vars() {
                    return vec![error_frame("`index` out of range for variables")];
                }
                let integer = bool_at(command, "integer", true);
                self.problem.integer_vars[index] = integer;
                vec![json!({"event":"applied","op":"set_integer","index":index,"integer":integer})]
            }
            "set_upper_bound" | "set_ub" => {
                let index = match usize_at(command, "index") {
                    Some(i) => i,
                    None => return vec![error_frame("`index` (usize) is required")],
                };
                if index >= self.n_vars() {
                    return vec![error_frame("`index` out of range for variables")];
                }
                let upper = f64_at(command, "ub", f64::INFINITY);
                if upper < 0.0 {
                    return vec![error_frame("`ub` must be non-negative")];
                }
                self.ensure_ub()[index] = upper;
                vec![json!({"event":"applied","op":"set_upper_bound","index":index,"ub":upper})]
            }
            other => vec![error_frame(format!(
                "unknown op `{other}` for streaming-milp"
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
    fn solves_simple_integer_program() {
        // max x  s.t. 2x <= 3, x >= 0, x integer  ->  x = 1, z = 1.
        let mut m = StreamingMilp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","sense":"max","c":[1],"a":[[2]],"b":[3],"integerVars":[true]}),
                json!({"op":"solve"}),
            ],
        );
        let sol = last_solution(&frames);
        assert_eq!(sol["status"], json!("optimal"));
        assert!((sol["z"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn streamed_variable_and_constraint_then_solve() {
        // Start: max 3x1 s.t. x1 <= 4 (integer). Add x2 (coef 2) and a joint cap.
        let mut m = StreamingMilp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","sense":"max","c":[3],"a":[[1]],"b":[4],"integerVars":[true]}),
                json!({"op":"add_variable","c":2,"integer":true,"column":[1]}),
                // now vars x1,x2 with constraint x1 + x2 <= 4
                json!({"op":"add_constraint","coefs":[1,1],"rhs":4}),
                json!({"op":"solve"}),
            ],
        );
        let sol = last_solution(&frames);
        assert_eq!(sol["status"], json!("optimal"));
        // max 3x1 + 2x2, x1<=4, x1+x2<=4, integer -> x1=4,x2=0 -> 12.
        assert!((sol["z"].as_f64().unwrap() - 12.0).abs() < 1e-6);
    }

    #[test]
    fn streamed_constraint_edits_re_solve_current_revision() {
        let mut m = StreamingMilp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","sense":"max","c":[1],"a":[[1]],"b":[4],"integerVars":[true]}),
                json!({"op":"set_rhs","index":0,"rhs":2}),
                json!({"op":"solve"}),
            ],
        );
        let sol = last_solution(&frames);
        assert_eq!(sol["status"], json!("optimal"));
        assert!((sol["z"].as_f64().unwrap() - 2.0).abs() < 1e-6);

        let frames = drive(
            &mut m,
            &[
                json!({"op":"remove_constraint","index":0}),
                json!({"op":"add_constraint","coefs":[2],"rhs":3}),
                json!({"op":"solve"}),
            ],
        );
        let sol = last_solution(&frames);
        assert_eq!(sol["status"], json!("optimal"));
        assert!((sol["z"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn streamed_variable_remove_updates_matrix_and_objective() {
        let mut m = StreamingMilp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","sense":"max","c":[1,10],"a":[[1,1]],"b":[1],"integerVars":[true,true]}),
                json!({"op":"remove_variable","index":1}),
                json!({"op":"set_upper_bound","index":0,"ub":1}),
                json!({"op":"solve"}),
            ],
        );
        let sol = last_solution(&frames);
        assert_eq!(sol["status"], json!("optimal"));
        assert_eq!(sol["x"].as_array().unwrap().len(), 1);
        assert!((sol["z"].as_f64().unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn emits_node_trace_when_requested() {
        let mut m = StreamingMilp::new();
        let frames = drive(
            &mut m,
            &[
                json!({"op":"init","sense":"max","c":[1],"a":[[2]],"b":[3],"integerVars":[true],"emitNodes":true}),
                json!({"op":"solve"}),
            ],
        );
        assert!(frames.iter().any(|f| f["event"] == json!("node")));
        assert!(frames.iter().any(|f| f["event"] == json!("trace")));
    }
}
