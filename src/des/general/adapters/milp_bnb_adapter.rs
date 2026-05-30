//! Port of `src/des/general/adapters/milp-bnb-adapter.ts`
//! (module `des::general::adapters::milp_bnb_adapter`).
//!
//! Two JSON adapters: the MILP branch-and-bound solver (`milp-bnb`) and the
//! explicit station-graph IP/MIP-DES solver (`ip-mip-des`).
//!
//! ## Conversion notes
//!
//!   * `sense: 'max' | 'min'` -> [`SenseParam`] (converted to the two distinct
//!     engine `Sense` enums: `milp_bnb::Sense` for MILP, `lp::Sense` for IP/MIP).
//!   * `branchRule` / `nodeSelection` / `lpAlgorithm` string unions -> the
//!     ported enums (`BranchRule`, `NodeSelection`, `LpRelaxationAlgorithm`).
//!   * Empty optional arrays coerced to `undefined`
//!     (`ub.length > 0 ? ub : undefined`) -> `Option` + `is_empty()` guard.
//!   * `Object.entries(result.lpAlgorithmUsage)` (insertion-ordered in JS) ->
//!     a `HashMap` whose entries are sorted by key spelling for a deterministic
//!     summary (see PORT NOTE below).
//!   * `Number.isFinite(z) ? z.toFixed(6) : z` -> [`fixed_or_raw`];
//!     `gap.toExponential(2)` -> [`to_exponential`].
//!   * `throw new Error(...)` for missing `{raw, knapsack}` -> `panic!`.
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the two
//! adapters are exposed via [`milp_adapter`] / [`ip_mip_des_adapter`].
//!
//! PORT NOTE: `IPMIPSolution::lp_algorithm_usage` is a `HashMap`, which has no
//! stable iteration order (the TS `Record` iterated in insertion order). The
//! summary sorts the entries by algorithm spelling so the output is
//! deterministic; this can differ in ordering from the TS for multi-backend
//! runs.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::des_base::stateful_token::TokenStateMode;
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::ip_mip_des::{
    build_binary_knapsack_ip, solve_ipmip_with_des, BranchRule as IpBranchRule, ConstraintNode,
    IPMIPProblem, IPMIPSolution, IPMIPSolveOptions, IPMIPTraceEvent, LpRelaxationAlgorithm,
    NodeSelection, TraceAction, VariableNode,
};
use crate::des::general::lp::Sense as LpSense;
use crate::des::general::milp_bnb::{
    build_knapsack_milp, solve_milp, BranchRule as MilpBranchRule, BranchType, LpStatus,
    MILPProblem, MILPSolution, MILPSolveOptions, NodeEvent, PrunedReason, Sense as MilpSense,
};

// =============================================================================
// Shared display helpers (JS number-formatting parity).
// =============================================================================

/// `String(v)` for a JS number (used by CSV cells / raw summary fields).
fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 { "Infinity".to_string() } else { "-Infinity".to_string() }
    } else {
        let s = v.to_string();
        if s == "-0" { "0".to_string() } else { s }
    }
}

/// `Number.isFinite(v) ? v.toFixed(d) : v`.
fn fixed_or_raw(v: f64, digits: usize) -> String {
    if v.is_finite() {
        format!("{:.*}", digits, v)
    } else {
        js_number(v)
    }
}

/// `v.toExponential(d)` (mantissa with `d` fraction digits, signed exponent).
fn to_exponential(v: f64, digits: usize) -> String {
    if !v.is_finite() {
        return js_number(v);
    }
    let raw = format!("{:.*e}", digits, v);
    match raw.split_once('e') {
        Some((mant, exp)) if !exp.starts_with('-') => format!("{mant}e+{exp}"),
        _ => raw,
    }
}

/// First-16 pretty `x*` vector (`v.toFixed(3)` / `N/A` for non-finite).
fn x_pretty(x: &[f64]) -> String {
    x.iter()
        .take(16)
        .map(|&v| if v.is_finite() { format!("{v:.3}") } else { "N/A".to_string() })
        .collect::<Vec<_>>()
        .join(", ")
}

// =============================================================================
// Shared param types
// =============================================================================

/// `sense: 'max' | 'min'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SenseParam {
    Max,
    Min,
}

impl SenseParam {
    fn to_milp(self) -> MilpSense {
        match self {
            SenseParam::Max => MilpSense::Max,
            SenseParam::Min => MilpSense::Min,
        }
    }
    fn to_lp(self) -> LpSense {
        match self {
            SenseParam::Max => LpSense::Max,
            SenseParam::Min => LpSense::Min,
        }
    }
}

/// `knapsack` convenience builder block (shared by both adapters).
#[derive(Clone, Debug)]
pub struct KnapsackParams {
    pub values: Vec<f64>,
    pub weights: Vec<f64>,
    pub capacity: f64,
}

// =============================================================================
// Shared schema fragments
// =============================================================================

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default: None, description: None }
}

fn arr(items: ParamSchema) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length: None, max_length: None, description: None }
}

fn str_enum(allowed: &[&str]) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: None,
        description: None,
    }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

/// The `raw` object schema, identical for both adapters.
fn raw_schema() -> ParamSchema {
    obj(
        vec![
            ("sense", str_enum(&["max", "min"])),
            ("c", arr(num(None, None, None))),
            ("A", arr(arr(num(None, None, None)))),
            ("b", arr(num(None, None, None))),
            ("integerVars", arr(ParamSchema::Boolean { default: None, description: None })),
            ("ub", arr(num(None, None, None))),
            ("varNames", arr(ParamSchema::String { allowed: None, default: None, description: None })),
            ("conNames", arr(ParamSchema::String { allowed: None, default: None, description: None })),
        ],
        vec!["sense", "c", "A", "b", "integerVars"],
    )
}

/// The `knapsack` object schema, identical for both adapters.
fn knapsack_schema() -> ParamSchema {
    obj(
        vec![
            ("values", arr(num(None, None, None))),
            ("weights", arr(num(None, None, None))),
            ("capacity", num(None, None, None)),
        ],
        vec!["values", "weights", "capacity"],
    )
}

// =============================================================================
// MILP branch-and-bound adapter
// =============================================================================

/// `raw` block for the MILP adapter.
#[derive(Clone, Debug)]
pub struct MilpRaw {
    pub sense: SenseParam,
    pub c: Vec<f64>,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub integer_vars: Vec<bool>,
    pub ub: Option<Vec<f64>>,
    pub var_names: Option<Vec<String>>,
    pub con_names: Option<Vec<String>>,
}

/// `options` block for the MILP adapter.
#[derive(Clone, Debug, Default)]
pub struct MilpOptionsParams {
    pub max_nodes: Option<usize>,
    pub lp_max_iters: Option<usize>,
    pub int_tol: Option<f64>,
    pub branch_rule: Option<MilpBranchRule>,
    pub initial_incumbent_z: Option<f64>,
}

/// `interface MILPParams`.
#[derive(Clone, Debug, Default)]
pub struct MILPParams {
    pub raw: Option<MilpRaw>,
    pub knapsack: Option<KnapsackParams>,
    pub options: Option<MilpOptionsParams>,
}

/// `const milpSchema`.
pub fn milp_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            ("raw".to_string(), raw_schema()),
            ("knapsack".to_string(), knapsack_schema()),
            (
                "options".to_string(),
                obj(
                    vec![
                        ("maxNodes", num(Some(1.0), None, Some(true))),
                        ("lpMaxIters", num(Some(1.0), None, Some(true))),
                        ("intTol", num(Some(0.0), None, None)),
                        ("branchRule", str_enum(&["most-fractional", "first-fractional"])),
                        ("initialIncumbentZ", num(None, None, None)),
                    ],
                    vec![],
                ),
            ),
        ],
        required: Some(vec![]),
        description: Some(
            "MILP solved via branch-and-bound with IncrementalLP relaxations at each node.".to_string(),
        ),
    }
}

fn milp_status_str(s: crate::des::general::milp_bnb::MILPStatus) -> &'static str {
    use crate::des::general::milp_bnb::MILPStatus;
    match s {
        MILPStatus::Optimal => "optimal",
        MILPStatus::Infeasible => "infeasible",
        MILPStatus::Unbounded => "unbounded",
        MILPStatus::MaxNodes => "maxnodes",
    }
}

fn milp_lp_status_str(s: LpStatus) -> &'static str {
    match s {
        LpStatus::Optimal => "optimal",
        LpStatus::Infeasible => "infeasible",
        LpStatus::Unbounded => "unbounded",
    }
}

fn branch_type_str(t: BranchType) -> &'static str {
    match t {
        BranchType::Le => "le",
        BranchType::Ge => "ge",
    }
}

fn pruned_reason_str(r: PrunedReason) -> &'static str {
    match r {
        PrunedReason::Infeasible => "infeasible",
        PrunedReason::Unbounded => "unbounded",
        PrunedReason::Bound => "bound",
        PrunedReason::IntegerFeasible => "integer-feasible",
    }
}

/// `const adapter` (`milp-bnb`).
pub struct MilpBnbAdapter;

/// Construct the MILP adapter (see the module PORT NOTE on registration).
pub fn milp_adapter() -> MilpBnbAdapter {
    MilpBnbAdapter
}

impl DESModelRegistration<MILPParams, MILPSolution> for MilpBnbAdapter {
    fn id(&self) -> &str {
        "milp-bnb"
    }

    fn description(&self) -> &str {
        "Mixed-integer LP via branch-and-bound, composing IncrementalLP for relaxations."
    }

    fn schema(&self) -> ParamSchema {
        milp_schema()
    }

    fn run(&self, params: MILPParams, _runtime: &DESRuntimeConfig) -> MILPSolution {
        let problem: MILPProblem = if let Some(k) = &params.knapsack {
            build_knapsack_milp(k.values.clone(), k.weights.clone(), k.capacity)
        } else if let Some(raw) = &params.raw {
            let ub = raw.ub.as_ref().filter(|u| !u.is_empty()).cloned();
            let var_names = raw.var_names.as_ref().filter(|v| !v.is_empty()).cloned();
            let con_names = raw.con_names.as_ref().filter(|v| !v.is_empty()).cloned();
            MILPProblem {
                sense: raw.sense.to_milp(),
                c: raw.c.clone(),
                a: raw.a.clone(),
                b: raw.b.clone(),
                integer_vars: raw.integer_vars.clone(),
                ub,
                var_names,
                con_names,
            }
        } else {
            panic!("milp-bnb: provide one of {{raw, knapsack}}");
        };

        let opts = params
            .options
            .as_ref()
            .map(|o| MILPSolveOptions {
                max_nodes: o.max_nodes,
                lp_max_iters: o.lp_max_iters,
                int_tol: o.int_tol,
                branch_rule: o.branch_rule,
                verbose: None,
                initial_incumbent_z: o.initial_incumbent_z,
                branch_seed: None,
            })
            .unwrap_or_default();

        solve_milp(&problem, opts)
    }

    fn summarize(&self, result: &MILPSolution, _params: &MILPParams) -> String {
        let xp = x_pretty(&result.x);
        [
            "MILP-BRANCH-AND-BOUND RUN SUMMARY".to_string(),
            "──────────────────────────────────".to_string(),
            format!("  Status:           {}", milp_status_str(result.status)),
            format!("  z*:               {}", fixed_or_raw(result.z, 6)),
            format!("  Best bound:       {}", format!("{:.6}", result.best_bound)),
            format!("  Optimality gap:   {}", to_exponential(result.gap, 2)),
            format!("  Nodes explored:   {}", result.nodes_explored),
            format!("  LP pivots total:  {}", result.total_pivots),
            "".to_string(),
            format!(
                "  x* (first 16):    [{}{}]",
                xp,
                if result.x.len() > 16 { ", …" } else { "" }
            ),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &MILPSolution, csv_path: &str) {
        let mut lines = vec!["var_index,value".to_string()];
        for (i, &xi) in result.x.iter().enumerate() {
            lines.push(csv_row([i.to_string(), js_number(xi)]));
        }
        lines.push(String::new());
        lines.push(
            "node_id,parent_id,depth,branch_var,branch_type,branch_value,lp_status,lp_z,fractional_count,pruned,pruned_reason,incumbent_updated"
                .to_string(),
        );
        for e in &result.trace {
            lines.push(csv_row(milp_trace_cells(e)));
        }
        write_csv_lines(csv_path, &lines);
    }

    fn examples(&self) -> Vec<RegistrationExample<MILPParams>> {
        vec![RegistrationExample {
            name: "milp-knapsack-4item".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "milp-bnb".to_string(),
                description: Some("4-item textbook knapsack".to_string()),
                parameters: MILPParams {
                    raw: None,
                    knapsack: Some(KnapsackParams {
                        values: vec![10.0, 40.0, 30.0, 50.0],
                        weights: vec![5.0, 4.0, 6.0, 3.0],
                        capacity: 10.0,
                    }),
                    options: None,
                },
                runtime: None,
                metadata: None,
            },
        }]
    }
}

fn milp_trace_cells(e: &NodeEvent) -> Vec<String> {
    vec![
        e.node_id.to_string(),
        e.parent_id.map(|p| p.to_string()).unwrap_or_default(),
        e.depth.to_string(),
        e.branch_var.map(|b| b.to_string()).unwrap_or_default(),
        e.branch_type.map(|t| branch_type_str(t).to_string()).unwrap_or_default(),
        e.branch_value.map(js_number).unwrap_or_default(),
        milp_lp_status_str(e.lp_status).to_string(),
        e.lp_z.map(js_number).unwrap_or_default(),
        e.fractional.len().to_string(),
        e.pruned.to_string(),
        e.pruned_reason.map(|r| pruned_reason_str(r).to_string()).unwrap_or_default(),
        e.incumbent_updated.to_string(),
    ]
}

// =============================================================================
// Explicit station-graph IP/MIP solver adapter
// =============================================================================

/// `raw` block for the IP/MIP adapter (mirrors `IPMIPProblem`).
#[derive(Clone, Debug)]
pub struct IpMipRaw {
    pub sense: SenseParam,
    pub c: Vec<f64>,
    pub a: Vec<Vec<f64>>,
    pub b: Vec<f64>,
    pub integer_vars: Vec<bool>,
    pub ub: Option<Vec<f64>>,
    pub var_names: Option<Vec<String>>,
    pub con_names: Option<Vec<String>>,
    /// Optional graph metadata carried through from `IPMIPProblem`.
    pub variable_nodes: Option<Vec<VariableNode>>,
    pub constraint_nodes: Option<Vec<ConstraintNode>>,
}

/// `options` block for the IP/MIP adapter.
#[derive(Clone, Debug, Default)]
pub struct IpMipOptionsParams {
    pub max_nodes: Option<usize>,
    pub max_ticks: Option<usize>,
    pub lp_max_iters: Option<usize>,
    pub int_tol: Option<f64>,
    pub branch_rule: Option<IpBranchRule>,
    pub node_selection: Option<NodeSelection>,
    pub lp_algorithm: Option<LpRelaxationAlgorithm>,
    pub max_cut_rounds: Option<usize>,
    pub max_cuts_per_node: Option<usize>,
    pub heuristic_passes: Option<usize>,
    pub allow_external_solvers: Option<bool>,
}

/// `interface IPMIPDESParams`.
#[derive(Clone, Debug, Default)]
pub struct IPMIPDESParams {
    pub raw: Option<IpMipRaw>,
    pub knapsack: Option<KnapsackParams>,
    pub options: Option<IpMipOptionsParams>,
}

/// `const ipMipDESSchema`.
pub fn ip_mip_des_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            ("raw".to_string(), raw_schema()),
            ("knapsack".to_string(), knapsack_schema()),
            (
                "options".to_string(),
                obj(
                    vec![
                        ("maxNodes", num(Some(1.0), None, Some(true))),
                        ("maxTicks", num(Some(1.0), None, Some(true))),
                        ("lpMaxIters", num(Some(1.0), None, Some(true))),
                        ("intTol", num(Some(0.0), None, None)),
                        ("branchRule", str_enum(&["most-fractional", "first-fractional"])),
                        ("nodeSelection", str_enum(&["dfs", "best-bound"])),
                        (
                            "lpAlgorithm",
                            str_enum(&[
                                "auto",
                                "incremental-primal-dual",
                                "des-simplex-dantzig",
                                "des-simplex-bland",
                                "internal-simplex",
                                "external-highs",
                                "external-highs-ds",
                                "external-highs-ipm",
                            ]),
                        ),
                        ("maxCutRounds", num(Some(0.0), None, Some(true))),
                        ("maxCutsPerNode", num(Some(0.0), None, Some(true))),
                        ("heuristicPasses", num(Some(0.0), None, Some(true))),
                        ("allowExternalSolvers", ParamSchema::Boolean { default: None, description: None }),
                    ],
                    vec![],
                ),
            ),
        ],
        required: Some(vec![]),
        description: Some(
            "Integer / mixed-integer program solved by an explicit DES station graph.".to_string(),
        ),
    }
}

fn trace_action_str(a: TraceAction) -> &'static str {
    match a {
        TraceAction::Branch => "branch",
        TraceAction::Cut => "cut",
        TraceAction::Prune => "prune",
        TraceAction::Incumbent => "incumbent",
        TraceAction::Unbounded => "unbounded",
    }
}

fn token_state_mode_str(m: TokenStateMode) -> &'static str {
    match m {
        TokenStateMode::Stateless => "stateless",
        TokenStateMode::Stateful => "stateful",
    }
}

/// `const ipMipDESAdapter` (`ip-mip-des`).
pub struct IpMipDesAdapter;

/// Construct the IP/MIP adapter (see the module PORT NOTE on registration).
pub fn ip_mip_des_adapter() -> IpMipDesAdapter {
    IpMipDesAdapter
}

impl DESModelRegistration<IPMIPDESParams, IPMIPSolution> for IpMipDesAdapter {
    fn id(&self) -> &str {
        "ip-mip-des"
    }

    fn description(&self) -> &str {
        "Integer/MIP solver graph: LP relaxation, rounding/repair, cuts, incumbent, and branching stations."
    }

    fn schema(&self) -> ParamSchema {
        ip_mip_des_schema()
    }

    fn run(&self, params: IPMIPDESParams, _runtime: &DESRuntimeConfig) -> IPMIPSolution {
        let problem: IPMIPProblem = if let Some(k) = &params.knapsack {
            build_binary_knapsack_ip(k.values.clone(), k.weights.clone(), k.capacity)
        } else if let Some(raw) = &params.raw {
            IPMIPProblem {
                sense: raw.sense.to_lp(),
                c: raw.c.clone(),
                a: raw.a.clone(),
                b: raw.b.clone(),
                integer_vars: raw.integer_vars.clone(),
                ub: raw.ub.as_ref().filter(|u| !u.is_empty()).cloned(),
                var_names: raw.var_names.as_ref().filter(|v| !v.is_empty()).cloned(),
                con_names: raw.con_names.as_ref().filter(|v| !v.is_empty()).cloned(),
                variable_nodes: raw.variable_nodes.clone(),
                constraint_nodes: raw.constraint_nodes.clone(),
            }
        } else {
            panic!("ip-mip-des: provide one of {{raw, knapsack}}");
        };

        let opts = params
            .options
            .as_ref()
            .map(|o| IPMIPSolveOptions {
                max_nodes: o.max_nodes,
                max_ticks: o.max_ticks,
                time_limit_ms: None,
                lp_max_iters: o.lp_max_iters,
                int_tol: o.int_tol,
                branch_rule: o.branch_rule,
                node_selection: o.node_selection,
                lp_algorithm: o.lp_algorithm,
                allow_external_solvers: o.allow_external_solvers,
                max_cut_rounds: o.max_cut_rounds,
                max_cuts_per_node: o.max_cuts_per_node,
                heuristic_passes: o.heuristic_passes,
                verbose: None,
            })
            .unwrap_or_default();

        solve_ipmip_with_des(problem, opts)
    }

    fn summarize(&self, result: &IPMIPSolution, _params: &IPMIPDESParams) -> String {
        let xp = x_pretty(&result.x);
        let usage = {
            let mut entries: Vec<(&'static str, u64)> = result
                .lp_algorithm_usage
                .iter()
                .map(|(k, v)| (k.as_str(), *v))
                .collect();
            entries.sort_by_key(|(k, _)| *k);
            let joined = entries
                .iter()
                .map(|(k, v)| format!("{k}={v}"))
                .collect::<Vec<_>>()
                .join(", ");
            if joined.is_empty() { "none".to_string() } else { joined }
        };
        [
            "IP/MIP DES SOLVER GRAPH".to_string(),
            "-----------------------".to_string(),
            format!("  Status:           {}", result.status.as_str()),
            format!("  Execution mode:   {}", result.execution_mode),
            format!("  In-house only:    {}", result.in_house_only),
            format!("  LP backend:       {}", result.lp_algorithm.as_str()),
            format!("  LP backend usage: {usage}"),
            format!(
                "  Technique plan:   {}{}",
                result.technique_plan.root_lp_algorithm.as_str(),
                if result.technique_plan.external_candidate { " (external candidate)" } else { "" }
            ),
            format!("  z*:               {}", fixed_or_raw(result.z, 6)),
            format!("  Best bound:       {}", fixed_or_raw(result.best_bound, 6)),
            format!("  Gap:              {}", to_exponential(result.gap, 2)),
            format!("  Nodes explored:   {}", result.nodes_explored),
            format!(
                "  Elapsed:          {} ms ({} nodes/s)",
                js_number(result.performance.elapsed_ms),
                format!("{:.2}", result.performance.nodes_per_second)
            ),
            format!("  LP solves:        {}", result.lp_solves),
            format!("  LP solver time:   {} ms", js_number(result.performance.total_lp_solver_ms)),
            format!("  LP iterations:    {}", result.total_lp_iterations),
            format!("  Cuts added:       {}", result.cuts_added),
            format!("  Candidates tried: {}", result.candidates_tried),
            format!(
                "  Solver tokens:    {} ({} stateful / {} stateless)",
                result.token_stats.created, result.token_stats.stateful, result.token_stats.stateless
            ),
            format!(
                "  Incumbent source: {}",
                result.incumbent_source.clone().unwrap_or_else(|| "none".to_string())
            ),
            format!(
                "  x* (first 16):    [{}{}]",
                xp,
                if result.x.len() > 16 { ", ..." } else { "" }
            ),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &IPMIPSolution, csv_path: &str) {
        let mut lines = vec![
            "node_id,parent_id,depth,lp_status,lp_z,solver,fractional_count,action,reason,children,cuts_added,node_token_id,lineage_root,token_generation,state_mode"
                .to_string(),
        ];
        for e in &result.trace {
            lines.push(csv_row(ip_mip_trace_cells(e)));
        }
        write_csv_lines(csv_path, &lines);
    }

    fn examples(&self) -> Vec<RegistrationExample<IPMIPDESParams>> {
        vec![RegistrationExample {
            name: "ip-mip-des-knapsack".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "ip-mip-des".to_string(),
                description: Some(
                    "4-item knapsack solved by the explicit IP/MIP DES station graph.".to_string(),
                ),
                parameters: IPMIPDESParams {
                    raw: None,
                    knapsack: Some(KnapsackParams {
                        values: vec![10.0, 40.0, 30.0, 50.0],
                        weights: vec![5.0, 4.0, 6.0, 3.0],
                        capacity: 10.0,
                    }),
                    options: Some(IpMipOptionsParams {
                        lp_algorithm: Some(LpRelaxationAlgorithm::Auto),
                        max_cut_rounds: Some(1),
                        ..Default::default()
                    }),
                },
                runtime: None,
                metadata: None,
            },
        }]
    }
}

fn ip_mip_trace_cells(e: &IPMIPTraceEvent) -> Vec<String> {
    vec![
        e.node_id.to_string(),
        e.parent_id.map(|p| p.to_string()).unwrap_or_default(),
        e.depth.to_string(),
        e.lp_status.as_str().to_string(),
        e.lp_z.map(js_number).unwrap_or_default(),
        e.solver.clone(),
        e.fractional.len().to_string(),
        trace_action_str(e.action).to_string(),
        e.reason.clone().unwrap_or_default(),
        e.children
            .as_ref()
            .map(|c| c.iter().map(|x| x.to_string()).collect::<Vec<_>>().join("|"))
            .unwrap_or_default(),
        e.cuts_added.map(|c| c.to_string()).unwrap_or_default(),
        e.node_token_id.clone().unwrap_or_default(),
        e.lineage_root.clone().unwrap_or_default(),
        e.token_generation.map(|g| g.to_string()).unwrap_or_default(),
        e.state_mode.map(|m| token_state_mode_str(m).to_string()).unwrap_or_default(),
    ]
}
