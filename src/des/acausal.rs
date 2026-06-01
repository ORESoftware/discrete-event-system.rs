//! `des::acausal` - a small equation-based modeling surface.
//!
//! This is the first ModelingToolkit-style seam in the Rust SDK: users can
//! submit a JSON model with variables, parameters, explicit differential
//! equations, algebraic assignments, and alias/connect equations. The compiler
//! performs a structural pass (alias elimination, dependency sorting, algebraic
//! loop detection) before generating a runnable fixed-step simulator and a
//! UI-facing workbench descriptor.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::panic::{catch_unwind, AssertUnwindSafe};

use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};

use crate::des::general::expr::{evaluate, parse, simplify, stringify, Env, Expr};
use crate::des::model::{CitizenError, ModelCitizen, ModelDescriptor, RunArtifact};
use crate::des::plugin::UiControl;

pub const ACAUSAL_SCHEMA: &str = "des/acausal-model/v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcausalVariableKind {
    State,
    Algebraic,
    Parameter,
    Input,
    Output,
}

impl AcausalVariableKind {
    fn as_str(self) -> &'static str {
        match self {
            AcausalVariableKind::State => "state",
            AcausalVariableKind::Algebraic => "algebraic",
            AcausalVariableKind::Parameter => "parameter",
            AcausalVariableKind::Input => "input",
            AcausalVariableKind::Output => "output",
        }
    }

    fn alias_priority(self) -> usize {
        match self {
            AcausalVariableKind::State => 0,
            AcausalVariableKind::Output => 1,
            AcausalVariableKind::Algebraic => 2,
            AcausalVariableKind::Input => 3,
            AcausalVariableKind::Parameter => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcausalVariableSpec {
    pub name: String,
    pub kind: AcausalVariableKind,
    #[serde(default)]
    pub initial: Option<f64>,
    #[serde(default)]
    pub value: Option<f64>,
    #[serde(default)]
    pub unit: Option<String>,
    #[serde(default)]
    pub description: Option<String>,
}

impl AcausalVariableSpec {
    pub fn state(name: &str, initial: f64, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: AcausalVariableKind::State,
            initial: Some(initial),
            value: None,
            unit: Some(unit.to_string()),
            description: None,
        }
    }

    pub fn parameter(name: &str, value: f64, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: AcausalVariableKind::Parameter,
            initial: None,
            value: Some(value),
            unit: Some(unit.to_string()),
            description: None,
        }
    }

    pub fn algebraic(name: &str, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: AcausalVariableKind::Algebraic,
            initial: None,
            value: None,
            unit: Some(unit.to_string()),
            description: None,
        }
    }

    pub fn output(name: &str, unit: &str) -> Self {
        Self {
            name: name.to_string(),
            kind: AcausalVariableKind::Output,
            initial: None,
            value: None,
            unit: Some(unit.to_string()),
            description: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcausalEquationKind {
    /// `d(lhs) / dt = rhs`, where `lhs` is a state variable.
    Derivative,
    /// `lhs = rhs`, where `lhs` is an algebraic/output variable.
    Assignment,
    /// Alias/connect equation. `rhs` must be a variable name.
    Alias,
    /// Reserved for future nonlinear DAE residual support. Currently rejected.
    Residual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcausalEquationSpec {
    pub kind: AcausalEquationKind,
    pub lhs: String,
    pub rhs: String,
    #[serde(default)]
    pub label: Option<String>,
}

impl AcausalEquationSpec {
    pub fn derivative(lhs: &str, rhs: &str) -> Self {
        Self {
            kind: AcausalEquationKind::Derivative,
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
            label: None,
        }
    }

    pub fn assignment(lhs: &str, rhs: &str) -> Self {
        Self {
            kind: AcausalEquationKind::Assignment,
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
            label: None,
        }
    }

    pub fn alias(lhs: &str, rhs: &str) -> Self {
        Self {
            kind: AcausalEquationKind::Alias,
            lhs: lhs.to_string(),
            rhs: rhs.to_string(),
            label: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum AcausalSolverKind {
    Euler,
    Rk4,
}

fn default_steps() -> usize {
    101
}

fn default_dt() -> f64 {
    0.01
}

fn default_solver() -> AcausalSolverKind {
    AcausalSolverKind::Rk4
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcausalModelSpec {
    #[serde(rename = "$schema", default)]
    pub schema: Option<String>,
    pub name: String,
    #[serde(default = "default_dt")]
    pub dt: f64,
    #[serde(default = "default_steps")]
    pub steps: usize,
    #[serde(default = "default_solver")]
    pub solver: AcausalSolverKind,
    pub variables: Vec<AcausalVariableSpec>,
    pub equations: Vec<AcausalEquationSpec>,
    #[serde(default)]
    pub metadata: Map<String, Value>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AliasElimination {
    pub variable: String,
    pub canonical: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StructuralDiagnostics {
    pub variables: usize,
    pub states: usize,
    pub algebraics: usize,
    pub parameters: usize,
    pub equations: usize,
    pub assignments: usize,
    pub derivatives: usize,
    pub aliases_eliminated: Vec<AliasElimination>,
    pub algebraic_order: Vec<String>,
    pub derivative_order: Vec<String>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug)]
struct CompiledAssignment {
    lhs: String,
    rhs: String,
    expr: Expr,
    dependencies: Vec<String>,
}

#[derive(Clone, Debug)]
struct CompiledDerivative {
    state: String,
    rhs: String,
    expr: Expr,
    dependencies: Vec<String>,
}

#[derive(Clone, Debug)]
pub struct CompiledAcausalModel {
    pub name: String,
    pub dt: f64,
    pub steps: usize,
    pub solver: AcausalSolverKind,
    pub variables: Vec<AcausalVariableSpec>,
    pub diagnostics: StructuralDiagnostics,
    state_names: Vec<String>,
    parameter_values: BTreeMap<String, f64>,
    initial_state: BTreeMap<String, f64>,
    assignments: Vec<CompiledAssignment>,
    derivatives: Vec<CompiledDerivative>,
    canonical_by_original: BTreeMap<String, String>,
    original_equations: Vec<AcausalEquationSpec>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AcausalError {
    InvalidSpec(String),
    DuplicateVariable(String),
    UnknownVariable {
        name: String,
        context: String,
    },
    DuplicateEquation {
        variable: String,
        kind: &'static str,
    },
    MissingDerivative(String),
    MissingAssignment(String),
    UnsupportedResidual(String),
    Parse {
        context: String,
        message: String,
    },
    AlgebraicLoop(Vec<String>),
    Run(String),
}

impl std::fmt::Display for AcausalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AcausalError::InvalidSpec(msg) => write!(f, "{msg}"),
            AcausalError::DuplicateVariable(name) => write!(f, "duplicate variable `{name}`"),
            AcausalError::UnknownVariable { name, context } => {
                write!(f, "unknown variable `{name}` in {context}")
            }
            AcausalError::DuplicateEquation { variable, kind } => {
                write!(f, "duplicate {kind} equation for `{variable}`")
            }
            AcausalError::MissingDerivative(name) => {
                write!(f, "state `{name}` has no derivative equation")
            }
            AcausalError::MissingAssignment(name) => {
                write!(f, "algebraic/output variable `{name}` has no assignment")
            }
            AcausalError::UnsupportedResidual(label) => write!(
                f,
                "residual equation `{label}` is not supported by the explicit simulator yet"
            ),
            AcausalError::Parse { context, message } => write!(f, "{context}: {message}"),
            AcausalError::AlgebraicLoop(nodes) => {
                write!(f, "algebraic loop detected: {}", nodes.join(" -> "))
            }
            AcausalError::Run(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for AcausalError {}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcausalPaletteParam {
    pub name: String,
    pub label: String,
    pub kind: String,
    pub default_value: Value,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcausalPaletteItem {
    pub kind: String,
    pub label: String,
    pub category: String,
    pub description: String,
    pub params: Vec<AcausalPaletteParam>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcausalWorkbenchDescriptor {
    pub schema: String,
    pub title: String,
    pub capabilities: Vec<String>,
    pub tabs: Vec<String>,
    pub palette: Vec<AcausalPaletteItem>,
    pub starter: AcausalModelSpec,
}

fn palette_param(name: &str, label: &str, kind: &str, default_value: Value) -> AcausalPaletteParam {
    AcausalPaletteParam {
        name: name.to_string(),
        label: label.to_string(),
        kind: kind.to_string(),
        default_value,
    }
}

fn palette_item(
    kind: &str,
    label: &str,
    category: &str,
    description: &str,
    params: Vec<AcausalPaletteParam>,
) -> AcausalPaletteItem {
    AcausalPaletteItem {
        kind: kind.to_string(),
        label: label.to_string(),
        category: category.to_string(),
        description: description.to_string(),
        params,
    }
}

/// UI metadata for an equation editor / schematic workbench.
pub fn acausal_palette() -> Vec<AcausalPaletteItem> {
    vec![
        palette_item(
            "state",
            "State",
            "Variables",
            "Dynamic variable with an initial condition and derivative equation.",
            vec![
                palette_param("name", "Name", "identifier", json!("x")),
                palette_param("initial", "Initial", "number", json!(0.0)),
                palette_param("unit", "Unit", "text", json!("")),
            ],
        ),
        palette_item(
            "algebraic",
            "Algebraic",
            "Variables",
            "Computed variable sorted by the structural compiler.",
            vec![
                palette_param("name", "Name", "identifier", json!("y")),
                palette_param("unit", "Unit", "text", json!("")),
            ],
        ),
        palette_item(
            "parameter",
            "Parameter",
            "Variables",
            "Constant value available to equations and sweeps.",
            vec![
                palette_param("name", "Name", "identifier", json!("k")),
                palette_param("value", "Value", "number", json!(1.0)),
                palette_param("unit", "Unit", "text", json!("")),
            ],
        ),
        palette_item(
            "derivative",
            "Derivative",
            "Equations",
            "Explicit differential equation d(state)/dt = expression.",
            vec![
                palette_param("lhs", "State", "identifier", json!("x")),
                palette_param("rhs", "Expression", "expression", json!("-k*x")),
            ],
        ),
        palette_item(
            "assignment",
            "Assignment",
            "Equations",
            "Algebraic equation lhs = expression, dependency-sorted before simulation.",
            vec![
                palette_param("lhs", "Variable", "identifier", json!("y")),
                palette_param("rhs", "Expression", "expression", json!("x + 1")),
            ],
        ),
        palette_item(
            "alias",
            "Connect / Alias",
            "Equations",
            "Equate two variable names and eliminate the alias structurally.",
            vec![
                palette_param("lhs", "Variable", "identifier", json!("a")),
                palette_param("rhs", "Variable", "identifier", json!("b")),
            ],
        ),
        palette_item(
            "scope",
            "Scope",
            "Analysis",
            "Plot selected state, algebraic, and output traces from a run artifact.",
            vec![palette_param(
                "signals",
                "Signals",
                "identifier-array",
                json!(["x"]),
            )],
        ),
        palette_item(
            "structural-diagnostics",
            "Structural Diagnostics",
            "Analysis",
            "Show alias eliminations, algebraic ordering, missing equations, and loops.",
            vec![],
        ),
    ]
}

/// A serializable descriptor an embedding UI can expose as its first-load state.
pub fn acausal_workbench_descriptor() -> AcausalWorkbenchDescriptor {
    AcausalWorkbenchDescriptor {
        schema: ACAUSAL_SCHEMA.to_string(),
        title: "Acausal Equation Workbench".to_string(),
        capabilities: vec![
            "explicit-ode".to_string(),
            "algebraic-assignment-sorting".to_string(),
            "alias-elimination".to_string(),
            "unit-metadata".to_string(),
            "rk4-and-euler".to_string(),
            "run-artifact-player".to_string(),
        ],
        tabs: vec![
            "diagram".to_string(),
            "equations".to_string(),
            "variables".to_string(),
            "diagnostics".to_string(),
            "simulation".to_string(),
        ],
        palette: acausal_palette(),
        starter: starter_acausal_model_spec(),
    }
}

/// A small model that exercises states, parameters, algebraics, aliases, and
/// output variables. It behaves like a damped spring-mass system.
pub fn starter_acausal_model_spec() -> AcausalModelSpec {
    AcausalModelSpec {
        schema: Some(ACAUSAL_SCHEMA.to_string()),
        name: "damped-mass-spring".to_string(),
        dt: 0.02,
        steps: 151,
        solver: AcausalSolverKind::Rk4,
        variables: vec![
            AcausalVariableSpec::state("x", 1.0, "m"),
            AcausalVariableSpec::state("v", 0.0, "m/s"),
            AcausalVariableSpec::algebraic("spring_force", "N"),
            AcausalVariableSpec::algebraic("damping_force", "N"),
            AcausalVariableSpec::output("position", "m"),
            AcausalVariableSpec::parameter("m", 1.0, "kg"),
            AcausalVariableSpec::parameter("k", 4.0, "N/m"),
            AcausalVariableSpec::parameter("c", 0.6, "N*s/m"),
        ],
        equations: vec![
            AcausalEquationSpec::assignment("spring_force", "k * x"),
            AcausalEquationSpec::assignment("damping_force", "c * v"),
            AcausalEquationSpec::derivative("x", "v"),
            AcausalEquationSpec::derivative("v", "-(spring_force + damping_force) / m"),
            AcausalEquationSpec::alias("position", "x"),
        ],
        metadata: Map::from_iter([("domain".to_string(), json!("mechanical-translational"))]),
    }
}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn panic_message(e: Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "expression parser failed".to_string()
    }
}

fn parse_expr(src: &str, context: &str) -> Result<Expr, AcausalError> {
    catch_unwind(AssertUnwindSafe(|| simplify(&parse(src)))).map_err(|e| AcausalError::Parse {
        context: context.to_string(),
        message: panic_message(e),
    })
}

fn collect_expr_vars(expr: &Expr, out: &mut BTreeSet<String>) {
    match expr {
        Expr::Num(_) => {}
        Expr::Var(name) => {
            out.insert(name.clone());
        }
        Expr::Neg(arg) => collect_expr_vars(arg, out),
        Expr::Func { arg, .. } => collect_expr_vars(arg, out),
        Expr::Bin { left, right, .. } => {
            collect_expr_vars(left, out);
            collect_expr_vars(right, out);
        }
    }
}

fn expr_vars(expr: &Expr) -> Vec<String> {
    let mut out = BTreeSet::new();
    collect_expr_vars(expr, &mut out);
    out.into_iter().collect()
}

fn rewrite_aliases(expr: &Expr, aliases: &BTreeMap<String, String>) -> Expr {
    match expr {
        Expr::Num(_) => expr.clone(),
        Expr::Var(name) => Expr::Var(aliases.get(name).cloned().unwrap_or_else(|| name.clone())),
        Expr::Neg(arg) => Expr::Neg(Box::new(rewrite_aliases(arg, aliases))),
        Expr::Func { name, arg } => Expr::Func {
            name: *name,
            arg: Box::new(rewrite_aliases(arg, aliases)),
        },
        Expr::Bin { op, left, right } => Expr::Bin {
            op: *op,
            left: Box::new(rewrite_aliases(left, aliases)),
            right: Box::new(rewrite_aliases(right, aliases)),
        },
    }
}

#[derive(Clone, Debug)]
struct UnionFind {
    parent: HashMap<String, String>,
}

impl UnionFind {
    fn new(names: impl Iterator<Item = String>) -> Self {
        let parent = names.map(|name| (name.clone(), name)).collect();
        Self { parent }
    }

    fn find(&mut self, x: &str) -> String {
        let p = self.parent.get(x).cloned().unwrap_or_else(|| x.to_string());
        if p == x {
            return p;
        }
        let root = self.find(&p);
        self.parent.insert(x.to_string(), root.clone());
        root
    }

    fn union(&mut self, a: &str, b: &str) {
        let ra = self.find(a);
        let rb = self.find(b);
        if ra != rb {
            self.parent.insert(rb, ra);
        }
    }
}

fn validate_variable_spec(var: &AcausalVariableSpec) -> Result<(), AcausalError> {
    if !is_ident(&var.name) {
        return Err(AcausalError::InvalidSpec(format!(
            "variable `{}` is not a valid identifier",
            var.name
        )));
    }
    if var.name == "t" {
        return Err(AcausalError::InvalidSpec(
            "`t` is reserved for simulation time".to_string(),
        ));
    }
    if let Some(v) = var.initial {
        if !v.is_finite() {
            return Err(AcausalError::InvalidSpec(format!(
                "variable `{}` has a non-finite initial value",
                var.name
            )));
        }
    }
    if let Some(v) = var.value {
        if !v.is_finite() {
            return Err(AcausalError::InvalidSpec(format!(
                "variable `{}` has a non-finite value",
                var.name
            )));
        }
    }
    Ok(())
}

fn validate_equation_names(
    equation: &AcausalEquationSpec,
    known: &HashSet<String>,
) -> Result<(), AcausalError> {
    if !known.contains(&equation.lhs) {
        return Err(AcausalError::UnknownVariable {
            name: equation.lhs.clone(),
            context: "equation left-hand side".to_string(),
        });
    }
    if equation.kind == AcausalEquationKind::Alias && !known.contains(&equation.rhs) {
        return Err(AcausalError::UnknownVariable {
            name: equation.rhs.clone(),
            context: "alias right-hand side".to_string(),
        });
    }
    Ok(())
}

fn build_alias_map(
    vars_by_name: &HashMap<String, AcausalVariableSpec>,
    equations: &[AcausalEquationSpec],
) -> Result<(BTreeMap<String, String>, Vec<AliasElimination>, Vec<String>), AcausalError> {
    let mut uf = UnionFind::new(vars_by_name.keys().cloned());
    for eq in equations {
        if eq.kind == AcausalEquationKind::Alias {
            if !is_ident(&eq.rhs) {
                return Err(AcausalError::InvalidSpec(format!(
                    "alias `{}` rhs `{}` must be a variable name",
                    eq.lhs, eq.rhs
                )));
            }
            uf.union(&eq.lhs, &eq.rhs);
        }
    }

    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for name in vars_by_name.keys() {
        let root = uf.find(name);
        groups.entry(root).or_default().push(name.clone());
    }

    let mut alias_map = BTreeMap::new();
    let mut eliminations = Vec::new();
    let mut warnings = Vec::new();

    for members in groups.values_mut() {
        members.sort_by(|a, b| {
            let ak = vars_by_name[a].kind.alias_priority();
            let bk = vars_by_name[b].kind.alias_priority();
            ak.cmp(&bk).then_with(|| a.cmp(b))
        });
        let canonical = members[0].clone();
        let state_count = members
            .iter()
            .filter(|name| vars_by_name[*name].kind == AcausalVariableKind::State)
            .count();
        if state_count > 1 {
            return Err(AcausalError::InvalidSpec(format!(
                "alias group `{}` contains multiple state variables",
                members.join(", ")
            )));
        }
        let mut constant_values = members
            .iter()
            .filter_map(|name| {
                let v = &vars_by_name[name];
                matches!(
                    v.kind,
                    AcausalVariableKind::Parameter | AcausalVariableKind::Input
                )
                .then_some((name, v.value.unwrap_or(0.0)))
            })
            .collect::<Vec<_>>();
        constant_values.sort_by(|a, b| a.0.cmp(b.0));
        if constant_values
            .windows(2)
            .any(|w| (w[0].1 - w[1].1).abs() > 1e-12)
        {
            warnings.push(format!(
                "alias group `{}` contains multiple parameter/input values; using `{}`",
                members.join(", "),
                canonical
            ));
        }
        for member in members {
            alias_map.insert(member.clone(), canonical.clone());
            if member != &canonical {
                eliminations.push(AliasElimination {
                    variable: member.clone(),
                    canonical: canonical.clone(),
                });
            }
        }
    }

    Ok((alias_map, eliminations, warnings))
}

fn canonical_variables(
    spec_vars: &[AcausalVariableSpec],
    alias_map: &BTreeMap<String, String>,
) -> Vec<AcausalVariableSpec> {
    let mut grouped: BTreeMap<String, Vec<AcausalVariableSpec>> = BTreeMap::new();
    for var in spec_vars {
        let canonical = alias_map
            .get(&var.name)
            .cloned()
            .unwrap_or_else(|| var.name.clone());
        grouped.entry(canonical).or_default().push(var.clone());
    }

    grouped
        .into_iter()
        .map(|(canonical, mut vars)| {
            vars.sort_by(|a, b| {
                a.kind
                    .alias_priority()
                    .cmp(&b.kind.alias_priority())
                    .then_with(|| a.name.cmp(&b.name))
            });
            let mut chosen = vars[0].clone();
            chosen.name = canonical;
            if chosen.initial.is_none() {
                chosen.initial = vars.iter().find_map(|v| v.initial);
            }
            if chosen.value.is_none() {
                chosen.value = vars.iter().find_map(|v| v.value);
            }
            if chosen.unit.is_none() {
                chosen.unit = vars.iter().find_map(|v| v.unit.clone());
            }
            if chosen.description.is_none() {
                chosen.description = vars.iter().find_map(|v| v.description.clone());
            }
            chosen
        })
        .collect()
}

fn ensure_known_expr_vars(
    expr: &Expr,
    known: &HashSet<String>,
    context: &str,
) -> Result<Vec<String>, AcausalError> {
    let vars = expr_vars(expr);
    for dep in &vars {
        if dep != "t" && !known.contains(dep) {
            return Err(AcausalError::UnknownVariable {
                name: dep.clone(),
                context: context.to_string(),
            });
        }
    }
    Ok(vars)
}

fn sort_assignments(
    assignments: Vec<CompiledAssignment>,
) -> Result<Vec<CompiledAssignment>, AcausalError> {
    let mut index_by_lhs = HashMap::new();
    for (idx, assignment) in assignments.iter().enumerate() {
        if index_by_lhs.insert(assignment.lhs.clone(), idx).is_some() {
            return Err(AcausalError::DuplicateEquation {
                variable: assignment.lhs.clone(),
                kind: "assignment",
            });
        }
    }

    let n = assignments.len();
    let mut indeg = vec![0usize; n];
    let mut adj = vec![Vec::<usize>::new(); n];
    for (idx, assignment) in assignments.iter().enumerate() {
        let mut seen = HashSet::new();
        for dep in &assignment.dependencies {
            if let Some(&dep_idx) = index_by_lhs.get(dep) {
                if dep_idx == idx {
                    return Err(AcausalError::AlgebraicLoop(vec![assignment.lhs.clone()]));
                }
                if seen.insert(dep_idx) {
                    adj[dep_idx].push(idx);
                    indeg[idx] += 1;
                }
            }
        }
    }

    let mut queue: VecDeque<usize> = indeg
        .iter()
        .enumerate()
        .filter_map(|(idx, d)| (*d == 0).then_some(idx))
        .collect();
    let mut order = Vec::with_capacity(n);
    while let Some(idx) = queue.pop_front() {
        order.push(idx);
        for &next in &adj[idx] {
            indeg[next] -= 1;
            if indeg[next] == 0 {
                queue.push_back(next);
            }
        }
    }
    if order.len() != n {
        let stuck = indeg
            .iter()
            .enumerate()
            .filter_map(|(idx, d)| (*d > 0).then_some(assignments[idx].lhs.clone()))
            .collect();
        return Err(AcausalError::AlgebraicLoop(stuck));
    }

    Ok(order
        .into_iter()
        .map(|idx| assignments[idx].clone())
        .collect())
}

/// Compile and structurally simplify a JSON-facing acausal model spec.
pub fn compile_acausal_model(
    spec: &AcausalModelSpec,
) -> Result<CompiledAcausalModel, AcausalError> {
    if spec.name.trim().is_empty() {
        return Err(AcausalError::InvalidSpec(
            "model name cannot be empty".to_string(),
        ));
    }
    if !spec.dt.is_finite() || spec.dt <= 0.0 {
        return Err(AcausalError::InvalidSpec(
            "dt must be a positive finite number".to_string(),
        ));
    }
    if spec.steps == 0 {
        return Err(AcausalError::InvalidSpec(
            "steps must be at least one".to_string(),
        ));
    }

    let mut vars_by_name = HashMap::new();
    for var in &spec.variables {
        validate_variable_spec(var)?;
        if vars_by_name.insert(var.name.clone(), var.clone()).is_some() {
            return Err(AcausalError::DuplicateVariable(var.name.clone()));
        }
    }
    let known_original: HashSet<String> = vars_by_name.keys().cloned().collect();
    for eq in &spec.equations {
        validate_equation_names(eq, &known_original)?;
    }

    let (alias_map, aliases_eliminated, mut warnings) =
        build_alias_map(&vars_by_name, &spec.equations)?;
    let variables = canonical_variables(&spec.variables, &alias_map);
    let vars_by_canonical: HashMap<String, AcausalVariableSpec> = variables
        .iter()
        .map(|v| (v.name.clone(), v.clone()))
        .collect();
    let known_canonical: HashSet<String> = vars_by_canonical.keys().cloned().collect();

    let mut state_names = Vec::new();
    let mut parameter_values = BTreeMap::new();
    let mut initial_state = BTreeMap::new();
    let mut assignment_required = BTreeSet::new();
    for var in &variables {
        match var.kind {
            AcausalVariableKind::State => {
                state_names.push(var.name.clone());
                initial_state.insert(var.name.clone(), var.initial.unwrap_or(0.0));
            }
            AcausalVariableKind::Parameter | AcausalVariableKind::Input => {
                parameter_values.insert(var.name.clone(), var.value.unwrap_or(0.0));
            }
            AcausalVariableKind::Algebraic | AcausalVariableKind::Output => {
                assignment_required.insert(var.name.clone());
            }
        }
    }

    let state_set: HashSet<String> = state_names.iter().cloned().collect();
    let mut derivative_by_state: HashMap<String, CompiledDerivative> = HashMap::new();
    let mut assignments = Vec::new();

    for eq in &spec.equations {
        let lhs = alias_map
            .get(&eq.lhs)
            .cloned()
            .unwrap_or_else(|| eq.lhs.clone());
        match eq.kind {
            AcausalEquationKind::Alias => {}
            AcausalEquationKind::Residual => {
                let label = eq.label.clone().unwrap_or_else(|| eq.lhs.clone());
                return Err(AcausalError::UnsupportedResidual(label));
            }
            AcausalEquationKind::Derivative => {
                if !state_set.contains(&lhs) {
                    return Err(AcausalError::InvalidSpec(format!(
                        "derivative lhs `{}` must be a state variable",
                        eq.lhs
                    )));
                }
                let parsed = parse_expr(&eq.rhs, &format!("derivative `{}`", eq.lhs))?;
                let expr = simplify(&rewrite_aliases(&parsed, &alias_map));
                let deps = ensure_known_expr_vars(
                    &expr,
                    &known_canonical,
                    &format!("derivative `{}`", eq.lhs),
                )?;
                let derivative = CompiledDerivative {
                    state: lhs.clone(),
                    rhs: stringify(&expr),
                    expr,
                    dependencies: deps,
                };
                if derivative_by_state
                    .insert(lhs.clone(), derivative)
                    .is_some()
                {
                    return Err(AcausalError::DuplicateEquation {
                        variable: lhs,
                        kind: "derivative",
                    });
                }
            }
            AcausalEquationKind::Assignment => {
                let Some(var) = vars_by_canonical.get(&lhs) else {
                    return Err(AcausalError::UnknownVariable {
                        name: lhs,
                        context: "assignment lhs".to_string(),
                    });
                };
                if matches!(
                    var.kind,
                    AcausalVariableKind::State
                        | AcausalVariableKind::Parameter
                        | AcausalVariableKind::Input
                ) {
                    return Err(AcausalError::InvalidSpec(format!(
                        "assignment lhs `{}` must be algebraic or output",
                        eq.lhs
                    )));
                }
                let parsed = parse_expr(&eq.rhs, &format!("assignment `{}`", eq.lhs))?;
                let expr = simplify(&rewrite_aliases(&parsed, &alias_map));
                let deps = ensure_known_expr_vars(
                    &expr,
                    &known_canonical,
                    &format!("assignment `{}`", eq.lhs),
                )?;
                assignments.push(CompiledAssignment {
                    lhs,
                    rhs: stringify(&expr),
                    expr,
                    dependencies: deps,
                });
            }
        }
    }

    let assignments = sort_assignments(assignments)?;
    let assigned: HashSet<String> = assignments.iter().map(|a| a.lhs.clone()).collect();
    for name in assignment_required {
        if !assigned.contains(&name) && !aliases_eliminated.iter().any(|a| a.variable == name) {
            return Err(AcausalError::MissingAssignment(name));
        }
    }
    for name in &state_names {
        if !derivative_by_state.contains_key(name) {
            return Err(AcausalError::MissingDerivative(name.clone()));
        }
    }

    let algebraic_lhs: HashSet<String> = assigned.clone();
    for derivative in derivative_by_state.values() {
        for dep in &derivative.dependencies {
            if vars_by_canonical.get(dep).is_some_and(|var| {
                matches!(
                    var.kind,
                    AcausalVariableKind::Algebraic | AcausalVariableKind::Output
                )
            }) && !algebraic_lhs.contains(dep)
            {
                return Err(AcausalError::MissingAssignment(dep.clone()));
            }
        }
    }

    let derivatives: Vec<CompiledDerivative> = state_names
        .iter()
        .filter_map(|name| derivative_by_state.get(name).cloned())
        .collect();
    let algebraic_order = assignments.iter().map(|a| a.lhs.clone()).collect();
    let derivative_order = derivatives.iter().map(|d| d.state.clone()).collect();
    let states = variables
        .iter()
        .filter(|v| v.kind == AcausalVariableKind::State)
        .count();
    let parameters = variables
        .iter()
        .filter(|v| {
            matches!(
                v.kind,
                AcausalVariableKind::Parameter | AcausalVariableKind::Input
            )
        })
        .count();
    let algebraics = variables.len().saturating_sub(states + parameters);

    if aliases_eliminated.is_empty() {
        warnings.push("no alias/connect equations were eliminated".to_string());
    }

    Ok(CompiledAcausalModel {
        name: spec.name.clone(),
        dt: spec.dt,
        steps: spec.steps,
        solver: spec.solver,
        variables,
        diagnostics: StructuralDiagnostics {
            variables: known_original.len(),
            states,
            algebraics,
            parameters,
            equations: spec.equations.len(),
            assignments: assignments.len(),
            derivatives: derivatives.len(),
            aliases_eliminated,
            algebraic_order,
            derivative_order,
            warnings,
        },
        state_names,
        parameter_values,
        initial_state,
        assignments,
        derivatives,
        canonical_by_original: alias_map,
        original_equations: spec.equations.clone(),
    })
}

#[derive(Clone, Debug)]
pub struct AcausalRun {
    pub times: Vec<f64>,
    pub series: BTreeMap<String, Vec<f64>>,
    pub frames: Vec<Value>,
    pub final_values: BTreeMap<String, f64>,
    pub diagnostics: StructuralDiagnostics,
}

impl AcausalRun {
    pub fn series(&self, name: &str) -> Option<&Vec<f64>> {
        self.series.get(name)
    }

    pub fn final_value(&self, name: &str) -> Option<f64> {
        self.final_values.get(name).copied()
    }

    pub fn to_artifact(&self, compiled: &CompiledAcausalModel) -> RunArtifact {
        let results = json!({
            "kind": "acausal",
            "schema": ACAUSAL_SCHEMA,
            "model": compiled.name,
            "solver": format!("{:?}", compiled.solver).to_lowercase(),
            "steps": compiled.steps,
            "dt": compiled.dt,
            "variables": compiled.variables.iter().map(|v| json!({
                "name": v.name,
                "kind": v.kind.as_str(),
                "initial": v.initial,
                "value": v.value,
                "unit": v.unit,
                "description": v.description,
            })).collect::<Vec<_>>(),
            "equations": compiled.original_equations.iter().map(|e| json!({
                "kind": format!("{:?}", e.kind).to_lowercase(),
                "lhs": e.lhs,
                "rhs": e.rhs,
                "label": e.label,
            })).collect::<Vec<_>>(),
            "compiled": {
                "assignments": compiled.assignments.iter().map(|a| json!({
                    "lhs": a.lhs,
                    "rhs": a.rhs,
                    "dependencies": a.dependencies,
                })).collect::<Vec<_>>(),
                "derivatives": compiled.derivatives.iter().map(|d| json!({
                    "state": d.state,
                    "rhs": d.rhs,
                    "dependencies": d.dependencies,
                })).collect::<Vec<_>>(),
                "canonicalByOriginal": compiled.canonical_by_original,
            },
            "structuralDiagnostics": self.diagnostics,
            "finalValues": self.final_values,
            "ui": acausal_workbench_descriptor(),
        });
        let summary = format!(
            "Acausal `{}`: {} state(s), {} algebraic/output variable(s), {} alias(es) eliminated.",
            compiled.name,
            compiled.diagnostics.states,
            compiled.diagnostics.algebraics,
            compiled.diagnostics.aliases_eliminated.len()
        );
        RunArtifact::sim(
            "acausal",
            &format!("Acausal Model: {}", compiled.name),
            "Equation-based model with structural diagnostics and generated simulation traces.",
            self.frames.clone(),
            results,
            vec![UiControl::range(
                "speed",
                "Speed (fps)",
                1.0,
                60.0,
                1.0,
                18.0,
            )],
            &summary,
        )
    }
}

fn eval_assignments(
    compiled: &CompiledAcausalModel,
    t: f64,
    states: &[f64],
) -> Result<Env, AcausalError> {
    let mut env = Env::new();
    env.insert("t".to_string(), t);
    for (name, value) in &compiled.parameter_values {
        env.insert(name.clone(), *value);
    }
    for (idx, name) in compiled.state_names.iter().enumerate() {
        env.insert(name.clone(), states.get(idx).copied().unwrap_or(0.0));
    }
    for assignment in &compiled.assignments {
        let value = evaluate_checked(&assignment.expr, &env, &assignment.lhs)?;
        env.insert(assignment.lhs.clone(), value);
    }
    Ok(env)
}

fn evaluate_checked(expr: &Expr, env: &Env, context: &str) -> Result<f64, AcausalError> {
    let value = catch_unwind(AssertUnwindSafe(|| evaluate(expr, env))).map_err(|e| {
        AcausalError::Run(format!(
            "failed to evaluate `{context}`: {}",
            panic_message(e)
        ))
    })?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(AcausalError::Run(format!(
            "equation `{context}` evaluated to a non-finite value"
        )))
    }
}

fn derivative_vector(
    compiled: &CompiledAcausalModel,
    t: f64,
    states: &[f64],
) -> Result<Vec<f64>, AcausalError> {
    let env = eval_assignments(compiled, t, states)?;
    compiled
        .derivatives
        .iter()
        .map(|d| evaluate_checked(&d.expr, &env, &d.state))
        .collect()
}

fn add_scaled(a: &[f64], scale: f64, b: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(x, dx)| x + scale * dx)
        .collect()
}

fn step_state(
    compiled: &CompiledAcausalModel,
    t: f64,
    y: &[f64],
) -> Result<Vec<f64>, AcausalError> {
    match compiled.solver {
        AcausalSolverKind::Euler => {
            let dy = derivative_vector(compiled, t, y)?;
            Ok(add_scaled(y, compiled.dt, &dy))
        }
        AcausalSolverKind::Rk4 => {
            let h = compiled.dt;
            let k1 = derivative_vector(compiled, t, y)?;
            let k2 = derivative_vector(compiled, t + 0.5 * h, &add_scaled(y, 0.5 * h, &k1))?;
            let k3 = derivative_vector(compiled, t + 0.5 * h, &add_scaled(y, 0.5 * h, &k2))?;
            let k4 = derivative_vector(compiled, t + h, &add_scaled(y, h, &k3))?;
            Ok(y.iter()
                .enumerate()
                .map(|(i, yi)| yi + h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
                .collect())
        }
    }
}

fn role_y(kind: AcausalVariableKind) -> f64 {
    match kind {
        AcausalVariableKind::Parameter | AcausalVariableKind::Input => 40.0,
        AcausalVariableKind::State => 145.0,
        AcausalVariableKind::Algebraic | AcausalVariableKind::Output => 250.0,
    }
}

fn role_fill(kind: AcausalVariableKind) -> &'static str {
    match kind {
        AcausalVariableKind::Parameter | AcausalVariableKind::Input => "#fef3c7",
        AcausalVariableKind::State => "#dbeafe",
        AcausalVariableKind::Algebraic => "#dcfce7",
        AcausalVariableKind::Output => "#fce7f3",
    }
}

fn build_frame(compiled: &CompiledAcausalModel, k: usize, t: f64, env: &Env) -> Value {
    let mut shapes = Vec::new();
    let mut counts: HashMap<&'static str, usize> = HashMap::new();
    let mut centers = HashMap::new();

    for var in &compiled.variables {
        let lane = var.kind.as_str();
        let idx = counts.entry(lane).or_insert(0);
        let x = 38.0 + (*idx as f64) * 150.0;
        *idx += 1;
        let y = role_y(var.kind);
        let w = 122.0;
        let h = 52.0;
        centers.insert(var.name.clone(), (x + w / 2.0, y + h / 2.0));
        let value = env.get(&var.name).copied().unwrap_or(0.0);
        shapes.push(json!({
            "kind": "rect", "x": x, "y": y, "w": w, "h": h, "rx": 8.0,
            "fill": role_fill(var.kind), "stroke": "#64748b", "strokeWidth": 1.3,
            "title": var.kind.as_str(),
        }));
        shapes.push(json!({
            "kind": "text", "x": x + w / 2.0, "y": y + 17.0,
            "text": var.name, "anchor": "middle", "fontSize": 12.0,
            "fontWeight": "bold", "fill": "#0f172a",
        }));
        shapes.push(json!({
            "kind": "text", "x": x + w / 2.0, "y": y + 36.0,
            "text": format!("{value:.4}"), "anchor": "middle", "fontSize": 11.0,
            "fill": "#1d4ed8",
        }));
    }

    for assignment in &compiled.assignments {
        if let Some(&(tx, ty)) = centers.get(&assignment.lhs) {
            for dep in &assignment.dependencies {
                if dep == "t" || dep == &assignment.lhs {
                    continue;
                }
                if let Some(&(sx, sy)) = centers.get(dep) {
                    shapes.push(json!({
                        "kind": "line", "x1": sx, "y1": sy, "x2": tx, "y2": ty,
                        "stroke": "#16a34a", "strokeWidth": 1.2, "opacity": 0.55,
                    }));
                }
            }
        }
    }
    for derivative in &compiled.derivatives {
        if let Some(&(tx, ty)) = centers.get(&derivative.state) {
            for dep in &derivative.dependencies {
                if dep == "t" || dep == &derivative.state {
                    continue;
                }
                if let Some(&(sx, sy)) = centers.get(dep) {
                    shapes.push(json!({
                        "kind": "line", "x1": sx, "y1": sy, "x2": tx, "y2": ty,
                        "stroke": "#2563eb", "strokeWidth": 1.5, "opacity": 0.65,
                    }));
                }
            }
        }
    }

    let mut frame = json!({
        "t": t,
        "step": k as f64,
        "caption": format!("t={t:.3}s"),
        "shapes": shapes,
    });
    if let Value::Object(map) = &mut frame {
        for var in &compiled.variables {
            if let Some(value) = env.get(&var.name) {
                map.insert(var.name.clone(), json!(value));
            }
        }
    }
    frame
}

/// Run a compiled acausal model with the solver selected in the spec.
pub fn simulate_acausal_model(compiled: &CompiledAcausalModel) -> Result<AcausalRun, AcausalError> {
    let mut y = compiled
        .state_names
        .iter()
        .map(|name| compiled.initial_state.get(name).copied().unwrap_or(0.0))
        .collect::<Vec<_>>();
    let mut times = Vec::with_capacity(compiled.steps);
    let mut series: BTreeMap<String, Vec<f64>> = compiled
        .variables
        .iter()
        .map(|v| (v.name.clone(), Vec::with_capacity(compiled.steps)))
        .collect();
    for alias in &compiled.diagnostics.aliases_eliminated {
        series
            .entry(alias.variable.clone())
            .or_insert_with(|| Vec::with_capacity(compiled.steps));
    }
    let mut frames = Vec::with_capacity(compiled.steps);

    for k in 0..compiled.steps {
        let t = k as f64 * compiled.dt;
        let env = eval_assignments(compiled, t, &y)?;
        times.push(t);
        for var in &compiled.variables {
            let value = env.get(&var.name).copied().unwrap_or(0.0);
            if let Some(xs) = series.get_mut(&var.name) {
                xs.push(value);
            }
        }
        for alias in &compiled.diagnostics.aliases_eliminated {
            let value = env.get(&alias.canonical).copied().unwrap_or(0.0);
            if let Some(xs) = series.get_mut(&alias.variable) {
                xs.push(value);
            }
        }
        let mut frame = build_frame(compiled, k, t, &env);
        if let Value::Object(map) = &mut frame {
            for alias in &compiled.diagnostics.aliases_eliminated {
                let value = env.get(&alias.canonical).copied().unwrap_or(0.0);
                map.insert(alias.variable.clone(), json!(value));
            }
        }
        frames.push(frame);
        if k + 1 < compiled.steps {
            y = step_state(compiled, t, &y)?;
        }
    }

    let final_values = series
        .iter()
        .filter_map(|(name, xs)| xs.last().copied().map(|v| (name.clone(), v)))
        .collect();

    Ok(AcausalRun {
        times,
        series,
        frames,
        final_values,
        diagnostics: compiled.diagnostics.clone(),
    })
}

/// Compile and run a spec in one step.
pub fn run_acausal_model(spec: &AcausalModelSpec) -> Result<AcausalRun, AcausalError> {
    let compiled = compile_acausal_model(spec)?;
    simulate_acausal_model(&compiled)
}

pub struct AcausalCitizen;

impl ModelCitizen for AcausalCitizen {
    fn descriptor(&self) -> ModelDescriptor {
        ModelDescriptor {
            kind: "acausal".to_string(),
            title: "Acausal Equation Model".to_string(),
            description: "JSON-first equation-based modeling with explicit ODEs, algebraic \
                          assignments, alias/connect elimination, structural diagnostics, \
                          RK4/Euler simulation, and UI workbench metadata."
                .to_string(),
            spec_schema: ACAUSAL_SCHEMA.to_string(),
            methods: vec![
                "simulate".to_string(),
                "structural-diagnostics".to_string(),
                "ui-metadata".to_string(),
            ],
            example_spec: serde_json::to_value(starter_acausal_model_spec())
                .unwrap_or_else(|_| json!({ "$schema": ACAUSAL_SCHEMA })),
        }
    }

    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError> {
        let spec = if spec
            .get("demo")
            .and_then(Value::as_str)
            .is_some_and(|demo| demo == "damped-mass-spring")
        {
            starter_acausal_model_spec()
        } else {
            serde_json::from_value::<AcausalModelSpec>(spec.clone()).map_err(|e| {
                CitizenError::InvalidSpec(format!("invalid acausal model spec: {e}"))
            })?
        };
        let compiled =
            compile_acausal_model(&spec).map_err(|e| CitizenError::InvalidSpec(e.to_string()))?;
        let run =
            simulate_acausal_model(&compiled).map_err(|e| CitizenError::Run(e.to_string()))?;
        Ok(run.to_artifact(&compiled))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn decay_spec() -> AcausalModelSpec {
        AcausalModelSpec {
            schema: Some(ACAUSAL_SCHEMA.to_string()),
            name: "decay".to_string(),
            dt: 0.1,
            steps: 11,
            solver: AcausalSolverKind::Rk4,
            variables: vec![
                AcausalVariableSpec::state("x", 1.0, "1"),
                AcausalVariableSpec::parameter("k", 1.0, "1/s"),
            ],
            equations: vec![AcausalEquationSpec::derivative("x", "-k*x")],
            metadata: Map::new(),
        }
    }

    #[test]
    fn starter_compiles_and_runs_with_structural_diagnostics() {
        let spec = starter_acausal_model_spec();
        let compiled = compile_acausal_model(&spec).expect("compile");
        assert_eq!(compiled.diagnostics.states, 2);
        assert_eq!(compiled.diagnostics.assignments, 2);
        assert_eq!(compiled.diagnostics.aliases_eliminated.len(), 1);
        assert_eq!(
            compiled.diagnostics.algebraic_order,
            vec!["spring_force", "damping_force"]
        );

        let run = simulate_acausal_model(&compiled).expect("simulate");
        assert_eq!(run.frames.len(), spec.steps);
        assert!(run.final_value("position").is_some());
        assert!(run.frames[0]["shapes"].is_array());
    }

    #[test]
    fn rk4_decay_tracks_the_analytic_solution() {
        let spec = decay_spec();
        let compiled = compile_acausal_model(&spec).expect("compile");
        let run = simulate_acausal_model(&compiled).expect("simulate");
        let final_x = run.final_value("x").unwrap();
        assert!(
            (final_x - std::f64::consts::E.powf(-1.0)).abs() < 2e-5,
            "final x {final_x}"
        );
    }

    #[test]
    fn algebraic_assignments_are_dependency_sorted() {
        let mut spec = decay_spec();
        spec.variables
            .push(AcausalVariableSpec::algebraic("y", "1"));
        spec.variables
            .push(AcausalVariableSpec::algebraic("z", "1"));
        spec.equations = vec![
            AcausalEquationSpec::assignment("z", "y + 1"),
            AcausalEquationSpec::assignment("y", "2 * x"),
            AcausalEquationSpec::derivative("x", "-z"),
        ];
        let compiled = compile_acausal_model(&spec).expect("compile");
        assert_eq!(compiled.diagnostics.algebraic_order, vec!["y", "z"]);
    }

    #[test]
    fn alias_equations_rewrite_rhs_expressions() {
        let mut spec = decay_spec();
        spec.variables.push(AcausalVariableSpec::output("out", "1"));
        spec.equations = vec![
            AcausalEquationSpec::derivative("x", "-k*out"),
            AcausalEquationSpec::alias("out", "x"),
        ];
        let compiled = compile_acausal_model(&spec).expect("compile");
        assert_eq!(compiled.diagnostics.aliases_eliminated.len(), 1);
        assert_eq!(compiled.derivatives[0].rhs, "-k * x");
    }

    #[test]
    fn algebraic_loops_are_rejected() {
        let mut spec = decay_spec();
        spec.variables
            .push(AcausalVariableSpec::algebraic("a", "1"));
        spec.variables
            .push(AcausalVariableSpec::algebraic("b", "1"));
        spec.equations = vec![
            AcausalEquationSpec::assignment("a", "b + 1"),
            AcausalEquationSpec::assignment("b", "a + 1"),
            AcausalEquationSpec::derivative("x", "-a"),
        ];
        assert!(matches!(
            compile_acausal_model(&spec),
            Err(AcausalError::AlgebraicLoop(nodes)) if nodes.len() == 2
        ));
    }

    #[test]
    fn workbench_descriptor_exposes_ui_palette_and_starter_spec() {
        let wb = acausal_workbench_descriptor();
        assert_eq!(wb.schema, ACAUSAL_SCHEMA);
        assert!(wb.tabs.contains(&"diagnostics".to_string()));
        assert!(wb.palette.iter().any(|p| p.kind == "alias"));
        assert_eq!(wb.starter.name, "damped-mass-spring");
    }

    #[test]
    fn citizen_runs_example_artifact() {
        let citizen = AcausalCitizen;
        let artifact = citizen
            .run_json(&citizen.descriptor().example_spec)
            .unwrap();
        assert_eq!(artifact.kind, "acausal");
        assert!(!artifact.frames.is_empty());
        assert!(artifact.results["structuralDiagnostics"].is_object());
        assert!(artifact.to_player_html().contains("Acausal Model"));
    }
}
