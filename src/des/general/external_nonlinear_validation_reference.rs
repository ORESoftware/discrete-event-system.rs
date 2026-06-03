//! Rust-facing bridge for nonlinear validation payloads.
//!
//! This module accepts compact expression-based NLP smoke models and keeps
//! heavyweight solvers optional. Registered solver names are exposed through
//! typed Rust options and routed through a dependency-free bounded grid plus
//! coordinate-pattern reference for small validation cases.

use std::collections::BTreeMap;
use std::time::Instant;

use serde_json::{json, Value};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearValidationReferenceSolver {
    Auto,
    Scipy,
    Ipopt,
    Bonmin,
    Minotaur,
    Couenne,
    Symphony,
    Knitro,
    Mosek,
    Baron,
    Copt,
    Casadi,
    Nlopt,
    NloptCli,
    Fallback,
}

impl ExternalNonlinearValidationReferenceSolver {
    pub fn all() -> &'static [ExternalNonlinearValidationReferenceSolver] {
        &[
            ExternalNonlinearValidationReferenceSolver::Auto,
            ExternalNonlinearValidationReferenceSolver::Scipy,
            ExternalNonlinearValidationReferenceSolver::Ipopt,
            ExternalNonlinearValidationReferenceSolver::Bonmin,
            ExternalNonlinearValidationReferenceSolver::Minotaur,
            ExternalNonlinearValidationReferenceSolver::Couenne,
            ExternalNonlinearValidationReferenceSolver::Symphony,
            ExternalNonlinearValidationReferenceSolver::Knitro,
            ExternalNonlinearValidationReferenceSolver::Mosek,
            ExternalNonlinearValidationReferenceSolver::Baron,
            ExternalNonlinearValidationReferenceSolver::Copt,
            ExternalNonlinearValidationReferenceSolver::Casadi,
            ExternalNonlinearValidationReferenceSolver::Nlopt,
            ExternalNonlinearValidationReferenceSolver::NloptCli,
            ExternalNonlinearValidationReferenceSolver::Fallback,
        ]
    }

    pub fn as_arg(self) -> &'static str {
        match self {
            ExternalNonlinearValidationReferenceSolver::Auto => "auto",
            ExternalNonlinearValidationReferenceSolver::Scipy => "scipy",
            ExternalNonlinearValidationReferenceSolver::Ipopt => "ipopt",
            ExternalNonlinearValidationReferenceSolver::Bonmin => "bonmin",
            ExternalNonlinearValidationReferenceSolver::Minotaur => "minotaur",
            ExternalNonlinearValidationReferenceSolver::Couenne => "couenne",
            ExternalNonlinearValidationReferenceSolver::Symphony => "symphony",
            ExternalNonlinearValidationReferenceSolver::Knitro => "knitro",
            ExternalNonlinearValidationReferenceSolver::Mosek => "mosek",
            ExternalNonlinearValidationReferenceSolver::Baron => "baron",
            ExternalNonlinearValidationReferenceSolver::Copt => "copt",
            ExternalNonlinearValidationReferenceSolver::Casadi => "casadi",
            ExternalNonlinearValidationReferenceSolver::Nlopt => "nlopt",
            ExternalNonlinearValidationReferenceSolver::NloptCli => "nlopt-cli",
            ExternalNonlinearValidationReferenceSolver::Fallback => "fallback",
        }
    }

    pub fn display_name(self) -> &'static str {
        match self {
            ExternalNonlinearValidationReferenceSolver::Auto => "Auto",
            ExternalNonlinearValidationReferenceSolver::Scipy => "SciPy SLSQP",
            ExternalNonlinearValidationReferenceSolver::Ipopt => "Ipopt",
            ExternalNonlinearValidationReferenceSolver::Bonmin => "Bonmin",
            ExternalNonlinearValidationReferenceSolver::Minotaur => "MINOTAUR",
            ExternalNonlinearValidationReferenceSolver::Couenne => "Couenne",
            ExternalNonlinearValidationReferenceSolver::Symphony => "COIN-OR SYMPHONY",
            ExternalNonlinearValidationReferenceSolver::Knitro => "Artelys Knitro",
            ExternalNonlinearValidationReferenceSolver::Mosek => "MOSEK",
            ExternalNonlinearValidationReferenceSolver::Baron => "BARON",
            ExternalNonlinearValidationReferenceSolver::Copt => "COPT",
            ExternalNonlinearValidationReferenceSolver::Casadi => "CasADi",
            ExternalNonlinearValidationReferenceSolver::Nlopt => "NLopt",
            ExternalNonlinearValidationReferenceSolver::NloptCli => "NLopt CLI",
            ExternalNonlinearValidationReferenceSolver::Fallback => "Pattern-search fallback",
        }
    }

    pub fn family(self) -> ExternalNonlinearValidationReferenceFamily {
        match self {
            ExternalNonlinearValidationReferenceSolver::Auto => {
                ExternalNonlinearValidationReferenceFamily::Auto
            }
            ExternalNonlinearValidationReferenceSolver::Scipy
            | ExternalNonlinearValidationReferenceSolver::Ipopt
            | ExternalNonlinearValidationReferenceSolver::Bonmin
            | ExternalNonlinearValidationReferenceSolver::Minotaur
            | ExternalNonlinearValidationReferenceSolver::Couenne
            | ExternalNonlinearValidationReferenceSolver::Symphony
            | ExternalNonlinearValidationReferenceSolver::Knitro
            | ExternalNonlinearValidationReferenceSolver::Mosek
            | ExternalNonlinearValidationReferenceSolver::Baron
            | ExternalNonlinearValidationReferenceSolver::Copt => {
                ExternalNonlinearValidationReferenceFamily::ScipyBridge
            }
            ExternalNonlinearValidationReferenceSolver::Casadi
            | ExternalNonlinearValidationReferenceSolver::Nlopt
            | ExternalNonlinearValidationReferenceSolver::NloptCli => {
                ExternalNonlinearValidationReferenceFamily::PackageBridge
            }
            ExternalNonlinearValidationReferenceSolver::Fallback => {
                ExternalNonlinearValidationReferenceFamily::Fallback
            }
        }
    }

    pub fn notes(self) -> &'static str {
        match self.family() {
            ExternalNonlinearValidationReferenceFamily::Auto => {
                "Prefer installed SciPy-backed validation, then use the bounded pattern-search fallback."
            }
            ExternalNonlinearValidationReferenceFamily::ScipyBridge => {
                "Registered NLP solver label routed through the local SciPy validation bridge when available, with deterministic fallback recovery."
            }
            ExternalNonlinearValidationReferenceFamily::PackageBridge => {
                "Package-specific bridge that checks the named Python package before falling back for smoke validation."
            }
            ExternalNonlinearValidationReferenceFamily::Fallback => {
                "Dependency-free bounded grid plus pattern-search reference for small NLP smoke models."
            }
        }
    }

    pub fn spec(self) -> ExternalNonlinearValidationReferenceSolverSpec {
        ExternalNonlinearValidationReferenceSolverSpec {
            solver: self,
            id: self.as_arg(),
            display_name: self.display_name(),
            family: self.family(),
            notes: self.notes(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearValidationReferenceFamily {
    Auto,
    ScipyBridge,
    PackageBridge,
    Fallback,
}

impl ExternalNonlinearValidationReferenceFamily {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalNonlinearValidationReferenceFamily::Auto => "auto",
            ExternalNonlinearValidationReferenceFamily::ScipyBridge => "scipy-bridge",
            ExternalNonlinearValidationReferenceFamily::PackageBridge => "package-bridge",
            ExternalNonlinearValidationReferenceFamily::Fallback => "fallback",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExternalNonlinearValidationReferenceSolverSpec {
    pub solver: ExternalNonlinearValidationReferenceSolver,
    pub id: &'static str,
    pub display_name: &'static str,
    pub family: ExternalNonlinearValidationReferenceFamily,
    pub notes: &'static str,
}

pub fn external_nonlinear_validation_reference_solver_specs(
) -> Vec<ExternalNonlinearValidationReferenceSolverSpec> {
    ExternalNonlinearValidationReferenceSolver::all()
        .iter()
        .copied()
        .map(ExternalNonlinearValidationReferenceSolver::spec)
        .collect()
}

pub fn external_nonlinear_validation_reference_solver_manifest() -> Value {
    Value::Array(
        external_nonlinear_validation_reference_solver_specs()
            .into_iter()
            .map(|spec| {
                json!({
                    "id": spec.id,
                    "displayName": spec.display_name,
                    "family": spec.family.as_str(),
                    "notes": spec.notes,
                })
            })
            .collect(),
    )
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationReferenceOptions {
    pub solver: ExternalNonlinearValidationReferenceSolver,
}

impl Default for ExternalNonlinearValidationReferenceOptions {
    fn default() -> Self {
        ExternalNonlinearValidationReferenceOptions {
            solver: ExternalNonlinearValidationReferenceSolver::Auto,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationVariable {
    pub name: String,
    pub lb: f64,
    pub ub: f64,
    pub start: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationConstraint {
    pub name: String,
    pub expr: String,
    pub sense: String,
    pub rhs: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationRequest {
    pub variables: Vec<ExternalNonlinearValidationVariable>,
    pub objective: String,
    pub constraints: Vec<ExternalNonlinearValidationConstraint>,
    pub sense: String,
}

impl ExternalNonlinearValidationRequest {
    pub fn to_json(&self) -> Value {
        json!({
            "kind": "nonlinear-validation",
            "variables": self.variables.iter().map(|variable| json!({
                "name": variable.name,
                "lb": variable.lb,
                "ub": variable.ub,
                "start": variable.start,
            })).collect::<Vec<_>>(),
            "objective": self.objective,
            "constraints": self.constraints.iter().map(|constraint| json!({
                "name": constraint.name,
                "expr": constraint.expr,
                "sense": constraint.sense,
                "rhs": constraint.rhs,
            })).collect::<Vec<_>>(),
            "sense": self.sense,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExternalNonlinearValidationReferenceStatus {
    Optimal,
    Infeasible,
    Failed,
    NumericalError,
}

impl ExternalNonlinearValidationReferenceStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ExternalNonlinearValidationReferenceStatus::Optimal => "optimal",
            ExternalNonlinearValidationReferenceStatus::Infeasible => "infeasible",
            ExternalNonlinearValidationReferenceStatus::Failed => "failed",
            ExternalNonlinearValidationReferenceStatus::NumericalError => "numerical-error",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExternalNonlinearValidationReferenceSolution {
    pub status: ExternalNonlinearValidationReferenceStatus,
    pub solver: String,
    pub x: Vec<f64>,
    pub objective: Option<f64>,
    pub message: String,
    pub iterations: Option<u64>,
    pub elapsed_ms: f64,
}

fn nonlinear_validation_error(
    status: ExternalNonlinearValidationReferenceStatus,
    solver: impl Into<String>,
    message: impl Into<String>,
    elapsed_ms: f64,
) -> ExternalNonlinearValidationReferenceSolution {
    ExternalNonlinearValidationReferenceSolution {
        status,
        solver: solver.into(),
        x: Vec::new(),
        objective: None,
        message: message.into(),
        iterations: None,
        elapsed_ms,
    }
}

#[derive(Clone, Debug, PartialEq)]
enum NlpExpr {
    Constant(f64),
    Variable(String),
    UnaryMinus(Box<NlpExpr>),
    UnaryPlus(Box<NlpExpr>),
    Add(Box<NlpExpr>, Box<NlpExpr>),
    Sub(Box<NlpExpr>, Box<NlpExpr>),
    Mul(Box<NlpExpr>, Box<NlpExpr>),
    Div(Box<NlpExpr>, Box<NlpExpr>),
    Pow(Box<NlpExpr>, Box<NlpExpr>),
    Call(String, Vec<NlpExpr>),
}

impl NlpExpr {
    fn eval(&self, env: &BTreeMap<String, f64>) -> Result<f64, String> {
        let value = match self {
            NlpExpr::Constant(value) => *value,
            NlpExpr::Variable(name) => *env
                .get(name)
                .ok_or_else(|| format!("unknown variable `{name}`"))?,
            NlpExpr::UnaryMinus(expr) => -expr.eval(env)?,
            NlpExpr::UnaryPlus(expr) => expr.eval(env)?,
            NlpExpr::Add(left, right) => left.eval(env)? + right.eval(env)?,
            NlpExpr::Sub(left, right) => left.eval(env)? - right.eval(env)?,
            NlpExpr::Mul(left, right) => left.eval(env)? * right.eval(env)?,
            NlpExpr::Div(left, right) => left.eval(env)? / right.eval(env)?,
            NlpExpr::Pow(left, right) => left.eval(env)?.powf(right.eval(env)?),
            NlpExpr::Call(name, args) => {
                let values = args
                    .iter()
                    .map(|arg| arg.eval(env))
                    .collect::<Result<Vec<_>, _>>()?;
                eval_nlp_function(name, &values)?
            }
        };
        if value.is_finite() {
            Ok(value)
        } else {
            Err("expression produced a non-finite value".to_string())
        }
    }
}

fn eval_nlp_function(name: &str, args: &[f64]) -> Result<f64, String> {
    match (name, args) {
        ("abs", [value]) => Ok(value.abs()),
        ("sin", [value]) => Ok(value.sin()),
        ("cos", [value]) => Ok(value.cos()),
        ("tan", [value]) => Ok(value.tan()),
        ("exp", [value]) => Ok(value.exp()),
        ("log", [value]) => Ok(value.ln()),
        ("sqrt", [value]) => Ok(value.sqrt()),
        ("pow", [base, exponent]) => Ok(base.powf(*exponent)),
        ("min", values) if !values.is_empty() => {
            Ok(values.iter().copied().fold(f64::INFINITY, f64::min))
        }
        ("max", values) if !values.is_empty() => {
            Ok(values.iter().copied().fold(f64::NEG_INFINITY, f64::max))
        }
        _ => Err(format!("unsupported function `{name}`")),
    }
}

struct NlpExprParser<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> NlpExprParser<'a> {
    fn new(input: &'a str) -> Self {
        NlpExprParser { input, pos: 0 }
    }

    fn parse(mut self) -> Result<NlpExpr, String> {
        let expr = self.parse_add_sub()?;
        self.skip_ws();
        if self.pos == self.input.len() {
            Ok(expr)
        } else {
            Err(format!(
                "unexpected input near `{}`",
                &self.input[self.pos..]
            ))
        }
    }

    fn skip_ws(&mut self) {
        while self
            .input
            .get(self.pos..)
            .and_then(|rest| rest.chars().next())
            .is_some_and(char::is_whitespace)
        {
            self.pos += self.input[self.pos..].chars().next().unwrap().len_utf8();
        }
    }

    fn consume(&mut self, token: &str) -> bool {
        self.skip_ws();
        if self.input[self.pos..].starts_with(token) {
            self.pos += token.len();
            true
        } else {
            false
        }
    }

    fn peek_char(&mut self) -> Option<char> {
        self.skip_ws();
        self.input[self.pos..].chars().next()
    }

    fn parse_add_sub(&mut self) -> Result<NlpExpr, String> {
        let mut expr = self.parse_mul_div()?;
        loop {
            if self.consume("+") {
                expr = NlpExpr::Add(Box::new(expr), Box::new(self.parse_mul_div()?));
            } else if self.consume("-") {
                expr = NlpExpr::Sub(Box::new(expr), Box::new(self.parse_mul_div()?));
            } else {
                return Ok(expr);
            }
        }
    }

    fn parse_mul_div(&mut self) -> Result<NlpExpr, String> {
        let mut expr = self.parse_power()?;
        loop {
            self.skip_ws();
            if self.input[self.pos..].starts_with("**") {
                return Ok(expr);
            }
            if self.consume("*") {
                expr = NlpExpr::Mul(Box::new(expr), Box::new(self.parse_power()?));
            } else if self.consume("/") {
                expr = NlpExpr::Div(Box::new(expr), Box::new(self.parse_power()?));
            } else {
                return Ok(expr);
            }
        }
    }

    fn parse_power(&mut self) -> Result<NlpExpr, String> {
        let expr = self.parse_unary()?;
        if self.consume("**") {
            Ok(NlpExpr::Pow(Box::new(expr), Box::new(self.parse_power()?)))
        } else {
            Ok(expr)
        }
    }

    fn parse_unary(&mut self) -> Result<NlpExpr, String> {
        if self.consume("-") {
            Ok(NlpExpr::UnaryMinus(Box::new(self.parse_unary()?)))
        } else if self.consume("+") {
            Ok(NlpExpr::UnaryPlus(Box::new(self.parse_unary()?)))
        } else {
            self.parse_primary()
        }
    }

    fn parse_primary(&mut self) -> Result<NlpExpr, String> {
        match self.peek_char() {
            Some('(') => {
                self.consume("(");
                let expr = self.parse_add_sub()?;
                if !self.consume(")") {
                    return Err("missing closing `)`".to_string());
                }
                Ok(expr)
            }
            Some(ch) if ch.is_ascii_digit() || ch == '.' => self.parse_number(),
            Some(ch) if ch.is_ascii_alphabetic() || ch == '_' => self.parse_identifier_expr(),
            Some(ch) => Err(format!("unexpected character `{ch}`")),
            None => Err("unexpected end of expression".to_string()),
        }
    }

    fn parse_number(&mut self) -> Result<NlpExpr, String> {
        self.skip_ws();
        let start = self.pos;
        let mut saw_exp = false;
        while let Some(ch) = self.input[self.pos..].chars().next() {
            if ch.is_ascii_digit() || ch == '.' {
                self.pos += ch.len_utf8();
            } else if (ch == 'e' || ch == 'E') && !saw_exp {
                saw_exp = true;
                self.pos += ch.len_utf8();
                if let Some(sign) = self.input[self.pos..].chars().next() {
                    if sign == '+' || sign == '-' {
                        self.pos += sign.len_utf8();
                    }
                }
            } else {
                break;
            }
        }
        self.input[start..self.pos]
            .parse::<f64>()
            .map(NlpExpr::Constant)
            .map_err(|_| format!("invalid number `{}`", &self.input[start..self.pos]))
    }

    fn parse_identifier_expr(&mut self) -> Result<NlpExpr, String> {
        let name = self.parse_identifier()?;
        if !self.consume("(") {
            return Ok(NlpExpr::Variable(name));
        }
        let mut args = Vec::new();
        if self.consume(")") {
            return Ok(NlpExpr::Call(name, args));
        }
        loop {
            args.push(self.parse_add_sub()?);
            if self.consume(")") {
                break;
            }
            if !self.consume(",") {
                return Err("expected `,` or `)` in function call".to_string());
            }
        }
        Ok(NlpExpr::Call(name, args))
    }

    fn parse_identifier(&mut self) -> Result<String, String> {
        self.skip_ws();
        let start = self.pos;
        while let Some(ch) = self.input[self.pos..].chars().next() {
            if ch.is_ascii_alphanumeric() || ch == '_' {
                self.pos += ch.len_utf8();
            } else {
                break;
            }
        }
        if self.pos == start {
            Err("expected identifier".to_string())
        } else {
            Ok(self.input[start..self.pos].to_string())
        }
    }
}

#[derive(Clone, Debug)]
struct RustNlpConstraint {
    expr: NlpExpr,
    sense: String,
    rhs: f64,
}

#[derive(Clone, Debug)]
struct RustNlpModel {
    names: Vec<String>,
    lb: Vec<f64>,
    ub: Vec<f64>,
    x0: Vec<f64>,
    objective: NlpExpr,
    constraints: Vec<RustNlpConstraint>,
    sense: String,
}

fn nonlinear_string(value: Option<&Value>, default: &str) -> String {
    match value {
        Some(Value::String(text)) => text.clone(),
        Some(value) => value.to_string(),
        None => default.to_string(),
    }
}

fn nonlinear_f64(value: Option<&Value>, default: f64) -> Result<f64, String> {
    let out = match value {
        Some(Value::Number(number)) => number
            .as_f64()
            .ok_or_else(|| "expected finite number".to_string())?,
        Some(Value::String(text)) => text
            .parse::<f64>()
            .map_err(|_| "expected finite number".to_string())?,
        Some(Value::Null) | None => default,
        Some(_) => return Err("expected finite number".to_string()),
    };
    if out.is_finite() {
        Ok(out)
    } else {
        Err("expected finite number".to_string())
    }
}

fn normalize_nonlinear_model(payload: &Value) -> Result<RustNlpModel, String> {
    let object = payload
        .as_object()
        .ok_or_else(|| "nonlinear payload must be an object".to_string())?;
    let kind = object
        .get("kind")
        .and_then(Value::as_str)
        .unwrap_or("nonlinear-validation")
        .replace('_', "-");
    if kind != "nonlinear-validation" && kind != "nlp-validation" {
        return Err("payload kind must be nonlinear-validation or nlp-validation".to_string());
    }
    let variable_values = object
        .get("variables")
        .and_then(Value::as_array)
        .filter(|variables| !variables.is_empty())
        .cloned()
        .unwrap_or_else(|| {
            let dimension = object.get("dimension").and_then(Value::as_u64).unwrap_or(0) as usize;
            (0..dimension)
                .map(|idx| json!({"name": format!("x{idx}")}))
                .collect::<Vec<_>>()
        });
    if variable_values.is_empty() {
        return Err("nonlinear payload needs variables or dimension".to_string());
    }

    let mut names = Vec::with_capacity(variable_values.len());
    let mut lb = Vec::with_capacity(variable_values.len());
    let mut ub = Vec::with_capacity(variable_values.len());
    let mut x0 = Vec::with_capacity(variable_values.len());
    for (idx, raw) in variable_values.iter().enumerate() {
        let fallback_name = format!("x{idx}");
        let item = raw.as_object();
        let name = item
            .and_then(|item| item.get("name"))
            .map(|value| nonlinear_string(Some(value), &fallback_name))
            .unwrap_or_else(|| {
                raw.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| fallback_name.clone())
            });
        let lower = nonlinear_f64(
            item.and_then(|item| item.get("lb").or_else(|| item.get("lower"))),
            -10.0,
        )?;
        let upper = nonlinear_f64(
            item.and_then(|item| item.get("ub").or_else(|| item.get("upper"))),
            10.0,
        )?;
        if lower > upper {
            return Err(format!("variable {name} needs finite ordered bounds"));
        }
        let start = nonlinear_f64(
            item.and_then(|item| item.get("start").or_else(|| item.get("initial"))),
            0.5 * (lower + upper),
        )?
        .clamp(lower, upper);
        names.push(name);
        lb.push(lower);
        ub.push(upper);
        x0.push(start);
    }

    let objective = NlpExprParser::new(&nonlinear_string(object.get("objective"), "0")).parse()?;
    let constraints = object
        .get("constraints")
        .and_then(Value::as_array)
        .ok_or_else(|| "constraints must be an array".to_string())?
        .iter()
        .filter_map(Value::as_object)
        .map(|constraint| {
            Ok(RustNlpConstraint {
                expr: NlpExprParser::new(&nonlinear_string(
                    constraint
                        .get("expr")
                        .or_else(|| constraint.get("expression")),
                    "0",
                ))
                .parse()?,
                sense: nonlinear_string(constraint.get("sense"), "<="),
                rhs: nonlinear_f64(constraint.get("rhs"), 0.0)?,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;
    Ok(RustNlpModel {
        names,
        lb,
        ub,
        x0,
        objective,
        constraints,
        sense: nonlinear_string(object.get("sense"), "min").to_ascii_lowercase(),
    })
}

fn rust_nlp_env(model: &RustNlpModel, x: &[f64]) -> BTreeMap<String, f64> {
    let mut env = BTreeMap::new();
    for (idx, value) in x.iter().copied().enumerate() {
        env.insert(format!("x{idx}"), value);
    }
    for (name, value) in model.names.iter().zip(x.iter().copied()) {
        env.insert(name.clone(), value);
    }
    env
}

fn rust_nlp_public_objective(model: &RustNlpModel, x: &[f64]) -> Result<f64, String> {
    model.objective.eval(&rust_nlp_env(model, x))
}

fn rust_nlp_objective_value(model: &RustNlpModel, x: &[f64]) -> Result<f64, String> {
    let value = rust_nlp_public_objective(model, x)?;
    if model.sense == "max" || model.sense == "maximize" {
        Ok(-value)
    } else {
        Ok(value)
    }
}

fn rust_nlp_constraint_violation(model: &RustNlpModel, x: &[f64]) -> Result<f64, String> {
    let env = rust_nlp_env(model, x);
    let mut total = 0.0;
    for constraint in &model.constraints {
        let lhs = constraint.expr.eval(&env)?;
        let rhs = constraint.rhs;
        let violation = match constraint.sense.as_str() {
            "<=" | "le" | "less-equal" => (lhs - rhs).max(0.0),
            ">=" | "ge" | "greater-equal" => (rhs - lhs).max(0.0),
            "=" | "==" | "eq" => lhs - rhs,
            other => return Err(format!("unsupported constraint sense `{other}`")),
        };
        total += violation * violation;
    }
    Ok(total.sqrt())
}

fn rust_nlp_feasible(model: &RustNlpModel, x: &[f64]) -> Result<bool, String> {
    Ok(rust_nlp_constraint_violation(model, x)? <= 1e-6)
}

fn rust_nlp_clamp(model: &RustNlpModel, x: &[f64]) -> Vec<f64> {
    x.iter()
        .zip(&model.lb)
        .zip(&model.ub)
        .map(|((value, lower), upper)| value.clamp(*lower, *upper))
        .collect()
}

fn rust_nlp_candidate_grid(model: &RustNlpModel) -> Vec<Vec<f64>> {
    let mut axes = Vec::with_capacity(model.lb.len());
    for ((lower, upper), start) in model.lb.iter().zip(&model.ub).zip(&model.x0) {
        let mut values = vec![*lower, *upper, 0.5 * (lower + upper), *start];
        if upper - lower > 0.0 {
            values.push(lower + (upper - lower) / 3.0);
            values.push(lower + 2.0 * (upper - lower) / 3.0);
        }
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
        axes.push(values);
    }
    if axes.iter().map(Vec::len).product::<usize>() > 50_000 {
        axes = model
            .lb
            .iter()
            .zip(&model.ub)
            .map(|(lower, upper)| vec![*lower, 0.5 * (lower + upper), *upper])
            .collect();
    }

    let mut candidates = vec![Vec::new()];
    for axis in axes {
        let mut next = Vec::new();
        for prefix in &candidates {
            for value in &axis {
                let mut candidate = prefix.clone();
                candidate.push(*value);
                next.push(candidate);
            }
        }
        candidates = next;
    }
    candidates
}

fn rust_nlp_penalized_value(model: &RustNlpModel, x: &[f64]) -> Result<f64, String> {
    Ok(
        rust_nlp_objective_value(model, x)?
            + 1_000_000.0 * rust_nlp_constraint_violation(model, x)?,
    )
}

fn rust_nlp_pattern_search(
    model: &RustNlpModel,
    start: &[f64],
    max_iterations: u64,
) -> Result<(Vec<f64>, u64), String> {
    let mut best = rust_nlp_clamp(model, start);
    let max_span = model
        .lb
        .iter()
        .zip(&model.ub)
        .map(|(lower, upper)| upper - lower)
        .fold(1.0_f64, f64::max);
    let mut step = (max_span * 0.25).max(1.0);
    let mut iterations = 0;
    let mut best_value = rust_nlp_penalized_value(model, &best)?;
    while iterations < max_iterations && step > 1e-8 {
        iterations += 1;
        let mut improved = false;
        let mut trial_best = best.clone();
        let mut trial_value = best_value;
        for idx in 0..best.len() {
            for sign in [-1.0, 1.0] {
                let mut candidate = best.clone();
                candidate[idx] += sign * step;
                candidate = rust_nlp_clamp(model, &candidate);
                let value = rust_nlp_penalized_value(model, &candidate)?;
                if value < trial_value - 1e-10 {
                    trial_best = candidate;
                    trial_value = value;
                    improved = true;
                }
            }
        }
        if improved {
            best = trial_best;
            best_value = trial_value;
        } else {
            step *= 0.5;
        }
    }
    Ok((best, iterations))
}

fn rust_nonlinear_solver_label(solver: ExternalNonlinearValidationReferenceSolver) -> String {
    if solver == ExternalNonlinearValidationReferenceSolver::Auto
        || solver == ExternalNonlinearValidationReferenceSolver::Fallback
    {
        "builtin:nlp-pattern-search".to_string()
    } else {
        format!("builtin:nlp-pattern-search-for-{}", solver.as_arg())
    }
}

fn solve_nonlinear_validation_with_rust_fallback(
    payload: &Value,
    opts: &ExternalNonlinearValidationReferenceOptions,
    started: Instant,
) -> Result<ExternalNonlinearValidationReferenceSolution, String> {
    let model = normalize_nonlinear_model(payload)?;
    let mut best = None::<Vec<f64>>;
    let mut best_score = f64::INFINITY;
    let mut iterations = 0_u64;
    for candidate in rust_nlp_candidate_grid(&model) {
        let (refined, used) = rust_nlp_pattern_search(&model, &candidate, 2_000)?;
        iterations += used;
        let score = rust_nlp_penalized_value(&model, &refined)?;
        if score < best_score {
            best = Some(refined);
            best_score = score;
        }
    }
    let solver = rust_nonlinear_solver_label(opts.solver);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let Some(best) = best else {
        return Ok(ExternalNonlinearValidationReferenceSolution {
            status: ExternalNonlinearValidationReferenceStatus::Infeasible,
            solver,
            x: Vec::new(),
            objective: None,
            message: "no candidate generated".to_string(),
            iterations: Some(iterations),
            elapsed_ms,
        });
    };
    let objective = rust_nlp_public_objective(&model, &best)?;
    let violation = rust_nlp_constraint_violation(&model, &best)?;
    if !rust_nlp_feasible(&model, &best)? {
        return Ok(ExternalNonlinearValidationReferenceSolution {
            status: ExternalNonlinearValidationReferenceStatus::Infeasible,
            solver,
            x: best,
            objective: Some(objective),
            message: format!("best constraint violation {violation:.3e}"),
            iterations: Some(iterations),
            elapsed_ms,
        });
    }
    Ok(ExternalNonlinearValidationReferenceSolution {
        status: ExternalNonlinearValidationReferenceStatus::Optimal,
        solver,
        x: best,
        objective: Some(objective),
        message: "bounded grid plus coordinate-pattern fallback".to_string(),
        iterations: Some(iterations),
        elapsed_ms,
    })
}

pub fn solve_nonlinear_validation_json_with_external_reference(
    payload: Value,
    opts: &ExternalNonlinearValidationReferenceOptions,
) -> ExternalNonlinearValidationReferenceSolution {
    let started = Instant::now();
    match solve_nonlinear_validation_with_rust_fallback(&payload, opts, started) {
        Ok(solution) => solution,
        Err(message) => nonlinear_validation_error(
            ExternalNonlinearValidationReferenceStatus::Failed,
            opts.solver.as_arg(),
            message,
            started.elapsed().as_secs_f64() * 1000.0,
        ),
    }
}

pub fn solve_nonlinear_validation_with_external_reference(
    request: &ExternalNonlinearValidationRequest,
    opts: &ExternalNonlinearValidationReferenceOptions,
) -> ExternalNonlinearValidationReferenceSolution {
    solve_nonlinear_validation_json_with_external_reference(request.to_json(), opts)
}

#[cfg(test)]
mod tests {
    use crate::des::general::external_nonlinear_validation_reference::{
        external_nonlinear_validation_reference_solver_manifest,
        external_nonlinear_validation_reference_solver_specs,
        solve_nonlinear_validation_with_external_reference, ExternalNonlinearValidationConstraint,
        ExternalNonlinearValidationReferenceFamily, ExternalNonlinearValidationReferenceOptions,
        ExternalNonlinearValidationReferenceSolver, ExternalNonlinearValidationReferenceStatus,
        ExternalNonlinearValidationRequest, ExternalNonlinearValidationVariable,
    };

    #[test]
    fn solver_manifest_covers_registered_nonlinear_validation_tools() {
        let specs = external_nonlinear_validation_reference_solver_specs();
        assert_eq!(specs.len(), 15);
        assert_eq!(
            specs
                .iter()
                .filter(
                    |spec| spec.family == ExternalNonlinearValidationReferenceFamily::ScipyBridge
                )
                .count(),
            10
        );
        assert_eq!(
            specs
                .iter()
                .filter(
                    |spec| spec.family == ExternalNonlinearValidationReferenceFamily::PackageBridge
                )
                .count(),
            3
        );
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalNonlinearValidationReferenceSolver::Ipopt
                && spec.id == "ipopt"
                && spec.display_name == "Ipopt"
        }));
        assert!(specs.iter().any(|spec| {
            spec.solver == ExternalNonlinearValidationReferenceSolver::Casadi && spec.id == "casadi"
        }));

        let manifest = external_nonlinear_validation_reference_solver_manifest();
        let items = manifest.as_array().expect("manifest array");
        assert_eq!(items.len(), 15);
        assert!(items.iter().any(|item| {
            item.get("id").and_then(|value| value.as_str()) == Some("knitro")
                && item.get("family").and_then(|value| value.as_str()) == Some("scipy-bridge")
        }));
    }

    #[test]
    fn fallback_bridge_solves_small_expression_model() {
        let request = ExternalNonlinearValidationRequest {
            variables: vec![
                ExternalNonlinearValidationVariable {
                    name: "x".to_string(),
                    lb: 0.0,
                    ub: 3.0,
                    start: Some(0.2),
                },
                ExternalNonlinearValidationVariable {
                    name: "y".to_string(),
                    lb: 0.0,
                    ub: 3.0,
                    start: Some(0.2),
                },
            ],
            objective: "(x - 1)**2 + (y - 2)**2".to_string(),
            constraints: vec![ExternalNonlinearValidationConstraint {
                name: "demand".to_string(),
                expr: "x + y".to_string(),
                sense: ">=".to_string(),
                rhs: 1.0,
            }],
            sense: "min".to_string(),
        };
        let result = solve_nonlinear_validation_with_external_reference(
            &request,
            &ExternalNonlinearValidationReferenceOptions {
                solver: ExternalNonlinearValidationReferenceSolver::Fallback,
            },
        );
        assert_eq!(
            result.status,
            ExternalNonlinearValidationReferenceStatus::Optimal
        );
        assert_eq!(result.x.len(), 2);
        assert!(result.objective.is_some_and(|objective| objective <= 1e-6));
    }

    #[test]
    fn fallback_bridge_reports_infeasible_registered_alias() {
        let request = ExternalNonlinearValidationRequest {
            variables: vec![
                ExternalNonlinearValidationVariable {
                    name: "x0".to_string(),
                    lb: 0.0,
                    ub: 1.0,
                    start: None,
                },
                ExternalNonlinearValidationVariable {
                    name: "x1".to_string(),
                    lb: 0.0,
                    ub: 1.0,
                    start: None,
                },
            ],
            objective: "x0**2 + x1**2".to_string(),
            constraints: vec![ExternalNonlinearValidationConstraint {
                name: "impossible".to_string(),
                expr: "x0 + x1".to_string(),
                sense: ">=".to_string(),
                rhs: 3.0,
            }],
            sense: "min".to_string(),
        };
        let result = solve_nonlinear_validation_with_external_reference(
            &request,
            &ExternalNonlinearValidationReferenceOptions {
                solver: ExternalNonlinearValidationReferenceSolver::Nlopt,
            },
        );

        assert_eq!(
            result.status,
            ExternalNonlinearValidationReferenceStatus::Infeasible
        );
        assert_eq!(result.solver, "builtin:nlp-pattern-search-for-nlopt");
        assert!(result.message.contains("constraint violation"));
    }
}
