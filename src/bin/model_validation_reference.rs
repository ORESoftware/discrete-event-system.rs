use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fmt;
use std::io::{self, Read};

use serde_json::{json, Value};

#[derive(Debug)]
struct CliError(String);

impl fmt::Display for CliError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl Error for CliError {}

fn usage(program: &str) -> String {
    format!("usage: {program} [--tool TOOL]")
}

fn normalize_tool_id(tool: Option<&str>) -> String {
    tool.unwrap_or("auto")
        .trim()
        .to_ascii_lowercase()
        .replace('_', "-")
}

fn result(
    status: &str,
    verdict: &str,
    validator: &str,
    message: impl Into<String>,
    stdout: impl Into<String>,
    stderr: impl Into<String>,
) -> Value {
    json!({
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message.into(),
        "stdout": stdout.into(),
        "stderr": stderr.into(),
    })
}

fn empty_error(message: impl Into<String>) -> Value {
    result(
        "failed",
        "failure",
        "rust:model-validation-reference",
        message,
        "",
        "",
    )
}

fn payload_text(payload: &Value, keys: &[&str]) -> String {
    keys.iter()
        .find_map(|key| payload.get(*key))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn parse_dimacs_cnf(text: &str) -> Result<(usize, Vec<Vec<i64>>), String> {
    let mut variables = 0usize;
    let mut clauses = Vec::<Vec<i64>>::new();
    let mut pending = Vec::<i64>::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        if line.starts_with('p') {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() >= 4 {
                variables = parts[2]
                    .parse::<usize>()
                    .map_err(|err| format!("invalid DIMACS variable count: {err}"))?;
            }
            continue;
        }
        for token in line.split_whitespace() {
            let literal = token
                .parse::<i64>()
                .map_err(|err| format!("invalid DIMACS literal {token:?}: {err}"))?;
            if literal == 0 {
                clauses.push(std::mem::take(&mut pending));
            } else {
                variables = variables.max(literal.unsigned_abs() as usize);
                pending.push(literal);
            }
        }
    }
    if !pending.is_empty() {
        clauses.push(pending);
    }
    Ok((variables, clauses))
}

fn dimacs_clause_satisfied(mask: u64, clause: &[i64]) -> bool {
    clause.iter().any(|literal| {
        let idx = literal.unsigned_abs() as usize - 1;
        let value = ((mask >> idx) & 1) == 1;
        value == (*literal > 0)
    })
}

fn brute_force_dimacs(text: &str) -> Value {
    let (variables, clauses) = match parse_dimacs_cnf(text) {
        Ok(parsed) => parsed,
        Err(message) => {
            return result(
                "failed",
                "failure",
                "rust:dimacs-small-cnf",
                message,
                "",
                "",
            )
        }
    };
    if variables > 24 {
        return result(
            "unavailable",
            "unknown",
            "rust:dimacs-small-cnf",
            format!("builtin CNF fallback is capped at 24 variables, got {variables}"),
            "",
            "",
        );
    }
    let total = 1u64 << variables;
    for mask in 0..total {
        if clauses
            .iter()
            .all(|clause| dimacs_clause_satisfied(mask, clause))
        {
            let model = (1..=variables)
                .map(|idx| {
                    if ((mask >> (idx - 1)) & 1) == 1 {
                        idx.to_string()
                    } else {
                        format!("-{idx}")
                    }
                })
                .collect::<Vec<_>>()
                .join(" ");
            return result(
                "ok",
                "sat",
                "rust:dimacs-small-cnf",
                "satisfying assignment found",
                format!("s SATISFIABLE\nv {model} 0\n"),
                "",
            );
        }
    }
    result(
        "ok",
        "unsat",
        "rust:dimacs-small-cnf",
        "all assignments exhausted",
        "",
        "",
    )
}

fn parse_wcnf(text: &str) -> Result<(usize, Option<i64>, Vec<(i64, Vec<i64>)>), String> {
    let mut variables = 0usize;
    let mut top_weight = None::<i64>;
    let mut clauses = Vec::<(i64, Vec<i64>)>::new();
    for raw_line in text.lines() {
        let line = raw_line.trim();
        if line.is_empty() || line.starts_with('c') {
            continue;
        }
        if line.starts_with('p') {
            let parts = line.split_whitespace().collect::<Vec<_>>();
            if parts.len() >= 4 && parts[1].eq_ignore_ascii_case("wcnf") {
                variables = parts[2]
                    .parse::<usize>()
                    .map_err(|err| format!("invalid WCNF variable count: {err}"))?;
                if parts.len() >= 5 {
                    top_weight = Some(
                        parts[4]
                            .parse::<i64>()
                            .map_err(|err| format!("invalid WCNF top weight: {err}"))?,
                    );
                }
            }
            continue;
        }
        let tokens = line
            .split_whitespace()
            .map(|token| {
                token
                    .parse::<i64>()
                    .map_err(|err| format!("invalid WCNF token {token:?}: {err}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        if tokens.len() < 2 || tokens.last() != Some(&0) {
            return Err("WCNF clauses must be '<weight> <lits...> 0'".to_string());
        }
        let weight = tokens[0];
        let clause = tokens[1..tokens.len() - 1].to_vec();
        variables = variables.max(
            clause
                .iter()
                .map(|literal| literal.unsigned_abs() as usize)
                .max()
                .unwrap_or(0),
        );
        clauses.push((weight, clause));
    }
    Ok((variables, top_weight, clauses))
}

fn brute_force_wcnf(text: &str) -> Value {
    let (variables, top_weight, clauses) = match parse_wcnf(text) {
        Ok(parsed) => parsed,
        Err(message) => {
            return result(
                "failed",
                "failure",
                "rust:wcnf-small-maxsat",
                message,
                "",
                "",
            )
        }
    };
    if variables > 24 {
        return result(
            "unavailable",
            "unknown",
            "rust:wcnf-small-maxsat",
            format!("builtin WCNF fallback is capped at 24 variables, got {variables}"),
            "",
            "",
        );
    }
    let mut best_cost = None::<i64>;
    let mut best_mask = 0u64;
    let total = 1u64 << variables;
    for mask in 0..total {
        let mut hard_failed = false;
        let mut cost = 0i64;
        for (weight, clause) in &clauses {
            if dimacs_clause_satisfied(mask, clause) {
                continue;
            }
            if top_weight.is_some_and(|top| *weight >= top) {
                hard_failed = true;
                break;
            }
            cost = cost.saturating_add(*weight);
        }
        if hard_failed {
            continue;
        }
        if best_cost.is_none_or(|best| cost < best) {
            best_cost = Some(cost);
            best_mask = mask;
        }
    }
    let Some(best_cost) = best_cost else {
        return result(
            "ok",
            "unsat",
            "rust:wcnf-small-maxsat",
            "hard clauses are unsatisfiable",
            "",
            "",
        );
    };
    let model = (1..=variables)
        .map(|idx| {
            if ((best_mask >> (idx - 1)) & 1) == 1 {
                idx.to_string()
            } else {
                format!("-{idx}")
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    result(
        "ok",
        "optimal",
        "rust:wcnf-small-maxsat",
        format!("optimum={best_cost}"),
        format!("o {best_cost}\ns OPTIMUM FOUND\nv {model} 0\n"),
        "",
    )
}

#[derive(Clone, Debug)]
struct OpbConstraint {
    terms: Vec<(i64, String)>,
    op: String,
    rhs: i64,
}

fn valid_opb_name(name: &str) -> bool {
    let mut chars = name.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    if !(first == '_' || first.is_ascii_alphabetic()) {
        return false;
    }
    chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn split_opb_constraint(line: &str) -> Option<(&str, &str, &str)> {
    for op in [">=", "<=", "="] {
        if let Some(idx) = line.find(op) {
            return Some((&line[..idx], op, &line[idx + op.len()..]));
        }
    }
    None
}

fn parse_opb(text: &str) -> Result<(Vec<String>, Vec<OpbConstraint>), String> {
    let mut variables = BTreeSet::<String>::new();
    let mut constraints = Vec::<OpbConstraint>::new();
    for raw_line in text.lines() {
        let line = raw_line.trim().trim_end_matches(';').trim();
        if line.is_empty() || line.starts_with('*') {
            continue;
        }
        let lower = line.to_ascii_lowercase();
        if lower.starts_with("min:") || lower.starts_with("max:") {
            continue;
        }
        let Some((lhs, op, rhs_text)) = split_opb_constraint(line) else {
            return Err(format!("unsupported OPB constraint {line:?}"));
        };
        let tokens = lhs.split_whitespace().collect::<Vec<_>>();
        if tokens.len() % 2 != 0 {
            return Err(format!("unsupported OPB term list {lhs:?}"));
        }
        let mut terms = Vec::<(i64, String)>::new();
        for pair in tokens.chunks(2) {
            let coeff = pair[0]
                .parse::<i64>()
                .map_err(|err| format!("invalid OPB coefficient {:?}: {err}", pair[0]))?;
            let name = pair[1].to_string();
            if !valid_opb_name(&name) {
                return Err(format!("unsupported OPB variable {name:?}"));
            }
            variables.insert(name.clone());
            terms.push((coeff, name));
        }
        let rhs = rhs_text
            .trim()
            .parse::<i64>()
            .map_err(|err| format!("invalid OPB rhs {:?}: {err}", rhs_text.trim()))?;
        constraints.push(OpbConstraint {
            terms,
            op: op.to_string(),
            rhs,
        });
    }
    if constraints.is_empty() {
        return Err("missing OPB constraints".to_string());
    }
    Ok((variables.into_iter().collect(), constraints))
}

fn opb_constraint_satisfied(
    constraint: &OpbConstraint,
    assignment: &BTreeMap<String, bool>,
) -> bool {
    let total = constraint
        .terms
        .iter()
        .map(|(coeff, name)| coeff * i64::from(*assignment.get(name).unwrap_or(&false)))
        .sum::<i64>();
    match constraint.op.as_str() {
        ">=" => total >= constraint.rhs,
        "<=" => total <= constraint.rhs,
        _ => total == constraint.rhs,
    }
}

fn brute_force_opb(text: &str) -> Value {
    let (variables, constraints) = match parse_opb(text) {
        Ok(parsed) => parsed,
        Err(message) => return result("failed", "failure", "rust:opb-small-pb", message, "", ""),
    };
    if variables.len() > 24 {
        return result(
            "unavailable",
            "unknown",
            "rust:opb-small-pb",
            format!(
                "builtin OPB fallback is capped at 24 variables, got {}",
                variables.len()
            ),
            "",
            "",
        );
    }
    let total = 1u64 << variables.len();
    for mask in 0..total {
        let assignment = variables
            .iter()
            .enumerate()
            .map(|(idx, name)| (name.clone(), ((mask >> idx) & 1) == 1))
            .collect::<BTreeMap<_, _>>();
        if constraints
            .iter()
            .all(|constraint| opb_constraint_satisfied(constraint, &assignment))
        {
            let model = variables
                .iter()
                .map(|name| format!("{name}={}", i32::from(assignment[name])))
                .collect::<Vec<_>>()
                .join(" ");
            return result(
                "ok",
                "sat",
                "rust:opb-small-pb",
                "satisfying assignment found",
                model,
                "",
            );
        }
    }
    result(
        "ok",
        "unsat",
        "rust:opb-small-pb",
        "all assignments exhausted",
        "",
        "",
    )
}

fn collapse_whitespace(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn builtin_smtlib(text: &str) -> Value {
    let lowered = collapse_whitespace(&text.to_ascii_lowercase());
    if lowered.contains("(assert false)") {
        return result(
            "ok",
            "unsat",
            "rust:smtlib-smoke",
            "assert false detected",
            "",
            "",
        );
    }
    let normalized = collapse_whitespace(text);
    let mut equalities = BTreeMap::<String, String>::new();
    let mut offset = 0usize;
    let marker = "(assert (= ";
    while let Some(found) = normalized[offset..].find(marker) {
        let start = offset + found + marker.len();
        let rest = &normalized[start..];
        let mut parts = rest.split_whitespace();
        let Some(name) = parts.next() else {
            break;
        };
        let Some(raw_value) = parts.next() else {
            break;
        };
        let value = raw_value.trim_end_matches(')').to_string();
        if value.parse::<i64>().is_ok() {
            if let Some(previous) = equalities.insert(name.to_string(), value.clone()) {
                if previous != value {
                    return result(
                        "ok",
                        "unsat",
                        "rust:smtlib-smoke",
                        format!("conflicting equalities for {name}"),
                        "",
                        "",
                    );
                }
            }
        }
        offset = start;
    }
    result(
        "ok",
        "sat",
        "rust:smtlib-smoke",
        "no contradiction found",
        "",
        "",
    )
}

fn parse_minizinc_domains(model: &str) -> Result<BTreeMap<String, Vec<i64>>, String> {
    let mut domains = BTreeMap::<String, Vec<i64>>::new();
    for statement in model.split(';') {
        let statement = statement.trim();
        let Some(rest) = statement.strip_prefix("var") else {
            continue;
        };
        let rest = rest.trim();
        let Some((range_text, name_text)) = rest.split_once(':') else {
            continue;
        };
        let Some((lo_text, hi_text)) = range_text.trim().split_once("..") else {
            continue;
        };
        let lo = lo_text
            .trim()
            .parse::<i64>()
            .map_err(|err| format!("invalid MiniZinc lower bound: {err}"))?;
        let hi = hi_text
            .trim()
            .parse::<i64>()
            .map_err(|err| format!("invalid MiniZinc upper bound: {err}"))?;
        if hi < lo {
            return Err("MiniZinc domain upper bound is below lower bound".to_string());
        }
        if hi - lo > 100 {
            return Err("builtin MiniZinc fallback supports domains of size <= 101".to_string());
        }
        let name = name_text.trim();
        if valid_opb_name(name) {
            domains.insert(name.to_string(), (lo..=hi).collect());
        }
    }
    Ok(domains)
}

fn parse_minizinc_constraints(model: &str) -> Vec<String> {
    model
        .split(';')
        .filter_map(|statement| statement.trim().strip_prefix("constraint"))
        .map(|expr| expr.trim().to_string())
        .collect()
}

fn eval_minizinc_constraint(
    expr: &str,
    assignment: &BTreeMap<String, i64>,
) -> Result<bool, String> {
    let operators = ["<=", ">=", "==", "=", "<", ">"];
    let Some((name, op, value_text)) = operators.iter().find_map(|op| {
        expr.find(op)
            .map(|idx| (&expr[..idx], *op, &expr[idx + op.len()..]))
    }) else {
        return Err(format!("unsupported MiniZinc constraint {expr:?}"));
    };
    let name = name.trim();
    let actual = assignment
        .get(name)
        .ok_or_else(|| format!("unknown MiniZinc variable {name:?}"))?;
    let expected = value_text
        .trim()
        .parse::<i64>()
        .map_err(|err| format!("invalid MiniZinc constraint rhs: {err}"))?;
    Ok(match op {
        "<=" => *actual <= expected,
        ">=" => *actual >= expected,
        "=" | "==" => *actual == expected,
        "<" => *actual < expected,
        _ => *actual > expected,
    })
}

fn search_minizinc(
    names: &[String],
    domains: &BTreeMap<String, Vec<i64>>,
    constraints: &[String],
    idx: usize,
    assignment: &mut BTreeMap<String, i64>,
) -> Result<Option<BTreeMap<String, i64>>, String> {
    if idx == names.len() {
        for constraint in constraints {
            if !eval_minizinc_constraint(constraint, assignment)? {
                return Ok(None);
            }
        }
        return Ok(Some(assignment.clone()));
    }
    let name = &names[idx];
    for value in &domains[name] {
        assignment.insert(name.clone(), *value);
        if let Some(solution) = search_minizinc(names, domains, constraints, idx + 1, assignment)? {
            return Ok(Some(solution));
        }
    }
    assignment.remove(name);
    Ok(None)
}

fn builtin_minizinc(model: &str) -> Value {
    let domains = match parse_minizinc_domains(model) {
        Ok(domains) => domains,
        Err(message) => return result("failed", "failure", "rust:minizinc-smoke", message, "", ""),
    };
    let constraints = parse_minizinc_constraints(model);
    if domains.is_empty() {
        if model.contains("constraint false;") {
            return result(
                "ok",
                "unsat",
                "rust:minizinc-smoke",
                "constraint false detected",
                "",
                "",
            );
        }
        return result(
            "ok",
            "sat",
            "rust:minizinc-smoke",
            "no finite-domain variables detected",
            "",
            "",
        );
    }
    let total = domains
        .values()
        .map(Vec::len)
        .try_fold(1usize, |acc, len| acc.checked_mul(len))
        .unwrap_or(usize::MAX);
    if total > 250_000 {
        return result(
            "unavailable",
            "unknown",
            "rust:minizinc-smoke",
            format!("search space too large: {total}"),
            "",
            "",
        );
    }
    let names = domains.keys().cloned().collect::<Vec<_>>();
    let mut assignment = BTreeMap::<String, i64>::new();
    match search_minizinc(&names, &domains, &constraints, 0, &mut assignment) {
        Ok(Some(solution)) => {
            let stdout = names
                .iter()
                .map(|name| format!("{name} = {};", solution[name]))
                .collect::<Vec<_>>()
                .join("\n")
                + "\n----------\n";
            result(
                "ok",
                "sat",
                "rust:minizinc-smoke",
                "satisfying assignment found",
                stdout,
                "",
            )
        }
        Ok(None) => result(
            "ok",
            "unsat",
            "rust:minizinc-smoke",
            "all assignments exhausted",
            "",
            "",
        ),
        Err(message) => result("failed", "failure", "rust:minizinc-smoke", message, "", ""),
    }
}

fn dispatch(payload: &Value, tool_override: Option<&str>) -> Value {
    let kind = normalize_tool_id(payload.get("kind").and_then(Value::as_str));
    let tool = normalize_tool_id(
        tool_override
            .or_else(|| payload.get("solver").and_then(Value::as_str))
            .or_else(|| payload.get("tool").and_then(Value::as_str)),
    );
    if kind == "minizinc-validation"
        || matches!(
            tool.as_str(),
            "minizinc"
                | "flatzinc"
                | "minizinc-solution-checker"
                | "gecode"
                | "chuffed"
                | "ortools-cp-sat"
                | "fzn-cp-sat"
        )
    {
        let model = payload_text(payload, &["model"]);
        if model.trim().is_empty() {
            return result(
                "failed",
                "failure",
                "minizinc",
                "payload needs model",
                "",
                "",
            );
        }
        return builtin_minizinc(&model);
    }
    if kind == "smtlib-validation"
        || kind == "smt-lib-validation"
        || matches!(
            tool.as_str(),
            "z3" | "cvc5"
                | "yices"
                | "bitwuzla"
                | "boolector"
                | "mathsat"
                | "optimathsat"
                | "opensmt"
                | "smtinterpol"
                | "princess"
        )
    {
        let text = payload_text(payload, &["script", "smtlib", "text", "model"]);
        if text.trim().is_empty() {
            return result(
                "failed",
                "failure",
                "smtlib",
                "payload needs script, smtlib, text, or model",
                "",
                "",
            );
        }
        return builtin_smtlib(&text);
    }
    if kind == "wcnf-validation"
        || kind == "dimacs-wcnf-validation"
        || kind == "maxsat-validation"
        || (matches!(tool.as_str(), "open-wbo" | "maxhs")
            && ["wcnf", "dimacs"]
                .iter()
                .any(|key| payload.get(*key).is_some()))
    {
        let text = payload_text(payload, &["wcnf", "dimacs", "text", "model"]);
        if text.trim().is_empty() {
            return result(
                "failed",
                "failure",
                "wcnf",
                "payload needs wcnf, dimacs, text, or model",
                "",
                "",
            );
        }
        return brute_force_wcnf(&text);
    }
    if kind == "opb-validation" || kind == "pseudo-boolean-validation" || tool == "roundingsat" {
        let text = payload_text(payload, &["opb", "pb", "text", "model"]);
        if text.trim().is_empty() {
            return result(
                "failed",
                "failure",
                "opb",
                "payload needs opb, pb, text, or model",
                "",
                "",
            );
        }
        return brute_force_opb(&text);
    }
    if kind == "dimacs-validation"
        || kind == "dimacs-cnf-validation"
        || matches!(
            tool.as_str(),
            "kissat" | "cadical" | "cryptominisat" | "minisat" | "glucose" | "maplesat" | "varisat"
        )
    {
        let text = payload_text(payload, &["dimacs", "cnf", "text", "model"]);
        if text.trim().is_empty() {
            return result(
                "failed",
                "failure",
                "dimacs",
                "payload needs dimacs, cnf, text, or model",
                "",
                "",
            );
        }
        return brute_force_dimacs(&text);
    }
    result(
        "unavailable",
        "unknown",
        &tool,
        format!("unknown model validation payload kind {kind:?}"),
        "",
        "",
    )
}

fn next_option_value(
    program: &str,
    option: &str,
    inline_value: Option<String>,
    values: &mut impl Iterator<Item = String>,
) -> Result<String, CliError> {
    if let Some(value) = inline_value {
        return Ok(value);
    }
    let value = values
        .next()
        .ok_or_else(|| CliError(format!("{option} requires a value\n{}", usage(program))))?;
    if value.starts_with("--") {
        return Err(CliError(format!(
            "{option} requires a value\n{}",
            usage(program)
        )));
    }
    Ok(value)
}

fn parse_args(
    program: &str,
    args: impl IntoIterator<Item = String>,
) -> Result<Option<String>, CliError> {
    let mut tool = None::<String>;
    let mut values = args.into_iter();
    while let Some(raw) = values.next() {
        if raw == "-h" || raw == "--help" {
            return Err(CliError(usage(program)));
        }
        let (key, inline_value) = if let Some((key, value)) = raw.split_once('=') {
            (key.to_string(), Some(value.to_string()))
        } else {
            (raw, None)
        };
        match key.as_str() {
            "--tool" => {
                tool = Some(next_option_value(
                    program,
                    "--tool",
                    inline_value,
                    &mut values,
                )?);
            }
            _ => {
                return Err(CliError(format!(
                    "unknown option {key}\n{}",
                    usage(program)
                )))
            }
        }
    }
    Ok(tool)
}

fn run(raw_args: Vec<String>, stdin: &str) -> Result<Value, CliError> {
    let program = raw_args
        .first()
        .cloned()
        .unwrap_or_else(|| "model_validation_reference".to_string());
    let tool = parse_args(&program, raw_args.into_iter().skip(1))?;
    let payload = serde_json::from_str::<Value>(stdin)
        .map_err(|err| CliError(format!("parse JSON: {err}")))?;
    Ok(dispatch(&payload, tool.as_deref()))
}

fn main() {
    let raw_args = env::args().collect::<Vec<_>>();
    if raw_args.iter().any(|arg| arg == "-h" || arg == "--help") {
        println!(
            "{}",
            usage(
                raw_args
                    .first()
                    .map(String::as_str)
                    .unwrap_or("model_validation_reference")
            )
        );
        return;
    }
    let mut stdin = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut stdin) {
        println!("{}", empty_error(format!("failed to read stdin: {err}")));
        std::process::exit(1);
    }
    match run(raw_args, &stdin) {
        Ok(output) => println!(
            "{}",
            serde_json::to_string(&output).expect("serialize model-validation output")
        ),
        Err(err) => {
            println!("{}", empty_error(err.to_string()));
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dimacs_builtin_finds_sat_assignment() {
        let payload = json!({
            "kind": "dimacs-validation",
            "dimacs": "p cnf 2 2\n1 2 0\n-1 2 0\n"
        });
        let output = dispatch(&payload, None);
        assert_eq!(output["status"], "ok");
        assert_eq!(output["verdict"], "sat");
        assert_eq!(output["validator"], "rust:dimacs-small-cnf");
    }

    #[test]
    fn wcnf_builtin_finds_optimum_cost() {
        let payload = json!({
            "kind": "wcnf-validation",
            "wcnf": "p wcnf 2 3 10\n10 1 0\n2 2 0\n3 -2 0\n"
        });
        let output = dispatch(&payload, None);
        assert_eq!(output["verdict"], "optimal");
        assert_eq!(output["message"], "optimum=2");
        assert_eq!(output["validator"], "rust:wcnf-small-maxsat");
    }

    #[test]
    fn opb_builtin_finds_satisfying_assignment() {
        let payload = json!({
            "kind": "opb-validation",
            "opb": "1 x 1 y >= 1;"
        });
        let output = dispatch(&payload, None);
        assert_eq!(output["verdict"], "sat");
        assert_eq!(output["validator"], "rust:opb-small-pb");
    }

    #[test]
    fn smtlib_builtin_detects_conflicting_equalities() {
        let payload = json!({
            "kind": "smtlib-validation",
            "script": "(assert (= x 1))\n(assert (= x 2))"
        });
        let output = dispatch(&payload, None);
        assert_eq!(output["verdict"], "unsat");
        assert_eq!(output["validator"], "rust:smtlib-smoke");
    }

    #[test]
    fn minizinc_builtin_searches_small_domains() {
        let payload = json!({
            "kind": "minizinc-validation",
            "model": "var 0..2: x; constraint x >= 2;"
        });
        let output = dispatch(&payload, None);
        assert_eq!(output["verdict"], "sat");
        assert_eq!(output["validator"], "rust:minizinc-smoke");
        assert!(output["stdout"].as_str().unwrap().contains("x = 2;"));
    }
}
