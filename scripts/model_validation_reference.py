#!/usr/bin/env python3
"""Reference bridge for external model validators.

The bridge keeps external executables optional. It calls local MiniZinc, SMT, or
SAT tools when available, and falls back to tiny dependency-free validators for
the smoke models used by the Rust cross-check suite.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def result(
    status: str,
    verdict: str,
    validator: str,
    message: str = "",
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message,
        "stdout": stdout,
        "stderr": stderr,
    }


def normalize_tool_id(tool: str | None) -> str:
    return (tool or "auto").strip().lower().replace("_", "-")


def command_from_env(tool: str) -> str | None:
    key = "ORES_" + re.sub(r"[^A-Z0-9]+", "_", tool.upper()).strip("_") + "_ADAPTER"
    configured = os.environ.get(key)
    if configured:
        return configured
    return None


def first_command(tool: str, aliases: list[str]) -> str | None:
    configured = command_from_env(tool)
    if configured and shutil.which(configured):
        return configured
    for alias in aliases:
        found = shutil.which(alias)
        if found:
            return found
    return None


def infer_sat_like(stdout: str, stderr: str, exit_success: bool) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "=====unsatisfiable=====" in text or "unsatisfiable" in text:
        return "unsat"
    if "----------" in stdout and exit_success:
        return "sat"
    for token in re.findall(r"[a-zA-Z_]+", text):
        if token in ("sat", "satisfiable"):
            return "sat"
        if token in ("unsat", "unsatisfiable"):
            return "unsat"
        if token == "unknown":
            return "unknown"
    return "success" if exit_success else "failure"


def run_command(command: str, args: list[str], stdin_text: str = "") -> tuple[bool, str, str]:
    completed = subprocess.run(
        [command, *args],
        input=stdin_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0, completed.stdout, completed.stderr


def parse_dimacs_cnf(text: str) -> tuple[int, list[list[int]]]:
    variables = 0
    clauses: list[list[int]] = []
    pending: list[int] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) >= 4:
                variables = int(parts[2])
            continue
        for token in line.split():
            literal = int(token)
            if literal == 0:
                clauses.append(pending)
                pending = []
            else:
                variables = max(variables, abs(literal))
                pending.append(literal)
    if pending:
        clauses.append(pending)
    return variables, clauses


def brute_force_dimacs(text: str) -> dict[str, Any]:
    variables, clauses = parse_dimacs_cnf(text)
    if variables > 24:
        return result(
            "unavailable",
            "unknown",
            "builtin:dimacs-small-cnf",
            f"builtin CNF fallback is capped at 24 variables, got {variables}",
        )
    for bits in itertools.product([False, True], repeat=variables):
        assignment = {idx + 1: value for idx, value in enumerate(bits)}
        if all(any(assignment[abs(lit)] == (lit > 0) for lit in clause) for clause in clauses):
            model = [idx if value else -idx for idx, value in assignment.items()]
            return result(
                "ok",
                "sat",
                "builtin:dimacs-small-cnf",
                "satisfying assignment found",
                "s SATISFIABLE\nv " + " ".join(str(value) for value in model) + " 0\n",
            )
    return result("ok", "unsat", "builtin:dimacs-small-cnf", "all assignments exhausted")


def validate_dimacs(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    text = str(payload.get("dimacs") or payload.get("cnf") or payload.get("text") or payload.get("model") or "")
    if not text.strip():
        return result("failed", "failure", "dimacs", "payload needs dimacs, cnf, text, or model")
    command = first_command(tool, [tool, "kissat", "cadical", "cryptominisat"])
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-dimacs-") as tmp:
            path = Path(tmp) / "problem.cnf"
            path.write_text(text, encoding="utf-8")
            ok, stdout, stderr = run_command(command, [str(path)])
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return brute_force_dimacs(text)


def builtin_smtlib(text: str) -> dict[str, Any]:
    lowered = re.sub(r"\s+", " ", text.lower())
    if "(assert false)" in lowered:
        return result("ok", "unsat", "builtin:smtlib-smoke", "assert false detected")
    equalities: dict[str, str] = {}
    for name, value in re.findall(r"\(assert\s+\(=\s+([a-zA-Z_][\w.-]*)\s+(-?\d+)\s*\)\s*\)", text):
        if name in equalities and equalities[name] != value:
            return result("ok", "unsat", "builtin:smtlib-smoke", f"conflicting equalities for {name}")
        equalities[name] = value
    return result("ok", "sat", "builtin:smtlib-smoke", "no contradiction found")


def validate_smtlib(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    text = str(payload.get("script") or payload.get("smtlib") or payload.get("text") or payload.get("model") or "")
    if not text.strip():
        return result("failed", "failure", "smtlib", "payload needs script, smtlib, text, or model")
    command = first_command(tool, [tool, "z3", "cvc5", "bitwuzla", "boolector"])
    if command:
        basename = Path(command).name.lower()
        if basename == "z3":
            args = ["-in", "-smt2"]
        elif basename == "cvc5":
            args = ["--lang=smt2", "-"]
        elif basename in ("bitwuzla", "boolector"):
            args = ["--smt2", "-"]
        else:
            args = []
        ok, stdout, stderr = run_command(command, args, text)
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return builtin_smtlib(text)


def minizinc_var_domains(model: str) -> dict[str, range]:
    domains: dict[str, range] = {}
    for lower, upper, name in re.findall(r"var\s+(-?\d+)\s*\.\.\s*(-?\d+)\s*:\s*([A-Za-z_]\w*)\s*;", model):
        lo = int(lower)
        hi = int(upper)
        if hi - lo > 100:
            raise ValueError("builtin MiniZinc fallback supports domains of size <= 101")
        domains[name] = range(lo, hi + 1)
    return domains


def eval_minizinc_constraint(expr: str, assignment: dict[str, int]) -> bool:
    match = re.fullmatch(r"\s*([A-Za-z_]\w*)\s*(<=|>=|=|==|<|>)\s*(-?\d+)\s*", expr)
    if not match:
        raise ValueError(f"unsupported MiniZinc constraint {expr!r}")
    name, op, value_text = match.groups()
    actual = assignment[name]
    expected = int(value_text)
    if op == "<=":
        return actual <= expected
    if op == ">=":
        return actual >= expected
    if op in ("=", "=="):
        return actual == expected
    if op == "<":
        return actual < expected
    return actual > expected


def builtin_minizinc(model: str) -> dict[str, Any]:
    domains = minizinc_var_domains(model)
    constraints = re.findall(r"constraint\s+([^;]+);", model)
    if not domains:
        if "constraint false;" in model:
            return result("ok", "unsat", "builtin:minizinc-smoke", "constraint false detected")
        return result("ok", "sat", "builtin:minizinc-smoke", "no finite-domain variables detected")
    names = list(domains)
    total = 1
    for domain in domains.values():
        total *= len(domain)
    if total > 250_000:
        return result("unavailable", "unknown", "builtin:minizinc-smoke", f"search space too large: {total}")
    for values in itertools.product(*(domains[name] for name in names)):
        assignment = dict(zip(names, values))
        if all(eval_minizinc_constraint(expr, assignment) for expr in constraints):
            stdout = "\n".join(f"{name} = {assignment[name]};" for name in names) + "\n----------\n"
            return result("ok", "sat", "builtin:minizinc-smoke", "satisfying assignment found", stdout)
    return result("ok", "unsat", "builtin:minizinc-smoke", "all assignments exhausted")


def validate_minizinc(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    model = str(payload.get("model") or "")
    data = str(payload.get("data") or "")
    solver = str(payload.get("solver") or "").strip()
    if not model.strip():
        return result("failed", "failure", "minizinc", "payload needs model")
    command = first_command(tool, [tool, "minizinc"])
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-minizinc-") as tmp:
            model_path = Path(tmp) / "model.mzn"
            data_path = Path(tmp) / "data.dzn"
            model_path.write_text(model, encoding="utf-8")
            args = []
            if solver:
                args.extend(["--solver", solver])
            args.append(str(model_path))
            if data.strip():
                data_path.write_text(data, encoding="utf-8")
                args.append(str(data_path))
            ok, stdout, stderr = run_command(command, args)
        verdict = infer_sat_like(stdout, stderr, ok)
        return result("ok" if ok else "failed", verdict, command, "", stdout, stderr)
    return builtin_minizinc(model)


def dispatch(payload: dict[str, Any], tool_override: str | None = None) -> dict[str, Any]:
    kind = normalize_tool_id(str(payload.get("kind", "")))
    tool = normalize_tool_id(tool_override or payload.get("solver") or payload.get("tool"))
    if kind == "minizinc-validation" or tool == "minizinc":
        return validate_minizinc(payload, "minizinc")
    if kind in ("smtlib-validation", "smt-lib-validation") or tool in ("z3", "cvc5", "bitwuzla", "boolector"):
        return validate_smtlib(payload, "z3" if tool == "auto" else tool)
    if kind in ("dimacs-validation", "dimacs-cnf-validation") or tool in ("kissat", "cadical", "cryptominisat"):
        return validate_dimacs(payload, "kissat" if tool == "auto" else tool)
    return result("unavailable", "unknown", tool, f"unknown model validation payload kind {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default=None)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        print(json.dumps(dispatch(payload, args.tool)))
    except Exception as exc:
        print(json.dumps(result("failed", "failure", args.tool or "model-validation", str(exc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
