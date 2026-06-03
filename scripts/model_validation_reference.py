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


MINIZINC_TOOL_IDS = {
    "minizinc",
    "flatzinc",
    "minizinc-solution-checker",
    "gecode",
    "chuffed",
    "ortools-cp-sat",
    "fzn-cp-sat",
}


ASP_TOOL_IDS = {
    "clingo",
    "clingcon",
}


def choose_minizinc_solver(command: str, requested: str = "") -> str:
    ok, stdout, stderr = run_command(command, ["--solvers"])
    catalog = f"{stdout}\n{stderr}".lower() if ok else ""
    aliases = {
        "cbc": "org.minizinc.mip.coin-bc",
        "coin-bc": "org.minizinc.mip.coin-bc",
        "coinbc": "org.minizinc.mip.coin-bc",
        "scip": "org.minizinc.mip.scip",
        "highs": "org.minizinc.mip.highs",
        "gecode": "org.gecode.gecode",
        "chuffed": "org.chuffed.chuffed",
        "cp-sat": "cp-sat",
        "ortools-cp-sat": "cp-sat",
        "or-tools-cp-sat": "cp-sat",
        "fzn-cp-sat": "cp-sat",
    }
    if requested:
        normalized = aliases.get(requested.lower(), requested)
        if not catalog or normalized.lower() in catalog or requested.lower() in catalog:
            return normalized
    for candidate in [
        "org.gecode.gecode",
        "org.chuffed.chuffed",
        "cp-sat",
        "org.minizinc.mip.coin-bc",
        "org.minizinc.mip.scip",
        "org.minizinc.mip.highs",
    ]:
        if candidate.lower() in catalog:
            return candidate
    return requested


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
    aliases_by_tool = {
        "kissat": ["kissat"],
        "cadical": ["cadical"],
        "cryptominisat": ["cryptominisat5", "cryptominisat"],
        "minisat": ["minisat"],
        "glucose": ["glucose", "glucose-syrup"],
        "maplesat": ["maplesat", "maple-sat", "maple-lcm"],
        "varisat": ["varisat"],
    }
    aliases = (
        [
            "kissat",
            "cadical",
            "cryptominisat5",
            "cryptominisat",
            "minisat",
            "glucose",
            "glucose-syrup",
            "maplesat",
            "maple-sat",
            "maple-lcm",
            "varisat",
        ]
        if tool == "auto"
        else aliases_by_tool.get(tool, [tool])
    )
    command = first_command(tool, aliases)
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-dimacs-") as tmp:
            path = Path(tmp) / "problem.cnf"
            path.write_text(text, encoding="utf-8")
            ok, stdout, stderr = run_command(command, [str(path)])
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return brute_force_dimacs(text)


def parse_wcnf(text: str) -> tuple[int, int | None, list[tuple[int, list[int]]]]:
    variables = 0
    top_weight: int | None = None
    clauses: list[tuple[int, list[int]]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("c"):
            continue
        if line.startswith("p"):
            parts = line.split()
            if len(parts) >= 4 and parts[1].lower() == "wcnf":
                variables = int(parts[2])
                if len(parts) >= 5:
                    top_weight = int(parts[4])
            continue
        tokens = [int(token) for token in line.split()]
        if len(tokens) < 2 or tokens[-1] != 0:
            raise ValueError("WCNF clauses must be '<weight> <lits...> 0'")
        weight = tokens[0]
        clause = tokens[1:-1]
        variables = max([variables, *[abs(literal) for literal in clause]])
        clauses.append((weight, clause))
    return variables, top_weight, clauses


def brute_force_wcnf(text: str) -> dict[str, Any]:
    variables, top_weight, clauses = parse_wcnf(text)
    if variables > 24:
        return result(
            "unavailable",
            "unknown",
            "builtin:wcnf-small-maxsat",
            f"builtin WCNF fallback is capped at 24 variables, got {variables}",
        )
    best_cost: int | None = None
    best_model: list[int] = []
    for bits in itertools.product([False, True], repeat=variables):
        assignment = {idx + 1: value for idx, value in enumerate(bits)}
        hard_failed = False
        cost = 0
        for weight, clause in clauses:
            satisfied = any(assignment[abs(lit)] == (lit > 0) for lit in clause)
            if satisfied:
                continue
            if top_weight is not None and weight >= top_weight:
                hard_failed = True
                break
            cost += weight
        if hard_failed:
            continue
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_model = [idx if value else -idx for idx, value in assignment.items()]
    if best_cost is None:
        return result("ok", "unsat", "builtin:wcnf-small-maxsat", "hard clauses are unsatisfiable")
    stdout = f"o {best_cost}\ns OPTIMUM FOUND\nv {' '.join(str(value) for value in best_model)} 0\n"
    return result("ok", "optimal", "builtin:wcnf-small-maxsat", f"optimum={best_cost}", stdout)


def validate_wcnf(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    text = str(payload.get("wcnf") or payload.get("dimacs") or payload.get("text") or payload.get("model") or "")
    if not text.strip():
        return result("failed", "failure", "wcnf", "payload needs wcnf, dimacs, text, or model")
    aliases_by_tool = {
        "open-wbo": ["open-wbo", "open-wbo_static"],
        "maxhs": ["maxhs"],
        "sat4j": ["sat4j", "sat4j-sat"],
        "pysat": ["pysat-adapter", "python-sat-adapter"],
    }
    aliases = ["open-wbo", "open-wbo_static", "maxhs"] if tool == "auto" else aliases_by_tool.get(tool, [tool])
    command = first_command(tool, aliases)
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-wcnf-") as tmp:
            path = Path(tmp) / "problem.wcnf"
            path.write_text(text, encoding="utf-8")
            ok, stdout, stderr = run_command(command, [str(path)])
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return brute_force_wcnf(text)


def parse_opb(text: str) -> tuple[list[str], list[tuple[list[tuple[int, str]], str, int]]]:
    variables: set[str] = set()
    constraints: list[tuple[list[tuple[int, str]], str, int]] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("*"):
            continue
        if line.lower().startswith(("min:", "max:")):
            continue
        match = re.fullmatch(r"(.+?)\s*(>=|<=|=)\s*(-?\d+)\s*;?", line)
        if not match:
            raise ValueError(f"unsupported OPB constraint {line!r}")
        lhs, op, rhs_text = match.groups()
        tokens = lhs.split()
        if len(tokens) % 2 != 0:
            raise ValueError(f"unsupported OPB term list {lhs!r}")
        terms: list[tuple[int, str]] = []
        for idx in range(0, len(tokens), 2):
            coeff = int(tokens[idx])
            name = tokens[idx + 1]
            if not re.fullmatch(r"[A-Za-z_]\w*", name):
                raise ValueError(f"unsupported OPB variable {name!r}")
            variables.add(name)
            terms.append((coeff, name))
        constraints.append((terms, op, int(rhs_text)))
    if not constraints:
        raise ValueError("missing OPB constraints")
    return sorted(variables), constraints


def opb_constraint_satisfied(
    constraint: tuple[list[tuple[int, str]], str, int],
    assignment: dict[str, bool],
) -> bool:
    terms, op, rhs = constraint
    total = sum(coeff * int(assignment[name]) for coeff, name in terms)
    if op == ">=":
        return total >= rhs
    if op == "<=":
        return total <= rhs
    return total == rhs


def brute_force_opb(text: str) -> dict[str, Any]:
    variables, constraints = parse_opb(text)
    if len(variables) > 24:
        return result(
            "unavailable",
            "unknown",
            "builtin:opb-small-pb",
            f"builtin OPB fallback is capped at 24 variables, got {len(variables)}",
        )
    for bits in itertools.product([False, True], repeat=len(variables)):
        assignment = dict(zip(variables, bits))
        if all(opb_constraint_satisfied(constraint, assignment) for constraint in constraints):
            model = " ".join(f"{name}={int(assignment[name])}" for name in variables)
            return result("ok", "sat", "builtin:opb-small-pb", "satisfying assignment found", model)
    return result("ok", "unsat", "builtin:opb-small-pb", "all assignments exhausted")


def validate_opb(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    text = str(payload.get("opb") or payload.get("pb") or payload.get("text") or payload.get("model") or "")
    if not text.strip():
        return result("failed", "failure", "opb", "payload needs opb, pb, text, or model")
    aliases_by_tool = {
        "roundingsat": ["roundingsat"],
        "sat4j": ["sat4j", "sat4j-pb"],
        "pysat": ["pysat-adapter", "python-sat-adapter"],
    }
    aliases = ["roundingsat"] if tool == "auto" else aliases_by_tool.get(tool, [tool])
    command = first_command(tool, aliases)
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-opb-") as tmp:
            path = Path(tmp) / "problem.opb"
            path.write_text(text, encoding="utf-8")
            ok, stdout, stderr = run_command(command, [str(path)])
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return brute_force_opb(text)


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
    aliases_by_tool = {
        "z3": ["z3"],
        "cvc5": ["cvc5"],
        "yices": ["yices-smt2", "yices"],
        "bitwuzla": ["bitwuzla"],
        "boolector": ["boolector"],
        "mathsat": ["mathsat"],
        "optimathsat": ["optimathsat", "optimathsat5"],
        "opensmt": ["opensmt", "opensmt2"],
        "smtinterpol": ["smtinterpol", "smtinterpol.sh"],
        "princess": ["princess", "princess-smt"],
    }
    aliases = (
        [
            "z3",
            "cvc5",
            "yices-smt2",
            "yices",
            "bitwuzla",
            "boolector",
            "mathsat",
            "optimathsat",
            "optimathsat5",
            "opensmt",
            "opensmt2",
            "smtinterpol",
            "smtinterpol.sh",
            "princess",
            "princess-smt",
        ]
        if tool == "auto"
        else aliases_by_tool.get(tool, [tool])
    )
    command = first_command(tool, aliases)
    if command:
        basename = Path(command).name.lower()
        if basename == "z3":
            args = ["-in", "-smt2"]
        elif basename == "cvc5":
            args = ["--lang=smt2", "-"]
        elif basename in ("yices-smt2", "yices"):
            args = []
        elif basename in ("bitwuzla", "boolector"):
            args = ["--smt2", "-"]
        elif basename in ("mathsat", "optimathsat", "optimathsat5"):
            args = ["-input=smt2"]
        elif basename in ("opensmt", "opensmt2"):
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
    tool = normalize_tool_id(tool)
    if not model.strip():
        return result("failed", "failure", "minizinc", "payload needs model")
    command_tool = "minizinc" if tool in MINIZINC_TOOL_IDS else tool
    configured = command_from_env(tool)
    command = (
        configured
        if configured and shutil.which(configured)
        else first_command(command_tool, ["minizinc"])
    )
    fzn_gecode = first_command("gecode", ["fzn-gecode"])
    fzn_cp_sat = first_command("ortools-cp-sat", ["fzn-cp-sat"])
    flatzinc_backend: tuple[str, str, str] | None = None
    if tool in {"flatzinc", "gecode"} and fzn_gecode:
        flatzinc_backend = (
            fzn_gecode,
            solver,
            "MiniZinc to FlatZinc compilation failed",
        )
    elif tool in {"ortools-cp-sat", "fzn-cp-sat"} and fzn_cp_sat:
        flatzinc_backend = (
            fzn_cp_sat,
            solver or "cp-sat",
            "MiniZinc to OR-Tools CP-SAT FlatZinc compilation failed",
        )
    if flatzinc_backend is not None:
        flatzinc_command, compile_solver_request, compile_error = flatzinc_backend
        minizinc = first_command("minizinc", ["minizinc"])
        if minizinc:
            with tempfile.TemporaryDirectory(prefix="ores-minizinc-gecode-") as tmp:
                model_path = Path(tmp) / "model.mzn"
                data_path = Path(tmp) / "data.dzn"
                fzn_path = Path(tmp) / "model.fzn"
                model_path.write_text(model, encoding="utf-8")
                compile_solver = choose_minizinc_solver(minizinc, compile_solver_request)
                compile_args = ["--compile", "--solver", compile_solver, "-o", str(fzn_path), str(model_path)]
                if data.strip():
                    data_path.write_text(data, encoding="utf-8")
                    compile_args.append(str(data_path))
                compile_ok, compile_stdout, compile_stderr = run_command(minizinc, compile_args)
                if compile_ok:
                    ok, stdout, stderr = run_command(flatzinc_command, [str(fzn_path)])
                    verdict = infer_sat_like(stdout, stderr, ok)
                    return result("ok" if ok else "failed", verdict, flatzinc_command, "", stdout, stderr)
                if tool in {"gecode", "ortools-cp-sat", "fzn-cp-sat"}:
                    return result(
                        "failed",
                        "failure",
                        minizinc,
                        compile_error,
                        compile_stdout,
                        compile_stderr,
                    )
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-minizinc-") as tmp:
            model_path = Path(tmp) / "model.mzn"
            data_path = Path(tmp) / "data.dzn"
            model_path.write_text(model, encoding="utf-8")
            args = []
            backend_solver = solver or (
                tool if tool in {"gecode", "chuffed", "ortools-cp-sat", "fzn-cp-sat"} else ""
            )
            backend_solver = choose_minizinc_solver(command, backend_solver)
            if backend_solver:
                args.extend(["--solver", backend_solver])
            args.append(str(model_path))
            if data.strip():
                data_path.write_text(data, encoding="utf-8")
                args.append(str(data_path))
            ok, stdout, stderr = run_command(command, args)
        if not ok and backend_solver and "no solver with tag" in stderr.lower():
            return builtin_minizinc(model)
        verdict = infer_sat_like(stdout, stderr, ok)
        return result("ok" if ok else "failed", verdict, command, "", stdout, stderr)
    return builtin_minizinc(model)


def validate_asp(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    program = str(
        payload.get("asp")
        or payload.get("program")
        or payload.get("model")
        or payload.get("text")
        or ""
    )
    tool = normalize_tool_id(tool)
    if not program.strip():
        return result("failed", "failure", "asp", "payload needs asp, program, model, or text")
    aliases_by_tool = {
        "clingo": ["clingo"],
        "clingcon": ["clingcon", "clingo"],
    }
    command = first_command(tool, aliases_by_tool.get(tool, [tool]))
    if command:
        ok, stdout, stderr = run_command(command, ["-", "0"], program)
        verdict = infer_sat_like(stdout, stderr, ok or stdout.strip() != "")
        if verdict in {"sat", "unsat"}:
            return result("ok", verdict, command, "", stdout, stderr)
        return result("failed", verdict, command, "", stdout, stderr)
    return result("unavailable", "unknown", tool, f"{tool} executable not found")


def dispatch(payload: dict[str, Any], tool_override: str | None = None) -> dict[str, Any]:
    kind = normalize_tool_id(str(payload.get("kind", "")))
    tool = normalize_tool_id(tool_override or payload.get("solver") or payload.get("tool"))
    if kind == "minizinc-validation" or tool in MINIZINC_TOOL_IDS:
        return validate_minizinc(payload, "minizinc" if tool == "auto" else tool)
    if kind in ("asp-validation", "clingo-validation") or tool in ASP_TOOL_IDS:
        return validate_asp(payload, "clingo" if tool == "auto" else tool)
    if kind in ("smtlib-validation", "smt-lib-validation") or tool in (
        "z3",
        "cvc5",
        "yices",
        "bitwuzla",
        "boolector",
        "mathsat",
        "optimathsat",
        "opensmt",
        "smtinterpol",
        "princess",
    ):
        return validate_smtlib(payload, tool)
    if kind in ("wcnf-validation", "dimacs-wcnf-validation", "maxsat-validation") or (
        tool in ("open-wbo", "maxhs") and any(key in payload for key in ("wcnf", "dimacs"))
    ):
        return validate_wcnf(payload, tool)
    if kind in ("opb-validation", "pseudo-boolean-validation") or tool in ("roundingsat",):
        return validate_opb(payload, tool)
    if kind in ("dimacs-validation", "dimacs-cnf-validation") or tool in (
        "kissat",
        "cadical",
        "cryptominisat",
        "minisat",
        "glucose",
        "maplesat",
        "varisat",
    ):
        return validate_dimacs(payload, tool)
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
