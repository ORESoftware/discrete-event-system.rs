#!/usr/bin/env python3
"""Direct LP/MIP bridge for installed solver CLIs.

The Rust optimization suite already cross-checks through Python APIs such as
SciPy and OR-Tools. This bridge exercises actual command-line solvers
(`highs`, `glpsol`, `scip`, `cbc`, LP-only `clp`, and optional commercial
CLIs such as `gurobi_cl` and `cplex`) on the same small validation models by
writing a CPLEX LP file, invoking the solver, and parsing the primal solution.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional, Sequence

from ip_mip_reference import expand_source_features
from lp_solve import dot, normalize_lp


COMMAND_ALIASES = {
    "glpk": ["glpsol"],
    "highs": ["highs"],
    "scip": ["scip"],
    "cbc": ["cbc"],
    "clp": ["clp"],
    "gurobi": ["gurobi_cl"],
    "cplex": ["cplex"],
    "xpress": ["optimizer", "xpress"],
    "lindo": ["lindo", "lindoapi"],
}

SUPPORTED_SOLVERS = {"glpk", "highs", "scip", "cbc", "clp", "gurobi", "cplex"}


def status_payload(status: str, solver: str, message: str = "") -> dict:
    return {
        "status": status,
        "solver": solver,
        "x": [],
        "objective": None,
        "message": message,
    }


def var_name(index: int) -> str:
    return f"x{index}"


def finite(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def term_expr(coefs: Sequence[float], names: Sequence[str]) -> str:
    parts: list[str] = []
    for coef, name in zip(coefs, names):
        coef = float(coef)
        if abs(coef) <= 1e-12:
            continue
        sign = "-" if coef < 0 else "+"
        mag = abs(coef)
        body = name if abs(mag - 1.0) <= 1e-12 else f"{mag:.12g} {name}"
        if not parts:
            parts.append(f"- {body}" if sign == "-" else body)
        else:
            parts.append(f"{sign} {body}")
    return " ".join(parts) if parts else f"0 {names[0] if names else 'x0'}"


def normalize_mip(raw: dict) -> tuple[str, list[float], list[list[float]], list[float], list[Optional[float]], list[Optional[float]], list[bool]]:
    p = expand_source_features(raw)
    c = [float(v) for v in p["c"]]
    n = len(c)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs = [float(v) for v in p.get("b", [])]
    if len(rows) != len(rhs):
        raise ValueError("MIP row/RHS length mismatch")
    for row in rows:
        if len(row) != n:
            raise ValueError("MIP row length mismatch")
    lb_raw = p.get("lb")
    ub_raw = p.get("ub")
    lbs = [0.0] * n if lb_raw is None else [0.0 if v is None else float(v) for v in lb_raw]
    ubs = [None] * n if ub_raw is None else [finite(v) for v in ub_raw]
    ints = [bool(v) for v in p.get("integer_vars", [False] * n)]
    if len(lbs) != n or len(ubs) != n or len(ints) != n:
        raise ValueError("MIP bound/integrality vector length mismatch")
    return p.get("sense", "max"), c, rows, rhs, lbs, ubs, ints


def write_cplex_lp(
    path: str,
    sense: str,
    c: Sequence[float],
    le_rows: Sequence[Sequence[float]],
    le_rhs: Sequence[float],
    eq_rows: Sequence[Sequence[float]],
    eq_rhs: Sequence[float],
    lbs: Sequence[Optional[float]],
    ubs: Sequence[Optional[float]],
    integer_vars: Sequence[bool],
) -> list[str]:
    n = len(c)
    names = [var_name(i) for i in range(n)]
    binary_vars = [
        i
        for i, is_int in enumerate(integer_vars)
        if is_int
        and (lbs[i] is None or abs(float(lbs[i])) <= 1e-12)
        and ubs[i] is not None
        and abs(float(ubs[i]) - 1.0) <= 1e-12
    ]
    binary_set = set(binary_vars)
    general_vars = [i for i, is_int in enumerate(integer_vars) if is_int and i not in binary_set]

    with open(path, "w", encoding="utf-8") as f:
        f.write("Maximize\n" if sense == "max" else "Minimize\n")
        f.write(f" obj: {term_expr(c, names)}\n")
        f.write("Subject To\n")
        for i, (row, rhs) in enumerate(zip(le_rows, le_rhs)):
            f.write(f" c{i}: {term_expr(row, names)} <= {float(rhs):.12g}\n")
        offset = len(le_rows)
        for i, (row, rhs) in enumerate(zip(eq_rows, eq_rhs)):
            f.write(f" e{i}: {term_expr(row, names)} = {float(rhs):.12g}\n")
        if not le_rows and not eq_rows:
            f.write(" c0: 0 x0 <= 0\n")
        f.write("Bounds\n")
        for i, name in enumerate(names):
            if i in binary_set:
                continue
            lb = lbs[i]
            ub = ubs[i]
            if lb is None and ub is None:
                f.write(f" {name} free\n")
            elif lb is None:
                f.write(f" {name} <= {float(ub):.12g}\n")
            elif ub is None:
                f.write(f" {float(lb):.12g} <= {name}\n")
            else:
                f.write(f" {float(lb):.12g} <= {name} <= {float(ub):.12g}\n")
        if general_vars:
            f.write("General\n")
            f.write(" " + " ".join(names[i] for i in general_vars) + "\n")
        if binary_vars:
            f.write("Binary\n")
            f.write(" " + " ".join(names[i] for i in binary_vars) + "\n")
        f.write("End\n")
    return names


def parse_highs_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    status = "unknown"
    in_primal = False
    in_columns = False
    remaining = 0
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    for i, line in enumerate(lines):
        if line == "Model status" and i + 1 < len(lines):
            status = lines[i + 1].lower()
        if line == "# Primal solution values":
            in_primal = True
            continue
        if in_primal and line.startswith("# Columns"):
            in_columns = True
            remaining = int(line.split()[2])
            continue
        if in_columns:
            if remaining <= 0 or line.startswith("# Rows"):
                break
            if line.startswith("#"):
                in_columns = False
                continue
            parts = line.split()
            if len(parts) >= 2 and parts[0].startswith("x"):
                idx = int(parts[0][1:])
                if 0 <= idx < n:
                    x[idx] = float(parts[1])
            remaining -= 1
    return status, x


def parse_glpk_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    status = "unknown"
    in_named_columns = False
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3 and parts[0] == "c" and parts[1] == "Status:":
                status = " ".join(parts[2:]).lower()
            elif len(parts) >= 2 and parts[0] == "Status:":
                status = " ".join(parts[1:]).lower()
            elif "Column name" in line:
                in_named_columns = True
            elif in_named_columns and stripped_starts(line, ("Integer feasibility", "KKT.", "End of output")):
                in_named_columns = False
            elif in_named_columns and len(parts) >= 3 and parts[0].isdigit() and parts[1].startswith("x"):
                suffix = parts[1][1:]
                if suffix.isdigit():
                    idx = int(suffix)
                    if 0 <= idx < n:
                        for token in parts[2:]:
                            if token != "*" and _is_number(token):
                                x[idx] = float(token)
                                break
            elif len(parts) >= 3 and parts[0] == "j":
                idx = int(parts[1]) - 1
                if 0 <= idx < n:
                    if len(parts) >= 4 and not _is_number(parts[2]):
                        x[idx] = float(parts[3])
                    else:
                        x[idx] = float(parts[2])
    return status, x


def parse_scip_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    status = "unknown"
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("solution status:"):
                status = stripped.split(":", 1)[1].strip().lower()
            elif stripped.startswith("x"):
                parts = stripped.split()
                if len(parts) >= 2 and parts[0][1:].isdigit():
                    idx = int(parts[0][1:])
                    if 0 <= idx < n:
                        x[idx] = float(parts[1])
    return status, x


def parse_cbc_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    status = "unknown"
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            stripped = line.strip()
            if line_no == 0 and stripped:
                status = stripped.lower()
                continue
            parts = stripped.split()
            if parts and parts[0] == "**":
                parts = parts[1:]
            if len(parts) >= 3 and parts[0].lstrip("-").isdigit() and parts[1].startswith("x"):
                suffix = parts[1][1:]
                if suffix.isdigit():
                    idx = int(suffix)
                    if 0 <= idx < n:
                        x[idx] = float(parts[2])
    return status, x


def parse_named_solution(path: str, n: int, default_status: str = "optimal") -> tuple[str, list[float]]:
    x = [0.0] * n
    status = default_status
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                lower = stripped.lower()
                if "objective" in lower and "value" in lower:
                    status = default_status
                continue
            parts = stripped.split()
            if len(parts) >= 2 and parts[0].startswith("x") and parts[0][1:].isdigit():
                idx = int(parts[0][1:])
                if 0 <= idx < n and _is_number(parts[1]):
                    x[idx] = float(parts[1])
    return status, x


def parse_cplex_solution(path: str, n: int) -> tuple[str, list[float]]:
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return parse_named_solution(path, n)

    x = [0.0] * n
    status = "optimal"
    for elem in root.iter():
        tag = elem.tag.rsplit("}", 1)[-1]
        if tag == "header":
            status = (
                elem.attrib.get("solutionStatusString")
                or elem.attrib.get("solutionStatus")
                or status
            ).lower()
        elif tag == "variable":
            name = elem.attrib.get("name", "")
            value = elem.attrib.get("value")
            if name.startswith("x") and name[1:].isdigit() and value is not None:
                idx = int(name[1:])
                if 0 <= idx < n:
                    x[idx] = float(value)
    return status, x


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def stripped_starts(text: str, prefixes: Sequence[str]) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(prefix) for prefix in prefixes)


def solver_available(solver: str) -> bool:
    return solver_command(solver) is not None


def solver_command(solver: str) -> Optional[str]:
    for command in COMMAND_ALIASES.get(solver, [solver]):
        resolved = shutil.which(command)
        if resolved is not None:
            return resolved
    return None


def run_solver(solver: str, model_path: str, solution_path: str, time_limit: float) -> tuple[str, str]:
    command = solver_command(solver)
    if command is None:
        raise ValueError(f"{solver} executable not found")
    if solver == "highs":
        cmd = [
            command,
            "--model_file",
            model_path,
            "--solution_file",
            solution_path,
            "--time_limit",
            str(time_limit),
        ]
    elif solver == "glpk":
        cmd = [command, "--lp", model_path, "-o", solution_path, "--tmlim", str(max(1, int(math.ceil(time_limit))))]
    elif solver == "scip":
        cmd = [
            command,
            "-q",
            "-c",
            f"read {model_path}",
            "-c",
            f"set limits time {time_limit}",
            "-c",
            "optimize",
            "-c",
            f"write solution {solution_path}",
            "-c",
            "quit",
        ]
    elif solver == "cbc":
        cmd = [
            command,
            model_path,
            "-seconds",
            str(time_limit),
            "-solve",
            "-solution",
            solution_path,
        ]
    elif solver == "clp":
        cmd = [
            command,
            model_path,
            "-seconds",
            str(time_limit),
            "-solve",
            "-solution",
            solution_path,
        ]
    elif solver == "gurobi":
        cmd = [
            command,
            f"ResultFile={solution_path}",
            f"TimeLimit={time_limit}",
            model_path,
        ]
    elif solver == "cplex":
        cmd = [
            command,
            "-c",
            f"read {model_path}",
            f"set timelimit {time_limit}",
            "optimize",
            f"write {solution_path}",
            "quit",
        ]
    else:
        raise ValueError(f"unknown CLI solver '{solver}'")
    run = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        check=False,
        cwd=os.path.dirname(model_path),
    )
    return run.stdout, run.stderr


def solve(kind: str, solver: str, raw: dict, time_limit: float) -> dict:
    if not solver_available(solver):
        return status_payload("unavailable", f"{solver}:cli", f"{solver} executable not found")
    if solver not in SUPPORTED_SOLVERS:
        return status_payload(
            "unavailable",
            f"{solver}:cli",
            f"{solver} executable found, but this bridge does not yet know the non-interactive solve command",
        )

    if kind == "lp":
        lp = raw.get("lp", raw)
        sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs = normalize_lp(lp)
        integer_vars = [False] * len(c)
    elif kind == "mip":
        if solver == "clp":
            return status_payload("unavailable", "clp:cli", "CLP is LP-only")
        sense, c, a_ub, b_ub, lbs, ubs, integer_vars = normalize_mip(raw)
        a_eq, b_eq = [], []
    else:
        raise ValueError("kind must be 'lp' or 'mip'")

    with tempfile.TemporaryDirectory(prefix="ores-linear-cli-") as tmp:
        model_path = os.path.join(tmp, "model.lp")
        solution_path = os.path.join(tmp, f"{solver}.sol")
        write_cplex_lp(model_path, sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs, integer_vars)
        stdout, stderr = run_solver(solver, model_path, solution_path, time_limit)
        if not os.path.exists(solution_path):
            return status_payload("unavailable", f"{solver}:cli", (stderr or stdout).strip())
        if solver == "highs":
            status, x = parse_highs_solution(solution_path, len(c))
        elif solver == "glpk":
            status, x = parse_glpk_solution(solution_path, len(c))
        elif solver == "scip":
            status, x = parse_scip_solution(solution_path, len(c))
        elif solver == "gurobi":
            status, x = parse_named_solution(solution_path, len(c))
        elif solver == "cplex":
            status, x = parse_cplex_solution(solution_path, len(c))
        else:
            status, x = parse_cbc_solution(solution_path, len(c))

    optimal = "optimal" in status
    if not optimal:
        return status_payload("infeasible" if "infeasible" in status else "unavailable", f"{solver}:cli", status)
    return {
        "status": "optimal",
        "solver": f"{solver}:cli",
        "x": x,
        "objective": dot(c, x),
        "message": status,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["lp", "mip"], required=True)
    parser.add_argument("--solver", choices=sorted(COMMAND_ALIASES.keys()), required=True)
    parser.add_argument("--problem")
    parser.add_argument("--time-limit", type=float, default=10.0)
    args = parser.parse_args()
    try:
        if args.problem:
            with open(args.problem, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = json.load(sys.stdin)
        print(json.dumps(solve(args.kind, args.solver, raw, args.time_limit), allow_nan=True))
        return 0
    except Exception as exc:
        print(json.dumps(status_payload("numerical-error", f"{args.solver}:cli", str(exc)), allow_nan=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
