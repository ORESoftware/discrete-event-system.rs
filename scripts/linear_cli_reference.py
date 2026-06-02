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
    "lindo": ["runlindo", "lindo", "lindoapi"],
}

SUPPORTED_SOLVERS = {
    "glpk",
    "highs",
    "scip",
    "cbc",
    "clp",
    "gurobi",
    "cplex",
    "xpress",
    "lindo",
}


def solver_env_names(solver: str) -> list[str]:
    upper = solver.upper().replace("-", "_")
    return [
        f"ORES_{upper}_BIN",
        f"DES_{upper}_BIN",
        f"{upper}_BIN",
    ]


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


def mps_number(value: float) -> str:
    return f"{float(value):.12g}"


def is_binary_var(
    index: int,
    lbs: Sequence[Optional[float]],
    ubs: Sequence[Optional[float]],
    integer_vars: Sequence[bool],
) -> bool:
    return (
        bool(integer_vars[index])
        and (lbs[index] is None or abs(float(lbs[index])) <= 1e-12)
        and ubs[index] is not None
        and abs(float(ubs[index]) - 1.0) <= 1e-12
    )


def write_mps(
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
    le_names = [f"c{i}" for i in range(len(le_rows))]
    eq_names = [f"e{i}" for i in range(len(eq_rows))]
    objective = [float(v) if sense == "min" else -float(v) for v in c]
    integer_indices = [i for i, is_int in enumerate(integer_vars) if is_int]
    integer_set = set(integer_indices)

    def write_column(f, index: int) -> None:
        name = names[index]
        obj = objective[index]
        if abs(obj) > 1e-12:
            f.write(f"    {name:<8}  OBJ       {mps_number(obj)}\n")
        for row_name, row in zip(le_names, le_rows):
            coef = float(row[index])
            if abs(coef) > 1e-12:
                f.write(f"    {name:<8}  {row_name:<8}  {mps_number(coef)}\n")
        for row_name, row in zip(eq_names, eq_rows):
            coef = float(row[index])
            if abs(coef) > 1e-12:
                f.write(f"    {name:<8}  {row_name:<8}  {mps_number(coef)}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("NAME          ORES\n")
        f.write("ROWS\n")
        f.write(" N  OBJ\n")
        for row_name in le_names:
            f.write(f" L  {row_name}\n")
        for row_name in eq_names:
            f.write(f" E  {row_name}\n")
        f.write("COLUMNS\n")
        for i in range(n):
            if i not in integer_set:
                write_column(f, i)
        if integer_indices:
            f.write("    MARK0000  'MARKER'                 'INTORG'\n")
            for i in integer_indices:
                write_column(f, i)
            f.write("    MARK0001  'MARKER'                 'INTEND'\n")
        if le_rows or eq_rows:
            f.write("RHS\n")
            for row_name, rhs in zip(le_names, le_rhs):
                f.write(f"    RHS1      {row_name:<8}  {mps_number(rhs)}\n")
            for row_name, rhs in zip(eq_names, eq_rhs):
                f.write(f"    RHS1      {row_name:<8}  {mps_number(rhs)}\n")
        f.write("BOUNDS\n")
        for i, name in enumerate(names):
            lb = lbs[i]
            ub = ubs[i]
            if is_binary_var(i, lbs, ubs, integer_vars):
                f.write(f" BV BND1      {name}\n")
                continue
            if lb is None and ub is None:
                f.write(f" FR BND1      {name}\n")
            elif lb is None:
                f.write(f" MI BND1      {name}\n")
                f.write(f" UP BND1      {name:<8}  {mps_number(ub)}\n")
            elif ub is None:
                if abs(float(lb)) > 1e-12:
                    f.write(f" LO BND1      {name:<8}  {mps_number(lb)}\n")
            elif abs(float(lb) - float(ub)) <= 1e-12:
                f.write(f" FX BND1      {name:<8}  {mps_number(lb)}\n")
            else:
                if abs(float(lb)) > 1e-12:
                    f.write(f" LO BND1      {name:<8}  {mps_number(lb)}\n")
                f.write(f" UP BND1      {name:<8}  {mps_number(ub)}\n")
        f.write("ENDATA\n")
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


def parse_report_solution(path: str, n: int, default_status: str = "optimal") -> tuple[str, list[float]]:
    x = [0.0] * n
    status = default_status
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            stripped = line.strip()
            lower = stripped.lower()
            if not stripped:
                continue
            if "infeasible" in lower and "not infeasible" not in lower:
                status = "infeasible"
            elif "unbounded" in lower:
                status = "unbounded"
            elif "optimal" in lower or "global optimum" in lower:
                status = default_status

            parts = stripped.replace(":", " ").split()
            for pos, token in enumerate(parts):
                name = token.rstrip(",;")
                if not (name.startswith("x") and name[1:].isdigit()):
                    continue
                idx = int(name[1:])
                if not 0 <= idx < n:
                    break
                for value_token in parts[pos + 1 :]:
                    value = value_token.strip(",;")
                    if value == "*" or value in ("=", ":"):
                        continue
                    if _is_number(value):
                        x[idx] = float(value)
                        break
                break
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


def classify_status(status: str, stdout: str, stderr: str) -> str:
    parsed = status.lower()
    if "primal infeasible" in parsed or ("infeasible" in parsed and "dual" not in parsed):
        return "infeasible"
    if "dual infeasible" in parsed or "unbounded" in parsed:
        return "unbounded"
    if "optimal" in parsed:
        return "optimal"

    text = f"{stdout}\n{stderr}".lower()
    infeasible_markers = (
        "no primal feasible",
        "primal infeasible",
        "linear relaxation infeasible",
        "no feasible solution",
        "no solution exists",
        "integer infeasible",
        "problem has no feasible",
    )
    if any(marker in text for marker in infeasible_markers):
        return "infeasible"
    unbounded_markers = (
        "has unbounded solution",
        "linear relaxation unbounded",
        "dual infeasible",
        "unbounded",
    )
    if any(marker in text for marker in unbounded_markers):
        return "unbounded"
    return "unknown"


def solver_available(solver: str) -> bool:
    return solver_command(solver) is not None


def solver_command(solver: str) -> Optional[str]:
    saw_configured = False
    for env_name in solver_env_names(solver):
        configured = os.environ.get(env_name)
        if configured:
            saw_configured = True
            expanded = os.path.expanduser(configured)
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return expanded
            resolved = shutil.which(configured)
            if resolved is not None:
                return resolved
    if saw_configured:
        return None
    for command in COMMAND_ALIASES.get(solver, [solver]):
        resolved = shutil.which(command)
        if resolved is not None:
            return resolved
    return None


def format_float(value: float) -> str:
    return f"{value:.17g}"


def normalize_node_limit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if value <= 0:
        raise ValueError("node limit must be positive")
    return value


def normalize_relative_gap(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if not math.isfinite(value) or value < 0.0:
        raise ValueError("relative gap must be finite and non-negative")
    return value


def normalize_threads(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if value <= 0:
        raise ValueError("thread count must be positive")
    return value


def normalize_random_seed(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    if value < 0 or value > 2_147_483_647:
        raise ValueError("random seed must be in [0, 2147483647]")
    return value


def run_solver(
    kind: str,
    solver: str,
    model_path: str,
    solution_path: str,
    time_limit: float,
    model_format: str,
    node_limit: Optional[int] = None,
    relative_gap: Optional[float] = None,
    threads: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> tuple[str, str]:
    command = solver_command(solver)
    if command is None:
        raise ValueError(f"{solver} executable not found")
    input_text = None
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
        if (
            node_limit is not None
            or relative_gap is not None
            or threads is not None
            or random_seed is not None
        ):
            options_path = os.path.join(os.path.dirname(model_path), "highs.options")
            with open(options_path, "w", encoding="utf-8") as f:
                if node_limit is not None:
                    f.write(f"mip_max_nodes = {node_limit}\n")
                if relative_gap is not None:
                    f.write(f"mip_rel_gap = {format_float(relative_gap)}\n")
                if threads is not None:
                    f.write(f"threads = {threads}\n")
                if random_seed is not None:
                    f.write(f"random_seed = {random_seed}\n")
            cmd.extend(["--options_file", options_path])
    elif solver == "glpk":
        format_arg = "--freemps" if model_format == "mps" else "--lp"
        solution_arg = "--write" if kind == "lp" else "-o"
        cmd = [
            command,
            format_arg,
            model_path,
            solution_arg,
            solution_path,
            "--tmlim",
            str(max(1, int(math.ceil(time_limit)))),
        ]
        if relative_gap is not None:
            cmd.extend(["--mipgap", format_float(relative_gap)])
    elif solver == "scip":
        commands = [
            f"read {model_path}",
            f"set limits time {time_limit}",
        ]
        if node_limit is not None:
            commands.append(f"set limits nodes {node_limit}")
        if relative_gap is not None:
            commands.append(f"set limits gap {format_float(relative_gap)}")
        if threads is not None:
            commands.append(f"set parallel maxnthreads {threads}")
        if random_seed is not None:
            commands.append(f"set randomization randomseedshift {random_seed}")
        commands.extend(["optimize", f"write solution {solution_path}", "quit"])
        cmd = [command, "-q"]
        for scip_command in commands:
            cmd.extend(["-c", scip_command])
    elif solver == "cbc":
        cmd = [
            command,
            model_path,
            "-seconds",
            str(time_limit),
        ]
        if node_limit is not None:
            cmd.extend(["-maxNodes", str(node_limit)])
        if relative_gap is not None:
            cmd.extend(["-ratioGap", format_float(relative_gap)])
        if threads is not None:
            cmd.extend(["-threads", str(threads)])
        if random_seed is not None:
            cmd.extend(["-randomSeed", str(random_seed), "-randomCbcSeed", str(random_seed)])
        cmd.extend(["-solve", "-solution", solution_path])
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
        ]
        if node_limit is not None:
            cmd.append(f"NodeLimit={node_limit}")
        if relative_gap is not None:
            cmd.append(f"MIPGap={format_float(relative_gap)}")
        if threads is not None:
            cmd.append(f"Threads={threads}")
        if random_seed is not None:
            cmd.append(f"Seed={random_seed}")
        cmd.append(model_path)
    elif solver == "cplex":
        commands = [
            f"read {model_path}",
            f"set timelimit {time_limit}",
        ]
        if node_limit is not None:
            commands.append(f"set mip limits nodes {node_limit}")
        if relative_gap is not None:
            commands.append(f"set mip tolerances mipgap {format_float(relative_gap)}")
        if threads is not None:
            commands.append(f"set threads {threads}")
        if random_seed is not None:
            commands.append(f"set randomseed {random_seed}")
        commands.extend(["optimize", f"write {solution_path}", "quit"])
        cmd = [command, "-c", *commands]
    elif solver == "xpress":
        solve_command = "mipoptimize" if kind == "mip" else "lpoptimize"
        commands = [
            f"readprob {model_path}",
            f"setparam MAXTIME {format_float(time_limit)}",
        ]
        if node_limit is not None:
            commands.append(f"setparam MAXNODE {node_limit}")
        if relative_gap is not None:
            commands.append(f"setparam MIPRELSTOP {format_float(relative_gap)}")
        if threads is not None:
            commands.append(f"setparam THREADS {threads}")
        if random_seed is not None:
            commands.append(f"setparam RANDOMSEED {random_seed}")
        commands.extend([solve_command, f"writeprtsol {solution_path}", "quit"])
        cmd = [command]
        input_text = "\n".join(commands) + "\n"
    elif solver == "lindo":
        cmd = [command, model_path, "-sol"]
    else:
        raise ValueError(f"unknown CLI solver '{solver}'")
    run = subprocess.run(
        cmd,
        text=True,
        input=input_text,
        capture_output=True,
        check=False,
        cwd=os.path.dirname(model_path),
        timeout=max(5.0, float(time_limit) + 5.0),
    )
    if solver == "lindo" and not os.path.exists(solution_path):
        automatic_solution_path = os.path.splitext(model_path)[0] + ".sol"
        if os.path.exists(automatic_solution_path):
            shutil.copyfile(automatic_solution_path, solution_path)
    return run.stdout, run.stderr


def solve(
    kind: str,
    solver: str,
    raw: dict,
    time_limit: float,
    model_format: str,
    node_limit: Optional[int] = None,
    relative_gap: Optional[float] = None,
    threads: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> dict:
    if not solver_available(solver):
        return status_payload("unavailable", f"{solver}:cli", f"{solver} executable not found")
    if solver not in SUPPORTED_SOLVERS:
        return status_payload(
            "unavailable",
            f"{solver}:cli",
            f"{solver} executable found, but this bridge does not yet know the non-interactive solve command",
        )

    solver_node_limit = None
    solver_relative_gap = None
    solver_threads = normalize_threads(threads)
    solver_random_seed = normalize_random_seed(random_seed)
    if kind == "lp":
        lp = raw.get("lp", raw)
        sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs = normalize_lp(lp)
        integer_vars = [False] * len(c)
    elif kind == "mip":
        if solver == "clp":
            return status_payload("unavailable", "clp:cli", "CLP is LP-only")
        solver_node_limit = normalize_node_limit(node_limit)
        solver_relative_gap = normalize_relative_gap(relative_gap)
        sense, c, a_ub, b_ub, lbs, ubs, integer_vars = normalize_mip(raw)
        a_eq, b_eq = [], []
    else:
        raise ValueError("kind must be 'lp' or 'mip'")

    with tempfile.TemporaryDirectory(prefix="ores-linear-cli-") as tmp:
        extension = "mps" if model_format == "mps" else "lp"
        model_path = os.path.join(tmp, f"model.{extension}")
        solution_path = os.path.join(tmp, f"{solver}.sol")
        if model_format == "mps":
            write_mps(model_path, sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs, integer_vars)
        else:
            write_cplex_lp(model_path, sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs, integer_vars)
        stdout, stderr = run_solver(
            kind,
            solver,
            model_path,
            solution_path,
            time_limit,
            model_format,
            solver_node_limit,
            solver_relative_gap,
            solver_threads,
            solver_random_seed,
        )
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
        elif solver in ("xpress", "lindo"):
            status, x = parse_report_solution(solution_path, len(c))
        else:
            status, x = parse_cbc_solution(solution_path, len(c))

    classified = classify_status(status, stdout, stderr)
    if classified != "optimal":
        return status_payload(
            classified if classified in ("infeasible", "unbounded") else "unavailable",
            f"{solver}:cli",
            status,
        )
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
    parser.add_argument("--model-format", choices=["lp", "mps"], default="lp")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--node-limit", type=int)
    parser.add_argument("--relative-gap", type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--random-seed", type=int)
    args = parser.parse_args()
    try:
        if args.problem:
            with open(args.problem, "r", encoding="utf-8") as f:
                raw = json.load(f)
        else:
            raw = json.load(sys.stdin)
        print(
            json.dumps(
                solve(
                    args.kind,
                    args.solver,
                    raw,
                    args.time_limit,
                    args.model_format,
                    args.node_limit,
                    args.relative_gap,
                    args.threads,
                    args.random_seed,
                ),
                allow_nan=True,
            )
        )
        return 0
    except Exception as exc:
        print(json.dumps(status_payload("numerical-error", f"{args.solver}:cli", str(exc)), allow_nan=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
