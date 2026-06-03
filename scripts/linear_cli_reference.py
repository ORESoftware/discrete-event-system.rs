#!/usr/bin/env python3
"""Direct LP/MIP bridge for installed solver CLIs.

The Rust optimization suite already cross-checks through Python APIs such as
SciPy and OR-Tools. This bridge exercises actual command-line solvers
(`highs`, `glpsol`, `scip`, `cbc`, LP-only `clp`/`soplex`/`qsopt_ex`, `lp_solve`, and optional commercial
CLIs such as `gurobi_cl`, `cplex`, FICO Xpress `optimizer`, and LINDO
`runlindo`) on the same small validation models by writing a solver-readable
LP/MPS file, invoking the solver, and parsing the primal solution.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
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
    "soplex": ["soplex"],
    "qsopt-ex": ["qsopt_ex", "qsopt-ex", "qsopt", "esolver"],
    "lp-solve": ["lp_solve", "lp-solve", "lpsolve"],
    "gurobi": ["gurobi_cl"],
    "cplex": ["cplex"],
    "xpress": ["optimizer", "xpress"],
    "lindo": ["runlindo", "lindo", "lindoapi"],
}

COMMAND_ENV_VARS = {
    "glpk": ["GLPSOL_CMD", "GLPK_CMD", "ORES_GLPK_CMD", "ORES_GLPK_BIN", "DES_GLPK_BIN", "GLPK_BIN"],
    "highs": ["HIGHS_CMD", "ORES_HIGHS_CMD", "ORES_HIGHS_BIN", "DES_HIGHS_BIN", "HIGHS_BIN"],
    "scip": ["SCIP_CMD", "ORES_SCIP_CMD", "ORES_SCIP_BIN", "DES_SCIP_BIN", "SCIP_BIN"],
    "cbc": ["CBC_CMD", "ORES_CBC_CMD", "ORES_CBC_BIN", "DES_CBC_BIN", "CBC_BIN"],
    "clp": ["CLP_CMD", "ORES_CLP_CMD", "ORES_CLP_BIN", "DES_CLP_BIN", "CLP_BIN"],
    "soplex": ["SOPLEX_CMD", "ORES_SOPLEX_CMD", "ORES_SOPLEX_BIN", "DES_SOPLEX_BIN", "SOPLEX_BIN"],
    "qsopt-ex": ["QSOPT_EX_CMD", "QSOPT_CMD", "ORES_QSOPT_EX_CMD", "ORES_QSOPT_EX_BIN", "DES_QSOPT_EX_BIN", "QSOPT_EX_BIN"],
    "lp-solve": ["LP_SOLVE_CMD", "LPSOLVE_CMD", "ORES_LP_SOLVE_CMD", "ORES_LPSOLVE_BIN", "DES_LPSOLVE_BIN", "LPSOLVE_BIN"],
    "gurobi": ["GUROBI_CL_CMD", "GUROBI_CMD", "ORES_GUROBI_CMD", "ORES_GUROBI_BIN", "DES_GUROBI_BIN", "GUROBI_BIN"],
    "cplex": ["CPLEX_CMD", "ORES_CPLEX_CMD", "ORES_CPLEX_BIN", "DES_CPLEX_BIN", "CPLEX_BIN"],
    "xpress": ["XPRESS_CMD", "XPRESS_OPTIMIZER_CMD", "ORES_XPRESS_CMD", "ORES_XPRESS_BIN", "DES_XPRESS_BIN", "XPRESS_BIN"],
    "lindo": ["RUNLINDO_CMD", "LINDO_CMD", "LINDOAPI_CMD", "ORES_LINDO_CMD", "ORES_LINDO_BIN", "DES_LINDO_BIN", "LINDO_BIN"],
}

COMMAND_DIR_ENV_VARS = {
    "glpk": ["GLPK_DIR", "GLPK_HOME"],
    "highs": ["HIGHS_DIR", "HIGHS_HOME"],
    "scip": ["SCIPOPTDIR", "SCIP_DIR", "SCIP_HOME"],
    "cbc": ["CBC_DIR", "CBC_HOME", "COINOR_DIR", "COINOR_HOME"],
    "clp": ["CLP_DIR", "CLP_HOME", "COINOR_DIR", "COINOR_HOME"],
    "soplex": ["SOPLEX_DIR", "SOPLEX_HOME"],
    "qsopt-ex": ["QSOPT_EX_DIR", "QSOPT_EX_HOME", "QSOPT_DIR", "QSOPT_HOME"],
    "lp-solve": ["LP_SOLVE_DIR", "LPSOLVE_DIR", "LP_SOLVE_HOME", "LPSOLVE_HOME"],
    "gurobi": ["GUROBI_HOME"],
    "cplex": ["CPLEX_STUDIO_DIR", "CPLEX_HOME"],
    "xpress": ["XPRESSDIR", "XPRESS_DIR", "XPRESS_HOME"],
    "lindo": ["LINDO_HOME", "LINDO_DIR", "LINDOAPI_HOME", "LINDOAPI_DIR"],
}

SUPPORTED_SOLVERS = {
    "glpk",
    "highs",
    "scip",
    "cbc",
    "clp",
    "soplex",
    "qsopt-ex",
    "lp-solve",
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


def basis_status_from_token(token: object) -> Optional[str]:
    text = str(token).strip().lower()
    highs_codes = {
        "0": "at_lower",
        "1": "basic",
        "2": "at_upper",
        "3": "zero",
        "4": "nonbasic",
    }
    glpk_codes = {
        "b": "basic",
        "bs": "basic",
        "l": "at_lower",
        "nl": "at_lower",
        "u": "at_upper",
        "nu": "at_upper",
        "f": "free",
        "nf": "free",
        "s": "fixed",
        "ns": "fixed",
    }
    named = {
        "basic": "basic",
        "lower": "at_lower",
        "at_lower": "at_lower",
        "upper": "at_upper",
        "at_upper": "at_upper",
        "zero": "zero",
        "nonbasic": "nonbasic",
        "free": "free",
        "fixed": "fixed",
        "superbasic": "superbasic",
    }
    return highs_codes.get(text) or glpk_codes.get(text) or named.get(text)


def status_payload(
    status: str,
    solver: str,
    message: str = "",
    solver_version: Optional[str] = None,
) -> dict:
    payload = {
        "status": status,
        "solver": solver,
        "x": [],
        "objective": None,
        "message": message,
    }
    if solver_version is not None:
        payload["solverVersion"] = solver_version
    return payload


def var_name(index: int) -> str:
    return f"x{index}"


def finite(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def normalized_node_limit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(0, int(value))


def normalized_solution_limit(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(1, int(value))


def normalized_solution_pool_size(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(1, int(value))


def normalized_relative_gap(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


def normalized_absolute_gap(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else None


def normalized_objective_limit(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def normalized_tolerance(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) and value > 0.0 else None


def normalized_threads(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(1, int(value))


def normalized_random_seed(value: Optional[int]) -> Optional[int]:
    if value is None:
        return None
    return max(0, int(value))


def normalized_presolve(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    aliases = {
        "auto": "auto",
        "default": "auto",
        "choose": "auto",
        "on": "on",
        "true": "on",
        "1": "on",
        "off": "off",
        "false": "off",
        "0": "off",
    }
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise ValueError(f"unknown presolve setting '{value}'")
    return normalized


def normalized_mip_switch(value: Optional[str], name: str) -> Optional[str]:
    if value is None:
        return None
    aliases = {
        "auto": "auto",
        "default": "auto",
        "on": "on",
        "true": "on",
        "1": "on",
        "off": "off",
        "false": "off",
        "0": "off",
    }
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise ValueError(f"unknown {name} setting '{value}'")
    return normalized


def normalized_lp_algorithm(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    aliases = {
        "simplex": "simplex",
        "dual-simplex": "simplex",
        "primal-simplex": "simplex",
        "ipm": "ipm",
        "interior": "ipm",
        "interior-point": "ipm",
        "barrier": "ipm",
    }
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise ValueError(f"unknown lp_algorithm '{value}'")
    return normalized


def normalized_branch_rule(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    aliases = {
        "first": "first-fractional",
        "first-fractional": "first-fractional",
        "most": "most-fractional",
        "most-fractional": "most-fractional",
    }
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise ValueError(f"unknown branch_rule '{value}'")
    return normalized


def normalized_branch_priorities(value: Optional[Sequence[int]], n: int) -> Optional[list[int]]:
    if value is None:
        return None
    if len(value) != n:
        raise ValueError(
            f"branch_priorities length {len(value)} does not match variable count {n}"
        )
    return [int(v) for v in value]


def normalized_node_selection(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    aliases = {
        "dfs": "dfs",
        "depth": "dfs",
        "best-bound": "best-bound",
        "best_bound": "best-bound",
        "bestb": "best-bound",
    }
    normalized = aliases.get(value.strip().lower())
    if normalized is None:
        raise ValueError(f"unknown node_selection '{value}'")
    return normalized


def normalized_mip_start(value: Optional[Sequence[float]], n: int) -> Optional[list[float]]:
    if value is None:
        return None
    if len(value) != n:
        raise ValueError(f"mip_start length {len(value)} does not match variable count {n}")
    start = [float(v) for v in value]
    if any(not math.isfinite(v) for v in start):
        raise ValueError("mip_start values must be finite")
    return start


def parse_mip_start_arg(text: Optional[str]) -> Optional[list[float]]:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] == "[":
        value = json.loads(stripped)
        if not isinstance(value, list):
            raise ValueError("--mip-start JSON must be a list")
        return [float(v) for v in value]
    return [float(part.strip()) for part in stripped.split(",") if part.strip()]


def parse_int_list_arg(text: Optional[str], name: str) -> Optional[list[int]]:
    if text is None:
        return None
    stripped = text.strip()
    if not stripped:
        return None
    if stripped[0] == "[":
        value = json.loads(stripped)
        if not isinstance(value, list):
            raise ValueError(f"--{name} JSON must be a list")
        return [int(v) for v in value]
    return [int(part.strip()) for part in stripped.split(",") if part.strip()]


def payload_solution(x: Sequence[float], objective: float) -> dict:
    return {
        "x": [float(value) for value in x],
        "objective": float(objective),
    }


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
    lazy_constraints = p.get("lazy_constraints") or p.get("lazyConstraints") or []
    for idx, constraint in enumerate(lazy_constraints):
        row = [float(v) for v in constraint["coefs"]]
        if len(row) != n:
            raise ValueError(f"lazy constraint {idx} coefficient length does not match variable count")
        rows.append(row)
        rhs.append(float(constraint["rhs"]))
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


def plain_mip_payload(
    sense: str,
    c: Sequence[float],
    rows: Sequence[Sequence[float]],
    rhs: Sequence[float],
    lbs: Sequence[Optional[float]],
    ubs: Sequence[Optional[float]],
    integer_vars: Sequence[bool],
) -> dict:
    return {
        "sense": sense,
        "c": [float(value) for value in c],
        "a": [[float(value) for value in row] for row in rows],
        "b": [float(value) for value in rhs],
        "lb": [None if value is None else float(value) for value in lbs],
        "ub": [None if value is None else float(value) for value in ubs],
        "integer_vars": [bool(value) for value in integer_vars],
    }


def normalized_multi_objectives(raw: dict, n: int) -> list[dict]:
    objectives = raw.get("multi_objectives") or raw.get("multiObjectives") or []
    normalized: list[dict] = []
    for idx, objective in enumerate(objectives):
        coeffs = [float(v) for v in objective["c"]]
        if len(coeffs) < n:
            coeffs.extend([0.0] * (n - len(coeffs)))
        if len(coeffs) != n:
            raise ValueError(
                f"multi_objectives[{idx}] coefficient length {len(coeffs)} "
                f"does not match variable count {n}"
            )
        sense = objective.get("sense", "max").strip().lower()
        if sense not in {"max", "min"}:
            raise ValueError(f"multi_objectives[{idx}] has unknown sense '{sense}'")
        normalized.append(
            {
                "sense": sense,
                "c": coeffs,
                "name": objective.get("name", f"multi_objective_{idx}"),
            }
        )
    return normalized


def add_objective_lock_rows(
    working: dict,
    coeffs: Sequence[float],
    optimum: float,
    name: str,
) -> None:
    rows = [[float(value) for value in row] for row in working.get("a", [])]
    rhs = [float(value) for value in working.get("b", [])]
    con_names = list(working.get("con_names") or [f"c{i}" for i in range(len(rows))])
    rows.append([float(value) for value in coeffs])
    rhs.append(float(optimum))
    con_names.append(f"{name}_le")
    rows.append([-float(value) for value in coeffs])
    rhs.append(float(-optimum))
    con_names.append(f"{name}_ge")
    working["a"] = rows
    working["b"] = rhs
    working["con_names"] = con_names


def solution_pool_integer_indices(integer_vars: Sequence[bool]) -> list[int]:
    return [idx for idx, is_integer in enumerate(integer_vars) if is_integer]


def validate_solution_pool_bounds(
    lbs: Sequence[Optional[float]],
    ubs: Sequence[Optional[float]],
    integer_indices: Sequence[int],
) -> Optional[str]:
    if not integer_indices:
        return "solution pool requires at least one integer variable"
    for idx in integer_indices:
        lb = 0.0 if lbs[idx] is None else float(lbs[idx])
        ub = ubs[idx]
        if ub is None or not math.isfinite(float(ub)):
            return f"solution pool requires finite upper bound for integer variable x{idx}"
        if not math.isfinite(lb):
            return f"solution pool requires finite lower bound for integer variable x{idx}"
        if abs(lb - round(lb)) > 1e-9 or abs(float(ub) - round(float(ub))) > 1e-9:
            return f"solution pool requires integral bounds for integer variable x{idx}"
    return None


def solution_pool_assignment_key(
    x: Sequence[float],
    integer_indices: Sequence[int],
) -> tuple[int, ...]:
    return tuple(int(round(float(x[idx]))) for idx in integer_indices)


def add_solution_pool_no_good_cut(
    working: dict,
    integer_indices: Sequence[int],
    assignment: Sequence[float],
) -> Optional[str]:
    c = [float(value) for value in working["c"]]
    rows = [[float(value) for value in row] for row in working.get("a", [])]
    rhs = [float(value) for value in working.get("b", [])]
    lbs = [0.0 if value is None else float(value) for value in working.get("lb", [])]
    ubs = [None if value is None else float(value) for value in working.get("ub", [])]
    integer_vars = [bool(value) for value in working.get("integer_vars", [])]
    n = len(c)
    if len(lbs) != n or len(ubs) != n or len(integer_vars) != n:
        return "solution pool no-good cut saw inconsistent working model dimensions"

    deviation_vars: list[int] = []
    for idx in integer_indices:
        if idx >= len(assignment):
            return f"solution pool assignment is missing integer variable x{idx}"
        value = round(float(assignment[idx]))
        lb = lbs[idx]
        ub = ubs[idx]
        if ub is None:
            return f"solution pool requires finite upper bound for integer variable x{idx}"
        if value < lb - 1e-9 or value > ub + 1e-9:
            return f"solution pool assignment for x{idx} is outside its bounds"

        if value > lb + 1e-9:
            deviation = len(c)
            c.append(0.0)
            lbs.append(0.0)
            ubs.append(1.0)
            integer_vars.append(True)
            for row in rows:
                row.append(0.0)
            row = [0.0] * len(c)
            row[idx] = 1.0
            row[deviation] = float(ub - value + 1.0)
            rows.append(row)
            rhs.append(float(ub))
            deviation_vars.append(deviation)

        if value < ub - 1e-9:
            deviation = len(c)
            c.append(0.0)
            lbs.append(0.0)
            ubs.append(1.0)
            integer_vars.append(True)
            for row in rows:
                row.append(0.0)
            row = [0.0] * len(c)
            row[idx] = -1.0
            row[deviation] = float(value + 1.0 - lb)
            rows.append(row)
            rhs.append(float(-lb))
            deviation_vars.append(deviation)

    if not deviation_vars:
        return "solution pool could not create a no-good cut for a singleton integer domain"

    row = [0.0] * len(c)
    for deviation in deviation_vars:
        row[deviation] = -1.0
    rows.append(row)
    rhs.append(-1.0)

    working["c"] = c
    working["a"] = rows
    working["b"] = rhs
    working["lb"] = lbs
    working["ub"] = ubs
    working["integer_vars"] = integer_vars
    return None


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


def write_lpsolve_lp(
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
    with open(path, "w", encoding="utf-8") as f:
        f.write(("max: " if sense == "max" else "min: ") + term_expr(c, names) + ";\n")
        for i, (row, rhs) in enumerate(zip(le_rows, le_rhs)):
            f.write(f"c{i}: {term_expr(row, names)} <= {float(rhs):.12g};\n")
        for i, (row, rhs) in enumerate(zip(eq_rows, eq_rhs)):
            f.write(f"e{i}: {term_expr(row, names)} = {float(rhs):.12g};\n")
        if not le_rows and not eq_rows:
            f.write(f"c0: 0 {names[0] if names else 'x0'} <= 0;\n")
        for i, name in enumerate(names):
            lb = lbs[i]
            ub = ubs[i]
            if lb is not None and math.isfinite(float(lb)):
                f.write(f"{name} >= {float(lb):.12g};\n")
            if ub is not None and math.isfinite(float(ub)):
                f.write(f"{name} <= {float(ub):.12g};\n")
        integer_names = [name for name, is_int in zip(names, integer_vars) if is_int]
        if integer_names:
            f.write("int " + ", ".join(integer_names) + ";\n")
    return names


def write_free_mps(
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
    include_objsense: bool = True,
) -> list[str]:
    n = len(c)
    names = [var_name(i) for i in range(n)]
    rows = [("L", f"c{i}", row, rhs) for i, (row, rhs) in enumerate(zip(le_rows, le_rhs))]
    rows.extend(("E", f"e{i}", row, rhs) for i, (row, rhs) in enumerate(zip(eq_rows, eq_rhs)))
    if not rows:
        rows = [("L", "c0", [0.0] * n, 0.0)]

    with open(path, "w", encoding="utf-8") as f:
        f.write("NAME          ORESCLI\n")
        if include_objsense:
            f.write("OBJSENSE\n")
            f.write(" MAX\n" if sense == "max" else " MIN\n")
        f.write("ROWS\n")
        f.write(" N  OBJ\n")
        for row_sense, row_name, _, _ in rows:
            f.write(f" {row_sense}  {row_name}\n")

        f.write("COLUMNS\n")
        in_integer = False
        marker = 0
        for idx, name in enumerate(names):
            if integer_vars[idx] and not in_integer:
                f.write(f"    MARK{marker:04d}  'MARKER'                 'INTORG'\n")
                marker += 1
                in_integer = True
            elif not integer_vars[idx] and in_integer:
                f.write(f"    MARK{marker:04d}  'MARKER'                 'INTEND'\n")
                marker += 1
                in_integer = False

            entries: list[tuple[str, float]] = []
            if abs(float(c[idx])) > 1e-12:
                entries.append(("OBJ", float(c[idx])))
            for _, row_name, row, _ in rows:
                value = float(row[idx])
                if abs(value) > 1e-12:
                    entries.append((row_name, value))
            if not entries:
                entries.append(("OBJ", 0.0))
            for pos in range(0, len(entries), 2):
                first_name, first_value = entries[pos]
                line = f"    {name}  {first_name}  {first_value:.17g}"
                if pos + 1 < len(entries):
                    second_name, second_value = entries[pos + 1]
                    line += f"  {second_name}  {second_value:.17g}"
                f.write(line + "\n")
        if in_integer:
            f.write(f"    MARK{marker:04d}  'MARKER'                 'INTEND'\n")

        f.write("RHS\n")
        for _, row_name, _, rhs in rows:
            f.write(f"    RHS1  {row_name}  {float(rhs):.17g}\n")

        f.write("BOUNDS\n")
        for idx, name in enumerate(names):
            lb = lbs[idx]
            ub = ubs[idx]
            is_binary = (
                integer_vars[idx]
                and lb is not None
                and abs(float(lb)) <= 1e-12
                and ub is not None
                and abs(float(ub) - 1.0) <= 1e-12
            )
            if is_binary:
                f.write(f" BV BND1  {name}\n")
                continue
            if lb is None and ub is None:
                f.write(f" FR BND1  {name}\n")
            elif lb is None:
                f.write(f" MI BND1  {name}\n")
            elif abs(float(lb)) > 1e-12:
                f.write(f" LO BND1  {name}  {float(lb):.17g}\n")
            if ub is not None:
                f.write(f" UP BND1  {name}  {float(ub):.17g}\n")
        f.write("ENDATA\n")
    return names


def parse_highs_solution(
    path: str,
    n: int,
    le_count: int = 0,
    eq_count: int = 0,
) -> tuple[str, list[float], dict[str, object]]:
    x = [0.0] * n
    status = "unknown"
    dual_columns: list[Optional[float]] = [None] * n
    dual_rows: dict[str, float] = {}
    var_basis: list[Optional[str]] = [None] * n
    row_basis: dict[str, str] = {}
    section: Optional[str] = None
    block: Optional[str] = None
    remaining = 0
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f]
    for i, line in enumerate(lines):
        if line == "Model status" and i + 1 < len(lines):
            status = lines[i + 1].lower()
        if line == "# Primal solution values":
            section = "primal"
            block = None
            continue
        if line == "# Dual solution values":
            section = "dual"
            block = None
            continue
        if line.startswith("# Basis"):
            section = "basis"
            block = None
            continue
        if section is None:
            continue
        if line.startswith("# Columns"):
            block = "columns"
            remaining = int(line.split()[2])
            continue
        if line.startswith("# Rows"):
            block = "rows"
            remaining = int(line.split()[2])
            continue
        if not line or line.startswith("#") or block is None:
            continue
        parts = line.split()
        if len(parts) >= 2:
            value = float(parts[1]) if _is_number(parts[1]) else None
            if block == "columns" and parts[0].startswith("x") and parts[0][1:].isdigit():
                idx = int(parts[0][1:])
                if 0 <= idx < n:
                    if value is not None and section == "primal":
                        x[idx] = value
                    elif value is not None and section == "dual":
                        dual_columns[idx] = value
                    elif section == "basis":
                        basis_status = basis_status_from_token(parts[1])
                        if basis_status is not None:
                            var_basis[idx] = basis_status
            elif section == "dual" and block == "rows":
                if value is not None:
                    dual_rows[parts[0]] = value
            elif section == "basis" and block == "rows":
                basis_status = basis_status_from_token(parts[1])
                if basis_status is not None:
                    row_basis[parts[0]] = basis_status
        remaining -= 1
        if remaining <= 0:
            block = None

    fields: dict[str, object] = {}
    if all(value is not None for value in dual_columns):
        fields["reducedCosts"] = [float(value) for value in dual_columns if value is not None]
    dual_ub = [dual_rows.get(f"c{i}") for i in range(le_count)]
    dual_eq = [dual_rows.get(f"e{i}") for i in range(eq_count)]
    if all(value is not None for value in dual_ub):
        fields["dualUB"] = [float(value) for value in dual_ub if value is not None]
    if all(value is not None for value in dual_eq):
        fields["dualEQ"] = [float(value) for value in dual_eq if value is not None]
    if all(value is not None for value in var_basis):
        fields["varBasis"] = [str(value) for value in var_basis if value is not None]
    row_statuses = [row_basis.get(f"c{i}") for i in range(le_count)]
    row_statuses.extend(row_basis.get(f"e{i}") for i in range(eq_count))
    if all(value is not None for value in row_statuses):
        fields["rowBasis"] = [str(value) for value in row_statuses if value is not None]
    return status, x, fields


def parse_glpk_solution(
    path: str,
    n: int,
    le_count: int = 0,
    eq_count: int = 0,
) -> tuple[str, list[float], dict[str, object]]:
    x = [0.0] * n
    status = "unknown"
    row_duals: list[Optional[float]] = [None] * (le_count + eq_count)
    reduced_costs: list[Optional[float]] = [None] * n
    var_basis: list[Optional[str]] = [None] * n
    row_basis: list[Optional[str]] = [None] * (le_count + eq_count)
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
                        status_token = basis_status_from_token(parts[2])
                        if status_token is not None:
                            var_basis[idx] = status_token
                        if len(parts) >= 5 and _is_number(parts[4]):
                            reduced_costs[idx] = float(parts[4])
                    else:
                        x[idx] = float(parts[2])
            elif len(parts) >= 5 and parts[0] == "i":
                idx = int(parts[1]) - 1
                if 0 <= idx < len(row_duals):
                    status_token = basis_status_from_token(parts[2])
                    if status_token is not None:
                        row_basis[idx] = status_token
                    if _is_number(parts[4]):
                        row_duals[idx] = float(parts[4])

    fields: dict[str, object] = {}
    if all(value is not None for value in reduced_costs):
        fields["reducedCosts"] = [
            float(value) for value in reduced_costs if value is not None
        ]
    dual_ub = row_duals[:le_count]
    dual_eq = row_duals[le_count:]
    if all(value is not None for value in dual_ub):
        fields["dualUB"] = [float(value) for value in dual_ub if value is not None]
    if all(value is not None for value in dual_eq):
        fields["dualEQ"] = [float(value) for value in dual_eq if value is not None]
    if all(value is not None for value in var_basis):
        fields["varBasis"] = [str(value) for value in var_basis if value is not None]
    if all(value is not None for value in row_basis):
        fields["rowBasis"] = [str(value) for value in row_basis if value is not None]
    return status, x, fields


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


def parse_cbc_basis(
    path: str,
    n: int,
    le_count: int,
    eq_count: int,
) -> dict[str, object]:
    var_basis: list[Optional[str]] = [None] * n
    row_basis: list[Optional[str]] = ["basic"] * le_count + ["fixed"] * eq_count
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if not parts or parts[0] in {"NAME", "ENDATA"}:
                continue
            code = parts[0].upper()
            if len(parts) >= 2 and parts[1].startswith("x") and parts[1][1:].isdigit():
                idx = int(parts[1][1:])
                if 0 <= idx < n:
                    if code in {"BS", "XL", "XU"}:
                        var_basis[idx] = "basic"
                    elif code == "LL":
                        var_basis[idx] = "at_lower"
                    elif code == "UL":
                        var_basis[idx] = "at_upper"
                    elif code == "FX":
                        var_basis[idx] = "fixed"
                    elif code == "FR":
                        var_basis[idx] = "free"
            if code in {"XL", "XU"} and len(parts) >= 3 and parts[2].startswith("c"):
                suffix = parts[2][1:]
                if suffix.isdigit():
                    row_idx = int(suffix)
                    if 0 <= row_idx < le_count:
                        row_basis[row_idx] = "at_lower" if code == "XL" else "at_upper"

    fields: dict[str, object] = {}
    if all(value is not None for value in var_basis):
        fields["varBasis"] = [str(value) for value in var_basis if value is not None]
    if all(value is not None for value in row_basis):
        fields["rowBasis"] = [str(value) for value in row_basis if value is not None]
    return fields


def parse_cbc_solution(
    path: str,
    n: int,
    le_count: int = 0,
    eq_count: int = 0,
    basis_path: Optional[str] = None,
) -> tuple[str, list[float], dict[str, object]]:
    x = [0.0] * n
    status = "unknown"
    row_duals: list[Optional[float]] = [None] * (le_count + eq_count)
    reduced_costs: list[Optional[float]] = [None] * n
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
                        if len(parts) >= 4 and _is_number(parts[3]):
                            reduced_costs[idx] = float(parts[3])
            elif len(parts) >= 4 and parts[0].lstrip("-").isdigit():
                row_name = parts[1]
                if row_name.startswith("c") and row_name[1:].isdigit():
                    idx = int(row_name[1:])
                elif row_name.startswith("e") and row_name[1:].isdigit():
                    idx = le_count + int(row_name[1:])
                else:
                    idx = -1
                if 0 <= idx < len(row_duals) and _is_number(parts[3]):
                    row_duals[idx] = float(parts[3])

    fields: dict[str, object] = {}
    if all(value is not None for value in reduced_costs):
        fields["reducedCosts"] = [
            float(value) for value in reduced_costs if value is not None
        ]
    dual_ub = row_duals[:le_count]
    dual_eq = row_duals[le_count:]
    if all(value is not None for value in dual_ub):
        fields["dualUB"] = [float(value) for value in dual_ub if value is not None]
    if all(value is not None for value in dual_eq):
        fields["dualEQ"] = [float(value) for value in dual_eq if value is not None]
    if basis_path is not None and os.path.exists(basis_path):
        fields.update(parse_cbc_basis(basis_path, n, le_count, eq_count))
    return status, x, fields


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


def parse_xpress_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    status = "optimal"
    header_path = os.path.splitext(path)[0] + ".hdr"
    if os.path.exists(header_path):
        with open(header_path, "r", encoding="utf-8", errors="replace") as f:
            header = f.read().lower()
        if "infeas" in header:
            status = "infeasible"
        elif "unbounded" in header:
            status = "unbounded"
        elif "optimal" in header:
            status = "optimal"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    for line in text.splitlines():
        fields = _split_xpress_solution_line(line)
        if len(fields) < 2:
            continue
        for name_pos, name in enumerate(fields):
            name = name.strip().strip('"')
            if not (name.startswith("x") and name[1:].isdigit()):
                continue
            idx = int(name[1:])
            if 0 <= idx < n:
                for value in fields[name_pos + 1 :]:
                    value = value.strip().strip('"')
                    if _is_number(value):
                        x[idx] = float(value)
                        break
                break
    return status, x


def parse_lindo_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lower = text.lower()
    if "infeasible" in lower or "no feasible" in lower:
        status = "infeasible"
    elif "unbounded" in lower:
        status = "unbounded"
    elif "optimal" in lower or "objective" in lower:
        status = "optimal"
    else:
        status = "unknown"

    for line in text.splitlines():
        parsed = _parse_named_value_line(line, n)
        if parsed is not None:
            idx, value = parsed
            x[idx] = value
    return status, x


def parse_lp_solve_solution(path: str, n: int) -> tuple[str, list[float]]:
    x = [0.0] * n
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lower = text.lower()
    if "infeasible" in lower or "no feasible" in lower:
        status = "infeasible"
    elif "unbounded" in lower:
        status = "unbounded"
    elif "value of objective function" in lower or "actual values of the variables" in lower:
        status = "optimal"
    else:
        status = "unknown"

    for line in text.splitlines():
        parsed = _parse_named_value_line(line, n)
        if parsed is not None:
            idx, value = parsed
            x[idx] = value
    return status, x


def parse_soplex_solution(path: str, n: int, stdout: str, stderr: str) -> tuple[str, list[float]]:
    x = [0.0] * n
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lower = f"{stdout}\n{stderr}\n{text}".lower()
    if "infeasible" in lower:
        status = "infeasible"
    elif "unbounded" in lower:
        status = "unbounded"
    elif "problem is solved [optimal]" in lower or "primal solution" in lower:
        status = "optimal"
    else:
        status = "unknown"

    for line in text.splitlines():
        parsed = _parse_named_value_line(line, n)
        if parsed is not None:
            idx, value = parsed
            x[idx] = value
    return status, x


def parse_qsopt_ex_solution(path: str, n: int, stdout: str, stderr: str) -> tuple[str, list[float]]:
    x = [0.0] * n
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    lower = f"{stdout}\n{stderr}\n{text}".lower()
    if "infeasible" in lower:
        status = "infeasible"
    elif "unbounded" in lower:
        status = "unbounded"
    elif "optimal" in lower or "objective" in lower or "primal solution" in lower:
        status = "optimal"
    else:
        status = "unknown"

    for line in text.splitlines():
        parsed = _parse_named_value_line(line, n)
        if parsed is not None:
            idx, value = parsed
            x[idx] = value
    return status, x


def _parse_named_value_line(line: str, n: int) -> Optional[tuple[int, float]]:
    match = re.search(r"\bx(\d+)\b", line, flags=re.IGNORECASE)
    if match is None:
        return None
    idx = int(match.group(1))
    if not 0 <= idx < n:
        return None
    after = line[match.end() :]
    for token in re.split(r"[\s,;:=]+", after):
        token = token.strip()
        if _is_number(token):
            return idx, float(token)
    before = line[: match.start()]
    numeric_before = [
        float(token)
        for token in re.split(r"[\s,;:=]+", before)
        if token.strip() and _is_number(token.strip())
    ]
    if numeric_before:
        return idx, numeric_before[-1]
    return None


def _split_xpress_solution_line(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return []
    for delimiter in (";", ","):
        if delimiter in stripped:
            return [
                field.strip()
                for field in next(csv.reader([stripped], delimiter=delimiter))
                if field.strip()
            ]
    return stripped.split()


def _is_number(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def stripped_starts(text: str, prefixes: Sequence[str]) -> bool:
    stripped = text.strip()
    return any(stripped.startswith(prefix) for prefix in prefixes)


def parse_solver_version(solver: str, stdout: str, stderr: str) -> Optional[str]:
    text = f"{stdout}\n{stderr}"
    patterns = {
        "highs": [
            (r"\bRunning HiGHS\s+([0-9][^\s,]*)", "HiGHS"),
            (r"\bHiGHS version\s+([0-9][^\s,]*)", "HiGHS"),
        ],
        "glpk": [(r"GLPSOL--GLPK LP/MIP Solver\s+([0-9][^\s,]*)", "GLPK")],
        "scip": [(r"\bSCIP version\s+([0-9][^\s\[]*)", "SCIP")],
        "cbc": [(r"\bVersion:\s+([0-9][^\s,]*)", "CBC")],
        "clp": [(r"\bCoin LP version\s+([0-9][^\s,]*)", "CLP")],
        "soplex": [(r"\bSoPlex version\s+([0-9][^\s,]*)", "SoPlex")],
        "gurobi": [(r"\bGurobi Optimizer version\s+([0-9][^\s,]*)", "Gurobi")],
        "cplex": [(r"\b(?:IBM ILOG )?CPLEX(?: Optimizer)?(?: Interactive Optimizer)?\s+([0-9][^\s,]*)", "CPLEX")],
        "xpress": [(r"\bXpress(?: Optimizer)?\s+([0-9][^\s,]*)", "Xpress")],
        "lindo": [(r"\bLINDO(?: API| Optimizer)?\s+([0-9][^\s,]*)", "LINDO")],
    }
    for pattern, label in patterns.get(solver, []):
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match is not None:
            return f"{label} {match.group(1)}"
    return None


def probe_solver_version(solver: str) -> Optional[str]:
    command = solver_command(solver)
    if command is None:
        return None
    version_args = {
        "highs": ["--version"],
        "glpk": ["--version"],
        "scip": ["--version"],
        "cbc": ["-version"],
        "clp": ["-version"],
        "soplex": ["-v0"],
        "gurobi": ["--version"],
    }.get(solver)
    if version_args is None:
        return None
    try:
        run = subprocess.run(
            [command, *version_args],
            text=True,
            capture_output=True,
            check=False,
            timeout=3.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return parse_solver_version(solver, run.stdout, run.stderr)


def solver_version_from_output(solver: str, stdout: str, stderr: str) -> Optional[str]:
    return parse_solver_version(solver, stdout, stderr) or probe_solver_version(solver)


def classify_status(status: str, stdout: str, stderr: str) -> str:
    parsed = status.lower()
    if "primal infeasible" in parsed or ("infeasible" in parsed and "dual" not in parsed):
        return "infeasible"
    if "dual infeasible" in parsed or "unbounded" in parsed:
        return "unbounded"
    if "optimal" in parsed:
        return "optimal"
    if "feasible" in parsed or "solution limit" in parsed:
        return "feasible"

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
    feasible_markers = (
        "stopped on solution limit",
        "solution limit reached",
        "exiting on maximum solutions",
        "partial search - best objective",
        "integer solution of",
        "feasibility pump exiting with objective",
    )
    if any(marker in text for marker in feasible_markers):
        return "feasible"
    return "unknown"


def _first_float_after_colon(line: str) -> Optional[float]:
    text = line.split(":", 1)[1] if ":" in line else line
    return _first_float(text)


def _first_float(text: str) -> Optional[float]:
    for token in re.split(r"[\s,()]+", text):
        cleaned = token.strip().rstrip("%")
        if not cleaned:
            continue
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


def parse_lp_iterations(solver: str, kind: str, stdout: str, stderr: str) -> dict:
    if kind != "lp":
        return {}
    text = f"{stdout}\n{stderr}"
    iterations: Optional[int] = None

    if solver == "highs":
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.lower().startswith("simplex") and "iterations" in stripped.lower():
                value = _first_float_after_colon(stripped)
                if value is not None:
                    iterations = int(round(value))
    elif solver == "glpk":
        for line in text.splitlines():
            match = re.match(r"^\*?\s*(\d+):\s+obj\b", line.strip())
            if match is not None:
                iterations = int(match.group(1))
    elif solver in {"cbc", "clp"}:
        for line in text.splitlines():
            match = re.search(r"-\s+(\d+)\s+iterations\b", line.strip(), flags=re.IGNORECASE)
            if match is not None:
                iterations = int(match.group(1))
    elif solver == "soplex":
        for line in text.splitlines():
            match = re.search(r"\bIterations\s*:\s*(\d+)\b", line.strip(), flags=re.IGNORECASE)
            if match is not None:
                iterations = int(match.group(1))

    if iterations is not None and iterations >= 0:
        return {"iterations": iterations}
    return {}


def parse_mip_quality(
    solver: str,
    kind: str,
    objective: Optional[float],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "mip":
        return {}
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    fields = {}

    best_bound = None
    mip_gap = None
    absolute_gap = None
    nodes_explored = None

    if solver == "highs":
        for line in text.splitlines():
            stripped = line.strip()
            lowered_line = stripped.lower()
            if lowered_line.startswith("dual bound"):
                best_bound = _first_float_after_colon(stripped)
            elif lowered_line.startswith("gap"):
                value = _first_float(stripped)
                if value is not None:
                    mip_gap = value / 100.0 if "%" in stripped else value
            elif lowered_line.startswith("nodes"):
                value = _first_float_after_colon(stripped)
                if value is not None:
                    nodes_explored = int(round(value))
    elif solver == "cbc":
        for line in text.splitlines():
            stripped = line.strip()
            lowered_line = stripped.lower()
            if lowered_line.startswith("enumerated nodes:"):
                value = _first_float_after_colon(stripped)
                if value is not None:
                    nodes_explored = int(round(value))
            elif lowered_line.startswith("lower bound:"):
                best_bound = _first_float_after_colon(stripped)
            elif "gap:" in lowered_line:
                value = _first_float_after_colon(stripped)
                if value is not None:
                    mip_gap = value / 100.0 if "%" in stripped else value
    elif solver == "scip":
        for line in text.splitlines():
            stripped = line.strip()
            lowered_line = stripped.lower()
            if lowered_line.startswith("solving nodes"):
                value = _first_float_after_colon(stripped)
                if value is not None:
                    nodes_explored = int(round(value))
            elif lowered_line.startswith("dual bound"):
                best_bound = _first_float_after_colon(stripped)
            elif lowered_line.startswith("gap"):
                value = _first_float_after_colon(stripped)
                if value is not None:
                    mip_gap = value / 100.0 if "%" in stripped else value

    if best_bound is not None and math.isfinite(best_bound):
        fields["bestBound"] = best_bound
    if best_bound is not None and objective is not None:
        absolute_gap = abs(best_bound - objective)
    if absolute_gap is not None and math.isfinite(absolute_gap):
        fields["absoluteGap"] = max(0.0, absolute_gap)
    if mip_gap is None and best_bound is not None and objective is not None:
        mip_gap = abs(best_bound - objective) / max(1.0, abs(objective))
    if mip_gap is not None and math.isfinite(mip_gap):
        fields["mipGap"] = max(0.0, mip_gap)
    if nodes_explored is not None and nodes_explored >= 0:
        fields["nodesExplored"] = nodes_explored
    return fields


def write_highs_mip_start(path: str, start: Sequence[float], objective: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("Model status\n")
        f.write("Unknown\n\n")
        f.write("# Primal solution values\n")
        f.write("Feasible\n")
        f.write(f"Objective {objective:.17g}\n")
        f.write(f"# Columns {len(start)}\n")
        for idx, value in enumerate(start):
            f.write(f"{var_name(idx)} {float(value):.17g}\n")
        f.write("# Rows 0\n\n")
        f.write("# Dual solution values\n")
        f.write("None\n\n")
        f.write("# Basis\n")
        f.write("HiGHS_basis_file v2\n")
        f.write("None\n")


def write_scip_mip_start(path: str, start: Sequence[float], objective: float) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("solution status: feasible\n")
        f.write(f"objective value: {objective:.17g}\n")
        for idx, value in enumerate(start):
            f.write(f"{var_name(idx)} {float(value):.17g}\n")


def write_cbc_mip_start(path: str, start: Sequence[float]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx, value in enumerate(start):
            f.write(f"{idx} {var_name(idx)} {float(value):.17g}\n")


def active_branch_priorities(
    branch_priorities: Optional[Sequence[int]],
    integer_vars: Optional[Sequence[bool]],
) -> list[tuple[int, int]]:
    if branch_priorities is None:
        return []
    active: list[tuple[int, int]] = []
    for idx, priority in enumerate(branch_priorities):
        if priority == 0:
            continue
        if integer_vars is not None and idx < len(integer_vars) and not integer_vars[idx]:
            continue
        active.append((idx, int(priority)))
    return active


def write_cbc_branch_priorities(
    path: str,
    branch_priorities: Sequence[int],
    integer_vars: Optional[Sequence[bool]],
) -> int:
    active = active_branch_priorities(branch_priorities, integer_vars)
    if not active:
        return 0
    highest = max(priority for _, priority in active)
    with open(path, "w", encoding="utf-8") as f:
        f.write("name,priority\n")
        for idx, priority in active:
            # CBC treats lower numeric priority as more important. The bridge
            # API follows the native DES convention: larger is more important.
            cbc_priority = highest - priority + 1
            f.write(f"{var_name(idx)},{int(cbc_priority)}\n")
    return len(active)


def mip_start_infeasibility_values(text: str) -> list[float]:
    values: list[float] = []
    for line in text.splitlines():
        if "infeasibilities" not in line.lower():
            continue
        numbers = re.findall(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", line)
        if numbers:
            values.append(float(numbers[0]))
    return values


def parse_mip_start_feedback(
    solver: str,
    kind: str,
    mip_start: Optional[Sequence[float]],
    start_objective: Optional[float],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "mip" or mip_start is None:
        return {}
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    accepted = False

    if solver == "highs" and "assessing feasibility of mip" in lowered:
        infeasibilities = mip_start_infeasibility_values(text)
        accepted = len(infeasibilities) >= 3 and all(
            abs(value) <= 1e-9 for value in infeasibilities[:3]
        )
    elif solver == "scip":
        accepted = "accepted as candidate" in lowered or "solution candidate storage" in lowered
    elif solver == "cbc":
        accepted = (
            "mipstart values read" in lowered
            and ("mipstart provided solution" in lowered or "integer solution" in lowered)
        )

    fields = {"mipStartAccepted": accepted}
    if start_objective is not None and math.isfinite(start_objective):
        fields["mipStartObjective"] = start_objective
    return fields


def parse_search_control_feedback(
    solver: str,
    kind: str,
    branch_rule: Optional[str],
    node_selection: Optional[str],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "mip":
        return {}
    text = f"{stdout}\n{stderr}"
    fields = {}

    if branch_rule is not None and solver == "glpk":
        flag = "--first" if branch_rule == "first-fractional" else "--mostf"
        if flag in text:
            fields["branchRule"] = branch_rule

    if node_selection is not None:
        if solver == "glpk":
            flag = "--dfs" if node_selection == "dfs" else "--bestb"
            if flag in text:
                fields["nodeSelection"] = node_selection
        elif solver == "cbc":
            cbc_name = "depth" if node_selection == "dfs" else "fewest"
            if f"-nodeStrategy {cbc_name}" in text or f"to {cbc_name}" in text:
                fields["nodeSelection"] = node_selection

    return fields


def parse_branch_priority_feedback(
    solver: str,
    kind: str,
    branch_priorities: Optional[Sequence[int]],
    integer_vars: Optional[Sequence[bool]],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "mip":
        return {}
    active_count = len(active_branch_priorities(branch_priorities, integer_vars))
    if active_count == 0:
        return {}
    text = f"{stdout}\n{stderr}".lower()
    if solver == "cbc" and "priorityin" in text:
        return {
            "branchPrioritiesAccepted": True,
            "branchPriorityCount": active_count,
        }
    if solver == "scip" and "branching priority of variable" in text:
        return {
            "branchPrioritiesAccepted": True,
            "branchPriorityCount": active_count,
        }
    if solver in {"gurobi", "cplex", "xpress"}:
        return {
            "branchPrioritiesAccepted": True,
            "branchPriorityCount": active_count,
        }
    return {}


def parse_operational_control_feedback(
    solver: str,
    kind: str,
    threads: Optional[int],
    random_seed: Optional[int],
    presolve: Optional[str],
    stdout: str,
    stderr: str,
) -> dict:
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    fields: dict[str, object] = {}

    if threads is not None:
        if solver == "highs" and f"set option threads to {threads}" in lowered:
            fields["threads"] = threads
        elif solver == "scip" and f"parallel/maxnthreads = {threads}" in text:
            fields["threads"] = threads
        elif solver == "cbc" and (
            f"threads was changed" in lowered or f"-threads {threads}" in text
        ):
            fields["threads"] = threads
        elif solver in {"gurobi", "cplex", "xpress"}:
            fields["threads"] = threads

    if random_seed is not None:
        if solver == "highs" and f"set option random_seed to {random_seed}" in lowered:
            fields["randomSeed"] = random_seed
        elif solver == "glpk" and f"--seed {random_seed}" in text:
            fields["randomSeed"] = random_seed
        elif solver == "scip" and f"randomization/randomseedshift = {random_seed}" in text:
            fields["randomSeed"] = random_seed
        elif solver == "cbc" and (
            f"randomseed was changed" in lowered
            or f"randomcbcseed was changed" in lowered
            or f"-randomS {random_seed}" in text
        ):
            fields["randomSeed"] = random_seed
        elif solver in {"gurobi", "cplex", "xpress"}:
            fields["randomSeed"] = random_seed

    if presolve is not None:
        highs_presolve = "choose" if presolve == "auto" else presolve
        if solver == "highs" and f"set option presolve to \"{highs_presolve}\"" in lowered:
            fields["presolve"] = presolve
        elif solver == "glpk":
            if kind == "mip" and (
                (presolve == "off" and "--nointopt" in text)
                or (presolve == "on" and "--intopt" in text)
            ):
                fields["presolve"] = presolve
            elif kind == "lp" and (
                (presolve == "off" and "--nopresol" in text)
                or (presolve == "on" and "--presol" in text)
            ):
                fields["presolve"] = presolve
        elif solver == "scip" and (
            (presolve == "off" and "presolving/maxrounds = 0" in text)
            or (presolve == "on" and "presolving/maxrounds = -1" in text)
        ):
            fields["presolve"] = presolve
        elif solver == "cbc" and (
            (presolve == "off" and "changed from" in lowered and "to off" in lowered)
            or (presolve == "on" and ("to on" in lowered or "-presolve on" in text))
        ):
            fields["presolve"] = presolve
        elif solver == "clp" and (
            (presolve == "off" and "-presolve off" in text)
            or (presolve == "on" and "-presolve on" in text)
        ):
            fields["presolve"] = presolve
        elif solver in {"gurobi", "cplex", "xpress"}:
            fields["presolve"] = presolve

    return fields


def parse_lp_algorithm_feedback(
    solver: str,
    kind: str,
    lp_algorithm: Optional[str],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "lp" or lp_algorithm is None:
        return {}
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()

    if solver == "highs":
        if f"set option solver to \"{lp_algorithm}\"" in lowered:
            return {"lpAlgorithm": lp_algorithm}
    elif solver == "glpk":
        flag = "--simplex" if lp_algorithm == "simplex" else "--interior"
        if flag in text:
            return {"lpAlgorithm": lp_algorithm}

    return {}


def parse_mip_strategy_feedback(
    solver: str,
    kind: str,
    cuts: Optional[str],
    heuristics: Optional[str],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "mip":
        return {}
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    fields: dict[str, object] = {}

    if cuts is not None:
        if solver == "cbc" and (
            f"-cuts {cuts}" in text
            or (
                "option for cutsonoff changed" in lowered
                and f"to {cuts}" in lowered
            )
        ):
            fields["cuts"] = cuts
        elif solver == "scip" and (
            (cuts == "off" and "separating/maxrounds = 0" in text and "separating/maxroundsroot = 0" in text)
            or (cuts == "on" and "separating/maxrounds = -1" in text and "separating/maxroundsroot = -1" in text)
        ):
            fields["cuts"] = cuts
        elif solver == "glpk" and cuts == "on" and "--cuts" in text:
            fields["cuts"] = cuts

    if heuristics is not None:
        if solver == "cbc" and (
            f"-heuristicsOnOff {heuristics}" in text
            or (
                "option for heuristicsonoff changed" in lowered
                and f"to {heuristics}" in lowered
            )
        ):
            fields["heuristics"] = heuristics
        elif solver == "scip" and heuristics == "off" and "heuristics/feaspump/freq = -1" in text:
            fields["heuristics"] = heuristics

    return fields


def parse_solution_limit_feedback(
    solver: str,
    kind: str,
    solution_limit: Optional[int],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "mip" or solution_limit is None:
        return {}
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    if solver == "cbc" and (
        f"-maxSolutions {solution_limit}" in text
        or f"maxsolutions was changed" in lowered
    ):
        return {"solutionLimit": solution_limit}
    if solver == "scip" and f"limits/solutions = {solution_limit}" in text:
        return {"solutionLimit": solution_limit}
    if solver in {"gurobi", "cplex", "xpress"}:
        return {"solutionLimit": solution_limit}
    return {}


def parse_objective_limit_feedback(
    solver: str,
    kind: str,
    objective_limit: Optional[float],
    stdout: str,
    stderr: str,
) -> dict:
    if kind != "mip" or objective_limit is None:
        return {}
    text = f"{stdout}\n{stderr}"
    lowered = text.lower()
    if solver == "highs" and "set option objective_target to" in lowered:
        return {"objectiveLimit": objective_limit}
    if solver == "scip" and f"limits/primal = {objective_limit:.12g}" in text:
        return {"objectiveLimit": objective_limit}
    if solver == "gurobi":
        return {"objectiveLimit": objective_limit}
    return {}


def parse_tolerance_feedback(
    solver: str,
    kind: str,
    primal_feasibility_tolerance: Optional[float],
    dual_feasibility_tolerance: Optional[float],
    integer_feasibility_tolerance: Optional[float],
    stdout: str,
    stderr: str,
) -> dict:
    _ = stdout, stderr
    fields: dict[str, object] = {}
    if primal_feasibility_tolerance is not None and solver in {
        "highs",
        "scip",
        "cbc",
        "clp",
        "gurobi",
        "cplex",
        "xpress",
    }:
        fields["primalFeasibilityTolerance"] = primal_feasibility_tolerance
    if dual_feasibility_tolerance is not None and solver in {
        "highs",
        "scip",
        "cbc",
        "clp",
        "gurobi",
        "cplex",
        "xpress",
    }:
        fields["dualFeasibilityTolerance"] = dual_feasibility_tolerance
    if (
        kind == "mip"
        and integer_feasibility_tolerance is not None
        and solver in {"highs", "cbc", "gurobi", "cplex", "xpress"}
    ):
        fields["integerFeasibilityTolerance"] = integer_feasibility_tolerance
    return fields


def solver_available(solver: str) -> bool:
    return solver_command(solver) is not None


def solver_command(solver: str) -> Optional[str]:
    configured_any = False
    for env_var in COMMAND_ENV_VARS.get(solver, []):
        configured = os.environ.get(env_var)
        if configured and configured.strip():
            configured_any = True
            expanded = os.path.expanduser(configured)
            if os.path.isfile(expanded) and os.access(expanded, os.X_OK):
                return expanded
            resolved = shutil.which(configured)
            if resolved is not None:
                return resolved
    for env_var in COMMAND_DIR_ENV_VARS.get(solver, []):
        configured = os.environ.get(env_var)
        if configured and configured.strip():
            configured_any = True
            resolved = command_in_install_dir(
                os.path.expanduser(configured),
                COMMAND_ALIASES.get(solver, [solver]),
            )
            if resolved is not None:
                return resolved
    if configured_any:
        return None
    for command in COMMAND_ALIASES.get(solver, [solver]):
        resolved = shutil.which(command)
        if resolved is not None:
            return resolved
    return None


def command_in_install_dir(root: str, aliases: Sequence[str]) -> Optional[str]:
    candidate_dirs = [root, os.path.join(root, "bin")]
    try:
        children = os.listdir(root)
    except OSError:
        children = []
    for child in children:
        child_path = os.path.join(root, child)
        if not os.path.isdir(child_path):
            continue
        child_bin = os.path.join(child_path, "bin")
        candidate_dirs.append(child_bin)
        try:
            platform_dirs = os.listdir(child_bin)
        except OSError:
            platform_dirs = []
        for platform_dir in platform_dirs:
            platform_path = os.path.join(child_bin, platform_dir)
            if os.path.isdir(platform_path):
                candidate_dirs.append(platform_path)
    for candidate_dir in candidate_dirs:
        for alias in aliases:
            candidate = os.path.join(candidate_dir, alias)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return candidate
    return None


def run_solver(
    solver: str,
    kind: str,
    sense: str,
    model_path: str,
    solution_path: str,
    time_limit: float,
    model_format: str,
    node_limit: Optional[int] = None,
    solution_limit: Optional[int] = None,
    relative_gap: Optional[float] = None,
    absolute_gap: Optional[float] = None,
    objective_limit: Optional[float] = None,
    primal_feasibility_tolerance: Optional[float] = None,
    dual_feasibility_tolerance: Optional[float] = None,
    integer_feasibility_tolerance: Optional[float] = None,
    lp_algorithm: Optional[str] = None,
    threads: Optional[int] = None,
    random_seed: Optional[int] = None,
    presolve: Optional[str] = None,
    cuts: Optional[str] = None,
    heuristics: Optional[str] = None,
    branch_rule: Optional[str] = None,
    branch_priorities: Optional[Sequence[int]] = None,
    node_selection: Optional[str] = None,
    mip_start: Optional[Sequence[float]] = None,
    mip_start_objective: Optional[float] = None,
    integer_vars: Optional[Sequence[bool]] = None,
) -> tuple[str, str]:
    command = solver_command(solver)
    if command is None:
        raise ValueError(f"{solver} executable not found")
    input_text = None
    if solver == "highs":
        start_path = None
        if kind == "mip" and mip_start is not None and mip_start_objective is not None:
            start_path = solution_path + ".start.sol"
            write_highs_mip_start(start_path, mip_start, mip_start_objective)
        options_path = None
        if (
            threads is not None
            or primal_feasibility_tolerance is not None
            or dual_feasibility_tolerance is not None
            or (
                kind == "mip"
                and (
                    node_limit is not None
                    or relative_gap is not None
                    or absolute_gap is not None
                    or objective_limit is not None
                    or integer_feasibility_tolerance is not None
                )
            )
        ):
            options_path = solution_path + ".options"
            with open(options_path, "w", encoding="utf-8") as f:
                if threads is not None:
                    f.write(f"threads = {int(threads)}\n")
                if primal_feasibility_tolerance is not None:
                    f.write(
                        f"primal_feasibility_tolerance = {float(primal_feasibility_tolerance):.17g}\n"
                    )
                if dual_feasibility_tolerance is not None:
                    f.write(
                        f"dual_feasibility_tolerance = {float(dual_feasibility_tolerance):.17g}\n"
                    )
                if kind == "mip" and node_limit is not None:
                    f.write(f"mip_max_nodes = {int(node_limit)}\n")
                if kind == "mip" and relative_gap is not None:
                    f.write(f"mip_rel_gap = {float(relative_gap):.17g}\n")
                if kind == "mip" and absolute_gap is not None:
                    f.write(f"mip_abs_gap = {float(absolute_gap):.17g}\n")
                if kind == "mip" and objective_limit is not None:
                    f.write(f"objective_target = {float(objective_limit):.17g}\n")
                if kind == "mip" and integer_feasibility_tolerance is not None:
                    f.write(
                        f"mip_feasibility_tolerance = {float(integer_feasibility_tolerance):.17g}\n"
                    )
        cmd = [
            command,
            "--model_file",
            model_path,
            "--solution_file",
            solution_path,
            "--time_limit",
            str(time_limit),
        ]
        if start_path is not None:
            cmd.extend(["--read_solution_file", start_path])
        if options_path is not None:
            cmd.extend(["--options_file", options_path])
        if kind == "lp" and lp_algorithm is not None:
            cmd.extend(["--solver", lp_algorithm])
        if random_seed is not None:
            cmd.extend(["--random_seed", str(int(random_seed))])
        if presolve is not None:
            cmd.extend(["--presolve", "choose" if presolve == "auto" else presolve])
    elif solver == "glpk":
        format_arg = "--freemps" if model_format == "mps" else "--lp"
        sense_arg = "--max" if sense == "max" else "--min"
        if kind == "lp":
            cmd = [
                command,
                format_arg,
                model_path,
                sense_arg,
                "--output",
                solution_path + ".report",
                "--write",
                solution_path,
                "--tmlim",
                str(max(1, int(math.ceil(time_limit)))),
            ]
            if presolve == "off":
                cmd.append("--nopresol")
            elif presolve == "on":
                cmd.append("--presol")
            if lp_algorithm == "simplex":
                cmd.append("--simplex")
            elif lp_algorithm == "ipm":
                cmd.append("--interior")
        else:
            cmd = [
                command,
                format_arg,
                model_path,
                sense_arg,
                "-o",
                solution_path,
                "--tmlim",
                str(max(1, int(math.ceil(time_limit)))),
            ]
            if presolve == "off":
                cmd.append("--nointopt")
            elif presolve == "on":
                cmd.append("--intopt")
            if branch_rule == "first-fractional":
                cmd.append("--first")
            elif branch_rule == "most-fractional":
                cmd.append("--mostf")
            if node_selection == "dfs":
                cmd.append("--dfs")
            elif node_selection == "best-bound":
                cmd.append("--bestb")
            if relative_gap is not None:
                cmd.extend(["--mipgap", f"{float(relative_gap):.17g}"])
            if cuts == "on":
                cmd.append("--cuts")
        if random_seed is not None:
            cmd.extend(["--seed", str(int(random_seed))])
    elif solver == "scip":
        cmd = [command]
        if presolve == "off":
            cmd.extend(["-c", "set presolving maxrounds 0"])
        elif presolve == "on":
            cmd.extend(["-c", "set presolving maxrounds -1"])
        if random_seed is not None:
            cmd.extend(["-c", f"set randomization randomseedshift {int(random_seed)}"])
        if threads is not None:
            cmd.extend(["-c", f"set parallel maxnthreads {int(threads)}"])
        if primal_feasibility_tolerance is not None:
            cmd.extend([
                "-c",
                f"set numerics feastol {float(primal_feasibility_tolerance):.17g}",
            ])
        if dual_feasibility_tolerance is not None:
            cmd.extend([
                "-c",
                f"set numerics dualfeastol {float(dual_feasibility_tolerance):.17g}",
            ])
        if kind == "mip":
            if cuts == "off":
                cmd.extend([
                    "-c",
                    "set separating maxrounds 0",
                    "-c",
                    "set separating maxroundsroot 0",
                ])
            elif cuts == "on":
                cmd.extend([
                    "-c",
                    "set separating maxrounds -1",
                    "-c",
                    "set separating maxroundsroot -1",
                ])
            if heuristics == "off":
                cmd.extend(["-c", "set heuristics emphasis off"])
        if kind == "mip":
            if node_limit is not None:
                cmd.extend(["-c", f"set limits nodes {int(node_limit)}"])
            if solution_limit is not None:
                cmd.extend(["-c", f"set limits solutions {int(solution_limit)}"])
            if relative_gap is not None:
                cmd.extend(["-c", f"set limits gap {float(relative_gap):.17g}"])
            if absolute_gap is not None:
                cmd.extend(["-c", f"set limits absgap {float(absolute_gap):.17g}"])
            if objective_limit is not None:
                cmd.extend(["-c", f"set limits primal {float(objective_limit):.17g}"])
            if mip_start is not None and mip_start_objective is not None:
                start_path = solution_path + ".start.sol"
                write_scip_mip_start(start_path, mip_start, mip_start_objective)
            else:
                start_path = None
        if kind == "lp":
            cmd.append("-q")
        cmd.extend([
            "-c",
            f"read {model_path}",
        ])
        if kind == "mip" and branch_priorities is not None:
            for idx, priority in active_branch_priorities(
                branch_priorities,
                integer_vars,
            ):
                cmd.extend([
                    "-c",
                    "set branching priority",
                    "-c",
                    var_name(idx),
                    "-c",
                    str(int(priority)),
                ])
        if kind == "mip" and start_path is not None:
            cmd.extend([
                "-c",
                f"read {start_path}",
            ])
        cmd.extend([
            "-c",
            f"set limits time {time_limit}",
            "-c",
            "optimize",
            "-c",
            f"write solution {solution_path}",
        ])
        if kind == "mip":
            cmd.extend([
                "-c",
                "display statistics",
            ])
        cmd.extend([
            "-c",
            "quit",
        ])
    elif solver == "cbc":
        cmd = [
            command,
            model_path,
            "-seconds",
            str(time_limit),
        ]
        if model_format == "mps":
            cmd.append("-max" if sense == "max" else "-min")
        if random_seed is not None:
            cmd.extend(["-randomS", str(int(random_seed)), "-randomC", str(int(random_seed))])
        if threads is not None:
            cmd.extend(["-threads", str(int(threads))])
        if primal_feasibility_tolerance is not None:
            cmd.extend(["-primalT", f"{float(primal_feasibility_tolerance):.17g}"])
        if dual_feasibility_tolerance is not None:
            cmd.extend(["-dualT", f"{float(dual_feasibility_tolerance):.17g}"])
        if presolve == "off":
            cmd.extend(["-presolve", "off"])
            if kind == "mip":
                cmd.extend(["-preprocess", "off"])
        elif presolve == "on":
            cmd.extend(["-presolve", "on"])
        if kind == "lp":
            cmd.extend(["-printingOptions", "all"])
        else:
            if cuts in {"on", "off"}:
                cmd.extend(["-cuts", cuts])
            if heuristics in {"on", "off"}:
                cmd.extend(["-heuristicsOnOff", heuristics])
            if branch_priorities is not None:
                priority_path = solution_path + ".priority.csv"
                if write_cbc_branch_priorities(
                    priority_path,
                    branch_priorities,
                    integer_vars,
                ) > 0:
                    cmd.extend(["-priorityIn", priority_path])
            if mip_start is not None:
                start_path = solution_path + ".start.sol"
                write_cbc_mip_start(start_path, mip_start)
                cmd.extend(["-mipstart", start_path])
            if node_limit is not None:
                cmd.extend(["-maxNodes", str(int(node_limit))])
            if solution_limit is not None:
                cmd.extend(["-maxSolutions", str(int(solution_limit))])
            if node_selection == "dfs":
                cmd.extend(["-nodeStrategy", "depth"])
            elif node_selection == "best-bound":
                cmd.extend(["-nodeStrategy", "fewest"])
            if integer_feasibility_tolerance is not None:
                cmd.extend(["-integerT", f"{float(integer_feasibility_tolerance):.17g}"])
        if kind == "mip" and relative_gap is not None:
            cmd.extend(["-ratioGap", f"{float(relative_gap):.17g}"])
        if kind == "mip" and absolute_gap is not None:
            cmd.extend(["-allowableGap", f"{float(absolute_gap):.17g}"])
        cmd.extend([
            "-solve",
            "-solution",
            solution_path,
        ])
        if kind == "lp":
            cmd.extend(["-basisOut", solution_path + ".basis"])
    elif solver == "clp":
        cmd = [
            command,
            model_path,
            "-seconds",
            str(time_limit),
            *(["-max" if sense == "max" else "-min"] if model_format == "mps" else []),
            "-printingOptions",
            "all",
            *(["-presolve", presolve] if presolve in {"on", "off"} else []),
            *(
                ["-primalT", f"{float(primal_feasibility_tolerance):.17g}"]
                if primal_feasibility_tolerance is not None
                else []
            ),
            *(
                ["-dualT", f"{float(dual_feasibility_tolerance):.17g}"]
                if dual_feasibility_tolerance is not None
                else []
            ),
            "-solve",
            "-solution",
            solution_path,
            "-basisOut",
            solution_path + ".basis",
        ]
    elif solver == "soplex":
        cmd = [
            command,
            "-v3",
            f"-t{float(time_limit):.17g}",
            f"-x={solution_path}",
            *(
                ["-f" + f"{float(primal_feasibility_tolerance):.17g}"]
                if primal_feasibility_tolerance is not None
                else []
            ),
            *(
                ["-o" + f"{float(dual_feasibility_tolerance):.17g}"]
                if dual_feasibility_tolerance is not None
                else []
            ),
            *(["-s0"] if presolve == "off" else ["-s1"] if presolve == "on" else []),
            model_path,
        ]
    elif solver == "qsopt-ex":
        cmd = [
            command,
            "-L",
            "-O",
            solution_path,
            model_path,
        ]
    elif solver == "lp-solve":
        cmd = [
            command,
            "-timeout",
            str(max(1, int(math.ceil(time_limit)))),
            model_path,
        ]
    elif solver == "gurobi":
        cmd = [
            command,
            f"ResultFile={solution_path}",
            f"TimeLimit={time_limit}",
            *([f"NodeLimit={int(node_limit)}"] if kind == "mip" and node_limit is not None else []),
            *([f"SolutionLimit={int(solution_limit)}"] if kind == "mip" and solution_limit is not None else []),
            *([f"MIPGap={float(relative_gap):.17g}"] if kind == "mip" and relative_gap is not None else []),
            *([f"MIPGapAbs={float(absolute_gap):.17g}"] if kind == "mip" and absolute_gap is not None else []),
            *([f"BestObjStop={float(objective_limit):.17g}"] if kind == "mip" and objective_limit is not None else []),
            *([f"FeasibilityTol={float(primal_feasibility_tolerance):.17g}"] if primal_feasibility_tolerance is not None else []),
            *([f"OptimalityTol={float(dual_feasibility_tolerance):.17g}"] if dual_feasibility_tolerance is not None else []),
            *([f"IntFeasTol={float(integer_feasibility_tolerance):.17g}"] if kind == "mip" and integer_feasibility_tolerance is not None else []),
            *([f"Threads={int(threads)}"] if threads is not None else []),
            *([f"Seed={int(random_seed)}"] if random_seed is not None else []),
            *([f"Presolve={-1 if presolve == 'auto' else 1 if presolve == 'on' else 0}"] if presolve is not None else []),
            model_path,
        ]
    elif solver == "cplex":
        cmd = [
            command,
            "-c",
            f"read {model_path}",
            f"set timelimit {time_limit}",
            *([f"set mip limits nodes {int(node_limit)}"] if kind == "mip" and node_limit is not None else []),
            *([f"set mip limits solutions {int(solution_limit)}"] if kind == "mip" and solution_limit is not None else []),
            *([f"set mip tolerances mipgap {float(relative_gap):.17g}"] if kind == "mip" and relative_gap is not None else []),
            *([f"set mip tolerances absmipgap {float(absolute_gap):.17g}"] if kind == "mip" and absolute_gap is not None else []),
            *([f"set simplex tolerances feasibility {float(primal_feasibility_tolerance):.17g}"] if primal_feasibility_tolerance is not None else []),
            *([f"set simplex tolerances optimality {float(dual_feasibility_tolerance):.17g}"] if dual_feasibility_tolerance is not None else []),
            *([f"set mip tolerances integrality {float(integer_feasibility_tolerance):.17g}"] if kind == "mip" and integer_feasibility_tolerance is not None else []),
            *([f"set threads {int(threads)}"] if threads is not None else []),
            *([f"set randomseed {int(random_seed)}"] if random_seed is not None else []),
            *([f"set preprocessing presolve {'-1' if presolve == 'auto' else '1' if presolve == 'on' else '0'}"] if presolve is not None else []),
            "optimize",
            f"write {solution_path}",
            "quit",
        ]
    elif solver == "xpress":
        script_path = os.path.join(os.path.dirname(model_path), "xpress_commands.txt")
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(f"MAXTIME = -{max(1, int(math.ceil(time_limit)))}\n")
            if kind == "mip" and solution_limit is not None:
                f.write(f"MAXSOLS = {int(solution_limit)}\n")
            if primal_feasibility_tolerance is not None:
                f.write(f"FEASTOL = {float(primal_feasibility_tolerance):.17g}\n")
            if dual_feasibility_tolerance is not None:
                f.write(f"OPTTOL = {float(dual_feasibility_tolerance):.17g}\n")
            if kind == "mip" and integer_feasibility_tolerance is not None:
                f.write(f"MIPTOL = {float(integer_feasibility_tolerance):.17g}\n")
            if threads is not None:
                f.write(f"THREADS = {int(threads)}\n")
            if random_seed is not None:
                f.write(f"RANDOMSEED = {int(random_seed)}\n")
            if presolve is not None:
                f.write(f"PRESOLVE = {-1 if presolve == 'auto' else 1 if presolve == 'on' else 0}\n")
            read_flag = "-m" if model_format == "mps" else "-l"
            f.write(f"readprob {read_flag} {model_path}\n")
            f.write("mipoptimize\n" if kind == "mip" else "lpoptimize\n")
            f.write(f"writesol {solution_path} -npa\n")
            f.write("quit\n")
        cmd = [command, f"@{script_path}"]
    elif solver == "lindo":
        cmd = [
            command,
            model_path,
            "-sol",
            "-max" if sense == "max" else "-min",
        ]
        if kind == "mip":
            cmd.append("-mip")
        else:
            cmd.append("-lp")
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
    if solver == "lp-solve":
        with open(solution_path, "w", encoding="utf-8") as f:
            f.write(run.stdout)
    return run.stdout, run.stderr


def solve_solution_pool(
    solver: str,
    raw: dict,
    time_limit: float,
    solution_pool_size: int,
    model_format: str = "lp",
    node_limit: Optional[int] = None,
    solution_limit: Optional[int] = None,
    relative_gap: Optional[float] = None,
    absolute_gap: Optional[float] = None,
    objective_limit: Optional[float] = None,
    primal_feasibility_tolerance: Optional[float] = None,
    dual_feasibility_tolerance: Optional[float] = None,
    integer_feasibility_tolerance: Optional[float] = None,
    threads: Optional[int] = None,
    random_seed: Optional[int] = None,
    presolve: Optional[str] = None,
    cuts: Optional[str] = None,
    heuristics: Optional[str] = None,
    branch_rule: Optional[str] = None,
    branch_priorities: Optional[Sequence[int]] = None,
    node_selection: Optional[str] = None,
    mip_start: Optional[Sequence[float]] = None,
) -> dict:
    sense, c, rows, rhs, lbs, ubs, integer_vars = normalize_mip(raw)
    integer_indices = solution_pool_integer_indices(integer_vars)
    bound_error = validate_solution_pool_bounds(lbs, ubs, integer_indices)
    if bound_error is not None:
        return status_payload("unavailable", f"{solver}:cli", bound_error)

    original_n = len(c)
    working = plain_mip_payload(sense, c, rows, rhs, lbs, ubs, integer_vars)
    solutions: list[dict] = []
    seen: set[tuple[int, ...]] = set()
    exhausted = False
    message = ""
    overall_status = "optimal"

    for pool_idx in range(solution_pool_size):
        working_branch_priorities = None
        if branch_priorities is not None:
            working_n = len(working["c"])
            working_branch_priorities = list(branch_priorities[:working_n])
            working_branch_priorities.extend([0] * (working_n - len(working_branch_priorities)))
        working_mip_start = mip_start if pool_idx == 0 else None
        result = solve(
            "mip",
            solver,
            working,
            time_limit,
            model_format=model_format,
            node_limit=node_limit,
            solution_limit=solution_limit,
            relative_gap=relative_gap,
            absolute_gap=absolute_gap,
            objective_limit=objective_limit,
            primal_feasibility_tolerance=primal_feasibility_tolerance,
            dual_feasibility_tolerance=dual_feasibility_tolerance,
            integer_feasibility_tolerance=integer_feasibility_tolerance,
            threads=threads,
            random_seed=random_seed,
            presolve=presolve,
            cuts=cuts,
            heuristics=heuristics,
            branch_rule=branch_rule,
            branch_priorities=working_branch_priorities,
            node_selection=node_selection,
            mip_start=working_mip_start,
        )
        if result["status"] in {"infeasible", "unbounded"}:
            exhausted = result["status"] == "infeasible"
            message = "pool exhausted by no-good cuts" if exhausted else result.get("message", "")
            break
        if result["status"] not in {"optimal", "feasible"}:
            overall_status = "feasible" if solutions else result["status"]
            message = result.get("message", "")
            break
        if result["status"] == "feasible":
            overall_status = "feasible"

        x = [float(value) for value in result["x"][:original_n]]
        key = solution_pool_assignment_key(x, integer_indices)
        if key in seen:
            overall_status = "feasible"
            message = "pool search stopped after duplicate integer assignment"
            break
        seen.add(key)

        objective = dot(c, x)
        solutions.append(payload_solution(x, objective))
        cut_error = add_solution_pool_no_good_cut(working, integer_indices, x)
        if cut_error is not None:
            message = cut_error
            exhausted = True
            break
        mip_start = None

    if len(solutions) == solution_pool_size and not exhausted:
        message = "pool reached solution_pool_size"
    if not solutions:
        return {
            **status_payload(
                "infeasible" if exhausted else overall_status,
                f"{solver}:cli",
                message,
                result.get("solverVersion") if "result" in locals() else None,
            ),
            "solutions": [],
            "solutionPoolSize": solution_pool_size,
            "exhausted": exhausted,
        }

    payload = {
        "status": overall_status,
        "solver": f"{solver}:cli",
        "x": solutions[0]["x"],
        "objective": solutions[0]["objective"],
        "message": message,
        "solutions": solutions,
        "solutionPoolSize": solution_pool_size,
        "exhausted": exhausted,
    }
    if "result" in locals() and result.get("solverVersion") is not None:
        payload["solverVersion"] = result["solverVersion"]
    for key in (
        "primalFeasibilityTolerance",
        "dualFeasibilityTolerance",
        "integerFeasibilityTolerance",
        "branchPrioritiesAccepted",
        "branchPriorityCount",
    ):
        if "result" in locals() and result.get(key) is not None:
            payload[key] = result[key]
    return payload


def solve_multi_objective(
    solver: str,
    sense: str,
    c: Sequence[float],
    rows: Sequence[Sequence[float]],
    rhs: Sequence[float],
    lbs: Sequence[Optional[float]],
    ubs: Sequence[Optional[float]],
    integer_vars: Sequence[bool],
    objectives: Sequence[dict],
    time_limit: float,
    node_limit: Optional[int] = None,
    solution_limit: Optional[int] = None,
    relative_gap: Optional[float] = None,
    absolute_gap: Optional[float] = None,
    objective_limit: Optional[float] = None,
    primal_feasibility_tolerance: Optional[float] = None,
    dual_feasibility_tolerance: Optional[float] = None,
    integer_feasibility_tolerance: Optional[float] = None,
    threads: Optional[int] = None,
    random_seed: Optional[int] = None,
    presolve: Optional[str] = None,
    cuts: Optional[str] = None,
    heuristics: Optional[str] = None,
    branch_rule: Optional[str] = None,
    branch_priorities: Optional[Sequence[int]] = None,
    node_selection: Optional[str] = None,
    mip_start: Optional[Sequence[float]] = None,
) -> dict:
    if not objectives:
        return status_payload("unavailable", f"{solver}:cli", "multi_objectives must be non-empty")

    working = plain_mip_payload(sense, c, rows, rhs, lbs, ubs, integer_vars)
    stage_results: list[dict] = []
    stage_mip_start = mip_start

    for idx, objective in enumerate(objectives):
        coeffs = [float(v) for v in objective["c"]]
        working["sense"] = objective["sense"]
        working["c"] = coeffs
        result = solve(
            "mip",
            solver,
            working,
            time_limit,
            node_limit=node_limit,
            solution_limit=solution_limit,
            relative_gap=relative_gap,
            absolute_gap=absolute_gap,
            objective_limit=objective_limit,
            primal_feasibility_tolerance=primal_feasibility_tolerance,
            dual_feasibility_tolerance=dual_feasibility_tolerance,
            integer_feasibility_tolerance=integer_feasibility_tolerance,
            threads=threads,
            random_seed=random_seed,
            presolve=presolve,
            cuts=cuts,
            heuristics=heuristics,
            branch_rule=branch_rule,
            branch_priorities=branch_priorities,
            node_selection=node_selection,
            mip_start=stage_mip_start,
        )
        stage_results.append(result)
        if result["status"] != "optimal":
            payload = status_payload(
                result["status"],
                result.get("solver", f"{solver}:cli"),
                result.get("message", ""),
                result.get("solverVersion"),
            )
            payload["x"] = result.get("x", [])
            payload["objective"] = result.get("objective")
            payload["objectiveValues"] = []
            payload["stageCount"] = len(stage_results)
            return payload

        x = [float(value) for value in result["x"]]
        optimum = dot(coeffs, x)
        name = str(objective.get("name") or f"multi_objective_{idx}")
        add_objective_lock_rows(working, coeffs, optimum, name)
        stage_mip_start = x

    final_x = [float(value) for value in stage_results[-1]["x"]]
    objective_values = [dot([float(v) for v in objective["c"]], final_x) for objective in objectives]
    payload = {
        "status": "optimal",
        "solver": f"{solver}:cli",
        "x": final_x,
        "objective": objective_values[-1] if objective_values else None,
        "objectiveValues": objective_values,
        "stageCount": len(stage_results),
        "message": "sequential lexicographic optimization",
    }
    if stage_results[-1].get("solverVersion") is not None:
        payload["solverVersion"] = stage_results[-1]["solverVersion"]
    for key in (
        "primalFeasibilityTolerance",
        "dualFeasibilityTolerance",
        "integerFeasibilityTolerance",
        "branchPrioritiesAccepted",
        "branchPriorityCount",
    ):
        if stage_results[-1].get(key) is not None:
            payload[key] = stage_results[-1][key]
    return payload


def solve(
    kind: str,
    solver: str,
    raw: dict,
    time_limit: float,
    model_format: str = "lp",
    node_limit: Optional[int] = None,
    solution_limit: Optional[int] = None,
    relative_gap: Optional[float] = None,
    absolute_gap: Optional[float] = None,
    objective_limit: Optional[float] = None,
    primal_feasibility_tolerance: Optional[float] = None,
    dual_feasibility_tolerance: Optional[float] = None,
    integer_feasibility_tolerance: Optional[float] = None,
    lp_algorithm: Optional[str] = None,
    threads: Optional[int] = None,
    random_seed: Optional[int] = None,
    presolve: Optional[str] = None,
    cuts: Optional[str] = None,
    heuristics: Optional[str] = None,
    branch_rule: Optional[str] = None,
    branch_priorities: Optional[Sequence[int]] = None,
    node_selection: Optional[str] = None,
    mip_start: Optional[Sequence[float]] = None,
    solution_pool_size: Optional[int] = None,
) -> dict:
    if not solver_available(solver):
        return status_payload("unavailable", f"{solver}:cli", f"{solver} executable not found")
    if solver not in SUPPORTED_SOLVERS:
        return status_payload(
            "unavailable",
            f"{solver}:cli",
            f"{solver} executable found, but this bridge does not yet know the non-interactive solve command",
        )
    if model_format not in {"lp", "mps"}:
        raise ValueError("model_format must be 'lp' or 'mps'")

    if kind == "lp":
        lp = raw.get("lp", raw)
        sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs = normalize_lp(lp)
        integer_vars = [False] * len(c)
        lp_algorithm = normalized_lp_algorithm(lp_algorithm)
        presolve = normalized_presolve(presolve)
        primal_feasibility_tolerance = normalized_tolerance(primal_feasibility_tolerance)
        dual_feasibility_tolerance = normalized_tolerance(dual_feasibility_tolerance)
        integer_feasibility_tolerance = None
    elif kind == "mip":
        if solver in {"clp", "soplex", "qsopt-ex"}:
            return status_payload("unavailable", f"{solver}:cli", f"{solver} is LP-only")
        sense, c, a_ub, b_ub, lbs, ubs, integer_vars = normalize_mip(raw)
        a_eq, b_eq = [], []
        node_limit = normalized_node_limit(node_limit)
        solution_limit = normalized_solution_limit(solution_limit)
        solution_pool_size = normalized_solution_pool_size(solution_pool_size)
        relative_gap = normalized_relative_gap(relative_gap)
        absolute_gap = normalized_absolute_gap(absolute_gap)
        objective_limit = normalized_objective_limit(objective_limit)
        primal_feasibility_tolerance = normalized_tolerance(primal_feasibility_tolerance)
        dual_feasibility_tolerance = normalized_tolerance(dual_feasibility_tolerance)
        integer_feasibility_tolerance = normalized_tolerance(integer_feasibility_tolerance)
        presolve = normalized_presolve(presolve)
        cuts = normalized_mip_switch(cuts, "cuts")
        heuristics = normalized_mip_switch(heuristics, "heuristics")
        branch_rule = normalized_branch_rule(branch_rule)
        if branch_priorities is None:
            branch_priorities = raw.get("branch_priorities") or raw.get("branchPriorities")
        branch_priorities = normalized_branch_priorities(branch_priorities, len(c))
        node_selection = normalized_node_selection(node_selection)
        if mip_start is None:
            mip_start = raw.get("mip_start") or raw.get("mipStart")
        mip_start = normalized_mip_start(mip_start, len(c))
        multi_objectives = normalized_multi_objectives(raw, len(c))
        if multi_objectives:
            if solution_pool_size is not None:
                return status_payload(
                    "unavailable",
                    f"{solver}:cli",
                    "solution pools for multi-objective MIPs are not supported",
                )
            return solve_multi_objective(
                solver,
                sense,
                c,
                a_ub,
                b_ub,
                lbs,
                ubs,
                integer_vars,
                multi_objectives,
                time_limit,
                node_limit=node_limit,
                solution_limit=solution_limit,
                relative_gap=relative_gap,
                absolute_gap=absolute_gap,
                objective_limit=objective_limit,
                primal_feasibility_tolerance=primal_feasibility_tolerance,
                dual_feasibility_tolerance=dual_feasibility_tolerance,
                integer_feasibility_tolerance=integer_feasibility_tolerance,
                threads=threads,
                random_seed=random_seed,
                presolve=presolve,
                cuts=cuts,
                heuristics=heuristics,
                branch_rule=branch_rule,
                branch_priorities=branch_priorities,
                node_selection=node_selection,
                mip_start=mip_start,
            )
        if solution_pool_size is not None:
            return solve_solution_pool(
                solver,
                raw,
                time_limit,
                solution_pool_size,
                model_format=model_format,
                node_limit=node_limit,
                solution_limit=solution_limit,
                relative_gap=relative_gap,
                absolute_gap=absolute_gap,
                objective_limit=objective_limit,
                primal_feasibility_tolerance=primal_feasibility_tolerance,
                dual_feasibility_tolerance=dual_feasibility_tolerance,
                integer_feasibility_tolerance=integer_feasibility_tolerance,
                threads=threads,
                random_seed=random_seed,
                presolve=presolve,
                cuts=cuts,
                heuristics=heuristics,
                branch_rule=branch_rule,
                branch_priorities=branch_priorities,
                node_selection=node_selection,
                mip_start=mip_start,
            )
    else:
        raise ValueError("kind must be 'lp' or 'mip'")
    threads = normalized_threads(threads)
    random_seed = normalized_random_seed(random_seed)
    mip_start_objective = dot(c, mip_start) if kind == "mip" and mip_start is not None else None

    with tempfile.TemporaryDirectory(prefix="ores-linear-cli-") as tmp:
        effective_model_format = (
            "mps"
            if solver == "lindo"
            else "lp"
            if solver in {"lp-solve", "qsopt-ex"}
            else model_format
        )
        extension = "mps" if effective_model_format == "mps" else "lp"
        model_path = os.path.join(tmp, f"model.{extension}")
        solution_path = (
            os.path.join(tmp, "xpress_solution")
            if solver == "xpress"
            else os.path.join(tmp, "model.sol")
            if solver == "lindo"
            else os.path.join(tmp, f"{solver}.sol")
        )
        if effective_model_format == "mps":
            write_free_mps(
                model_path,
                sense,
                c,
                a_ub,
                b_ub,
                a_eq,
                b_eq,
                lbs,
                ubs,
                integer_vars,
                include_objsense=solver != "glpk",
            )
        elif solver == "lp-solve":
            write_lpsolve_lp(
                model_path,
                sense,
                c,
                a_ub,
                b_ub,
                a_eq,
                b_eq,
                lbs,
                ubs,
                integer_vars,
            )
        else:
            write_cplex_lp(model_path, sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs, integer_vars)
        stdout, stderr = run_solver(
            solver,
            kind,
            sense,
            model_path,
            solution_path,
            time_limit,
            effective_model_format,
            node_limit,
            solution_limit,
            relative_gap,
            absolute_gap,
            objective_limit,
            primal_feasibility_tolerance,
            dual_feasibility_tolerance,
            integer_feasibility_tolerance,
            lp_algorithm,
            threads,
            random_seed,
            presolve,
            cuts,
            heuristics,
            branch_rule,
            branch_priorities,
            node_selection,
            mip_start,
            mip_start_objective,
            integer_vars,
        )
        if solver == "xpress":
            solution_path = _first_existing_path(
                [solution_path + ".asc", solution_path, solution_path + ".sol"]
            ) or solution_path
        elif solver == "lindo":
            solution_path = _first_existing_path(
                [solution_path, os.path.splitext(model_path)[0] + ".sol"]
            ) or solution_path
        if not os.path.exists(solution_path):
            classified = classify_status("", stdout, stderr)
            solver_version = solver_version_from_output(solver, stdout, stderr)
            if classified in ("infeasible", "unbounded"):
                return status_payload(
                    classified,
                    f"{solver}:cli",
                    (stderr or stdout).strip(),
                    solver_version,
                )
            return status_payload(
                "unavailable",
                f"{solver}:cli",
                (stderr or stdout).strip(),
                solver_version,
            )
        certificate_fields: dict[str, object] = {}
        if solver == "highs":
            status, x, certificate_fields = parse_highs_solution(
                solution_path,
                len(c),
                len(a_ub),
                len(a_eq),
            )
        elif solver == "glpk":
            status, x, certificate_fields = parse_glpk_solution(
                solution_path,
                len(c),
                len(a_ub),
                len(a_eq),
            )
        elif solver == "scip":
            status, x = parse_scip_solution(solution_path, len(c))
        elif solver == "gurobi":
            status, x = parse_named_solution(solution_path, len(c))
        elif solver == "cplex":
            status, x = parse_cplex_solution(solution_path, len(c))
        elif solver == "xpress":
            status, x = parse_xpress_solution(solution_path, len(c))
        elif solver == "lindo":
            status, x = parse_lindo_solution(solution_path, len(c))
        elif solver == "lp-solve":
            status, x = parse_lp_solve_solution(solution_path, len(c))
        elif solver == "soplex":
            status, x = parse_soplex_solution(solution_path, len(c), stdout, stderr)
        elif solver == "qsopt-ex":
            status, x = parse_qsopt_ex_solution(solution_path, len(c), stdout, stderr)
        else:
            status, x, certificate_fields = parse_cbc_solution(
                solution_path,
                len(c),
                len(a_ub),
                len(a_eq),
                solution_path + ".basis" if kind == "lp" else None,
            )

    classified = classify_status(status, stdout, stderr)
    solver_version = solver_version_from_output(solver, stdout, stderr)
    if classified not in {"optimal", "feasible"}:
        return status_payload(
            classified if classified in ("infeasible", "unbounded") else "unavailable",
            f"{solver}:cli",
            status,
            solver_version,
        )
    objective = dot(c, x)
    result = {
        "status": classified,
        "solver": f"{solver}:cli",
        "x": x,
        "objective": objective,
        "message": status,
    }
    if solver_version is not None:
        result["solverVersion"] = solver_version
    if kind == "lp":
        result.update(certificate_fields)
        result.update(parse_lp_iterations(solver, kind, stdout, stderr))
    result.update(
        parse_lp_algorithm_feedback(
            solver,
            kind,
            lp_algorithm,
            stdout,
            stderr,
        )
    )
    result.update(
        parse_operational_control_feedback(
            solver,
            kind,
            threads,
            random_seed,
            presolve,
            stdout,
            stderr,
        )
    )
    result.update(
        parse_mip_strategy_feedback(
            solver,
            kind,
            cuts,
            heuristics,
            stdout,
            stderr,
        )
    )
    result.update(
        parse_solution_limit_feedback(
            solver,
            kind,
            solution_limit,
            stdout,
            stderr,
        )
    )
    result.update(
        parse_objective_limit_feedback(
            solver,
            kind,
            objective_limit,
            stdout,
            stderr,
        )
    )
    result.update(
        parse_tolerance_feedback(
            solver,
            kind,
            primal_feasibility_tolerance,
            dual_feasibility_tolerance,
            integer_feasibility_tolerance,
            stdout,
            stderr,
        )
    )
    result.update(
        parse_mip_start_feedback(
            solver,
            kind,
            mip_start,
            mip_start_objective,
            stdout,
            stderr,
        )
    )
    result.update(
        parse_search_control_feedback(
            solver,
            kind,
            branch_rule,
            node_selection,
            stdout,
            stderr,
        )
    )
    result.update(
        parse_branch_priority_feedback(
            solver,
            kind,
            branch_priorities,
            integer_vars,
            stdout,
            stderr,
        )
    )
    result.update(parse_mip_quality(solver, kind, objective, stdout, stderr))
    return result


def _first_existing_path(paths: Sequence[str]) -> Optional[str]:
    for path in paths:
        if os.path.exists(path):
            return path
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["lp", "mip"], required=True)
    parser.add_argument("--solver", choices=sorted(COMMAND_ALIASES.keys()), required=True)
    parser.add_argument("--problem")
    parser.add_argument("--model-format", choices=["lp", "mps"], default="lp")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--node-limit", type=int)
    parser.add_argument("--solution-limit", type=int)
    parser.add_argument("--solution-pool-size", type=int)
    parser.add_argument("--relative-gap", type=float)
    parser.add_argument("--absolute-gap", type=float)
    parser.add_argument("--objective-limit", type=float)
    parser.add_argument("--primal-feasibility-tolerance", type=float)
    parser.add_argument("--dual-feasibility-tolerance", type=float)
    parser.add_argument("--integer-feasibility-tolerance", type=float)
    parser.add_argument("--lp-algorithm", choices=["simplex", "ipm"])
    parser.add_argument("--threads", type=int)
    parser.add_argument("--random-seed", type=int)
    parser.add_argument("--presolve", choices=["auto", "on", "off"])
    parser.add_argument("--cuts", choices=["auto", "on", "off"])
    parser.add_argument("--heuristics", choices=["auto", "on", "off"])
    parser.add_argument("--branch-rule", choices=["first-fractional", "most-fractional"])
    parser.add_argument("--branch-priorities")
    parser.add_argument("--node-selection", choices=["dfs", "best-bound"])
    parser.add_argument("--mip-start")
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
                    args.solution_limit,
                    args.relative_gap,
                    args.absolute_gap,
                    args.objective_limit,
                    args.primal_feasibility_tolerance,
                    args.dual_feasibility_tolerance,
                    args.integer_feasibility_tolerance,
                    args.lp_algorithm,
                    args.threads,
                    args.random_seed,
                    args.presolve,
                    args.cuts,
                    args.heuristics,
                    args.branch_rule,
                    parse_int_list_arg(args.branch_priorities, "branch-priorities"),
                    args.node_selection,
                    parse_mip_start_arg(args.mip_start),
                    args.solution_pool_size,
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
