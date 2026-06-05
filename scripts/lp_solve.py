#!/usr/bin/env python3
"""Small LP bridge used by the Rust validation harness.

Input is JSON on stdin:
  {"lp": {...}, "method": "highs"}

The bridge supports SciPy/HiGHS methods plus OR-Tools GLOP/PDLP. If the requested
solver is unavailable, it falls back to the Rust internal simplex reference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple


def status_payload(status: str, solver: str, message: str = "") -> dict:
    return {
        "status": status,
        "x": [],
        "objective": None,
        "iters": None,
        "solver": solver,
        "message": message,
    }


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "lp_solve_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "lp.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "lp_solve_reference"
    explicit = os.environ.get("LP_SOLVE_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def rust_reference(lp: dict, method: str = "fallback") -> dict:
    command = rust_reference_command()
    cwd = None
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)
    completed = subprocess.run(
        [*command, "--method", method],
        input=json.dumps({"lp": lp, "method": method}),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    try:
        parsed = json.loads(completed.stdout)
    except Exception as exc:
        return status_payload(
            "numerical-error",
            "rust:internal-simplex",
            f"failed to parse Rust LP output: {exc}; stderr={completed.stderr.strip()}",
        )
    if completed.returncode != 0 and not parsed.get("message"):
        parsed["message"] = completed.stderr.strip()
    return parsed


def objective_offset(lp: dict) -> float:
    return float(lp.get("objective_offset", lp.get("objectiveOffset", 0.0)) or 0.0)


def apply_objective_offset(lp: dict, result: dict) -> dict:
    if result.get("objective") is not None:
        result["objective"] = float(result["objective"]) + objective_offset(lp)
    return result


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def solve_square(a: Sequence[Sequence[float]], b: Sequence[float]) -> Optional[List[float]]:
    n = len(b)
    if n == 0:
        return []
    aug = [list(map(float, row)) + [float(bi)] for row, bi in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) <= 1e-10:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor == 0:
                continue
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def normalize_lp(lp: dict) -> Tuple[str, List[float], List[List[float]], List[float], List[List[float]], List[float], List[Optional[float]], List[Optional[float]]]:
    c = [float(v) for v in lp.get("c", [])]
    n = len(c)
    sense = lp.get("sense", "max")
    a_ub = [list(map(float, row)) for row in lp.get("A_ub", lp.get("a_ub", [])) or []]
    b_ub = [float(v) for v in (lp.get("b_ub", []) or [])]
    a_eq = [list(map(float, row)) for row in lp.get("A_eq", lp.get("a_eq", [])) or []]
    b_eq = [float(v) for v in (lp.get("b_eq", []) or [])]
    for idx, row_bound in enumerate(lp.get("linear_constraints", []) or []):
        row = [float(v) for v in row_bound["coefs"]]
        if len(row) != n:
            raise ValueError(f"linear constraint {idx} row length mismatch")
        lower_raw = row_bound.get("lower")
        upper_raw = row_bound.get("upper")
        lower = None if lower_raw is None else float(lower_raw)
        upper = None if upper_raw is None else float(upper_raw)
        if lower is None and upper is None:
            raise ValueError(f"linear constraint {idx} needs lower or upper bound")
        if lower is not None and upper is not None and lower > upper + 1e-9:
            raise ValueError(f"linear constraint {idx} lower exceeds upper")
        if lower is not None and upper is not None and abs(lower - upper) <= 1e-9:
            a_eq.append(row)
            b_eq.append(upper)
            continue
        if upper is not None:
            a_ub.append(row[:])
            b_ub.append(upper)
        if lower is not None:
            a_ub.append([-v for v in row])
            b_ub.append(-lower)
    lb = lp.get("lb")
    ub = lp.get("ub")
    lbs = [0.0] * n if lb is None else [None if v is None else float(v) for v in lb]
    ubs = [None] * n if ub is None else [None if v is None else float(v) for v in ub]
    if len(lbs) != n or len(ubs) != n:
        raise ValueError("bound vector length mismatch")
    if len(a_ub) != len(b_ub) or len(a_eq) != len(b_eq):
        raise ValueError("constraint matrix/vector length mismatch")
    for row in a_ub + a_eq:
        if len(row) != n:
            raise ValueError("constraint row length mismatch")
    return sense, c, a_ub, b_ub, a_eq, b_eq, lbs, ubs


def recover_certificate(lp: dict, x: Sequence[float], tol: float = 1e-8) -> dict:
    sense, c, a_ub, b_ub, a_eq, b_eq, lb, ub = normalize_lp(lp)
    n = len(c)
    if len(x) != n:
        return {}
    bound_state = [0] * n
    for j, xj in enumerate(x):
        xj = float(xj)
        if lb[j] is not None:
            lower = float(lb[j])
            if xj < lower - 10.0 * tol:
                return {}
            if abs(xj - lower) <= 10.0 * tol:
                bound_state[j] = -1
        if ub[j] is not None:
            upper = float(ub[j])
            if xj > upper + 10.0 * tol:
                return {}
            if abs(xj - upper) <= 10.0 * tol:
                bound_state[j] = 2 if bound_state[j] == -1 else 1

    active = []
    for i, (row, rhs) in enumerate(zip(a_ub, b_ub)):
        lhs = dot(row, x)
        if lhs > rhs + 10.0 * tol:
            return {}
        if abs(lhs - rhs) <= 10.0 * tol:
            active.append(i)
    for row, rhs in zip(a_eq, b_eq):
        if abs(dot(row, x) - rhs) > 10.0 * tol:
            return {}

    interior = [j for j, state in enumerate(bound_state) if state == 0]
    unknowns = len(active) + len(a_eq)
    if unknowns != len(interior):
        return {}
    system = [[0.0 for _ in range(unknowns)] for _ in interior]
    for col, row_idx in enumerate(active):
        for eq_row, j in enumerate(interior):
            system[eq_row][col] = a_ub[row_idx][j]
    for eq_idx, row in enumerate(a_eq):
        col = len(active) + eq_idx
        for eq_row, j in enumerate(interior):
            system[eq_row][col] = row[j]
    gradient = [coef if sense == "max" else -coef for coef in c]
    rhs = [gradient[j] for j in interior]
    if unknowns == 0:
        solution = []
    else:
        solution = solve_square(system, rhs)
        if solution is None:
            return {}

    dual_ub = [0.0] * len(a_ub)
    for col, row_idx in enumerate(active):
        if solution[col] < -1e-7:
            return {}
        dual_ub[row_idx] = max(0.0, float(solution[col]))
    dual_eq = [float(v) for v in solution[len(active):]]
    reduced = gradient[:]
    for row, dual in zip(a_ub, dual_ub):
        if dual == 0.0:
            continue
        for j in range(n):
            reduced[j] -= dual * row[j]
    for row, dual in zip(a_eq, dual_eq):
        for j in range(n):
            reduced[j] -= dual * row[j]
    for j, state in enumerate(bound_state):
        if state == 0 and abs(reduced[j]) > 1e-7:
            return {}
        if state == -1 and reduced[j] > 1e-7:
            return {}
        if state == 1 and reduced[j] < -1e-7:
            return {}
    return {
        "dualUB": dual_ub,
        "dualEQ": dual_eq,
        "reducedCosts": reduced,
    }


def scipy_linprog(lp: dict, method: str) -> Optional[dict]:
    try:
        from scipy.optimize import linprog  # type: ignore
    except Exception:
        return None
    sense, c, a_ub, b_ub, a_eq, b_eq, lb, ub = normalize_lp(lp)
    scipy_c = [-v for v in c] if sense == "max" else c[:]
    bounds = [
        (
            None if l is None else float(l),
            None if u is None else float(u),
        )
        for l, u in zip(lb, ub)
    ]
    result = linprog(
        scipy_c,
        A_ub=a_ub or None,
        b_ub=b_ub or None,
        A_eq=a_eq or None,
        b_eq=b_eq or None,
        bounds=bounds,
        method=method,
    )
    status_map = {0: "optimal", 2: "infeasible", 3: "unbounded", 1: "iter-limit"}
    status = status_map.get(int(result.status), "numerical-error")
    x = [] if result.x is None else [float(v) for v in result.x]
    objective = None
    if status == "optimal":
        objective = dot(c, x)
    result_payload = {
        "status": status,
        "x": x,
        "objective": objective,
        "iters": int(getattr(result, "nit", 0) or 0),
        "solver": f"scipy:{method}",
        "message": str(result.message),
    }
    if status == "optimal":
        result_payload.update(recover_certificate(lp, x))
    return apply_objective_offset(lp, result_payload)


def ortools_linear_solver(lp: dict, backend: str) -> Optional[dict]:
    try:
        from ortools.linear_solver import pywraplp  # type: ignore
    except Exception:
        return None

    solver_name = backend.upper()
    solver = pywraplp.Solver.CreateSolver(solver_name)
    if solver is None:
        return None

    sense, c, a_ub, b_ub, a_eq, b_eq, lb, ub = normalize_lp(lp)
    inf = solver.infinity()
    xs = []
    for j, (lower, upper) in enumerate(zip(lb, ub)):
        xs.append(
            solver.NumVar(
                -inf if lower is None else float(lower),
                inf if upper is None else float(upper),
                f"x{j}",
            )
        )

    for i, (row, rhs) in enumerate(zip(a_ub, b_ub)):
        constraint = solver.RowConstraint(-inf, float(rhs), f"ub{i}")
        for coef, var in zip(row, xs):
            if abs(coef) > 1e-12:
                constraint.SetCoefficient(var, float(coef))

    for i, (row, rhs) in enumerate(zip(a_eq, b_eq)):
        constraint = solver.RowConstraint(float(rhs), float(rhs), f"eq{i}")
        for coef, var in zip(row, xs):
            if abs(coef) > 1e-12:
                constraint.SetCoefficient(var, float(coef))

    objective = solver.Objective()
    for coef, var in zip(c, xs):
        if abs(coef) > 1e-12:
            objective.SetCoefficient(var, float(coef))
    if sense == "max":
        objective.SetMaximization()
    else:
        objective.SetMinimization()

    status_code = solver.Solve()
    status_map = {
        pywraplp.Solver.OPTIMAL: "optimal",
        pywraplp.Solver.FEASIBLE: "feasible",
        pywraplp.Solver.INFEASIBLE: "infeasible",
        pywraplp.Solver.UNBOUNDED: "unbounded",
        pywraplp.Solver.ABNORMAL: "numerical-error",
        pywraplp.Solver.NOT_SOLVED: "iter-limit",
    }
    status = status_map.get(status_code, "numerical-error")
    x = [float(var.solution_value()) for var in xs] if status in ("optimal", "feasible") else []
    iters = solver.iterations() if hasattr(solver, "iterations") else 0
    result_payload = {
        "status": status,
        "x": x,
        "objective": dot(c, x) if status in ("optimal", "feasible") else None,
        "iters": int(iters or 0),
        "solver": f"ortools:{backend.lower()}",
        "message": f"{solver_name} status code {status_code}",
    }
    if status == "optimal":
        result_payload.update(recover_certificate(lp, x))
    return apply_objective_offset(lp, result_payload)


def solve_external(lp: dict, method: str) -> Optional[dict]:
    normalized = method.lower().replace("_", "-")
    if normalized in ("rust", "fallback", "internal", "internal-simplex", "rust-internal"):
        return rust_reference(lp, method)
    if normalized in ("glop", "ortools-glop", "ortools:glop"):
        return ortools_linear_solver(lp, "glop")
    if normalized in ("pdlp", "ortools-pdlp", "ortools:pdlp"):
        return ortools_linear_solver(lp, "pdlp")
    return scipy_linprog(lp, method)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="highs")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        lp = payload.get("lp", payload)
        method = payload.get("method", args.method)
        result = solve_external(lp, method)
        if result is None:
            result = rust_reference(lp, "fallback")
        print(json.dumps(result, allow_nan=True))
        return 0
    except Exception as exc:
        print(json.dumps(status_payload("numerical-error", "python:lp-bridge", str(exc)), allow_nan=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
