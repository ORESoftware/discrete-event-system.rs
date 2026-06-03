#!/usr/bin/env python3
"""Small LP bridge used by the Rust validation harness.

Input is JSON on stdin:
  {"lp": {...}, "method": "highs"}

The bridge supports SciPy/HiGHS methods plus OR-Tools GLOP. If the requested
solver is unavailable, it falls back to dependency-free vertex enumeration,
which is intended for small validation models rather than production-scale LPs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
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


def with_bound_inequalities(
    n: int,
    a_ub: List[List[float]],
    b_ub: List[float],
    lb: Sequence[Optional[float]],
    ub: Sequence[Optional[float]],
) -> Tuple[List[List[float]], List[float]]:
    rows = [row[:] for row in a_ub]
    rhs = b_ub[:]
    for j in range(n):
        if ub[j] is not None:
            row = [0.0] * n
            row[j] = 1.0
            rows.append(row)
            rhs.append(float(ub[j]))
        if lb[j] is not None:
            row = [0.0] * n
            row[j] = -1.0
            rows.append(row)
            rhs.append(-float(lb[j]))
    return rows, rhs


def feasible(
    x: Sequence[float],
    a_ub: Sequence[Sequence[float]],
    b_ub: Sequence[float],
    a_eq: Sequence[Sequence[float]],
    b_eq: Sequence[float],
) -> bool:
    for row, rhs in zip(a_ub, b_ub):
        if dot(row, x) > rhs + 1e-7:
            return False
    for row, rhs in zip(a_eq, b_eq):
        if abs(dot(row, x) - rhs) > 1e-7:
            return False
    return True


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


def rank(rows: Sequence[Sequence[float]]) -> int:
    if not rows:
        return 0
    work = [list(map(float, row)) for row in rows]
    m, n = len(work), len(work[0])
    r = 0
    for c in range(n):
        pivot = max(range(r, m), key=lambda i: abs(work[i][c]), default=r)
        if pivot >= m or abs(work[pivot][c]) <= 1e-10:
            continue
        work[r], work[pivot] = work[pivot], work[r]
        pv = work[r][c]
        for j in range(c, n):
            work[r][j] /= pv
        for i in range(m):
            if i == r:
                continue
            factor = work[i][c]
            for j in range(c, n):
                work[i][j] -= factor * work[r][j]
        r += 1
        if r == m:
            break
    return r


def null_vector_rank_n_minus_one(rows: Sequence[Sequence[float]], n: int) -> Optional[List[float]]:
    if n == 0:
        return None
    work = [list(map(float, row)) for row in rows if any(abs(float(v)) > 1e-10 for v in row)]
    pivots: List[int] = []
    r = 0
    for c in range(n):
        pivot = max(range(r, len(work)), key=lambda i: abs(work[i][c]), default=r)
        if pivot >= len(work) or abs(work[pivot][c]) <= 1e-10:
            continue
        work[r], work[pivot] = work[pivot], work[r]
        pv = work[r][c]
        for j in range(c, n):
            work[r][j] /= pv
        for i in range(len(work)):
            if i == r:
                continue
            factor = work[i][c]
            if abs(factor) <= 1e-15:
                continue
            for j in range(c, n):
                work[i][j] -= factor * work[r][j]
        pivots.append(c)
        r += 1
        if r == len(work):
            break
    if len(pivots) != n - 1:
        return None
    free_cols = [c for c in range(n) if c not in pivots]
    if len(free_cols) != 1:
        return None
    free = free_cols[0]
    d = [0.0] * n
    d[free] = 1.0
    for row_idx, pivot_col in enumerate(pivots):
        d[pivot_col] = -work[row_idx][free]
    norm = max(abs(v) for v in d)
    if norm <= 1e-12:
        return None
    return [v / norm for v in d]


def improving_recession_ray(
    sense: str,
    c: Sequence[float],
    a_ub: Sequence[Sequence[float]],
    a_eq: Sequence[Sequence[float]],
) -> Optional[List[float]]:
    n = len(c)
    if n == 0:
        return None
    objective_sign = 1.0 if sense == "max" else -1.0
    active_needed = max(0, n - 1 - rank(a_eq))
    if active_needed > len(a_ub):
        candidates = [()]
    else:
        candidates = itertools.combinations(range(len(a_ub)), active_needed)
    for active in candidates:
        rows = [list(row) for row in a_eq] + [list(a_ub[i]) for i in active]
        ray = null_vector_rank_n_minus_one(rows, n)
        if ray is None:
            continue
        for direction in (ray, [-v for v in ray]):
            if all(dot(row, direction) <= 1e-8 for row in a_ub) and all(
                abs(dot(row, direction)) <= 1e-8 for row in a_eq
            ):
                improvement = objective_sign * dot(c, direction)
                if improvement > 1e-8:
                    return direction
    return None


def vertex_enumeration(lp: dict) -> dict:
    sense, c, raw_a_ub, raw_b_ub, a_eq, b_eq, lb, ub = normalize_lp(lp)
    n = len(c)
    solver = "python:vertex-enumeration"
    a_ub, b_ub = with_bound_inequalities(n, raw_a_ub, raw_b_ub, lb, ub)
    if n == 0:
        if feasible([], a_ub, b_ub, a_eq, b_eq):
            return apply_objective_offset(lp, {"status": "optimal", "x": [], "objective": 0.0, "iters": 0, "solver": solver})
        return status_payload("infeasible", solver, "empty LP violates constraints")

    eq_rank = rank(a_eq)
    if eq_rank > n:
        return status_payload("infeasible", solver, "equality system rank exceeds variable count")
    need = n - eq_rank
    candidates: List[List[float]] = []
    if need < 0:
        return status_payload("infeasible", solver, "too many independent equalities")
    if need == 0:
        x = solve_square(a_eq[:n], b_eq[:n]) if len(a_eq) >= n else None
        if x is not None and feasible(x, a_ub, b_ub, a_eq, b_eq):
            candidates.append(x)
    else:
        if need > len(a_ub):
            return status_payload(
                "numerical-error",
                solver,
                "not enough finite active constraints for dependency-free vertex enumeration",
            )
        for active in itertools.combinations(range(len(a_ub)), need):
            mat = [row[:] for row in a_eq] + [a_ub[i][:] for i in active]
            rhs = b_eq[:] + [b_ub[i] for i in active]
            if len(mat) != n:
                continue
            x = solve_square(mat, rhs)
            if x is not None and feasible(x, a_ub, b_ub, a_eq, b_eq):
                candidates.append(x)
    if not candidates:
        return status_payload("infeasible", solver, "no feasible vertex found")

    ray = improving_recession_ray(sense, c, a_ub, a_eq)
    if ray is not None:
        return {
            "status": "unbounded",
            "x": [],
            "objective": None,
            "iters": len(candidates),
            "solver": solver,
            "unboundedRay": ray,
            "message": "dependency-free recession-ray fallback",
        }

    sign = 1.0 if sense == "max" else -1.0
    best = max(candidates, key=lambda x: sign * dot(c, x))
    result = {
        "status": "optimal",
        "x": best,
        "objective": dot(c, best),
        "iters": len(candidates),
        "solver": solver,
        "message": "dependency-free vertex enumeration fallback",
    }
    result.update(recover_certificate(lp, best))
    return apply_objective_offset(lp, result)


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


def ortools_glop(lp: dict) -> Optional[dict]:
    try:
        from ortools.linear_solver import pywraplp  # type: ignore
    except Exception:
        return None

    solver = pywraplp.Solver.CreateSolver("GLOP")
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
        "solver": "ortools:glop",
        "message": f"GLOP status code {status_code}",
    }
    if status == "optimal":
        result_payload.update(recover_certificate(lp, x))
    return apply_objective_offset(lp, result_payload)


def solve_external(lp: dict, method: str) -> Optional[dict]:
    normalized = method.lower().replace("_", "-")
    if normalized in ("glop", "ortools-glop", "ortools:glop"):
        return ortools_glop(lp)
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
            result = vertex_enumeration(lp)
        print(json.dumps(result, allow_nan=True))
        return 0
    except Exception as exc:
        print(json.dumps(status_payload("numerical-error", "python:lp-bridge", str(exc)), allow_nan=True))
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
