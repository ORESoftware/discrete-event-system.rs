#!/usr/bin/env python3
"""Reference bridge for small convex quadratic programs.

Input JSON:
  {
    "Q": [[...]], "c": [...],
    "A_ub": [[...]], "b_ub": [...],
    "A_eq": [[...]], "b_eq": [...],
    "lb": [0, null], "ub": [1, null]
  }

The bridge prefers scipy.optimize.minimize when SciPy is installed, and falls
back to dependency-free active-set enumeration for small dense QPs.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from typing import List, Optional, Sequence, Tuple


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mat_vec(a: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    return [dot(row, x) for row in a]


def objective(qp: dict, x: Sequence[float]) -> float:
    qx = mat_vec(qp["Q"], x)
    return 0.5 * dot(x, qx) + dot(qp["c"], x)


def solve_square(a: Sequence[Sequence[float]], b: Sequence[float], tol: float = 1e-10) -> Optional[List[float]]:
    n = len(b)
    aug = [list(map(float, row)) + [float(rhs)] for row, rhs in zip(a, b)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < tol:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        pv = aug[col][col]
        for j in range(col, n + 1):
            aug[col][j] /= pv
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            for j in range(col, n + 1):
                aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def normalize(qp: dict) -> dict:
    q = [list(map(float, row)) for row in qp.get("Q", qp.get("q", []))]
    c = [float(v) for v in qp["c"]]
    n = len(c)
    out = {
        "Q": q,
        "c": c,
        "A_ub": [list(map(float, row)) for row in qp.get("A_ub", qp.get("a_ub", [])) or []],
        "b_ub": [float(v) for v in qp.get("b_ub", []) or []],
        "A_eq": [list(map(float, row)) for row in qp.get("A_eq", qp.get("a_eq", [])) or []],
        "b_eq": [float(v) for v in qp.get("b_eq", []) or []],
        "lb": [0.0] * n if qp.get("lb") is None else [None if v is None else float(v) for v in qp["lb"]],
        "ub": [None] * n if qp.get("ub") is None else [None if v is None else float(v) for v in qp["ub"]],
    }
    return out


def active_items(qp: dict) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    for i in range(len(qp["A_ub"])):
        items.append(("ineq", i))
    for i, v in enumerate(qp["lb"]):
        if v is not None:
            items.append(("lb", i))
    for i, v in enumerate(qp["ub"]):
        if v is not None:
            items.append(("ub", i))
    return items


def active_row(qp: dict, item: Tuple[str, int]) -> Tuple[List[float], float]:
    kind, i = item
    n = len(qp["c"])
    if kind == "ineq":
        return qp["A_ub"][i][:], qp["b_ub"][i]
    row = [0.0] * n
    row[i] = 1.0
    if kind == "lb":
        return row, qp["lb"][i]
    return row, qp["ub"][i]


def feasible(qp: dict, x: Sequence[float], tol: float = 1e-7) -> bool:
    for i, value in enumerate(x):
        if qp["lb"][i] is not None and value < qp["lb"][i] - tol:
            return False
        if qp["ub"][i] is not None and value > qp["ub"][i] + tol:
            return False
    for row, rhs in zip(qp["A_ub"], qp["b_ub"]):
        if dot(row, x) > rhs + tol:
            return False
    for row, rhs in zip(qp["A_eq"], qp["b_eq"]):
        if abs(dot(row, x) - rhs) > tol:
            return False
    return True


def normalize_socp(raw: dict) -> dict:
    c = [float(v) for v in raw["c"]]
    n = len(c)
    return {
        "c": c,
        "A_ub": [list(map(float, row)) for row in raw.get("A_ub", raw.get("a_ub", [])) or []],
        "b_ub": [float(v) for v in raw.get("b_ub", []) or []],
        "A_eq": [list(map(float, row)) for row in raw.get("A_eq", raw.get("a_eq", [])) or []],
        "b_eq": [float(v) for v in raw.get("b_eq", []) or []],
        "lb": [None] * n if raw.get("lb") is None else [None if v is None else float(v) for v in raw["lb"]],
        "ub": [None] * n if raw.get("ub") is None else [None if v is None else float(v) for v in raw["ub"]],
        "cones": [
            {
                "A": [list(map(float, row)) for row in cone.get("A", cone.get("a", []))],
                "b": [float(v) for v in cone.get("b", [])],
                "c": [float(v) for v in cone["c"]],
                "d": float(cone["d"]),
                "name": cone.get("name"),
            }
            for cone in raw.get("cones", [])
        ],
    }


def socp_objective(p: dict, x: Sequence[float]) -> float:
    return dot(p["c"], x)


def socp_feasible(p: dict, x: Sequence[float], tol: float = 1e-7) -> bool:
    for i, value in enumerate(x):
        if p["lb"][i] is not None and value < p["lb"][i] - tol:
            return False
        if p["ub"][i] is not None and value > p["ub"][i] + tol:
            return False
    for row, rhs in zip(p["A_ub"], p["b_ub"]):
        if dot(row, x) > rhs + tol:
            return False
    for row, rhs in zip(p["A_eq"], p["b_eq"]):
        if abs(dot(row, x) - rhs) > tol:
            return False
    for cone in p["cones"]:
        ax = mat_vec(cone["A"], x)
        lhs = math.sqrt(sum((ai + bi) ** 2 for ai, bi in zip(ax, cone["b"])))
        rhs = dot(cone["c"], x) + cone["d"]
        if rhs < -tol or lhs > rhs + tol:
            return False
    return True


def socp_initial_point(p: dict, tol: float = 1e-7) -> Optional[List[float]]:
    x = []
    for lb, ub in zip(p["lb"], p["ub"]):
        if lb is not None and ub is not None:
            x.append(0.5 * (lb + ub))
        elif lb is not None and lb > 0:
            x.append(lb)
        elif ub is not None and ub < 0:
            x.append(ub)
        else:
            x.append(0.0)
    if socp_feasible(p, x, tol):
        return x
    values = []
    for lb, ub in zip(p["lb"], p["ub"]):
        vals = [0.0]
        if lb is not None:
            vals.append(lb)
        if ub is not None:
            vals.append(ub)
        if lb is not None and ub is not None:
            vals.append(0.5 * (lb + ub))
        values.append(sorted(set(vals)))
    for candidate in itertools.product(*values):
        x = [float(v) for v in candidate]
        if socp_feasible(p, x, tol):
            return x
    return None


def socp_pattern_reference(raw: dict) -> dict:
    p = normalize_socp(raw)
    x = socp_initial_point(p)
    if x is None:
        return {"status": "infeasible", "solver": "python:socp-pattern-search", "x": [], "objective": None, "iterations": 0}
    dirs = []
    n = len(p["c"])
    for i in range(n):
        plus = [0.0] * n
        plus[i] = 1.0
        dirs.append(plus)
        minus = [0.0] * n
        minus[i] = -1.0
        dirs.append(minus)
    norm = math.sqrt(sum(v * v for v in p["c"]))
    if norm > 1e-12:
        dirs.append([-v / norm for v in p["c"]])
    spans = [
        ub - lb
        for lb, ub in zip(p["lb"], p["ub"])
        if lb is not None and ub is not None and ub > lb
    ]
    step = max(1.0, 0.5 * max(spans)) if spans else 1.0
    best = x
    best_obj = socp_objective(p, best)
    iterations = 0
    tol = 1e-7
    while iterations < 20_000 and step > tol:
        iterations += 1
        improved = False
        trial_best = best
        trial_obj = best_obj
        for direction in dirs:
            cand = [xi + step * di for xi, di in zip(best, direction)]
            for i, (lb, ub) in enumerate(zip(p["lb"], p["ub"])):
                if lb is not None:
                    cand[i] = max(cand[i], lb)
                if ub is not None:
                    cand[i] = min(cand[i], ub)
            if not socp_feasible(p, cand, tol):
                continue
            obj = socp_objective(p, cand)
            if obj < trial_obj - tol:
                trial_best = cand
                trial_obj = obj
                improved = True
        if improved:
            best = trial_best
            best_obj = trial_obj
        else:
            step *= 0.5
    status = "optimal" if step <= tol else "numerical-error"
    return {
        "status": status,
        "solver": "python:socp-pattern-search",
        "x": best,
        "objective": best_obj,
        "iterations": iterations,
        "message": "dependency-free SOCP pattern-search fallback",
    }


def normalize_qcp(raw: dict) -> dict:
    c = [float(v) for v in raw["c"]]
    n = len(c)
    return {
        "Q": [list(map(float, row)) for row in raw.get("Q", raw.get("q", []))],
        "c": c,
        "A_ub": [list(map(float, row)) for row in raw.get("A_ub", raw.get("a_ub", [])) or []],
        "b_ub": [float(v) for v in raw.get("b_ub", []) or []],
        "A_eq": [list(map(float, row)) for row in raw.get("A_eq", raw.get("a_eq", [])) or []],
        "b_eq": [float(v) for v in raw.get("b_eq", []) or []],
        "lb": [None] * n if raw.get("lb") is None else [None if v is None else float(v) for v in raw["lb"]],
        "ub": [None] * n if raw.get("ub") is None else [None if v is None else float(v) for v in raw["ub"]],
        "quadratic_constraints": [
            {
                "Q": [list(map(float, row)) for row in qc.get("Q", qc.get("q", []))],
                "c": [float(v) for v in qc.get("c", [])],
                "rhs": float(qc["rhs"]),
                "name": qc.get("name"),
            }
            for qc in raw.get("quadratic_constraints", raw.get("q_constraints", []))
        ],
    }


def qcp_objective(p: dict, x: Sequence[float]) -> float:
    return objective({"Q": p["Q"], "c": p["c"]}, x)


def quadratic_constraint_value(qc: dict, x: Sequence[float]) -> float:
    qx = mat_vec(qc["Q"], x)
    return dot(x, qx) + dot(qc["c"], x)


def qcp_feasible(p: dict, x: Sequence[float], tol: float = 1e-7) -> bool:
    for i, value in enumerate(x):
        if p["lb"][i] is not None and value < p["lb"][i] - tol:
            return False
        if p["ub"][i] is not None and value > p["ub"][i] + tol:
            return False
    for row, rhs in zip(p["A_ub"], p["b_ub"]):
        if dot(row, x) > rhs + tol:
            return False
    for row, rhs in zip(p["A_eq"], p["b_eq"]):
        if abs(dot(row, x) - rhs) > tol:
            return False
    for qc in p["quadratic_constraints"]:
        if quadratic_constraint_value(qc, x) > qc["rhs"] + tol:
            return False
    return True


def qcp_initial_point(p: dict, tol: float = 1e-7) -> Optional[List[float]]:
    x = []
    for lb, ub in zip(p["lb"], p["ub"]):
        if lb is not None and ub is not None:
            x.append(0.5 * (lb + ub))
        elif lb is not None and lb > 0:
            x.append(lb)
        elif ub is not None and ub < 0:
            x.append(ub)
        else:
            x.append(0.0)
    if qcp_feasible(p, x, tol):
        return x
    values = []
    for lb, ub in zip(p["lb"], p["ub"]):
        vals = [0.0]
        if lb is not None:
            vals.append(lb)
        if ub is not None:
            vals.append(ub)
        if lb is not None and ub is not None:
            vals.append(0.5 * (lb + ub))
        values.append(sorted(set(vals)))
    for candidate in itertools.product(*values):
        x = [float(v) for v in candidate]
        if qcp_feasible(p, x, tol):
            return x
    return None


def qcp_gradient(p: dict, x: Sequence[float]) -> List[float]:
    qx = mat_vec(p["Q"], x)
    return [qi + ci for qi, ci in zip(qx, p["c"])]


def qcp_pattern_reference(raw: dict) -> dict:
    p = normalize_qcp(raw)
    x = qcp_initial_point(p)
    if x is None:
        return {"status": "infeasible", "solver": "python:qcp-pattern-search", "x": [], "objective": None, "iterations": 0}
    spans = [
        ub - lb
        for lb, ub in zip(p["lb"], p["ub"])
        if lb is not None and ub is not None and ub > lb
    ]
    step = max(1.0, 0.5 * max(spans)) if spans else 1.0
    best = x
    best_obj = qcp_objective(p, best)
    iterations = 0
    tol = 1e-7
    n = len(p["c"])
    while iterations < 20_000 and step > tol:
        iterations += 1
        dirs = []
        for i in range(n):
            plus = [0.0] * n
            plus[i] = 1.0
            dirs.append(plus)
            minus = [0.0] * n
            minus[i] = -1.0
            dirs.append(minus)
        grad = qcp_gradient(p, best)
        norm = math.sqrt(sum(v * v for v in grad))
        if norm > 1e-12:
            dirs.append([-v / norm for v in grad])

        improved = False
        trial_best = best
        trial_obj = best_obj
        for direction in dirs:
            cand = [xi + step * di for xi, di in zip(best, direction)]
            for i, (lb, ub) in enumerate(zip(p["lb"], p["ub"])):
                if lb is not None:
                    cand[i] = max(cand[i], lb)
                if ub is not None:
                    cand[i] = min(cand[i], ub)
            if not qcp_feasible(p, cand, tol):
                continue
            obj = qcp_objective(p, cand)
            if obj < trial_obj - tol:
                trial_best = cand
                trial_obj = obj
                improved = True
        if improved:
            best = trial_best
            best_obj = trial_obj
        else:
            step *= 0.5
    status = "optimal" if step <= tol else "numerical-error"
    return {
        "status": status,
        "solver": "python:qcp-pattern-search",
        "x": best,
        "objective": best_obj,
        "iterations": iterations,
        "message": "dependency-free QCP pattern-search fallback",
    }


def solve_kkt(qp: dict, active: Sequence[Tuple[str, int]]) -> Optional[List[float]]:
    n = len(qp["c"])
    eq_rows = [row[:] for row in qp["A_eq"]]
    eq_rhs = qp["b_eq"][:]
    for item in active:
        row, rhs = active_row(qp, item)
        eq_rows.append(row)
        eq_rhs.append(rhs)
    m = len(eq_rows)
    dim = n + m
    kkt = [[0.0] * dim for _ in range(dim)]
    rhs = [0.0] * dim
    for i in range(n):
        for j in range(n):
            kkt[i][j] = qp["Q"][i][j]
        rhs[i] = -qp["c"][i]
    for r, row in enumerate(eq_rows):
        for j in range(n):
            kkt[j][n + r] = row[j]
            kkt[n + r][j] = row[j]
        rhs[n + r] = eq_rhs[r]
    sol = solve_square(kkt, rhs)
    if sol is None:
        return None
    return sol[:n]


def enumerate_active_sets(qp: dict) -> dict:
    qp = normalize(qp)
    items = active_items(qp)
    n = len(qp["c"])
    best_x = None
    best_obj = math.inf
    iterations = 0
    for r in range(min(n, len(items)) + 1):
        for active in itertools.combinations(items, r):
            iterations += 1
            x = solve_kkt(qp, active)
            if x is None or not feasible(qp, x):
                continue
            obj = objective(qp, x)
            if obj < best_obj - 1e-8:
                best_obj = obj
                best_x = x
    if best_x is None:
        return {"status": "infeasible", "solver": "python:qp-active-set", "x": [], "objective": None, "iterations": iterations}
    return {
        "status": "optimal",
        "solver": "python:qp-active-set",
        "x": best_x,
        "objective": best_obj,
        "iterations": iterations,
        "message": "dependency-free active-set enumeration fallback",
    }


def scipy_reference(qp_raw: dict) -> Optional[dict]:
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import Bounds, LinearConstraint, minimize  # type: ignore
    except Exception:
        return None
    qp = normalize(qp_raw)
    q = np.array(qp["Q"], dtype=float)
    c = np.array(qp["c"], dtype=float)
    n = len(c)
    bounds = Bounds(
        [-np.inf if v is None else v for v in qp["lb"]],
        [np.inf if v is None else v for v in qp["ub"]],
    )
    constraints = []
    if qp["A_ub"]:
        constraints.append(LinearConstraint(np.array(qp["A_ub"], dtype=float), -np.inf, np.array(qp["b_ub"], dtype=float)))
    if qp["A_eq"]:
        constraints.append(LinearConstraint(np.array(qp["A_eq"], dtype=float), np.array(qp["b_eq"], dtype=float), np.array(qp["b_eq"], dtype=float)))

    x0 = np.zeros(n)
    for i in range(n):
        if qp["lb"][i] is not None and x0[i] < qp["lb"][i]:
            x0[i] = qp["lb"][i]
        if qp["ub"][i] is not None and x0[i] > qp["ub"][i]:
            x0[i] = qp["ub"][i]

    def fun(x):
        return float(0.5 * x @ q @ x + c @ x)

    def jac(x):
        return q @ x + c

    result = minimize(fun, x0, jac=jac, bounds=bounds, constraints=constraints, method="SLSQP", options={"ftol": 1e-10, "maxiter": 500})
    if not result.success:
        return {"status": "numerical-error", "solver": "scipy:SLSQP", "x": [], "objective": None, "message": str(result.message)}
    x = [float(v) for v in result.x]
    return {"status": "optimal", "solver": "scipy:SLSQP", "x": x, "objective": objective(qp, x), "iterations": int(result.nit), "message": str(result.message)}


def scipy_socp_reference(raw: dict) -> Optional[dict]:
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize  # type: ignore
    except Exception:
        return None
    p = normalize_socp(raw)
    n = len(p["c"])
    bounds = Bounds(
        [-np.inf if v is None else v for v in p["lb"]],
        [np.inf if v is None else v for v in p["ub"]],
    )
    constraints = []
    if p["A_ub"]:
        constraints.append(LinearConstraint(np.array(p["A_ub"], dtype=float), -np.inf, np.array(p["b_ub"], dtype=float)))
    if p["A_eq"]:
        constraints.append(LinearConstraint(np.array(p["A_eq"], dtype=float), np.array(p["b_eq"], dtype=float), np.array(p["b_eq"], dtype=float)))
    for cone in p["cones"]:
        a = np.array(cone["A"], dtype=float)
        b = np.array(cone["b"], dtype=float)
        c = np.array(cone["c"], dtype=float)
        d = float(cone["d"])

        def fun(x, a=a, b=b, c=c, d=d):
            return float(c @ x + d - np.linalg.norm(a @ x + b))

        constraints.append(NonlinearConstraint(fun, 0.0, np.inf))

    x0 = socp_initial_point(p) or [0.0] * n

    def fun(x):
        return float(np.array(p["c"], dtype=float) @ x)

    def jac(_x):
        return np.array(p["c"], dtype=float)

    result = minimize(fun, np.array(x0, dtype=float), jac=jac, bounds=bounds, constraints=constraints, method="SLSQP", options={"ftol": 1e-10, "maxiter": 500})
    if not result.success:
        return {"status": "numerical-error", "solver": "scipy:SLSQP-socp", "x": [], "objective": None, "message": str(result.message)}
    x = [float(v) for v in result.x]
    return {"status": "optimal", "solver": "scipy:SLSQP-socp", "x": x, "objective": socp_objective(p, x), "iterations": int(result.nit), "message": str(result.message)}


def scipy_qcp_reference(raw: dict) -> Optional[dict]:
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, minimize  # type: ignore
    except Exception:
        return None
    p = normalize_qcp(raw)
    q = np.array(p["Q"], dtype=float)
    c = np.array(p["c"], dtype=float)
    bounds = Bounds(
        [-np.inf if v is None else v for v in p["lb"]],
        [np.inf if v is None else v for v in p["ub"]],
    )
    constraints = []
    if p["A_ub"]:
        constraints.append(LinearConstraint(np.array(p["A_ub"], dtype=float), -np.inf, np.array(p["b_ub"], dtype=float)))
    if p["A_eq"]:
        constraints.append(LinearConstraint(np.array(p["A_eq"], dtype=float), np.array(p["b_eq"], dtype=float), np.array(p["b_eq"], dtype=float)))
    for qc in p["quadratic_constraints"]:
        q_con = np.array(qc["Q"], dtype=float)
        c_con = np.array(qc["c"], dtype=float)
        rhs = float(qc["rhs"])

        def fun(x, q_con=q_con, c_con=c_con, rhs=rhs):
            return float(rhs - x @ q_con @ x - c_con @ x)

        constraints.append(NonlinearConstraint(fun, 0.0, np.inf))
    x0 = qcp_initial_point(p) or [0.0] * len(c)

    def fun(x):
        return float(0.5 * x @ q @ x + c @ x)

    def jac(x):
        return q @ x + c

    result = minimize(fun, np.array(x0, dtype=float), jac=jac, bounds=bounds, constraints=constraints, method="SLSQP", options={"ftol": 1e-10, "maxiter": 500})
    if not result.success:
        return {"status": "numerical-error", "solver": "scipy:SLSQP-qcp", "x": [], "objective": None, "message": str(result.message)}
    x = [float(v) for v in result.x]
    return {"status": "optimal", "solver": "scipy:SLSQP-qcp", "x": x, "objective": qcp_objective(p, x), "iterations": int(result.nit), "message": str(result.message)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="auto")
    args = parser.parse_args()
    qp = json.load(sys.stdin)
    result = None
    if qp.get("cones"):
        if args.solver in ("auto", "scipy", "scipy-slsqp"):
            result = scipy_socp_reference(qp)
            if args.solver != "auto" and result is None:
                result = {"status": "unavailable", "solver": "scipy:SLSQP-socp", "x": [], "objective": None, "message": "scipy is not installed"}
        if result is None:
            result = socp_pattern_reference(qp)
        print(json.dumps(result))
        return 0 if result.get("status") != "unavailable" else 2
    if qp.get("quadratic_constraints") or qp.get("q_constraints"):
        if args.solver in ("auto", "scipy", "scipy-slsqp"):
            result = scipy_qcp_reference(qp)
            if args.solver != "auto" and result is None:
                result = {"status": "unavailable", "solver": "scipy:SLSQP-qcp", "x": [], "objective": None, "message": "scipy is not installed"}
        if result is None:
            result = qcp_pattern_reference(qp)
        print(json.dumps(result))
        return 0 if result.get("status") != "unavailable" else 2
    if args.solver in ("auto", "scipy", "scipy-slsqp"):
        result = scipy_reference(qp)
        if args.solver != "auto" and result is None:
            result = {"status": "unavailable", "solver": "scipy:SLSQP", "x": [], "objective": None, "message": "scipy is not installed"}
    if result is None:
        result = enumerate_active_sets(qp)
    print(json.dumps(result))
    return 0 if result.get("status") != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
