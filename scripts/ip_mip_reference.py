#!/usr/bin/env python3
"""External IP/MIP reference solver for small Rust validation cases.

The preferred external engines are open-source packages when present:
OR-Tools CP-SAT or scipy.optimize.milp. The dependency-free fallback is exact
bounded enumeration of integer variables, solving any remaining continuous
subproblem with the vertex-enumeration LP reference.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

from lp_solve import dot, vertex_enumeration


def payload(status: str, solver: str, x=None, objective=None, message="", enumerated=0, **extra) -> dict:
    result = {
        "status": status,
        "solver": solver,
        "x": x,
        "objective": objective,
        "message": message,
        "enumerated": enumerated,
    }
    result.update(extra)
    return {"result": result}


def load_problem(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def bounds(p: dict) -> Tuple[List[float], List[Optional[float]]]:
    n = len(p["c"])
    lb = p.get("lb")
    ub = p.get("ub")
    lbs = [0.0] * n if lb is None else [0.0 if v is None else float(v) for v in lb]
    ubs = [None] * n if ub is None else [None if v is None else float(v) for v in ub]
    return lbs, ubs


def objective(p: dict, x: Sequence[float]) -> float:
    return dot([float(v) for v in p["c"]], x)


def objective_from_coefficients(c: Sequence[float], x: Sequence[float]) -> float:
    return dot([float(v) for v in c], x)


def max_lhs_over_bounds(p: dict, row: Sequence[float], rhs: float, name: str) -> float:
    lbs, ubs = bounds(p)
    max_lhs = 0.0
    for j, coeff_raw in enumerate(row):
        coeff = float(coeff_raw)
        if coeff > 0.0:
            upper = ubs[j]
            if upper is None or not math.isfinite(float(upper)):
                raise ValueError(f"indicator {name} needs a finite upper bound for variable x{j}")
            max_lhs += coeff * float(upper)
        elif coeff < 0.0:
            max_lhs += coeff * lbs[j]
    return max(0.0, max_lhs - rhs)


def append_indicator_le_row(p: dict, rows: list, rhs_values: list, names: list, indicator: dict, row: list[float], rhs: float, suffix: str | None) -> None:
    binary_var = int(indicator["binary_var"])
    name = indicator.get("name", f"indicator_{len(rows)}")
    m = max_lhs_over_bounds(p, row, rhs, name)
    compiled = list(row)
    if bool(indicator.get("active_value", True)):
        compiled[binary_var] += m
        compiled_rhs = rhs + m
    else:
        compiled[binary_var] -= m
        compiled_rhs = rhs
    rows.append(compiled)
    rhs_values.append(compiled_rhs)
    row_name = name if suffix is None else f"{name}_{suffix}"
    names.append(row_name)


def expand_indicators(p: dict) -> dict:
    indicators = p.get("indicators") or []
    if not indicators:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])
    n = len(p["c"])
    _, ubs = bounds(p)
    for idx, indicator in enumerate(indicators):
        binary_var = int(indicator["binary_var"])
        if binary_var < 0 or binary_var >= n:
            raise ValueError(f"indicator {idx} binary_var out of range")
        if not bool(p["integer_vars"][binary_var]):
            raise ValueError(f"indicator {idx} trigger variable must be integer/binary")
        ub = ubs[binary_var]
        if ub is None or not math.isfinite(float(ub)) or float(ub) > 1.0 + 1e-9:
            raise ValueError(f"indicator {idx} trigger variable must have finite binary upper bound <= 1")
        row = [float(v) for v in indicator["coefs"]]
        if len(row) != n:
            raise ValueError(f"indicator {idx} coefficient length does not match variable count")
        rhs = float(indicator["rhs"])
        sense = indicator.get("sense", "le")
        if sense == "le":
            append_indicator_le_row(p, rows, rhs_values, names, indicator, row, rhs, None)
        elif sense == "ge":
            append_indicator_le_row(p, rows, rhs_values, names, indicator, [-v for v in row], -rhs, None)
        elif sense == "eq":
            append_indicator_le_row(p, rows, rhs_values, names, indicator, row, rhs, "le")
            append_indicator_le_row(p, rows, rhs_values, names, indicator, [-v for v in row], -rhs, "ge")
        else:
            raise ValueError(f"indicator {idx} has unknown sense {sense}")
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["con_names"] = names
    expanded.pop("indicators", None)
    return expanded


def sos_ordered_vars(sos: dict) -> list[int]:
    vars_ = [int(v) for v in sos["vars"]]
    weights = sos.get("weights")
    if weights is None:
        return vars_
    if len(weights) != len(vars_):
        raise ValueError("sos weight length does not match variable count")
    pairs = sorted((float(w), v) for w, v in zip(weights, vars_))
    for (wa, _), (wb, _) in zip(pairs, pairs[1:]):
        if abs(wa - wb) <= 1e-12:
            raise ValueError("sos weights must be unique")
    return [v for _, v in pairs]


def add_binary_helper(rows: list, c: list, integer_vars: list, ubs: list, var_names: list, name: str) -> int:
    for row in rows:
        row.append(0.0)
    idx = len(c)
    c.append(0.0)
    integer_vars.append(True)
    ubs.append(1.0)
    var_names.append(name)
    return idx


def add_continuous_helper(rows: list, c: list, integer_vars: list, ubs: list, var_names: list, name: str, upper: float) -> int:
    for row in rows:
        row.append(0.0)
    idx = len(c)
    c.append(0.0)
    integer_vars.append(False)
    ubs.append(float(upper))
    var_names.append(name)
    return idx


def add_compiled_row(rows: list, rhs_values: list, names: list, row: list[float], rhs: float, name: str) -> None:
    rows.append(row)
    rhs_values.append(rhs)
    names.append(name)


def add_compiled_equality(rows: list, rhs_values: list, names: list, row: list[float], rhs: float, name: str) -> None:
    add_compiled_row(rows, rhs_values, names, list(row), rhs, f"{name}_le")
    add_compiled_row(rows, rhs_values, names, [-v for v in row], -rhs, f"{name}_ge")


def expand_linear_constraints(p: dict) -> dict:
    constraints = p.get("linear_constraints") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])
    n = len(p["c"])

    for idx, constraint in enumerate(constraints):
        row = [float(v) for v in constraint["coefs"]]
        if len(row) != n:
            raise ValueError(f"linear constraint {idx} coefficient length does not match variable count")
        if any(not math.isfinite(v) for v in row):
            raise ValueError(f"linear constraint {idx} coefficients must be finite")
        lower = constraint.get("lower")
        upper = constraint.get("upper")
        lower = None if lower is None else float(lower)
        upper = None if upper is None else float(upper)
        if lower is None and upper is None:
            raise ValueError(f"linear constraint {idx} needs at least one bound")
        if lower is not None and not math.isfinite(lower):
            raise ValueError(f"linear constraint {idx} lower bound must be finite")
        if upper is not None and not math.isfinite(upper):
            raise ValueError(f"linear constraint {idx} upper bound must be finite")
        if lower is not None and upper is not None and lower > upper + 1e-9:
            raise ValueError(f"linear constraint {idx} lower bound exceeds upper bound")
        name = constraint.get("name", f"linear_row_{idx}")
        if upper is not None:
            add_compiled_row(rows, rhs_values, names, list(row), upper, f"{name}_upper")
        if lower is not None:
            add_compiled_row(rows, rhs_values, names, [-v for v in row], -lower, f"{name}_lower")

    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["con_names"] = names
    expanded.pop("linear_constraints", None)
    return expanded


def finite_sos_ub(ubs: Sequence[Optional[float]], var: int, idx: int) -> float:
    if var < 0 or var >= len(ubs):
        raise ValueError(f"sos {idx} variable {var} out of range")
    upper = ubs[var]
    if upper is None or not math.isfinite(float(upper)):
        raise ValueError(f"sos {idx} variable x{var} needs a finite upper bound")
    if float(upper) < 0.0:
        raise ValueError(f"sos {idx} variable x{var} has a negative upper bound")
    return float(upper)


def expand_sos(p: dict) -> dict:
    sets = p.get("sos") or []
    if not sets:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    if any(abs(lb) > 1e-12 for lb in lbs):
        raise ValueError("sos bridge currently expects non-negative variables")
    ubs = [None if ub is None else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    for idx, sos in enumerate(sets):
        ordered = sos_ordered_vars(sos)
        if not ordered:
            raise ValueError(f"sos {idx} has no variables")
        if len(set(ordered)) != len(ordered):
            raise ValueError(f"sos {idx} contains duplicate variables")
        kind = sos.get("kind", "sos1")
        name = sos.get("name", kind)
        if kind == "sos1":
            selectors = []
            for pos, var in enumerate(ordered):
                upper = finite_sos_ub(ubs, var, idx)
                y = add_binary_helper(rows, c, integer_vars, ubs, var_names, f"{name}_sel_{pos}")
                selectors.append(y)
                row = [0.0] * len(c)
                row[var] = 1.0
                row[y] = -upper
                add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_link_{pos}")
            row = [0.0] * len(c)
            for y in selectors:
                row[y] = 1.0
            add_compiled_row(rows, rhs_values, row_names, row, 1.0, f"{name}_at_most_one")
        elif kind == "sos2":
            for var in ordered:
                finite_sos_ub(ubs, var, idx)
            if len(ordered) <= 2:
                continue
            segments = [
                add_binary_helper(rows, c, integer_vars, ubs, var_names, f"{name}_seg_{pos}")
                for pos in range(len(ordered) - 1)
            ]
            row = [0.0] * len(c)
            for segment in segments:
                row[segment] = 1.0
            add_compiled_row(rows, rhs_values, row_names, row, 1.0, f"{name}_one_segment")
            for pos, var in enumerate(ordered):
                upper = finite_sos_ub(ubs, var, idx)
                row = [0.0] * len(c)
                row[var] = 1.0
                if pos > 0:
                    row[segments[pos - 1]] -= upper
                if pos + 1 < len(ordered):
                    row[segments[pos]] -= upper
                add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_link_{pos}")
        else:
            raise ValueError(f"sos {idx} has unknown kind {kind}")
    expanded["c"] = c
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["integer_vars"] = integer_vars
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded.pop("sos", None)
    return expanded


def expand_pwl(p: dict) -> dict:
    constraints = p.get("pwl") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    if any(abs(lb) > 1e-12 for lb in lbs):
        raise ValueError("pwl bridge currently expects non-negative variables")
    ubs = [None if ub is None else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])
    sos_sets = list(p.get("sos") or [])

    for idx, pwl in enumerate(constraints):
        x_var = int(pwl["x_var"])
        y_var = int(pwl["y_var"])
        if x_var < 0 or x_var >= len(c):
            raise ValueError(f"pwl {idx} x_var out of range")
        if y_var < 0 or y_var >= len(c):
            raise ValueError(f"pwl {idx} y_var out of range")
        if x_var == y_var:
            raise ValueError(f"pwl {idx} x_var and y_var must be distinct")
        if integer_vars[x_var] or integer_vars[y_var]:
            raise ValueError(f"pwl {idx} x_var and y_var must be continuous")
        points = [(float(pt["x"]), float(pt["y"])) for pt in pwl["points"]]
        if len(points) < 2:
            raise ValueError(f"pwl {idx} needs at least two breakpoints")
        for pos, (px, py) in enumerate(points):
            if not math.isfinite(px) or not math.isfinite(py):
                raise ValueError(f"pwl {idx} breakpoint {pos} must be finite")
            if px < -1e-12 or py < -1e-12:
                raise ValueError(f"pwl {idx} breakpoint {pos} must be non-negative")
            if pos > 0 and px <= points[pos - 1][0] + 1e-12:
                raise ValueError(f"pwl {idx} breakpoint x values must be strictly increasing")
        name = pwl.get("name", f"pwl_{idx}")
        lambdas = [
            add_continuous_helper(rows, c, integer_vars, ubs, var_names, f"{name}_lambda_{pos}", 1.0)
            for pos, _ in enumerate(points)
        ]

        row = [0.0] * len(c)
        for lam in lambdas:
            row[lam] = 1.0
        add_compiled_equality(rows, rhs_values, row_names, row, 1.0, f"{name}_lambda_sum")

        row = [0.0] * len(c)
        row[x_var] = 1.0
        for lam, (px, _) in zip(lambdas, points):
            row[lam] -= px
        add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_x_interp")

        row = [0.0] * len(c)
        row[y_var] = 1.0
        for lam, (_, py) in zip(lambdas, points):
            row[lam] -= py
        add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_y_interp")

        sos_sets.append(
            {
                "kind": "sos2",
                "vars": lambdas,
                "weights": [px for px, _ in points],
                "name": f"{name}_sos2",
            }
        )

    expanded["c"] = c
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["integer_vars"] = integer_vars
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded["sos"] = sos_sets
    expanded.pop("pwl", None)
    return expanded


def expand_semi_variables(p: dict) -> dict:
    semis = p.get("semi_variables") or []
    if not semis:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    for idx, semi in enumerate(semis):
        var = int(semi["var"])
        if var < 0 or var >= len(c):
            raise ValueError(f"semi variable {idx} index out of range")
        if abs(lbs[var]) > 1e-12:
            raise ValueError(f"semi variable {idx} expects ordinary lower bound 0")
        lower = float(semi["lower"])
        if not math.isfinite(lower) or lower <= 0.0:
            raise ValueError(f"semi variable {idx} lower bound must be finite and positive")
        upper = ubs[var]
        if upper is None or not math.isfinite(float(upper)):
            raise ValueError(f"semi variable {idx} needs a finite upper bound")
        upper = float(upper)
        if upper + 1e-9 < lower:
            raise ValueError(f"semi variable {idx} lower bound exceeds upper bound")
        kind = semi.get("kind", "semi_continuous")
        if kind == "semi_integer":
            integer_vars[var] = True
        elif kind == "semi_continuous":
            integer_vars[var] = False
        else:
            raise ValueError(f"semi variable {idx} has unknown kind {kind}")
        name = semi.get("name", f"semi_{var}")
        y = add_binary_helper(rows, c, integer_vars, ubs, var_names, f"{name}_active")

        row = [0.0] * len(c)
        row[var] = 1.0
        row[y] = -upper
        add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_upper_link")

        row = [0.0] * len(c)
        row[var] = -1.0
        row[y] = lower
        add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_lower_link")

    expanded["c"] = c
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["integer_vars"] = integer_vars
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded.pop("semi_variables", None)
    return expanded


def feasible(p: dict, x: Sequence[float], tol: float = 1e-7) -> bool:
    lbs, ubs = bounds(p)
    for j, v in enumerate(x):
        if v < lbs[j] - tol:
            return False
        if ubs[j] is not None and v > ubs[j] + tol:
            return False
        if p["integer_vars"][j] and abs(v - round(v)) > tol:
            return False
    for row, rhs in zip(p.get("a", []), p.get("b", [])):
        if dot(row, x) > rhs + tol:
            return False
    return True


def solve_continuous_remainder(p: dict, fixed: dict[int, float]) -> Optional[Tuple[List[float], float]]:
    n = len(p["c"])
    cont = [j for j in range(n) if j not in fixed]
    x = [0.0] * n
    for j, v in fixed.items():
        x[j] = v
    if not cont:
        return (x, objective(p, x)) if feasible(p, x) else None

    c_cont = [float(p["c"][j]) for j in cont]
    a_ub = []
    b_ub = []
    for row, rhs in zip(p.get("a", []), p.get("b", [])):
        adjusted = float(rhs) - sum(float(row[j]) * v for j, v in fixed.items())
        a_ub.append([float(row[j]) for j in cont])
        b_ub.append(adjusted)
    lbs, ubs = bounds(p)
    lp = {
        "sense": p.get("sense", "max"),
        "c": c_cont,
        "A_ub": a_ub,
        "b_ub": b_ub,
        "lb": [lbs[j] for j in cont],
        "ub": [ubs[j] for j in cont],
    }
    result = vertex_enumeration(lp)
    if result["status"] != "optimal":
        return None
    for k, j in enumerate(cont):
        x[j] = float(result["x"][k])
    if not feasible(p, x):
        return None
    return x, objective(p, x)


def brute_force(p: dict, max_enumerations: int) -> dict:
    int_vars = [j for j, is_int in enumerate(p["integer_vars"]) if is_int]
    lbs, ubs = bounds(p)
    domains = []
    for j in int_vars:
        if ubs[j] is None:
            return payload("unavailable", "python:bounded-enumeration", message=f"x{j} has no finite upper bound")
        lo = math.ceil(lbs[j])
        hi = math.floor(float(ubs[j]))
        if hi < lo:
            return payload("infeasible", "python:bounded-enumeration", enumerated=0)
        domains.append(range(lo, hi + 1))

    best_x = None
    best_obj = -math.inf if p.get("sense", "max") == "max" else math.inf
    enumerated = 0
    for values in itertools.product(*domains):
        enumerated += 1
        if enumerated > max_enumerations:
            return payload("unavailable", "python:bounded-enumeration", message="enumeration cap reached", enumerated=enumerated)
        fixed = {j: float(v) for j, v in zip(int_vars, values)}
        candidate = solve_continuous_remainder(p, fixed)
        if candidate is None:
            continue
        x, obj = candidate
        if p.get("sense", "max") == "max":
            better = obj > best_obj + 1e-9
        else:
            better = obj < best_obj - 1e-9
        if best_x is None or better:
            best_x, best_obj = x, obj
    if best_x is None:
        return payload("infeasible", "python:bounded-enumeration", enumerated=enumerated)
    return payload("optimal", "python:bounded-enumeration", best_x, best_obj, "exact bounded enumeration", enumerated)


def try_ortools_cp_sat(p: dict) -> Optional[dict]:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return None
    if not all(p["integer_vars"]):
        return None
    scale = 1000
    try:
        c = [int(round(float(v) * scale)) for v in p["c"]]
        a = [[int(round(float(v) * scale)) for v in row] for row in p.get("a", [])]
        b = [int(round(float(v) * scale)) for v in p.get("b", [])]
        lbs, ubs = bounds(p)
        if any(ub is None for ub in ubs):
            return None
        model = cp_model.CpModel()
        xs = [
            model.NewIntVar(int(math.ceil(lbs[j])), int(math.floor(float(ubs[j]))), f"x{j}")
            for j in range(len(c))
        ]
        for row, rhs in zip(a, b):
            model.Add(sum(row[j] * xs[j] for j in range(len(c))) <= rhs)
        expr = sum(c[j] * xs[j] for j in range(len(c)))
        if p.get("sense", "max") == "max":
            model.Maximize(expr)
        else:
            model.Minimize(expr)
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = 10.0
        status = solver.Solve(model)
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            x = [float(solver.Value(v)) for v in xs]
            return payload("optimal" if status == cp_model.OPTIMAL else "feasible", "ortools:cp-sat", x, objective(p, x))
        if status == cp_model.INFEASIBLE:
            return payload("infeasible", "ortools:cp-sat")
        return payload("unavailable", "ortools:cp-sat", message="CP-SAT did not prove a solution")
    except Exception as exc:
        return payload("unavailable", "ortools:cp-sat", message=str(exc))


def try_scipy_milp(p: dict) -> Optional[dict]:
    try:
        import numpy as np  # type: ignore
        from scipy.optimize import Bounds, LinearConstraint, milp  # type: ignore
    except Exception:
        return None
    try:
        c = np.array(p["c"], dtype=float)
        if p.get("sense", "max") == "max":
            c = -c
        a = np.array(p.get("a", []), dtype=float)
        b = np.array(p.get("b", []), dtype=float)
        lbs, ubs = bounds(p)
        upper = [np.inf if u is None else float(u) for u in ubs]
        constraints = []
        if len(a):
            constraints.append(LinearConstraint(a, -np.inf, b))
        result = milp(
            c=c,
            integrality=np.array([1 if v else 0 for v in p["integer_vars"]], dtype=int),
            bounds=Bounds(lbs, upper),
            constraints=constraints,
        )
        if result.success and result.x is not None:
            x = [float(v) for v in result.x]
            return payload("optimal", "scipy:milp", x, objective(p, x), str(result.message))
        if int(result.status) == 2:
            return payload("infeasible", "scipy:milp", message=str(result.message))
        return payload("unavailable", "scipy:milp", message=str(result.message))
    except Exception as exc:
        return payload("unavailable", "scipy:milp", message=str(exc))


def expand_source_features(p: dict) -> dict:
    p = expand_linear_constraints(p)
    p = expand_indicators(p)
    p = expand_pwl(p)
    p = expand_sos(p)
    p = expand_semi_variables(p)
    return p


def solve_expanded(p: dict, solver: str, max_enumerations: int) -> dict:
    if solver in ("auto", "ortools", "ortools-cp-sat"):
        r = try_ortools_cp_sat(p)
        if r and r["result"]["status"] in ("optimal", "feasible", "infeasible"):
            return r
        if solver in ("ortools", "ortools-cp-sat"):
            return r or payload("unavailable", "ortools:cp-sat", message="ortools is not installed")
    if solver in ("auto", "scipy", "scipy-milp"):
        r = try_scipy_milp(p)
        if r and r["result"]["status"] in ("optimal", "infeasible"):
            return r
        if solver in ("scipy", "scipy-milp"):
            return r or payload("unavailable", "scipy:milp", message="scipy is not installed")
    if solver in ("auto", "brute-force", "enumeration"):
        return brute_force(p, max_enumerations)
    return payload("unavailable", solver, message=f"unknown or unavailable solver '{solver}'")


def solve_multi_objective(p: dict, objectives: list[dict], solver: str, max_enumerations: int) -> dict:
    if not objectives:
        return payload("unavailable", "multi-objective", message="multi_objectives must be non-empty")
    p = dict(p)
    p.pop("multi_objectives", None)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])
    p["a"] = rows
    p["b"] = rhs_values
    p["con_names"] = row_names
    stage_results = []

    for idx, objective_spec in enumerate(objectives):
        coeffs = [float(v) for v in objective_spec["c"]]
        if len(coeffs) < len(p["c"]):
            coeffs.extend([0.0] * (len(p["c"]) - len(coeffs)))
        if len(coeffs) != len(p["c"]):
            return payload(
                "unavailable",
                "multi-objective",
                message=f"objective {idx} coefficient length does not match variable count",
            )
        p["c"] = coeffs
        p["sense"] = objective_spec.get("sense", "max")
        result = solve_expanded(p, solver, max_enumerations)
        stage_results.append(result["result"])
        if result["result"]["status"] != "optimal":
            return payload(
                result["result"]["status"],
                result["result"]["solver"],
                x=result["result"].get("x"),
                objective=result["result"].get("objective"),
                message=result["result"].get("message", ""),
                enumerated=result["result"].get("enumerated", 0),
                objective_values=[],
                stages=stage_results,
            )
        x = result["result"]["x"]
        optimum = objective_from_coefficients(coeffs, x)
        name = objective_spec.get("name", f"multi_objective_{idx}")
        add_compiled_equality(rows, rhs_values, row_names, coeffs, optimum, name)

    final_x = stage_results[-1]["x"]
    values = [
        objective_from_coefficients(
            ([float(v) for v in objective_spec["c"]] + [0.0] * len(p["c"]))[: len(p["c"])],
            final_x,
        )
        for objective_spec in objectives
    ]
    return payload(
        "optimal",
        "python:lexicographic-multi-objective",
        final_x,
        values[-1] if values else None,
        "sequential lexicographic optimization",
        sum(int(stage.get("enumerated") or 0) for stage in stage_results),
        objective_values=values,
        stages=stage_results,
    )


def solve(p: dict, solver: str, max_enumerations: int) -> dict:
    try:
        p = expand_source_features(p)
    except Exception as exc:
        return payload("unavailable", "source-linearization", message=str(exc))
    objectives = p.get("multi_objectives") or []
    if objectives:
        return solve_multi_objective(p, objectives, solver, max_enumerations)
    return solve_expanded(p, solver, max_enumerations)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--max-enumerations", type=int, default=1_000_000)
    args = parser.parse_args()
    p = load_problem(args.problem)
    result = solve(p, args.solver, args.max_enumerations)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)
        f.write("\n")
    print(json.dumps(result["result"], allow_nan=True))
    return 0 if result["result"]["status"] != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
