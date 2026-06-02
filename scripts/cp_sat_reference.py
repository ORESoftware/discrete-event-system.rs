#!/usr/bin/env python3
"""Reference bridge for small CP-SAT-style finite-domain models.

The bridge prefers OR-Tools CP-SAT when installed and falls back to exact
enumeration with the same JSON model contract.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from typing import Dict, List, Optional, Sequence


def objective_value(model: dict, assignment: Sequence[int]) -> Optional[int]:
    obj = model.get("objective")
    if not obj:
        return None
    return sum(int(t["coeff"]) * int(assignment[int(t["var"])]) for t in obj.get("terms", []))


def linear_bounds(model: dict, partial: Sequence[Optional[int]], terms: Sequence[dict]) -> tuple[int, int]:
    lo = hi = 0
    for term in terms:
        var = int(term["var"])
        coeff = int(term["coeff"])
        if partial[var] is not None:
            lo += coeff * int(partial[var])
            hi += coeff * int(partial[var])
            continue
        dom = model["variables"][var]["domain"]
        dmin, dmax = min(dom), max(dom)
        if coeff >= 0:
            lo += coeff * dmin
            hi += coeff * dmax
        else:
            lo += coeff * dmax
            hi += coeff * dmin
    return lo, hi


def literal_truth(partial: Sequence[Optional[int]], lit: dict) -> Optional[bool]:
    value = partial[int(lit["var"])]
    if value is None:
        return None
    return (int(value) == 1) if bool(lit.get("positive", True)) else (int(value) == 0)


def enforcement_state(partial: Sequence[Optional[int]], literals: Sequence[dict]) -> Optional[bool]:
    unknown = False
    for lit in literals:
        truth = literal_truth(partial, lit)
        if truth is False:
            return False
        if truth is None:
            unknown = True
    return None if unknown else True


def linear_partial_ok(model: dict, partial: Sequence[Optional[int]], c: dict) -> bool:
    lo, hi = linear_bounds(model, partial, c["terms"])
    rhs = int(c["rhs"])
    sense = c["sense"]
    if sense == "le" and lo > rhs:
        return False
    if sense == "ge" and hi < rhs:
        return False
    if sense == "eq" and not (lo <= rhs <= hi):
        return False
    return True


def partial_ok(model: dict, partial: Sequence[Optional[int]]) -> bool:
    for c in model.get("constraints", []):
        kind = c["kind"]
        if kind == "linear":
            if not linear_partial_ok(model, partial, c):
                return False
        elif kind == "enforced_linear":
            active = enforcement_state(partial, c["enforcement"])
            if active is True and not linear_partial_ok(model, partial, c):
                return False
        elif kind == "all_different":
            seen = set()
            for v in c["vars"]:
                value = partial[int(v)]
                if value is None:
                    continue
                if value in seen:
                    return False
                seen.add(value)
        elif kind == "bool_or":
            unknown = False
            satisfied = False
            for lit in c["literals"]:
                value = partial[int(lit["var"])]
                if value is None:
                    unknown = True
                    continue
                truth = literal_truth(partial, lit)
                if truth:
                    satisfied = True
                    break
            if not satisfied and not unknown:
                return False
        elif kind == "allowed_assignments":
            vars_ = [int(v) for v in c["vars"]]
            tuples = [[int(v) for v in row] for row in c["tuples"]]
            ok = False
            for row in tuples:
                if all(partial[var] is None or int(partial[var]) == value for var, value in zip(vars_, row)):
                    ok = True
                    break
            if not ok:
                return False
        elif kind == "element":
            index_var = int(c["index"])
            target_var = int(c["target"])
            values = [int(v) for v in c["values"]]
            index_value = partial[index_var]
            target_value = partial[target_var]
            if index_value is not None and target_value is not None:
                index = int(index_value)
                if index < 0 or index >= len(values) or values[index] != int(target_value):
                    return False
            elif index_value is not None:
                index = int(index_value)
                if index < 0 or index >= len(values):
                    return False
            elif target_value is not None:
                if not any(
                    0 <= int(index) < len(values) and values[int(index)] == int(target_value)
                    for index in model["variables"][index_var]["domain"]
                ):
                    return False
            elif not any(0 <= int(index) < len(values) for index in model["variables"][index_var]["domain"]):
                return False
        elif kind == "no_overlap":
            intervals = c["intervals"]
            for i, a in enumerate(intervals):
                start_a = partial[int(a["start"])]
                if start_a is None:
                    continue
                end_a = int(start_a) + int(a["duration"])
                for b in intervals[i + 1:]:
                    start_b = partial[int(b["start"])]
                    if start_b is None:
                        continue
                    end_b = int(start_b) + int(b["duration"])
                    if not (end_a <= int(start_b) or end_b <= int(start_a)):
                        return False
        elif kind == "cumulative":
            assigned = []
            for interval in c["intervals"]:
                start = partial[int(interval["start"])]
                if start is None:
                    continue
                assigned.append((int(start), int(start) + int(interval["duration"]), int(interval["demand"])))
            points = sorted({point for start, end, _ in assigned for point in (start, end)})
            capacity = int(c["capacity"])
            for t in points:
                load = sum(demand for start, end, demand in assigned if start <= t < end)
                if load > capacity:
                    return False
        else:
            raise ValueError(f"unknown constraint kind {kind}")
    return True


def enumerate_reference(model: dict) -> dict:
    n = len(model["variables"])
    partial: List[Optional[int]] = [None] * n
    best = None
    best_obj = None
    nodes = 0

    def better(value: int, incumbent: int) -> bool:
        sense = model.get("objective", {}).get("sense", "min")
        return value < incumbent if sense == "min" else value > incumbent

    def dfs() -> None:
        nonlocal best, best_obj, nodes
        nodes += 1
        if not partial_ok(model, partial):
            return
        try:
            var = min(
                (i for i, v in enumerate(partial) if v is None),
                key=lambda i: len(model["variables"][i]["domain"]),
            )
        except ValueError:
            full = [int(v) for v in partial]  # type: ignore[arg-type]
            obj = objective_value(model, full)
            if best is None or (obj is not None and best_obj is not None and better(obj, best_obj)):
                best = full
                best_obj = obj
            elif best is None:
                best = full
                best_obj = obj
            return
        for value in model["variables"][var]["domain"]:
            partial[var] = int(value)
            dfs()
            partial[var] = None

    dfs()
    if best is None:
        return {"status": "infeasible", "assignment": [], "objective": None, "nodes": nodes, "solver": "python:cp-enumeration"}
    return {
        "status": "optimal" if model.get("objective") else "feasible",
        "assignment": best,
        "objective": best_obj,
        "nodes": nodes,
        "solver": "python:cp-enumeration",
        "message": "dependency-free exact enumeration fallback",
    }


def ortools_reference(model: dict) -> Optional[dict]:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception:
        return None
    cp = cp_model.CpModel()
    xs = []
    for var in model["variables"]:
        dom = cp_model.Domain.FromValues([int(v) for v in var["domain"]])
        xs.append(cp.NewIntVarFromDomain(dom, var.get("name", f"x{len(xs)}")))
    for c in model.get("constraints", []):
        kind = c["kind"]
        if kind == "linear":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            if c["sense"] == "le":
                cp.Add(expr <= int(c["rhs"]))
            elif c["sense"] == "ge":
                cp.Add(expr >= int(c["rhs"]))
            else:
                cp.Add(expr == int(c["rhs"]))
        elif kind == "enforced_linear":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            if c["sense"] == "le":
                constraint = cp.Add(expr <= int(c["rhs"]))
            elif c["sense"] == "ge":
                constraint = cp.Add(expr >= int(c["rhs"]))
            else:
                constraint = cp.Add(expr == int(c["rhs"]))
            enforcement = []
            for lit in c["enforcement"]:
                x = xs[int(lit["var"])]
                enforcement.append(x if bool(lit.get("positive", True)) else x.Not())
            constraint.OnlyEnforceIf(enforcement)
        elif kind == "all_different":
            cp.AddAllDifferent([xs[int(v)] for v in c["vars"]])
        elif kind == "bool_or":
            lits = []
            for lit in c["literals"]:
                x = xs[int(lit["var"])]
                lits.append(x if bool(lit.get("positive", True)) else x.Not())
            cp.AddBoolOr(lits)
        elif kind == "allowed_assignments":
            cp.AddAllowedAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
        elif kind == "element":
            cp.AddElement(
                xs[int(c["index"])],
                [int(v) for v in c["values"]],
                xs[int(c["target"])],
            )
        elif kind == "no_overlap":
            intervals = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(
                    cp.NewFixedSizeIntervalVar(
                        xs[int(interval["start"])],
                        int(interval["duration"]),
                        name,
                    )
                )
            cp.AddNoOverlap(intervals)
        elif kind == "cumulative":
            intervals = []
            demands = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(
                    cp.NewFixedSizeIntervalVar(
                        xs[int(interval["start"])],
                        int(interval["duration"]),
                        name,
                    )
                )
                demands.append(int(interval["demand"]))
            cp.AddCumulative(intervals, demands, int(c["capacity"]))
    if model.get("objective"):
        expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in model["objective"]["terms"])
        if model["objective"].get("sense", "min") == "min":
            cp.Minimize(expr)
        else:
            cp.Maximize(expr)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(cp)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignment = [int(solver.Value(x)) for x in xs]
        return {
            "status": "optimal" if status == cp_model.OPTIMAL else "feasible",
            "assignment": assignment,
            "objective": objective_value(model, assignment),
            "nodes": int(solver.NumBranches()),
            "solver": "ortools:cp-sat",
        }
    if status == cp_model.INFEASIBLE:
        return {"status": "infeasible", "assignment": [], "objective": None, "nodes": int(solver.NumBranches()), "solver": "ortools:cp-sat"}
    return {"status": "unavailable", "assignment": [], "objective": None, "nodes": int(solver.NumBranches()), "solver": "ortools:cp-sat", "message": "CP-SAT did not prove a result"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="auto")
    args = parser.parse_args()
    model = json.load(sys.stdin)
    result = None
    if args.solver in ("auto", "ortools", "ortools-cp-sat"):
        result = ortools_reference(model)
        if args.solver != "auto" and result is None:
            result = {"status": "unavailable", "assignment": [], "objective": None, "nodes": 0, "solver": "ortools:cp-sat", "message": "ortools is not installed"}
    if result is None:
        result = enumerate_reference(model)
    print(json.dumps(result))
    return 0 if result.get("status") != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
