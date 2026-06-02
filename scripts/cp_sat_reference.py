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


def product_range(bounds: Sequence[tuple[int, int]]) -> tuple[int, int]:
    lo = 1
    hi = 1
    for next_lo, next_hi in bounds:
        candidates = [
            lo * next_lo,
            lo * next_hi,
            hi * next_lo,
            hi * next_hi,
        ]
        lo = min(candidates)
        hi = max(candidates)
    return lo, hi


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
        elif kind == "bool_and":
            for lit in c["literals"]:
                truth = literal_truth(partial, lit)
                if truth is False:
                    return False
        elif kind == "bool_xor":
            true_count = 0
            unknown = False
            for lit in c["literals"]:
                truth = literal_truth(partial, lit)
                if truth is True:
                    true_count += 1
                elif truth is None:
                    unknown = True
            if not unknown and true_count % 2 == 0:
                return False
        elif kind == "at_most_one":
            true_count = sum(1 for lit in c["literals"] if literal_truth(partial, lit) is True)
            if true_count > 1:
                return False
        elif kind == "exactly_one":
            true_count = 0
            unknown = False
            for lit in c["literals"]:
                truth = literal_truth(partial, lit)
                if truth is True:
                    true_count += 1
                elif truth is None:
                    unknown = True
            if true_count > 1 or (true_count == 0 and not unknown):
                return False
        elif kind == "implication":
            antecedent = literal_truth(partial, c["antecedent"])
            consequent = literal_truth(partial, c["consequent"])
            if antecedent is True and consequent is False:
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
        elif kind == "forbidden_assignments":
            vars_ = [int(v) for v in c["vars"]]
            tuples = [[int(v) for v in row] for row in c["tuples"]]
            for row in tuples:
                if all(partial[var] is not None and int(partial[var]) == value for var, value in zip(vars_, row)):
                    return False
        elif kind == "inverse":
            direct = [int(v) for v in c["direct"]]
            inverse = [int(v) for v in c["inverse"]]
            n = len(direct)
            seen_direct = set()
            for i, var in enumerate(direct):
                value = partial[var]
                if value is None:
                    continue
                j = int(value)
                if j < 0 or j >= n or j in seen_direct:
                    return False
                seen_direct.add(j)
                inverse_value = partial[inverse[j]]
                if inverse_value is not None and int(inverse_value) != i:
                    return False
            seen_inverse = set()
            for j, var in enumerate(inverse):
                value = partial[var]
                if value is None:
                    continue
                i = int(value)
                if i < 0 or i >= n or i in seen_inverse:
                    return False
                seen_inverse.add(i)
                direct_value = partial[direct[i]]
                if direct_value is not None and int(direct_value) != j:
                    return False
        elif kind in ("max_equality", "min_equality"):
            target = int(c["target"])
            vars_ = [int(v) for v in c["vars"]]
            target_value = partial[target]
            if all(partial[var] is not None for var in vars_):
                values = [int(partial[var]) for var in vars_]  # type: ignore[arg-type]
                expected = max(values) if kind == "max_equality" else min(values)
                if target_value is not None and int(target_value) != expected:
                    return False
                if target_value is None and expected not in model["variables"][target]["domain"]:
                    return False
            elif target_value is not None:
                t = int(target_value)
                ranges = []
                for var in vars_:
                    if partial[var] is None:
                        dom = [int(v) for v in model["variables"][var]["domain"]]
                        ranges.append((min(dom), max(dom)))
                    else:
                        v = int(partial[var])
                        ranges.append((v, v))
                if kind == "max_equality":
                    if any(lo > t for lo, _ in ranges) or not any(lo <= t <= hi for lo, hi in ranges):
                        return False
                else:
                    if any(hi < t for _, hi in ranges) or not any(lo <= t <= hi for lo, hi in ranges):
                        return False
        elif kind == "abs_equality":
            target = int(c["target"])
            var = int(c["var"])
            target_value = partial[target]
            var_value = partial[var]
            if target_value is not None and int(target_value) < 0:
                return False
            if target_value is not None and var_value is not None:
                if int(target_value) != abs(int(var_value)):
                    return False
            elif target_value is not None:
                t = int(target_value)
                if not any(abs(int(value)) == t for value in model["variables"][var]["domain"]):
                    return False
            elif var_value is not None:
                expected = abs(int(var_value))
                if expected not in model["variables"][target]["domain"]:
                    return False
        elif kind == "multiplication_equality":
            target = int(c["target"])
            vars_ = [int(v) for v in c["vars"]]
            target_value = partial[target]
            product = 1
            all_assigned = True
            bounds = []
            for var in vars_:
                value = partial[var]
                if value is None:
                    all_assigned = False
                    dom = [int(v) for v in model["variables"][var]["domain"]]
                    bounds.append((min(dom), max(dom)))
                else:
                    value = int(value)
                    product *= value
                    bounds.append((value, value))
            if all_assigned:
                if target_value is not None:
                    if int(target_value) != product:
                        return False
                elif product not in [int(v) for v in model["variables"][target]["domain"]]:
                    return False
            else:
                product_lo, product_hi = product_range(bounds)
                if target_value is not None:
                    if not (product_lo <= int(target_value) <= product_hi):
                        return False
                else:
                    dom = [int(v) for v in model["variables"][target]["domain"]]
                    if max(dom) < product_lo or min(dom) > product_hi:
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
        elif kind == "bool_and":
            lits = []
            for lit in c["literals"]:
                x = xs[int(lit["var"])]
                lits.append(x if bool(lit.get("positive", True)) else x.Not())
            cp.AddBoolAnd(lits)
        elif kind == "bool_xor":
            lits = []
            for lit in c["literals"]:
                x = xs[int(lit["var"])]
                lits.append(x if bool(lit.get("positive", True)) else x.Not())
            cp.AddBoolXOr(lits)
        elif kind == "at_most_one":
            lits = []
            for lit in c["literals"]:
                x = xs[int(lit["var"])]
                lits.append(x if bool(lit.get("positive", True)) else x.Not())
            cp.AddAtMostOne(lits)
        elif kind == "exactly_one":
            lits = []
            for lit in c["literals"]:
                x = xs[int(lit["var"])]
                lits.append(x if bool(lit.get("positive", True)) else x.Not())
            cp.AddExactlyOne(lits)
        elif kind == "implication":
            antecedent_var = xs[int(c["antecedent"]["var"])]
            consequent_var = xs[int(c["consequent"]["var"])]
            antecedent = antecedent_var if bool(c["antecedent"].get("positive", True)) else antecedent_var.Not()
            consequent = consequent_var if bool(c["consequent"].get("positive", True)) else consequent_var.Not()
            cp.AddImplication(antecedent, consequent)
        elif kind == "allowed_assignments":
            cp.AddAllowedAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
        elif kind == "forbidden_assignments":
            cp.AddForbiddenAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
        elif kind == "inverse":
            cp.AddInverse(
                [xs[int(v)] for v in c["direct"]],
                [xs[int(v)] for v in c["inverse"]],
            )
        elif kind == "max_equality":
            cp.AddMaxEquality(
                xs[int(c["target"])],
                [xs[int(v)] for v in c["vars"]],
            )
        elif kind == "min_equality":
            cp.AddMinEquality(
                xs[int(c["target"])],
                [xs[int(v)] for v in c["vars"]],
            )
        elif kind == "abs_equality":
            cp.AddAbsEquality(
                xs[int(c["target"])],
                xs[int(c["var"])],
            )
        elif kind == "multiplication_equality":
            cp.AddMultiplicationEquality(
                xs[int(c["target"])],
                [xs[int(v)] for v in c["vars"]],
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
