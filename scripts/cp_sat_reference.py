#!/usr/bin/env python3
"""Reference bridge for small CP-SAT-style finite-domain models.

Rust handles native exact enumeration, solution pools, and assumption cores.
This Python layer remains as the optional OR-Tools CP-SAT adapter for the
same JSON model contract.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Optional, Sequence


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "cp_sat_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_cp_sat_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "cp_sat.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def exec_rust_reference(args: argparse.Namespace) -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "cp_sat_reference"
    command_args = ["--solver", args.solver]
    if args.enumerate_solutions is not None:
        command_args.extend(["--enumerate-solutions", str(args.enumerate_solutions)])
    if args.assumption_core:
        command_args.append("--assumption-core")
    explicit = os.environ.get("CP_SAT_REFERENCE_RUST_BIN")
    if explicit:
        os.execv(explicit, [explicit, *command_args])
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        os.execv(local_binary, [local_binary, *command_args])
    os.chdir(repo_root)
    os.execvp(
        "cargo",
        ["cargo", "run", "--quiet", "--bin", binary_name, "--", *command_args],
    )


def package_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def external_rust_fallback_enabled() -> bool:
    value = os.environ.get("CP_SAT_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


def objective_value(model: dict, assignment: Sequence[int]) -> Optional[int]:
    obj = model.get("objective")
    if not obj:
        return None
    return sum(int(t["coeff"]) * int(assignment[int(t["var"])]) for t in obj.get("terms", []))


def validate_decision_strategies(model: dict) -> list[dict]:
    strategies = list(model.get("decision_strategies") or [])
    seen = set()
    for strategy in strategies:
        vars_ = [int(v) for v in strategy["vars"]]
        if not vars_:
            raise ValueError("decision strategy has no variables")
        for var in vars_:
            if var < 0 or var >= len(model["variables"]):
                raise ValueError(f"decision strategy variable {var} out of range")
            if var in seen:
                raise ValueError(f"duplicate decision strategy for variable {var}")
            seen.add(var)
    return strategies


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
    for hint in model.get("solution_hint", []):
        cp.AddHint(xs[int(hint["var"])], int(hint["value"]))

    def variable_strategy_constant(name: str):
        mapping = {
            "first": cp_model.CHOOSE_FIRST,
            "min_domain_size": cp_model.CHOOSE_MIN_DOMAIN_SIZE,
            "max_domain_size": cp_model.CHOOSE_MAX_DOMAIN_SIZE,
            "lowest_min": cp_model.CHOOSE_LOWEST_MIN,
            "highest_max": cp_model.CHOOSE_HIGHEST_MAX,
        }
        if name not in mapping:
            raise ValueError(f"unknown decision variable strategy {name}")
        return mapping[name]

    def domain_strategy_constant(name: str):
        mapping = {
            "min_value": cp_model.SELECT_MIN_VALUE,
            "max_value": cp_model.SELECT_MAX_VALUE,
            "lower_half": cp_model.SELECT_LOWER_HALF,
            "upper_half": cp_model.SELECT_UPPER_HALF,
            "median_value": cp_model.SELECT_MEDIAN_VALUE,
        }
        if name not in mapping:
            raise ValueError(f"unknown decision domain strategy {name}")
        return mapping[name]

    decision_strategies = validate_decision_strategies(model)
    covered_strategy_vars = set()
    for strategy in decision_strategies:
        vars_ = [int(v) for v in strategy["vars"]]
        covered_strategy_vars.update(vars_)
        cp.AddDecisionStrategy(
            [xs[var] for var in vars_],
            variable_strategy_constant(strategy.get("variable_strategy", "first")),
            domain_strategy_constant(strategy.get("domain_strategy", "min_value")),
        )
    if decision_strategies:
        uncovered = [idx for idx in range(len(xs)) if idx not in covered_strategy_vars]
        if uncovered:
            cp.AddDecisionStrategy(
                [xs[idx] for idx in uncovered],
                cp_model.CHOOSE_MIN_DOMAIN_SIZE,
                cp_model.SELECT_MIN_VALUE,
            )

    def cp_literal(lit: dict):
        var = xs[int(lit["var"])]
        return var if bool(lit.get("positive", True)) else var.Not()

    def literal_expr(lit: dict):
        var = xs[int(lit["var"])]
        return var if bool(lit.get("positive", True)) else 1 - var

    def enforcement_literals(literals: Sequence[dict]):
        return [cp_literal(lit) for lit in literals]

    def fixed_size_interval(start_var, duration: int, name: str, item: dict):
        presence = item.get("presence")
        if presence is None:
            return cp.NewFixedSizeIntervalVar(start_var, duration, name)
        return cp.NewOptionalFixedSizeIntervalVar(
            start_var,
            duration,
            cp_literal(presence),
            name,
        )

    def variable_size_interval(item: dict, name: str):
        start = xs[int(item["start"])]
        size = xs[int(item["duration"])]
        end = xs[int(item["end"])]
        presence = item.get("presence")
        if presence is None:
            return cp.NewIntervalVar(start, size, end, name)
        return cp.NewOptionalIntervalVar(start, size, end, cp_literal(presence), name)

    def variable_axis_interval(
        item: dict,
        start_key: str,
        size_key: str,
        end_key: str,
        name: str,
    ):
        start = xs[int(item[start_key])]
        size = xs[int(item[size_key])]
        end = xs[int(item[end_key])]
        presence = item.get("presence")
        if presence is None:
            return cp.NewIntervalVar(start, size, end, name)
        return cp.NewOptionalIntervalVar(start, size, end, cp_literal(presence), name)

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
        elif kind == "linear_domain":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            cp.AddLinearExpressionInDomain(
                expr,
                cp_model.Domain.FromIntervals(
                    [[int(interval["lb"]), int(interval["ub"])] for interval in c["intervals"]]
                ),
            )
        elif kind == "enforced_linear_domain":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            constraint = cp.AddLinearExpressionInDomain(
                expr,
                cp_model.Domain.FromIntervals(
                    [[int(interval["lb"]), int(interval["ub"])] for interval in c["intervals"]]
                ),
            )
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "map_domain":
            cp.AddMapDomain(
                xs[int(c["var"])],
                [xs[int(v)] for v in c["bools"]],
                int(c.get("offset", 0)),
            )
        elif kind == "enforced_linear":
            expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in c["terms"])
            if c["sense"] == "le":
                constraint = cp.Add(expr <= int(c["rhs"]))
            elif c["sense"] == "ge":
                constraint = cp.Add(expr >= int(c["rhs"]))
            else:
                constraint = cp.Add(expr == int(c["rhs"]))
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_bool_or":
            constraint = cp.AddBoolOr([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_bool_and":
            constraint = cp.AddBoolAnd([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_bool_xor":
            constraint = cp.AddBoolXOr([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_at_most_one":
            constraint = cp.AddAtMostOne([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_at_least_one":
            constraint = cp.AddAtLeastOne([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_exactly_one":
            constraint = cp.AddExactlyOne([cp_literal(lit) for lit in c["literals"]])
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "all_different":
            cp.AddAllDifferent([xs[int(v)] for v in c["vars"]])
        elif kind == "bool_or":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddBoolOr(lits)
        elif kind == "bool_and":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddBoolAnd(lits)
        elif kind == "bool_xor":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddBoolXOr(lits)
        elif kind == "at_most_one":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddAtMostOne(lits)
        elif kind == "at_least_one":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddAtLeastOne(lits)
        elif kind == "exactly_one":
            lits = [cp_literal(lit) for lit in c["literals"]]
            cp.AddExactlyOne(lits)
        elif kind == "implication":
            cp.AddImplication(cp_literal(c["antecedent"]), cp_literal(c["consequent"]))
        elif kind == "circuit":
            arcs = []
            for arc in c["arcs"]:
                arcs.append((int(arc["tail"]), int(arc["head"]), cp_literal(arc["literal"])))
            cp.AddCircuit(arcs)
        elif kind == "multiple_circuit":
            arcs = []
            for arc in c["arcs"]:
                arcs.append((int(arc["tail"]), int(arc["head"]), cp_literal(arc["literal"])))
            cp.AddMultipleCircuit(arcs)
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
        elif kind == "enforced_allowed_assignments":
            constraint = cp.AddAllowedAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
        elif kind == "enforced_forbidden_assignments":
            constraint = cp.AddForbiddenAssignments(
                [xs[int(v)] for v in c["vars"]],
                [[int(v) for v in row] for row in c["tuples"]],
            )
            constraint.OnlyEnforceIf(enforcement_literals(c["enforcement"]))
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
        elif kind == "division_equality":
            cp.AddDivisionEquality(
                xs[int(c["target"])],
                xs[int(c["numerator"])],
                xs[int(c["denominator"])],
            )
        elif kind == "modulo_equality":
            cp.AddModuloEquality(
                xs[int(c["target"])],
                xs[int(c["var"])],
                xs[int(c["modulus"])],
            )
        elif kind == "automaton":
            cp.AddAutomaton(
                [xs[int(v)] for v in c["vars"]],
                int(c["starting_state"]),
                [int(s) for s in c["final_states"]],
                [
                    (int(t["tail"]), int(t["label"]), int(t["head"]))
                    for t in c["transitions"]
                ],
            )
        elif kind == "element":
            cp.AddElement(
                xs[int(c["index"])],
                [int(v) for v in c["values"]],
                xs[int(c["target"])],
            )
        elif kind == "variable_element":
            cp.AddElement(
                xs[int(c["index"])],
                [xs[int(v)] for v in c["vars"]],
                xs[int(c["target"])],
            )
        elif kind == "alternative":
            name = c.get("name", "alternative")
            parent_start = xs[int(c["start"])]
            parent_size = xs[int(c["duration"])]
            parent_end = xs[int(c["end"])]
            parent_presence = c.get("presence")
            if parent_presence is None:
                cp.NewIntervalVar(parent_start, parent_size, parent_end, f"{name}_parent")
            else:
                cp.NewOptionalIntervalVar(
                    parent_start,
                    parent_size,
                    parent_end,
                    cp_literal(parent_presence),
                    f"{name}_parent",
                )
            mode_literals = []
            mode_literal_exprs = []
            for i, mode in enumerate(c["alternatives"]):
                mode_name = mode.get("name", f"{name}_mode_{i}")
                presence = mode["presence"]
                active = cp_literal(presence)
                mode_literals.append(active)
                mode_literal_exprs.append(literal_expr(presence))
                cp.NewOptionalIntervalVar(
                    xs[int(mode["start"])],
                    xs[int(mode["duration"])],
                    xs[int(mode["end"])],
                    active,
                    mode_name,
                )
                cp.Add(parent_start == xs[int(mode["start"])]).OnlyEnforceIf(active)
                cp.Add(parent_size == xs[int(mode["duration"])]).OnlyEnforceIf(active)
                cp.Add(parent_end == xs[int(mode["end"])]).OnlyEnforceIf(active)
            if parent_presence is None:
                cp.AddExactlyOne(mode_literals)
            else:
                cp.Add(sum(mode_literal_exprs) == literal_expr(parent_presence))
        elif kind == "no_overlap":
            intervals = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(
                    fixed_size_interval(
                        xs[int(interval["start"])],
                        int(interval["duration"]),
                        name,
                        interval,
                    )
                )
            cp.AddNoOverlap(intervals)
        elif kind == "no_overlap_variable":
            intervals = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(variable_size_interval(interval, name))
            cp.AddNoOverlap(intervals)
        elif kind == "no_overlap_2d":
            x_intervals = []
            y_intervals = []
            for i, rectangle in enumerate(c["rectangles"]):
                name = rectangle.get("name", f"rectangle_{i}")
                x_intervals.append(
                    fixed_size_interval(
                        xs[int(rectangle["x_start"])],
                        int(rectangle["width"]),
                        f"{name}_x",
                        rectangle,
                    )
                )
                y_intervals.append(
                    fixed_size_interval(
                        xs[int(rectangle["y_start"])],
                        int(rectangle["height"]),
                        f"{name}_y",
                        rectangle,
                    )
                )
            cp.AddNoOverlap2D(x_intervals, y_intervals)
        elif kind == "no_overlap_2d_variable":
            x_intervals = []
            y_intervals = []
            for i, rectangle in enumerate(c["rectangles"]):
                name = rectangle.get("name", f"rectangle_{i}")
                x_intervals.append(
                    variable_axis_interval(
                        rectangle,
                        "x_start",
                        "x_size",
                        "x_end",
                        f"{name}_x",
                    )
                )
                y_intervals.append(
                    variable_axis_interval(
                        rectangle,
                        "y_start",
                        "y_size",
                        "y_end",
                        f"{name}_y",
                    )
                )
            cp.AddNoOverlap2D(x_intervals, y_intervals)
        elif kind == "cumulative":
            intervals = []
            demands = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(
                    fixed_size_interval(
                        xs[int(interval["start"])],
                        int(interval["duration"]),
                        name,
                        interval,
                    )
                )
                demands.append(int(interval["demand"]))
            cp.AddCumulative(intervals, demands, int(c["capacity"]))
        elif kind == "cumulative_variable":
            intervals = []
            demands = []
            for i, interval in enumerate(c["intervals"]):
                name = interval.get("name", f"interval_{i}")
                intervals.append(variable_size_interval(interval, name))
                demands.append(xs[int(interval["demand"])])
            cp.AddCumulative(intervals, demands, xs[int(c["capacity"])])
        elif kind == "reservoir":
            times = [xs[int(event["time"])] for event in c["events"]]
            level_changes = [int(event["level_change"]) for event in c["events"]]
            if any(event.get("active") is not None for event in c["events"]):
                actives = []
                for event in c["events"]:
                    active = event.get("active")
                    if active is None:
                        literal = cp.NewConstant(1)
                    else:
                        var = xs[int(active["var"])]
                        literal = var if bool(active.get("positive", True)) else var.Not()
                    actives.append(literal)
                cp.AddReservoirConstraintWithActive(
                    times,
                    level_changes,
                    actives,
                    int(c["min_level"]),
                    int(c["max_level"]),
                )
            else:
                cp.AddReservoirConstraint(
                    times,
                    level_changes,
                    int(c["min_level"]),
                    int(c["max_level"]),
                )
    if model.get("objective"):
        expr = sum(int(t["coeff"]) * xs[int(t["var"])] for t in model["objective"]["terms"])
        if model["objective"].get("sense", "min") == "min":
            cp.Minimize(expr)
        else:
            cp.Maximize(expr)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    if decision_strategies:
        solver.parameters.search_branching = cp_model.FIXED_SEARCH
        solver.parameters.cp_model_presolve = False
        solver.parameters.num_search_workers = 1
    status = solver.Solve(cp)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        assignment = [int(solver.Value(x)) for x in xs]
        status_label = "optimal" if model.get("objective") and status == cp_model.OPTIMAL else "feasible"
        return {
            "status": status_label,
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
    parser.add_argument("--enumerate-solutions", type=int)
    parser.add_argument("--assumption-core", action="store_true")
    args = parser.parse_args()
    if os.environ.get("CP_SAT_REFERENCE_DISABLE_RUST_EXEC") != "1":
        if (
            args.solver in ("auto", "rust-enumeration", "rust-exact", "python-enumeration")
            or args.enumerate_solutions is not None
            or args.assumption_core
        ):
            exec_rust_reference(args)
        if (
            external_rust_fallback_enabled()
            and args.solver in ("ortools", "ortools-cp-sat")
            and not package_available("ortools")
        ):
            args.solver = "rust-enumeration"
            exec_rust_reference(args)

    model = json.load(sys.stdin)
    if args.assumption_core:
        result = {
            "status": "unavailable",
            "assignment": [],
            "objective": None,
            "nodes": 0,
            "solver": "rust:cp-native-assumption-core",
            "message": "Rust exec is disabled; Python exact CP fallback has been removed",
        }
        print(json.dumps(result))
        return 2

    if args.enumerate_solutions is not None:
        result = {
            "status": "unavailable",
            "assignment": [],
            "objective": None,
            "nodes": 0,
            "solver": "rust:cp-native-solution-enumeration",
            "message": "Rust exec is disabled; Python exact CP fallback has been removed",
        }
        print(json.dumps(result))
        return 2

    result = None
    if args.solver in ("auto", "ortools", "ortools-cp-sat"):
        result = ortools_reference(model)
        if args.solver != "auto" and result is None:
            result = {
                "status": "unavailable",
                "assignment": [],
                "objective": None,
                "nodes": 0,
                "solver": "ortools:cp-sat",
                "message": "ortools is not installed",
            }
    if result is None:
        result = {
            "status": "unavailable",
            "assignment": [],
            "objective": None,
            "nodes": 0,
            "solver": "rust:cp-native-enumeration",
            "message": "Rust exec is disabled; Python exact CP fallback has been removed",
        }
    print(json.dumps(result))
    return 0 if result.get("status") != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
