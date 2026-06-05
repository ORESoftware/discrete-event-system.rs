#!/usr/bin/env python3
"""External IP/MIP reference solver for small Rust validation cases.

The preferred external engines are open-source packages when present:
OR-Tools CP-SAT or scipy.optimize.milp. Built-in bounded enumeration and any
remaining continuous subproblems are delegated to the Rust reference binary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple

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


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_problem(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "ip_mip_reference"
    explicit = os.environ.get("IP_MIP_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def rust_bounded_reference(
    p: dict,
    solver: str,
    max_enumerations: int,
    pool_size: int | None = None,
) -> dict:
    command = rust_reference_command() + [
        "--solver",
        solver,
        "--max-enumerations",
        str(max_enumerations),
    ]
    if pool_size is not None:
        command.extend(["--pool-size", str(pool_size)])
    cwd = None
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)
    completed = subprocess.run(
        command,
        input=json.dumps(p, allow_nan=True),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    try:
        result = json.loads(completed.stdout)
    except Exception as exc:
        return payload(
            "unavailable",
            "rust:bounded-enumeration",
            message=f"failed to parse Rust IP/MIP reference output: {exc}; stderr={completed.stderr.strip()}",
        )
    if completed.returncode not in (0, 2) and not result.get("message"):
        result["message"] = completed.stderr.strip()
    return {"result": result}


def bounds(p: dict) -> Tuple[List[float], List[Optional[float]]]:
    n = len(p["c"])
    lb = p.get("lb")
    ub = p.get("ub")
    lbs = [0.0] * n if lb is None else [0.0 if v is None else float(v) for v in lb]
    ubs = [None] * n if ub is None else [None if v is None else float(v) for v in ub]
    return lbs, ubs


def objective(p: dict, x: Sequence[float]) -> float:
    return dot([float(v) for v in p["c"]], x)


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


def add_binary_helper(rows: list, c: list, integer_vars: list, lbs: list, ubs: list, var_names: list, name: str) -> int:
    for row in rows:
        row.append(0.0)
    idx = len(c)
    c.append(0.0)
    integer_vars.append(True)
    lbs.append(0.0)
    ubs.append(1.0)
    var_names.append(name)
    return idx


def add_continuous_helper(rows: list, c: list, integer_vars: list, lbs: list, ubs: list, var_names: list, name: str, upper: float) -> int:
    for row in rows:
        row.append(0.0)
    idx = len(c)
    c.append(0.0)
    integer_vars.append(False)
    lbs.append(0.0)
    ubs.append(float(upper))
    var_names.append(name)
    return idx


def add_bounded_continuous_helper(rows: list, c: list, integer_vars: list, lbs: list, ubs: list, var_names: list, name: str, lower: float, upper: float) -> int:
    for row in rows:
        row.append(0.0)
    idx = len(c)
    c.append(0.0)
    integer_vars.append(False)
    lbs.append(float(lower))
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


def expand_abs_constraints(p: dict) -> dict:
    constraints = p.get("abs") or p.get("absolute_values") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None or not math.isfinite(float(ub)) else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    for idx, constraint in enumerate(constraints):
        arg = int(constraint["arg_var"])
        target = int(constraint["target_var"])
        if arg < 0 or arg >= len(c):
            raise ValueError(f"abs {idx} arg_var out of range")
        if target < 0 or target >= len(c):
            raise ValueError(f"abs {idx} target_var out of range")
        if arg == target:
            raise ValueError(f"abs {idx} arg_var and target_var must be distinct")
        lower = float(lbs[arg])
        upper = ubs[arg]
        if upper is not None and float(upper) + 1e-9 < lower:
            raise ValueError(f"abs {idx} argument lower bound exceeds upper bound")
        name = constraint.get("name", f"abs_{idx}")

        if lower >= -1e-12:
            row = [0.0] * len(c)
            row[target] = 1.0
            row[arg] = -1.0
            add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_nonnegative")
            continue
        if upper is not None and float(upper) <= 1e-12:
            row = [0.0] * len(c)
            row[target] = 1.0
            row[arg] = 1.0
            add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_nonpositive")
            continue
        if upper is None:
            raise ValueError(f"abs {idx} mixed-sign argument needs a finite upper bound")
        upper = float(upper)

        selector = add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_sign")

        row = [0.0] * len(c)
        row[arg] = 1.0
        row[target] = -1.0
        add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_ge_arg")

        row = [0.0] * len(c)
        row[arg] = -1.0
        row[target] = -1.0
        add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_ge_neg_arg")

        row = [0.0] * len(c)
        row[target] = 1.0
        row[arg] = -1.0
        row[selector] = -2.0 * lower
        add_compiled_row(
            rows,
            rhs_values,
            row_names,
            row,
            -2.0 * lower,
            f"{name}_target_le_positive_branch",
        )

        row = [0.0] * len(c)
        row[target] = 1.0
        row[arg] = 1.0
        row[selector] = -2.0 * upper
        add_compiled_row(
            rows,
            rhs_values,
            row_names,
            row,
            0.0,
            f"{name}_target_le_negative_branch",
        )

    expanded["c"] = c
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["integer_vars"] = integer_vars
    expanded["lb"] = lbs
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded.pop("abs", None)
    expanded.pop("absolute_values", None)
    return expanded


def max_constraint_candidates(constraint: dict) -> list[tuple[str, float | int]]:
    candidates: list[tuple[str, float | int]] = [("var", int(v)) for v in constraint.get("arg_vars", [])]
    if constraint.get("constant") is not None:
        candidates.append(("constant", float(constraint["constant"])))
    return candidates


def expand_maximum_constraints(p: dict) -> dict:
    constraints = p.get("maximums") or p.get("max_constraints") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None or not math.isfinite(float(ub)) else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    for idx, constraint in enumerate(constraints):
        target = int(constraint["target_var"])
        if target < 0 or target >= len(c):
            raise ValueError(f"maximum {idx} target_var out of range")
        candidates = max_constraint_candidates(constraint)
        if not candidates:
            raise ValueError(f"maximum {idx} needs at least one argument or a constant")
        seen_vars: set[int] = set()
        for kind, value in candidates:
            if kind == "var":
                var = int(value)
                if var < 0 or var >= len(c):
                    raise ValueError(f"maximum {idx} argument variable out of range")
                if var == target:
                    raise ValueError(f"maximum {idx} target_var must be distinct from argument variables")
                if var in seen_vars:
                    raise ValueError(f"maximum {idx} duplicate argument variable {var}")
                seen_vars.add(var)
            elif not math.isfinite(float(value)):
                raise ValueError(f"maximum {idx} constant must be finite")
        name = constraint.get("name", f"maximum_{idx}")

        def candidate_lower(candidate: tuple[str, float | int]) -> float:
            kind, value = candidate
            return lbs[int(value)] if kind == "var" else float(value)

        def candidate_upper(candidate: tuple[str, float | int]) -> Optional[float]:
            kind, value = candidate
            return ubs[int(value)] if kind == "var" else float(value)

        if len(candidates) == 1:
            kind, value = candidates[0]
            row = [0.0] * len(c)
            row[target] = 1.0
            if kind == "var":
                row[int(value)] = -1.0
                add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_single_arg")
            else:
                add_compiled_equality(rows, rhs_values, row_names, row, float(value), f"{name}_constant")
            continue

        target_upper = ubs[target]
        if target_upper is not None:
            max_upper = float(target_upper)
        else:
            uppers = [candidate_upper(candidate) for candidate in candidates]
            if any(upper is None for upper in uppers):
                raise ValueError(f"maximum {idx} needs finite argument uppers or a finite target upper bound")
            max_upper = max(float(upper) for upper in uppers if upper is not None)

        selectors = [
            add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_select_{pos}")
            for pos, _ in enumerate(candidates)
        ]

        for pos, candidate in enumerate(candidates):
            kind, value = candidate
            row = [0.0] * len(c)
            row[target] = -1.0
            if kind == "var":
                row[int(value)] = 1.0
                add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_ge_arg_{pos}")
            else:
                add_compiled_row(rows, rhs_values, row_names, row, -float(value), f"{name}_target_ge_constant")

            big_m = max(0.0, max_upper - candidate_lower(candidate))
            row = [0.0] * len(c)
            row[target] = 1.0
            row[selectors[pos]] = big_m
            if kind == "var":
                row[int(value)] = -1.0
                rhs = big_m
            else:
                rhs = float(value) + big_m
            add_compiled_row(rows, rhs_values, row_names, row, rhs, f"{name}_target_le_candidate_{pos}")

        row = [0.0] * len(c)
        for selector in selectors:
            row[selector] = 1.0
        add_compiled_equality(rows, rhs_values, row_names, row, 1.0, f"{name}_one_active")

    expanded["c"] = c
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["integer_vars"] = integer_vars
    expanded["lb"] = lbs
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded.pop("maximums", None)
    expanded.pop("max_constraints", None)
    return expanded


def min_constraint_candidates(constraint: dict) -> list[tuple[str, float | int]]:
    candidates: list[tuple[str, float | int]] = [("var", int(v)) for v in constraint.get("arg_vars", [])]
    if constraint.get("constant") is not None:
        candidates.append(("constant", float(constraint["constant"])))
    return candidates


def expand_minimum_constraints(p: dict) -> dict:
    constraints = p.get("minimums") or p.get("min_constraints") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None or not math.isfinite(float(ub)) else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    for idx, constraint in enumerate(constraints):
        target = int(constraint["target_var"])
        if target < 0 or target >= len(c):
            raise ValueError(f"minimum {idx} target_var out of range")
        candidates = min_constraint_candidates(constraint)
        if not candidates:
            raise ValueError(f"minimum {idx} needs at least one argument or a constant")
        seen_vars: set[int] = set()
        for kind, value in candidates:
            if kind == "var":
                var = int(value)
                if var < 0 or var >= len(c):
                    raise ValueError(f"minimum {idx} argument variable out of range")
                if var == target:
                    raise ValueError(f"minimum {idx} target_var must be distinct from argument variables")
                if var in seen_vars:
                    raise ValueError(f"minimum {idx} duplicate argument variable {var}")
                seen_vars.add(var)
            elif not math.isfinite(float(value)):
                raise ValueError(f"minimum {idx} constant must be finite")
        name = constraint.get("name", f"minimum_{idx}")

        def candidate_upper(candidate: tuple[str, float | int]) -> float:
            kind, value = candidate
            if kind == "constant":
                return float(value)
            upper = ubs[int(value)]
            if upper is None:
                raise ValueError(f"minimum {idx} argument variable {int(value)} needs a finite upper bound")
            return float(upper)

        if len(candidates) == 1:
            kind, value = candidates[0]
            row = [0.0] * len(c)
            row[target] = 1.0
            if kind == "var":
                row[int(value)] = -1.0
                add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_single_arg")
            else:
                add_compiled_equality(rows, rhs_values, row_names, row, float(value), f"{name}_constant")
            continue

        selectors = [
            add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_select_{pos}")
            for pos, _ in enumerate(candidates)
        ]

        for pos, candidate in enumerate(candidates):
            kind, value = candidate
            row = [0.0] * len(c)
            row[target] = 1.0
            if kind == "var":
                row[int(value)] = -1.0
                add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_le_arg_{pos}")
            else:
                add_compiled_row(rows, rhs_values, row_names, row, float(value), f"{name}_target_le_constant")

            big_m = max(0.0, candidate_upper(candidate) - lbs[target])
            row = [0.0] * len(c)
            row[target] = -1.0
            row[selectors[pos]] = big_m
            if kind == "var":
                row[int(value)] = 1.0
                rhs = big_m
            else:
                rhs = -float(value) + big_m
            add_compiled_row(rows, rhs_values, row_names, row, rhs, f"{name}_target_ge_candidate_{pos}")

        row = [0.0] * len(c)
        for selector in selectors:
            row[selector] = 1.0
        add_compiled_equality(rows, rhs_values, row_names, row, 1.0, f"{name}_one_active")

    expanded["c"] = c
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["integer_vars"] = integer_vars
    expanded["lb"] = lbs
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded.pop("minimums", None)
    expanded.pop("min_constraints", None)
    return expanded


def expand_logical_constraints(p: dict) -> dict:
    constraints = p.get("logical") or p.get("logic_constraints") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None or not math.isfinite(float(ub)) else float(ub) for ub in ubs]
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    def validate_binary(var: int, idx: int, role: str) -> None:
        if var < 0 or var >= len(c):
            raise ValueError(f"logical {idx} {role} variable out of range")
        if abs(lbs[var]) > 1e-12:
            raise ValueError(f"logical {idx} {role} variable must have lower bound 0")
        if not integer_vars[var]:
            raise ValueError(f"logical {idx} {role} variable must be integer/binary")
        upper = ubs[var]
        if upper is None or float(upper) > 1.0 + 1e-9:
            raise ValueError(f"logical {idx} {role} variable must have finite binary upper bound <= 1")

    for idx, constraint in enumerate(constraints):
        target = int(constraint["target_var"])
        validate_binary(target, idx, "target")
        args = [int(v) for v in constraint.get("arg_vars", [])]
        if not args:
            raise ValueError(f"logical {idx} needs at least one argument")
        seen: set[int] = set()
        for var in args:
            validate_binary(var, idx, "argument")
            if var == target:
                raise ValueError(f"logical {idx} target_var must be distinct from argument variables")
            if var in seen:
                raise ValueError(f"logical {idx} duplicate argument variable {var}")
            seen.add(var)
        name = constraint.get("name", f"logical_{idx}")
        kind = constraint.get("kind", "and")

        if kind == "and":
            for pos, var in enumerate(args):
                row = [0.0] * len(c)
                row[target] = 1.0
                row[var] = -1.0
                add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_le_arg_{pos}")
            row = [0.0] * len(c)
            for var in args:
                row[var] = 1.0
            row[target] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, float(len(args) - 1), f"{name}_target_ge_all_args")
        elif kind == "or":
            for pos, var in enumerate(args):
                row = [0.0] * len(c)
                row[var] = 1.0
                row[target] = -1.0
                add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_ge_arg_{pos}")
            row = [0.0] * len(c)
            row[target] = 1.0
            for var in args:
                row[var] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_le_any_arg")
        else:
            raise ValueError(f"logical {idx} has unknown kind {kind}")

    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["con_names"] = row_names
    expanded.pop("logical", None)
    expanded.pop("logic_constraints", None)
    return expanded


def l1_abs_helper_upper(lower: float, upper: Optional[float], idx: int, pos: int) -> float:
    if not math.isfinite(lower):
        raise ValueError(f"l1_norm {idx} argument {pos} lower bound must be finite")
    if upper is not None and float(upper) + 1e-9 < lower:
        raise ValueError(f"l1_norm {idx} argument {pos} lower bound exceeds upper bound")
    if lower >= -1e-12:
        return math.inf if upper is None else max(0.0, float(upper))
    if upper is not None and float(upper) <= 1e-12:
        return max(0.0, -lower)
    if upper is None:
        raise ValueError(f"l1_norm {idx} mixed-sign argument {pos} needs a finite upper bound")
    return max(0.0, -lower, float(upper))


def expand_l1_norm_constraints(p: dict) -> dict:
    constraints = p.get("l1_norms") or p.get("l1_norm_constraints") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None or not math.isfinite(float(ub)) else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    for idx, constraint in enumerate(constraints):
        target = int(constraint["target_var"])
        if target < 0 or target >= len(c):
            raise ValueError(f"l1_norm {idx} target_var out of range")
        if not math.isfinite(lbs[target]):
            raise ValueError(f"l1_norm {idx} target lower bound must be finite")
        args = [int(v) for v in constraint.get("arg_vars", [])]
        if not args:
            raise ValueError(f"l1_norm {idx} needs at least one argument")
        seen: set[int] = set()
        for pos, var in enumerate(args):
            if var < 0 or var >= len(c):
                raise ValueError(f"l1_norm {idx} argument variable {var} out of range")
            if var == target:
                raise ValueError(f"l1_norm {idx} target_var must be distinct from argument variables")
            if var in seen:
                raise ValueError(f"l1_norm {idx} duplicate argument variable {var}")
            seen.add(var)
            l1_abs_helper_upper(float(lbs[var]), ubs[var], idx, pos)

        name = constraint.get("name", f"l1_norm_{idx}")
        helpers: list[int] = []
        for pos, arg in enumerate(args):
            lower = float(lbs[arg])
            upper = ubs[arg]
            helper_upper = l1_abs_helper_upper(lower, upper, idx, pos)
            helper = add_continuous_helper(
                rows,
                c,
                integer_vars,
                lbs,
                ubs,
                var_names,
                f"{name}_abs_{pos}",
                helper_upper,
            )
            helpers.append(helper)

            if lower >= -1e-12:
                row = [0.0] * len(c)
                row[helper] = 1.0
                row[arg] = -1.0
                add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_abs_{pos}_nonnegative")
                continue
            if upper is not None and float(upper) <= 1e-12:
                row = [0.0] * len(c)
                row[helper] = 1.0
                row[arg] = 1.0
                add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_abs_{pos}_nonpositive")
                continue
            if upper is None:
                raise ValueError(f"l1_norm {idx} mixed-sign argument {pos} needs a finite upper bound")
            upper = float(upper)
            selector = add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_abs_{pos}_sign")

            row = [0.0] * len(c)
            row[arg] = 1.0
            row[helper] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_abs_{pos}_target_ge_arg")

            row = [0.0] * len(c)
            row[arg] = -1.0
            row[helper] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_abs_{pos}_target_ge_neg_arg")

            row = [0.0] * len(c)
            row[helper] = 1.0
            row[arg] = -1.0
            row[selector] = -2.0 * lower
            add_compiled_row(
                rows,
                rhs_values,
                row_names,
                row,
                -2.0 * lower,
                f"{name}_abs_{pos}_target_le_positive_branch",
            )

            row = [0.0] * len(c)
            row[helper] = 1.0
            row[arg] = 1.0
            row[selector] = -2.0 * upper
            add_compiled_row(
                rows,
                rhs_values,
                row_names,
                row,
                0.0,
                f"{name}_abs_{pos}_target_le_negative_branch",
            )

        row = [0.0] * len(c)
        row[target] = 1.0
        for helper in helpers:
            row[helper] = -1.0
        add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_target_sum")

    expanded["c"] = c
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["integer_vars"] = integer_vars
    expanded["lb"] = lbs
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded.pop("l1_norms", None)
    expanded.pop("l1_norm_constraints", None)
    return expanded


def linf_abs_helper_upper(lower: float, upper: Optional[float], idx: int, pos: int) -> float:
    if not math.isfinite(lower):
        raise ValueError(f"linf_norm {idx} argument {pos} lower bound must be finite")
    if upper is not None and float(upper) + 1e-9 < lower:
        raise ValueError(f"linf_norm {idx} argument {pos} lower bound exceeds upper bound")
    if lower >= -1e-12:
        return math.inf if upper is None else max(0.0, float(upper))
    if upper is not None and float(upper) <= 1e-12:
        return max(0.0, -lower)
    if upper is None:
        raise ValueError(f"linf_norm {idx} mixed-sign argument {pos} needs a finite upper bound")
    return max(0.0, -lower, float(upper))


def finite_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def expand_linf_norm_constraints(p: dict) -> dict:
    constraints = p.get("linf_norms") or p.get("linf_norm_constraints") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None or not math.isfinite(float(ub)) else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    for idx, constraint in enumerate(constraints):
        target = int(constraint["target_var"])
        if target < 0 or target >= len(c):
            raise ValueError(f"linf_norm {idx} target_var out of range")
        if not math.isfinite(lbs[target]):
            raise ValueError(f"linf_norm {idx} target lower bound must be finite")
        args = [int(v) for v in constraint.get("arg_vars", [])]
        if not args:
            raise ValueError(f"linf_norm {idx} needs at least one argument")
        seen: set[int] = set()
        for pos, var in enumerate(args):
            if var < 0 or var >= len(c):
                raise ValueError(f"linf_norm {idx} argument variable {var} out of range")
            if var == target:
                raise ValueError(f"linf_norm {idx} target_var must be distinct from argument variables")
            if var in seen:
                raise ValueError(f"linf_norm {idx} duplicate argument variable {var}")
            seen.add(var)
            linf_abs_helper_upper(float(lbs[var]), ubs[var], idx, pos)

        name = constraint.get("name", f"linf_norm_{idx}")
        helpers: list[int] = []
        for pos, arg in enumerate(args):
            lower = float(lbs[arg])
            upper = ubs[arg]
            helper_upper = linf_abs_helper_upper(lower, upper, idx, pos)
            helper = add_continuous_helper(
                rows,
                c,
                integer_vars,
                lbs,
                ubs,
                var_names,
                f"{name}_abs_{pos}",
                helper_upper,
            )
            helpers.append(helper)

            if lower >= -1e-12:
                row = [0.0] * len(c)
                row[helper] = 1.0
                row[arg] = -1.0
                add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_abs_{pos}_nonnegative")
                continue
            if upper is not None and float(upper) <= 1e-12:
                row = [0.0] * len(c)
                row[helper] = 1.0
                row[arg] = 1.0
                add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_abs_{pos}_nonpositive")
                continue
            if upper is None:
                raise ValueError(f"linf_norm {idx} mixed-sign argument {pos} needs a finite upper bound")
            upper = float(upper)
            selector = add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_abs_{pos}_sign")

            row = [0.0] * len(c)
            row[arg] = 1.0
            row[helper] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_abs_{pos}_target_ge_arg")

            row = [0.0] * len(c)
            row[arg] = -1.0
            row[helper] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_abs_{pos}_target_ge_neg_arg")

            row = [0.0] * len(c)
            row[helper] = 1.0
            row[arg] = -1.0
            row[selector] = -2.0 * lower
            add_compiled_row(
                rows,
                rhs_values,
                row_names,
                row,
                -2.0 * lower,
                f"{name}_abs_{pos}_target_le_positive_branch",
            )

            row = [0.0] * len(c)
            row[helper] = 1.0
            row[arg] = 1.0
            row[selector] = -2.0 * upper
            add_compiled_row(
                rows,
                rhs_values,
                row_names,
                row,
                0.0,
                f"{name}_abs_{pos}_target_le_negative_branch",
            )

        if len(helpers) == 1:
            row = [0.0] * len(c)
            row[target] = 1.0
            row[helpers[0]] = -1.0
            add_compiled_equality(rows, rhs_values, row_names, row, 0.0, f"{name}_max_abs_single_arg")
            continue

        target_upper = finite_optional(ubs[target])
        if target_upper is not None:
            max_upper = target_upper
        else:
            helper_uppers = [finite_optional(ubs[helper]) for helper in helpers]
            if any(upper is None for upper in helper_uppers):
                raise ValueError(f"linf_norm {idx} needs finite helper uppers or a finite target upper bound")
            max_upper = max(float(upper) for upper in helper_uppers if upper is not None)

        selectors = [
            add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_max_abs_select_{pos}")
            for pos, _ in enumerate(helpers)
        ]

        for pos, helper in enumerate(helpers):
            row = [0.0] * len(c)
            row[target] = -1.0
            row[helper] = 1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_max_abs_target_ge_arg_{pos}")

            row = [0.0] * len(c)
            row[target] = 1.0
            row[helper] = -1.0
            row[selectors[pos]] = max_upper
            add_compiled_row(
                rows,
                rhs_values,
                row_names,
                row,
                max_upper,
                f"{name}_max_abs_target_le_candidate_{pos}",
            )

        row = [0.0] * len(c)
        for selector in selectors:
            row[selector] = 1.0
        add_compiled_equality(rows, rhs_values, row_names, row, 1.0, f"{name}_max_abs_one_active")

    expanded["c"] = c
    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["integer_vars"] = integer_vars
    expanded["lb"] = lbs
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded.pop("linf_norms", None)
    expanded.pop("linf_norm_constraints", None)
    return expanded


def expand_quadratic_objective_terms(p: dict) -> dict:
    terms = p.get("quadratic_objective") or p.get("quadratic_objective_terms") or []
    if not terms:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None or not math.isfinite(float(ub)) else float(ub) for ub in ubs]
    var_names = list(p.get("var_names") or [f"x{i}" for i in range(len(c))])
    products = list(p.get("products") or p.get("product_constraints") or [])

    def is_binary(var: int) -> bool:
        upper = ubs[var]
        return (
            abs(lbs[var]) <= 1e-12
            and integer_vars[var]
            and upper is not None
            and float(upper) <= 1.0 + 1e-9
        )

    def finite_factor_bounds(var: int, idx: int) -> tuple[float, float]:
        lower = float(lbs[var])
        upper = ubs[var]
        if not math.isfinite(lower):
            raise ValueError(f"quadratic objective term {idx} continuous factor {var} lower bound must be finite")
        if upper is None:
            raise ValueError(f"quadratic objective term {idx} continuous factor {var} needs a finite upper bound")
        upper = float(upper)
        if upper + 1e-9 < lower:
            raise ValueError(
                f"quadratic objective term {idx} continuous factor {var} lower bound exceeds upper bound"
            )
        return lower, upper

    for idx, term in enumerate(terms):
        x_var = int(term["x_var"])
        y_var = int(term["y_var"])
        for role, var in (("x_var", x_var), ("y_var", y_var)):
            if var < 0 or var >= len(c):
                raise ValueError(f"quadratic objective term {idx} {role} out of range")
        coeff = float(term["coeff"])
        if not math.isfinite(coeff):
            raise ValueError(f"quadratic objective term {idx} coeff must be finite")

        x_binary = is_binary(x_var)
        y_binary = is_binary(y_var)
        if x_var == y_var:
            if not x_binary:
                raise ValueError(f"quadratic objective term {idx} square is exact only for binary variables")
            c[x_var] += coeff
            continue
        if not x_binary and not y_binary:
            raise ValueError(
                f"quadratic objective term {idx} exact linearization needs at least one binary factor; continuous-continuous products are nonconvex"
            )

        if x_binary and y_binary:
            product_lb, product_ub = 0.0, 1.0
        else:
            continuous = y_var if x_binary else x_var
            lower, upper = finite_factor_bounds(continuous, idx)
            product_lb, product_ub = min(0.0, lower), max(0.0, upper)

        name = term.get("name") or f"quadratic_objective_{idx}"
        helper = add_bounded_continuous_helper(
            rows,
            c,
            integer_vars,
            lbs,
            ubs,
            var_names,
            name,
            product_lb,
            product_ub,
        )
        c[helper] = coeff
        products.append(
            {
                "target_var": helper,
                "x_var": x_var,
                "y_var": y_var,
                "name": name,
            }
        )

    expanded["c"] = c
    expanded["a"] = rows
    expanded["integer_vars"] = integer_vars
    expanded["lb"] = lbs
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["products"] = products
    expanded.pop("quadratic_objective", None)
    expanded.pop("quadratic_objective_terms", None)
    expanded.pop("product_constraints", None)
    return expanded


def expand_product_constraints(p: dict) -> dict:
    constraints = p.get("products") or p.get("product_constraints") or []
    if not constraints:
        return p
    expanded = dict(p)
    rows = [[float(v) for v in row] for row in p.get("a", [])]
    rhs_values = [float(v) for v in p.get("b", [])]
    c = [float(v) for v in p["c"]]
    integer_vars = [bool(v) for v in p["integer_vars"]]
    lbs, ubs = bounds(p)
    ubs = [None if ub is None or not math.isfinite(float(ub)) else float(ub) for ub in ubs]
    row_names = list(p.get("con_names") or [f"c{i}" for i in range(len(rows))])

    def is_binary(var: int) -> bool:
        upper = ubs[var]
        return (
            abs(lbs[var]) <= 1e-12
            and integer_vars[var]
            and upper is not None
            and float(upper) <= 1.0 + 1e-9
        )

    def finite_factor_bounds(var: int, idx: int) -> tuple[float, float]:
        lower = float(lbs[var])
        upper = ubs[var]
        if not math.isfinite(lower):
            raise ValueError(f"product {idx} continuous factor {var} lower bound must be finite")
        if upper is None:
            raise ValueError(f"product {idx} continuous factor {var} needs a finite upper bound")
        upper = float(upper)
        if upper + 1e-9 < lower:
            raise ValueError(f"product {idx} continuous factor {var} lower bound exceeds upper bound")
        return lower, upper

    for idx, constraint in enumerate(constraints):
        target = int(constraint["target_var"])
        x_var = int(constraint["x_var"])
        y_var = int(constraint["y_var"])
        for role, var in (("target_var", target), ("x_var", x_var), ("y_var", y_var)):
            if var < 0 or var >= len(c):
                raise ValueError(f"product {idx} {role} out of range")
        if target == x_var or target == y_var:
            raise ValueError(f"product {idx} target_var must be distinct from factor variables")
        if x_var == y_var:
            raise ValueError(f"product {idx} x_var and y_var must be distinct")
        if not math.isfinite(float(lbs[target])):
            raise ValueError(f"product {idx} target lower bound must be finite")

        x_binary = is_binary(x_var)
        y_binary = is_binary(y_var)
        if not x_binary and not y_binary:
            raise ValueError(
                f"product {idx} exact linearization needs at least one binary factor; continuous-continuous products are nonconvex"
            )

        name = constraint.get("name", f"product_{idx}")
        if x_binary and y_binary:
            row = [0.0] * len(c)
            row[target] = 1.0
            row[x_var] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_le_x")

            row = [0.0] * len(c)
            row[target] = 1.0
            row[y_var] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_le_y")

            row = [0.0] * len(c)
            row[target] = -1.0
            add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_ge_zero")

            row = [0.0] * len(c)
            row[target] = -1.0
            row[x_var] = 1.0
            row[y_var] = 1.0
            add_compiled_row(rows, rhs_values, row_names, row, 1.0, f"{name}_target_ge_xy")
            continue

        binary = x_var if x_binary else y_var
        continuous = y_var if x_binary else x_var
        lower, upper = finite_factor_bounds(continuous, idx)

        row = [0.0] * len(c)
        row[target] = 1.0
        row[binary] = -upper
        add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_le_ub_binary")

        row = [0.0] * len(c)
        row[target] = -1.0
        row[binary] = lower
        add_compiled_row(rows, rhs_values, row_names, row, 0.0, f"{name}_target_ge_lb_binary")

        row = [0.0] * len(c)
        row[target] = 1.0
        row[continuous] = -1.0
        row[binary] = -lower
        add_compiled_row(rows, rhs_values, row_names, row, -lower, f"{name}_target_le_active")

        row = [0.0] * len(c)
        row[target] = -1.0
        row[continuous] = 1.0
        row[binary] = upper
        add_compiled_row(rows, rhs_values, row_names, row, upper, f"{name}_target_ge_active")

    expanded["a"] = rows
    expanded["b"] = rhs_values
    expanded["con_names"] = row_names
    expanded.pop("products", None)
    expanded.pop("product_constraints", None)
    return expanded


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
        for var in ordered:
            if abs(lbs[var]) > 1e-12:
                raise ValueError(f"sos {idx} variable x{var} must have lower bound 0")
        if kind == "sos1":
            selectors = []
            for pos, var in enumerate(ordered):
                upper = finite_sos_ub(ubs, var, idx)
                y = add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_sel_{pos}")
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
                add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_seg_{pos}")
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
    expanded["lb"] = lbs
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
            add_continuous_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_lambda_{pos}", 1.0)
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
    expanded["lb"] = lbs
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
        y = add_binary_helper(rows, c, integer_vars, lbs, ubs, var_names, f"{name}_active")

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
    expanded["lb"] = lbs
    expanded["ub"] = ubs
    expanded["var_names"] = var_names
    expanded["con_names"] = row_names
    expanded.pop("semi_variables", None)
    return expanded


def brute_force(p: dict, max_enumerations: int) -> dict:
    return rust_bounded_reference(p, "enumeration", max_enumerations)


def brute_force_pool(p: dict, pool_size: int, max_enumerations: int) -> dict:
    return rust_bounded_reference(p, "enumeration", max_enumerations, pool_size)


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
        if int(result.status) == 3:
            return payload("unbounded", "scipy:milp", message=str(result.message))
        return payload("unavailable", "scipy:milp", message=str(result.message))
    except Exception as exc:
        return payload("unavailable", "scipy:milp", message=str(exc))


def expand_source_features(p: dict) -> dict:
    p = expand_linear_constraints(p)
    p = expand_indicators(p)
    p = expand_abs_constraints(p)
    p = expand_maximum_constraints(p)
    p = expand_minimum_constraints(p)
    p = expand_logical_constraints(p)
    p = expand_l1_norm_constraints(p)
    p = expand_linf_norm_constraints(p)
    p = expand_quadratic_objective_terms(p)
    p = expand_product_constraints(p)
    p = expand_pwl(p)
    p = expand_sos(p)
    p = expand_semi_variables(p)
    return p


def rust_can_parse_source_features(p: dict) -> bool:
    unsupported_feature_keys = ()
    return not any(p.get(key) for key in unsupported_feature_keys)


def solve_expanded(p: dict, solver: str, max_enumerations: int) -> dict:
    if solver in ("auto", "brute-force", "enumeration", "rust-enumeration"):
        return rust_bounded_reference(p, solver, max_enumerations)
    if solver in ("auto", "ortools", "ortools-cp-sat"):
        r = try_ortools_cp_sat(p)
        if r and r["result"]["status"] in ("optimal", "feasible", "infeasible"):
            return r
        if solver in ("ortools", "ortools-cp-sat"):
            return r or payload("unavailable", "ortools:cp-sat", message="ortools is not installed")
    if solver in ("auto", "scipy", "scipy-milp"):
        r = try_scipy_milp(p)
        if r and r["result"]["status"] in ("optimal", "infeasible", "unbounded"):
            return r
        if solver in ("scipy", "scipy-milp"):
            return r or payload("unavailable", "scipy:milp", message="scipy is not installed")
    return payload("unavailable", solver, message=f"unknown or unavailable solver '{solver}'")


def solve(p: dict, solver: str, max_enumerations: int, pool_size: int | None = None) -> dict:
    if (
        solver in ("auto", "brute-force", "enumeration", "rust-enumeration")
        and rust_can_parse_source_features(p)
    ):
        return rust_bounded_reference(p, solver, max_enumerations, pool_size)
    try:
        p = expand_source_features(p)
    except Exception as exc:
        return payload("unavailable", "source-linearization", message=str(exc))
    objectives = p.get("multi_objectives") or []
    if objectives:
        if pool_size is not None:
            return payload("unavailable", "rust:bounded-enumeration-pool", message="solution pools for multi-objective MIPs are not supported")
        rust_solver = (
            solver
            if solver in ("auto", "brute-force", "enumeration", "rust-enumeration")
            else "rust-enumeration"
        )
        return rust_bounded_reference(p, rust_solver, max_enumerations)
    if pool_size is not None:
        if solver in ("auto", "brute-force", "enumeration", "rust-enumeration"):
            return rust_bounded_reference(p, solver, max_enumerations, pool_size)
        return payload("unavailable", solver, message=f"solution pools are unavailable for solver '{solver}'")
    return solve_expanded(p, solver, max_enumerations)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--problem", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--max-enumerations", type=int, default=1_000_000)
    parser.add_argument("--pool-size", type=int)
    args = parser.parse_args()
    p = load_problem(args.problem)
    result = solve(p, args.solver, args.max_enumerations, args.pool_size)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, allow_nan=True)
        f.write("\n")
    print(json.dumps(result["result"], allow_nan=True))
    return 0 if result["result"]["status"] != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
