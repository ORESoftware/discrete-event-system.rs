#!/usr/bin/env python3
"""Reference bridge for two-stage stochastic linear programs.

The deterministic oracle builds the extensive-form sample-average LP and solves
it with SciPy's HiGHS-backed ``linprog`` when available. This gives the Rust
validation suite a same-input open-source reference for native monolithic SAA
and Benders/L-shaped stochastic LP solves without vendoring solver executables.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Any


def get_any(raw: dict[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in raw:
            return raw[key]
    return default


def numbers(values: Any, name: str) -> list[float]:
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list")
    out = [float(v) for v in values]
    if not all(math.isfinite(v) for v in out):
        raise ValueError(f"{name} must contain finite numbers")
    return out


def matrix(values: Any, name: str, cols: int | None = None) -> list[list[float]]:
    if values is None:
        return []
    if not isinstance(values, list):
        raise ValueError(f"{name} must be a list of rows")
    rows = [numbers(row, f"{name}[{i}]") for i, row in enumerate(values)]
    if cols is not None:
        for i, row in enumerate(rows):
            if len(row) != cols:
                raise ValueError(f"{name}[{i}] length {len(row)} != {cols}")
    return rows


def normalize(raw: dict[str, Any]) -> dict[str, Any]:
    c_first = numbers(get_any(raw, "cFirst", "c_first", "c"), "cFirst")
    q_second = numbers(get_any(raw, "qSecond", "q_second", "q"), "qSecond")
    n_first = len(c_first)
    n_second = len(q_second)
    if n_first == 0 or n_second == 0:
        raise ValueError("cFirst and qSecond must be non-empty")

    a_first = matrix(get_any(raw, "aFirst", "a_first", "A"), "aFirst", n_first)
    b_first = numbers(get_any(raw, "bFirst", "b_first", "b", default=[]), "bFirst")
    if len(a_first) != len(b_first):
        raise ValueError(f"aFirst rows {len(a_first)} != bFirst length {len(b_first)}")

    w_second = matrix(get_any(raw, "wSecond", "w_second", "W"), "wSecond", n_second)
    if not w_second:
        raise ValueError("wSecond must be non-empty")

    raw_scenarios = get_any(raw, "scenarios", "scenarioSet")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenarios must be a non-empty list")

    scenarios = []
    default_prob = 1.0 / len(raw_scenarios)
    for s, raw_scenario in enumerate(raw_scenarios):
        if not isinstance(raw_scenario, dict):
            raise ValueError(f"scenarios[{s}] must be an object")
        t = matrix(get_any(raw_scenario, "t", "T"), f"scenarios[{s}].t", n_first)
        h = numbers(get_any(raw_scenario, "h"), f"scenarios[{s}].h")
        if len(t) != len(w_second) or len(h) != len(w_second):
            raise ValueError(
                f"scenarios[{s}] must have {len(w_second)} recourse rows; "
                f"got T={len(t)} h={len(h)}"
            )
        prob = float(get_any(raw_scenario, "prob", "probability", default=default_prob))
        if not math.isfinite(prob) or prob < 0.0:
            raise ValueError(f"scenarios[{s}].prob must be finite and non-negative")
        scenarios.append({"t": t, "h": h, "prob": prob})

    return {
        "c_first": c_first,
        "a_first": a_first,
        "b_first": b_first,
        "q_second": q_second,
        "w_second": w_second,
        "scenarios": scenarios,
    }


def result(
    status: str,
    solver: str,
    x: list[float] | None = None,
    objective: float | None = None,
    c_first_x: float | None = None,
    expected_q: float | None = None,
    y_by_scenario: list[list[float]] | None = None,
    scenario_values: list[float] | None = None,
    iterations: int | None = None,
    message: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "solver": solver,
        "x": [] if x is None else [float(v) for v in x],
        "objective": None if objective is None else float(objective),
        "cFirstX": None if c_first_x is None else float(c_first_x),
        "expectedQ": None if expected_q is None else float(expected_q),
        "yByScenario": [] if y_by_scenario is None else y_by_scenario,
        "scenarioValues": [] if scenario_values is None else [float(v) for v in scenario_values],
        "iterations": iterations,
        "message": message,
    }


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "stochastic_lp_reference"
    explicit = os.environ.get("STOCHASTIC_LP_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "stochastic_lp_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_stochastic_lp_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "stochastic_lp.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference(raw: dict[str, Any], solver: str = "auto") -> dict[str, Any]:
    command = rust_reference_command()
    cwd = None
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)
    completed = subprocess.run(
        [*command, "--solver", solver],
        input=json.dumps(raw),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    try:
        parsed = json.loads(completed.stdout)
    except Exception as exc:
        return result(
            "numerical-error",
            "rust:stochastic-lp-reference",
            message=f"failed to parse Rust stochastic LP output: {exc}; stderr={completed.stderr.strip()}",
        )
    if completed.returncode != 0 and not parsed.get("message"):
        parsed["message"] = completed.stderr.strip()
    return parsed


def scipy_status(code: int) -> str:
    if code == 0:
        return "optimal"
    if code == 1:
        return "iteration-limit"
    if code == 2:
        return "infeasible"
    if code == 3:
        return "unbounded"
    return "numerical-error"


def solve_scipy(problem: dict[str, Any]) -> dict[str, Any]:
    try:
        from scipy import optimize  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on local env
        return result("unavailable", "scipy:highs-slp", message=f"SciPy unavailable: {exc}")

    c_first = problem["c_first"]
    a_first = problem["a_first"]
    b_first = problem["b_first"]
    q_second = problem["q_second"]
    w_second = problem["w_second"]
    scenarios = problem["scenarios"]
    n_first = len(c_first)
    n_second = len(q_second)
    total_vars = n_first + len(scenarios) * n_second

    c = [0.0 for _ in range(total_vars)]
    for j, value in enumerate(c_first):
        c[j] = -value
    for s, scenario in enumerate(scenarios):
        for j, value in enumerate(q_second):
            c[n_first + s * n_second + j] = -scenario["prob"] * value

    a_ub: list[list[float]] = []
    b_ub: list[float] = []
    for row, rhs in zip(a_first, b_first):
        out = [0.0 for _ in range(total_vars)]
        out[:n_first] = row
        a_ub.append(out)
        b_ub.append(rhs)

    for s, scenario in enumerate(scenarios):
        y_offset = n_first + s * n_second
        for t_row, w_row, rhs in zip(scenario["t"], w_second, scenario["h"]):
            out = [0.0 for _ in range(total_vars)]
            out[:n_first] = t_row
            out[y_offset : y_offset + n_second] = w_row
            a_ub.append(out)
            b_ub.append(rhs)

    try:
        sol = optimize.linprog(
            c,
            A_ub=a_ub if a_ub else None,
            b_ub=b_ub if b_ub else None,
            bounds=[(0.0, None) for _ in range(total_vars)],
            method="highs",
        )
    except Exception as exc:
        return result("numerical-error", "scipy:highs-slp", message=str(exc))

    status = scipy_status(int(sol.status))
    if status != "optimal":
        return result(status, "scipy:highs-slp", iterations=getattr(sol, "nit", None), message=str(sol.message))

    values = [float(v) for v in sol.x]
    x = values[:n_first]
    y_by_scenario: list[list[float]] = []
    scenario_values: list[float] = []
    for s in range(len(scenarios)):
        lo = n_first + s * n_second
        y = values[lo : lo + n_second]
        y_by_scenario.append(y)
        scenario_values.append(sum(q * yj for q, yj in zip(q_second, y)))

    c_first_x = sum(cj * xj for cj, xj in zip(c_first, x))
    expected_q = sum(
        scenario["prob"] * value for scenario, value in zip(scenarios, scenario_values)
    )
    return result(
        "optimal",
        "scipy:highs-slp",
        x=x,
        objective=c_first_x + expected_q,
        c_first_x=c_first_x,
        expected_q=expected_q,
        y_by_scenario=y_by_scenario,
        scenario_values=scenario_values,
        iterations=getattr(sol, "nit", None),
        message=str(sol.message),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=[
            "auto",
            "rust",
            "rust-monolithic",
            "monolithic",
            "scipy",
            "scipy-highs",
            "highs",
            "fallback",
            "rust-fallback",
        ],
        default="auto",
    )
    args = parser.parse_args()

    try:
        raw = json.load(sys.stdin)
        if args.solver in ("scipy", "scipy-highs", "highs"):
            problem = normalize(raw)
            output = solve_scipy(problem)
        else:
            output = rust_reference(raw, args.solver)
        print(json.dumps(output, sort_keys=True))
        return 0 if output["status"] in {
            "optimal",
            "infeasible",
            "unbounded",
            "iteration-limit",
            "unsupported",
            "unavailable",
        } else 1
    except Exception as exc:
        print(
            json.dumps(
                result(
                    "numerical-error",
                    "stochastic-lp-reference",
                    message=str(exc),
                ),
                sort_keys=True,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
