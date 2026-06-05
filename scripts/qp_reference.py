#!/usr/bin/env python3
"""Reference bridge for small convex quadratic programs.

Input JSON:
  {
    "Q": [[...]], "c": [...],
    "A_ub": [[...]], "b_ub": [...],
    "A_eq": [[...]], "b_eq": [...],
    "lb": [0, null], "ub": [1, null]
  }

The bridge delegates default and built-in fallback solves to the Rust
reference binary, while keeping Python ecosystem adapters for optional
external solvers such as HiGHS, SciPy, OSQP, and CVXPY.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
from typing import List, Optional, Sequence, Tuple

CVXPY_SOLVER_ALIASES = {
    "osqp": "OSQP",
    "scs": "SCS",
    "clarabel": "CLARABEL",
    "ecos": "ECOS",
    "proxqp": "PROXQP",
    "sdpa": "SDPA",
    "mosek": "MOSEK",
    "copt": "COPT",
}

CVXPY_REFERENCE_SOLVERS = ("cvxpy", "scs", "clarabel", "ecos", "mosek", "copt")
REGISTERED_CONIC_REFERENCE_SOLVERS = ("qpoases", "proxqp", "cosmo", "sdpa", "csdp")
RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust", "rust-internal", "rust-fallback")


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "qp_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_optimization_tools.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_quadratic_reference.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


@contextlib.contextmanager
def redirect_process_stdout_to_stderr():
    """Keep native solver banners off the JSON stdout protocol."""
    stdout_fd = sys.stdout.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    try:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(sys.stderr.fileno(), stdout_fd)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout_fd, stdout_fd)
        os.close(saved_stdout_fd)


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "qp_reference"
    explicit = os.environ.get("QP_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def package_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def python_bridge_disabled() -> bool:
    for name in ("QP_REFERENCE_RUST_FIRST", "ORES_EXTERNAL_REFERENCE_RUST_FIRST"):
        value = os.environ.get(name)
        if value and value.strip().lower() not in ("0", "false", "off", "disabled"):
            return True
    value = os.environ.get("QP_REFERENCE_PYTHON_BRIDGE", "auto")
    return value.strip().lower() in ("0", "false", "off", "disabled", "rust")


def rust_reference(qp: dict, solver: str = "auto", max_enumerations: int = 1_000_000) -> dict:
    command = rust_reference_command()
    cwd = None
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)
    completed = subprocess.run(
        [
            *command,
            "--solver",
            solver,
            "--max-enumerations",
            str(max_enumerations),
        ],
        input=json.dumps(qp),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    try:
        parsed = json.loads(completed.stdout)
    except Exception as exc:
        return {
            "status": "numerical-error",
            "solver": "rust:quadratic-reference",
            "x": [],
            "objective": None,
            "message": f"failed to parse Rust quadratic output: {exc}; stderr={completed.stderr.strip()}",
        }
    if completed.returncode != 0 and not parsed.get("message"):
        parsed["message"] = completed.stderr.strip()
    return parsed


def exec_rust_reference(raw_stdin: str, solver: str, max_enumerations: int = 1_000_000) -> None:
    command = rust_reference_command()
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    with tempfile.TemporaryFile(mode="w+b") as stdin_file:
        stdin_file.write(raw_stdin.encode("utf-8"))
        stdin_file.flush()
        stdin_file.seek(0)
        os.dup2(stdin_file.fileno(), sys.stdin.fileno())
        os.execvp(
            command[0],
            [
                *command,
                "--solver",
                solver,
                "--max-enumerations",
                str(max_enumerations),
            ],
        )


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def mat_vec(a: Sequence[Sequence[float]], x: Sequence[float]) -> List[float]:
    return [dot(row, x) for row in a]


def objective(qp: dict, x: Sequence[float]) -> float:
    qx = mat_vec(qp["Q"], x)
    return 0.5 * dot(x, qx) + dot(qp["c"], x)


def qp_gradient(qp: dict, x: Sequence[float]) -> List[float]:
    qx = mat_vec(qp["Q"], x)
    return [qi + ci for qi, ci in zip(qx, qp["c"])]


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


def recover_qp_certificate(qp_raw: dict, x: Sequence[float], tol: float = 1e-8) -> dict:
    qp = normalize(qp_raw)
    n = len(qp["c"])
    if len(x) != n:
        return {}

    active: List[Tuple[str, int]] = []
    for i, value in enumerate(x):
        if qp["lb"][i] is not None:
            lower = float(qp["lb"][i])
            if value < lower - 10.0 * tol:
                return {}
            if abs(value - lower) <= 10.0 * tol:
                active.append(("lb", i))
        if qp["ub"][i] is not None:
            upper = float(qp["ub"][i])
            if value > upper + 10.0 * tol:
                return {}
            if abs(value - upper) <= 10.0 * tol:
                active.append(("ub", i))

    for row, rhs in zip(qp["A_eq"], qp["b_eq"]):
        if abs(dot(row, x) - rhs) > 10.0 * tol:
            return {}
    for i, (row, rhs) in enumerate(zip(qp["A_ub"], qp["b_ub"])):
        lhs = dot(row, x)
        if lhs > rhs + 10.0 * tol:
            return {}
        if abs(lhs - rhs) <= 10.0 * tol:
            active.append(("ineq", i))

    rows = [row[:] for row in qp["A_eq"]]
    rows.extend(active_row(qp, item)[0] for item in active)
    gradient = qp_gradient(qp, x)
    unknowns = len(rows)
    if unknowns == 0:
        if any(abs(v) > 10.0 * tol for v in gradient):
            return {}
        solution: List[float] = []
    else:
        normal = [[0.0 for _ in range(unknowns)] for _ in range(unknowns)]
        rhs = [0.0 for _ in range(unknowns)]
        for col, row in enumerate(rows):
            rhs[col] = -dot(row, gradient)
            for other, other_row in enumerate(rows):
                normal[col][other] = dot(row, other_row)
        solved = solve_square(normal, rhs, tol=max(tol, 1e-10))
        if solved is None:
            return {}
        residual = gradient[:]
        for row, dual in zip(rows, solved):
            for j in range(n):
                residual[j] += dual * row[j]
        if any(abs(v) > 1e-6 for v in residual):
            return {}
        solution = solved

    dual_eq = [float(v) for v in solution[: len(qp["A_eq"])]]
    dual_ub = [0.0] * len(qp["A_ub"])
    dual_lower = [0.0] * n
    dual_upper = [0.0] * n
    for offset, item in enumerate(active):
        kind, idx = item
        multiplier = float(solution[len(qp["A_eq"]) + offset])
        if kind == "ineq":
            if multiplier < -1e-7:
                return {}
            dual_ub[idx] = max(0.0, multiplier)
        elif kind == "lb":
            dual = -multiplier
            if dual < -1e-7:
                return {}
            dual_lower[idx] = max(0.0, dual)
        elif kind == "ub":
            if multiplier < -1e-7:
                return {}
            dual_upper[idx] = max(0.0, multiplier)

    reduced = gradient[:]
    for row, dual in zip(qp["A_ub"], dual_ub):
        if dual == 0.0:
            continue
        for j in range(n):
            reduced[j] += dual * row[j]
    for row, dual in zip(qp["A_eq"], dual_eq):
        for j in range(n):
            reduced[j] += dual * row[j]
    for j in range(n):
        stationarity = reduced[j] - dual_lower[j] + dual_upper[j]
        if abs(stationarity) > 1e-6:
            return {}

    return {
        "dualUB": dual_ub,
        "dualEQ": dual_eq,
        "dualLowerBounds": dual_lower,
        "dualUpperBounds": dual_upper,
        "reducedGradient": reduced,
    }


def with_qp_certificate(result: dict, qp_raw: dict) -> dict:
    if result.get("status") == "optimal" and result.get("x") is not None:
        result.update(recover_qp_certificate(qp_raw, result["x"]))
    return result


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


def highs_qp_reference(qp_raw: dict) -> Optional[dict]:
    try:
        import highspy  # type: ignore
    except Exception:
        return None

    qp = normalize(qp_raw)
    n = len(qp["c"])
    highs = highspy.Highs()
    highs.setOptionValue("output_flag", False)
    highs.setOptionValue("primal_feasibility_tolerance", 1e-10)
    highs.setOptionValue("dual_feasibility_tolerance", 1e-10)
    highs.setOptionValue("ipm_optimality_tolerance", 1e-12)
    inf = highs.getInfinity()

    lower = [-inf if v is None else float(v) for v in qp["lb"]]
    upper = [inf if v is None else float(v) for v in qp["ub"]]
    status = highs.addCols(
        n,
        qp["c"],
        lower,
        upper,
        0,
        [0] * (n + 1),
        [],
        [],
    )
    if status != highspy.HighsStatus.kOk:
        return {
            "status": "numerical-error",
            "solver": "highs:qp",
            "x": [],
            "objective": None,
            "message": f"Highs.addCols returned {status}",
        }

    rows = []
    row_lower = []
    row_upper = []
    for row, rhs in zip(qp["A_ub"], qp["b_ub"]):
        rows.append(row)
        row_lower.append(-inf)
        row_upper.append(float(rhs))
    for row, rhs in zip(qp["A_eq"], qp["b_eq"]):
        rows.append(row)
        row_lower.append(float(rhs))
        row_upper.append(float(rhs))
    if rows:
        starts = [0]
        indices = []
        values = []
        for row in rows:
            for j, value in enumerate(row):
                if abs(value) > 0.0:
                    indices.append(j)
                    values.append(float(value))
            starts.append(len(indices))
        status = highs.addRows(
            len(rows),
            row_lower,
            row_upper,
            len(indices),
            starts,
            indices,
            values,
        )
        if status != highspy.HighsStatus.kOk:
            return {
                "status": "numerical-error",
                "solver": "highs:qp",
                "x": [],
                "objective": None,
                "message": f"Highs.addRows returned {status}",
            }

    h_starts = [0]
    h_indices = []
    h_values = []
    for col in range(n):
        for row in range(col + 1):
            value = float(qp["Q"][row][col])
            if abs(value) > 0.0:
                h_indices.append(row)
                h_values.append(value)
        h_starts.append(len(h_indices))
    if h_values:
        status = highs.passHessian(
            n,
            len(h_indices),
            highspy.HessianFormat.kTriangular,
            h_starts,
            h_indices,
            h_values,
        )
        if status != highspy.HighsStatus.kOk:
            return {
                "status": "numerical-error",
                "solver": "highs:qp",
                "x": [],
                "objective": None,
                "message": f"Highs.passHessian returned {status}",
            }

    status = highs.run()
    if status != highspy.HighsStatus.kOk:
        return {
            "status": "numerical-error",
            "solver": "highs:qp",
            "x": [],
            "objective": None,
            "message": f"Highs.run returned {status}",
        }

    model_status = highs.getModelStatus()
    if model_status == highspy.HighsModelStatus.kOptimal:
        solution = highs.getSolution()
        x = [float(v) for v in solution.col_value]
        return with_qp_certificate({
            "status": "optimal",
            "solver": "highs:qp",
            "x": x,
            "objective": objective(qp, x),
            "message": highs.modelStatusToString(model_status),
        }, qp)
    if model_status == highspy.HighsModelStatus.kInfeasible:
        return {
            "status": "infeasible",
            "solver": "highs:qp",
            "x": [],
            "objective": None,
            "message": highs.modelStatusToString(model_status),
        }
    if model_status in (
        highspy.HighsModelStatus.kUnbounded,
        highspy.HighsModelStatus.kUnboundedOrInfeasible,
    ):
        return {
            "status": "unbounded",
            "solver": "highs:qp",
            "x": [],
            "objective": None,
            "message": highs.modelStatusToString(model_status),
        }
    return {
        "status": "numerical-error",
        "solver": "highs:qp",
        "x": [],
        "objective": None,
        "message": highs.modelStatusToString(model_status),
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
    return with_qp_certificate({"status": "optimal", "solver": "scipy:SLSQP", "x": x, "objective": objective(qp, x), "iterations": scipy_iterations(result), "message": str(result.message)}, qp)


def unavailable_reference(solver: str, message: str) -> dict:
    return {
        "status": "unavailable",
        "solver": solver,
        "x": [],
        "objective": None,
        "message": message,
    }


def osqp_reference(qp_raw: dict) -> Optional[dict]:
    try:
        import numpy as np  # type: ignore
        import osqp  # type: ignore
        from scipy import sparse  # type: ignore
    except Exception:
        return None

    qp = normalize(qp_raw)
    n = len(qp["c"])
    rows = []
    lower = []
    upper = []
    inf = float("inf")
    for row, rhs in zip(qp["A_ub"], qp["b_ub"]):
        rows.append(row)
        lower.append(-inf)
        upper.append(float(rhs))
    for row, rhs in zip(qp["A_eq"], qp["b_eq"]):
        rows.append(row)
        lower.append(float(rhs))
        upper.append(float(rhs))
    for idx in range(n):
        row = [0.0] * n
        row[idx] = 1.0
        rows.append(row)
        lower.append(-inf if qp["lb"][idx] is None else float(qp["lb"][idx]))
        upper.append(inf if qp["ub"][idx] is None else float(qp["ub"][idx]))

    p = sparse.csc_matrix(np.array(qp["Q"], dtype=float))
    q = np.array(qp["c"], dtype=float)
    a = sparse.csc_matrix(np.array(rows, dtype=float)) if rows else sparse.csc_matrix((0, n))
    l = np.array(lower, dtype=float)
    u = np.array(upper, dtype=float)
    solver = osqp.OSQP()
    try:
        solver.setup(P=p, q=q, A=a, l=l, u=u, verbose=False, polish=True, eps_abs=1e-9, eps_rel=1e-9)
        result = solver.solve()
    except Exception as exc:
        return {
            "status": "numerical-error",
            "solver": "osqp",
            "x": [],
            "objective": None,
            "message": str(exc),
        }

    status = str(result.info.status).lower()
    if "solved" in status:
        x = [float(v) for v in result.x]
        return with_qp_certificate({
            "status": "optimal",
            "solver": "osqp",
            "x": x,
            "objective": objective(qp, x),
            "iterations": int(result.info.iter),
            "message": result.info.status,
        }, qp)
    if "primal infeasible" in status:
        return {"status": "infeasible", "solver": "osqp", "x": [], "objective": None, "message": result.info.status}
    if "dual infeasible" in status:
        return {"status": "unbounded", "solver": "osqp", "x": [], "objective": None, "message": result.info.status}
    return {"status": "numerical-error", "solver": "osqp", "x": [], "objective": None, "message": result.info.status}


def cvxpy_solver_name(requested: str, installed: Sequence[str]) -> Optional[str]:
    if requested == "cvxpy":
        for candidate in ("CLARABEL", "OSQP", "SCS", "ECOS"):
            if candidate in installed:
                return candidate
        return None
    return CVXPY_SOLVER_ALIASES.get(requested)


def cvxpy_status_payload(problem, solver_label: str, x_value) -> dict:
    status = str(problem.status).lower()
    if status in ("optimal", "optimal_inaccurate"):
        x = [float(v) for v in x_value]
        return {
            "status": "optimal",
            "solver": solver_label,
            "x": x,
            "objective": float(problem.value),
            "message": str(problem.status),
        }
    if "infeasible" in status:
        return {"status": "infeasible", "solver": solver_label, "x": [], "objective": None, "message": str(problem.status)}
    if "unbounded" in status:
        return {"status": "unbounded", "solver": solver_label, "x": [], "objective": None, "message": str(problem.status)}
    return {"status": "numerical-error", "solver": solver_label, "x": [], "objective": None, "message": str(problem.status)}


def cvxpy_reference(qp_raw: dict, requested_solver: str) -> Optional[dict]:
    try:
        import cvxpy as cp  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    qp = normalize(qp_raw)
    installed = set(cp.installed_solvers())
    solver_name = cvxpy_solver_name(requested_solver, installed)
    if solver_name is None or solver_name not in installed:
        return unavailable_reference(
            f"cvxpy:{requested_solver}",
            f"cvxpy solver '{requested_solver}' is not installed",
        )
    x = cp.Variable(len(qp["c"]))
    q = np.array(qp["Q"], dtype=float)
    c = np.array(qp["c"], dtype=float)
    constraints = []
    if qp["A_ub"]:
        constraints.append(np.array(qp["A_ub"], dtype=float) @ x <= np.array(qp["b_ub"], dtype=float))
    if qp["A_eq"]:
        constraints.append(np.array(qp["A_eq"], dtype=float) @ x == np.array(qp["b_eq"], dtype=float))
    for idx, value in enumerate(qp["lb"]):
        if value is not None:
            constraints.append(x[idx] >= float(value))
    for idx, value in enumerate(qp["ub"]):
        if value is not None:
            constraints.append(x[idx] <= float(value))
    problem = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, q) + c @ x), constraints)
    try:
        with redirect_process_stdout_to_stderr():
            problem.solve(solver=solver_name, verbose=False)
    except Exception as exc:
        return {
            "status": "numerical-error",
            "solver": f"cvxpy:{solver_name.lower()}",
            "x": [],
            "objective": None,
            "message": str(exc),
        }
    result = cvxpy_status_payload(problem, f"cvxpy:{solver_name.lower()}", x.value)
    if result.get("status") == "optimal":
        result["objective"] = objective(qp, result["x"])
        result = with_qp_certificate(result, qp)
    return result


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
    return {"status": "optimal", "solver": "scipy:SLSQP-socp", "x": x, "objective": socp_objective(p, x), "iterations": scipy_iterations(result), "message": str(result.message)}


def cvxpy_socp_reference(raw: dict, requested_solver: str) -> Optional[dict]:
    try:
        import cvxpy as cp  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    p = normalize_socp(raw)
    installed = set(cp.installed_solvers())
    solver_name = cvxpy_solver_name(requested_solver, installed)
    if solver_name is None or solver_name not in installed:
        return unavailable_reference(
            f"cvxpy:{requested_solver}",
            f"cvxpy solver '{requested_solver}' is not installed",
        )
    x = cp.Variable(len(p["c"]))
    constraints = []
    if p["A_ub"]:
        constraints.append(np.array(p["A_ub"], dtype=float) @ x <= np.array(p["b_ub"], dtype=float))
    if p["A_eq"]:
        constraints.append(np.array(p["A_eq"], dtype=float) @ x == np.array(p["b_eq"], dtype=float))
    for idx, value in enumerate(p["lb"]):
        if value is not None:
            constraints.append(x[idx] >= float(value))
    for idx, value in enumerate(p["ub"]):
        if value is not None:
            constraints.append(x[idx] <= float(value))
    for cone in p["cones"]:
        constraints.append(
            cp.norm(np.array(cone["A"], dtype=float) @ x + np.array(cone["b"], dtype=float))
            <= np.array(cone["c"], dtype=float) @ x + float(cone["d"])
        )
    problem = cp.Problem(cp.Minimize(np.array(p["c"], dtype=float) @ x), constraints)
    try:
        with redirect_process_stdout_to_stderr():
            problem.solve(solver=solver_name, verbose=False)
    except Exception as exc:
        return {
            "status": "numerical-error",
            "solver": f"cvxpy:{solver_name.lower()}",
            "x": [],
            "objective": None,
            "message": str(exc),
        }
    result = cvxpy_status_payload(problem, f"cvxpy:{solver_name.lower()}", x.value)
    if result.get("status") == "optimal":
        result["objective"] = socp_objective(p, result["x"])
    return result


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
    return {"status": "optimal", "solver": "scipy:SLSQP-qcp", "x": x, "objective": qcp_objective(p, x), "iterations": scipy_iterations(result), "message": str(result.message)}


def cvxpy_qcp_reference(raw: dict, requested_solver: str) -> Optional[dict]:
    try:
        import cvxpy as cp  # type: ignore
        import numpy as np  # type: ignore
    except Exception:
        return None

    p = normalize_qcp(raw)
    installed = set(cp.installed_solvers())
    solver_name = cvxpy_solver_name(requested_solver, installed)
    if solver_name is None or solver_name not in installed:
        return unavailable_reference(
            f"cvxpy:{requested_solver}",
            f"cvxpy solver '{requested_solver}' is not installed",
        )
    x = cp.Variable(len(p["c"]))
    constraints = []
    if p["A_ub"]:
        constraints.append(np.array(p["A_ub"], dtype=float) @ x <= np.array(p["b_ub"], dtype=float))
    if p["A_eq"]:
        constraints.append(np.array(p["A_eq"], dtype=float) @ x == np.array(p["b_eq"], dtype=float))
    for idx, value in enumerate(p["lb"]):
        if value is not None:
            constraints.append(x[idx] >= float(value))
    for idx, value in enumerate(p["ub"]):
        if value is not None:
            constraints.append(x[idx] <= float(value))
    for qc in p["quadratic_constraints"]:
        constraints.append(
            cp.quad_form(x, np.array(qc["Q"], dtype=float))
            + np.array(qc["c"], dtype=float) @ x
            <= float(qc["rhs"])
        )
    problem = cp.Problem(
        cp.Minimize(0.5 * cp.quad_form(x, np.array(p["Q"], dtype=float)) + np.array(p["c"], dtype=float) @ x),
        constraints,
    )
    try:
        with redirect_process_stdout_to_stderr():
            problem.solve(solver=solver_name, verbose=False)
    except Exception as exc:
        return {
            "status": "numerical-error",
            "solver": f"cvxpy:{solver_name.lower()}",
            "x": [],
            "objective": None,
            "message": str(exc),
        }
    result = cvxpy_status_payload(problem, f"cvxpy:{solver_name.lower()}", x.value)
    if result.get("status") == "optimal":
        result["objective"] = qcp_objective(p, result["x"])
    return result


def relabel_registered_fallback(result: dict, solver: str, fallback_kind: str) -> dict:
    output = dict(result)
    output["solver"] = f"builtin:{fallback_kind}-for-{solver}"
    message = str(output.get("message") or "")
    suffix = "registered external solver fallback"
    output["message"] = f"{message}; {suffix}" if message else suffix
    return output


def registered_qp_reference(qp_raw: dict, requested_solver: str) -> dict:
    cvxpy = cvxpy_reference(qp_raw, requested_solver)
    if cvxpy is not None and cvxpy.get("status") not in ("unavailable", "numerical-error"):
        return cvxpy
    return relabel_registered_fallback(rust_reference(qp_raw, "fallback"), requested_solver, "qp-active-set")


def registered_socp_reference(raw: dict, requested_solver: str) -> dict:
    cvxpy = cvxpy_socp_reference(raw, requested_solver)
    if cvxpy is not None and cvxpy.get("status") not in ("unavailable", "numerical-error"):
        return cvxpy
    return relabel_registered_fallback(rust_reference(raw, "fallback"), requested_solver, "socp-pattern-search")


def registered_qcp_reference(raw: dict, requested_solver: str) -> dict:
    cvxpy = cvxpy_qcp_reference(raw, requested_solver)
    if cvxpy is not None and cvxpy.get("status") not in ("unavailable", "numerical-error"):
        return cvxpy
    return relabel_registered_fallback(rust_reference(raw, "fallback"), requested_solver, "qcp-pattern-search")


def scipy_iterations(result) -> int:
    try:
        return int(result.get("nit", 0))
    except AttributeError:
        return int(getattr(result, "nit", 0))


def should_try_next_auto_reference(result: Optional[dict], solver: str) -> bool:
    return solver == "auto" and result is not None and result.get("status") in ("unavailable", "numerical-error")


def continuous_socp_reference(raw: dict, solver: str) -> dict:
    result = None
    if solver in ("auto", "scipy", "scipy-slsqp"):
        result = scipy_socp_reference(raw)
        if solver != "auto" and result is None:
            return {
                "status": "unavailable",
                "solver": "scipy:SLSQP-socp",
                "x": [],
                "objective": None,
                "message": "scipy is not installed",
            }
        if should_try_next_auto_reference(result, solver):
            result = None
    if result is None and solver in ("auto", "cvxpy", "scs", "clarabel", "ecos"):
        result = cvxpy_socp_reference(raw, solver if solver != "auto" else "cvxpy")
        if solver != "auto" and result is None:
            return unavailable_reference(f"cvxpy:{solver}", "cvxpy is not installed")
        if should_try_next_auto_reference(result, solver):
            result = None
    if result is None and solver in REGISTERED_CONIC_REFERENCE_SOLVERS:
        result = registered_socp_reference(raw, solver)
    if result is None:
        result = rust_reference(raw, "fallback")
    return result


def continuous_qcp_reference(raw: dict, solver: str) -> dict:
    result = None
    if solver in ("auto", "scipy", "scipy-slsqp"):
        result = scipy_qcp_reference(raw)
        if solver != "auto" and result is None:
            return {
                "status": "unavailable",
                "solver": "scipy:SLSQP-qcp",
                "x": [],
                "objective": None,
                "message": "scipy is not installed",
            }
        if should_try_next_auto_reference(result, solver):
            result = None
    if result is None and solver in ("auto", "cvxpy", "scs", "clarabel", "ecos"):
        result = cvxpy_qcp_reference(raw, solver if solver != "auto" else "cvxpy")
        if solver != "auto" and result is None:
            return unavailable_reference(f"cvxpy:{solver}", "cvxpy is not installed")
        if should_try_next_auto_reference(result, solver):
            result = None
    if result is None and solver in REGISTERED_CONIC_REFERENCE_SOLVERS:
        result = registered_qcp_reference(raw, solver)
    if result is None:
        result = rust_reference(raw, "fallback")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", default="auto")
    parser.add_argument("--max-enumerations", type=int, default=1_000_000)
    args = parser.parse_args()
    args.solver = args.solver.strip().lower().replace("_", "-")
    raw_stdin = sys.stdin.read()
    if args.solver in RUST_REFERENCE_SOLVERS:
        exec_rust_reference(raw_stdin, args.solver, args.max_enumerations)
    qp = json.loads(raw_stdin)
    if python_bridge_disabled():
        if args.solver in REGISTERED_CONIC_REFERENCE_SOLVERS:
            os.environ["QP_REFERENCE_REGISTERED_FALLBACK"] = "rust"
            exec_rust_reference(raw_stdin, args.solver, args.max_enumerations)
        exec_rust_reference(raw_stdin, "fallback", args.max_enumerations)
    if (
        args.solver in REGISTERED_CONIC_REFERENCE_SOLVERS
        and not qp.get("integer_vars")
        and (python_bridge_disabled() or not package_available("cvxpy"))
    ):
        os.environ["QP_REFERENCE_REGISTERED_FALLBACK"] = "rust"
        exec_rust_reference(raw_stdin, args.solver, args.max_enumerations)
    result = None
    if qp.get("integer_vars") and qp.get("cones"):
        exec_rust_reference(raw_stdin, "fallback", args.max_enumerations)
    if qp.get("integer_vars") and (qp.get("quadratic_constraints") or qp.get("q_constraints")):
        exec_rust_reference(raw_stdin, "fallback", args.max_enumerations)
    if qp.get("integer_vars"):
        exec_rust_reference(raw_stdin, "fallback", args.max_enumerations)
    if qp.get("cones"):
        result = continuous_socp_reference(qp, args.solver)
        print(json.dumps(result))
        return 0 if result.get("status") != "unavailable" else 2
    if qp.get("quadratic_constraints") or qp.get("q_constraints"):
        result = continuous_qcp_reference(qp, args.solver)
        print(json.dumps(result))
        return 0 if result.get("status") != "unavailable" else 2
    if args.solver in ("auto", "highs", "highspy", "highs-qp"):
        result = highs_qp_reference(qp)
        if args.solver != "auto" and result is None:
            result = {"status": "unavailable", "solver": "highs:qp", "x": [], "objective": None, "message": "highspy is not installed"}
    if result is None and args.solver in ("auto", "scipy", "scipy-slsqp"):
        result = scipy_reference(qp)
        if args.solver != "auto" and result is None:
            result = {"status": "unavailable", "solver": "scipy:SLSQP", "x": [], "objective": None, "message": "scipy is not installed"}
    if result is None and args.solver in ("auto", "osqp"):
        result = osqp_reference(qp)
        if args.solver != "auto" and result is None:
            result = unavailable_reference("osqp", "osqp is not installed")
    if result is None and args.solver in ("auto", *CVXPY_REFERENCE_SOLVERS):
        result = cvxpy_reference(qp, args.solver if args.solver != "auto" else "cvxpy")
        if args.solver != "auto" and result is None:
            result = unavailable_reference(f"cvxpy:{args.solver}", "cvxpy is not installed")
    if result is None and args.solver in REGISTERED_CONIC_REFERENCE_SOLVERS:
        result = registered_qp_reference(qp, args.solver)
    if result is None:
        exec_rust_reference(raw_stdin, "fallback", args.max_enumerations)
    print(json.dumps(result))
    return 0 if result.get("status") != "unavailable" else 2


if __name__ == "__main__":
    raise SystemExit(main())
