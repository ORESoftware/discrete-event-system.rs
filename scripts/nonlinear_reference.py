#!/usr/bin/env python3
"""Reference bridge for small nonlinear and derivative-free optimization models.

Input JSON chooses one model family:

  {"kind": "rosenbrock", "x0": [-1.2, 1.0]}
  {"kind": "least_squares", "points": [{"x": 0, "y": 2}], "initial": [1, -0.2]}
  {"kind": "global_benchmark", "objective": "sphere", "dimension": 3,
   "lower": -5, "upper": 5}

The bridge delegates default and built-in fallback solves to the Rust
reference binary, while keeping Python ecosystem adapters for optional SciPy
and NLopt checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from typing import Callable, Optional, Sequence

RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust", "rust-fallback", "rust-reference")


def local_rust_binary_is_current(repo_root: str, binary_path: str) -> bool:
    if not os.path.exists(binary_path):
        return False
    binary_mtime = os.path.getmtime(binary_path)
    source_paths = [
        os.path.join(repo_root, "src", "bin", "nonlinear_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_optimization_tools.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "nonlinear_reference"
    explicit = os.environ.get("NONLINEAR_REFERENCE_RUST_BIN")
    if explicit:
        return [explicit]
    local_binary = os.path.join(repo_root, "target", "debug", binary_name)
    if local_rust_binary_is_current(repo_root, local_binary):
        return [local_binary]
    return ["cargo", "run", "--quiet", "--bin", binary_name, "--"]


def exec_rust_reference(
    solver: str = "auto",
    max_iterations: int = 200,
    raw_stdin: Optional[str] = None,
) -> None:
    command = rust_reference_command()
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    args = [*command, "--solver", solver, "--max-iterations", str(max_iterations)]
    if raw_stdin is None:
        os.execvp(command[0], args)
    with tempfile.TemporaryFile(mode="w+b") as stdin_file:
        stdin_file.write(raw_stdin.encode("utf-8"))
        stdin_file.flush()
        stdin_file.seek(0)
        os.dup2(stdin_file.fileno(), sys.stdin.fileno())
        os.execvp(command[0], args)


def rosenbrock(x: Sequence[float]) -> float:
    return sum(
        100.0 * (x[i + 1] - x[i] * x[i]) ** 2 + (1.0 - x[i]) ** 2
        for i in range(len(x) - 1)
    )


def rosenbrock_grad(x: Sequence[float]) -> list[float]:
    n = len(x)
    g = [0.0] * n
    for i in range(n - 1):
        g[i] += -400.0 * x[i] * (x[i + 1] - x[i] * x[i]) - 2.0 * (1.0 - x[i])
        g[i + 1] += 200.0 * (x[i + 1] - x[i] * x[i])
    return g


def sphere(x: Sequence[float]) -> float:
    return sum(v * v for v in x)


def rastrigin(x: Sequence[float]) -> float:
    return 10.0 * len(x) + sum(v * v - 10.0 * math.cos(2.0 * math.pi * v) for v in x)


def norm2(values: Sequence[float]) -> float:
    return math.sqrt(sum(v * v for v in values))


def result(
    status: str,
    solver: str,
    x: Optional[Sequence[float]] = None,
    objective: Optional[float] = None,
    gradient_norm: Optional[float] = None,
    residual_norm: Optional[float] = None,
    iterations: Optional[int] = None,
    evaluations: Optional[int] = None,
    message: str = "",
) -> dict:
    return {
        "status": status,
        "solver": solver,
        "x": [] if x is None else [float(v) for v in x],
        "objective": None if objective is None else float(objective),
        "gradientNorm": None if gradient_norm is None else float(gradient_norm),
        "residualNorm": None if residual_norm is None else float(residual_norm),
        "iterations": iterations,
        "evaluations": evaluations,
        "message": message,
    }


def load_scipy():
    try:
        import numpy as np  # type: ignore
        from scipy import optimize  # type: ignore

        return optimize, np, None
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, None, exc


def load_nlopt():
    try:
        import nlopt  # type: ignore

        return nlopt, None
    except Exception as exc:  # pragma: no cover - depends on local env
        return None, exc


def solve_rosenbrock_scipy(raw: dict, max_iterations: int) -> Optional[dict]:
    optimize, np, exc = load_scipy()
    if optimize is None:
        return None
    x0 = np.array([float(v) for v in raw.get("x0", [-1.2, 1.0])], dtype=float)
    try:
        sol = optimize.minimize(
            lambda z: rosenbrock(z),
            x0,
            jac=lambda z: np.array(rosenbrock_grad(z), dtype=float),
            method="BFGS",
            options={"gtol": 1e-10, "maxiter": max_iterations},
        )
    except Exception as err:
        return result("numerical-error", "scipy:BFGS", message=str(err))
    x = [float(v) for v in sol.x]
    obj = rosenbrock(x)
    grad = rosenbrock_grad(x)
    status = "optimal" if bool(sol.success) or obj <= 1e-12 else "feasible"
    return result(
        status,
        "scipy:BFGS",
        x=x,
        objective=obj,
        gradient_norm=norm2(grad),
        iterations=getattr(sol, "nit", None),
        evaluations=getattr(sol, "nfev", None),
        message=str(sol.message),
    )


def solve_rosenbrock_nlopt(raw: dict, max_iterations: int) -> Optional[dict]:
    nlopt, exc = load_nlopt()
    if nlopt is None:
        return None
    x0 = [float(v) for v in raw.get("x0", [-1.2, 1.0])]
    opt = nlopt.opt(nlopt.LD_LBFGS, len(x0))
    opt.set_min_objective(
        lambda x, grad: (
            grad.__setitem__(slice(None), rosenbrock_grad(x)) if len(grad) else None,
            rosenbrock(x),
        )[1]
    )
    opt.set_xtol_rel(1e-10)
    opt.set_maxeval(max_iterations)
    try:
        x = [float(v) for v in opt.optimize(x0)]
    except Exception as err:
        return result("numerical-error", "nlopt:LD_LBFGS", message=str(err))
    obj = rosenbrock(x)
    return result(
        "optimal" if obj <= 1e-10 else "feasible",
        "nlopt:LD_LBFGS",
        x=x,
        objective=obj,
        gradient_norm=norm2(rosenbrock_grad(x)),
        evaluations=opt.get_numevals(),
        message=str(opt.last_optimize_result()),
    )


def solve_rosenbrock(raw: dict, solver: str, max_iterations: int) -> dict:
    if solver in ("auto", "scipy"):
        scipy = solve_rosenbrock_scipy(raw, max_iterations)
        if scipy is not None:
            return scipy
    if solver in ("auto", "nlopt"):
        nlopt = solve_rosenbrock_nlopt(raw, max_iterations)
        if nlopt is not None:
            return nlopt
    if solver in ("scipy", "nlopt"):
        return result("unavailable", solver, message=f"{solver} is not installed")
    return result("unavailable", solver, message=f"unknown or unavailable solver: {solver}")


def default_points() -> list[dict]:
    return [
        {"x": 0.0, "y": 2.00},
        {"x": 1.0, "y": 1.22},
        {"x": 2.0, "y": 0.74},
        {"x": 3.0, "y": 0.45},
        {"x": 4.0, "y": 0.27},
    ]


def residuals(params: Sequence[float], points: Sequence[dict]) -> list[float]:
    a, b = params[0], params[1]
    return [a * math.exp(b * float(p["x"])) - float(p["y"]) for p in points]


def jacobian(params: Sequence[float], points: Sequence[dict]) -> list[list[float]]:
    a, b = params[0], params[1]
    out = []
    for point in points:
        x = float(point["x"])
        e = math.exp(b * x)
        out.append([e, a * x * e])
    return out


def least_squares_stats(params: Sequence[float], points: Sequence[dict]) -> tuple[float, float, float]:
    r = residuals(params, points)
    j = jacobian(params, points)
    gradient = [0.0, 0.0]
    for row, ri in zip(j, r):
        gradient[0] += 2.0 * row[0] * ri
        gradient[1] += 2.0 * row[1] * ri
    return sum(v * v for v in r), norm2(r), norm2(gradient)


def solve_least_squares_scipy(raw: dict, max_iterations: int) -> Optional[dict]:
    optimize, np, exc = load_scipy()
    if optimize is None:
        return None
    points = raw.get("points") or default_points()
    initial = np.array([float(v) for v in raw.get("initial", [1.0, -0.2])], dtype=float)
    try:
        sol = optimize.least_squares(
            lambda z: np.array(residuals(z, points), dtype=float),
            initial,
            jac=lambda z: np.array(jacobian(z, points), dtype=float),
            max_nfev=max_iterations,
            xtol=1e-12,
            ftol=1e-12,
            gtol=1e-12,
        )
    except Exception as err:
        return result("numerical-error", "scipy:least_squares", message=str(err))
    x = [float(v) for v in sol.x]
    sse, residual_norm, gradient_norm = least_squares_stats(x, points)
    status = "optimal" if bool(sol.success) or gradient_norm <= 1e-8 else "feasible"
    return result(
        status,
        "scipy:least_squares",
        x=x,
        objective=sse,
        gradient_norm=gradient_norm,
        residual_norm=residual_norm,
        evaluations=getattr(sol, "nfev", None),
        message=str(sol.message),
    )


def solve_least_squares_nlopt(raw: dict, max_iterations: int) -> Optional[dict]:
    nlopt, exc = load_nlopt()
    if nlopt is None:
        return None
    points = raw.get("points") or default_points()
    initial = [float(v) for v in raw.get("initial", [1.0, -0.2])]
    opt = nlopt.opt(nlopt.LD_LBFGS, 2)

    def objective(x, grad):
        if len(grad):
            r = residuals(x, points)
            j = jacobian(x, points)
            g0 = 0.0
            g1 = 0.0
            for row, ri in zip(j, r):
                g0 += 2.0 * row[0] * ri
                g1 += 2.0 * row[1] * ri
            grad[0] = g0
            grad[1] = g1
        sse, _, _ = least_squares_stats(x, points)
        return sse

    opt.set_min_objective(objective)
    opt.set_xtol_rel(1e-10)
    opt.set_maxeval(max_iterations)
    try:
        x = [float(v) for v in opt.optimize(initial)]
    except Exception as err:
        return result("numerical-error", "nlopt:LD_LBFGS-sse", message=str(err))
    sse, residual_norm, gradient_norm = least_squares_stats(x, points)
    return result(
        "optimal" if gradient_norm <= 1e-6 else "feasible",
        "nlopt:LD_LBFGS-sse",
        x=x,
        objective=sse,
        gradient_norm=gradient_norm,
        residual_norm=residual_norm,
        evaluations=opt.get_numevals(),
        message=str(opt.last_optimize_result()),
    )


def solve_least_squares(raw: dict, solver: str, max_iterations: int) -> dict:
    if solver in ("auto", "scipy"):
        scipy = solve_least_squares_scipy(raw, max_iterations)
        if scipy is not None:
            return scipy
    if solver in ("auto", "nlopt"):
        nlopt = solve_least_squares_nlopt(raw, max_iterations)
        if nlopt is not None:
            return nlopt
    if solver == "scipy":
        return result("unavailable", "scipy", message="SciPy is not installed")
    if solver == "nlopt":
        return result("unavailable", "nlopt", message="NLopt is not installed")
    return result("unavailable", solver, message=f"unknown or unavailable solver: {solver}")


def benchmark_function(name: str) -> Callable[[Sequence[float]], float]:
    if name == "sphere":
        return sphere
    if name == "rastrigin":
        return rastrigin
    if name == "rosenbrock":
        return rosenbrock
    raise ValueError(f"unknown objective: {name}")


def known_global_solution(name: str, dimension: int, lower: float, upper: float) -> Optional[list[float]]:
    if name in ("sphere", "rastrigin") and lower <= 0.0 <= upper:
        return [0.0] * dimension
    if name == "rosenbrock" and lower <= 1.0 <= upper:
        return [1.0] * dimension
    return None


def solve_global_scipy(raw: dict, max_iterations: int) -> Optional[dict]:
    optimize, np, exc = load_scipy()
    if optimize is None:
        return None
    name = str(raw.get("objective", "sphere"))
    dimension = int(raw.get("dimension", 3))
    lower = float(raw.get("lower", -5.0))
    upper = float(raw.get("upper", 5.0))
    fun = benchmark_function(name)
    try:
        sol = optimize.differential_evolution(
            lambda z: fun(z),
            [(lower, upper)] * dimension,
            seed=0,
            maxiter=max_iterations,
            popsize=8,
            tol=1e-9,
            polish=True,
            updating="immediate",
            workers=1,
        )
    except Exception as err:
        return result("numerical-error", "scipy:differential_evolution", message=str(err))
    x = [float(v) for v in sol.x]
    obj = fun(x)
    known = known_global_solution(name, dimension, lower, upper)
    status = "optimal" if known is not None and obj <= fun(known) + 1e-8 else "feasible"
    return result(
        status,
        "scipy:differential_evolution",
        x=x,
        objective=obj,
        iterations=getattr(sol, "nit", None),
        evaluations=getattr(sol, "nfev", None),
        message=str(sol.message),
    )


def solve_global_nlopt(raw: dict, max_iterations: int) -> Optional[dict]:
    nlopt, exc = load_nlopt()
    if nlopt is None:
        return None
    name = str(raw.get("objective", "sphere"))
    dimension = int(raw.get("dimension", 3))
    lower = float(raw.get("lower", -5.0))
    upper = float(raw.get("upper", 5.0))
    fun = benchmark_function(name)
    opt = nlopt.opt(nlopt.GN_DIRECT_L, dimension)
    opt.set_lower_bounds([lower] * dimension)
    opt.set_upper_bounds([upper] * dimension)
    opt.set_min_objective(lambda x, grad: fun(x))
    opt.set_xtol_rel(1e-9)
    opt.set_maxeval(max(1, max_iterations * max(1, dimension)))
    try:
        x = [float(v) for v in opt.optimize([(lower + upper) / 2.0] * dimension)]
    except Exception as err:
        return result("numerical-error", "nlopt:GN_DIRECT_L", message=str(err))
    obj = fun(x)
    known = known_global_solution(name, dimension, lower, upper)
    status = "optimal" if known is not None and obj <= fun(known) + 1e-7 else "feasible"
    return result(
        status,
        "nlopt:GN_DIRECT_L",
        x=x,
        objective=obj,
        evaluations=opt.get_numevals(),
        message=str(opt.last_optimize_result()),
    )


def solve_global(raw: dict, solver: str, max_iterations: int) -> dict:
    if solver in ("auto", "scipy"):
        scipy = solve_global_scipy(raw, max_iterations)
        if scipy is not None:
            return scipy
    if solver in ("auto", "nlopt"):
        nlopt = solve_global_nlopt(raw, max_iterations)
        if nlopt is not None:
            return nlopt
    if solver == "scipy":
        return result("unavailable", "scipy", message="SciPy is not installed")
    if solver == "nlopt":
        return result("unavailable", "nlopt", message="NLopt is not installed")
    return result("unavailable", solver, message=f"unknown or unavailable solver: {solver}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        default="auto",
        choices=["auto", "scipy", "nlopt", "fallback", "rust", "rust-fallback", "rust-reference"],
    )
    parser.add_argument("--max-iterations", type=int, default=200)
    args = parser.parse_args()
    if args.solver in RUST_REFERENCE_SOLVERS:
        exec_rust_reference(args.solver, args.max_iterations)

    try:
        raw_stdin = sys.stdin.read()
        raw = json.loads(raw_stdin)
        kind = str(raw.get("kind", "rosenbrock"))
        if kind == "pareto_portfolio":
            exec_rust_reference("fallback", args.max_iterations, raw_stdin)
        elif kind == "rosenbrock":
            out = solve_rosenbrock(raw, args.solver, args.max_iterations)
        elif kind == "least_squares":
            out = solve_least_squares(raw, args.solver, args.max_iterations)
        elif kind == "global_benchmark":
            out = solve_global(raw, args.solver, args.max_iterations)
        else:
            out = result("unsupported", "nonlinear-reference", message=f"unknown kind: {kind}")
    except Exception as exc:
        out = result("numerical-error", "nonlinear-reference", message=str(exc))
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
