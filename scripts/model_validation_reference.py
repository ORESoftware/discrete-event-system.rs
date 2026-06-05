#!/usr/bin/env python3
"""Reference bridge for external model validators.

The bridge keeps external executables optional. It calls local MiniZinc, SMT, or
SAT tools when available, and falls back to tiny dependency-free validators for
the smoke models used by the Rust cross-check suite.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


def result(
    status: str,
    verdict: str,
    validator: str,
    message: str = "",
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "status": status,
        "verdict": verdict,
        "validator": validator,
        "message": message,
        "stdout": stdout,
        "stderr": stderr,
    }


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "model_validation_reference"
    explicit = os.environ.get("MODEL_VALIDATION_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "model_validation_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_validation_tools.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def exec_rust_builtin_reference(payload: dict[str, Any], tool: str | None = None) -> None:
    command = rust_reference_command()
    args = []
    if tool:
        args.extend(["--tool", tool])
    stdin_file = tempfile.TemporaryFile(mode="w+", encoding="utf-8")
    with stdin_file:
        json.dump(payload, stdin_file)
        stdin_file.flush()
        stdin_file.seek(0)
        os.dup2(stdin_file.fileno(), sys.stdin.fileno())
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
    os.execvp(command[0], [*command, *args])


def normalize_tool_id(tool: str | None) -> str:
    return (tool or "auto").strip().lower().replace("_", "-")


def rust_first_requested(tool: str | None) -> bool:
    normalized = normalize_tool_id(tool)
    if normalized in {"rust", "rust-reference", "rust-fallback", "fallback"}:
        return True
    value = os.environ.get("MODEL_VALIDATION_REFERENCE_RUST_FIRST", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


def command_from_env(tool: str) -> str | None:
    key = "ORES_" + re.sub(r"[^A-Z0-9]+", "_", tool.upper()).strip("_") + "_ADAPTER"
    configured = os.environ.get(key)
    if configured:
        return configured
    return None


def first_command(tool: str, aliases: list[str]) -> str | None:
    configured = command_from_env(tool)
    if configured and shutil.which(configured):
        return configured
    for alias in aliases:
        found = shutil.which(alias)
        if found:
            return found
    return None


MINIZINC_TOOL_IDS = {
    "minizinc",
    "flatzinc",
    "minizinc-solution-checker",
    "gecode",
    "chuffed",
    "ortools-cp-sat",
    "fzn-cp-sat",
}


ASP_TOOL_IDS = {
    "clingo",
    "clingcon",
}


def choose_minizinc_solver(command: str, requested: str = "") -> str:
    ok, stdout, stderr = run_command(command, ["--solvers"])
    catalog = f"{stdout}\n{stderr}".lower() if ok else ""
    aliases = {
        "cbc": "org.minizinc.mip.coin-bc",
        "coin-bc": "org.minizinc.mip.coin-bc",
        "coinbc": "org.minizinc.mip.coin-bc",
        "scip": "org.minizinc.mip.scip",
        "highs": "org.minizinc.mip.highs",
        "gecode": "org.gecode.gecode",
        "chuffed": "org.chuffed.chuffed",
        "cp-sat": "cp-sat",
        "ortools-cp-sat": "cp-sat",
        "or-tools-cp-sat": "cp-sat",
        "fzn-cp-sat": "cp-sat",
    }
    if requested:
        normalized = aliases.get(requested.lower(), requested)
        if not catalog or normalized.lower() in catalog or requested.lower() in catalog:
            return normalized
    for candidate in [
        "org.gecode.gecode",
        "org.chuffed.chuffed",
        "cp-sat",
        "org.minizinc.mip.coin-bc",
        "org.minizinc.mip.scip",
        "org.minizinc.mip.highs",
    ]:
        if candidate.lower() in catalog:
            return candidate
    return requested


def infer_sat_like(stdout: str, stderr: str, exit_success: bool) -> str:
    text = f"{stdout}\n{stderr}".lower()
    if "=====unsatisfiable=====" in text or "unsatisfiable" in text:
        return "unsat"
    if "----------" in stdout and exit_success:
        return "sat"
    for token in re.findall(r"[a-zA-Z_]+", text):
        if token in ("sat", "satisfiable"):
            return "sat"
        if token in ("unsat", "unsatisfiable"):
            return "unsat"
        if token == "unknown":
            return "unknown"
    return "success" if exit_success else "failure"


def run_command(command: str, args: list[str], stdin_text: str = "") -> tuple[bool, str, str]:
    completed = subprocess.run(
        [command, *args],
        input=stdin_text,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return completed.returncode == 0, completed.stdout, completed.stderr


def brute_force_dimacs(text: str) -> dict[str, Any]:
    return exec_rust_builtin_reference({"kind": "dimacs-validation", "dimacs": text})


def validate_dimacs(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    text = str(payload.get("dimacs") or payload.get("cnf") or payload.get("text") or payload.get("model") or "")
    if not text.strip():
        return result("failed", "failure", "dimacs", "payload needs dimacs, cnf, text, or model")
    aliases_by_tool = {
        "kissat": ["kissat"],
        "cadical": ["cadical"],
        "cryptominisat": ["cryptominisat5", "cryptominisat"],
        "minisat": ["minisat"],
        "glucose": ["glucose", "glucose-syrup"],
        "maplesat": ["maplesat", "maple-sat", "maple-lcm"],
        "varisat": ["varisat"],
    }
    aliases = (
        [
            "kissat",
            "cadical",
            "cryptominisat5",
            "cryptominisat",
            "minisat",
            "glucose",
            "glucose-syrup",
            "maplesat",
            "maple-sat",
            "maple-lcm",
            "varisat",
        ]
        if tool == "auto"
        else aliases_by_tool.get(tool, [tool])
    )
    command = first_command(tool, aliases)
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-dimacs-") as tmp:
            path = Path(tmp) / "problem.cnf"
            path.write_text(text, encoding="utf-8")
            ok, stdout, stderr = run_command(command, [str(path)])
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return brute_force_dimacs(text)


def brute_force_wcnf(text: str) -> dict[str, Any]:
    return exec_rust_builtin_reference({"kind": "wcnf-validation", "wcnf": text})


def validate_wcnf(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    text = str(payload.get("wcnf") or payload.get("dimacs") or payload.get("text") or payload.get("model") or "")
    if not text.strip():
        return result("failed", "failure", "wcnf", "payload needs wcnf, dimacs, text, or model")
    aliases_by_tool = {
        "open-wbo": ["open-wbo", "open-wbo_static"],
        "maxhs": ["maxhs"],
        "sat4j": ["sat4j", "sat4j-sat"],
        "pysat": ["pysat-adapter", "python-sat-adapter"],
    }
    aliases = ["open-wbo", "open-wbo_static", "maxhs"] if tool == "auto" else aliases_by_tool.get(tool, [tool])
    command = first_command(tool, aliases)
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-wcnf-") as tmp:
            path = Path(tmp) / "problem.wcnf"
            path.write_text(text, encoding="utf-8")
            ok, stdout, stderr = run_command(command, [str(path)])
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return brute_force_wcnf(text)


def brute_force_opb(text: str) -> dict[str, Any]:
    return exec_rust_builtin_reference({"kind": "opb-validation", "opb": text})


def validate_opb(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    text = str(payload.get("opb") or payload.get("pb") or payload.get("text") or payload.get("model") or "")
    if not text.strip():
        return result("failed", "failure", "opb", "payload needs opb, pb, text, or model")
    aliases_by_tool = {
        "roundingsat": ["roundingsat"],
        "sat4j": ["sat4j", "sat4j-pb"],
        "pysat": ["pysat-adapter", "python-sat-adapter"],
    }
    aliases = ["roundingsat"] if tool == "auto" else aliases_by_tool.get(tool, [tool])
    command = first_command(tool, aliases)
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-opb-") as tmp:
            path = Path(tmp) / "problem.opb"
            path.write_text(text, encoding="utf-8")
            ok, stdout, stderr = run_command(command, [str(path)])
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return brute_force_opb(text)


def builtin_smtlib(text: str) -> dict[str, Any]:
    return exec_rust_builtin_reference({"kind": "smtlib-validation", "script": text})


def validate_smtlib(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    text = str(payload.get("script") or payload.get("smtlib") or payload.get("text") or payload.get("model") or "")
    if not text.strip():
        return result("failed", "failure", "smtlib", "payload needs script, smtlib, text, or model")
    aliases_by_tool = {
        "z3": ["z3"],
        "cvc5": ["cvc5"],
        "yices": ["yices-smt2", "yices"],
        "bitwuzla": ["bitwuzla"],
        "boolector": ["boolector"],
        "mathsat": ["mathsat"],
        "optimathsat": ["optimathsat", "optimathsat5"],
        "opensmt": ["opensmt", "opensmt2"],
        "smtinterpol": ["smtinterpol", "smtinterpol.sh"],
        "princess": ["princess", "princess-smt"],
    }
    aliases = (
        [
            "z3",
            "cvc5",
            "yices-smt2",
            "yices",
            "bitwuzla",
            "boolector",
            "mathsat",
            "optimathsat",
            "optimathsat5",
            "opensmt",
            "opensmt2",
            "smtinterpol",
            "smtinterpol.sh",
            "princess",
            "princess-smt",
        ]
        if tool == "auto"
        else aliases_by_tool.get(tool, [tool])
    )
    command = first_command(tool, aliases)
    if command:
        basename = Path(command).name.lower()
        if basename == "z3":
            args = ["-in", "-smt2"]
        elif basename == "cvc5":
            args = ["--lang=smt2", "-"]
        elif basename in ("yices-smt2", "yices"):
            args = []
        elif basename in ("bitwuzla", "boolector"):
            args = ["--smt2", "-"]
        elif basename in ("mathsat", "optimathsat", "optimathsat5"):
            args = ["-input=smt2"]
        elif basename in ("opensmt", "opensmt2"):
            args = ["--smt2", "-"]
        else:
            args = []
        ok, stdout, stderr = run_command(command, args, text)
        return result("ok" if ok else "failed", infer_sat_like(stdout, stderr, ok), command, "", stdout, stderr)
    return builtin_smtlib(text)


def builtin_minizinc(model: str) -> dict[str, Any]:
    return exec_rust_builtin_reference({"kind": "minizinc-validation", "model": model})


def validate_minizinc(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    model = str(payload.get("model") or "")
    data = str(payload.get("data") or "")
    solver = str(payload.get("solver") or "").strip()
    tool = normalize_tool_id(tool)
    if not model.strip():
        return result("failed", "failure", "minizinc", "payload needs model")
    command_tool = "minizinc" if tool in MINIZINC_TOOL_IDS else tool
    configured = command_from_env(tool)
    command = (
        configured
        if configured and shutil.which(configured)
        else first_command(command_tool, ["minizinc"])
    )
    fzn_gecode = first_command("gecode", ["fzn-gecode"])
    fzn_cp_sat = first_command("ortools-cp-sat", ["fzn-cp-sat"])
    flatzinc_backend: tuple[str, str, str] | None = None
    if tool in {"flatzinc", "gecode"} and fzn_gecode:
        flatzinc_backend = (
            fzn_gecode,
            solver,
            "MiniZinc to FlatZinc compilation failed",
        )
    elif tool in {"ortools-cp-sat", "fzn-cp-sat"} and fzn_cp_sat:
        flatzinc_backend = (
            fzn_cp_sat,
            solver or "cp-sat",
            "MiniZinc to OR-Tools CP-SAT FlatZinc compilation failed",
        )
    if flatzinc_backend is not None:
        flatzinc_command, compile_solver_request, compile_error = flatzinc_backend
        minizinc = first_command("minizinc", ["minizinc"])
        if minizinc:
            with tempfile.TemporaryDirectory(prefix="ores-minizinc-gecode-") as tmp:
                model_path = Path(tmp) / "model.mzn"
                data_path = Path(tmp) / "data.dzn"
                fzn_path = Path(tmp) / "model.fzn"
                model_path.write_text(model, encoding="utf-8")
                compile_solver = choose_minizinc_solver(minizinc, compile_solver_request)
                compile_args = ["--compile", "--solver", compile_solver, "-o", str(fzn_path), str(model_path)]
                if data.strip():
                    data_path.write_text(data, encoding="utf-8")
                    compile_args.append(str(data_path))
                compile_ok, compile_stdout, compile_stderr = run_command(minizinc, compile_args)
                if compile_ok:
                    ok, stdout, stderr = run_command(flatzinc_command, [str(fzn_path)])
                    verdict = infer_sat_like(stdout, stderr, ok)
                    return result("ok" if ok else "failed", verdict, flatzinc_command, "", stdout, stderr)
                if tool in {"gecode", "ortools-cp-sat", "fzn-cp-sat"}:
                    return result(
                        "failed",
                        "failure",
                        minizinc,
                        compile_error,
                        compile_stdout,
                        compile_stderr,
                    )
    if command:
        with tempfile.TemporaryDirectory(prefix="ores-minizinc-") as tmp:
            model_path = Path(tmp) / "model.mzn"
            data_path = Path(tmp) / "data.dzn"
            model_path.write_text(model, encoding="utf-8")
            args = []
            backend_solver = solver or (
                tool if tool in {"gecode", "chuffed", "ortools-cp-sat", "fzn-cp-sat"} else ""
            )
            backend_solver = choose_minizinc_solver(command, backend_solver)
            if backend_solver:
                args.extend(["--solver", backend_solver])
            args.append(str(model_path))
            if data.strip():
                data_path.write_text(data, encoding="utf-8")
                args.append(str(data_path))
            ok, stdout, stderr = run_command(command, args)
        if not ok and backend_solver and "no solver with tag" in stderr.lower():
            return builtin_minizinc(model)
        verdict = infer_sat_like(stdout, stderr, ok)
        return result("ok" if ok else "failed", verdict, command, "", stdout, stderr)
    return builtin_minizinc(model)


def validate_asp(payload: dict[str, Any], tool: str) -> dict[str, Any]:
    program = str(
        payload.get("asp")
        or payload.get("program")
        or payload.get("model")
        or payload.get("text")
        or ""
    )
    tool = normalize_tool_id(tool)
    if not program.strip():
        return result("failed", "failure", "asp", "payload needs asp, program, model, or text")
    aliases_by_tool = {
        "clingo": ["clingo"],
        "clingcon": ["clingcon", "clingo"],
    }
    command = first_command(tool, aliases_by_tool.get(tool, [tool]))
    if command:
        ok, stdout, stderr = run_command(command, ["-", "0"], program)
        verdict = infer_sat_like(stdout, stderr, ok or stdout.strip() != "")
        if verdict in {"sat", "unsat"}:
            return result("ok", verdict, command, "", stdout, stderr)
        return result("failed", verdict, command, "", stdout, stderr)
    return result("unavailable", "unknown", tool, f"{tool} executable not found")


def dispatch(payload: dict[str, Any], tool_override: str | None = None) -> dict[str, Any]:
    kind = normalize_tool_id(str(payload.get("kind", "")))
    tool = normalize_tool_id(tool_override or payload.get("solver") or payload.get("tool"))
    if kind == "minizinc-validation" or tool in MINIZINC_TOOL_IDS:
        return validate_minizinc(payload, "minizinc" if tool == "auto" else tool)
    if kind in ("asp-validation", "clingo-validation") or tool in ASP_TOOL_IDS:
        return validate_asp(payload, "clingo" if tool == "auto" else tool)
    if kind in ("smtlib-validation", "smt-lib-validation") or tool in (
        "z3",
        "cvc5",
        "yices",
        "bitwuzla",
        "boolector",
        "mathsat",
        "optimathsat",
        "opensmt",
        "smtinterpol",
        "princess",
    ):
        return validate_smtlib(payload, tool)
    if kind in ("wcnf-validation", "dimacs-wcnf-validation", "maxsat-validation") or (
        tool in ("open-wbo", "maxhs") and any(key in payload for key in ("wcnf", "dimacs"))
    ):
        return validate_wcnf(payload, tool)
    if kind in ("opb-validation", "pseudo-boolean-validation") or tool in ("roundingsat",):
        return validate_opb(payload, tool)
    if kind in ("dimacs-validation", "dimacs-cnf-validation") or tool in (
        "kissat",
        "cadical",
        "cryptominisat",
        "minisat",
        "glucose",
        "maplesat",
        "varisat",
    ):
        return validate_dimacs(payload, tool)
    return result("unavailable", "unknown", tool, f"unknown model validation payload kind {kind!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tool", default=None)
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        if rust_first_requested(args.tool):
            exec_rust_builtin_reference(payload, args.tool)
        print(json.dumps(dispatch(payload, args.tool)))
    except Exception as exc:
        print(json.dumps(result("failed", "failure", args.tool or "model-validation", str(exc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
