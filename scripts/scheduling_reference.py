#!/usr/bin/env python3
"""Reference bridge for small scheduling instances.

The deterministic exact job-shop and flow-shop oracles live in Rust. This
Python bridge remains as thin adapter glue for explicit OR-Tools CP-SAT checks
without vendoring solver executables.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
from typing import Optional


SCALE = 1000
RUST_REFERENCE_SOLVERS = ("auto", "fallback", "rust-exact")


def rust_reference_command() -> list[str]:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)
    binary_name = "scheduling_reference"
    explicit = os.environ.get("SCHEDULING_REFERENCE_RUST_BIN")
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
        os.path.join(repo_root, "src", "bin", "scheduling_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "external_scheduling_reference.rs"),
        os.path.join(repo_root, "src", "des", "general", "scheduling.rs"),
    ]
    return all(
        not os.path.exists(source_path) or os.path.getmtime(source_path) <= binary_mtime
        for source_path in source_paths
    )


def exec_rust_reference(solver: str, kind: str) -> None:
    command = rust_reference_command() + ["--solver", solver, "--kind", kind]
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(os.path.dirname(script_dir))
        os.execvp(command[0], command)
    os.execv(command[0], command)


def package_available(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except Exception:
        return False


def external_rust_fallback_enabled() -> bool:
    value = os.environ.get("SCHEDULING_REFERENCE_EXTERNAL_FALLBACK", "")
    return value.strip().lower() in ("1", "true", "yes", "on", "rust")


def external_rust_first_enabled() -> bool:
    values = (
        os.environ.get("SCHEDULING_REFERENCE_RUST_FIRST", ""),
        os.environ.get("ORES_EXTERNAL_REFERENCE_RUST_FIRST", ""),
    )
    return any(value.strip().lower() in ("1", "true", "yes", "on", "rust") for value in values)


def rust_reference(raw: dict, kind: str) -> dict:
    command = rust_reference_command() + ["--solver", "rust-exact", "--kind", kind]
    cwd = None
    if command[0] == "cargo":
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cwd = os.path.dirname(script_dir)
    completed = subprocess.run(
        command,
        input=json.dumps(raw),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd,
        check=False,
    )
    try:
        output = json.loads(completed.stdout)
    except Exception as exc:
        return {
            "status": "error",
            "solver": "rust:scheduling-reference",
            "schedule": [],
            "sequence": [],
            "makespan": None,
            "totalFlowTime": None,
            "message": f"failed to parse Rust reference output: {exc}; stderr={completed.stderr.strip()}",
        }
    if completed.returncode != 0 and not output.get("message"):
        output["message"] = completed.stderr.strip()
    return output


def normalize_job_shop(raw: dict) -> list[dict]:
    jobs = raw.get("jobs") or []
    if not jobs:
        raise ValueError("jobs must be non-empty")
    normalized = []
    seen = set()
    for job_index, raw_job in enumerate(jobs):
        job_id = str(raw_job.get("id", f"J{job_index + 1}"))
        if not job_id.strip():
            raise ValueError(f"jobs[{job_index}].id must be non-empty")
        if job_id in seen:
            raise ValueError(f"duplicate job id {job_id!r}")
        seen.add(job_id)
        operations = []
        for op_index, raw_op in enumerate(raw_job.get("operations") or []):
            machine = str(raw_op.get("machine", ""))
            duration = float(raw_op.get("duration", 0.0))
            if not machine.strip():
                raise ValueError(
                    f"jobs[{job_index}].operations[{op_index}].machine must be non-empty"
                )
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError(
                    f"jobs[{job_index}].operations[{op_index}].duration must be finite and >= 0"
                )
            operations.append({"machine": machine, "duration": duration})
        if not operations:
            raise ValueError(f"jobs[{job_index}].operations must be non-empty")
        normalized.append(
            {
                "id": job_id,
                "due": raw_job.get("due"),
                "operations": operations,
            }
        )
    return normalized


def normalize_flow_shop(raw: dict) -> list[dict]:
    jobs = raw.get("jobs") or []
    if not jobs:
        raise ValueError("jobs must be non-empty")
    normalized = []
    seen = set()
    machine_count: Optional[int] = None
    for job_index, raw_job in enumerate(jobs):
        job_id = str(raw_job.get("id", f"F{job_index + 1}"))
        if not job_id.strip():
            raise ValueError(f"jobs[{job_index}].id must be non-empty")
        if job_id in seen:
            raise ValueError(f"duplicate job id {job_id!r}")
        seen.add(job_id)
        raw_times = raw_job.get("processingTimes", raw_job.get("processing_times"))
        processing_times = [float(v) for v in (raw_times or [])]
        if not processing_times:
            raise ValueError(f"jobs[{job_index}].processingTimes must be non-empty")
        if machine_count is None:
            machine_count = len(processing_times)
        if len(processing_times) != machine_count:
            raise ValueError(
                f"jobs[{job_index}].processingTimes length {len(processing_times)} != {machine_count}"
            )
        for machine_index, duration in enumerate(processing_times):
            if not math.isfinite(duration) or duration < 0.0:
                raise ValueError(
                    f"jobs[{job_index}].processingTimes[{machine_index}] must be finite and >= 0"
                )
        normalized.append(
            {
                "id": job_id,
                "due": raw_job.get("due"),
                "processingTimes": processing_times,
            }
        )
    return normalized


def schedule_result(
    status: str,
    solver: str,
    schedule: list[dict],
    message: str = "",
    sequence: Optional[list[str]] = None,
) -> dict:
    makespan = max((op["finish"] for op in schedule), default=0.0)
    completions = {}
    for op in schedule:
        completions[op["jobId"]] = max(completions.get(op["jobId"], 0.0), op["finish"])
    return {
        "status": status,
        "solver": solver,
        "schedule": schedule,
        "sequence": [] if sequence is None else list(sequence),
        "makespan": float(makespan),
        "totalFlowTime": float(sum(completions.values())),
        "message": message,
    }


def flow_shop_schedule(sequence: list[dict]) -> list[dict]:
    if not sequence:
        return []
    machine_count = len(sequence[0]["processingTimes"])
    machine_ready = [0.0 for _ in range(machine_count)]
    schedule = []
    for job in sequence:
        job_ready = 0.0
        for machine_index, duration in enumerate(job["processingTimes"]):
            start = max(machine_ready[machine_index], job_ready)
            finish = start + duration
            schedule.append(
                {
                    "jobId": job["id"],
                    "opIndex": machine_index,
                    "machine": f"M{machine_index + 1}",
                    "start": float(start),
                    "finish": float(finish),
                }
            )
            machine_ready[machine_index] = finish
            job_ready = finish
    return schedule


def scaled_duration(duration: float) -> int:
    return int(round(duration * SCALE))


def ortools_cp_sat(jobs: list[dict]) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return {"status": "unavailable", "message": f"OR-Tools CP-SAT unavailable: {exc}"}

    horizon = sum(scaled_duration(op["duration"]) for job in jobs for op in job["operations"])
    model = cp_model.CpModel()
    operations = {}
    machine_intervals: dict[str, list] = {}
    last_ends = []

    for job_index, job in enumerate(jobs):
        previous_end = None
        for op_index, op in enumerate(job["operations"]):
            duration = scaled_duration(op["duration"])
            suffix = f"j{job_index}_o{op_index}"
            start = model.NewIntVar(0, horizon, f"start_{suffix}")
            end = model.NewIntVar(0, horizon, f"end_{suffix}")
            interval = model.NewIntervalVar(start, duration, end, f"interval_{suffix}")
            operations[(job_index, op_index)] = (start, end, duration)
            machine_intervals.setdefault(op["machine"], []).append(interval)
            if previous_end is not None:
                model.Add(start >= previous_end)
            previous_end = end
        if previous_end is not None:
            last_ends.append(previous_end)

    for intervals in machine_intervals.values():
        model.AddNoOverlap(intervals)

    makespan = model.NewIntVar(0, horizon, "makespan")
    model.AddMaxEquality(makespan, last_ends)
    model.Minimize(makespan)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status_name, "message": f"OR-Tools CP-SAT status {status_name}"}

    schedule = []
    for job_index, job in enumerate(jobs):
        for op_index, op in enumerate(job["operations"]):
            start_var, end_var, _ = operations[(job_index, op_index)]
            start = solver.Value(start_var) / SCALE
            finish = solver.Value(end_var) / SCALE
            schedule.append(
                {
                    "jobId": job["id"],
                    "opIndex": op_index,
                    "machine": op["machine"],
                    "start": float(start),
                    "finish": float(finish),
                }
            )
    schedule.sort(key=lambda op: (op["start"], op["finish"], op["machine"], op["jobId"], op["opIndex"]))
    result = schedule_result(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat",
        schedule,
        f"OR-Tools CP-SAT status {status_name}",
    )
    result["objectiveBound"] = solver.BestObjectiveBound() / SCALE
    return result


def ortools_flow_shop_cp_sat(jobs: list[dict]) -> dict:
    try:
        from ortools.sat.python import cp_model  # type: ignore
    except Exception as exc:
        return {"status": "unavailable", "message": f"OR-Tools CP-SAT unavailable: {exc}"}

    n = len(jobs)
    machine_count = len(jobs[0]["processingTimes"])
    scaled = [
        [scaled_duration(duration) for duration in job["processingTimes"]]
        for job in jobs
    ]
    horizon = sum(sum(row) for row in scaled)
    model = cp_model.CpModel()
    assigned = {
        (job, pos): model.NewBoolVar(f"assign_j{job}_p{pos}")
        for job in range(n)
        for pos in range(n)
    }
    for job in range(n):
        model.AddExactlyOne(assigned[(job, pos)] for pos in range(n))
    for pos in range(n):
        model.AddExactlyOne(assigned[(job, pos)] for job in range(n))

    completion = [
        [model.NewIntVar(0, horizon, f"c_p{pos}_m{machine}") for machine in range(machine_count)]
        for pos in range(n)
    ]
    for pos in range(n):
        for machine in range(machine_count):
            duration_expr = sum(assigned[(job, pos)] * scaled[job][machine] for job in range(n))
            if pos == 0 and machine == 0:
                model.Add(completion[pos][machine] >= duration_expr)
            elif pos == 0:
                model.Add(completion[pos][machine] >= completion[pos][machine - 1] + duration_expr)
            elif machine == 0:
                model.Add(completion[pos][machine] >= completion[pos - 1][machine] + duration_expr)
            else:
                model.Add(completion[pos][machine] >= completion[pos - 1][machine] + duration_expr)
                model.Add(completion[pos][machine] >= completion[pos][machine - 1] + duration_expr)

    makespan = completion[n - 1][machine_count - 1]
    model.Minimize(makespan)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    solver.parameters.num_search_workers = 1
    status_code = solver.Solve(model)
    status_name = solver.StatusName(status_code).lower()
    if status_code not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return {"status": status_name, "message": f"OR-Tools CP-SAT status {status_name}"}

    sequence = []
    for pos in range(n):
        job_index = next(job for job in range(n) if solver.BooleanValue(assigned[(job, pos)]))
        sequence.append(jobs[job_index])
    schedule = flow_shop_schedule(sequence)
    result = schedule_result(
        "optimal" if status_code == cp_model.OPTIMAL else "feasible",
        "ortools:cp-sat-flow-shop",
        schedule,
        f"OR-Tools CP-SAT status {status_name}",
        [job["id"] for job in sequence],
    )
    result["objectiveBound"] = solver.BestObjectiveBound() / SCALE
    return result


def infer_kind(raw: dict, requested: str) -> str:
    if requested != "auto":
        return requested
    raw_kind = str(raw.get("kind", raw.get("type", ""))).lower().replace("_", "-")
    if raw_kind in ("flow-shop", "flowshop", "permutation-flow-shop"):
        return "flow-shop"
    if raw_kind in ("job-shop", "jobshop"):
        return "job-shop"
    jobs = raw.get("jobs") or []
    if jobs and any("processingTimes" in job or "processing_times" in job for job in jobs):
        return "flow-shop"
    return "job-shop"


def rust_reference_embedded() -> bool:
    value = os.environ.get("SCHEDULING_REFERENCE_RUST_REFERENCE_EMBEDDED", "")
    return value.strip().lower() in ("1", "true", "yes", "on")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--solver",
        choices=["auto", "ortools", "fallback", "rust-exact"],
        default="auto",
    )
    parser.add_argument("--kind", choices=["auto", "job-shop", "flow-shop"], default="auto")
    args = parser.parse_args()
    if args.solver in RUST_REFERENCE_SOLVERS:
        exec_rust_reference(args.solver, args.kind)
    if external_rust_first_enabled() and args.solver == "ortools":
        os.environ["SCHEDULING_REFERENCE_EXTERNAL_FALLBACK"] = "rust"
        exec_rust_reference(args.solver, args.kind)
    if (
        external_rust_fallback_enabled()
        and args.solver == "ortools"
        and not package_available("ortools")
    ):
        exec_rust_reference("rust-exact", args.kind)

    try:
        raw = json.load(sys.stdin)
        kind = infer_kind(raw, args.kind)
        if kind == "flow-shop":
            jobs = normalize_flow_shop(raw)
            ortools_fn = ortools_flow_shop_cp_sat
        else:
            jobs = normalize_job_shop(raw)
            ortools_fn = ortools_cp_sat

        ortools = ortools_fn(jobs)
        output = dict(ortools)
        output.setdefault(
            "solver",
            "ortools:cp-sat-flow-shop" if kind == "flow-shop" else "ortools:cp-sat",
        )
        output.setdefault("sequence", [])
        output.setdefault("schedule", [])
        output.setdefault("makespan", None)
        output.setdefault("totalFlowTime", None)
        if not rust_reference_embedded():
            reference = rust_reference(raw, kind)
            output["referenceStatus"] = reference.get("status")
            output["referenceMakespan"] = reference.get("makespan")
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "unavailable") else 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "status": "error",
                    "solver": "scheduling-reference",
                    "schedule": [],
                    "makespan": None,
                    "totalFlowTime": None,
                    "message": str(exc),
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
