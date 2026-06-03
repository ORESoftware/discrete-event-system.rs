#!/usr/bin/env python3
"""Reference bridge for small job-shop scheduling instances.

The deterministic oracle is an exact depth-first branch-and-bound over
semi-active schedules. When OR-Tools is installed, the same input is also sent
to CP-SAT with interval variables and no-overlap machine resources so Rust
validation can cross-check against a real CP engine without vendoring solver
executables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


EPS = 1e-9
SCALE = 1000


def normalize(raw: dict) -> list[dict]:
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


def schedule_result(status: str, solver: str, schedule: list[dict], message: str = "") -> dict:
    makespan = max((op["finish"] for op in schedule), default=0.0)
    completions = {}
    for op in schedule:
        completions[op["jobId"]] = max(completions.get(op["jobId"], 0.0), op["finish"])
    return {
        "status": status,
        "solver": solver,
        "schedule": schedule,
        "makespan": float(makespan),
        "totalFlowTime": float(sum(completions.values())),
        "message": message,
    }


def dispatch_spt(jobs: list[dict]) -> dict:
    machine_ready: dict[str, float] = {}
    job_ready = [0.0 for _ in jobs]
    next_ops = [0 for _ in jobs]
    schedule = []
    total_ops = sum(len(job["operations"]) for job in jobs)
    while len(schedule) < total_ops:
        candidates = []
        for job_index, job in enumerate(jobs):
            op_index = next_ops[job_index]
            if op_index >= len(job["operations"]):
                continue
            op = job["operations"][op_index]
            start = max(job_ready[job_index], machine_ready.get(op["machine"], 0.0))
            finish = start + op["duration"]
            candidates.append((op["duration"], finish, start, job_index))
        _, _, start, job_index = min(candidates)
        op_index = next_ops[job_index]
        job = jobs[job_index]
        op = job["operations"][op_index]
        finish = start + op["duration"]
        schedule.append(
            {
                "jobId": job["id"],
                "opIndex": op_index,
                "machine": op["machine"],
                "start": float(start),
                "finish": float(finish),
            }
        )
        machine_ready[op["machine"]] = finish
        job_ready[job_index] = finish
        next_ops[job_index] += 1
    return schedule_result("feasible", "python:spt-dispatch", schedule)


def exact_job_shop(jobs: list[dict]) -> dict:
    total_ops = sum(len(job["operations"]) for job in jobs)
    if total_ops > 20:
        return {
            "status": "unsupported",
            "solver": "python:exact-job-shop",
            "schedule": [],
            "makespan": None,
            "totalFlowTime": None,
            "message": f"exact job-shop only practical for <= 20 operations, got {total_ops}",
        }

    incumbent = dispatch_spt(jobs)
    best_schedule = list(incumbent["schedule"])
    best_makespan = float(incumbent["makespan"])
    best_total_flow = float(incumbent["totalFlowTime"])

    next_ops = [0 for _ in jobs]
    machine_ready: dict[str, float] = {}
    job_ready = [0.0 for _ in jobs]
    schedule: list[dict] = []

    def lower_bound() -> float:
        bound = max(job_ready, default=0.0)
        for job_index, job in enumerate(jobs):
            remaining = sum(op["duration"] for op in job["operations"][next_ops[job_index] :])
            bound = max(bound, job_ready[job_index] + remaining)
        machine_work: dict[str, float] = {}
        for job_index, job in enumerate(jobs):
            for op in job["operations"][next_ops[job_index] :]:
                machine_work[op["machine"]] = machine_work.get(op["machine"], 0.0) + op["duration"]
        for machine, work in machine_work.items():
            bound = max(bound, machine_ready.get(machine, 0.0) + work)
        return bound

    def recurse() -> None:
        nonlocal best_schedule, best_makespan, best_total_flow
        if len(schedule) == total_ops:
            makespan = max(job_ready, default=0.0)
            total_flow = sum(job_ready)
            if makespan < best_makespan - EPS or (
                abs(makespan - best_makespan) <= EPS and total_flow < best_total_flow - EPS
            ):
                best_schedule = [dict(op) for op in schedule]
                best_makespan = makespan
                best_total_flow = total_flow
            return
        if lower_bound() > best_makespan + EPS:
            return

        candidates = []
        for job_index, job in enumerate(jobs):
            op_index = next_ops[job_index]
            if op_index >= len(job["operations"]):
                continue
            op = job["operations"][op_index]
            start = max(job_ready[job_index], machine_ready.get(op["machine"], 0.0))
            finish = start + op["duration"]
            candidates.append((finish, start, job_index))
        candidates.sort()

        for _, start, job_index in candidates:
            op_index = next_ops[job_index]
            job = jobs[job_index]
            op = job["operations"][op_index]
            machine = op["machine"]
            finish = start + op["duration"]
            previous_machine_ready = machine_ready.get(machine)
            previous_job_ready = job_ready[job_index]

            machine_ready[machine] = finish
            job_ready[job_index] = finish
            next_ops[job_index] += 1
            schedule.append(
                {
                    "jobId": job["id"],
                    "opIndex": op_index,
                    "machine": machine,
                    "start": float(start),
                    "finish": float(finish),
                }
            )

            recurse()

            schedule.pop()
            next_ops[job_index] -= 1
            job_ready[job_index] = previous_job_ready
            if previous_machine_ready is None:
                machine_ready.pop(machine, None)
            else:
                machine_ready[machine] = previous_machine_ready

    recurse()
    return {
        "status": "optimal",
        "solver": "python:exact-job-shop",
        "schedule": best_schedule,
        "makespan": float(best_makespan),
        "totalFlowTime": float(best_total_flow),
        "message": "exact job-shop branch-and-bound",
    }


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    args = parser.parse_args()

    try:
        jobs = normalize(json.load(sys.stdin))
        exact = exact_job_shop(jobs)
        if args.solver == "fallback":
            output = dict(exact)
            output["solver"] = "python:exact-job-shop"
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "feasible", "unsupported") else 1

        ortools = ortools_cp_sat(jobs)
        if args.solver == "ortools":
            output = dict(ortools)
            output.setdefault("schedule", [])
            output.setdefault("makespan", None)
            output.setdefault("totalFlowTime", None)
            output["referenceStatus"] = exact.get("status")
            output["referenceMakespan"] = exact.get("makespan")
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "feasible", "unavailable") else 1

        solver = (
            "ortools:cp-sat+python:exact-job-shop"
            if ortools.get("status") != "unavailable"
            else "python:exact-job-shop"
        )
        output = dict(exact)
        output["solver"] = solver
        output["ortoolsStatus"] = ortools.get("status")
        output["ortoolsMakespan"] = ortools.get("makespan")
        output["ortoolsTotalFlowTime"] = ortools.get("totalFlowTime")
        output["ortoolsSchedule"] = ortools.get("schedule", [])
        output["ortoolsMessage"] = ortools.get("message", "")
        output["ortoolsObjectiveBound"] = ortools.get("objectiveBound")
        print(json.dumps(output))
        return 0 if output["status"] in ("optimal", "feasible", "unsupported") else 1
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
