#!/usr/bin/env python3
"""Reference bridge for small scheduling instances.

For job-shop inputs, the deterministic oracle is an exact depth-first
branch-and-bound over semi-active schedules. For flow-shop inputs, the oracle
enumerates small permutation schedules. When OR-Tools is installed, the same
input is also sent to CP-SAT so Rust validation can cross-check against a real
CP engine without vendoring solver executables.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Optional


EPS = 1e-9
SCALE = 1000


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


def exact_flow_shop(jobs: list[dict]) -> dict:
    if len(jobs) > 10:
        return {
            "status": "unsupported",
            "solver": "python:exact-flow-shop",
            "sequence": [],
            "schedule": [],
            "makespan": None,
            "totalFlowTime": None,
            "message": f"exact flow-shop only practical for <= 10 jobs, got {len(jobs)}",
        }

    best_sequence: list[dict] = []
    best_schedule: list[dict] = []
    best_makespan = math.inf
    best_total_flow = math.inf
    used = [False for _ in jobs]
    current: list[dict] = []

    def consider(sequence: list[dict]) -> None:
        nonlocal best_sequence, best_schedule, best_makespan, best_total_flow
        schedule = flow_shop_schedule(sequence)
        result = schedule_result("optimal", "python:exact-flow-shop", schedule)
        makespan = float(result["makespan"])
        total_flow = float(result["totalFlowTime"])
        seq_ids = [job["id"] for job in sequence]
        best_ids = [job["id"] for job in best_sequence]
        if makespan < best_makespan - EPS or (
            abs(makespan - best_makespan) <= EPS
            and (total_flow < best_total_flow - EPS or (abs(total_flow - best_total_flow) <= EPS and seq_ids < best_ids))
        ):
            best_sequence = [dict(job) for job in sequence]
            best_schedule = schedule
            best_makespan = makespan
            best_total_flow = total_flow

    def recurse() -> None:
        if len(current) == len(jobs):
            consider(current)
            return
        for index, job in enumerate(jobs):
            if used[index]:
                continue
            used[index] = True
            current.append(job)
            recurse()
            current.pop()
            used[index] = False

    recurse()
    return schedule_result(
        "optimal",
        "python:exact-flow-shop",
        best_schedule,
        "exact permutation flow-shop enumeration",
        [job["id"] for job in best_sequence],
    )


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["auto", "ortools", "fallback"], default="auto")
    parser.add_argument("--kind", choices=["auto", "job-shop", "flow-shop"], default="auto")
    args = parser.parse_args()

    try:
        raw = json.load(sys.stdin)
        kind = infer_kind(raw, args.kind)
        if kind == "flow-shop":
            jobs = normalize_flow_shop(raw)
            exact = exact_flow_shop(jobs)
            ortools_fn = ortools_flow_shop_cp_sat
            fallback_solver = "python:exact-flow-shop"
            combined_solver = "ortools:cp-sat-flow-shop+python:exact-flow-shop"
        else:
            jobs = normalize_job_shop(raw)
            exact = exact_job_shop(jobs)
            ortools_fn = ortools_cp_sat
            fallback_solver = "python:exact-job-shop"
            combined_solver = "ortools:cp-sat+python:exact-job-shop"
        if args.solver == "fallback":
            output = dict(exact)
            output["solver"] = fallback_solver
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "feasible", "unsupported") else 1

        ortools = ortools_fn(jobs)
        if args.solver == "ortools":
            output = dict(ortools)
            output.setdefault("sequence", [])
            output.setdefault("schedule", [])
            output.setdefault("makespan", None)
            output.setdefault("totalFlowTime", None)
            output["referenceStatus"] = exact.get("status")
            output["referenceMakespan"] = exact.get("makespan")
            print(json.dumps(output))
            return 0 if output["status"] in ("optimal", "feasible", "unavailable") else 1

        solver = combined_solver if ortools.get("status") != "unavailable" else fallback_solver
        output = dict(exact)
        output["solver"] = solver
        output["ortoolsStatus"] = ortools.get("status")
        output["ortoolsSequence"] = ortools.get("sequence", [])
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
