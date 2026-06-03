#!/usr/bin/env python3
"""Reference bridge for simulation validation payloads.

The built-in paths handle deterministic smoke models encoded as JSON:
single-station queues, mobility/transport routes, building energy balance, and
simple robot/physics trajectories, agent-based models, distributed-system
models, and process-flow models. They are intentionally small enough to be
dependency-free while still validating the same output contract used for local
SimPy, SUMO, EnergyPlus, MuJoCo, Mesa, SimGrid, NeqSim, and related adapter
commands.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from typing import Any


EVENT_ENGINES = {
    "simpy",
    "salabim",
    "simmer",
    "jaamsim",
    "anylogic",
    "simio",
    "simul8",
    "arena",
    "flexsim",
    "plant-simulation",
    "extendsim",
    "gpss-world",
    "simulink",
    "ptolemy-ii",
}
MOBILITY_ENGINES = {"ns-3", "ns3", "omnetpp", "omnet++", "sumo", "matsim", "carla"}
ENERGY_ENGINES = {
    "energyplus",
    "openstudio",
    "openmodelica",
    "fmi-fmu",
    "fmi",
    "fmu",
    "omsimulator",
    "simulink",
    "gridlabd",
    "opendss",
    "pandapower",
}
PHYSICS_ENGINES = {
    "gazebo",
    "webots",
    "mujoco",
    "drake",
    "pybullet",
    "carla",
    "isaac-sim",
    "airsim",
}
AGENT_BASED_ENGINES = {"mesa", "repast", "repast-simphony", "mason", "netlogo", "agentpy"}
DISTRIBUTED_SYSTEM_ENGINES = {"simgrid", "cloudsim", "batsim", "gem5", "ptolemy-ii"}
PROCESS_ENGINES = {"neqsim", "dwsim", "cape-open", "copasi", "tellurium"}


def result(
    status: str,
    verdict: str,
    simulator: str,
    message: str = "",
    metrics: dict[str, float] | None = None,
    checks: list[dict[str, Any]] | None = None,
    trace: list[dict[str, Any]] | None = None,
) -> dict:
    return {
        "status": status,
        "verdict": verdict,
        "simulator": simulator,
        "message": message,
        "metrics": metrics or {},
        "checks": checks or [],
        "trace": trace or [],
    }


def finite_float(value: Any, default: float | None = None) -> float:
    if value is None:
        if default is None:
            raise ValueError("expected finite number")
        return default
    out = float(value)
    if not math.isfinite(out):
        raise ValueError("expected finite number")
    return out


def normalize_model(model: dict[str, Any]) -> tuple[int, list[float], list[float]]:
    servers = int(model.get("servers", 1))
    if servers <= 0:
        raise ValueError("servers must be positive")
    arrivals = model.get("arrival_times")
    services = model.get("service_times")
    if arrivals is None or services is None:
        jobs = int(model.get("jobs", 5))
        interarrival = finite_float(model.get("interarrival_time"), 1.0)
        service_time = finite_float(model.get("service_time"), 1.0)
        arrivals = [idx * interarrival for idx in range(jobs)]
        services = [service_time for _ in range(jobs)]
    arrivals = [finite_float(value) for value in arrivals]
    services = [finite_float(value) for value in services]
    if len(arrivals) != len(services):
        raise ValueError("arrival_times and service_times length mismatch")
    if any(service < 0.0 for service in services):
        raise ValueError("service times must be non-negative")
    if any(arrivals[idx] > arrivals[idx + 1] for idx in range(len(arrivals) - 1)):
        raise ValueError("arrival_times must be sorted")
    return servers, arrivals, services


def simulate_single_station(model: dict[str, Any]) -> dict[str, Any]:
    servers, arrivals, services = normalize_model(model)
    available_at = [0.0 for _ in range(servers)]
    jobs = []
    trace = []
    for job_id, (arrival, service) in enumerate(zip(arrivals, services)):
        server = min(range(servers), key=lambda idx: available_at[idx])
        start = max(arrival, available_at[server])
        departure = start + service
        available_at[server] = departure
        wait = start - arrival
        jobs.append(
            {
                "job": job_id,
                "server": server,
                "arrival": arrival,
                "start": start,
                "departure": departure,
                "wait": wait,
                "service": service,
            }
        )
        trace.append({"time": arrival, "event": "arrival", "job": job_id})
        trace.append({"time": start, "event": "service_start", "job": job_id, "server": server})
        trace.append({"time": departure, "event": "departure", "job": job_id, "server": server})
    trace.sort(key=lambda event: (event["time"], event["event"], event["job"]))
    waits = [job["wait"] for job in jobs]
    sojourns = [job["departure"] - job["arrival"] for job in jobs]
    metrics = {
        "jobs_completed": float(len(jobs)),
        "mean_wait": sum(waits) / len(waits) if waits else 0.0,
        "max_wait": max(waits) if waits else 0.0,
        "mean_sojourn": sum(sojourns) / len(sojourns) if sojourns else 0.0,
        "makespan": max((job["departure"] for job in jobs), default=0.0),
        "utilization_lower_bound": sum(services) / (servers * max(max(available_at), 1.0)),
    }
    return {"jobs": jobs, "trace": trace, "metrics": metrics}


def check_trace_property(name: str, simulation: dict[str, Any]) -> dict[str, Any]:
    jobs = simulation["jobs"]
    if name == "queue_length_never_negative":
        passed = all(job["wait"] >= -1e-9 for job in jobs)
    elif name == "departures_after_arrivals":
        passed = all(job["departure"] + 1e-9 >= job["arrival"] for job in jobs)
    elif name == "service_starts_after_arrivals":
        passed = all(job["start"] + 1e-9 >= job["arrival"] for job in jobs)
    elif name == "single_station_fcfs":
        starts = [job["start"] for job in jobs]
        passed = all(starts[idx] <= starts[idx + 1] + 1e-9 for idx in range(len(starts) - 1))
    else:
        return {"name": name, "passed": False, "message": "unknown trace property"}
    return {"name": name, "passed": bool(passed), "message": ""}


def simulate_mobility(model: dict[str, Any]) -> dict[str, Any]:
    routes = model.get("routes")
    if not isinstance(routes, list) or not routes:
        raise ValueError("mobility model needs non-empty routes")
    vehicles = []
    trace = []
    for idx, route in enumerate(routes):
        depart = finite_float(route.get("depart"), 0.0)
        segments = route.get("segments", route.get("travel_times", []))
        if not isinstance(segments, list) or not segments:
            raise ValueError("each mobility route needs segments or travel_times")
        travel_time = 0.0
        time = depart
        trace.append({"time": time, "event": "vehicle_depart", "vehicle": idx})
        for segment_idx, segment in enumerate(segments):
            if isinstance(segment, dict):
                segment_time = finite_float(segment.get("travel_time"), 0.0)
            else:
                segment_time = finite_float(segment)
            if segment_time < 0.0:
                raise ValueError("travel times must be non-negative")
            travel_time += segment_time
            time += segment_time
            trace.append(
                {
                    "time": time,
                    "event": "segment_arrive",
                    "vehicle": idx,
                    "segment": segment_idx,
                }
            )
        vehicles.append(
            {
                "vehicle": idx,
                "depart": depart,
                "arrival": time,
                "travel_time": travel_time,
            }
        )
        trace.append({"time": time, "event": "vehicle_arrive", "vehicle": idx})
    trace.sort(key=lambda event: (event["time"], event["event"], event["vehicle"]))
    travel_times = [vehicle["travel_time"] for vehicle in vehicles]
    metrics = {
        "vehicles_completed": float(len(vehicles)),
        "mean_travel_time": sum(travel_times) / len(travel_times),
        "max_travel_time": max(travel_times),
        "min_travel_time": min(travel_times),
        "last_arrival": max(vehicle["arrival"] for vehicle in vehicles),
    }
    return {"vehicles": vehicles, "trace": trace, "metrics": metrics}


def check_mobility_property(name: str, simulation: dict[str, Any]) -> dict[str, Any]:
    vehicles = simulation["vehicles"]
    if name == "departures_before_arrivals":
        passed = all(vehicle["arrival"] + 1e-9 >= vehicle["depart"] for vehicle in vehicles)
    elif name == "travel_times_nonnegative":
        passed = all(vehicle["travel_time"] >= -1e-9 for vehicle in vehicles)
    elif name == "vehicles_complete":
        passed = all(math.isfinite(vehicle["arrival"]) for vehicle in vehicles)
    else:
        return {"name": name, "passed": False, "message": "unknown mobility property"}
    return {"name": name, "passed": bool(passed), "message": ""}


def simulate_energy_balance(model: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    zones = model.get("zones")
    if not isinstance(zones, list) or not zones:
        zones = [model]
    horizon = finite_float(scenario.get("horizon"), finite_float(model.get("horizon"), 4.0))
    step = finite_float(scenario.get("step"), finite_float(model.get("step"), 1.0))
    if horizon <= 0.0 or step <= 0.0:
        raise ValueError("energy horizon and step must be positive")
    steps = max(1, int(round(horizon / step)))
    trace = []
    final_errors = []
    energy_kwh = 0.0
    min_temp = math.inf
    max_temp = -math.inf
    for zone_idx, zone in enumerate(zones):
        temp = finite_float(zone.get("initial_temp"), 20.0)
        setpoint = finite_float(zone.get("setpoint"), 21.0)
        outdoor = finite_float(zone.get("outdoor_temp"), 10.0)
        ua = finite_float(zone.get("ua"), 0.2)
        heat_capacity = finite_float(zone.get("heat_capacity"), 5.0)
        hvac_power = finite_float(zone.get("hvac_power"), 4.0)
        internal_gain = finite_float(zone.get("internal_gain"), 0.0)
        if heat_capacity <= 0.0:
            raise ValueError("heat_capacity must be positive")
        for step_idx in range(steps):
            error = setpoint - temp
            hvac = max(-hvac_power, min(hvac_power, error * hvac_power))
            temp += ((ua * (outdoor - temp)) + internal_gain + hvac) * step / heat_capacity
            energy_kwh += abs(hvac) * step
            min_temp = min(min_temp, temp)
            max_temp = max(max_temp, temp)
            trace.append(
                {
                    "time": (step_idx + 1) * step,
                    "event": "zone_temperature",
                    "zone": zone_idx,
                    "temperature": temp,
                    "hvac": hvac,
                }
            )
        final_errors.append(abs(temp - setpoint))
    metrics = {
        "energy_kwh": energy_kwh,
        "max_abs_setpoint_error": max(final_errors) if final_errors else 0.0,
        "min_temperature": min_temp if math.isfinite(min_temp) else 0.0,
        "max_temperature": max_temp if math.isfinite(max_temp) else 0.0,
        "zones": float(len(zones)),
    }
    return {"trace": trace, "metrics": metrics}


def check_energy_property(name: str, simulation: dict[str, Any]) -> dict[str, Any]:
    metrics = simulation["metrics"]
    trace = simulation["trace"]
    if name == "energy_nonnegative":
        passed = metrics["energy_kwh"] >= -1e-9
    elif name == "temperatures_finite":
        passed = all(math.isfinite(event.get("temperature", 0.0)) for event in trace)
    elif name == "temperature_within_bounds":
        passed = metrics["min_temperature"] >= -100.0 and metrics["max_temperature"] <= 100.0
    else:
        return {"name": name, "passed": False, "message": "unknown energy property"}
    return {"name": name, "passed": bool(passed), "message": ""}


def simulate_physics_trajectory(model: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    dt = finite_float(scenario.get("dt"), finite_float(model.get("dt"), 0.1))
    steps = int(scenario.get("steps", model.get("steps", 10)))
    if dt <= 0.0 or steps <= 0:
        raise ValueError("trajectory dt and steps must be positive")
    position = finite_float(model.get("initial_position"), 0.0)
    velocity = finite_float(model.get("initial_velocity"), 0.0)
    acceleration = finite_float(model.get("acceleration"), finite_float(model.get("acceleration_command"), 0.0))
    floor = finite_float(model.get("floor"), -math.inf)
    trace = [{"time": 0.0, "event": "state", "position": position, "velocity": velocity}]
    positions = [position]
    path_length = 0.0
    for step_idx in range(steps):
        previous = position
        velocity += acceleration * dt
        position += velocity * dt
        if position < floor:
            position = floor
            velocity = max(0.0, velocity)
        path_length += abs(position - previous)
        positions.append(position)
        trace.append(
            {
                "time": (step_idx + 1) * dt,
                "event": "state",
                "position": position,
                "velocity": velocity,
            }
        )
    metrics = {
        "final_position": position,
        "final_velocity": velocity,
        "max_position": max(positions),
        "min_position": min(positions),
        "path_length": path_length,
    }
    return {"trace": trace, "metrics": metrics}


def check_physics_property(name: str, simulation: dict[str, Any]) -> dict[str, Any]:
    trace = simulation["trace"]
    if name == "positions_finite":
        passed = all(math.isfinite(event.get("position", 0.0)) for event in trace)
    elif name == "velocities_finite":
        passed = all(math.isfinite(event.get("velocity", 0.0)) for event in trace)
    elif name == "path_length_nonnegative":
        passed = simulation["metrics"]["path_length"] >= -1e-9
    elif name == "stays_above_floor":
        floor = min(event.get("position", 0.0) for event in trace)
        passed = floor >= -1e-9
    else:
        return {"name": name, "passed": False, "message": "unknown physics property"}
    return {"name": name, "passed": bool(passed), "message": ""}


def simulate_agent_based(model: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    agents = model.get("agents", [])
    if not isinstance(agents, list) or not agents:
        raise ValueError("agent-based model needs non-empty agents")
    steps = int(scenario.get("steps", model.get("steps", 1)))
    if steps < 0:
        raise ValueError("agent-based steps must be non-negative")
    interactions = model.get("interactions", model.get("edges", []))
    if interactions is None:
        interactions = []
    if not isinstance(interactions, list):
        raise ValueError("agent-based interactions must be an array")
    trace = []
    for step_idx in range(steps + 1):
        trace.append({"time": float(step_idx), "event": "step", "agents": len(agents)})
    stateful_agents = sum(1 for agent in agents if isinstance(agent, dict) and bool(agent.get("state", agent.get("type", ""))))
    metrics = {
        "agents": float(len(agents)),
        "steps": float(steps),
        "interactions": float(len(interactions)),
        "stateful_agents": float(stateful_agents),
    }
    return {"trace": trace, "metrics": metrics, "agents": agents, "interactions": interactions}


def check_agent_property(name: str, simulation: dict[str, Any]) -> dict[str, Any]:
    metrics = simulation["metrics"]
    if name == "agents_nonempty":
        passed = metrics["agents"] > 0.0
    elif name == "states_present":
        passed = metrics["stateful_agents"] == metrics["agents"]
    elif name == "steps_nonnegative":
        passed = metrics["steps"] >= 0.0
    elif name == "interactions_reference_agents":
        count = int(metrics["agents"])
        passed = True
        for edge in simulation["interactions"]:
            if not isinstance(edge, dict):
                passed = False
                break
            src = int(edge.get("source", edge.get("from", -1)))
            dst = int(edge.get("target", edge.get("to", -1)))
            if src < 0 or dst < 0 or src >= count or dst >= count:
                passed = False
                break
    else:
        return {"name": name, "passed": False, "message": "unknown agent-based property"}
    return {"name": name, "passed": bool(passed), "message": ""}


def simulate_distributed_system(model: dict[str, Any]) -> dict[str, Any]:
    hosts = model.get("hosts", [])
    links = model.get("links", [])
    tasks = model.get("tasks", model.get("workloads", []))
    if not isinstance(hosts, list) or not hosts:
        raise ValueError("distributed-system model needs non-empty hosts")
    if not isinstance(links, list):
        raise ValueError("distributed-system links must be an array")
    if not isinstance(tasks, list):
        raise ValueError("distributed-system tasks/workloads must be an array")
    total_capacity = 0.0
    min_bandwidth = math.inf
    total_work = 0.0
    for host in hosts:
        if not isinstance(host, dict):
            raise ValueError("distributed-system host must be an object")
        total_capacity += finite_float(host.get("capacity"), finite_float(host.get("cores"), 1.0))
    for link in links:
        if not isinstance(link, dict):
            raise ValueError("distributed-system link must be an object")
        min_bandwidth = min(min_bandwidth, finite_float(link.get("bandwidth"), 0.0))
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("distributed-system task must be an object")
        total_work += finite_float(task.get("work"), finite_float(task.get("duration"), 0.0))
    metrics = {
        "hosts": float(len(hosts)),
        "links": float(len(links)),
        "tasks": float(len(tasks)),
        "total_capacity": total_capacity,
        "min_bandwidth": min_bandwidth if math.isfinite(min_bandwidth) else 0.0,
        "total_work": total_work,
    }
    trace = [{"time": 0.0, "event": "distributed_model_loaded", "hosts": len(hosts), "tasks": len(tasks)}]
    return {"trace": trace, "metrics": metrics}


def check_distributed_property(name: str, simulation: dict[str, Any]) -> dict[str, Any]:
    metrics = simulation["metrics"]
    if name == "hosts_have_capacity":
        passed = metrics["total_capacity"] > 0.0
    elif name == "links_nonnegative":
        passed = metrics["min_bandwidth"] >= 0.0
    elif name == "tasks_schedulable":
        passed = metrics["tasks"] == 0.0 or metrics["total_capacity"] > 0.0
    else:
        return {"name": name, "passed": False, "message": "unknown distributed-system property"}
    return {"name": name, "passed": bool(passed), "message": ""}


def simulate_process_flow(model: dict[str, Any]) -> dict[str, Any]:
    units = model.get("units", [])
    streams = model.get("streams", [])
    if not isinstance(units, list) or not units:
        raise ValueError("process-flow model needs non-empty units")
    if not isinstance(streams, list):
        raise ValueError("process-flow streams must be an array")
    inlet = 0.0
    outlet = 0.0
    min_flow = math.inf
    for stream in streams:
        if not isinstance(stream, dict):
            raise ValueError("process-flow stream must be an object")
        flow = finite_float(stream.get("flow", stream.get("mass_flow", 0.0)))
        min_flow = min(min_flow, flow)
        if stream.get("to") in (None, "", "sink"):
            outlet += flow
        if stream.get("from") in (None, "", "source"):
            inlet += flow
    metrics = {
        "units": float(len(units)),
        "streams": float(len(streams)),
        "inlet_flow": inlet,
        "outlet_flow": outlet,
        "mass_balance_error": abs(inlet - outlet),
        "min_stream_flow": min_flow if math.isfinite(min_flow) else 0.0,
    }
    trace = [{"time": 0.0, "event": "process_model_loaded", "units": len(units), "streams": len(streams)}]
    return {"trace": trace, "metrics": metrics}


def check_process_property(name: str, simulation: dict[str, Any]) -> dict[str, Any]:
    metrics = simulation["metrics"]
    if name == "units_present":
        passed = metrics["units"] > 0.0
    elif name == "streams_nonnegative":
        passed = metrics["min_stream_flow"] >= -1e-9
    elif name == "mass_balance_closed":
        passed = metrics["mass_balance_error"] <= 1e-9
    else:
        return {"name": name, "passed": False, "message": "unknown process-flow property"}
    return {"name": name, "passed": bool(passed), "message": ""}


def check_metric(expectation: dict[str, Any], metrics: dict[str, float]) -> dict[str, Any]:
    name = str(expectation.get("name", ""))
    comparison = str(expectation.get("comparison", "within-absolute"))
    target = finite_float(expectation.get("target"), 0.0)
    tolerance = abs(finite_float(expectation.get("tolerance"), 0.0))
    actual = metrics.get(name)
    if actual is None:
        return {"name": name, "passed": False, "actual": None, "target": target, "message": "metric missing"}
    if comparison == "within-absolute":
        passed = abs(actual - target) <= tolerance
    elif comparison in ("less-equal", "at-most", "<="):
        passed = actual <= target + tolerance
    elif comparison in ("greater-equal", "at-least", ">="):
        passed = actual + tolerance >= target
    elif comparison in ("equal", "=="):
        passed = abs(actual - target) <= tolerance
    else:
        return {
            "name": name,
            "passed": False,
            "actual": actual,
            "target": target,
            "message": f"unknown comparison {comparison!r}",
        }
    return {
        "name": name,
        "passed": bool(passed),
        "actual": actual,
        "target": target,
        "tolerance": tolerance,
        "comparison": comparison,
        "message": "",
    }


def validate_simulation(payload: dict[str, Any], engine_override: str | None = None) -> dict[str, Any]:
    engine = (engine_override or str(payload.get("engine") or payload.get("engine_id") or "builtin")).lower()
    model_format = str(payload.get("model_format", "json-event-network"))
    model = payload.get("model", {})
    scenario = payload.get("scenario") or {}
    if not isinstance(model, dict):
        raise ValueError("simulation model must be an object")
    if not isinstance(scenario, dict):
        scenario = {}
    simulator = f"builtin:{model_format}"
    if model_format == "json-event-network":
        simulation = simulate_single_station(model)
        property_checker = check_trace_property
        simulator = "builtin:single-station-des"
        if engine in EVENT_ENGINES:
            simulator = f"builtin:single-station-des-for-{engine}"
    elif model_format == "json-mobility-network":
        simulation = simulate_mobility(model)
        property_checker = check_mobility_property
        if engine not in MOBILITY_ENGINES:
            simulator = "builtin:mobility-network"
        else:
            simulator = f"builtin:mobility-network-for-{engine}"
    elif model_format == "json-energy-balance":
        simulation = simulate_energy_balance(model, scenario)
        property_checker = check_energy_property
        if engine not in ENERGY_ENGINES:
            simulator = "builtin:energy-balance"
        else:
            simulator = f"builtin:energy-balance-for-{engine}"
    elif model_format == "json-physics-trajectory":
        simulation = simulate_physics_trajectory(model, scenario)
        property_checker = check_physics_property
        if engine not in PHYSICS_ENGINES:
            simulator = "builtin:physics-trajectory"
        else:
            simulator = f"builtin:physics-trajectory-for-{engine}"
    elif model_format == "json-agent-based":
        simulation = simulate_agent_based(model, scenario)
        property_checker = check_agent_property
        if engine not in AGENT_BASED_ENGINES:
            simulator = "builtin:agent-based"
        else:
            simulator = f"builtin:agent-based-for-{engine}"
    elif model_format == "json-distributed-system":
        simulation = simulate_distributed_system(model)
        property_checker = check_distributed_property
        if engine not in DISTRIBUTED_SYSTEM_ENGINES:
            simulator = "builtin:distributed-system"
        else:
            simulator = f"builtin:distributed-system-for-{engine}"
    elif model_format == "json-process-flow":
        simulation = simulate_process_flow(model)
        property_checker = check_process_property
        if engine not in PROCESS_ENGINES:
            simulator = "builtin:process-flow"
        else:
            simulator = f"builtin:process-flow-for-{engine}"
    else:
        return result("unavailable", "unknown", engine, f"unsupported model_format {model_format!r}")
    checks = []
    for name in payload.get("expected_trace_properties", []):
        checks.append(property_checker(str(name), simulation))
    for expectation in payload.get("metric_expectations", []):
        checks.append(check_metric(expectation, simulation["metrics"]))
    verdict = "valid" if all(check["passed"] for check in checks) else "invalid"
    return result(
        "ok",
        verdict,
        simulator,
        "",
        simulation["metrics"],
        checks,
        simulation["trace"],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine")
    args = parser.parse_args()
    try:
        payload = json.load(sys.stdin)
        print(json.dumps(validate_simulation(payload, args.engine)))
    except Exception as exc:
        print(json.dumps(result("failed", "failure", args.engine or "simulation", str(exc))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
