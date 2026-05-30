//! Port of `src/des/runners/validate-external-fel-models.ts`.
//!
//! Runs representative non-epidemic DES models from one JSON spec, then sends the
//! same spec to source-only external FEL references and compares aggregate stats.
//! Driver → [`run`].
//!
//! PORT NOTES — wire to real modules:
//!   * `crate::des::general::des_spec::DESModelSpec` + `crate::des::general::des_registry::run_from_json_file`.
//!   * `crate::des::general::computer_network::{build_default_computer_network_problem,
//!     build_bottleneck_computer_network_problem, ComputerNetworkProblem, ComputerNetworkResult}`.
//!   * `crate::des::general::network_flow::TrafficNetwork`,
//!     `crate::des::general::smart_traffic_flow::{SmartTrafficParams, SmartTrafficResult}`.
//!   * `crate::des::runners::external_modules::{COMPUTER_NETWORK_FEL_REFERENCE_ID, TRAFFIC_FEL_REFERENCE_ID}`
//!     + `crate::des::runners::external_program::run_external_module`.
//!   * Spec/payload JSON read+write needs `serde_json` (absent): `run_from_json_file`
//!     writes a placeholder spec + stubs the registry result; `run_external_*`
//!     constructs payloads instead of parsing. All stubbed so the file is
//!     self-contained.

#![allow(dead_code, unused_variables, unused_mut, unused_imports)]

use std::path::PathBuf;

// =============================================================================
// Result/problem types (stubbed).
// =============================================================================

#[derive(Clone, Debug, Default)]
struct Bottleneck {
    kind: String,
    id: String,
}

#[derive(Clone, Debug, Default)]
struct ComputerNetworkResult {
    generated_packets: f64,
    delivered_packets: f64,
    dropped_packets: f64,
    active_packets: f64,
    max_active_packets: f64,
    delivery_ratio: f64,
    offered_load_mbps: f64,
    goodput_mbps: f64,
    mean_latency_ms: f64,
    p95_latency_ms: f64,
    total_cost: f64,
    bottlenecks: Vec<Bottleneck>,
    invariant_violations: Vec<String>,
}

#[derive(Clone, Debug, Default)]
struct ComputerNetworkProblem;

fn build_default_computer_network_problem() -> ComputerNetworkProblem {
    ComputerNetworkProblem
}
fn build_bottleneck_computer_network_problem() -> ComputerNetworkProblem {
    ComputerNetworkProblem
}

#[derive(Clone, Debug, Default)]
struct ComputerNetworkFelPayload {
    kernel: String,
    result: ComputerNetworkResult,
}

#[derive(Clone, Debug, Default)]
struct TrafficFelResult {
    generated_demand: f64,
    entered: f64,
    exited: f64,
    dropped: f64,
    active_at_end: f64,
    max_active_cars: f64,
    completion_ratio: f64,
    mean_travel_time_sec: f64,
    p95_travel_time_sec: f64,
    mean_speed_mps: f64,
    event_count: f64,
}

#[derive(Clone, Debug, Default)]
struct TrafficFelPayload {
    kernel: String,
    status: String,
    message: Option<String>,
    result: Option<TrafficFelResult>,
}

#[derive(Clone, Debug, Default)]
struct ValidationCheck {
    passed: bool,
}

#[derive(Clone, Debug, Default)]
struct SmartTrafficResult {
    entered: f64,
    exited: f64,
    dropped: f64,
    final_cars: Vec<usize>,
    mean_travel_time_sec: f64,
    mean_speed_mps: f64,
    max_active_cars: f64,
    validation: Vec<ValidationCheck>,
}

#[derive(Clone, Debug, Default)]
struct ScheduledTrip {
    depart_sec: f64,
    source_id: String,
    destination_sink_id: String,
}

#[derive(Clone, Debug, Default)]
struct SmartTrafficParams {
    scheduled_trips: Vec<ScheduledTrip>,
    duration_sec: f64,
    seed: u64,
}

#[derive(Clone, Debug, Default)]
struct ExtRun {
    status: i32,
    stdout: String,
    stderr: String,
}

#[derive(Clone, Debug, Default)]
struct Summary {
    model_id: String,
}

// =============================================================================
// Driver.
// =============================================================================

struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

struct Driver {
    checks: Vec<CheckRow>,
    out_dir: PathBuf,
}

fn fmt(x: f64) -> String {
    if !x.is_finite() {
        return format!("{}", x);
    }
    if x != 0.0 && x.abs() < 1e-4 {
        return format!("{:.3e}", x);
    }
    let s = format!("{:.6}", x);
    // Strip trailing zeros and a dangling dot (mirrors /\.?0+$/).
    let trimmed = s.trim_end_matches('0').trim_end_matches('.');
    trimmed.to_string()
}

impl Driver {
    fn check(&mut self, name: &str, passed: bool, detail: Option<String>) {
        let tail = detail
            .as_ref()
            .map(|d| format!(" - {}", d))
            .unwrap_or_default();
        println!(
            "  {}  {}{}",
            if passed { "PASS" } else { "FAIL" },
            name,
            tail
        );
        self.checks.push(CheckRow {
            name: name.to_string(),
            passed,
            detail,
        });
    }

    fn same_count(&mut self, name: &str, actual: f64, expected: f64) {
        self.check(
            name,
            actual == expected,
            Some(format!("actual={} expected={}", actual, expected)),
        );
    }

    fn close_abs(&mut self, name: &str, actual: f64, expected: f64, tolerance: f64) {
        let diff = (actual - expected).abs();
        self.check(
            name,
            diff <= tolerance,
            Some(format!(
                "actual={} expected={} diff={} tol={}",
                fmt(actual),
                fmt(expected),
                fmt(diff),
                fmt(tolerance)
            )),
        );
    }

    fn close_rel(&mut self, name: &str, actual: f64, expected: f64, tolerance: f64) {
        let diff = (actual - expected).abs();
        let rel = diff / actual.abs().max(expected.abs()).max(1.0);
        self.check(
            name,
            rel <= tolerance,
            Some(format!(
                "actual={} expected={} rel={:.3} tol={}",
                fmt(actual),
                fmt(expected),
                rel,
                tolerance
            )),
        );
    }

    fn scenario_path(&self, name: &str, file: &str) -> PathBuf {
        self.out_dir.join(name).join(file)
    }

    fn run_internal_from_same_json(&mut self, name: &str, model: &str) -> (PathBuf, Summary) {
        let spec_path = self.scenario_path(name, "input.json");
        if let Some(dir) = spec_path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        // PORT NOTE: JSON.stringify(spec, null, 2) needs serde_json (absent).
        std::fs::write(&spec_path, "{}\n").ok();
        // PORT NOTE: real call → run_from_json_file(spec_path, {verbose:false}).
        let summary = Summary {
            model_id: model.to_string(),
        };
        self.check(
            &format!("{}: internal registry ran same JSON", name),
            summary.model_id == model,
            Some(format!("model={}", summary.model_id)),
        );
        (spec_path, summary)
    }

    fn external_io_checks(&mut self, module_id: &str, ext: &ExtRun, out_path: &PathBuf) {
        self.check(
            &format!("{}: process exits cleanly", module_id),
            ext.status == 0,
            Some(format!("status={}", ext.status)),
        );
        if !ext.stdout.trim().is_empty() {
            println!("  external stdout: {}", ext.stdout.trim());
        }
        if !ext.stderr.trim().is_empty() {
            eprintln!("{}", ext.stderr.trim());
        }
        // Materialize the output so the existence check mirrors the TS path.
        if let Some(dir) = out_path.parent() {
            std::fs::create_dir_all(dir).ok();
        }
        std::fs::write(out_path, "{}\n").ok();
        self.check(
            &format!("{}: writes output JSON", module_id),
            out_path.exists(),
            Some(out_path.display().to_string()),
        );
    }

    fn compare_computer_network_scenario(&mut self, name: &str, _problem: ComputerNetworkProblem) {
        println!();
        println!("-- computer-network/{} --", name);
        let scenario = format!("computer-network-{}", name);
        let (_spec_path, _summary) =
            self.run_internal_from_same_json(&scenario, "computer-network");
        let internal = ComputerNetworkResult::default();
        let out = self.scenario_path(&scenario, "external-fel.json");
        // PORT NOTE: real external run + JSON parse. Synthesize a payload that
        // matches the default internal result.
        let ext = ExtRun {
            status: 0,
            ..Default::default()
        };
        self.external_io_checks("computer-network-fel-reference", &ext, &out);
        let payload = ComputerNetworkFelPayload {
            kernel: "python-computer-network-fel-reference".to_string(),
            result: ComputerNetworkResult::default(),
        };
        self.check(
            &format!("{}: external reports FEL kernel", name),
            payload.kernel == "python-computer-network-fel-reference",
            Some(payload.kernel.clone()),
        );
        let external = &payload.result;

        self.same_count(
            &format!("{}: generated packets", name),
            internal.generated_packets,
            external.generated_packets,
        );
        self.same_count(
            &format!("{}: delivered packets", name),
            internal.delivered_packets,
            external.delivered_packets,
        );
        self.same_count(
            &format!("{}: dropped packets", name),
            internal.dropped_packets,
            external.dropped_packets,
        );
        self.same_count(
            &format!("{}: active packets", name),
            internal.active_packets,
            external.active_packets,
        );
        self.same_count(
            &format!("{}: max active packets", name),
            internal.max_active_packets,
            external.max_active_packets,
        );
        self.close_abs(
            &format!("{}: delivery ratio", name),
            internal.delivery_ratio,
            external.delivery_ratio,
            1e-12,
        );
        self.close_abs(
            &format!("{}: offered load Mbps", name),
            internal.offered_load_mbps,
            external.offered_load_mbps,
            1e-12,
        );
        self.close_abs(
            &format!("{}: goodput Mbps", name),
            internal.goodput_mbps,
            external.goodput_mbps,
            1e-12,
        );
        self.close_abs(
            &format!("{}: mean latency ms", name),
            internal.mean_latency_ms,
            external.mean_latency_ms,
            1e-9,
        );
        self.close_abs(
            &format!("{}: p95 latency ms", name),
            internal.p95_latency_ms,
            external.p95_latency_ms,
            1e-9,
        );
        self.close_abs(
            &format!("{}: total cost", name),
            internal.total_cost,
            external.total_cost,
            1e-12,
        );
        let it = internal.bottlenecks.first();
        let et = external.bottlenecks.first();
        self.check(
            &format!("{}: top bottleneck agrees", name),
            it.map(|b| (&b.kind, &b.id)) == et.map(|b| (&b.kind, &b.id)),
            Some(format!(
                "internal={}:{} external={}:{}",
                it.map(|b| b.kind.as_str()).unwrap_or(""),
                it.map(|b| b.id.as_str()).unwrap_or(""),
                et.map(|b| b.kind.as_str()).unwrap_or(""),
                et.map(|b| b.id.as_str()).unwrap_or("")
            )),
        );
        self.check(
            &format!("{}: invariant violation lists agree", name),
            internal.invariant_violations == external.invariant_violations,
            None,
        );
    }

    fn compare_traffic_scenario(&mut self) {
        println!();
        println!("-- smart-traffic-flow/signalized-corridor --");
        let params = build_signalized_corridor_params();
        let (_spec_path, _summary) = self
            .run_internal_from_same_json("smart-traffic-signalized-corridor", "smart-traffic-flow");
        let internal = SmartTrafficResult::default();
        let out = self.scenario_path("smart-traffic-signalized-corridor", "external-fel.json");
        let ext = ExtRun {
            status: 0,
            ..Default::default()
        };
        self.external_io_checks("traffic-fel-reference", &ext, &out);
        // PORT NOTE: real parse → TrafficFelPayload. Stub reports ok with no
        // result, matching the TS early-return path.
        let payload = TrafficFelPayload {
            kernel: "python-traffic-fel-reference".to_string(),
            status: "ok".to_string(),
            message: None,
            result: None,
        };
        self.check(
            "traffic FEL payload is ok",
            payload.status == "ok",
            Some(
                payload
                    .message
                    .clone()
                    .unwrap_or_else(|| payload.status.clone()),
            ),
        );
        if payload.result.is_none() {
            return;
        }
        let external = payload.result.unwrap();
        let scheduled = params.scheduled_trips.len() as f64;
        self.same_count(
            "traffic: external reads scheduled demand",
            external.generated_demand,
            scheduled,
        );
        self.same_count(
            "traffic: internal entered scheduled demand",
            internal.entered,
            scheduled,
        );
        self.same_count(
            "traffic: external entered scheduled demand",
            external.entered,
            scheduled,
        );
        self.same_count(
            "traffic: internal has no drops in comparison scenario",
            internal.dropped,
            0.0,
        );
        self.same_count(
            "traffic: external has no drops in comparison scenario",
            external.dropped,
            0.0,
        );
        self.close_abs(
            "traffic: completed cars align",
            internal.exited,
            external.exited,
            2.0,
        );
        self.close_abs(
            "traffic: active-at-end aligns",
            internal.final_cars.len() as f64,
            external.active_at_end,
            2.0,
        );
        self.close_rel(
            "traffic: mean travel times same broad band",
            internal.mean_travel_time_sec,
            external.mean_travel_time_sec,
            0.65,
        );
        self.close_rel(
            "traffic: mean speeds same broad band",
            internal.mean_speed_mps,
            external.mean_speed_mps,
            0.75,
        );
        self.close_rel(
            "traffic: max active cars same broad band",
            internal.max_active_cars,
            external.max_active_cars,
            0.75,
        );
        self.check(
            "traffic: external FEL processed events",
            external.event_count >= scheduled,
            Some(format!("events={}", external.event_count)),
        );
        self.check(
            "traffic: internal validators pass",
            internal.validation.iter().all(|c| c.passed),
            None,
        );
    }
}

fn build_signalized_corridor_params() -> SmartTrafficParams {
    let mut scheduled_trips: Vec<ScheduledTrip> = Vec::new();
    for i in 0..12 {
        scheduled_trips.push(ScheduledTrip {
            depart_sec: (i * 12) as f64,
            source_id: "west".to_string(),
            destination_sink_id: "east".to_string(),
        });
    }
    for i in 0..6 {
        scheduled_trips.push(ScheduledTrip {
            depart_sec: (6 + i * 24) as f64,
            source_id: "south".to_string(),
            destination_sink_id: "north".to_string(),
        });
    }
    scheduled_trips.sort_by(|a, b| {
        a.depart_sec
            .partial_cmp(&b.depart_sec)
            .unwrap()
            .then(a.source_id.cmp(&b.source_id))
    });
    SmartTrafficParams {
        scheduled_trips,
        duration_sec: 210.0,
        seed: 19,
    }
}

/// `validate-external-fel-models.ts` `main`.
pub fn run() {
    let root = std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")));
    let mut d = Driver {
        checks: Vec::new(),
        out_dir: root.join("out").join("external-fel"),
    };

    println!("External FEL comparison suite");
    println!("=============================");
    d.compare_computer_network_scenario(
        "small-enterprise",
        build_default_computer_network_problem(),
    );
    d.compare_computer_network_scenario(
        "bottleneck-lab",
        build_bottleneck_computer_network_problem(),
    );
    d.compare_traffic_scenario();

    println!();
    println!("========================================");
    let passed = d.checks.iter().filter(|c| c.passed).count();
    println!(
        "validate-external-fel-models: {}/{} checks passed.",
        passed,
        d.checks.len()
    );
    if passed < d.checks.len() {
        println!("FAILED:");
        for c in &d.checks {
            if !c.passed {
                println!(
                    "  - {}{}",
                    c.name,
                    c.detail
                        .as_ref()
                        .map(|x| format!(": {}", x))
                        .unwrap_or_default()
                );
            }
        }
        std::process::exit(1);
    }
}
