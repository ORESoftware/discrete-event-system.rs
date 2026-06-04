//! Port of `src/des/runners/validate-external-fel-models.ts`.
//!
//! Runs representative non-epidemic DES models through the real Rust kernels and
//! compares against external FEL outputs when a real reference payload is
//! available. Missing external reference scripts/artifacts are reported as
//! skips, not as synthetic passes.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::des::general::computer_network::{
    build_bottleneck_computer_network_problem, build_default_computer_network_problem,
    run_computer_network_simulation, ComputerNetworkProblem, ComputerNetworkResult,
};
use crate::des::general::network_flow::{
    build_five_intersection_traffic_network, TrafficParams, TrafficScheduledTrip,
};
use crate::des::general::smart_traffic_flow::{
    run_smart_traffic_flow, SmartTrafficParams, SmartTrafficResult,
};

const COMPUTER_NETWORK_MODULE_ID: &str = "computer-network-fel-reference";
const TRAFFIC_MODULE_ID: &str = "traffic-fel-reference";

const COMPUTER_NETWORK_SCRIPT: &str =
    "external-references/computer-network/network_fel_reference.py";
const TRAFFIC_SCRIPT: &str = "external-references/traffic/fel_traffic_reference.py";

#[derive(Clone, Debug)]
struct CheckRow {
    name: String,
    passed: bool,
    detail: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
struct FelBottleneck {
    kind: String,
    id: String,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ComputerNetworkFelResult {
    #[serde(alias = "generated_packets")]
    generated_packets: f64,
    #[serde(alias = "delivered_packets")]
    delivered_packets: f64,
    #[serde(alias = "dropped_packets")]
    dropped_packets: f64,
    #[serde(alias = "active_packets")]
    active_packets: f64,
    #[serde(alias = "max_active_packets")]
    max_active_packets: f64,
    #[serde(alias = "delivery_ratio")]
    delivery_ratio: f64,
    #[serde(alias = "offered_load_mbps")]
    offered_load_mbps: f64,
    #[serde(alias = "goodput_mbps")]
    goodput_mbps: f64,
    #[serde(alias = "mean_latency_ms")]
    mean_latency_ms: f64,
    #[serde(alias = "p95_latency_ms")]
    p95_latency_ms: f64,
    #[serde(alias = "total_cost")]
    total_cost: f64,
    #[serde(default)]
    bottlenecks: Vec<FelBottleneck>,
    #[serde(default, alias = "invariant_violations")]
    invariant_violations: Vec<String>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ComputerNetworkFelPayload {
    kernel: Option<String>,
    result: Option<ComputerNetworkFelResult>,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrafficFelResult {
    #[serde(alias = "generated_demand")]
    generated_demand: f64,
    entered: f64,
    exited: f64,
    dropped: f64,
    #[serde(alias = "active_at_end")]
    active_at_end: f64,
    #[serde(alias = "max_active_cars")]
    max_active_cars: f64,
    #[serde(default, alias = "completion_ratio")]
    completion_ratio: f64,
    #[serde(alias = "mean_travel_time_sec")]
    mean_travel_time_sec: f64,
    #[serde(default, alias = "p95_travel_time_sec")]
    p95_travel_time_sec: f64,
    #[serde(alias = "mean_speed_mps")]
    mean_speed_mps: f64,
    #[serde(alias = "event_count")]
    event_count: f64,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct TrafficFelPayload {
    kernel: Option<String>,
    status: Option<String>,
    message: Option<String>,
    result: Option<TrafficFelResult>,
}

struct Driver {
    checks: Vec<CheckRow>,
    skipped: usize,
    root: PathBuf,
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
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

impl Driver {
    fn check(&mut self, name: &str, passed: bool, detail: impl Into<Option<String>>) {
        let detail = detail.into();
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

    fn skip(&mut self, name: &str, detail: impl Into<Option<String>>) {
        let detail = detail.into();
        let tail = detail
            .as_ref()
            .map(|d| format!(" - {}", d))
            .unwrap_or_default();
        println!("  SKIP  {}{}", name, tail);
        self.skipped += 1;
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

    fn write_input_note(&mut self, scenario: &str, model: &str) {
        let path = self.scenario_path(scenario, "input.json");
        if let Some(dir) = path.parent() {
            if let Err(err) = fs::create_dir_all(dir) {
                self.check(
                    &format!("{scenario}: creates output directory"),
                    false,
                    Some(err.to_string()),
                );
                return;
            }
        }
        let body = format!(
            "{{\n  \"model\": \"{}\",\n  \"scenario\": \"{}\",\n  \"runner\": \"validate_external_fel_models\"\n}}\n",
            model, scenario
        );
        self.check(
            &format!("{scenario}: writes internal scenario note"),
            fs::write(&path, body).is_ok(),
            Some(path.display().to_string()),
        );
    }

    fn external_script_missing(&mut self, module_id: &str, script: &str) -> bool {
        let path = self.root.join(script);
        if path.exists() {
            return false;
        }
        self.skip(
            &format!("{module_id}: external FEL script unavailable"),
            Some(path.display().to_string()),
        );
        true
    }

    fn load_external_payload<T: for<'de> Deserialize<'de>>(
        &mut self,
        scenario: &str,
        module_id: &str,
    ) -> Option<T> {
        let path = self.scenario_path(scenario, "external-fel.json");
        if !path.exists() {
            self.skip(
                &format!("{module_id}: external FEL output unavailable"),
                Some(path.display().to_string()),
            );
            return None;
        }
        let text = match fs::read_to_string(&path) {
            Ok(text) => text,
            Err(err) => {
                self.check(
                    &format!("{module_id}: reads external FEL output"),
                    false,
                    Some(err.to_string()),
                );
                return None;
            }
        };
        let trimmed = text.trim();
        if trimmed.is_empty() || trimmed == "{}" || trimmed == "null" {
            self.skip(
                &format!("{module_id}: external FEL payload empty"),
                Some(path.display().to_string()),
            );
            return None;
        }
        match serde_json::from_str::<T>(trimmed) {
            Ok(payload) => {
                self.check(
                    &format!("{module_id}: parses external FEL payload"),
                    true,
                    Some(path.display().to_string()),
                );
                Some(payload)
            }
            Err(err) => {
                self.check(
                    &format!("{module_id}: parses external FEL payload"),
                    false,
                    Some(err.to_string()),
                );
                None
            }
        }
    }

    fn compare_computer_network_scenario(&mut self, name: &str, problem: ComputerNetworkProblem) {
        println!();
        println!("-- computer-network/{name} --");
        let scenario = format!("computer-network-{name}");
        self.write_input_note(&scenario, "computer-network");

        let internal = run_computer_network_simulation(&problem);
        self.check_computer_network_internal(name, &internal);

        if self.external_script_missing(COMPUTER_NETWORK_MODULE_ID, COMPUTER_NETWORK_SCRIPT) {
            return;
        }
        let Some(payload) = self.load_external_payload::<ComputerNetworkFelPayload>(
            &scenario,
            COMPUTER_NETWORK_MODULE_ID,
        ) else {
            return;
        };
        let kernel = payload.kernel.unwrap_or_default();
        self.check(
            &format!("{name}: external reports FEL kernel"),
            kernel == "python-computer-network-fel-reference",
            Some(kernel),
        );
        let Some(external) = payload.result else {
            self.skip(
                &format!("{name}: external computer-network result absent"),
                Some("payload.result missing".to_string()),
            );
            return;
        };
        self.compare_computer_network_stats(name, &internal, &external);
    }

    fn check_computer_network_internal(&mut self, name: &str, result: &ComputerNetworkResult) {
        self.check(
            &format!("{name}: Rust network generated packets"),
            result.generated_packets > 0.0,
            Some(format!("generated={}", result.generated_packets)),
        );
        self.check(
            &format!("{name}: Rust network conserves packets"),
            result.generated_packets
                == result.delivered_packets + result.dropped_packets + result.active_packets,
            Some(format!(
                "generated={} delivered={} dropped={} active={}",
                result.generated_packets,
                result.delivered_packets,
                result.dropped_packets,
                result.active_packets
            )),
        );
        self.check(
            &format!("{name}: Rust network delivery ratio finite"),
            result.delivery_ratio.is_finite() && (0.0..=1.0).contains(&result.delivery_ratio),
            Some(format!("ratio={}", fmt(result.delivery_ratio))),
        );
        self.check(
            &format!("{name}: Rust network latency finite"),
            result.mean_latency_ms.is_finite() && result.p95_latency_ms.is_finite(),
            Some(format!(
                "mean={} p95={}",
                fmt(result.mean_latency_ms),
                fmt(result.p95_latency_ms)
            )),
        );
        self.check(
            &format!("{name}: Rust network invariants pass"),
            result.invariant_violations.is_empty(),
            Some(format!("violations={}", result.invariant_violations.len())),
        );
    }

    fn compare_computer_network_stats(
        &mut self,
        name: &str,
        internal: &ComputerNetworkResult,
        external: &ComputerNetworkFelResult,
    ) {
        self.same_count(
            &format!("{name}: generated packets"),
            internal.generated_packets,
            external.generated_packets,
        );
        self.same_count(
            &format!("{name}: delivered packets"),
            internal.delivered_packets,
            external.delivered_packets,
        );
        self.same_count(
            &format!("{name}: dropped packets"),
            internal.dropped_packets,
            external.dropped_packets,
        );
        self.same_count(
            &format!("{name}: active packets"),
            internal.active_packets,
            external.active_packets,
        );
        self.same_count(
            &format!("{name}: max active packets"),
            internal.max_active_packets,
            external.max_active_packets,
        );
        self.close_abs(
            &format!("{name}: delivery ratio"),
            internal.delivery_ratio,
            external.delivery_ratio,
            1e-12,
        );
        self.close_abs(
            &format!("{name}: offered load Mbps"),
            internal.offered_load_mbps,
            external.offered_load_mbps,
            1e-12,
        );
        self.close_abs(
            &format!("{name}: goodput Mbps"),
            internal.goodput_mbps,
            external.goodput_mbps,
            1e-12,
        );
        self.close_abs(
            &format!("{name}: mean latency ms"),
            internal.mean_latency_ms,
            external.mean_latency_ms,
            1e-9,
        );
        self.close_abs(
            &format!("{name}: p95 latency ms"),
            internal.p95_latency_ms,
            external.p95_latency_ms,
            1e-9,
        );
        self.close_abs(
            &format!("{name}: total cost"),
            internal.total_cost,
            external.total_cost,
            1e-12,
        );
        let it = internal.bottlenecks.first();
        let et = external.bottlenecks.first();
        self.check(
            &format!("{name}: top bottleneck agrees"),
            it.map(|b| (b.kind.as_str(), b.id.as_str()))
                == et.map(|b| (b.kind.as_str(), b.id.as_str())),
            Some(format!(
                "internal={}:{} external={}:{}",
                it.map(|b| b.kind.as_str()).unwrap_or(""),
                it.map(|b| b.id.as_str()).unwrap_or(""),
                et.map(|b| b.kind.as_str()).unwrap_or(""),
                et.map(|b| b.id.as_str()).unwrap_or("")
            )),
        );
        self.check(
            &format!("{name}: invariant violation lists agree"),
            internal.invariant_violations == external.invariant_violations,
            Some(format!(
                "internal={} external={}",
                internal.invariant_violations.len(),
                external.invariant_violations.len()
            )),
        );
    }

    fn compare_traffic_scenario(&mut self) {
        println!();
        println!("-- smart-traffic-flow/signalized-corridor --");
        let scenario = "smart-traffic-signalized-corridor";
        self.write_input_note(scenario, "smart-traffic-flow");

        let params = build_signalized_corridor_params();
        let scheduled = params.base.scheduled_trips.as_ref().map_or(0, Vec::len);
        let internal = run_smart_traffic_flow(params, None);
        self.check_traffic_internal(&internal, scheduled);

        if self.external_script_missing(TRAFFIC_MODULE_ID, TRAFFIC_SCRIPT) {
            return;
        }
        let Some(payload) =
            self.load_external_payload::<TrafficFelPayload>(scenario, TRAFFIC_MODULE_ID)
        else {
            return;
        };
        let status = payload.status.unwrap_or_default();
        self.check(
            "traffic FEL payload is ok",
            status == "ok",
            Some(payload.message.unwrap_or(status)),
        );
        let kernel = payload.kernel.unwrap_or_default();
        self.check(
            "traffic: external reports FEL kernel",
            kernel == "python-traffic-fel-reference",
            Some(kernel),
        );
        let Some(external) = payload.result else {
            self.skip(
                "traffic: external FEL result absent",
                Some("payload.result missing".to_string()),
            );
            return;
        };
        self.compare_traffic_stats(&internal, scheduled, &external);
    }

    fn check_traffic_internal(&mut self, result: &SmartTrafficResult, scheduled: usize) {
        self.check(
            "traffic: Rust reads scheduled demand",
            scheduled > 0,
            Some(format!("scheduled={scheduled}")),
        );
        self.check(
            "traffic: Rust entered at least one car",
            result.entered > 0,
            Some(format!("entered={}", result.entered)),
        );
        self.check(
            "traffic: Rust conserves cars",
            result.entered
                == result.exited + result.crashed + result.dropped + result.final_cars.len(),
            Some(format!(
                "entered={} exited={} crashed={} dropped={} active={}",
                result.entered,
                result.exited,
                result.crashed,
                result.dropped,
                result.final_cars.len()
            )),
        );
        self.check(
            "traffic: Rust speeds finite",
            result.mean_speed_mps.is_finite() && result.mean_speed_mps >= 0.0,
            Some(format!("meanSpeed={}", fmt(result.mean_speed_mps))),
        );
        self.check(
            "traffic: Rust validators pass",
            result.validation.iter().all(|c| c.passed),
            Some(format!(
                "failed={}",
                result.validation.iter().filter(|c| !c.passed).count()
            )),
        );
    }

    fn compare_traffic_stats(
        &mut self,
        internal: &SmartTrafficResult,
        scheduled: usize,
        external: &TrafficFelResult,
    ) {
        let scheduled = scheduled as f64;
        self.same_count(
            "traffic: external reads scheduled demand",
            external.generated_demand,
            scheduled,
        );
        self.same_count(
            "traffic: internal entered scheduled demand",
            internal.entered as f64,
            scheduled,
        );
        self.same_count(
            "traffic: external entered scheduled demand",
            external.entered,
            scheduled,
        );
        self.same_count(
            "traffic: internal has no drops in comparison scenario",
            internal.dropped as f64,
            0.0,
        );
        self.same_count(
            "traffic: external has no drops in comparison scenario",
            external.dropped,
            0.0,
        );
        self.close_abs(
            "traffic: completed cars align",
            internal.exited as f64,
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
            internal.max_active_cars as f64,
            external.max_active_cars,
            0.75,
        );
        self.check(
            "traffic: external FEL processed events",
            external.event_count >= scheduled,
            Some(format!("events={}", external.event_count)),
        );
        self.check(
            "traffic: external completion ratio finite",
            external.completion_ratio.is_finite()
                && (0.0..=1.0).contains(&external.completion_ratio),
            Some(format!(
                "completionRatio={} p95TravelTime={}",
                fmt(external.completion_ratio),
                fmt(external.p95_travel_time_sec)
            )),
        );
    }
}

fn build_signalized_corridor_params() -> SmartTrafficParams {
    let mut scheduled_trips: Vec<TrafficScheduledTrip> = Vec::new();
    for i in 0..12 {
        scheduled_trips.push(TrafficScheduledTrip {
            depart_sec: (i * 12) as f64,
            source_id: "west".to_string(),
            destination_sink_id: "east".to_string(),
        });
    }
    for i in 0..6 {
        scheduled_trips.push(TrafficScheduledTrip {
            depart_sec: (6 + i * 24) as f64,
            source_id: "south0".to_string(),
            destination_sink_id: "north1".to_string(),
        });
    }
    scheduled_trips.sort_by(|a, b| {
        a.depart_sec
            .partial_cmp(&b.depart_sec)
            .unwrap()
            .then(a.source_id.cmp(&b.source_id))
    });
    SmartTrafficParams {
        base: TrafficParams {
            builtin: None,
            network: Some(build_five_intersection_traffic_network()),
            duration_sec: 210.0,
            dt_sec: 0.2,
            seed: 19.0,
            max_cars: 80,
            car_length_m: None,
            car_width_m: None,
            lane_width_m: None,
            min_gap_m: None,
            max_accel_mps2: None,
            max_decel_mps2: None,
            max_jerk_mps3: Some(4.0),
            reaction_time_sec: Some(0.8),
            time_headway_sec: Some(1.1),
            grid_cell_size_m: Some(0.3048),
            grid_look_ahead_m: None,
            spawn_rate_multiplier: Some(0.0),
            scheduled_trips: Some(scheduled_trips),
        },
        smart_car_pool_size: Some(120),
        actor_shuffle_seed: Some(2026.0),
        accident_risk_scale: Some(0.0),
        accident_probability: Some(0.0),
        accident_accel_boost_mps2: None,
        accident_fault_duration_sec: None,
        distance_preference_spread: Some(0.54),
        start_preference_spread: Some(0.65),
        accident_flash_seconds: None,
    }
}

fn root_from_env() -> PathBuf {
    std::env::var("REPO_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")))
}

fn ensure_out_dir(path: &Path) {
    let _ = fs::create_dir_all(path);
}

/// `validate-external-fel-models.ts` `main`.
pub fn run() {
    let root = root_from_env();
    let out_dir = root.join("out").join("external-fel");
    ensure_out_dir(&out_dir);
    let mut d = Driver {
        checks: Vec::new(),
        skipped: 0,
        root,
        out_dir,
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
        "validate-external-fel-models: {}/{} checks passed, {} skipped.",
        passed,
        d.checks.len(),
        d.skipped
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
