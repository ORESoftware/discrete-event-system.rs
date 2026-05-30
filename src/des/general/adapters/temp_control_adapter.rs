//! Port of `src/des/general/adapters/temp-control-adapter.ts`
//! (module `des::general::adapters::temp_control_adapter`).
//!
//! JSON adapter for the indoor temperature-control DES (bang-bang / PID / fuzzy
//! / MDP-MPC controllers).
//!
//! ## Conversion notes
//!
//!   * `ControllerSpec` is a discriminated union encoded as a `oneOf` schema;
//!     it reuses the ported [`ControllerSpec`] enum and matches on it in
//!     [`describe_controller`].
//!   * `run` copies params into a [`SimConfig`] and injects `seed: runtime.seed`
//!     (coerced `f64 -> u32`). `params.band` is already resolved by schema
//!     defaults, so the TS `?? 2` / `?? 1` fallbacks are no-ops here.
//!   * `Math.min(...result.T_in)` / `Math.max(...)` -> `fold` with `f64::min` /
//!     `f64::max`.
//!
//! PORT NOTE: `animate` depends on the un-ported animation subsystem
//! (`animation/scenes/temp-control-scene`, `animation/frame-recorder`); it is a
//! documented no-op here. `registerModel` / the model registry is also not
//! ported — the adapter is exposed via [`adapter()`].

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::temp_control::{
    run_temp_control, ControllerSpec, HouseParamsPartial, OutdoorPatternPartial, RunResult,
    SimConfig,
};

/// `interface TempControlParams`. `band` / `dt_min` / `cost_per_kWh` /
/// `comfort_penalty` are resolved by schema defaults, so they are plain `f64`.
#[derive(Clone, Debug)]
pub struct TempControlParams {
    pub t_target: f64,
    pub band: f64,
    pub duration_h: f64,
    pub dt_min: f64,
    pub cost_per_kwh: f64,
    pub comfort_penalty: f64,
    pub controller: ControllerSpec,
    pub house: Option<HouseParamsPartial>,
    pub outdoor: Option<OutdoorPatternPartial>,
    pub sensor_noise_std: Option<f64>,
    pub forecast_noise_std: Option<f64>,
    pub forecast_horizon_h: Option<f64>,
}

fn num(
    min: Option<f64>,
    max: Option<f64>,
    integer: Option<bool>,
    default: Option<f64>,
    description: Option<&str>,
) -> ParamSchema {
    ParamSchema::Number {
        min,
        max,
        integer,
        default,
        description: description.map(|s| s.to_string()),
    }
}

fn str_enum(allowed: &[&str]) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: None,
        description: None,
    }
}

/// `const controllerSchema` (a `oneOf` over the four controller kinds).
fn controller_schema() -> ParamSchema {
    use crate::des::general::des_spec::OneOfVariant;
    ParamSchema::OneOf {
        description: Some("Controller type and its hyperparameters.".to_string()),
        variants: vec![
            OneOfVariant {
                tag: "bang-bang".to_string(),
                tag_field: None,
                description: None,
                schema: ParamSchema::Object {
                    fields: vec![("kind".to_string(), str_enum(&["bang-bang"]))],
                    required: Some(vec!["kind".to_string()]),
                    description: None,
                },
            },
            OneOfVariant {
                tag: "pid".to_string(),
                tag_field: None,
                description: None,
                schema: ParamSchema::Object {
                    fields: vec![
                        ("kind".to_string(), str_enum(&["pid"])),
                        (
                            "Kp".to_string(),
                            num(
                                Some(0.0),
                                None,
                                None,
                                None,
                                Some("Proportional gain (kW/°F)"),
                            ),
                        ),
                        (
                            "Ki".to_string(),
                            num(Some(0.0), None, None, None, Some("Integral gain (kW/°F·h)")),
                        ),
                        (
                            "Kd".to_string(),
                            num(
                                Some(0.0),
                                None,
                                None,
                                None,
                                Some("Derivative gain (kW·h/°F)"),
                            ),
                        ),
                    ],
                    required: Some(vec![
                        "kind".to_string(),
                        "Kp".to_string(),
                        "Ki".to_string(),
                        "Kd".to_string(),
                    ]),
                    description: None,
                },
            },
            OneOfVariant {
                tag: "fuzzy".to_string(),
                tag_field: None,
                description: None,
                schema: ParamSchema::Object {
                    fields: vec![("kind".to_string(), str_enum(&["fuzzy"]))],
                    required: Some(vec!["kind".to_string()]),
                    description: None,
                },
            },
            OneOfVariant {
                tag: "mdp-mpc".to_string(),
                tag_field: None,
                description: None,
                schema: ParamSchema::Object {
                    fields: vec![
                        ("kind".to_string(), str_enum(&["mdp-mpc"])),
                        (
                            "horizon_h".to_string(),
                            num(
                                Some(0.1),
                                None,
                                None,
                                None,
                                Some("Lookahead horizon in hours"),
                            ),
                        ),
                        (
                            "nLevels".to_string(),
                            num(
                                Some(2.0),
                                Some(20.0),
                                Some(true),
                                None,
                                Some("Number of discrete heater levels (2-20)"),
                            ),
                        ),
                        (
                            "comfort_penalty".to_string(),
                            num(Some(0.0), None, None, None, None),
                        ),
                        (
                            "cost_per_kWh".to_string(),
                            num(Some(0.0), None, None, None, None),
                        ),
                        (
                            "trackWeight".to_string(),
                            num(
                                Some(0.0),
                                None,
                                None,
                                Some(1.0),
                                Some("Soft tracking weight inside band"),
                            ),
                        ),
                    ],
                    required: Some(vec![
                        "kind".to_string(),
                        "horizon_h".to_string(),
                        "nLevels".to_string(),
                        "comfort_penalty".to_string(),
                        "cost_per_kWh".to_string(),
                    ]),
                    description: None,
                },
            },
        ],
    }
}

/// `const tempControlSchema`.
pub fn temp_control_schema() -> ParamSchema {
    let house_schema = ParamSchema::Object {
        fields: vec![
            (
                "tau".to_string(),
                num(
                    Some(0.01),
                    None,
                    None,
                    Some(12.0),
                    Some("Thermal time constant (h)"),
                ),
            ),
            (
                "G".to_string(),
                num(
                    Some(0.0),
                    None,
                    None,
                    Some(1.0),
                    Some("Heater gain (°F per kW per hour)"),
                ),
            ),
            (
                "Q_max".to_string(),
                num(
                    Some(0.0),
                    None,
                    None,
                    Some(5.0),
                    Some("Max heater power (kW)"),
                ),
            ),
            (
                "T_init".to_string(),
                num(
                    None,
                    None,
                    None,
                    Some(70.0),
                    Some("Initial indoor temperature (°F)"),
                ),
            ),
        ],
        required: Some(vec![]),
        description: None,
    };
    let outdoor_schema = ParamSchema::Object {
        fields: vec![
            ("mean".to_string(), num(None, None, None, Some(25.0), None)),
            (
                "amp".to_string(),
                num(Some(0.0), None, None, Some(15.0), None),
            ),
            ("phase".to_string(), num(None, None, None, Some(9.0), None)),
            (
                "noiseStd".to_string(),
                num(Some(0.0), None, None, Some(1.5), None),
            ),
        ],
        required: Some(vec![]),
        description: None,
    };
    ParamSchema::Object {
        fields: vec![
            (
                "T_target".to_string(),
                num(
                    None,
                    None,
                    None,
                    None,
                    Some("Target indoor temperature (°F)"),
                ),
            ),
            (
                "band".to_string(),
                num(
                    Some(0.0),
                    None,
                    None,
                    Some(2.0),
                    Some("±band defining comfort interval (°F)"),
                ),
            ),
            (
                "duration_h".to_string(),
                num(
                    Some(0.0),
                    None,
                    None,
                    None,
                    Some("Simulated duration (hours)"),
                ),
            ),
            (
                "dt_min".to_string(),
                num(
                    Some(0.0),
                    None,
                    None,
                    Some(1.0),
                    Some("Tick length (minutes)"),
                ),
            ),
            (
                "cost_per_kWh".to_string(),
                num(
                    Some(0.0),
                    None,
                    None,
                    Some(0.15),
                    Some("Energy price ($/kWh)"),
                ),
            ),
            (
                "comfort_penalty".to_string(),
                num(
                    Some(0.0),
                    None,
                    None,
                    Some(0.5),
                    Some("Comfort violation penalty ($/(°F)²/h)"),
                ),
            ),
            ("controller".to_string(), controller_schema()),
            ("house".to_string(), house_schema),
            ("outdoor".to_string(), outdoor_schema),
            (
                "sensorNoiseStd".to_string(),
                num(Some(0.0), None, None, Some(0.0), None),
            ),
            (
                "forecastNoiseStd".to_string(),
                num(Some(0.0), None, None, Some(0.0), None),
            ),
            (
                "forecastHorizon_h".to_string(),
                num(Some(0.1), None, None, Some(6.0), None),
            ),
        ],
        required: Some(vec![
            "T_target".to_string(),
            "duration_h".to_string(),
            "controller".to_string(),
        ]),
        description: Some("Temperature-control simulation parameters.".to_string()),
    }
}

/// `function describeController(c)`.
pub fn describe_controller(c: &ControllerSpec) -> String {
    match c {
        ControllerSpec::BangBang => "Bang-bang".to_string(),
        ControllerSpec::Pid { kp, ki, kd } => format!("PID (Kp={kp}, Ki={ki}, Kd={kd})"),
        ControllerSpec::Fuzzy => "Fuzzy-PI (Mamdani)".to_string(),
        ControllerSpec::MdpMpc {
            horizon_h,
            n_levels,
            track_weight,
            ..
        } => {
            format!(
                "MDP-MPC (H={}h, levels={}, w={})",
                horizon_h,
                n_levels,
                track_weight.unwrap_or(1.0)
            )
        }
    }
}

/// `const adapter: DESModelRegistration<TempControlParams, RunResult>`.
pub struct TempControlAdapter;

/// Construct the adapter (see the module's PORT NOTE about registration).
pub fn adapter() -> TempControlAdapter {
    TempControlAdapter
}

impl DESModelRegistration<TempControlParams, RunResult> for TempControlAdapter {
    fn id(&self) -> &str {
        "temp-control"
    }

    fn description(&self) -> &str {
        "Indoor temperature-control DES with bang-bang / PID / Fuzzy / MDP-MPC controllers."
    }

    fn schema(&self) -> ParamSchema {
        temp_control_schema()
    }

    fn run(&self, params: TempControlParams, runtime: &DESRuntimeConfig) -> RunResult {
        let cfg = SimConfig {
            t_target: params.t_target,
            band: Some(params.band),
            duration_h: params.duration_h,
            dt_min: params.dt_min,
            controller: params.controller,
            house: params.house,
            outdoor: params.outdoor,
            cost_per_kwh: params.cost_per_kwh,
            comfort_penalty: params.comfort_penalty,
            sensor_noise_std: params.sensor_noise_std,
            forecast_noise_std: params.forecast_noise_std,
            forecast_horizon_h: params.forecast_horizon_h,
            seed: runtime.seed.map(|s| s as u32),
        };
        run_temp_control(cfg)
    }

    fn summarize(&self, result: &RunResult, params: &TempControlParams) -> String {
        let min_t = result.t_in.iter().copied().fold(f64::INFINITY, f64::min);
        let max_t = result
            .t_in
            .iter()
            .copied()
            .fold(f64::NEG_INFINITY, f64::max);
        let lines = vec![
            "TEMPERATURE-CONTROL RUN SUMMARY".to_string(),
            "──────────────────────────────────".to_string(),
            format!(
                "  Controller:      {}",
                describe_controller(&params.controller)
            ),
            format!(
                "  Target:          {:.2}°F ± {:.2}°F",
                params.t_target, params.band
            ),
            format!(
                "  Duration:        {:.2} h  ({} ticks of {} min)",
                params.duration_h,
                result.trace.len(),
                params.dt_min
            ),
            format!("  Indoor range:    [{min_t:.2}, {max_t:.2}] °F"),
            format!("  Energy used:     {:.2} kWh", result.energy_kwh),
            format!(
                "  Comfort:         {:.2}% in band",
                100.0 * result.comfort_pct
            ),
            format!(
                "  Violation:       {:.3} °F·h outside band",
                result.violation_fh
            ),
            format!("  Total cost:      ${:.2}", result.cost),
        ];
        lines.join("\n")
    }

    fn write_csv(&self, result: &RunResult, csv_path: &str) {
        let mut lines =
            vec!["tick,t_h,T_out,T_in,Q,energy_cum_kWh,error,in_band,violation_Fh".to_string()];
        for r in &result.trace {
            lines.push(csv_row([
                r.tick.to_string(),
                format!("{:.4}", r.t_h),
                format!("{:.3}", r.t_out_true),
                format!("{:.3}", r.t_in_true),
                format!("{:.3}", r.q),
                format!("{:.3}", r.energy_cum_kwh),
                format!("{:.3}", r.error),
                if r.in_band {
                    "1".to_string()
                } else {
                    "0".to_string()
                },
                format!("{:.4}", r.violation_fh),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }

    fn examples(&self) -> Vec<RegistrationExample<TempControlParams>> {
        vec![RegistrationExample {
            name: "PID winter day".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "temp-control".to_string(),
                description: Some("24-hour winter day, PID controller".to_string()),
                parameters: TempControlParams {
                    t_target: 70.0,
                    band: 2.0,
                    duration_h: 24.0,
                    dt_min: 1.0,
                    cost_per_kwh: 0.15,
                    comfort_penalty: 0.5,
                    controller: ControllerSpec::Pid {
                        kp: 3.0,
                        ki: 0.5,
                        kd: 0.5,
                    },
                    house: None,
                    outdoor: None,
                    sensor_noise_std: Some(0.2),
                    forecast_noise_std: Some(1.5),
                    forecast_horizon_h: Some(6.0),
                },
                runtime: Some(DESRuntimeConfig {
                    seed: Some(42.0),
                    ..Default::default()
                }),
                metadata: None,
            },
        }]
    }
}
