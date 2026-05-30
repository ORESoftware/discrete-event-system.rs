//! Port of `src/des/general/adapters/simulated-annealing-adapter.ts`
//! (module `des::general::adapters::simulated_annealing_adapter`).
//!
//! JSON adapter registering the simulated-annealing solver (built-in TSP /
//! knapsack problems).
//!
//! ## Conversion notes
//!
//!   * The TS `SAResult<unknown>` (erased state) becomes [`SAAdapterRaw`], an
//!     enum over the two concrete state types the solver produces:
//!     `SAResult<Tour>` (TSP, `Vec<usize>`) and `SAResult<Vec<f64>>` (knapsack
//!     binary vector). The scalar fields the summary / CSV read are exposed via
//!     accessor methods so neither branch needs `as`-casting.
//!   * `CoolingSchedule` is the ported discriminated-union enum; `problem` /
//!     `tsp.builtin` / `init` / `moves` literals become enums.
//!   * `runSimulatedAnnealing(problem, opts)` takes an `Rc<dyn SAProblem<S>>`,
//!     so the concrete problem is boxed before the call.
//!   * `buildKnapsackSAProblem(inst)` defaults `penalty = 1e6`;
//!     `buildPentagonTSP(n)` defaults `radius = 50` — both passed explicitly.
//!   * `throw new Error(...)` for missing config -> `panic!`.
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the
//! adapter is exposed via [`adapter()`].

#![allow(dead_code)]

use std::rc::Rc;

use crate::des::general::adapters::adapter_utils::write_csv_lines;
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, OneOfVariant, ParamSchema,
    RegistrationExample, DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::genetic_tsp::{
    build_pentagon_tsp, build_random_tsp, tour_length, InitMode, TSPInstance, Tour,
};
use crate::des::general::simulated_annealing::{
    build_knapsack_sa_problem, build_tsp_sa_problem, run_simulated_annealing, CoolingSchedule,
    KnapsackInstance, SAMove, SAProblem, SAResult, SASolverOptions, TSPSAProblemOptions,
};

/// `problem: 'tsp' | 'knapsack'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SAProblemKind {
    Tsp,
    Knapsack,
}

impl SAProblemKind {
    pub fn as_str(self) -> &'static str {
        match self {
            SAProblemKind::Tsp => "tsp",
            SAProblemKind::Knapsack => "knapsack",
        }
    }
}

/// `tsp.builtin?: 'pentagon' | 'random'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TspBuiltin {
    Pentagon,
    Random,
}

/// `tsp` parameter group.
#[derive(Clone, Debug, Default)]
pub struct TspParams {
    pub builtin: Option<TspBuiltin>,
    pub n: Option<usize>,
    pub seed: Option<u32>,
    pub coordinates: Option<Vec<(f64, f64)>>,
    pub distance: Option<Vec<Vec<f64>>>,
    pub precedence: Option<Vec<(usize, usize)>>,
    pub init: Option<InitMode>,
    pub moves: Option<SAMove>,
    pub penalty_per_violation: Option<f64>,
}

/// `knapsack` parameter group.
#[derive(Clone, Debug)]
pub struct KnapsackParams {
    pub values: Vec<f64>,
    pub weights: Vec<f64>,
    pub capacity: f64,
}

/// `options` parameter group.
#[derive(Clone, Debug, Default)]
pub struct SAOptions {
    pub max_iterations: usize,
    pub seed: Option<u32>,
    pub stall_limit: Option<usize>,
    pub record_trace: Option<bool>,
    pub trace_stride: Option<usize>,
}

/// `interface SAParams`.
#[derive(Clone, Debug)]
pub struct SAParams {
    pub problem: SAProblemKind,
    pub tsp: Option<TspParams>,
    pub knapsack: Option<KnapsackParams>,
    pub cooling: CoolingSchedule,
    pub options: SAOptions,
}

/// `raw: SAResult<unknown>` — one of the two concrete state types.
pub enum SAAdapterRaw {
    Tsp(SAResult<Tour>),
    Knapsack(SAResult<Vec<f64>>),
}

impl SAAdapterRaw {
    pub fn iterations(&self) -> usize {
        match self {
            SAAdapterRaw::Tsp(r) => r.iterations,
            SAAdapterRaw::Knapsack(r) => r.iterations,
        }
    }
    pub fn accepted_count(&self) -> usize {
        match self {
            SAAdapterRaw::Tsp(r) => r.accepted_count,
            SAAdapterRaw::Knapsack(r) => r.accepted_count,
        }
    }
    pub fn improve_count(&self) -> usize {
        match self {
            SAAdapterRaw::Tsp(r) => r.improve_count,
            SAAdapterRaw::Knapsack(r) => r.improve_count,
        }
    }
    pub fn best_cost(&self) -> f64 {
        match self {
            SAAdapterRaw::Tsp(r) => r.best_cost,
            SAAdapterRaw::Knapsack(r) => r.best_cost,
        }
    }
    pub fn final_cost(&self) -> f64 {
        match self {
            SAAdapterRaw::Tsp(r) => r.final_cost,
            SAAdapterRaw::Knapsack(r) => r.final_cost,
        }
    }
    pub fn temperature_history(&self) -> &[f64] {
        match self {
            SAAdapterRaw::Tsp(r) => &r.temperature_history,
            SAAdapterRaw::Knapsack(r) => &r.temperature_history,
        }
    }
    pub fn best_history(&self) -> &[f64] {
        match self {
            SAAdapterRaw::Tsp(r) => &r.best_history,
            SAAdapterRaw::Knapsack(r) => &r.best_history,
        }
    }
    pub fn current_history(&self) -> &[f64] {
        match self {
            SAAdapterRaw::Tsp(r) => &r.current_history,
            SAAdapterRaw::Knapsack(r) => &r.current_history,
        }
    }
}

/// TSP-specific extras: tour length and instance size.
#[derive(Clone, Copy, Debug)]
pub struct TspExtras {
    pub tour_length: f64,
    pub n: usize,
}

/// Knapsack-specific extras.
#[derive(Clone, Copy, Debug)]
pub struct KnapExtras {
    pub value: f64,
    pub weight: f64,
    pub capacity: f64,
}

/// `interface SAAdapterResult`.
pub struct SAAdapterResult {
    pub problem: SAProblemKind,
    pub raw: SAAdapterRaw,
    pub tsp_extras: Option<TspExtras>,
    pub knap_extras: Option<KnapExtras>,
}

fn num(min: Option<f64>, max: Option<f64>, integer: Option<bool>) -> ParamSchema {
    ParamSchema::Number { min, max, integer, default: None, description: None }
}

fn str_enum(allowed: &[&str]) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: None,
        description: None,
    }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields.into_iter().map(|(k, v)| (k.to_string(), v)).collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn arr(items: ParamSchema) -> ParamSchema {
    ParamSchema::Array { items: Box::new(items), min_length: None, max_length: None, description: None }
}

/// `const coolingSchema` (the `oneOf` over cooling-schedule kinds).
fn cooling_schema() -> ParamSchema {
    let variant = |tag: &str, schema: ParamSchema| OneOfVariant {
        tag: tag.to_string(),
        tag_field: None,
        description: None,
        schema,
    };
    ParamSchema::OneOf {
        description: None,
        variants: vec![
            variant(
                "geometric",
                obj(
                    vec![
                        ("kind", str_enum(&["geometric"])),
                        ("T0", num(Some(0.0), None, None)),
                        ("alpha", num(Some(0.0), Some(1.0), None)),
                        ("Tmin", num(Some(0.0), None, None)),
                    ],
                    vec!["kind", "T0", "alpha"],
                ),
            ),
            variant(
                "logarithmic",
                obj(
                    vec![
                        ("kind", str_enum(&["logarithmic"])),
                        ("T0", num(Some(0.0), None, None)),
                        ("Tmin", num(Some(0.0), None, None)),
                    ],
                    vec!["kind", "T0"],
                ),
            ),
            variant(
                "linear",
                obj(
                    vec![
                        ("kind", str_enum(&["linear"])),
                        ("T0", num(Some(0.0), None, None)),
                        ("rate", num(Some(0.0), None, None)),
                        ("Tmin", num(Some(0.0), None, None)),
                    ],
                    vec!["kind", "T0", "rate"],
                ),
            ),
            variant(
                "exp-restart",
                obj(
                    vec![
                        ("kind", str_enum(&["exp-restart"])),
                        ("T0", num(Some(0.0), None, None)),
                        ("alpha", num(Some(0.0), Some(1.0), None)),
                        ("period", num(Some(1.0), None, Some(true))),
                        ("Tmin", num(Some(0.0), None, None)),
                    ],
                    vec!["kind", "T0", "alpha", "period"],
                ),
            ),
        ],
    }
}

/// `const saSchema`.
pub fn sa_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            ("problem".to_string(), str_enum(&["tsp", "knapsack"])),
            (
                "tsp".to_string(),
                obj(
                    vec![
                        ("builtin", str_enum(&["pentagon", "random"])),
                        ("n", num(Some(3.0), None, Some(true))),
                        ("seed", num(None, None, Some(true))),
                        ("coordinates", arr(arr(num(None, None, None)))),
                        ("distance", arr(arr(num(None, None, None)))),
                        ("precedence", arr(arr(num(Some(0.0), None, Some(true))))),
                        ("init", str_enum(&["random", "nearest-neighbor"])),
                        ("moves", str_enum(&["2-opt", "or-opt", "mixed"])),
                        ("penaltyPerViolation", num(Some(0.0), None, None)),
                    ],
                    vec![],
                ),
            ),
            (
                "knapsack".to_string(),
                obj(
                    vec![
                        ("values", arr(num(None, None, None))),
                        ("weights", arr(num(None, None, None))),
                        ("capacity", num(Some(0.0), None, None)),
                    ],
                    vec!["values", "weights", "capacity"],
                ),
            ),
            ("cooling".to_string(), cooling_schema()),
            (
                "options".to_string(),
                obj(
                    vec![
                        ("maxIterations", num(Some(1.0), None, Some(true))),
                        ("seed", num(None, None, Some(true))),
                        ("stallLimit", num(Some(0.0), None, Some(true))),
                        ("recordTrace", ParamSchema::Boolean { default: None, description: None }),
                        ("traceStride", num(Some(1.0), None, Some(true))),
                    ],
                    vec!["maxIterations"],
                ),
            ),
        ],
        required: Some(vec!["problem".to_string(), "cooling".to_string(), "options".to_string()]),
        description: Some(
            "Simulated annealing on a generic combinatorial problem (TSP / knapsack built-in).".to_string(),
        ),
    }
}

fn cooling_kind(c: &CoolingSchedule) -> &'static str {
    match c {
        CoolingSchedule::Geometric { .. } => "geometric",
        CoolingSchedule::Logarithmic { .. } => "logarithmic",
        CoolingSchedule::Linear { .. } => "linear",
        CoolingSchedule::ExpRestart { .. } => "exp-restart",
    }
}

fn solver_options(params: &SAParams) -> SASolverOptions {
    SASolverOptions {
        max_iterations: params.options.max_iterations,
        cooling: params.cooling,
        seed: params.options.seed,
        stall_limit: params.options.stall_limit,
        verbose: None,
        record_trace: params.options.record_trace,
        trace_stride: params.options.trace_stride,
    }
}

/// `const adapter`.
pub struct SimulatedAnnealingAdapter;

/// Construct the adapter (see the module's PORT NOTE about registration).
pub fn adapter() -> SimulatedAnnealingAdapter {
    SimulatedAnnealingAdapter
}

impl DESModelRegistration<SAParams, SAAdapterResult> for SimulatedAnnealingAdapter {
    fn id(&self) -> &str {
        "simulated-annealing"
    }

    fn description(&self) -> &str {
        "Simulated annealing on built-in TSP / knapsack problems (extensible to others via TS subclassing)."
    }

    fn schema(&self) -> ParamSchema {
        sa_schema()
    }

    fn run(&self, params: SAParams, _runtime: &DESRuntimeConfig) -> SAAdapterResult {
        match params.problem {
            SAProblemKind::Tsp => {
                let tsp = params.tsp.as_ref();
                let builtin = tsp.and_then(|t| t.builtin);
                let inst: TSPInstance = if builtin == Some(TspBuiltin::Pentagon) {
                    build_pentagon_tsp(tsp.and_then(|t| t.n).unwrap_or(5), 50.0)
                } else if builtin == Some(TspBuiltin::Random) {
                    build_random_tsp(tsp.and_then(|t| t.n).unwrap_or(20), tsp.and_then(|t| t.seed).unwrap_or(42), None)
                } else if let (Some(coords), Some(dist)) =
                    (tsp.and_then(|t| t.coordinates.clone()), tsp.and_then(|t| t.distance.clone()))
                {
                    TSPInstance {
                        n: coords.len(),
                        coordinates: coords,
                        distance: dist,
                        precedence: tsp.and_then(|t| t.precedence.clone()),
                    }
                } else {
                    panic!("simulated-annealing: tsp params must specify builtin or (coordinates + distance)");
                };
                let n = inst.n;
                let problem = build_tsp_sa_problem(
                    inst.clone(),
                    TSPSAProblemOptions {
                        init: Some(tsp.and_then(|t| t.init).unwrap_or(InitMode::NearestNeighbor)),
                        moves: Some(tsp.and_then(|t| t.moves).unwrap_or(SAMove::Mixed)),
                        penalty_per_violation: tsp.and_then(|t| t.penalty_per_violation),
                    },
                );
                let r = run_simulated_annealing(Rc::new(problem) as Rc<dyn SAProblem<Tour>>, solver_options(&params));
                let tour_len = tour_length(&inst, &r.best_state);
                SAAdapterResult {
                    problem: SAProblemKind::Tsp,
                    raw: SAAdapterRaw::Tsp(r),
                    tsp_extras: Some(TspExtras { tour_length: tour_len, n }),
                    knap_extras: None,
                }
            }
            SAProblemKind::Knapsack => {
                let knap = params
                    .knapsack
                    .as_ref()
                    .unwrap_or_else(|| panic!("simulated-annealing: knapsack params required"));
                let inst = KnapsackInstance {
                    values: knap.values.clone(),
                    weights: knap.weights.clone(),
                    capacity: knap.capacity,
                };
                let problem = build_knapsack_sa_problem(inst, 1e6);
                let r = run_simulated_annealing(
                    Rc::new(problem) as Rc<dyn SAProblem<Vec<f64>>>,
                    solver_options(&params),
                );
                let x = &r.best_state;
                let mut v = 0.0;
                let mut w = 0.0;
                for i in 0..x.len() {
                    v += knap.values[i] * x[i];
                    w += knap.weights[i] * x[i];
                }
                let capacity = knap.capacity;
                SAAdapterResult {
                    problem: SAProblemKind::Knapsack,
                    raw: SAAdapterRaw::Knapsack(r),
                    tsp_extras: None,
                    knap_extras: Some(KnapExtras { value: v, weight: w, capacity }),
                }
            }
        }
    }

    fn summarize(&self, result: &SAAdapterResult, params: &SAParams) -> String {
        let mut lines = vec![
            "SIMULATED-ANNEALING RUN SUMMARY".to_string(),
            "──────────────────────────────────".to_string(),
            format!("  Problem:           {}", result.problem.as_str()),
            format!("  Cooling:           {}", cooling_kind(&params.cooling)),
            format!("  Iterations:        {}", result.raw.iterations()),
            format!("  Accepted:          {}", result.raw.accepted_count()),
            format!("  Improvements:      {}", result.raw.improve_count()),
            format!("  Best cost:         {:.4}", result.raw.best_cost()),
            format!("  Final cost:        {:.4}", result.raw.final_cost()),
        ];
        if let Some(t) = &result.tsp_extras {
            lines.push(format!("  Tour length (n={}):  {:.4}", t.n, t.tour_length));
        }
        if let Some(k) = &result.knap_extras {
            lines.push(format!(
                "  Knapsack value:    {:.2}    weight: {:.2} / {}",
                k.value, k.weight, k.capacity
            ));
        }
        lines.join("\n")
    }

    fn write_csv(&self, result: &SAAdapterResult, csv_path: &str) {
        let mut lines = vec!["k,T,best_cost,current_cost".to_string()];
        let t = result.raw.temperature_history();
        let b = result.raw.best_history();
        let c = result.raw.current_history();
        for i in 0..b.len() {
            let temp = t.get(i).map(|v| v.to_string()).unwrap_or_default();
            lines.push(format!("{},{},{},{}", i, temp, b[i], c[i]));
        }
        write_csv_lines(csv_path, &lines);
    }

    fn examples(&self) -> Vec<RegistrationExample<SAParams>> {
        vec![RegistrationExample {
            name: "sa-tsp-random20".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "simulated-annealing".to_string(),
                description: Some("SA on a 20-city random TSP".to_string()),
                parameters: SAParams {
                    problem: SAProblemKind::Tsp,
                    tsp: Some(TspParams {
                        builtin: Some(TspBuiltin::Random),
                        n: Some(20),
                        seed: Some(5),
                        init: Some(InitMode::NearestNeighbor),
                        moves: Some(SAMove::Mixed),
                        ..Default::default()
                    }),
                    knapsack: None,
                    cooling: CoolingSchedule::Geometric { t0: 100.0, alpha: 0.999, t_min: None },
                    options: SAOptions { max_iterations: 30000, seed: Some(1), ..Default::default() },
                },
                runtime: None,
                metadata: None,
            },
        }]
    }
}
