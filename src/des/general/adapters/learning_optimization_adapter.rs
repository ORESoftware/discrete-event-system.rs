//! Port of `src/des/general/adapters/learning-optimization-adapter.ts`
//! (module `des::general::adapters::learning_optimization_adapter`).
//!
//! Registers six JSON adapters: regression (linear / ridge / logistic / MLP)
//! and RL (policy-gradient corridor / expected-SARSA gridworld).
//!
//! ## Conversion notes
//!
//!   * `optimizer: 'sgd' | 'adam'` -> the engine [`Optimizer`] enum.
//!   * `rlSummary`'s inline `topology: {stations, movables}` -> a
//!     `&StationGraphSummary` (the `RLTopology` alias).
//!   * Param shapes reuse the engine `*Params` structs directly; defaults live
//!     in the schema and the model functions.
//!   * CSV writers zip parallel history arrays.
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the six
//! adapters are exposed via the `*_adapter()` constructors.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::des_base::learning_optimization::{Optimizer, StationGraphSummary};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::learning_optimization_models::{
    run_backprop_mlp_classifier, run_linear_regression_ls, run_logistic_regression_sgd,
    run_ridge_regression_ls, BackpropMLPParams, GradientTrainingResult, LinearRegressionParams,
    LinearRegressionResult, LogisticRegressionSGDParams, RidgeRegressionParams,
};
use crate::des::general::rl_learning_models::{
    run_expected_sarsa_gridworld, run_policy_gradient_corridor, ExpectedSarsaGridParams,
    ExpectedSarsaGridResult, PolicyGradientCorridorParams, PolicyGradientCorridorResult,
};

// =============================================================================
// Formatting helpers (JS parity).
// =============================================================================

fn js_number(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v.is_infinite() {
        if v > 0.0 {
            "Infinity".to_string()
        } else {
            "-Infinity".to_string()
        }
    } else {
        let s = v.to_string();
        if s == "-0" {
            "0".to_string()
        } else {
            s
        }
    }
}

fn to_exponential(v: f64, digits: usize) -> String {
    if !v.is_finite() {
        return js_number(v);
    }
    let raw = format!("{:.*e}", digits, v);
    match raw.split_once('e') {
        Some((mant, exp)) if !exp.starts_with('-') => format!("{mant}e+{exp}"),
        _ => raw,
    }
}

// =============================================================================
// Schema helpers
// =============================================================================

fn num(
    min: Option<f64>,
    max: Option<f64>,
    integer: Option<bool>,
    default: Option<f64>,
) -> ParamSchema {
    ParamSchema::Number {
        min,
        max,
        integer,
        default,
        description: None,
    }
}

fn boolean(default: Option<bool>) -> ParamSchema {
    ParamSchema::Boolean {
        default,
        description: None,
    }
}

fn arr(items: ParamSchema, min_length: Option<usize>) -> ParamSchema {
    ParamSchema::Array {
        items: Box::new(items),
        min_length,
        max_length: None,
        description: None,
    }
}

fn str_enum(allowed: &[&str], default: &str) -> ParamSchema {
    ParamSchema::String {
        allowed: Some(allowed.iter().map(|s| s.to_string()).collect()),
        default: Some(default.to_string()),
        description: None,
    }
}

fn obj(fields: Vec<(&str, ParamSchema)>, required: Vec<&str>) -> ParamSchema {
    ParamSchema::Object {
        fields: fields
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect(),
        required: Some(required.iter().map(|s| s.to_string()).collect()),
        description: None,
    }
}

fn supervised_sample_schema() -> ParamSchema {
    obj(
        vec![
            ("x", arr(num(None, None, None, None), Some(1))),
            ("y", num(None, None, None, None)),
        ],
        vec!["x", "y"],
    )
}

fn samples_schema() -> ParamSchema {
    arr(supervised_sample_schema(), Some(1))
}

// =============================================================================
// Shared summary / CSV helpers
// =============================================================================

/// `function gradientSummary(title, result)`.
fn gradient_summary(title: &str, result: &GradientTrainingResult) -> String {
    let weights = result
        .weights
        .iter()
        .take(8)
        .map(|v| format!("{v:.4}"))
        .collect::<Vec<_>>()
        .join(", ");
    [
        title.to_string(),
        "----------------------------------------".to_string(),
        format!("  Steps:          {}", result.loss_history.len()),
        format!("  Final loss:     {:.6}", result.final_loss),
        format!("  Accuracy:       {:.1}%", 100.0 * result.accuracy),
        format!("  Bias:           {:.6}", result.bias),
        format!(
            "  Weights:        [{}{}]",
            weights,
            if result.weights.len() > 8 {
                ", ..."
            } else {
                ""
            }
        ),
        format!(
            "  Stations:       {}",
            result.topology.stations.join(" -> ")
        ),
        format!("  Movables:       {}", result.topology.movables.join(", ")),
    ]
    .join("\n")
}

/// `function writeGradientCsv(result, csvPath)`.
fn write_gradient_csv(result: &GradientTrainingResult, csv_path: &str) {
    let mut lines = vec![csv_row(["step", "loss", "gradient_norm"])];
    for (i, &loss) in result.loss_history.iter().enumerate() {
        lines.push(csv_row([
            (i + 1).to_string(),
            js_number(loss),
            js_number(result.gradient_norm_history[i]),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

/// `function rlSummary(...)`.
fn rl_summary(
    title: &str,
    episodes: usize,
    success_rate: f64,
    mean_length: f64,
    updates: u64,
    topology: &StationGraphSummary,
) -> String {
    [
        title.to_string(),
        "----------------------------------------".to_string(),
        format!("  Episodes:       {episodes}"),
        format!("  Greedy success: {:.1}%", 100.0 * success_rate),
        format!("  Greedy length:  {:.2}", mean_length),
        format!("  Updates:        {updates}"),
        format!("  Stations:       {}", topology.stations.join(" -> ")),
        format!("  Movables:       {}", topology.movables.join(", ")),
    ]
    .join("\n")
}

/// `[csvRow(['sample','prediction','residual']), ...]` for the regression models.
fn write_regression_csv(result: &LinearRegressionResult, csv_path: &str) {
    let mut lines = vec![csv_row(["sample", "prediction", "residual"])];
    for (i, &p) in result.predictions.iter().enumerate() {
        lines.push(csv_row([
            i.to_string(),
            js_number(p),
            js_number(result.residuals[i]),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

fn linear_like_summary(title: &str, result: &LinearRegressionResult) -> String {
    [
        title.to_string(),
        "------------------------------------".to_string(),
        format!("  Samples:        {}", result.sample_count),
        format!(
            "  Coefficients:   [{}]",
            result
                .coefficients
                .iter()
                .map(|v| format!("{v:.6}"))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        format!("  Intercept:      {:.6}", result.intercept),
        format!("  MSE:            {}", to_exponential(result.mse, 3)),
        format!(
            "  Stations:       {}",
            result.topology.stations.join(" -> ")
        ),
        format!("  Movables:       {}", result.topology.movables.join(", ")),
    ]
    .join("\n")
}

fn rl_history_csv(reward_history: &[f64], length_history: &[f64], csv_path: &str) {
    let mut lines = vec![csv_row(["episode", "reward", "length"])];
    for (i, &r) in reward_history.iter().enumerate() {
        lines.push(csv_row([
            i.to_string(),
            js_number(r),
            js_number(length_history[i]),
        ]));
    }
    write_csv_lines(csv_path, &lines);
}

fn example<P>(name: &str, model: &str, description: &str, parameters: P) -> RegistrationExample<P> {
    RegistrationExample {
        name: name.to_string(),
        spec: DESModelSpec {
            schema: DES_MODEL_SPEC_SCHEMA.to_string(),
            model: model.to_string(),
            description: Some(description.to_string()),
            parameters,
            runtime: None,
            metadata: None,
        },
    }
}

// =============================================================================
// 1. linear-regression-ls
// =============================================================================

pub struct LinearRegressionLsAdapter;

pub fn linear_regression_ls_adapter() -> LinearRegressionLsAdapter {
    LinearRegressionLsAdapter
}

impl DESModelRegistration<LinearRegressionParams, LinearRegressionResult>
    for LinearRegressionLsAdapter
{
    fn id(&self) -> &str {
        "linear-regression-ls"
    }
    fn description(&self) -> &str {
        "Least-squares linear regression as DES sample tokens, normal-equation accumulator, and fit sink."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("samples", samples_schema()),
                ("fitIntercept", boolean(Some(true))),
                ("ridge", num(Some(0.0), None, None, Some(0.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        params: LinearRegressionParams,
        _runtime: &DESRuntimeConfig,
    ) -> LinearRegressionResult {
        run_linear_regression_ls(params)
    }
    fn summarize(
        &self,
        result: &LinearRegressionResult,
        _params: &LinearRegressionParams,
    ) -> String {
        linear_like_summary("LINEAR REGRESSION (least squares DES)", result)
    }
    fn write_csv(&self, result: &LinearRegressionResult, csv_path: &str) {
        write_regression_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<LinearRegressionParams>> {
        vec![example(
            "line-y-2x-plus-1",
            "linear-regression-ls",
            "Fit y = 2x + 1 through DES sample and fit tokens.",
            LinearRegressionParams::default(),
        )]
    }
}

// =============================================================================
// 2. ridge-regression-ls
// =============================================================================

pub struct RidgeRegressionLsAdapter;

pub fn ridge_regression_ls_adapter() -> RidgeRegressionLsAdapter {
    RidgeRegressionLsAdapter
}

impl DESModelRegistration<RidgeRegressionParams, LinearRegressionResult>
    for RidgeRegressionLsAdapter
{
    fn id(&self) -> &str {
        "ridge-regression-ls"
    }
    fn description(&self) -> &str {
        "Ridge-regularized least squares using the shared DES sample and normal-equation stations."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("samples", samples_schema()),
                ("fitIntercept", boolean(Some(true))),
                ("ridge", num(Some(0.0), None, None, Some(0.1))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        params: RidgeRegressionParams,
        _runtime: &DESRuntimeConfig,
    ) -> LinearRegressionResult {
        run_ridge_regression_ls(params)
    }
    fn summarize(
        &self,
        result: &LinearRegressionResult,
        _params: &RidgeRegressionParams,
    ) -> String {
        linear_like_summary("RIDGE REGRESSION (least squares DES)", result)
    }
    fn write_csv(&self, result: &LinearRegressionResult, csv_path: &str) {
        write_regression_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<RidgeRegressionParams>> {
        vec![example(
            "regularized-line-fit",
            "ridge-regression-ls",
            "Fit a regularized line through the shared DES least-squares station graph.",
            RidgeRegressionParams {
                samples: None,
                fit_intercept: None,
                ridge: Some(0.1),
            },
        )]
    }
}

// =============================================================================
// 3. logistic-regression-sgd
// =============================================================================

pub struct LogisticRegressionSgdAdapter;

pub fn logistic_regression_sgd_adapter() -> LogisticRegressionSgdAdapter {
    LogisticRegressionSgdAdapter
}

impl DESModelRegistration<LogisticRegressionSGDParams, GradientTrainingResult>
    for LogisticRegressionSgdAdapter
{
    fn id(&self) -> &str {
        "logistic-regression-sgd"
    }
    fn description(&self) -> &str {
        "Binary logistic regression trained by mini-batch gradient tokens and SGD/Adam updates."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("samples", samples_schema()),
                ("epochs", num(Some(1.0), None, Some(true), Some(120.0))),
                ("batchSize", num(Some(1.0), None, Some(true), Some(4.0))),
                ("learningRate", num(Some(1e-9), None, None, Some(0.2))),
                ("optimizer", str_enum(&["sgd", "adam"], "sgd")),
                ("l2", num(Some(0.0), None, None, Some(0.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        params: LogisticRegressionSGDParams,
        _runtime: &DESRuntimeConfig,
    ) -> GradientTrainingResult {
        run_logistic_regression_sgd(params)
    }
    fn summarize(
        &self,
        result: &GradientTrainingResult,
        _params: &LogisticRegressionSGDParams,
    ) -> String {
        gradient_summary("LOGISTIC REGRESSION (mini-batch gradient DES)", result)
    }
    fn write_csv(&self, result: &GradientTrainingResult, csv_path: &str) {
        write_gradient_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<LogisticRegressionSGDParams>> {
        vec![example(
            "separable-points",
            "logistic-regression-sgd",
            "Train a binary linear classifier through sample, batch, and gradient-update stations.",
            LogisticRegressionSGDParams {
                epochs: Some(120),
                batch_size: Some(3),
                learning_rate: Some(0.2),
                ..Default::default()
            },
        )]
    }
}

// =============================================================================
// 4. backprop-mlp-classifier
// =============================================================================

pub struct BackpropMlpClassifierAdapter;

pub fn backprop_mlp_classifier_adapter() -> BackpropMlpClassifierAdapter {
    BackpropMlpClassifierAdapter
}

impl DESModelRegistration<BackpropMLPParams, GradientTrainingResult>
    for BackpropMlpClassifierAdapter
{
    fn id(&self) -> &str {
        "backprop-mlp-classifier"
    }
    fn description(&self) -> &str {
        "One-hidden-layer MLP trained by explicit backprop gradient tokens over mini-batches."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("samples", samples_schema()),
                ("hiddenUnits", num(Some(1.0), None, Some(true), Some(4.0))),
                ("epochs", num(Some(1.0), None, Some(true), Some(800.0))),
                ("batchSize", num(Some(1.0), None, Some(true), Some(4.0))),
                ("learningRate", num(Some(1e-9), None, None, Some(0.08))),
                ("optimizer", str_enum(&["sgd", "adam"], "adam")),
                ("seed", num(None, None, Some(true), Some(7.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        params: BackpropMLPParams,
        _runtime: &DESRuntimeConfig,
    ) -> GradientTrainingResult {
        run_backprop_mlp_classifier(params)
    }
    fn summarize(&self, result: &GradientTrainingResult, _params: &BackpropMLPParams) -> String {
        gradient_summary("BACKPROP MLP CLASSIFIER (DES)", result)
    }
    fn write_csv(&self, result: &GradientTrainingResult, csv_path: &str) {
        write_gradient_csv(result, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<BackpropMLPParams>> {
        vec![example(
            "xor",
            "backprop-mlp-classifier",
            "Train XOR with explicit mini-batch and backprop update stations.",
            BackpropMLPParams {
                hidden_units: Some(4),
                epochs: Some(800),
                batch_size: Some(4),
                learning_rate: Some(0.08),
                optimizer: Some(Optimizer::Adam),
                seed: Some(7),
                ..Default::default()
            },
        )]
    }
}

// =============================================================================
// 5. policy-gradient-corridor
// =============================================================================

pub struct PolicyGradientCorridorAdapter;

pub fn policy_gradient_corridor_adapter() -> PolicyGradientCorridorAdapter {
    PolicyGradientCorridorAdapter
}

impl DESModelRegistration<PolicyGradientCorridorParams, PolicyGradientCorridorResult>
    for PolicyGradientCorridorAdapter
{
    fn id(&self) -> &str {
        "policy-gradient-corridor"
    }
    fn description(&self) -> &str {
        "REINFORCE-style softmax policy-gradient agent on a corridor environment with train/resume tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("numEpisodes", num(Some(1.0), None, Some(true), Some(300.0))),
                (
                    "maxStepsPerEpisode",
                    num(Some(1.0), None, Some(true), Some(40.0)),
                ),
                ("rolloutLen", num(Some(1.0), None, Some(true), Some(12.0))),
                ("alpha", num(Some(1e-9), None, None, Some(0.04))),
                ("gamma", num(Some(0.0), Some(1.0), None, Some(0.95))),
                ("seed", num(None, None, Some(true), Some(1.0))),
                ("length", num(Some(2.0), None, Some(true), Some(7.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        params: PolicyGradientCorridorParams,
        _runtime: &DESRuntimeConfig,
    ) -> PolicyGradientCorridorResult {
        run_policy_gradient_corridor(params)
    }
    fn summarize(
        &self,
        result: &PolicyGradientCorridorResult,
        _params: &PolicyGradientCorridorParams,
    ) -> String {
        rl_summary(
            "POLICY GRADIENT CORRIDOR (DES)",
            result.reward_history.len(),
            result.greedy_success_rate,
            result.greedy_mean_length,
            result.updates,
            &result.topology,
        )
    }
    fn write_csv(&self, result: &PolicyGradientCorridorResult, csv_path: &str) {
        rl_history_csv(&result.reward_history, &result.length_history, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<PolicyGradientCorridorParams>> {
        vec![example(
            "corridor",
            "policy-gradient-corridor",
            "Softmax policy-gradient agent connected to a corridor environment through DES action/transition tokens.",
            PolicyGradientCorridorParams {
                num_episodes: Some(300),
                max_steps_per_episode: None,
                rollout_len: Some(12),
                alpha: Some(0.04),
                gamma: Some(0.95),
                seed: Some(1),
                length: Some(7),
            },
        )]
    }
}

// =============================================================================
// 6. expected-sarsa-gridworld
// =============================================================================

pub struct ExpectedSarsaGridworldAdapter;

pub fn expected_sarsa_gridworld_adapter() -> ExpectedSarsaGridworldAdapter {
    ExpectedSarsaGridworldAdapter
}

impl DESModelRegistration<ExpectedSarsaGridParams, ExpectedSarsaGridResult>
    for ExpectedSarsaGridworldAdapter
{
    fn id(&self) -> &str {
        "expected-sarsa-gridworld"
    }
    fn description(&self) -> &str {
        "Expected SARSA control on GridWorld using environment state/action/transition tokens."
    }
    fn schema(&self) -> ParamSchema {
        obj(
            vec![
                ("numEpisodes", num(Some(1.0), None, Some(true), Some(900.0))),
                (
                    "maxStepsPerEpisode",
                    num(Some(1.0), None, Some(true), Some(80.0)),
                ),
                ("alpha", num(Some(1e-9), None, None, Some(0.2))),
                ("gamma", num(Some(0.0), Some(1.0), None, Some(0.95))),
                ("epsilon", num(Some(0.0), Some(1.0), None, Some(0.35))),
                ("epsilonDecay", num(Some(0.0), Some(1.0), None, Some(0.995))),
                ("epsilonMin", num(Some(0.0), Some(1.0), None, Some(0.02))),
                ("seed", num(None, None, Some(true), Some(1.0))),
            ],
            vec![],
        )
    }
    fn run(
        &self,
        params: ExpectedSarsaGridParams,
        _runtime: &DESRuntimeConfig,
    ) -> ExpectedSarsaGridResult {
        run_expected_sarsa_gridworld(params)
    }
    fn summarize(
        &self,
        result: &ExpectedSarsaGridResult,
        _params: &ExpectedSarsaGridParams,
    ) -> String {
        [
            "EXPECTED SARSA GRIDWORLD (DES)".to_string(),
            "----------------------------------------".to_string(),
            format!("  Episodes:       {}", result.reward_history.len()),
            format!(
                "  Greedy reaches: {} in {} steps",
                if result.greedy_reached { "yes" } else { "no" },
                result.greedy_len
            ),
            format!(
                "  Q(start):       [{}]",
                result
                    .q_start
                    .iter()
                    .map(|v| format!("{v:.3}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            format!(
                "  Stations:       {}",
                result.topology.stations.join(" -> ")
            ),
            format!("  Movables:       {}", result.topology.movables.join(", ")),
        ]
        .join("\n")
    }
    fn write_csv(&self, result: &ExpectedSarsaGridResult, csv_path: &str) {
        rl_history_csv(&result.reward_history, &result.length_history, csv_path);
    }
    fn examples(&self) -> Vec<RegistrationExample<ExpectedSarsaGridParams>> {
        vec![example(
            "gridworld",
            "expected-sarsa-gridworld",
            "Expected SARSA learns GridWorld through DES state/action/transition movables.",
            ExpectedSarsaGridParams {
                num_episodes: Some(900),
                max_steps_per_episode: None,
                alpha: Some(0.2),
                gamma: Some(0.95),
                epsilon: Some(0.35),
                epsilon_decay: Some(0.995),
                epsilon_min: None,
                seed: Some(1),
            },
        )]
    }
}
