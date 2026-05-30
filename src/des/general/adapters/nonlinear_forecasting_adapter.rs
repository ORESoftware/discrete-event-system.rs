//! Port of `src/des/general/adapters/nonlinear-forecasting-adapter.ts`
//! (module `des::general::adapters::nonlinear_forecasting_adapter`).
//!
//! JSON adapter for the nonlinear MDP/POMDP forecasting model.
//!
//! ## Conversion notes
//!
//!   * Thin adapter: `run` calls the model fn; the schema carries the defaults.
//!   * `row.split` / `row.beliefMode` are enums in the Rust tree; their TS
//!     string form comes from `as_str()`. Empty CSV cells (`''`) for the fit
//!     rows that lack a lower/upper band become empty strings.
//!
//! PORT NOTE: `animate` (`animateNonlinearForecast` + `buildForecastFrame` +
//! the `drawStationPipeline` / `drawVariables` / `drawActivePanels` helpers)
//! depends on the un-ported animation subsystem (`animation/frame-recorder`,
//! `animation/types::Shape`). It is a documented no-op here; the integrator
//! should port those helpers once the animation crate exists. `registerModel` /
//! the model registry is also not ported — the adapter is exposed via
//! [`adapter()`].

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{csv_row, write_csv_lines};
use crate::des::general::des_spec::{DESModelRegistration, DESRuntimeConfig, ParamSchema};
use crate::des::general::nonlinear_forecasting_model::{
    run_nonlinear_mdp_pomdp_forecast, NonlinearMDPPOMDPForecastParams,
    NonlinearMDPPOMDPForecastResult,
};

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

/// `const nonlinearForecastSchema`.
pub fn nonlinear_forecast_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            (
                "trainingPeriods".to_string(),
                num(Some(18.0), Some(200.0), Some(true), Some(42.0)),
            ),
            (
                "forecastHorizon".to_string(),
                num(Some(1.0), Some(80.0), Some(true), Some(8.0)),
            ),
            (
                "mdpBudget".to_string(),
                num(Some(1.0), Some(10.0), Some(true), Some(6.0)),
            ),
            ("ridge".to_string(), num(Some(0.0), None, None, Some(0.03))),
            (
                "fineTuneIterations".to_string(),
                num(Some(1.0), Some(200.0), Some(true), Some(18.0)),
            ),
            (
                "validationShare".to_string(),
                num(Some(0.1), Some(0.5), None, Some(0.25)),
            ),
        ],
        required: Some(vec![]),
        description: None,
    }
}

/// `registerModel<NonlinearMDPPOMDPForecastParams, NonlinearMDPPOMDPForecastResult>`.
pub struct NonlinearForecastAdapter;

/// Construct the adapter (see the module's PORT NOTE about registration).
pub fn adapter() -> NonlinearForecastAdapter {
    NonlinearForecastAdapter
}

impl DESModelRegistration<NonlinearMDPPOMDPForecastParams, NonlinearMDPPOMDPForecastResult>
    for NonlinearForecastAdapter
{
    fn id(&self) -> &str {
        "nonlinear-mdp-pomdp-forecast"
    }

    fn description(&self) -> &str {
        "Nonlinear forecasting: POMDP latent-variable discovery plus MDP feature selection and equation fine-tuning."
    }

    fn schema(&self) -> ParamSchema {
        nonlinear_forecast_schema()
    }

    fn run(
        &self,
        params: NonlinearMDPPOMDPForecastParams,
        _runtime: &DESRuntimeConfig,
    ) -> NonlinearMDPPOMDPForecastResult {
        run_nonlinear_mdp_pomdp_forecast(params)
    }

    fn summarize(
        &self,
        result: &NonlinearMDPPOMDPForecastResult,
        _params: &NonlinearMDPPOMDPForecastParams,
    ) -> String {
        [
            "NONLINEAR MDP/POMDP FORECAST".to_string(),
            "----------------------------------------".to_string(),
            format!(
                "  Selected variables:       {}",
                result.selected_variables.join(", ")
            ),
            format!(
                "  Validation MSE:           {:.4} (baseline {:.4})",
                result.metrics.validation_mse, result.metrics.baseline_validation_mse
            ),
            format!(
                "  Forecast MSE:             {:.4} (baseline {:.4})",
                result.metrics.forecast_mse, result.metrics.baseline_forecast_mse
            ),
            format!(
                "  POMDP final entropy:      {:.4}",
                result.metrics.final_belief_entropy
            ),
            format!(
                "  MDP states/actions:       {}/{}",
                result.mdp.states, result.mdp.actions
            ),
            format!(
                "  Equation:                 {}",
                result.equation.equation_text
            ),
            format!(
                "  Stations:                 {}",
                result.topology.stations.join(" -> ")
            ),
            format!(
                "  Movables:                 {}",
                result.topology.movables.join(", ")
            ),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &NonlinearMDPPOMDPForecastResult, csv_path: &str) {
        let mut lines = vec![csv_row([
            "kind",
            "t",
            "actual",
            "predicted",
            "lower",
            "upper",
            "split_or_belief_mode",
        ])];
        for row in &result.equation.fitted {
            lines.push(csv_row([
                "fit".to_string(),
                row.t.to_string(),
                row.actual.to_string(),
                row.predicted.to_string(),
                String::new(),
                String::new(),
                row.split.as_str().to_string(),
            ]));
        }
        for row in &result.projection {
            lines.push(csv_row([
                "forecast".to_string(),
                row.t.to_string(),
                row.actual.to_string(),
                row.forecast.to_string(),
                row.lower.to_string(),
                row.upper.to_string(),
                row.belief_mode.as_str().to_string(),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }

    // PORT NOTE: `animate` is intentionally left as the trait's no-op default;
    // see the module-level PORT NOTE about the un-ported animation subsystem.
}
