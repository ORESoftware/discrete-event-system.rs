//! Port of src/des/test/nonlinear-forecasting-test.ts
//!
//! Tests the nonlinear MDP/POMDP forecasting model.
//!
//! PORT NOTE: the "registry and JSON smoke" group (get_model / run_from_spec)
//! depends on `des-registry`, which is not yet ported; it is deferred.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::nonlinear_forecasting_model::{
        run_nonlinear_mdp_pomdp_forecast, NonlinearMDPPOMDPForecastParams, VariableSource,
    };

    #[test]
    fn nonlinear_mdp_pomdp_forecast() {
        let r = run_nonlinear_mdp_pomdp_forecast(NonlinearMDPPOMDPForecastParams::default());

        let has_station = |id: &str| r.topology.stations.iter().any(|s| s == id);
        assert!(has_station("nonlinear-forecast-data-source"));
        assert!(has_station("pomdp-latent-variable-station"));
        assert!(has_station("mdp-variable-discovery-station"));
        assert!(has_station("nonlinear-equation-tuning-station"));
        assert!(has_station("forecast-projection-station") && has_station("forecast-result-sink"));

        for m in [
            "ForecastDataToken",
            "LatentBeliefTraceToken",
            "DiscoveredVariablesToken",
            "FineTunedEquationToken",
            "ForecastProjectionToken",
        ] {
            assert!(r.topology.movables.iter().any(|x| x == m), "missing movable {m}");
        }

        assert_eq!(r.pomdp.points.len(), 42);
        assert!(r.mdp.states >= 512 && r.mdp.actions >= 8, "states={} actions={}", r.mdp.states, r.mdp.actions);
        assert!(r.discovered_variables.iter().any(|v| v.source == VariableSource::Pomdp));
        assert!(r.metrics.validation_mse < r.metrics.baseline_validation_mse);
        assert!(r.metrics.forecast_mse < r.metrics.baseline_forecast_mse);
        assert_eq!(r.projection.len(), 8);
        assert!(r
            .projection
            .iter()
            .all(|p| p.forecast.is_finite() && p.lower.is_finite() && p.upper.is_finite()));
        let trace = &r.equation.trace;
        assert!(trace[trace.len() - 1].mse < trace[0].mse);
    }
}
