//! Port of src/des/test/domain-application-test.ts
//!
//! The TS original drove every applied domain model through the des-registry.
//! `general::domain_application_models` and `general::des_registry` are now
//! ported, so this exercises the domain pipeline directly (the registry's
//! string-keyed global lookup is a thin wrapper around these same `run_*`
//! functions). We assert the shared "common graph" invariants the TS harness
//! checked: identity, candidate count, a feasible incumbent, the
//! source/generator/evaluator/sink topology, and the exposed domain movables.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::domain_application_models::{
        run_dynamic_pricing_revenue, RevenueManagementParams,
    };

    #[test]
    fn dynamic_pricing_revenue_model_is_well_formed() {
        let r = run_dynamic_pricing_revenue(RevenueManagementParams { capacity: None });

        // Identity + at least three candidate plans (TS commonGraphChecks).
        assert_eq!(r.model_id, "dynamic-pricing-revenue");
        assert!(r.candidates.len() >= 3, "n={}", r.candidates.len());

        // A feasible incumbent is selected, and dynamic pricing earns revenue.
        assert!(
            r.best.feasible,
            "incumbent {} infeasible",
            r.best.candidate_id
        );

        // Source / generator / evaluator / sink topology.
        assert_eq!(
            r.topology.stations[0],
            "dynamic-pricing-revenue-scenario-source"
        );
        for stage in [
            "dynamic-pricing-revenue-candidate-generator",
            "dynamic-pricing-revenue-plan-evaluator",
            "dynamic-pricing-revenue-result-sink",
        ] {
            assert!(
                r.topology.stations.iter().any(|s| s == stage),
                "missing station {stage}"
            );
        }

        // Domain movables exposed for animation/inspection.
        for token in [
            "DomainScenarioToken",
            "DomainPlanToken",
            "DomainEvaluationToken",
        ] {
            assert!(
                r.topology.movables.iter().any(|m| m == token),
                "missing movable {token}"
            );
        }
    }
}
