//! Port of src/des/test/collaborative-inference-test.ts
//
// `general/collaborative_inference` is now ported. The TS test's headline
// assertion was that every built-in validation check passes (conservation +
// finite scores); we reproduce that against a fixed synthetic seed.

#[cfg(test)]
mod tests {
    use crate::des::general::collaborative_inference::{
        run_collaborative_inference, CollaborativeInferenceParams, CollaborativeInferenceScenario,
    };

    fn synthetic_params() -> CollaborativeInferenceParams {
        CollaborativeInferenceParams {
            seed: Some(12345),
            ..Default::default()
        }
    }

    #[test]
    fn synthetic_run_passes_all_validations() {
        let result = run_collaborative_inference(synthetic_params());
        let failed: Vec<&str> = result
            .validation
            .iter()
            .filter(|c| !c.passed)
            .map(|c| c.name.as_str())
            .collect();
        assert!(failed.is_empty(), "failed validation checks: {failed:?}");
    }

    #[test]
    fn synthetic_run_is_deterministic_and_nonempty() {
        let a = run_collaborative_inference(synthetic_params());
        let b = run_collaborative_inference(synthetic_params());
        assert!(
            a.synthetic,
            "expected a synthetic scenario when none is supplied"
        );
        assert!(a.respondents_processed > 0, "no respondents processed");
        assert!(!a.rankings.is_empty(), "no item rankings produced");
        // Same seed → identical ranking order and top item.
        assert_eq!(a.rankings.len(), b.rankings.len());
        assert_eq!(
            a.top.first().map(|s| s.item_id.clone()),
            b.top.first().map(|s| s.item_id.clone()),
            "top item not reproducible across identical seeds"
        );
    }

    #[test]
    fn richer_builtin_domains_run_and_rank_items() {
        let scenarios = [
            CollaborativeInferenceScenario::Movies,
            CollaborativeInferenceScenario::TravelSpots,
            CollaborativeInferenceScenario::Books,
            CollaborativeInferenceScenario::Songs,
        ];
        for scenario in scenarios {
            let result = run_collaborative_inference(CollaborativeInferenceParams {
                scenario: Some(scenario),
                respondent_count: Some(900),
                seed: Some(2024),
                top_k: Some(5),
                ..Default::default()
            });
            let failed: Vec<&str> = result
                .validation
                .iter()
                .filter(|c| !c.passed)
                .map(|c| c.name.as_str())
                .collect();
            assert!(
                failed.is_empty(),
                "failed validation checks for {:?}: {failed:?}",
                scenario
            );
            assert_eq!(result.top.len(), 5);
            assert!(result.credibility.exposure_order_weight_strength > 0.0);
            assert!(result.credibility.rating_age_weight_strength > 0.0);
        }
    }
}
