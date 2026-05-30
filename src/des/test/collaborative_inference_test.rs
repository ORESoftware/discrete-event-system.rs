//! Port of src/des/test/collaborative-inference-test.ts
//
// `general/collaborative_inference` is now ported. The TS test's headline
// assertion was that every built-in validation check passes (conservation +
// finite scores); we reproduce that against a fixed synthetic seed.

#[cfg(test)]
mod tests {
    use crate::des::general::collaborative_inference::{
        run_collaborative_inference, CollaborativeInferenceParams,
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
}
