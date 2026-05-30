//! Port of src/des/test/validation-test.ts
//!
//! End-to-end test of the validator protocol. Group [1] (the validator factory
//! primitives operating on plain data) is ported faithfully.
//!
//! PORT NOTE:
//!   - `ground_truth_validator` and `external_reference_validator` are not yet
//!     ported in `des_base::validation`, so cases 1.7/1.8 (ground-truth vector
//!     comparison) and groups [5]/[6] (external-reference validators + file I/O)
//!     are deferred.
//!   - groups [2]-[4] and [6] subclass `DESStation`, `TSPSAOptimizer`,
//!     `ValueIterationStation`, `FixedPointIterationStation` etc. and assert on
//!     the `run_iterative_des` validation summary / auto-attached validators;
//!     those rely on TS subclass overrides and the runner's validation
//!     aggregation, and are deferred here.
#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::des_base::validation::{
        bound_validator, format_validation_report, intrinsic_check, monotonicity_validator,
        numeric_validator, run_validators, Monotonicity, NumericMode, ValidationCheck, Validator,
    };

    struct Stub {
        x: f64,
        history: Vec<f64>,
    }

    impl Stub {
        fn scalar(x: f64) -> Self {
            Stub { x, history: Vec::new() }
        }
        fn series(history: Vec<f64>) -> Self {
            Stub { x: 0.0, history }
        }
    }

    struct BrokenValidator;

    impl Validator<Stub> for BrokenValidator {
        fn name(&self) -> &str {
            "t.broken"
        }
        fn validate(&self, _s: &Stub) -> Vec<ValidationCheck> {
            panic!("boom");
        }
    }

    // [1] Validator factories — pure data, no DES.
    #[test]
    fn validator_factories() {
        // numericValidator — absolute tol pass.
        let num_abs = numeric_validator::<Stub>("t.numAbs", |s| s.x, |_| 1.0, 1e-9, NumericMode::Absolute, None);
        assert!(num_abs.validate(&Stub::scalar(1.0))[0].passed);

        // numericValidator — relative tol fail.
        let num_rel = numeric_validator::<Stub>("t.numRel", |s| s.x, |_| 100.0, 1e-3, NumericMode::Relative, None);
        assert!(!num_rel.validate(&Stub::scalar(101.0))[0].passed);

        // boundValidator — inside / outside.
        let bnd = bound_validator::<Stub>("t.bnd", |s| s.x, 0.0, 10.0, true, None);
        assert!(bnd.validate(&Stub::scalar(5.0))[0].passed);
        assert!(!bnd.validate(&Stub::scalar(11.0))[0].passed);

        // monotonicityValidator — non-increasing.
        let mono = monotonicity_validator::<Stub>(
            "t.mono",
            |s| s.history.clone(),
            Monotonicity::NonIncreasing,
            1e-9,
            None,
        );
        assert!(mono.validate(&Stub::series(vec![5.0, 4.0, 3.0, 3.0, 1.0]))[0].passed);
        assert!(!mono.validate(&Stub::series(vec![5.0, 4.0, 5.0, 3.0]))[0].passed);

        // intrinsicCheck — wraps a predicate.
        let ic = intrinsic_check::<Stub>("t.ic", |s| s.x > 0.0, None, None, None, None);
        assert!(ic.validate(&Stub::scalar(7.0))[0].passed);
        assert!(!ic.validate(&Stub::scalar(-1.0))[0].passed);

        // A validator that panics is captured as a failed `/threw` check.
        let validators: Vec<Box<dyn Validator<Stub>>> = vec![Box::new(BrokenValidator)];
        let out = run_validators(&Stub::scalar(0.0), &validators);
        assert!(out.len() == 1 && !out[0].passed && out[0].name.ends_with("/threw"));

        // formatValidationReport renders pass/fail counts.
        let txt = format_validation_report(&[
            ValidationCheck { name: "a".into(), passed: true, ..Default::default() },
            ValidationCheck {
                name: "b".into(),
                passed: false,
                observed: Some("5".into()),
                expected: Some("6".into()),
                details: Some("oops".into()),
                ..Default::default()
            },
        ]);
        assert!(txt.contains("1 passed") && txt.contains("1 failed"));
    }
}
