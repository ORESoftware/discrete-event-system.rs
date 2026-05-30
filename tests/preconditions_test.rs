//! TypeScript source: `src/des/test/preconditions-test.ts`
//! Rust target: `tests/preconditions_test.rs`

use discrete_event_system_rs::des::general::des_base::preconditions::{
    PreconditionError, Preconditions,
};

fn expect_precondition_error(
    result: Result<(), PreconditionError>,
    param_fragment: &str,
) -> PreconditionError {
    let err = result.expect_err("expected a PreconditionError");
    assert!(
        err.to_string()
            .to_lowercase()
            .contains(&param_fragment.to_lowercase()),
        "error `{err}` did not mention `{param_fragment}`"
    );
    err
}

#[test]
fn low_level_guards_reject_bad_numeric_inputs() {
    expect_precondition_error(Preconditions::finite("m", "x", f64::NAN), "x");
    expect_precondition_error(Preconditions::finite("m", "x", f64::INFINITY), "x");
    expect_precondition_error(Preconditions::positive("m", "x", 0.0), "x");
    expect_precondition_error(Preconditions::positive("m", "x", -0.5), "x");
    expect_precondition_error(Preconditions::in_range("m", "x", 1.2, 0.0, 1.0), "x");
    expect_precondition_error(Preconditions::in_range("m", "x", -0.2, 0.0, 1.0), "x");
    expect_precondition_error(Preconditions::integer("m", "k", 3.7), "k");
    expect_precondition_error(Preconditions::integer_in_range("m", "k", 11.0, 0, 10), "k");
    expect_precondition_error(Preconditions::not_div_by_zero("m", "d", 0.0, 1e-12), "d");
}

#[test]
fn low_level_guards_reject_bad_vectors_and_matrices() {
    expect_precondition_error(
        Preconditions::probability_vector("m", "p", &[0.5, 0.4, 0.09], 1e-6),
        "p",
    );
    expect_precondition_error(
        Preconditions::probability_vector("m", "p", &[0.5, -0.1, 0.6], 1e-6),
        "p",
    );
    expect_precondition_error(
        Preconditions::symmetric_matrix("m", "M", &[vec![1.0, 2.0], vec![3.0, 4.0]], 1e-9),
        "M",
    );
    expect_precondition_error(
        Preconditions::positive_definite_cholesky("m", "M", &[vec![0.0, 0.0], vec![0.0, 1.0]]),
        "M",
    );
    expect_precondition_error(
        Preconditions::positive_definite_cholesky("m", "M", &[vec![1.0, 2.0], vec![2.0, 1.0]]),
        "M",
    );
    expect_precondition_error(Preconditions::length_eq("m", "arr", &[1, 2], 3), "arr");
}

#[test]
fn valid_inputs_do_not_error() {
    Preconditions::finite("m", "x", 1.5).unwrap();
    Preconditions::positive("m", "x", 0.001).unwrap();
    Preconditions::in_range("m", "x", 0.5, 0.0, 1.0).unwrap();
    Preconditions::probability_vector("m", "p", &[0.4, 0.3, 0.3], 1e-6).unwrap();
    Preconditions::symmetric_matrix("m", "M", &[vec![1.0, 2.0], vec![2.0, 3.0]], 1e-9).unwrap();
    Preconditions::positive_definite_cholesky("m", "M", &[vec![2.0, 1.0], vec![1.0, 2.0]]).unwrap();
    Preconditions::not_div_by_zero("m", "d", 0.5, 1e-12).unwrap();
}

#[test]
fn error_keeps_structured_fields_for_migration_callers() {
    let err = expect_precondition_error(Preconditions::positive("Plant", "dt", 0.0), "dt");
    assert_eq!(err.model, "Plant");
    assert_eq!(err.param, "dt");
    assert!(err.condition.contains("> 0"));
    assert_eq!(err.observed.unwrap(), serde_json::json!(0.0));
}
