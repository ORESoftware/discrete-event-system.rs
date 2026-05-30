//! Numeric policy tests for exact decimals, exact fractions, and deliberate f64 use.

use discrete_event_system_rs::numeric::{
    absolute_decimal, approximately_equal_f64, compensated_sum, decimal_from_f64, decimal_from_str,
    decimal_mean, decimal_sum, decimal_to_f64, rational, rational_mean,
};
use discrete_event_system_rs::DesDecimal;

#[test]
fn decimal_arithmetic_keeps_base10_values_exact() {
    let one_tenth = decimal_from_str("0.1", "test").expect("decimal parses");
    let two_tenths = decimal_from_str("0.2", "test").expect("decimal parses");
    let three_tenths = decimal_from_str("0.3", "test").expect("decimal parses");

    assert_eq!(one_tenth + two_tenths, three_tenths);
    assert_eq!(
        decimal_mean(&[one_tenth, two_tenths, three_tenths], "test").expect("mean"),
        decimal_from_str("0.2", "test").expect("decimal parses")
    );
}

#[test]
fn rational_arithmetic_keeps_fractional_values_exact() {
    let one_third = rational(1, 3, "test").expect("rational");
    let one_sixth = rational(1, 6, "test").expect("rational");
    let one_half = rational(1, 2, "test").expect("rational");

    assert_eq!(one_third + one_sixth, one_half);
    assert_eq!(
        rational_mean(
            &[
                rational(1, 3, "test").expect("rational"),
                rational(2, 3, "test").expect("rational"),
            ],
            "test"
        )
        .expect("mean"),
        rational(1, 2, "test").expect("rational")
    );
}

#[test]
fn decimal_from_f64_rejects_non_finite_values() {
    assert!(decimal_from_f64(f64::NAN, "test").is_err());
    assert!(decimal_from_f64(f64::INFINITY, "test").is_err());
}

#[test]
fn rational_constructor_rejects_zero_denominator() {
    assert!(rational(1, 0, "test").is_err());
}

#[test]
fn compensated_sum_limits_binary_float_accumulation_error() {
    let values = [1e16, 1.0, -1e16];
    let naive: f64 = values.iter().copied().sum();

    assert_eq!(naive, 0.0);
    assert_eq!(compensated_sum(values).expect("finite values"), 1.0);
}

#[test]
fn decimal_policy_keeps_integer_conversion_lossless() {
    assert_eq!(
        DesDecimal::from(42),
        decimal_from_str("42", "test").unwrap()
    );
}

#[test]
fn decimal_sum_accumulates_base10_values_exactly() {
    let values = ["0.10", "0.20", "0.30", "0.40"]
        .into_iter()
        .map(|value| decimal_from_str(value, "test").expect("decimal parses"));

    assert_eq!(
        decimal_sum(values),
        decimal_from_str("1.00", "test").expect("decimal parses")
    );
}

#[test]
fn decimal_mean_rejects_empty_inputs() {
    let err = decimal_mean(&[], "empty decimal mean").expect_err("empty input is rejected");

    assert!(format!("{err}").contains("expected at least one decimal value"));
}

#[test]
fn decimal_to_f64_preserves_simple_boundary_values() {
    let value = decimal_from_str("12.5", "test").expect("decimal parses");

    assert_eq!(
        decimal_to_f64(value, "test").expect("decimal converts"),
        12.5
    );
}

#[test]
fn absolute_decimal_normalizes_negative_values() {
    let value = decimal_from_str("-123.456", "test").expect("decimal parses");

    assert_eq!(
        absolute_decimal(value),
        decimal_from_str("123.456", "test").expect("decimal parses")
    );
}

#[test]
fn approximate_f64_equality_rejects_invalid_inputs() {
    assert!(approximately_equal_f64(100.0, 100.000_000_1, 1e-8));
    assert!(!approximately_equal_f64(100.0, 100.1, 1e-8));
    assert!(!approximately_equal_f64(f64::NAN, 1.0, 1e-8));
    assert!(!approximately_equal_f64(1.0, f64::INFINITY, 1e-8));
    assert!(!approximately_equal_f64(1.0, 1.0, -1e-8));
}
