//! Numeric policy tests for exact decimals, exact fractions, and deliberate f64 use.

use discrete_event_system_rs::numeric::{
    compensated_sum, decimal_from_f64, decimal_from_str, decimal_mean, rational, rational_mean,
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
