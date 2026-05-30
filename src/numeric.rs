//! Numeric policy for the Rust migration.
//!
//! Use `DesDecimal` for base-10 quantities that are compared, accumulated, or
//! serialized as model state. Use `DesRational` when a value is fundamentally a
//! fraction and should remain exact across compound calculations. Keep `f64`
//! for continuous numerical algorithms, random samples, geometry, and library
//! boundaries, and use the helpers here when crossing those boundaries.

use crate::core::{DesDecimal, DesError, DesResult};
use bigdecimal::BigDecimal;
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::Zero;
use rust_decimal::prelude::ToPrimitive;
use std::str::FromStr;

pub type DesBigDecimal = BigDecimal;
pub type DesRational = BigRational;

pub fn decimal_from_f64(value: f64, context: &'static str) -> DesResult<DesDecimal> {
    if !value.is_finite() {
        return Err(DesError::InvalidState {
            context,
            message: format!("expected a finite f64, got {value}"),
        });
    }

    DesDecimal::from_f64_retain(value).ok_or_else(|| DesError::InvalidState {
        context,
        message: format!("could not represent f64 {value} as Decimal"),
    })
}

pub fn decimal_from_str(value: &str, context: &'static str) -> DesResult<DesDecimal> {
    DesDecimal::from_str(value).map_err(|err| DesError::InvalidState {
        context,
        message: format!("could not parse decimal {value:?}: {err}"),
    })
}

pub fn decimal_to_f64(value: DesDecimal, context: &'static str) -> DesResult<f64> {
    value.to_f64().ok_or_else(|| DesError::InvalidState {
        context,
        message: format!("could not represent Decimal {value} as f64"),
    })
}

pub fn decimal_sum<I>(values: I) -> DesDecimal
where
    I: IntoIterator<Item = DesDecimal>,
{
    values
        .into_iter()
        .fold(DesDecimal::ZERO, |sum, value| sum + value)
}

pub fn decimal_mean(values: &[DesDecimal], context: &'static str) -> DesResult<DesDecimal> {
    if values.is_empty() {
        return Err(DesError::InvalidState {
            context,
            message: "expected at least one decimal value".to_owned(),
        });
    }

    Ok(decimal_sum(values.iter().copied()) / DesDecimal::from(values.len() as u64))
}

pub fn rational(
    numerator: i128,
    denominator: i128,
    context: &'static str,
) -> DesResult<DesRational> {
    if denominator == 0 {
        return Err(DesError::InvalidState {
            context,
            message: "rational denominator must be non-zero".to_owned(),
        });
    }

    Ok(DesRational::new(
        BigInt::from(numerator),
        BigInt::from(denominator),
    ))
}

pub fn rational_sum<I>(values: I) -> DesRational
where
    I: IntoIterator<Item = DesRational>,
{
    values
        .into_iter()
        .fold(DesRational::zero(), |sum, value| sum + value)
}

pub fn rational_mean(values: &[DesRational], context: &'static str) -> DesResult<DesRational> {
    if values.is_empty() {
        return Err(DesError::InvalidState {
            context,
            message: "expected at least one rational value".to_owned(),
        });
    }

    Ok(
        rational_sum(values.iter().cloned())
            / DesRational::from_integer(BigInt::from(values.len())),
    )
}

pub fn absolute_decimal(value: DesDecimal) -> DesDecimal {
    value.abs()
}

pub fn approximately_equal_f64(left: f64, right: f64, tolerance: f64) -> bool {
    if !left.is_finite() || !right.is_finite() || tolerance < 0.0 {
        return false;
    }

    let scale = left.abs().max(right.abs()).max(1.0);
    (left - right).abs() <= tolerance * scale
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, value: f64) -> DesResult<()> {
        if !value.is_finite() {
            return Err(DesError::InvalidState {
                context: "CompensatedSum::add",
                message: format!("expected a finite f64, got {value}"),
            });
        }

        let next = self.sum + value;
        if self.sum.abs() >= value.abs() {
            self.correction += (self.sum - next) + value;
        } else {
            self.correction += (value - next) + self.sum;
        }
        self.sum = next;
        Ok(())
    }

    pub fn total(self) -> f64 {
        self.sum + self.correction
    }
}

pub fn compensated_sum<I>(values: I) -> DesResult<f64>
where
    I: IntoIterator<Item = f64>,
{
    let mut sum = CompensatedSum::new();
    for value in values {
        sum.add(value)?;
    }
    Ok(sum.total())
}
