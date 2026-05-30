//! Port of src/des/test/test.ts
//!
//! Originally an ad-hoc throughput / distribution probe for the exponential
//! random variables: it sampled 100k inter-event quantities from
//! `ExponentialRandomVariable` and `ExponentialRandomVariable2`, bucketed them,
//! and printed timings — it had no assertions (a benchmark, not a test). The
//! port keeps the sampling loop but turns the histogram into a robust property
//! check: every sampled quantity is a valid (>= -1) integer count and at least
//! one distinct bucket is populated. `Date.now()` timing is dropped.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use crate::des::general::general::bgn;
    use crate::des::random_variables::rv::{
        ExponentialRandomVariable, ExponentialRandomVariable2, RandomVariable,
    };
    use crate::des::shared::capabilities::SeededRandom;

    #[test]
    fn exponential_random_variable_sampling() {
        let ts = bgn(600.0);
        let lambda = bgn(1.0) / bgn(500.0);

        let mut erv =
            ExponentialRandomVariable::new(lambda, ts, Box::new(SeededRandom::new(600)));
        let mut bucket1: HashMap<i64, i64> = HashMap::new();
        for _ in 0..100_000 {
            let val = erv.get_next_event_quantity(ts);
            assert!(val >= -1, "quantity should be >= -1, got {val}");
            *bucket1.entry(val).or_insert(0) += 1;
        }
        assert!(!bucket1.is_empty());

        let mut erv2 =
            ExponentialRandomVariable2::new(lambda, ts, Box::new(SeededRandom::new(601)));
        let mut bucket2: HashMap<i64, i64> = HashMap::new();
        for _ in 0..100_000 {
            let val = erv2.get_next_event_quantity(ts);
            assert!(val >= -1, "quantity should be >= -1, got {val}");
            *bucket2.entry(val).or_insert(0) += 1;
        }
        assert!(!bucket2.is_empty());
    }
}
