//! Canonical use path: `crate::des::shared::iterable_int::IterableInt`
//!
//! Port shim for the npm package `iterable.int` (no crate exists).
//!
//! `IterableInt::new(start, end)` is a half-open integer range iterator yielding
//! `start, start+1, ..., end-1` (i.e. `start..end`), matching the JS package's
//! behaviour of iterating the half-open interval.

#![allow(dead_code)]

/// Half-open integer range iterator (`start..end`).
pub struct IterableInt {
    current: i64,
    end: i64,
}

impl IterableInt {
    /// Construct an iterator over `start..end` (end-exclusive).
    pub fn new(start: i64, end: i64) -> Self {
        IterableInt {
            current: start,
            end,
        }
    }
}

impl Iterator for IterableInt {
    type Item = i64;

    fn next(&mut self) -> Option<i64> {
        if self.current < self.end {
            let v = self.current;
            self.current += 1;
            Some(v)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn yields_half_open_range() {
        let collected: Vec<i64> = IterableInt::new(2, 6).collect();
        assert_eq!(collected, vec![2, 3, 4, 5]);
    }

    #[test]
    fn empty_when_start_ge_end() {
        let collected: Vec<i64> = IterableInt::new(5, 5).collect();
        assert!(collected.is_empty());
    }
}
