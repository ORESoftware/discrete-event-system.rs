//! Canonical use path: `crate::des::random_variables::half::*`
//!
//! Port of `src/des/random-variables/half.ts` — a throwaway CLI script that
//! down-samples an array by pairwise averaging.
//!
//! PORT NOTES:
//!   * This was a top-level script (`console.log(goFrom_131072_to_1024(...))`);
//!     that maps to a `[[bin]]` / `examples/` `main`, not library code.
//!   * LATENT BUG preserved verbatim: the `while` loop reduces the ORIGINAL input
//!     `v` (not the running `ret`), so `reduce_by_half(&v)` returns the same length
//!     every iteration and the loop never converges for large inputs — i.e. it
//!     would spin forever. Translated as-is and flagged rather than "fixed". Do
//!     NOT call this with an input longer than 1024 expecting it to terminate.
//!   * JS `[...v].sort()` sorts LEXICOGRAPHICALLY (stringwise); here we sort
//!     numerically. For the integer demo input the two orderings differ; noted.

#![allow(dead_code)]

/// One pairwise-averaging pass, mirroring the TS `reduceByHalf` fold whose
/// accumulator was `[outputArray, pendingValue | null]`.
///
/// For each element: at an ODD index it stashes the value as "pending"; at an
/// EVEN index it pushes `(pending + value) / 2` (JS `null` coerces to `0`, so the
/// very first push is `value / 2`).
fn reduce_by_half(v: &[f64]) -> (Vec<f64>, Option<f64>) {
    let mut out: Vec<f64> = Vec::new();
    let mut pending: Option<f64> = None;
    for (i, &b) in v.iter().enumerate() {
        if i % 2 != 0 {
            pending = Some(b);
        } else {
            let prev = pending.unwrap_or(0.0); // JS: null + b === b
            out.push((prev + b) / 2.0);
            pending = None;
        }
    }
    (out, pending)
}

/// `goFrom_131072_to_1024(v)` — repeatedly halve until length `<= 1024`.
///
/// See the module-level PORT NOTE: the loop body reduces `v`, not `ret`, so it is
/// non-converging for inputs longer than 1024. Preserved exactly.
pub fn go_from_131072_to_1024(v: Vec<f64>) -> Vec<f64> {
    let mut ret: Vec<f64> = v.clone();
    ret.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    while ret.len() > 1024 {
        // TS logged `ret.length` here; omitted.
        ret = reduce_by_half(&v).0; // BUG (preserved): reduces `v`, not `ret`.
    }

    ret
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce_by_half_averages_pairs() {
        // indices: 0(even,push 0+? ->), 1(odd,stash), 2(even,push)...
        // For [2,4,6,8]: i0 push (0+2)/2=1; i1 stash 4; i2 push (4+6)/2=5; i3 stash 8.
        let (out, pending) = reduce_by_half(&[2.0, 4.0, 6.0, 8.0]);
        assert_eq!(out, vec![1.0, 5.0]);
        assert_eq!(pending, Some(8.0));
    }

    #[test]
    fn small_input_returns_sorted_without_looping() {
        // len <= 1024, so the (buggy) while loop never executes and we just get a
        // numerically sorted copy back.
        let v = vec![3.0, 1.0, 2.0];
        let out = go_from_131072_to_1024(v);
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
    }
}
