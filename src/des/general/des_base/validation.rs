//! Port of `src/des/general/des-base/validation.ts`.
//!
//! Validator protocol for DES stations + factory helpers. `interface
//! Validator<S>` → `trait Validator<S>`; the factories (which capture
//! `extract`/`predicate`/`compare` closures) become a closure-backed
//! [`FnValidator<S>`]. `runValidators`' `try/catch` → `catch_unwind` so one
//! buggy validator never blocks the rest.

use std::panic::{catch_unwind, AssertUnwindSafe};

/// A single pass/fail check produced by a `Validator`.
#[derive(Clone, Debug, Default)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub observed: Option<String>,
    pub expected: Option<String>,
    pub group: Option<String>,
    pub details: Option<String>,
}

/// A pluggable validator for a DES station. `S` is the station type (often
/// `dyn DESStation`, with the closure downcasting to the concrete station).
pub trait Validator<S: ?Sized> {
    fn name(&self) -> &str;
    fn validate(&self, station: &S) -> Vec<ValidationCheck>;
}

/// Closure-backed validator — the Rust home for all the TS factory helpers,
/// each of which returned an object literal capturing closures.
pub struct FnValidator<S: ?Sized + 'static> {
    name: String,
    f: Box<dyn Fn(&S) -> Vec<ValidationCheck>>,
}

impl<S: ?Sized + 'static> FnValidator<S> {
    pub fn new(name: impl Into<String>, f: impl Fn(&S) -> Vec<ValidationCheck> + 'static) -> Self {
        FnValidator { name: name.into(), f: Box::new(f) }
    }

    /// Box as a trait object for storage on a station.
    pub fn boxed(self) -> Box<dyn Validator<S>> {
        Box::new(self)
    }
}

impl<S: ?Sized + 'static> Validator<S> for FnValidator<S> {
    fn name(&self) -> &str {
        &self.name
    }
    fn validate(&self, station: &S) -> Vec<ValidationCheck> {
        (self.f)(station)
    }
}

/// Run a list of validators, capturing panics as failed checks.
pub fn run_validators<S: ?Sized>(station: &S, validators: &[Box<dyn Validator<S>>]) -> Vec<ValidationCheck> {
    let mut out = Vec::new();
    for v in validators {
        let result = catch_unwind(AssertUnwindSafe(|| v.validate(station)));
        match result {
            Ok(checks) => out.extend(checks),
            Err(e) => {
                let msg = panic_message(&e);
                eprintln!("[validation] validator \"{}\" panicked during validate(): {msg} — recording as a failed check.", v.name());
                out.push(ValidationCheck {
                    name: format!("{}/threw", v.name()),
                    passed: false,
                    details: Some(msg),
                    ..Default::default()
                });
            }
        }
    }
    out
}

fn panic_message(e: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = e.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic".to_string()
    }
}

/// Format a `ValidationCheck` list as a multi-line report.
pub fn format_validation_report(checks: &[ValidationCheck]) -> String {
    if checks.is_empty() {
        return "(no validators registered)".to_string();
    }
    let mut lines: Vec<String> = Vec::new();
    let mut pass = 0;
    let mut fail = 0;
    let mut cur_group = String::new();
    for c in checks {
        let g = c.group.clone().unwrap_or_default();
        if g != cur_group {
            if !g.is_empty() {
                lines.push(format!("  ─── {g} ───"));
            }
            cur_group = g;
        }
        let tag = if c.passed { "PASS" } else { "FAIL" };
        let obs_part = c.observed.as_ref().map(|o| format!("  observed={o}")).unwrap_or_default();
        let exp_part = c.expected.as_ref().map(|e| format!("  expected={e}")).unwrap_or_default();
        let det_part = if c.passed { String::new() } else {
            c.details.as_ref().map(|d| format!("  ({d})")).unwrap_or_default()
        };
        lines.push(format!("  {tag}  {}{obs_part}{exp_part}{det_part}", c.name));
        if c.passed {
            pass += 1;
        } else {
            fail += 1;
        }
    }
    lines.push(format!("  {}", "-".repeat(64)));
    lines.push(format!("  {pass} passed, {fail} failed"));
    lines.join("\n")
}

// -----------------------------------------------------------------------------
// FACTORIES
// -----------------------------------------------------------------------------

/// Wrap an arbitrary `(station) → bool` predicate into a Validator.
pub fn intrinsic_check<S: ?Sized + 'static>(
    name: impl Into<String>,
    predicate: impl Fn(&S) -> bool + 'static,
    expected: Option<String>,
    observed_fn: Option<Box<dyn Fn(&S) -> String>>,
    group: Option<String>,
    details: Option<String>,
) -> FnValidator<S> {
    let name = name.into();
    let n2 = name.clone();
    FnValidator::new(name, move |s: &S| {
        let passed = predicate(s);
        vec![ValidationCheck {
            name: n2.clone(),
            passed,
            observed: observed_fn.as_ref().map(|f| f(s)),
            expected: expected.clone(),
            group: group.clone(),
            details: if passed { None } else { details.clone() },
        }]
    })
}

/// Comparison mode for [`numeric_validator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NumericMode {
    Absolute,
    Relative,
}

/// Compare a scalar extracted from the station against a (possibly
/// station-dependent) reference.
pub fn numeric_validator<S: ?Sized + 'static>(
    name: impl Into<String>,
    extract: impl Fn(&S) -> f64 + 'static,
    expected: impl Fn(&S) -> f64 + 'static,
    tol: f64,
    mode: NumericMode,
    group: Option<String>,
) -> FnValidator<S> {
    let name = name.into();
    let n2 = name.clone();
    FnValidator::new(name, move |s: &S| {
        let obs = extract(s);
        let exp = expected(s);
        if !obs.is_finite() || !exp.is_finite() {
            return vec![ValidationCheck {
                name: n2.clone(),
                passed: false,
                observed: Some(obs.to_string()),
                expected: Some(exp.to_string()),
                group: group.clone(),
                details: Some("non-finite value".to_string()),
            }];
        }
        let diff = (obs - exp).abs();
        let denom = if mode == NumericMode::Relative { exp.abs().max(1e-12) } else { 1.0 };
        let err = diff / denom;
        let passed = err <= tol;
        vec![ValidationCheck {
            name: n2.clone(),
            passed,
            observed: Some(format!("{obs:.8}")),
            expected: Some(format!("{exp:.8}")),
            group: group.clone(),
            details: if passed { None } else { Some(format!("{mode:?}-err={err:.2e} > tol={tol}")) },
        }]
    })
}

/// Assert a numeric extract is in `[low, high]` (closed by default).
pub fn bound_validator<S: ?Sized + 'static>(
    name: impl Into<String>,
    extract: impl Fn(&S) -> f64 + 'static,
    low: f64,
    high: f64,
    inclusive: bool,
    group: Option<String>,
) -> FnValidator<S> {
    let name = name.into();
    let n2 = name.clone();
    FnValidator::new(name, move |s: &S| {
        let v = extract(s);
        let in_lo = if inclusive { v >= low } else { v > low };
        let in_hi = if inclusive { v <= high } else { v < high };
        let passed = in_lo && in_hi;
        let (ob, cb) = if inclusive { ('[', ']') } else { ('(', ')') };
        vec![ValidationCheck {
            name: n2.clone(),
            passed,
            observed: Some(v.to_string()),
            expected: Some(format!("{ob}{low}, {high}{cb}")),
            group: group.clone(),
            details: if passed { None } else {
                Some(format!("value {v} outside {} interval [{low}, {high}]", if inclusive { "closed" } else { "open" }))
            },
        }]
    })
}

/// Monotonicity direction for [`monotonicity_validator`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Monotonicity {
    NonIncreasing,
    NonDecreasing,
}

/// Assert that an extracted sequence is monotone in `direction`.
pub fn monotonicity_validator<S: ?Sized + 'static>(
    name: impl Into<String>,
    extract: impl Fn(&S) -> Vec<f64> + 'static,
    direction: Monotonicity,
    tol: f64,
    group: Option<String>,
) -> FnValidator<S> {
    let name = name.into();
    let n2 = name.clone();
    FnValidator::new(name, move |s: &S| {
        let xs = extract(s);
        let mut first_violation: isize = -1;
        for i in 1..xs.len() {
            let d = xs[i] - xs[i - 1];
            let ok = match direction {
                Monotonicity::NonIncreasing => d <= tol,
                Monotonicity::NonDecreasing => d >= -tol,
            };
            if !ok {
                first_violation = i as isize;
                break;
            }
        }
        let passed = first_violation == -1;
        let dir = format!("{direction:?}");
        vec![ValidationCheck {
            name: n2.clone(),
            passed,
            observed: Some(if passed { format!("{dir} (n={})", xs.len()) } else { format!("breaks at i={first_violation}") }),
            expected: Some(dir),
            group: group.clone(),
            details: if passed { None } else {
                let i = first_violation as usize;
                Some(format!("xs[{}]={}  xs[{}]={}", i - 1, xs[i - 1], i, xs[i]))
            },
        }]
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dummy {
        value: f64,
        series: Vec<f64>,
    }

    #[test]
    fn numeric_pass_fail() {
        let v = numeric_validator::<Dummy>("v", |s| s.value, |_| 1.0, 1e-6, NumericMode::Absolute, None);
        let good = Dummy { value: 1.0, series: vec![] };
        let bad = Dummy { value: 2.0, series: vec![] };
        assert!(v.validate(&good)[0].passed);
        assert!(!v.validate(&bad)[0].passed);
    }

    #[test]
    fn monotonic_and_report() {
        let v = monotonicity_validator::<Dummy>("m", |s| s.series.clone(), Monotonicity::NonIncreasing, 1e-12, None);
        let good = Dummy { value: 0.0, series: vec![3.0, 2.0, 1.0] };
        let bad = Dummy { value: 0.0, series: vec![1.0, 2.0] };
        assert!(v.validate(&good)[0].passed);
        let checks = v.validate(&bad);
        assert!(!checks[0].passed);
        let report = format_validation_report(&checks);
        assert!(report.contains("FAIL"));
    }

    #[test]
    fn captures_panicking_validator() {
        let bad: Box<dyn Validator<Dummy>> = FnValidator::new("boom", |_: &Dummy| panic!("nope")).boxed();
        let s = Dummy { value: 0.0, series: vec![] };
        let checks = run_validators(&s, std::slice::from_ref(&bad));
        assert_eq!(checks.len(), 1);
        assert!(!checks[0].passed);
        assert!(checks[0].name.ends_with("/threw"));
    }
}
