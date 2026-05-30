//! Rust port of `src/des/general/des-base/validation.ts`.

use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use std::{fmt::Debug, fs, marker::PhantomData};

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/des-base/validation.ts",
    "src/des/general/des_base/validation.rs",
    &[
        "ValidationCheck is a nominal Rust struct with optional string fields.",
        "Validator is a trait returning Result so failed validators become checks.",
        "Factory helpers return boxed trait objects to mirror TS pluggability.",
        "External references use std::fs and serde_json at the module boundary.",
    ],
    &[
        "ValidationCheck",
        "Validator",
        "boundValidator",
        "externalReferenceValidator",
        "formatValidationReport",
        "groundTruthValidator",
        "intrinsicCheck",
        "monotonicityValidator",
        "numericValidator",
        "runValidators",
    ],
);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ValidationCheck {
    pub name: String,
    pub passed: bool,
    pub observed: Option<String>,
    pub expected: Option<String>,
    pub group: Option<String>,
    pub details: Option<String>,
}

impl ValidationCheck {
    pub fn new(name: impl Into<String>, passed: bool) -> Self {
        Self {
            name: name.into(),
            passed,
            observed: None,
            expected: None,
            group: None,
            details: None,
        }
    }

    pub fn with_observed(mut self, observed: impl Into<String>) -> Self {
        self.observed = Some(observed.into());
        self
    }

    pub fn with_expected(mut self, expected: impl Into<String>) -> Self {
        self.expected = Some(expected.into());
        self
    }

    pub fn with_group(mut self, group: impl Into<String>) -> Self {
        self.group = Some(group.into());
        self
    }

    pub fn with_details(mut self, details: impl Into<String>) -> Self {
        self.details = Some(details.into());
        self
    }
}

pub trait Validator<S> {
    fn name(&self) -> &str;
    fn validate(&self, station: &S) -> Result<Vec<ValidationCheck>, String>;
}

pub struct ClosureValidator<S, F>
where
    F: Fn(&S) -> Result<Vec<ValidationCheck>, String>,
{
    name: String,
    validate_fn: F,
    _station: PhantomData<fn(&S)>,
}

impl<S, F> ClosureValidator<S, F>
where
    F: Fn(&S) -> Result<Vec<ValidationCheck>, String>,
{
    pub fn new(name: impl Into<String>, validate_fn: F) -> Self {
        Self {
            name: name.into(),
            validate_fn,
            _station: PhantomData,
        }
    }
}

impl<S, F> Validator<S> for ClosureValidator<S, F>
where
    F: Fn(&S) -> Result<Vec<ValidationCheck>, String>,
{
    fn name(&self) -> &str {
        &self.name
    }

    fn validate(&self, station: &S) -> Result<Vec<ValidationCheck>, String> {
        (self.validate_fn)(station)
    }
}

pub fn run_validators<S>(
    station: &S,
    validators: &[Box<dyn Validator<S>>],
) -> Vec<ValidationCheck> {
    let mut out = Vec::new();
    for validator in validators {
        match validator.validate(station) {
            Ok(mut checks) => out.append(&mut checks),
            Err(err) => {
                eprintln!(
                    "[validation] validator \"{}\" returned an error during validate(): {} - recording as a failed check.",
                    validator.name(),
                    err
                );
                out.push(
                    ValidationCheck::new(format!("{}/threw", validator.name()), false)
                        .with_details(err),
                );
            }
        }
    }
    out
}

pub fn format_validation_report(checks: &[ValidationCheck]) -> String {
    if checks.is_empty() {
        return "(no validators registered)".to_owned();
    }

    let mut lines = Vec::new();
    let mut pass = 0usize;
    let mut fail = 0usize;
    let mut current_group = String::new();

    for check in checks {
        let group = check.group.as_deref().unwrap_or("");
        if group != current_group {
            if !group.is_empty() {
                lines.push(format!("  --- {group} ---"));
            }
            current_group = group.to_owned();
        }

        let tag = if check.passed { "PASS" } else { "FAIL" };
        let observed = check
            .observed
            .as_ref()
            .map(|value| format!("  observed={value}"))
            .unwrap_or_default();
        let expected = check
            .expected
            .as_ref()
            .map(|value| format!("  expected={value}"))
            .unwrap_or_default();
        let details = if check.passed {
            String::new()
        } else {
            check
                .details
                .as_ref()
                .map(|value| format!("  ({value})"))
                .unwrap_or_default()
        };
        lines.push(format!(
            "  {tag}  {}{observed}{expected}{details}",
            check.name
        ));

        if check.passed {
            pass += 1;
        } else {
            fail += 1;
        }
    }

    lines.push(format!("  {}", "-".repeat(64)));
    lines.push(format!("  {pass} passed, {fail} failed"));
    lines.join("\n")
}

pub type PredicateFn<S> = Box<dyn Fn(&S) -> bool>;
pub type ObservedStringFn<S> = Box<dyn Fn(&S) -> String>;
pub type NumericExtractFn<S> = Box<dyn Fn(&S) -> f64>;
pub type SeriesExtractFn<S> = Box<dyn Fn(&S) -> Vec<f64>>;
pub type GroundTruthExtractFn<S, T> = Box<dyn Fn(&S) -> T>;
pub type GroundTruthCompareFn<T> = Box<dyn Fn(&T, &T) -> Option<String>>;
pub type GroundTruthFormatFn<T> = Box<dyn Fn(&T) -> String>;
pub type ExternalReferenceCompareFn<S> = Box<dyn Fn(&S, &JsonValue) -> Vec<ValidationCheck>>;

pub struct IntrinsicCheckOptions<S> {
    pub name: String,
    pub predicate: PredicateFn<S>,
    pub expected: Option<String>,
    pub observed_fn: Option<ObservedStringFn<S>>,
    pub group: Option<String>,
    pub details: Option<String>,
}

impl<S> IntrinsicCheckOptions<S> {
    pub fn new<F>(name: impl Into<String>, predicate: F) -> Self
    where
        F: Fn(&S) -> bool + 'static,
    {
        Self {
            name: name.into(),
            predicate: Box::new(predicate),
            expected: None,
            observed_fn: None,
            group: None,
            details: None,
        }
    }
}

pub fn intrinsic_check<S: 'static>(opts: IntrinsicCheckOptions<S>) -> Box<dyn Validator<S>> {
    let name = opts.name;
    let predicate = opts.predicate;
    let expected = opts.expected;
    let observed_fn = opts.observed_fn;
    let group = opts.group;
    let details = opts.details;
    let validator_name = name.clone();

    Box::new(ClosureValidator::new(validator_name, move |station: &S| {
        let passed = predicate(station);
        let mut check = ValidationCheck::new(name.clone(), passed);
        if let Some(observed_fn) = observed_fn.as_ref() {
            check = check.with_observed(observed_fn(station));
        }
        if let Some(expected) = expected.as_ref() {
            check = check.with_expected(expected.clone());
        }
        if let Some(group) = group.as_ref() {
            check = check.with_group(group.clone());
        }
        if !passed {
            if let Some(details) = details.as_ref() {
                check = check.with_details(details.clone());
            }
        }
        Ok(vec![check])
    }))
}

pub enum ExpectedNumber<S> {
    Value(f64),
    Extractor(Box<dyn Fn(&S) -> f64>),
}

impl<S> ExpectedNumber<S> {
    fn resolve(&self, station: &S) -> f64 {
        match self {
            Self::Value(value) => *value,
            Self::Extractor(extract) => extract(station),
        }
    }
}

impl<S> From<f64> for ExpectedNumber<S> {
    fn from(value: f64) -> Self {
        Self::Value(value)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericMode {
    Absolute,
    Relative,
}

impl NumericMode {
    fn label(self) -> &'static str {
        match self {
            Self::Absolute => "absolute",
            Self::Relative => "relative",
        }
    }
}

pub struct NumericValidatorOptions<S> {
    pub name: String,
    pub extract: NumericExtractFn<S>,
    pub expected: ExpectedNumber<S>,
    pub tol: f64,
    pub mode: NumericMode,
    pub group: Option<String>,
}

impl<S> NumericValidatorOptions<S> {
    pub fn new<F, E>(name: impl Into<String>, extract: F, expected: E) -> Self
    where
        F: Fn(&S) -> f64 + 'static,
        E: Into<ExpectedNumber<S>>,
    {
        Self {
            name: name.into(),
            extract: Box::new(extract),
            expected: expected.into(),
            tol: 1e-9,
            mode: NumericMode::Absolute,
            group: None,
        }
    }
}

pub fn numeric_validator<S: 'static>(opts: NumericValidatorOptions<S>) -> Box<dyn Validator<S>> {
    let name = opts.name;
    let extract = opts.extract;
    let expected = opts.expected;
    let tol = opts.tol;
    let mode = opts.mode;
    let group = opts.group;
    let validator_name = name.clone();

    Box::new(ClosureValidator::new(validator_name, move |station: &S| {
        let observed = extract(station);
        let expected_value = expected.resolve(station);
        if !observed.is_finite() || !expected_value.is_finite() {
            let mut check = ValidationCheck::new(name.clone(), false)
                .with_observed(observed.to_string())
                .with_expected(expected_value.to_string())
                .with_details("non-finite value");
            if let Some(group) = group.as_ref() {
                check = check.with_group(group.clone());
            }
            return Ok(vec![check]);
        }

        let diff = (observed - expected_value).abs();
        let denom = if mode == NumericMode::Relative {
            expected_value.abs().max(1e-12)
        } else {
            1.0
        };
        let err = diff / denom;
        let passed = err <= tol;
        let mut check = ValidationCheck::new(name.clone(), passed)
            .with_observed(to_precision(observed, 8))
            .with_expected(to_precision(expected_value, 8));
        if let Some(group) = group.as_ref() {
            check = check.with_group(group.clone());
        }
        if !passed {
            check = check.with_details(format!("{}-err={:.2e} > tol={}", mode.label(), err, tol));
        }
        Ok(vec![check])
    }))
}

pub struct BoundValidatorOptions<S> {
    pub name: String,
    pub extract: NumericExtractFn<S>,
    pub low: Option<f64>,
    pub high: Option<f64>,
    pub inclusive: bool,
    pub group: Option<String>,
}

impl<S> BoundValidatorOptions<S> {
    pub fn new<F>(name: impl Into<String>, extract: F) -> Self
    where
        F: Fn(&S) -> f64 + 'static,
    {
        Self {
            name: name.into(),
            extract: Box::new(extract),
            low: None,
            high: None,
            inclusive: true,
            group: None,
        }
    }
}

pub fn bound_validator<S: 'static>(opts: BoundValidatorOptions<S>) -> Box<dyn Validator<S>> {
    let name = opts.name;
    let extract = opts.extract;
    let low = opts.low.unwrap_or(f64::NEG_INFINITY);
    let high = opts.high.unwrap_or(f64::INFINITY);
    let inclusive = opts.inclusive;
    let group = opts.group;
    let validator_name = name.clone();

    Box::new(ClosureValidator::new(validator_name, move |station: &S| {
        let value = extract(station);
        let in_low = if inclusive { value >= low } else { value > low };
        let in_high = if inclusive {
            value <= high
        } else {
            value < high
        };
        let passed = in_low && in_high;
        let mut check = ValidationCheck::new(name.clone(), passed)
            .with_observed(value.to_string())
            .with_expected(format!(
                "{}{}, {}{}",
                if inclusive { "[" } else { "(" },
                low,
                high,
                if inclusive { "]" } else { ")" }
            ));
        if let Some(group) = group.as_ref() {
            check = check.with_group(group.clone());
        }
        if !passed {
            check = check.with_details(format!(
                "value {value} outside {} interval [{low}, {high}]",
                if inclusive { "closed" } else { "open" }
            ));
        }
        Ok(vec![check])
    }))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MonotonicityDirection {
    NonIncreasing,
    NonDecreasing,
}

impl MonotonicityDirection {
    fn label(self) -> &'static str {
        match self {
            Self::NonIncreasing => "non-increasing",
            Self::NonDecreasing => "non-decreasing",
        }
    }
}

pub struct MonotonicityValidatorOptions<S> {
    pub name: String,
    pub extract: SeriesExtractFn<S>,
    pub direction: MonotonicityDirection,
    pub tol: f64,
    pub group: Option<String>,
}

impl<S> MonotonicityValidatorOptions<S> {
    pub fn new<F>(name: impl Into<String>, extract: F, direction: MonotonicityDirection) -> Self
    where
        F: Fn(&S) -> Vec<f64> + 'static,
    {
        Self {
            name: name.into(),
            extract: Box::new(extract),
            direction,
            tol: 1e-12,
            group: None,
        }
    }
}

pub fn monotonicity_validator<S: 'static>(
    opts: MonotonicityValidatorOptions<S>,
) -> Box<dyn Validator<S>> {
    let name = opts.name;
    let extract = opts.extract;
    let direction = opts.direction;
    let tol = opts.tol;
    let group = opts.group;
    let validator_name = name.clone();

    Box::new(ClosureValidator::new(validator_name, move |station: &S| {
        let xs = extract(station);
        let first_violation = xs.windows(2).position(|pair| {
            let diff = pair[1] - pair[0];
            match direction {
                MonotonicityDirection::NonIncreasing => diff > tol,
                MonotonicityDirection::NonDecreasing => diff < -tol,
            }
        });

        let passed = first_violation.is_none();
        let mut check = ValidationCheck::new(name.clone(), passed).with_expected(direction.label());
        if let Some(group) = group.as_ref() {
            check = check.with_group(group.clone());
        }

        if let Some(window_index) = first_violation {
            let index = window_index + 1;
            check = check
                .with_observed(format!("breaks at i={index}"))
                .with_details(format!(
                    "xs[{}]={}  xs[{index}]={}",
                    index - 1,
                    xs[index - 1],
                    xs[index]
                ));
        } else {
            check = check.with_observed(format!("{} (n={})", direction.label(), xs.len()));
        }
        Ok(vec![check])
    }))
}

pub enum GroundTruthExpected<S, T> {
    Value(T),
    Extractor(Box<dyn Fn(&S) -> T>),
}

impl<S, T> GroundTruthExpected<S, T>
where
    T: Clone,
{
    fn resolve(&self, station: &S) -> T {
        match self {
            Self::Value(value) => value.clone(),
            Self::Extractor(extract) => extract(station),
        }
    }
}

impl<S, T> From<T> for GroundTruthExpected<S, T> {
    fn from(value: T) -> Self {
        Self::Value(value)
    }
}

pub struct GroundTruthValidatorOptions<S, T> {
    pub name: String,
    pub extract: GroundTruthExtractFn<S, T>,
    pub expected: GroundTruthExpected<S, T>,
    pub compare: GroundTruthCompareFn<T>,
    pub format: GroundTruthFormatFn<T>,
    pub group: Option<String>,
}

impl<S, T> GroundTruthValidatorOptions<S, T>
where
    T: Clone + Debug + 'static,
{
    pub fn new<F, E, C>(name: impl Into<String>, extract: F, expected: E, compare: C) -> Self
    where
        F: Fn(&S) -> T + 'static,
        E: Into<GroundTruthExpected<S, T>>,
        C: Fn(&T, &T) -> Option<String> + 'static,
    {
        Self {
            name: name.into(),
            extract: Box::new(extract),
            expected: expected.into(),
            compare: Box::new(compare),
            format: Box::new(|value| format!("{value:?}")),
            group: None,
        }
    }
}

pub fn ground_truth_validator<S: 'static, T>(
    opts: GroundTruthValidatorOptions<S, T>,
) -> Box<dyn Validator<S>>
where
    T: Clone + Debug + 'static,
{
    let name = opts.name;
    let extract = opts.extract;
    let expected = opts.expected;
    let compare = opts.compare;
    let format_value = opts.format;
    let group = opts.group;
    let validator_name = name.clone();

    Box::new(ClosureValidator::new(validator_name, move |station: &S| {
        let observed = extract(station);
        let expected_value = expected.resolve(station);
        let failure = compare(&observed, &expected_value);
        let passed = failure.is_none();
        let mut check = ValidationCheck::new(name.clone(), passed)
            .with_observed(format_value(&observed))
            .with_expected(format_value(&expected_value));
        if let Some(group) = group.as_ref() {
            check = check.with_group(group.clone());
        }
        if let Some(failure) = failure {
            check = check.with_details(failure);
        }
        Ok(vec![check])
    }))
}

pub struct ExternalReferenceValidatorOptions<S> {
    pub name: String,
    pub reference_path: String,
    pub compare: ExternalReferenceCompareFn<S>,
    pub silent_if_missing: bool,
    pub group: Option<String>,
}

impl<S> ExternalReferenceValidatorOptions<S> {
    pub fn new<F>(name: impl Into<String>, reference_path: impl Into<String>, compare: F) -> Self
    where
        F: Fn(&S, &JsonValue) -> Vec<ValidationCheck> + 'static,
    {
        Self {
            name: name.into(),
            reference_path: reference_path.into(),
            compare: Box::new(compare),
            silent_if_missing: false,
            group: None,
        }
    }
}

pub fn external_reference_validator<S: 'static>(
    opts: ExternalReferenceValidatorOptions<S>,
) -> Box<dyn Validator<S>> {
    let name = opts.name;
    let reference_path = opts.reference_path;
    let compare = opts.compare;
    let silent_if_missing = opts.silent_if_missing;
    let group = opts.group;
    let validator_name = name.clone();

    Box::new(ClosureValidator::new(validator_name, move |station: &S| {
        let source = match fs::read_to_string(&reference_path) {
            Ok(source) => source,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
                if silent_if_missing {
                    return Ok(Vec::new());
                }
                eprintln!(
                    "[validation] external reference file not found for \"{}\": {} - comparison check will fail. Generate the reference or set silent_if_missing.",
                    name,
                    reference_path
                );
                let mut check = ValidationCheck::new(format!("{name}/reference-missing"), false)
                    .with_observed("absent")
                    .with_expected("present")
                    .with_details(format!("reference file {reference_path} not found"));
                if let Some(group) = group.as_ref() {
                    check = check.with_group(group.clone());
                }
                return Ok(vec![check]);
            }
            Err(err) => return Err(err.to_string()),
        };

        let reference: JsonValue = serde_json::from_str(&source).map_err(|err| err.to_string())?;
        let mut checks = compare(station, &reference);
        if let Some(group) = group.as_ref() {
            for check in &mut checks {
                if check.group.is_none() {
                    check.group = Some(group.clone());
                }
            }
        }
        Ok(checks)
    }))
}

fn to_precision(value: f64, significant_digits: usize) -> String {
    if !value.is_finite() {
        return value.to_string();
    }
    if value == 0.0 {
        return format!("{:.*}", significant_digits.saturating_sub(1), 0.0);
    }

    let abs = value.abs();
    if !(1e-6..1e21).contains(&abs) {
        return format!("{:.*e}", significant_digits.saturating_sub(1), value);
    }

    let digits_before_decimal = abs.log10().floor().max(0.0) as usize + 1;
    let decimals = significant_digits.saturating_sub(digits_before_decimal);
    format!("{value:.decimals$}")
}
