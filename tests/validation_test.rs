//! TypeScript source: `src/des/test/validation-test.ts`
//! Rust target: `tests/validation_test.rs`

use discrete_event_system_rs::des::general::des_base::{
    fixed_point::{FixedPointHooks, FixedPointIterationStation, FixedPointOptions},
    runner::{run_iterative_des, IterativeRunOptions, IterativeRunReason},
    station::{DESRunLoopEntity, DESStation, HasRunTimeStep, StationCore},
    validation::{
        bound_validator, external_reference_validator, format_validation_report,
        ground_truth_validator, intrinsic_check, monotonicity_validator, numeric_validator,
        run_validators, BoundValidatorOptions, ClosureValidator, ExternalReferenceValidatorOptions,
        GroundTruthValidatorOptions, IntrinsicCheckOptions, MonotonicityDirection,
        MonotonicityValidatorOptions, NumericMode, NumericValidatorOptions, ValidationCheck,
        Validator,
    },
};
use serde_json::json;
use std::{fs, path::PathBuf};

#[derive(Debug)]
struct Stub {
    x: f64,
    history: Vec<f64>,
}

impl Stub {
    fn new(x: f64, history: Vec<f64>) -> Self {
        Self { x, history }
    }
}

#[test]
fn validator_factories_match_typescript_protocol() {
    let mut num_abs_opts = NumericValidatorOptions::<Stub>::new("t.numAbs", |s| s.x, 1.0);
    num_abs_opts.tol = 1e-9;
    let num_abs = numeric_validator(num_abs_opts);
    let result = num_abs.validate(&Stub::new(1.0, vec![])).unwrap();
    assert!(result[0].passed);
    assert_eq!(result[0].observed.as_deref(), Some("1.0000000"));

    let mut num_rel_opts = NumericValidatorOptions::<Stub>::new("t.numRel", |s| s.x, 100.0);
    num_rel_opts.tol = 1e-3;
    num_rel_opts.mode = NumericMode::Relative;
    let num_rel = numeric_validator(num_rel_opts);
    let result = num_rel.validate(&Stub::new(101.0, vec![])).unwrap();
    assert!(!result[0].passed);
    assert_eq!(
        result[0].details.as_deref(),
        Some("relative-err=1.00e-2 > tol=0.001")
    );

    let mut bound_opts = BoundValidatorOptions::<Stub>::new("t.bnd", |s| s.x);
    bound_opts.low = Some(0.0);
    bound_opts.high = Some(10.0);
    let bound = bound_validator(bound_opts);
    assert!(bound.validate(&Stub::new(5.0, vec![])).unwrap()[0].passed);
    assert!(!bound.validate(&Stub::new(11.0, vec![])).unwrap()[0].passed);

    let mono = monotonicity_validator(MonotonicityValidatorOptions::<Stub>::new(
        "t.mono",
        |s| s.history.clone(),
        MonotonicityDirection::NonIncreasing,
    ));
    assert!(
        mono.validate(&Stub::new(0.0, vec![5.0, 4.0, 3.0, 3.0, 1.0]))
            .unwrap()[0]
            .passed
    );
    let mono_fail = mono
        .validate(&Stub::new(0.0, vec![5.0, 4.0, 5.0, 3.0]))
        .unwrap();
    assert!(!mono_fail[0].passed);
    assert_eq!(mono_fail[0].details.as_deref(), Some("xs[1]=4  xs[2]=5"));

    let ground_truth = ground_truth_validator(GroundTruthValidatorOptions::<Stub, Vec<f64>>::new(
        "t.gt",
        |s| s.history.clone(),
        vec![1.0, 2.0, 3.0],
        |observed, expected| {
            if observed.len() != expected.len() {
                return Some(format!("len {} vs {}", observed.len(), expected.len()));
            }
            observed
                .iter()
                .zip(expected.iter())
                .position(|(a, b)| (a - b).abs() > 1e-9)
                .map(|index| format!("idx {index}"))
        },
    ));
    assert!(
        ground_truth
            .validate(&Stub::new(0.0, vec![1.0, 2.0, 3.0]))
            .unwrap()[0]
            .passed
    );
    assert!(
        !ground_truth
            .validate(&Stub::new(0.0, vec![1.0, 2.0, 4.0]))
            .unwrap()[0]
            .passed
    );

    let intrinsic = intrinsic_check(IntrinsicCheckOptions::<Stub>::new("t.ic", |s| s.x > 0.0));
    assert!(intrinsic.validate(&Stub::new(7.0, vec![])).unwrap()[0].passed);
    assert!(!intrinsic.validate(&Stub::new(-1.0, vec![])).unwrap()[0].passed);

    let broken: Box<dyn Validator<Stub>> =
        Box::new(ClosureValidator::new("t.broken", |_station: &Stub| {
            Err("boom".to_owned())
        }));
    let validators = vec![broken];
    let out = run_validators(&Stub::new(0.0, vec![]), &validators);
    assert_eq!(out.len(), 1);
    assert!(!out[0].passed);
    assert_eq!(out[0].name, "t.broken/threw");

    let report = format_validation_report(&[
        ValidationCheck::new("a", true),
        ValidationCheck::new("b", false)
            .with_observed("5")
            .with_expected("6")
            .with_details("oops"),
    ]);
    assert!(report.contains("1 passed"));
    assert!(report.contains("1 failed"));
}

struct CounterStation {
    core: StationCore<Self>,
    count: usize,
    cap: usize,
    attach_validator_on_finalize: bool,
    finalized: bool,
}

impl CounterStation {
    fn new(id: &str, cap: usize) -> Self {
        Self {
            core: StationCore::new(id),
            count: 0,
            cap,
            attach_validator_on_finalize: false,
            finalized: false,
        }
    }

    fn finalizing(id: &str) -> Self {
        let mut station = Self::new(id, 1);
        station.attach_validator_on_finalize = true;
        station
    }
}

impl HasRunTimeStep for CounterStation {
    fn run_time_step(&mut self) {
        self.count += 1;
    }
}

impl DESRunLoopEntity for CounterStation {
    fn id(&self) -> Option<&str> {
        Some(self.core.id())
    }

    fn has_work(&self) -> bool {
        self.count < self.cap
    }

    fn on_finalize(&mut self) {
        self.finalized = true;
        if self.attach_validator_on_finalize {
            self.add_validator(intrinsic_check(
                IntrinsicCheckOptions::<CounterStation>::new(
                    "fin.attached-on-finalize",
                    |station| station.finalized,
                ),
            ));
        }
    }

    fn num_validators(&self) -> usize {
        self.core.num_validators()
    }

    fn run_validation(&self) -> Vec<ValidationCheck> {
        self.core.run_validation(self)
    }
}

impl DESStation for CounterStation {
    fn core(&self) -> &StationCore<Self> {
        &self.core
    }

    fn core_mut(&mut self) -> &mut StationCore<Self> {
        &mut self.core
    }
}

#[test]
fn des_station_runner_aggregates_validation_like_typescript() {
    let mut station = CounterStation::new("counter", 10);
    let mut final_count = NumericValidatorOptions::<CounterStation>::new(
        "counter.final",
        |station| station.count as f64,
        10.0,
    );
    final_count.tol = 0.0;
    station.add_validator(numeric_validator(final_count));

    let mut in_range =
        BoundValidatorOptions::<CounterStation>::new("counter.in-range", |station| {
            station.count as f64
        });
    in_range.low = Some(0.0);
    in_range.high = Some(100.0);
    station.add_validator(bound_validator(in_range));
    assert_eq!(station.num_validators(), 2);

    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 1] = [&mut station];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                shuffle: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(summary.reason, IterativeRunReason::Done);
    assert_eq!(summary.ticks, 10);
    assert_eq!(summary.validation_ok, Some(true));
    assert_eq!(summary.validation.as_ref().unwrap().len(), 2);

    let mut wrong = CounterStation::new("counter2", 10);
    let mut wrong_count = NumericValidatorOptions::<CounterStation>::new(
        "counter2.WRONG",
        |station| station.count as f64,
        999.0,
    );
    wrong_count.tol = 0.0;
    wrong.add_validator(numeric_validator(wrong_count));
    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 1] = [&mut wrong];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                shuffle: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(summary.validation_ok, Some(false));
    let failed = &summary.validation.as_ref().unwrap()[0];
    assert_eq!(failed.observed.as_deref(), Some("10.000000"));
    assert_eq!(failed.expected.as_deref(), Some("999.00000"));

    let mut suppressed = CounterStation::new("counter3", 5);
    suppressed.add_validator(numeric_validator(
        NumericValidatorOptions::<CounterStation>::new("x", |station| station.count as f64, 5.0),
    ));
    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 1] = [&mut suppressed];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                shuffle: false,
                run_validators: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert!(summary.validation.is_none());

    let mut no_validator = CounterStation::new("nv", 3);
    let mut with_validator = CounterStation::new("wv", 3);
    with_validator.add_validator(numeric_validator(
        NumericValidatorOptions::<CounterStation>::new(
            "wv.eq",
            |station| station.count as f64,
            3.0,
        ),
    ));
    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 2] =
            [&mut no_validator, &mut with_validator];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                shuffle: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let validation = summary.validation.unwrap();
    assert_eq!(validation.len(), 1);
    assert_eq!(validation[0].name, "wv.eq");

    let mut finalizing = CounterStation::finalizing("f");
    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 1] = [&mut finalizing];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                shuffle: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(summary.validation_ok, Some(true));
    assert!(summary
        .validation
        .unwrap()
        .iter()
        .any(|check| check.name == "fin.attached-on-finalize"));
}

struct HalvingHooks;

impl FixedPointHooks<f64> for HalvingHooks {
    fn initial_state(&mut self) -> f64 {
        1.0
    }

    fn apply_operator(&mut self, prev: &f64) -> f64 {
        prev * 0.5
    }

    fn delta(&self, prev: &f64, next: &f64) -> f64 {
        (next - prev).abs()
    }
}

type MiniFixedPoint = FixedPointIterationStation<f64, HalvingHooks>;

fn validation_tmp_path(file_name: &str) -> PathBuf {
    std::env::temp_dir().join(format!(
        "discrete-event-system-rs-validation-{}-{file_name}",
        std::process::id()
    ))
}

#[test]
fn external_reference_validator_matches_fixed_point_behavior() {
    let missing_path = validation_tmp_path("missing.json");
    let mut loud = MiniFixedPoint::new(
        "mini-loud",
        HalvingHooks,
        FixedPointOptions {
            tol: 1e-6,
            max_iter: 30,
            max_history_len: None,
        },
    );
    loud.add_validator(external_reference_validator(
        ExternalReferenceValidatorOptions::<MiniFixedPoint>::new(
            "5a.loud",
            missing_path.to_string_lossy().to_string(),
            |_station, _reference| Vec::new(),
        ),
    ));
    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 1] = [&mut loud];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                shuffle: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    let missing = &summary.validation.unwrap()[0];
    assert_eq!(missing.name, "5a.loud/reference-missing");
    assert!(!missing.passed);

    let mut quiet_opts = ExternalReferenceValidatorOptions::<MiniFixedPoint>::new(
        "5b.quiet",
        missing_path.to_string_lossy().to_string(),
        |_station, _reference| Vec::new(),
    );
    quiet_opts.silent_if_missing = true;
    let mut quiet = MiniFixedPoint::new(
        "mini-quiet",
        HalvingHooks,
        FixedPointOptions {
            tol: 1e-6,
            max_iter: 30,
            max_history_len: None,
        },
    );
    quiet.add_validator(external_reference_validator(quiet_opts));
    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 1] = [&mut quiet];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                shuffle: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert!(summary.validation.is_none());

    let reference_path = validation_tmp_path("present.json");
    fs::write(&reference_path, json!({"fixedPoint": 0.0}).to_string()).unwrap();
    let mut present = MiniFixedPoint::new(
        "mini-pres",
        HalvingHooks,
        FixedPointOptions {
            tol: 1e-6,
            max_iter: 60,
            max_history_len: None,
        },
    );
    present.add_validator(external_reference_validator(
        ExternalReferenceValidatorOptions::<MiniFixedPoint>::new(
            "5c.pres",
            reference_path.to_string_lossy().to_string(),
            |station, reference| {
                let observed = *station.current();
                let expected = reference["fixedPoint"].as_f64().unwrap();
                let passed = (observed - expected).abs() < 1e-5;
                let mut check = ValidationCheck::new("5c.pres", passed)
                    .with_observed(observed.to_string())
                    .with_expected(expected.to_string());
                if !passed {
                    check = check.with_details("mismatch");
                }
                vec![check]
            },
        ),
    ));
    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 1] = [&mut present];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                shuffle: false,
                ..Default::default()
            },
        )
        .unwrap()
    };
    assert_eq!(summary.validation_ok, Some(true));
    assert!(present.current().abs() < 1e-5);
}
