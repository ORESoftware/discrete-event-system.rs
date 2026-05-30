//! Rust port of `src/des/general/des-base/runner.ts`.

use super::{station::DESRunLoopEntity, validation::ValidationCheck};
use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/des-base/runner.ts",
    "src/des/general/des_base/runner.rs",
    &[
        "IterativeRunOptions and IterativeRunSummary are nominal structs.",
        "run_iterative_des operates over DESRunLoopEntity trait objects.",
        "Math.random is an injected FnMut returning f64.",
        "Validation aggregation delegates to each participant after finalize.",
    ],
    &[
        "DESResultStation",
        "IterativeDESParticipant",
        "IterativeRunOptions",
        "IterativeRunSummary",
        "assertNoValidationFailures",
        "failedValidationChecks",
        "runIterativeDES",
        "runResultStation",
        "validationFailureNames",
    ],
);

pub type IterativeDESParticipant = dyn DESRunLoopEntity;

pub struct IterativeRunOptions<'a> {
    pub max_ticks: Option<usize>,
    pub stop_when: Option<Box<dyn FnMut(usize) -> bool + 'a>>,
    pub rng: Option<Box<dyn FnMut() -> f64 + 'a>>,
    pub shuffle: bool,
    pub on_tick: Option<Box<dyn FnMut(usize) + 'a>>,
    pub run_validators: bool,
}

impl Default for IterativeRunOptions<'_> {
    fn default() -> Self {
        Self {
            max_ticks: None,
            stop_when: None,
            rng: None,
            shuffle: true,
            on_tick: None,
            run_validators: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IterativeRunReason {
    Done,
    MaxTicks,
    StopWhen,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IterativeRunSummary {
    pub ticks: usize,
    pub reason: IterativeRunReason,
    pub validation: Option<Vec<ValidationCheck>>,
    pub validation_ok: Option<bool>,
}

pub trait DESResultStation<R>: DESRunLoopEntity {
    fn result(&self, validation: &[ValidationCheck]) -> R;
}

fn shuffle_in_place<T>(arr: &mut [T], rng: &mut dyn FnMut() -> f64) {
    for i in (1..arr.len()).rev() {
        let j = ((rng() * ((i + 1) as f64)).floor() as usize).min(i);
        arr.swap(i, j);
    }
}

pub fn run_iterative_des(
    stations: &mut [&mut dyn DESRunLoopEntity],
    mut opts: IterativeRunOptions<'_>,
) -> Result<IterativeRunSummary, String> {
    for station in stations.iter() {
        station.assert_preconditions()?;
    }

    let mut default_rng = || rand::random::<f64>();
    let mut tick = 0usize;
    let reason;

    loop {
        if let Some(stop_when) = opts.stop_when.as_mut() {
            if stop_when(tick) {
                reason = IterativeRunReason::StopWhen;
                break;
            }
        }

        if let Some(max_ticks) = opts.max_ticks {
            if tick >= max_ticks {
                eprintln!(
                    "[run_iterative_des] hit max_ticks={max_ticks} before the system went quiescent ({} participants still active) - increase max_ticks or check for a non-terminating model.",
                    stations.len()
                );
                reason = IterativeRunReason::MaxTicks;
                break;
            }
        }

        let any_work = stations.iter().any(|station| station.has_work());
        if !any_work {
            reason = IterativeRunReason::Done;
            break;
        }

        let mut order: Vec<usize> = (0..stations.len()).collect();
        if opts.shuffle {
            if let Some(rng) = opts.rng.as_mut() {
                shuffle_in_place(&mut order, rng.as_mut());
            } else {
                shuffle_in_place(&mut order, &mut default_rng);
            }
        }

        for index in order {
            stations[index].run_time_step();
        }

        if let Some(on_tick) = opts.on_tick.as_mut() {
            on_tick(tick);
        }
        tick += 1;
    }

    for station in stations.iter_mut() {
        station.on_finalize();
    }

    let mut summary = IterativeRunSummary {
        ticks: tick,
        reason,
        validation: None,
        validation_ok: None,
    };

    if opts.run_validators {
        let mut all_checks = Vec::new();
        for station in stations.iter() {
            if station.num_validators() == 0 {
                continue;
            }
            all_checks.extend(station.run_validation());
        }

        if !all_checks.is_empty() {
            let validation_ok = all_checks.iter().all(|check| check.passed);
            if !validation_ok {
                let failed: Vec<&str> = all_checks
                    .iter()
                    .filter(|check| !check.passed)
                    .map(|check| check.name.as_str())
                    .collect();
                eprintln!(
                    "[run_iterative_des] {}/{} validators FAILED after {} ticks: {}",
                    failed.len(),
                    all_checks.len(),
                    tick,
                    failed.join(", ")
                );
            }
            summary.validation = Some(all_checks);
            summary.validation_ok = Some(validation_ok);
        }
    }

    Ok(summary)
}

pub fn run_result_station<R, S>(station: &mut S, opts: IterativeRunOptions<'_>) -> Result<R, String>
where
    S: DESResultStation<R>,
{
    let summary = {
        let mut stations: [&mut dyn DESRunLoopEntity; 1] = [station];
        run_iterative_des(&mut stations, opts)?
    };
    Ok(station.result(summary.validation.as_deref().unwrap_or(&[])))
}

pub fn failed_validation_checks(summary: &IterativeRunSummary) -> Vec<ValidationCheck> {
    summary
        .validation
        .as_ref()
        .map(|checks| {
            checks
                .iter()
                .filter(|check| !check.passed)
                .cloned()
                .collect()
        })
        .unwrap_or_default()
}

pub fn validation_failure_names(summary: &IterativeRunSummary) -> String {
    failed_validation_checks(summary)
        .into_iter()
        .map(|check| check.name)
        .collect::<Vec<_>>()
        .join(", ")
}

pub fn assert_no_validation_failures(
    summary: &IterativeRunSummary,
    model_name: &str,
) -> Result<(), String> {
    let names = validation_failure_names(summary);
    if names.is_empty() {
        return Ok(());
    }

    eprintln!(
        "[{model_name}] post-run validation failed ({} checks): {names}",
        failed_validation_checks(summary).len()
    );
    Err(format!("{model_name} validation failed: {names}"))
}
