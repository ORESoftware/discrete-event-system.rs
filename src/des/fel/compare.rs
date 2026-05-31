//! Head-to-head: **FEL** (next-event) vs the existing **time-stepped** engine on
//! the same M/M/1 queue, both against the closed-form analytical truth.
//!
//! The two paradigms model the *same* system differently:
//!
//! | aspect            | FEL ([`super::mm1`])            | time-stepped ([`super::time_stepped_mm1`]) |
//! |-------------------|---------------------------------|--------------------------------------------|
//! | clock advance     | jumps to next event time        | fixed Δt seconds per tick                  |
//! | randomness        | exact Exp(λ)/Exp(μ) inter-times | Poisson(λΔt)/Poisson(μΔt) counts per tick  |
//! | accuracy          | exact (no time discretization)  | O(Δt) bias; → exact as Δt → 0              |
//! | work per unit time| ~ 2·λ events                    | (#stations)/Δt updates, regardless of load |
//!
//! This harness runs the FEL once and the time-stepped engine at several Δt, and
//! reports each estimate next to the analytical values and the work performed.

use serde::Serialize;

use super::mm1::{run_fel_mm1, FelMm1Result};
use super::time_stepped_mm1::{run_time_stepped_mm1, TimeSteppedMm1Result};

/// Closed-form M/M/1 steady-state metrics (the ground truth).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AnalyticalMm1 {
    pub lambda: f64,
    pub mu: f64,
    pub rho: f64,
    pub l: f64,
    pub lq: f64,
    pub w: f64,
    pub wq: f64,
}

/// Standard M/M/1 formulas: ρ=λ/μ, L=ρ/(1−ρ), Lq=ρ²/(1−ρ), W=1/(μ−λ),
/// Wq=ρ/(μ−λ).
pub fn analytical_mm1(lambda: f64, mu: f64) -> AnalyticalMm1 {
    assert!(lambda < mu, "unstable queue: need lambda < mu");
    let rho = lambda / mu;
    AnalyticalMm1 {
        lambda,
        mu,
        rho,
        l: rho / (1.0 - rho),
        lq: rho * rho / (1.0 - rho),
        w: 1.0 / (mu - lambda),
        wq: rho / (mu - lambda),
    }
}

/// Full comparison report: analytical truth, the FEL run, and time-stepped runs
/// at several Δt, plus a relative-work figure.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ComparisonReport {
    pub model: String,
    pub horizon: f64,
    pub analytical: AnalyticalMm1,
    pub fel: FelMm1Result,
    pub time_stepped: Vec<TimeSteppedMm1Result>,
    /// time-stepped station-updates ÷ FEL events, for the finest Δt run.
    pub work_ratio_finest_dt: f64,
}

/// Run the standard comparison: M/M/1 with the given rates, FEL to `horizon`,
/// and the time-stepped engine at each Δt in `dts` (each run covers the same
/// `horizon` of simulated time).
pub fn compare_mm1(lambda: f64, mu: f64, horizon: f64, dts: &[f64], seed: u32) -> ComparisonReport {
    let analytical = analytical_mm1(lambda, mu);
    let fel = run_fel_mm1(lambda, mu, horizon, seed);

    let mut time_stepped = Vec::new();
    for (i, &dt) in dts.iter().enumerate() {
        let num_ticks = (horizon / dt).ceil() as u64;
        time_stepped.push(run_time_stepped_mm1(
            lambda,
            mu,
            dt,
            num_ticks,
            seed.wrapping_add(1 + i as u32),
        ));
    }

    // Finest Δt = smallest dt = most ticks = most work.
    let finest = time_stepped
        .iter()
        .max_by(|a, b| a.station_updates.cmp(&b.station_updates));
    let work_ratio_finest_dt = match finest {
        Some(ts) if fel.events > 0 => ts.station_updates as f64 / fel.events as f64,
        _ => 0.0,
    };

    ComparisonReport {
        model: "M/M/1".to_string(),
        horizon,
        analytical,
        fel,
        time_stepped,
        work_ratio_finest_dt,
    }
}

fn fmt_row(label: &str, rho: f64, l: f64, lq: f64, w: f64, wq: f64, work: String) -> String {
    format!(
        "{label:<26} ρ={rho:>6.3}  L={l:>7.3}  Lq={lq:>7.3}  W={w:>7.3}  Wq={wq:>7.3}  {work}"
    )
}

/// Print a human-readable table plus the full JSON report. Suitable as a
/// `main_*`-style entry point.
pub fn run() {
    let lambda = 0.8;
    let mu = 1.0;
    let horizon = 50_000.0;
    let dts = [1.0, 0.25, 0.05];
    let report = compare_mm1(lambda, mu, horizon, &dts, 20_240_530);

    println!("=== FEL vs time-stepped DES — M/M/1 (λ={lambda}, μ={mu}, ρ={}) ===", lambda / mu);
    println!("horizon = {horizon} simulated time units\n");
    let a = &report.analytical;
    println!(
        "{}",
        fmt_row("analytical (exact)", a.rho, a.l, a.lq, a.w, a.wq, String::new())
    );
    let f = &report.fel;
    println!(
        "{}",
        fmt_row(
            "FEL (next-event)",
            f.rho,
            f.l,
            f.lq,
            f.w,
            f.wq,
            format!("events={}", f.events)
        )
    );
    for ts in &report.time_stepped {
        println!(
            "{}",
            fmt_row(
                &format!("time-step Δt={:<5}", ts.dt),
                ts.rho,
                ts.l,
                ts.lq,
                ts.w,
                ts.wq,
                format!("ticks={} updates={}", ts.ticks, ts.station_updates)
            )
        );
    }
    println!(
        "\nWork to cover the same horizon at the finest Δt: time-stepped did {:.1}× the \nFEL's event count (the FEL skips idle time; the time-step engine cannot).",
        report.work_ratio_finest_dt
    );
    println!("\n--- JSON ---");
    println!(
        "{}",
        serde_json::to_string_pretty(&report).unwrap_or_else(|_| "{}".to_string())
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn analytical_formulas_are_correct() {
        let a = analytical_mm1(0.8, 1.0);
        assert!((a.rho - 0.8).abs() < 1e-12);
        assert!((a.l - 4.0).abs() < 1e-9);
        assert!((a.lq - 3.2).abs() < 1e-9);
        assert!((a.w - 5.0).abs() < 1e-9);
        assert!((a.wq - 4.0).abs() < 1e-9);
    }

    #[test]
    fn fel_and_time_stepped_agree_with_analytical_and_each_other() {
        // ρ=0.5 mixes fast → tight estimates at a modest horizon.
        let report = compare_mm1(0.5, 1.0, 40_000.0, &[0.05], 4242);
        let a = &report.analytical;
        let fel = &report.fel;
        let ts = &report.time_stepped[0];

        // Each engine close to the analytical utilization & throughput.
        assert!((fel.rho - a.rho).abs() < 0.05, "FEL rho {}", fel.rho);
        assert!((ts.rho - a.rho).abs() < 0.06, "TS rho {}", ts.rho);
        assert!((fel.throughput - a.lambda).abs() < 0.05);
        assert!((ts.throughput - a.lambda).abs() < 0.06);

        // The two engines agree with each other on mean queue length.
        assert!((fel.lq - ts.lq).abs() < 0.25, "fel.lq={} ts.lq={}", fel.lq, ts.lq);
    }

    #[test]
    fn finer_time_steps_track_the_exact_fel_more_closely() {
        // The "compete" claim made concrete: the time-step bias is O(Δt), so a
        // finer Δt should land closer to the exact (FEL) utilization than a
        // coarse one. Same seed family, same horizon — only Δt changes.
        let report = compare_mm1(0.7, 1.0, 60_000.0, &[1.0, 0.1], 20_260_530);
        let fel_rho = report.fel.rho;
        let coarse = &report.time_stepped[0]; // Δt = 1.0
        let fine = &report.time_stepped[1]; // Δt = 0.1
        assert!(coarse.dt > fine.dt);
        let coarse_err = (coarse.rho - fel_rho).abs();
        let fine_err = (fine.rho - fel_rho).abs();
        assert!(
            fine_err <= coarse_err + 1e-6,
            "finer Δt should be at least as accurate: fine_err={fine_err} coarse_err={coarse_err}"
        );
        // And the fine step does strictly more work to get there.
        assert!(fine.station_updates > coarse.station_updates);
    }

    #[test]
    fn fel_does_far_less_work_than_time_stepping() {
        // The headline efficiency contrast: covering the same horizon, the
        // fixed-step engine performs many more operations than the FEL, which
        // only does work when an event actually occurs.
        let report = compare_mm1(0.5, 1.0, 20_000.0, &[0.05], 7);
        assert!(
            report.work_ratio_finest_dt > 10.0,
            "expected time-stepping to do >10x the work, got {}",
            report.work_ratio_finest_dt
        );
    }
}
