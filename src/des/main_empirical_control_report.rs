//! Port of `src/des/main-empirical-control-report.ts`.
//!
//! Report tool: runs the empirical controllability/observability demo and
//! writes a styled HTML report into `out/`.
//!
//! Delegates page rendering to `crate::des::animation::run_report::RunReportPage`.
//!
//! PORT NOTE: the TS spawns the sibling script with
//! `execFileSync(ts-node, ['main-empirical-control.ts'])` and captures its
//! stdout. This is a library crate (no sibling binary), so we invoke the demo
//! in-process via [`crate::des::main_empirical_control::run`] (whose output goes
//! to this process's stdout) and embed an explanatory note where the captured
//! log would appear. Wire real stdout capture if a binary target is added.

#![allow(dead_code)]

use crate::des::animation::run_report::{ReportSection, RunReportPage};

struct EmpiricalControlReport;

impl EmpiricalControlReport {
    fn run(&self) {
        // In-process stand-in for `execFileSync` (see PORT NOTE): run the demo,
        // emitting its report to stdout.
        crate::des::main_empirical_control::run();
        let log = "(run executed in-process; output emitted to stdout — see PORT NOTE)".to_string();

        let mut page = RunReportPage::new(
            "Empirical Controllability & Observability",
            "Quantitative degree (Gramian eigenvalues) and trial-based estimates vs the analytic Kalman tests.",
        );
        page.add_section(ReportSection {
            heading: "What this run measures".to_string(),
            description: Some(
                "Instead of the binary Kalman rank verdict, this computes how controllable/observable each direction is: \
controllability/observability Gramian eigenvalues (min = weakest direction, max = strongest), the empirical reached-state \
covariance from thousands of random control rollouts (∝ W_c), least-squares target hit rate, noisy state-reconstruction error, \
MDP random-policy reach degree, and POMDP belief-tracking hit-probability / residual entropy."
                    .to_string(),
            ),
            metrics: None,
            log: None,
        });
        page.add_section(ReportSection {
            heading: "Run output".to_string(),
            description: None,
            metrics: None,
            log: Some(log),
        });

        let out = std::path::Path::new("out")
            .join("empirical-control")
            .join("report.html");
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&out, page.to_html());
        let resolved = std::fs::canonicalize(&out).unwrap_or(out);
        println!("Empirical-control report: {}", resolved.display());
    }
}

/// Entry point (TS top-level script).
pub fn run() {
    EmpiricalControlReport.run();
}
