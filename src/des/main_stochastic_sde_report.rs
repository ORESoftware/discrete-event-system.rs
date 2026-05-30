//! Port of `src/des/main-stochastic-sde-report.ts`.
//!
//! Report tool: runs the stochastic-SDE + 3-ML-algorithm demo and writes a
//! styled HTML report into `out/stochastic-sde/report.html`.
//!
//! Conversion notes:
//!   - `class StochasticSdeReport` → struct + impl; `fs` write → `std::fs`.
//!   - `RunReportPage` → `crate::des::animation::run_report::RunReportPage`.
//!
//! PORT NOTE: the TS shells out (`execFileSync(ts-node,
//! ['src/des/main-stochastic-sde.ts'])`) to capture the sibling script's stdout
//! as the "Run output" log. This crate is a single library — there is no
//! separate `main-stochastic-sde` binary to exec, and capturing the current
//! process's own stdout needs an external crate. So we invoke the ported
//! [`crate::des::main_stochastic_sde::run`] directly (its output streams to the
//! console) and record a note in the log section. Wire real stdout capture once
//! a binary target or a capture helper exists.

use crate::des::animation::run_report::{ReportSection, RunReportPage};

struct StochasticSdeReport;

impl StochasticSdeReport {
    fn run(&self) {
        // Execute the demo (streams its textual output to stdout).
        crate::des::main_stochastic_sde::run();

        let log = "Run output streamed to stdout (see console). \
                   Subprocess/stdout capture is unavailable in the library crate — see PORT NOTE."
            .to_string();

        let mut page = RunReportPage::new(
            "Stochastic Differential Equations + 3 ML algorithms",
            "Euler–Maruyama SDE engine with system identification, ensemble filtering, and score-based diffusion.",
        );
        page.add_section(ReportSection {
            heading: "What this run covers".to_string(),
            description: Some(
                "Models dX = f(X,t)dt + g(X,t)dW where the solution is a random process. Three machine-learning paradigms run on it: \
                 (1) maximum-likelihood SDE parameter recovery (system identification); (2) an Ensemble Kalman Filter that estimates a hidden \
                 motor current from noisy speed-only measurements (filtering/inference); (3) a denoising-diffusion generative model that learns a \
                 bimodal target and samples it by integrating the reverse-time SDE."
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

        let out = std::path::Path::new("out").join("stochastic-sde").join("report.html");
        if let Some(dir) = out.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        let _ = std::fs::write(&out, page.to_html());
        let abs = std::fs::canonicalize(&out)
            .map(|p| p.to_string_lossy().into_owned())
            .unwrap_or_else(|_| out.to_string_lossy().into_owned());
        println!("Stochastic-SDE report: {}", abs);
    }
}

/// Entry point (TS top-level script).
pub fn run() {
    StochasticSdeReport.run();
}
