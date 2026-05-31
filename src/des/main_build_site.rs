//! Port of `src/des/main-build-site.ts`.
//!
//! Build tool: regenerates every simulation HTML page into `out/` and writes the
//! curated landing index (`out/index.html`).
//!
//! Delegates page rendering to `crate::des::animation::run_report`
//! (`SimulationIndexPage`, `IndexEntry`, `CatalogEntry`). `process.env.*` →
//! `std::env::var`; `fs` → `std::fs`.
//!
//! PORT NOTE: the TS regenerates artifacts with
//! `execFileSync(ts-node, [siblingScript], …)`. Rust-native generators are
//! called directly from this module as they become available; scripts still
//! represented only by TS-style placeholders are logged by
//! [`SimulationSiteBuilder::run_script`].

#![allow(dead_code)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::des::animation::run_report::{
    CatalogEntry, CatalogSection, IndexEntry, IndexGroup, SimulationIndexPage,
};

struct SimulationSiteBuilder;

impl SimulationSiteBuilder {
    /// `run(script, env)` — see PORT NOTE: logs the intended invocation only.
    fn run_script(&self, script: &str, env: &[(&str, &str)]) {
        let env_str = env
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(" ");
        eprintln!("  • {script} {env_str}");
        // PORT NOTE: would `execFileSync(ts-node, [script], { stdio: 'inherit', … })`.
    }

    fn build(&self) {
        if std::env::var("INDEX_ONLY").as_deref() == Ok("1") {
            eprintln!("INDEX_ONLY=1: rebuilding out/index.html only...");
            self.write_index();
            return;
        }
        eprintln!("Regenerating animations...");
        self.run_script("src/des/main-wind-mppt-anim.ts", &[]);
        self.run_script("src/des/main-wind-mppt-anim.ts", &[("CONTROLLER", "pi")]);
        self.generate_dc_motor_pages();
        crate::des::main_observability_controllability_anim::run();

        eprintln!("Generating run reports...");
        self.run_script("src/des/main-empirical-control-report.ts", &[]);
        self.run_script("src/des/main-stochastic-sde-report.ts", &[]);

        eprintln!("Generating traffic simulations...");
        match crate::des::main_traffic::write_traffic_html_pages() {
            Ok((traffic, smart)) => {
                eprintln!("  • {traffic}");
                eprintln!("  • {smart}");
            }
            Err(e) => eprintln!("  ! traffic HTML generation failed: {e}"),
        }

        self.write_index();
    }

    fn generate_dc_motor_pages(&self) {
        let original_mode = std::env::var("MODE").ok();
        std::env::remove_var("MODE");
        crate::des::main_dc_motor_anim::run();
        std::env::set_var("MODE", "open");
        crate::des::main_dc_motor_anim::run();
        match original_mode {
            Some(value) => std::env::set_var("MODE", value),
            None => std::env::remove_var("MODE"),
        }
    }

    fn link_if_exists(&self, entry: IndexEntry) -> Option<IndexEntry> {
        if Path::new("out").join(&entry.href).exists() {
            Some(entry)
        } else {
            None
        }
    }

    /// Recursively collect every `*.html` under `dir` as forward-slash paths
    /// relative to `base`.
    fn scan_html(&self, dir: &Path, base: &Path, acc: &mut Vec<String>) {
        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => return,
        };
        for entry in entries.flatten() {
            let full = entry.path();
            if full.is_dir() {
                self.scan_html(&full, base, acc);
            } else if full.extension().and_then(|e| e.to_str()) == Some("html") {
                if let Ok(rel) = full.strip_prefix(base) {
                    let parts: Vec<String> = rel
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .collect();
                    acc.push(parts.join("/"));
                }
            }
        }
    }

    fn human_size(&self, bytes: u64) -> String {
        if bytes < 1024 {
            format!("{bytes} B")
        } else if bytes < 1024 * 1024 {
            format!("{:.0} KB", bytes as f64 / 1024.0)
        } else {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        }
    }

    fn catalog_entries(&self, featured: &HashSet<String>) -> Vec<CatalogEntry> {
        let out_dir = std::fs::canonicalize("out").unwrap_or_else(|_| PathBuf::from("out"));
        let mut found: Vec<String> = Vec::new();
        self.scan_html(&out_dir, &out_dir, &mut found);
        let mut rels: Vec<String> = found
            .into_iter()
            .filter(|rel| rel != "index.html" && !featured.contains(rel))
            .collect();
        rels.sort();
        rels.into_iter()
            .map(|rel| {
                let size = std::fs::metadata(out_dir.join(&rel))
                    .ok()
                    .map(|m| self.human_size(m.len()));
                CatalogEntry {
                    href: rel.clone(),
                    label: rel.strip_suffix(".html").unwrap_or(&rel).to_string(),
                    size,
                }
            })
            .collect()
    }

    fn write_index(&self) {
        let animations: Vec<IndexEntry> = vec![
            IndexEntry {
                kind: "animation".into(),
                title: "Wind MPPT — optimal torque".into(),
                href: "wind-mppt/animation-optimal-torque.html".into(),
                description:
                    "Variable-speed PMSG turbine tracking optimal tip-speed ratio via T = K_opt·ω²."
                        .into(),
            },
            IndexEntry {
                kind: "animation".into(),
                title: "Wind MPPT — PI speed loop".into(),
                href: "wind-mppt/animation-pi.html".into(),
                description: "Same turbine driven by a PI controller tracking ω* = λ*·V/R.".into(),
            },
            IndexEntry {
                kind: "animation".into(),
                title: "DC motor — closed-loop PI".into(),
                href: "dc-motor/animation-closed.html".into(),
                description:
                    "Back-EMF ODE motor; PI speed control tracking 60→100 rad/s with a load step."
                        .into(),
            },
            IndexEntry {
                kind: "animation".into(),
                title: "DC motor — open loop".into(),
                href: "dc-motor/animation-open.html".into(),
                description:
                    "Step-voltage response showing back-EMF rise throttling armature current."
                        .into(),
            },
            IndexEntry {
                kind: "animation".into(),
                title: "Controllability & Observability".into(),
                href: "obs-ctrl/animation.html".into(),
                description:
                    "Kalman rank tests, MDP reachability, and POMDP distinguishability storyboard."
                        .into(),
            },
            IndexEntry {
                kind: "animation".into(),
                title: "Empirical control — structured run".into(),
                href: "empirical-control/player.html".into(),
                description:
                    "Playable frame stream for LTI Gramian, MDP reachability, and POMDP belief checks."
                        .into(),
            },
            IndexEntry {
                kind: "animation".into(),
                title: "Temperature control — winter heat".into(),
                href: "temp-control/animation.html".into(),
                description:
                    "Heating-only indoor temperature control over a cold 24-hour winter day."
                        .into(),
            },
            IndexEntry {
                kind: "animation".into(),
                title: "Temperature control — heat/cool".into(),
                href: "temp-control/animation-heat-cool.html".into(),
                description:
                    "Bidirectional heat-pump control with night heating and afternoon cooling."
                        .into(),
            },
        ];
        let runs: Vec<IndexEntry> = vec![
            IndexEntry { kind: "simulation".into(), title: "Traffic flow — five intersection".into(), href: "traffic-flow-five-intersection.html".into(), description: "Signalized five-intersection road network with moving car snapshots and lane-phase highlights.".into() },
            IndexEntry { kind: "simulation".into(), title: "Smart traffic flow".into(), href: "smart-traffic-flow.html".into(), description: "Smart movable cars with shuffled actor updates, accident instrumentation, and live traffic metrics.".into() },
            IndexEntry { kind: "run report".into(), title: "DC motor — shadow controllability & observability".into(), href: "dc-motor/shadow-observability-controllability.html".into(), description: "Dual evaluator for the back-EMF plant: Kalman rank tests plus Gramian degree metrics for weak/strong directions.".into() },
            IndexEntry { kind: "run report".into(), title: "Empirical controllability & observability".into(), href: "empirical-control/report.html".into(), description: "Gramian degree (min/max directions) and Monte-Carlo trial estimates vs analytic Kalman tests.".into() },
            IndexEntry { kind: "run report".into(), title: "Stochastic SDEs + 3 ML algorithms".into(), href: "stochastic-sde/report.html".into(), description: "Euler–Maruyama engine with MLE system-id, Ensemble Kalman filtering, and a diffusion model.".into() },
        ];

        let mut page = SimulationIndexPage::new(
            "Discrete-Event-System — Simulations & Runs",
            "Control-system animations and numerical / machine-learning runs, generated from the discrete-event-system submodule.",
        );
        let present = |es: Vec<IndexEntry>| -> Vec<IndexEntry> {
            es.into_iter()
                .filter_map(|e| self.link_if_exists(e))
                .collect()
        };
        let featured_anims = present(animations);
        let featured_runs = present(runs);
        page.add_group(IndexGroup {
            heading: "Control-system animations".into(),
            blurb: "Interactive HTML players (play / pause / scrub / speed) built on the DES animation engine.".into(),
            entries: featured_anims.clone(),
        });
        page.add_group(IndexGroup {
            heading: "Numerical & machine-learning runs".into(),
            blurb: "Reproducible run reports with the full console output of each simulation."
                .into(),
            entries: featured_runs.clone(),
        });

        let mut featured_hrefs: HashSet<String> = HashSet::new();
        for e in featured_anims.iter().chain(featured_runs.iter()) {
            featured_hrefs.insert(e.href.clone());
        }
        page.add_catalog(CatalogSection {
            heading: "All rendered runs".into(),
            blurb: "Every other HTML artifact in out/ — DES models, optimization solvers, signal transforms, \
epidemic/traffic/network simulations, and more. Click any to open its rendered page."
                .into(),
            entries: self.catalog_entries(&featured_hrefs),
        });

        let out = Path::new("out").join("index.html");
        if let Some(parent) = out.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        let _ = std::fs::write(&out, page.to_html(&iso_now()));
        let resolved = std::fs::canonicalize(&out).unwrap_or(out);
        eprintln!("\nLanding page: {}", resolved.display());
        eprintln!("Served on cluster at /des/out/ (and /des/out/index.html).");
    }
}

/// `new Date().toISOString()` — UTC timestamp `YYYY-MM-DDTHH:MM:SSZ`.
///
/// PORT NOTE: no `chrono` dependency assumed; computed from
/// `SystemTime::now()` via the civil-from-days algorithm (sub-second precision
/// dropped relative to JS's millisecond `toISOString`).
fn iso_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    format!("{year:04}-{m:02}-{d:02}T{hh:02}:{mm:02}:{ss:02}Z")
}

/// Entry point (TS top-level script).
pub fn run() {
    SimulationSiteBuilder.build();
}
