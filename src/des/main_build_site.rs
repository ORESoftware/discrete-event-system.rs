//! Port of `src/des/main-build-site.ts`.
//!
//! Build tool: regenerates every simulation HTML page into `out/` and writes the
//! curated landing index (`out/index.html`).
//!
//! Featured cards and HTML generators are defined in [`crate::des::html_index`];
//! this module filters entries by file existence and scans `out/` for any
//! additional HTML artifacts.

#![allow(dead_code)]

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::des::animation::run_report::{
    CatalogEntry, CatalogSection, IndexEntry, SimulationIndexPage,
};
use crate::des::html_index::{generate_html_artifacts, html_index_groups, to_index_entries};

struct SimulationSiteBuilder;

impl SimulationSiteBuilder {
    fn build(&self) {
        if std::env::var("INDEX_ONLY").as_deref() == Ok("1") {
            eprintln!("INDEX_ONLY=1: rebuilding out/index.html only...");
            self.write_index();
            return;
        }
        generate_html_artifacts();
        self.write_index();
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
        let mut page = SimulationIndexPage::new(
            "Discrete-Event-System — Simulations & Runs",
            "Control-system animations and numerical / machine-learning runs, generated from the discrete-event-system submodule.",
        );

        let mut featured_hrefs: HashSet<String> = HashSet::new();
        for group in html_index_groups() {
            let entries: Vec<IndexEntry> = to_index_entries(group.entries)
                .into_iter()
                .filter_map(|e| self.link_if_exists(e))
                .collect();
            for e in &entries {
                featured_hrefs.insert(e.href.clone());
            }
            page.add_group(crate::des::animation::run_report::IndexGroup {
                heading: group.heading.to_string(),
                blurb: group.blurb.to_string(),
                entries,
            });
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
fn iso_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0) as i64;
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (hh, mm, ss) = (rem / 3600, (rem % 3600) / 60, rem % 60);
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
