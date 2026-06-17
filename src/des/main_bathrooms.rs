//! Entry point for the household bathroom occupancy study.
//!
//! Runs the [`crate::des::bathrooms`] Monte-Carlo discrete-event simulation
//! (8 people, 2 bathrooms, 3 visits/day of 20 minutes), prints the headline
//! occupancy probabilities next to the closed-form binomial reference, and
//! writes the self-contained animation to `out/bathrooms.html`.

use crate::des::bathrooms::{render_bathrooms_html, run_bathroom_montecarlo, BathroomConfig};
use std::path::Path;

/// Entry point.
pub fn run() {
    let cfg = BathroomConfig::default();
    let report = run_bathroom_montecarlo(&cfg);
    let occ = &report.occupancy_fraction;
    let a = &report.analytic;

    println!(
        "Bathroom occupancy — {} people, {} bathrooms, {}×{}min/day",
        cfg.people, cfg.bathrooms, cfg.visits_per_day, cfg.service_min as i64
    );
    println!(
        "  per-person occupancy p = {:.4}  ({} replications × {} days)",
        a.p_person, cfg.replications, cfg.days_per_rep
    );
    println!("  {:<22} {:>10} {:>10}", "state", "simulated", "binomial");
    println!(
        "  {:<22} {:>9.1}% {:>9.1}%",
        "no bathroom busy",
        100.0 * occ[0],
        100.0 * a.p0
    );
    println!(
        "  {:<22} {:>9.1}% {:>9.1}%",
        "one bathroom busy",
        100.0 * occ[1],
        100.0 * a.p1
    );
    println!(
        "  {:<22} {:>9.1}% {:>9.1}%",
        "both bathrooms busy",
        100.0 * occ.get(2).copied().unwrap_or(0.0),
        100.0 * a.p2_plus
    );
    println!(
        "  mean bathrooms busy: sim {:.3} vs theory {:.3}; requests rejected: {:.2}%",
        report.mean_occupied,
        a.expected_occupied,
        100.0 * report.blocked_fraction
    );

    let out_dir = Path::new("out");
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("[main_bathrooms] could not create out/: {e}");
        return;
    }
    let path = out_dir.join("bathrooms.html");
    match std::fs::write(&path, render_bathrooms_html(&report)) {
        Ok(()) => println!("  wrote {}", path.display()),
        Err(e) => eprintln!("[main_bathrooms] could not write {}: {e}", path.display()),
    }
}
