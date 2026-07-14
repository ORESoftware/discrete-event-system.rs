//! Entry point for the framework-based two-bathroom occupancy animation.
//!
//! Same household question and physics as [`crate::des::main_bathrooms`], but
//! built on the engine's reusable entity + animation frameworks (MovingEntity
//! people, StationaryEntity bathrooms, visual-block rendering through the shared
//! animation player). Writes `out/two-bathrooms.html`.

use crate::des::bathrooms::{run_bathroom_montecarlo, BathroomConfig};
use crate::des::two_bathrooms::render_two_bathrooms_html;
use std::path::Path;

/// Entry point.
pub fn run() {
    let cfg = BathroomConfig::default();
    // Print the headline so the catalogue run is informative.
    let report = run_bathroom_montecarlo(&cfg);
    let occ = &report.occupancy_fraction;
    println!(
        "Two bathrooms (framework) — none {:.1}% / one {:.1}% / both {:.1}%  (binomial {:.1}/{:.1}/{:.1})",
        100.0 * occ[0],
        100.0 * occ[1],
        100.0 * occ.get(2).copied().unwrap_or(0.0),
        100.0 * report.analytic.p0,
        100.0 * report.analytic.p1,
        100.0 * report.analytic.p2_plus,
    );

    let out_dir = Path::new("out");
    if let Err(e) = std::fs::create_dir_all(out_dir) {
        eprintln!("[main_two_bathrooms] could not create out/: {e}");
        return;
    }
    let path = out_dir.join("two-bathrooms.html");
    match std::fs::write(&path, render_two_bathrooms_html(&cfg)) {
        Ok(()) => println!("  wrote {}", path.display()),
        Err(e) => eprintln!(
            "[main_two_bathrooms] could not write {}: {e}",
            path.display()
        ),
    }
}
