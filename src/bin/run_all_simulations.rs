//! Binary entry point that runs every simulation in series.
//!
//! `cargo run --bin run_all_simulations`

fn main() {
    let outcomes = des_engine::des::simulations::run_all_simulations();
    let failed = outcomes.iter().filter(|o| !o.ok).count();
    if failed > 0 {
        std::process::exit(1);
    }
}
