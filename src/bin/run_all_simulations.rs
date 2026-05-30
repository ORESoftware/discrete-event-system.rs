//! Binary entry point that runs simulations in series.
//!
//! `cargo run --bin run_all_simulations`            — run the whole catalogue
//! `cargo run --bin run_all_simulations -- <name>`  — run only sims whose name
//!                                                    contains `<name>`

fn main() {
    let filter = std::env::args().nth(1);
    let outcomes = match filter {
        Some(name) => des_engine::des::simulations::run_simulations_matching(&name),
        None => des_engine::des::simulations::run_all_simulations(),
    };
    let failed = outcomes.iter().filter(|o| !o.ok).count();
    if failed > 0 {
        std::process::exit(1);
    }
}
