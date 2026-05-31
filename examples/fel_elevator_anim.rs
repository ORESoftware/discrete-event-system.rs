//! **FEL elevator** simulation + **MDP/POMDP** elevator-dispatch models.
//!
//! ```sh
//! cargo run --example fel_elevator_anim
//! # writes:
//! #   out/elevator-fel/animation.html    (the next-event elevator sim, animated)
//! #   out/elevator-fel/mdp-player.html    (elevator-dispatch MDP, value-iterated)
//! #   out/elevator-fel/pomdp-player.html  (elevator-dispatch POMDP, belief-tracked)
//! #   out/elevator-fel/mdp.json, pomdp.json  (the canonical specs)
//! ```
//!
//! This is purely additive — the existing time-stepped elevator
//! (`des::main_elevator` / `main_elevator_highrise`) is untouched. The new
//! next-event elevator lives in [`des_engine::des::fel::elevator`], and the two
//! decision models run on the existing first-class model citizens
//! ([`des_engine::des::model`]).

use std::fs;
use std::path::Path;

use des_engine::des::fel::elevator::{
    elevator_mdp_spec, elevator_pomdp_spec, render_elevator_html, run_fel_elevator, ElevatorConfig,
};
use des_engine::des::model::with_builtins;

fn main() {
    let dir = Path::new("out/elevator-fel");
    fs::create_dir_all(dir).expect("create out/elevator-fel");

    // 1) FEL next-event elevator simulation -> animated HTML.
    let cfg = ElevatorConfig::default();
    let data = run_fel_elevator(&cfg);
    let meta = &data["meta"];
    let html = render_elevator_html(&data);
    let anim_path = dir.join("animation.html");
    fs::write(&anim_path, html).expect("write animation.html");

    println!("FEL elevator simulation");
    println!(
        "  floors={}  capacity={}  horizon={}s  λ={}/s",
        meta["floors"], meta["capacity"], meta["horizon"], meta["arrivalRate"]
    );
    println!(
        "  events={}  arrivals={}  served={}  mean wait={:.1}s",
        meta["events"],
        meta["arrivals"],
        meta["served"],
        meta["meanWait"].as_f64().unwrap_or(0.0)
    );
    println!("  frames={}", data["frames"].as_array().map(|a| a.len()).unwrap_or(0));
    println!("  -> {}", anim_path.display());

    // 2) MDP / POMDP elevator-dispatch models on the first-class citizens.
    let reg = with_builtins();

    let mdp_spec = elevator_mdp_spec();
    fs::write(dir.join("mdp.json"), serde_json::to_string_pretty(&mdp_spec).unwrap())
        .expect("write mdp.json");
    let mdp = reg.run("mdp", &mdp_spec).expect("elevator MDP solves");
    fs::write(dir.join("mdp-player.html"), mdp.to_player_html()).expect("write mdp player");
    println!("\nElevator-dispatch MDP  ({} states)", mdp_spec["numStates"]);
    println!("  {}", mdp.summary);
    println!("  -> {}", dir.join("mdp-player.html").display());

    let pomdp_spec = elevator_pomdp_spec();
    fs::write(dir.join("pomdp.json"), serde_json::to_string_pretty(&pomdp_spec).unwrap())
        .expect("write pomdp.json");
    let pomdp = reg.run("pomdp", &pomdp_spec).expect("elevator POMDP solves");
    fs::write(dir.join("pomdp-player.html"), pomdp.to_player_html()).expect("write pomdp player");
    println!("\nElevator-dispatch POMDP  (noisy call button)");
    println!("  {}", pomdp.summary);
    println!("  -> {}", dir.join("pomdp-player.html").display());
}
