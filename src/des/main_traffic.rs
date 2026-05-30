//! Port of `src/des/main-traffic.ts`.
//!
//! Thin runner: small traffic-flow simulation over DES cell stations (cars
//! carry position / velocity / acceleration / jerk).
//!
//! Conversion notes:
//!   - top-level `main()` → [`run`]; the fixed seed is passed through to the
//!     seeded sim inside `general::network_flow`.
//!   - delegates to `general::network_flow`.

use crate::des::general::network_flow::{
    build_five_intersection_traffic_network, run_traffic_flow, TrafficParams, TrafficResult,
};

/// `fmt(x, digits=2)` — finite numbers to `digits` decimals, else `"n/a"`.
fn fmt(x: f64, digits: usize) -> String {
    if x.is_finite() {
        format!("{:.*}", digits, x)
    } else {
        "n/a".to_string()
    }
}

fn mean_abs_jerk(result: &TrafficResult) -> f64 {
    let jerks: Vec<f64> = result
        .trace
        .iter()
        .flat_map(|row| row.cars.iter().map(|car| car.jerk_mps3.abs()))
        .collect();
    if jerks.is_empty() {
        0.0
    } else {
        jerks.iter().sum::<f64>() / jerks.len() as f64
    }
}

fn min_leader_gap(result: &TrafficResult) -> f64 {
    let gaps: Vec<f64> = result
        .trace
        .iter()
        .flat_map(|row| row.cars.iter().filter_map(|car| car.leader_gap_m))
        .collect();
    if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let network = build_five_intersection_traffic_network();
    let result = run_traffic_flow(
        TrafficParams {
            builtin: Some("five-intersection".to_string()),
            network: Some(network),
            duration_sec: 180.0,
            dt_sec: 0.25,
            seed: 19.0,
            max_cars: 250,
            car_length_m: None,
            car_width_m: None,
            lane_width_m: None,
            min_gap_m: None,
            max_accel_mps2: None,
            max_decel_mps2: None,
            max_jerk_mps3: Some(2.5),
            reaction_time_sec: Some(1.0),
            time_headway_sec: Some(1.2),
            grid_cell_size_m: Some(0.3048),
            grid_look_ahead_m: None,
            spawn_rate_multiplier: Some(1.0),
            scheduled_trips: None,
        },
        None,
    );
    let mean_abs_jerk_mps3 = mean_abs_jerk(&result);
    let min_leader_gap_m = min_leader_gap(&result);

    println!("# Traffic-flow DES");
    println!("# TrafficCellStation grid + moving car snapshots; kinematics stepped at dt");
    println!(
        "# nodes={}, lanes={}, sources={}, cells={}",
        result.network.nodes.len(),
        result.network.lanes.len(),
        result.network.sources.len(),
        result.cell_stats.created_cell_stations
    );
    println!(
        "# dt={}s, cell={}m, configured cap={}, max active={}",
        result.params.dt_sec,
        fmt(result.cell_stats.cell_size_m, 4),
        result.params.max_cars,
        result.max_active_cars
    );
    println!();

    println!("## Demand and throughput");
    println!("  entered cars:         {}", result.entered);
    println!("  exited cars:          {}", result.exited);
    println!("  active at stop:       {}", result.final_cars.len());
    println!("  dropped attempts:     {}", result.dropped);
    println!();

    println!("## Kinematics");
    println!(
        "  mean travel:       {} sec",
        fmt(result.mean_travel_time_sec, 1)
    );
    println!("  mean speed:        {} m/s", fmt(result.mean_speed_mps, 2));
    println!("  mean |jerk|:       {} m/s^3", fmt(mean_abs_jerk_mps3, 2));
    println!("  min leader gap:    {} m", fmt(min_leader_gap_m, 3));
    println!(
        "  max cell occup.:   {}",
        result.cell_stats.max_cell_occupancy
    );
    println!();

    println!("## Final sample");
    for car in result.final_cars.iter().take(10) {
        let cells: Vec<String> = car.grid_cell_ids.iter().take(3).cloned().collect();
        println!(
            "  car={:>3} lane={:<6} x={}m v={}m/s a={}m/s^2 cells={}",
            car.id,
            car.lane_id,
            fmt(car.position_m, 2),
            fmt(car.speed_mps, 2),
            fmt(car.acceleration_mps2, 2),
            cells.join("|")
        );
    }
}
