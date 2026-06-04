//! Port of `src/des/runners/compare-elevator-dispatch.ts`.
//!
//! Sweeps seeds and arrival rates to quantify coordinated vs uncoordinated
//! elevator dispatch, printing a table + per-rate summary and writing a JSON
//! dump. The TS top-level `main()` becomes [`run`].
//!
//! ## PORT NOTE
//!
//! This driver imports `ElevatorConfig`/`build_schedule`/`run_elevator` from the
//! real Rust `crate::des::main_elevator` port and keeps only the comparison
//! sweep/reporting logic here.
//!
//! Notes:
//!   * `process.env.{SEEDS,LAMBDAS,SIM_T}` → `std::env::var` + split/parse.
//!   * `fs`/`path` + `JSON.stringify(.., null, 2)` → `std::fs` + `JsonValue`.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use crate::des::main_elevator::{build_schedule, run_elevator, Aggregates, ElevatorConfig};
use crate::des::observability::logger::JsonValue;

struct TrialAggregate {
    seed: u32,
    lambda: f64,
    uncoord: Aggregates,
    coord: Aggregates,
}

fn parse_list(env: Option<String>, def: &[f64]) -> Vec<f64> {
    match env {
        None => def.to_vec(),
        Some(s) => s
            .split(',')
            .filter_map(|x| x.trim().parse::<f64>().ok())
            .filter(|x| x.is_finite())
            .collect(),
    }
}

fn pct(a: f64, b: f64) -> String {
    if b == 0.0 {
        "n/a".to_string()
    } else {
        format!("{:.1}%", (a / b - 1.0) * 100.0)
    }
}

fn pad_start(s: &str, width: usize) -> String {
    if s.chars().count() >= width {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(width - s.chars().count()))
    }
}

fn jn(v: f64) -> JsonValue {
    JsonValue::Number(v)
}

fn agg_json(a: &Aggregates) -> JsonValue {
    JsonValue::Object(vec![
        ("meanWait".to_string(), jn(a.mean_wait)),
        ("p95Wait".to_string(), jn(a.p95_wait)),
        ("meanTotal".to_string(), jn(a.mean_total)),
    ])
}

/// `main()` — run the dispatch sweep.
pub fn run() {
    let seeds = parse_list(std::env::var("SEEDS").ok(), &[1.0, 2.0, 3.0, 4.0, 5.0])
        .into_iter()
        .map(|x| x as u32)
        .collect::<Vec<_>>();
    let lambdas = parse_list(std::env::var("LAMBDAS").ok(), &[0.2, 0.4]);
    let sim_t: f64 = std::env::var("SIM_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1800.0);

    let mut trials: Vec<TrialAggregate> = Vec::new();
    println!("# elevator dispatch comparison");
    println!(
        "#   seeds = {}",
        seeds
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!(
        "#   λ     = {} arrivals/s",
        lambdas
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("#   simT  = {sim_t}s");
    println!();
    println!("  λ     seed   meanWait u→c     p95Wait u→c     meanTotal u→c    Δmean   Δp95");
    println!(
        "  ────  ────  ─────────────────  ─────────────────  ──────────────────  ──────  ──────"
    );

    for &lambda in &lambdas {
        for &seed in &seeds {
            let base = ElevatorConfig {
                n_floors: 4,
                n_elevators: 3,
                capacity: 8,
                floor_travel_time: 4.0,
                service_time: 3.0,
                arrival_rate: lambda,
                sim_t,
                step_size: 0.5,
                seed,
                dispatch_mode: String::new(),
            };
            let schedule = build_schedule(&base);
            let mut uncoord_cfg = base.clone();
            uncoord_cfg.dispatch_mode = "uncoordinated".to_string();
            let mut coord_cfg = base;
            coord_cfg.dispatch_mode = "coordinated".to_string();
            let u = run_elevator(uncoord_cfg, schedule.clone()).aggregates;
            let c = run_elevator(coord_cfg, schedule).aggregates;
            trials.push(TrialAggregate {
                seed,
                lambda,
                uncoord: u.clone(),
                coord: c.clone(),
            });

            let mw = format!(
                "{} → {}",
                pad_start(&format!("{:.2}", u.mean_wait), 5),
                pad_start(&format!("{:.2}", c.mean_wait), 5)
            );
            let pw = format!(
                "{} → {}",
                pad_start(&format!("{:.1}", u.p95_wait), 5),
                pad_start(&format!("{:.1}", c.p95_wait), 5)
            );
            let mt = format!(
                "{} → {}",
                pad_start(&format!("{:.2}", u.mean_total), 6),
                pad_start(&format!("{:.2}", c.mean_total), 6)
            );
            let dm = pad_start(&pct(c.mean_wait, u.mean_wait), 7);
            let dp = pad_start(&pct(c.p95_wait, u.p95_wait), 7);
            println!(
                "  {:.2}  {}  {mw}   {pw}   {mt}  {dm} {dp}",
                lambda,
                pad_start(&seed.to_string(), 4)
            );
        }
    }

    // Aggregate by lambda (insertion order preserved via Vec of keys).
    let mut by_lambda: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
    let mut lambda_order: Vec<u64> = Vec::new();
    for (i, tr) in trials.iter().enumerate() {
        let key = tr.lambda.to_bits();
        if !by_lambda.contains_key(&key) {
            lambda_order.push(key);
        }
        by_lambda.entry(key).or_default().push(i);
    }
    println!();
    println!("  Per-rate summary (mean over seeds):");
    for key in &lambda_order {
        let idxs = &by_lambda[key];
        let lambda = f64::from_bits(*key);
        let mean_of = |sel: &dyn Fn(&TrialAggregate) -> f64| -> f64 {
            idxs.iter().map(|&i| sel(&trials[i])).sum::<f64>() / idxs.len() as f64
        };
        let u_mw = mean_of(&|t| t.uncoord.mean_wait);
        let c_mw = mean_of(&|t| t.coord.mean_wait);
        let u_p95 = mean_of(&|t| t.uncoord.p95_wait);
        let c_p95 = mean_of(&|t| t.coord.p95_wait);
        let u_mt = mean_of(&|t| t.uncoord.mean_total);
        let c_mt = mean_of(&|t| t.coord.mean_total);
        println!(
            "    λ={:.2}: meanWait {:.2}→{:.2} ({}), p95Wait {:.1}→{:.1} ({}), meanTotal {:.2}→{:.2} ({})",
            lambda, u_mw, c_mw, pct(c_mw, u_mw), u_p95, c_p95, pct(c_p95, u_p95), u_mt, c_mt, pct(c_mt, u_mt)
        );
    }

    let out_dir = Path::new("out");
    if let Err(e) = fs::create_dir_all(out_dir) {
        eprintln!("# could not create out dir: {e}");
        return;
    }
    let out_path = out_dir.join("elevator-dispatch-sweep.json");
    let trials_json = JsonValue::Array(
        trials
            .iter()
            .map(|t| {
                JsonValue::Object(vec![
                    ("seed".to_string(), jn(t.seed as f64)),
                    ("lambda".to_string(), jn(t.lambda)),
                    ("uncoord".to_string(), agg_json(&t.uncoord)),
                    ("coord".to_string(), agg_json(&t.coord)),
                ])
            })
            .collect(),
    );
    let payload = JsonValue::Object(vec![
        (
            "seeds".to_string(),
            JsonValue::Array(seeds.iter().map(|s| jn(*s as f64)).collect()),
        ),
        (
            "lambdas".to_string(),
            JsonValue::Array(lambdas.iter().map(|l| jn(*l)).collect()),
        ),
        ("simT".to_string(), jn(sim_t)),
        ("trials".to_string(), trials_json),
    ]);
    if let Err(e) = fs::write(&out_path, payload.to_string_pretty(2)) {
        eprintln!("# could not write artifact: {e}");
        return;
    }
    println!();
    println!("# wrote {}", out_path.display());
}
