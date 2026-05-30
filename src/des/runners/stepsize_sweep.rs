//! Port of `src/des/runners/stepsize-sweep.ts`.
//!
//! Sweeps the framework kernel's `stepSize` against the (stepSize-free) FEL
//! reference, comparing time-averaged compartment populations; emits a markdown
//! ratio table, ASCII bar charts, and CSV/SVG artifacts. The TS top-level
//! `main()` becomes [`run`].
//!
//! ## PORT NOTE
//!
//!   * `process.env.{N,STEPSIZES}` → `std::env::var` (+ comma split/parse).
//!   * `fs`/`path` CSV/SVG writes → `std::fs`/`std::path`; artifacts go to
//!     `./out/` relative to the working directory.
//!   * seeds `0x40000+i` / `0x50000+i` kept verbatim.
//!   * `console.log` → `println!`.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::time::Instant;

use super::fel_runner::run_fel_once;
use super::framework_runner::run_framework_once;
use super::stats::{mean, stddev};
use super::types::{default_config, RunOpts, RunResult, SimConfig, COMPARTMENT_ORDER};

/// One sweep point (`interface SweepPoint`).
struct SweepPoint {
    step_size: f64,
    fw_mean: HashMap<String, f64>,
    fw_sd: HashMap<String, f64>,
    fel_mean: HashMap<String, f64>,
    fel_sd: HashMap<String, f64>,
    ratio: HashMap<String, f64>,
    fw_wall_ms: u128,
    fel_wall_ms: u128,
}

fn fmt(n: f64, d: usize) -> String {
    if n.is_finite() {
        format!("{n:.d$}")
    } else {
        n.to_string()
    }
}

fn pad_end(s: &str, width: usize) -> String {
    if s.chars().count() >= width {
        s.to_string()
    } else {
        format!("{s}{}", " ".repeat(width - s.chars().count()))
    }
}

fn pad_start(s: &str, width: usize) -> String {
    if s.chars().count() >= width {
        s.to_string()
    } else {
        format!("{}{s}", " ".repeat(width - s.chars().count()))
    }
}

fn get(m: &HashMap<String, f64>, c: &str) -> f64 {
    m.get(c).copied().unwrap_or(0.0)
}

fn ascii_bars(label: &str, values: &[f64], step_sizes: &[f64], max_len: usize) -> String {
    let max = values.iter().copied().fold(1.0_f64, f64::max);
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let bars = ((v / max) * max_len as f64).round() as usize;
            format!(
                "  ss={}  {label}={}  |{}",
                pad_start(&fmt(step_sizes[i], 2), 5),
                pad_start(&fmt(v, 3), 7),
                "#".repeat(bars)
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// `main()` — run the sweep and write artifacts.
pub fn run() {
    let n: usize = std::env::var("N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    let step_sizes: Vec<f64> = std::env::var("STEPSIZES")
        .unwrap_or_else(|_| "1.0,0.5,0.1,0.05".to_string())
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    let step_sizes_json = format!(
        "[{}]",
        step_sizes
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("stepsize-sweep.ts: {n} reps per stepSize, sweeping {step_sizes_json}");

    let default_cfg = default_config();

    // FEL is stepSize-independent; run it once with N reps for comparison.
    let mut fel_runs_per_step: Vec<RunResult> = Vec::new();
    let t0 = Instant::now();
    for i in 0..n {
        fel_runs_per_step.push(run_fel_once(
            &default_cfg,
            &RunOpts {
                seed: Some(0x40000 + i as u64),
                ..Default::default()
            },
        ));
    }
    let fel_wall = t0.elapsed().as_millis();

    let mut fel_mean: HashMap<String, f64> = HashMap::new();
    let mut fel_sd: HashMap<String, f64> = HashMap::new();
    for c in COMPARTMENT_ORDER {
        let xs: Vec<f64> = fel_runs_per_step
            .iter()
            .map(|r| get(&r.time_avg_populations, c))
            .collect();
        fel_mean.insert(c.to_string(), mean(&xs));
        fel_sd.insert(c.to_string(), stddev(&xs));
    }
    println!("fel reference: {n} reps, total wall {fel_wall} ms");

    let mut sweep: Vec<SweepPoint> = Vec::new();

    for &ss in &step_sizes {
        let cfg = SimConfig {
            step_size: ss,
            ..default_cfg.clone()
        };
        let t_start = Instant::now();
        let mut reps: Vec<RunResult> = Vec::new();
        for i in 0..n {
            reps.push(run_framework_once(
                &cfg,
                &RunOpts {
                    seed: Some(0x50000 + i as u64),
                    ..Default::default()
                },
            ));
        }
        let fw_wall = t_start.elapsed().as_millis();

        let mut fw_mean: HashMap<String, f64> = HashMap::new();
        let mut fw_sd: HashMap<String, f64> = HashMap::new();
        let mut ratio: HashMap<String, f64> = HashMap::new();
        for c in COMPARTMENT_ORDER {
            let xs: Vec<f64> = reps
                .iter()
                .map(|r| get(&r.time_avg_populations, c))
                .collect();
            let m = mean(&xs);
            fw_mean.insert(c.to_string(), m);
            fw_sd.insert(c.to_string(), stddev(&xs));
            let fm = get(&fel_mean, c);
            ratio.insert(c.to_string(), if fm > 0.0 { m / fm } else { f64::NAN });
        }

        println!(
            "  stepSize={}  framework wall {fw_wall} ms (mean per rep {} ms)",
            pad_start(&ss.to_string(), 5),
            fmt(fw_wall as f64 / n as f64, 1)
        );

        sweep.push(SweepPoint {
            step_size: ss,
            fw_mean,
            fw_sd,
            fel_mean: fel_mean.clone(),
            fel_sd: fel_sd.clone(),
            ratio,
            fw_wall_ms: fw_wall,
            fel_wall_ms: fel_wall,
        });
    }

    // ---- Markdown table --------------------------------------------------
    println!();
    println!("=== framework / fel time-averaged population ratios ===");
    println!("(1.000 = perfect agreement; > 1 means framework over-estimates)");
    println!();
    let mut header_cells: Vec<String> = vec!["stepSize".to_string()];
    for c in COMPARTMENT_ORDER {
        header_cells.push(c.to_string());
    }
    println!(
        "{}",
        header_cells
            .iter()
            .map(|h| pad_end(h, 10))
            .collect::<Vec<_>>()
            .join("  ")
    );
    for sp in &sweep {
        let mut cells: Vec<String> = vec![fmt(sp.step_size, 3)];
        for c in COMPARTMENT_ORDER {
            cells.push(fmt(get(&sp.ratio, c), 3));
        }
        println!(
            "{}",
            cells
                .iter()
                .map(|s| pad_end(s, 10))
                .collect::<Vec<_>>()
                .join("  ")
        );
    }

    // ---- Per-compartment bar charts --------------------------------------
    let fw_s: Vec<f64> = sweep.iter().map(|s| get(&s.fw_mean, "S")).collect();
    println!();
    println!("=== ASCII bar chart: framework <S>(t) vs stepSize ===");
    println!("{}", ascii_bars("<S>", &fw_s, &step_sizes, 40));
    println!(
        "  fel <S>={}     <-- this is the target",
        fmt(get(&fel_mean, "S"), 3)
    );

    let fw_e: Vec<f64> = sweep.iter().map(|s| get(&s.fw_mean, "E")).collect();
    println!();
    println!("=== ASCII bar chart: framework <E>(t) vs stepSize ===");
    println!("{}", ascii_bars("<E>", &fw_e, &step_sizes, 40));
    println!(
        "  fel <E>={}     <-- this is the target",
        fmt(get(&fel_mean, "E"), 3)
    );

    let fw_ip: Vec<f64> = sweep.iter().map(|s| get(&s.fw_mean, "I-P")).collect();
    println!();
    println!("=== ASCII bar chart: framework <I-P>(t) vs stepSize ===");
    println!("{}", ascii_bars("<I-P>", &fw_ip, &step_sizes, 40));
    println!(
        "  fel <I-P>={}     <-- this is the target",
        fmt(get(&fel_mean, "I-P"), 3)
    );

    let ratio_ip: Vec<f64> = sweep.iter().map(|s| get(&s.ratio, "I-P")).collect();
    println!();
    println!("=== ASCII bar chart: framework / fel ratio for <I-P> ===");
    println!("{}", ascii_bars("ratio<I-P>", &ratio_ip, &step_sizes, 40));

    // ---- Persist as CSV --------------------------------------------------
    let out_dir = Path::new("out");
    if let Err(e) = fs::create_dir_all(out_dir) {
        eprintln!("[stepsize-sweep] could not create out dir: {e}");
        return;
    }
    let csv_path = out_dir.join("stepsize-sweep.csv");
    let mut cols: Vec<String> = vec!["stepSize".to_string()];
    for c in COMPARTMENT_ORDER {
        cols.push(format!("fw_{c}_mean"));
    }
    for c in COMPARTMENT_ORDER {
        cols.push(format!("fw_{c}_sd"));
    }
    for c in COMPARTMENT_ORDER {
        cols.push(format!("fel_{c}_mean"));
    }
    for c in COMPARTMENT_ORDER {
        cols.push(format!("fel_{c}_sd"));
    }
    for c in COMPARTMENT_ORDER {
        cols.push(format!("ratio_{c}"));
    }
    let mut lines: Vec<String> = vec![cols.join(",")];
    for sp in &sweep {
        let mut row: Vec<String> = vec![sp.step_size.to_string()];
        for c in COMPARTMENT_ORDER {
            row.push(format!("{:.6}", get(&sp.fw_mean, c)));
        }
        for c in COMPARTMENT_ORDER {
            row.push(format!("{:.6}", get(&sp.fw_sd, c)));
        }
        for c in COMPARTMENT_ORDER {
            row.push(format!("{:.6}", get(&sp.fel_mean, c)));
        }
        for c in COMPARTMENT_ORDER {
            row.push(format!("{:.6}", get(&sp.fel_sd, c)));
        }
        for c in COMPARTMENT_ORDER {
            row.push(format!("{:.6}", get(&sp.ratio, c)));
        }
        lines.push(row.join(","));
    }
    let csv = lines.join("\n") + "\n";
    if let Err(e) = fs::write(&csv_path, csv) {
        eprintln!("[stepsize-sweep] could not write CSV: {e}");
        return;
    }

    // ---- Persist as SVG --------------------------------------------------
    let svg = render_svg(&sweep, &fel_mean);
    let svg_path = out_dir.join("stepsize-sweep.svg");
    if let Err(e) = fs::write(&svg_path, svg) {
        eprintln!("[stepsize-sweep] could not write SVG: {e}");
        return;
    }

    println!("\nartifacts written:");
    println!("  {}", csv_path.display());
    println!("  {}", svg_path.display());
}

fn render_svg(sweep: &[SweepPoint], fel_mean: &HashMap<String, f64>) -> String {
    let w = 760.0_f64;
    let h = 420.0_f64;
    let pad = 60.0_f64;
    let compartments = ["S", "E", "I-P", "I-A", "I-S", "I-H"];
    let colors = [
        "#d62728", "#ff7f0e", "#2ca02c", "#1f77b4", "#9467bd", "#8c564b",
    ];

    let xs: Vec<f64> = sweep.iter().map(|s| s.step_size.log10()).collect();
    let xmin = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let xmax = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut yvals: Vec<f64> = Vec::new();
    for c in compartments {
        for sp in sweep {
            yvals.push(get(&sp.fw_mean, c));
        }
        yvals.push(get(fel_mean, c));
    }
    let ymax = yvals.iter().copied().fold(0.001_f64, f64::max);

    let x_to_px = |x: f64| -> f64 {
        if xmax == xmin {
            pad + w / 2.0
        } else {
            pad + ((x - xmin) / (xmax - xmin)) * (w - 2.0 * pad)
        }
    };
    let y_to_px = |y: f64| -> f64 { h - pad - (y / ymax) * (h - 2.0 * pad) };

    let mut svg = format!(
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 {w} {h}' font-family='monospace' font-size='11'>"
    );
    svg.push_str("<rect width='100%' height='100%' fill='white'/>");
    svg.push_str(&format!(
        "<line x1='{pad}' y1='{}' x2='{}' y2='{}' stroke='black'/>",
        h - pad,
        w - pad,
        h - pad
    ));
    svg.push_str(&format!(
        "<line x1='{pad}' y1='{pad}' x2='{pad}' y2='{}' stroke='black'/>",
        h - pad
    ));
    svg.push_str(&format!(
        "<text x='{}' y='{}' text-anchor='middle' font-size='14' font-weight='bold'>Framework time-averaged compartment populations vs stepSize (FEL reference dashed)</text>",
        w / 2.0,
        pad - 25.0
    ));
    svg.push_str(&format!(
        "<text x='{}' y='{}' text-anchor='middle'>log10(stepSize, days)</text>",
        w / 2.0,
        h - 15.0
    ));
    svg.push_str(&format!(
        "<text x='15' y='{}' text-anchor='middle' transform='rotate(-90 15 {})'>mean population</text>",
        h / 2.0,
        h / 2.0
    ));
    for sp in sweep {
        let px = x_to_px(sp.step_size.log10());
        svg.push_str(&format!(
            "<line x1='{px}' y1='{}' x2='{px}' y2='{}' stroke='black'/>",
            h - pad,
            h - pad + 5.0
        ));
        svg.push_str(&format!(
            "<text x='{px}' y='{}' text-anchor='middle'>{}</text>",
            h - pad + 18.0,
            sp.step_size
        ));
    }
    for i in 0..=5 {
        let yv = (ymax * i as f64) / 5.0;
        let py = y_to_px(yv);
        svg.push_str(&format!(
            "<line x1='{}' y1='{py}' x2='{pad}' y2='{py}' stroke='black'/>",
            pad - 5.0
        ));
        svg.push_str(&format!(
            "<text x='{}' y='{}' text-anchor='end'>{:.2}</text>",
            pad - 8.0,
            py + 4.0,
            yv
        ));
    }
    for (i, c) in compartments.iter().enumerate() {
        let color = colors[i];
        let points: Vec<String> = sweep
            .iter()
            .map(|sp| {
                format!(
                    "{},{}",
                    x_to_px(sp.step_size.log10()),
                    y_to_px(get(&sp.fw_mean, c))
                )
            })
            .collect();
        svg.push_str(&format!(
            "<polyline points='{}' fill='none' stroke='{color}' stroke-width='2'/>",
            points.join(" ")
        ));
        for sp in sweep {
            let cx = x_to_px(sp.step_size.log10());
            let cy = y_to_px(get(&sp.fw_mean, c));
            svg.push_str(&format!(
                "<circle cx='{cx}' cy='{cy}' r='4' fill='{color}'/>"
            ));
        }
        let py = y_to_px(get(fel_mean, c));
        svg.push_str(&format!(
            "<line x1='{pad}' y1='{py}' x2='{}' y2='{py}' stroke='{color}' stroke-width='1' stroke-dasharray='5,3' opacity='0.6'/>",
            w - pad
        ));
    }
    let lx = w - pad - 110.0;
    let ly = pad + 10.0;
    svg.push_str(&format!(
        "<rect x='{}' y='{}' width='120' height='{}' fill='white' stroke='#888'/>",
        lx - 10.0,
        ly - 14.0,
        compartments.len() as f64 * 16.0 + 8.0
    ));
    for (i, c) in compartments.iter().enumerate() {
        svg.push_str(&format!(
            "<line x1='{lx}' y1='{}' x2='{}' y2='{}' stroke='{}' stroke-width='2'/>",
            ly + i as f64 * 16.0 - 4.0,
            lx + 18.0,
            ly + i as f64 * 16.0 - 4.0,
            colors[i]
        ));
        svg.push_str(&format!(
            "<text x='{}' y='{}'><{c}>(t)  fel={:.3}</text>",
            lx + 24.0,
            ly + i as f64 * 16.0,
            get(fel_mean, c)
        ));
    }
    svg.push_str("</svg>");
    svg
}
