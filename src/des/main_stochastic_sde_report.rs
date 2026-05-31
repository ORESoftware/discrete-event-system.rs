//! Port of `src/des/main-stochastic-sde-report.ts`.
//!
//! Report tool: runs the stochastic-SDE + 3-ML-algorithm demo and writes a
//! styled, data-backed HTML report into `out/stochastic-sde/report.html`.
//!
//! Conversion notes:
//!   - `class StochasticSdeReport` → struct + impl; `fs` write → `std::fs`.
//!   - `RunReportPage` → `crate::des::animation::run_report::RunReportPage`.
//!
//! PORT NOTE: the TS shells out (`execFileSync(ts-node,
//! ['src/des/main-stochastic-sde.ts'])`) to capture the sibling script's stdout.
//! This Rust report computes the same seeded run in-process and renders both the
//! textual summary and visualization traces from typed data, avoiding stdout
//! capture while keeping the report inspectable from `file://`.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::rc::Rc;

use serde_json::{json, Value};

use crate::des::animation::run_report::RunReportPage;
use crate::des::general::control_systems::empirical_control::Mulberry32;
use crate::des::general::control_systems::sde_learning::{
    DenoisingDiffusionModel, DiffusionOptions, DiffusionTrainOptions, EnkfOptions,
    EnsembleKalmanFilter, EnsembleKalmanFilterStation, GbmFamily, SdeMaximumLikelihoodEstimator,
    SdeMleOptions,
};
use crate::des::general::control_systems::stochastic_sde::{
    EulerMaruyamaIntegrator, GeometricBrownianMotion, SdeChannels, SdeEstimateSinkStation,
    SdePlantOptions, SdePlantStation, StochasticDcMotor, StochasticDcMotorSpec,
};
use crate::des::general::des_base::runner::{run_iterative_des, IterativeRunOptions};
use crate::des::general::des_base::station::{DESStation, StationRef};

const TITLE: &str = "Stochastic Differential Equations + 3 ML algorithms";
const SUBTITLE: &str =
    "Euler–Maruyama SDE engine with system identification, ensemble filtering, and score-based diffusion.";

struct StochasticSdeReport;

impl StochasticSdeReport {
    fn run(&self) {
        let data = build_report_data();
        let log = render_run_output(&data);
        println!("{log}");

        let out = std::path::Path::new("out")
            .join("stochastic-sde")
            .join("report.html");
        match write_report_html(&out, &data, &log) {
            Ok(abs) => println!("Stochastic-SDE report: {}", abs),
            Err(e) => eprintln!("Stochastic-SDE report write failed: {e}"),
        }
    }
}

#[derive(Clone, Debug)]
struct GbmTracePoint {
    t: f64,
    representative_path: f64,
    empirical_mean: f64,
    analytic_mean: f64,
}

#[derive(Clone, Debug)]
struct GbmEngineReport {
    horizon: f64,
    empirical_mean: f64,
    analytic_mean: f64,
    empirical_var: f64,
    analytic_var: f64,
    paths: usize,
    trace: Vec<GbmTracePoint>,
}

#[derive(Clone, Debug)]
struct MleReport {
    true_mu: f64,
    true_sigma: f64,
    learned_mu: f64,
    learned_sigma: f64,
    final_neg_log_lik: f64,
    iterations: usize,
}

#[derive(Clone, Debug)]
struct EnkfTracePoint {
    t: f64,
    true_current: f64,
    estimated_current: f64,
    true_speed: f64,
    estimated_speed: f64,
}

#[derive(Clone, Debug)]
struct EnkfReport {
    current_rmse: f64,
    speed_rmse: f64,
    baseline_current_rmse: f64,
    trace: Vec<EnkfTracePoint>,
}

#[derive(Clone, Debug)]
struct HistogramBin {
    center: f64,
    target_density: f64,
    learned_density: f64,
}

#[derive(Clone, Debug)]
struct DiffusionReport {
    data_mean: f64,
    data_std: f64,
    sample_mean: f64,
    sample_std: f64,
    final_loss: f64,
    near_negative_mode: f64,
    histogram: Vec<HistogramBin>,
}

#[derive(Clone, Debug)]
struct ReportData {
    gbm: GbmEngineReport,
    mle: MleReport,
    enkf: EnkfReport,
    diffusion: DiffusionReport,
}

fn build_report_data() -> ReportData {
    ReportData {
        gbm: build_gbm_engine_report(),
        mle: build_mle_report(),
        enkf: build_enkf_report(),
        diffusion: build_diffusion_report(),
    }
}

fn build_gbm_engine_report() -> GbmEngineReport {
    let gbm = GeometricBrownianMotion::new(0.1, 0.3);
    let em = EulerMaruyamaIntegrator::new();
    let x0 = 1.0;
    let dt = 0.002;
    let steps = 1000usize;
    let paths = 4000usize;
    let sample_every = 25usize;
    let sample_steps: Vec<usize> = (0..=steps).step_by(sample_every).collect();
    let mut mean_sum = vec![0.0; sample_steps.len()];
    let mut representative_path = vec![0.0; sample_steps.len()];
    let mut sum = 0.0;
    let mut sum_sq = 0.0;
    for p in 0..paths {
        let mut rng = Mulberry32::new(1000 + p as u32);
        let res = em.simulate(&gbm, &[x0], dt, steps, &mut rng);
        let x_t = res.path[res.path.len() - 1][0];
        sum += x_t;
        sum_sq += x_t * x_t;
        for (i, step_idx) in sample_steps.iter().enumerate() {
            let v = res.path[*step_idx][0];
            mean_sum[i] += v;
            if p == 0 {
                representative_path[i] = v;
            }
        }
    }
    let empirical_mean = sum / paths as f64;
    let empirical_var = sum_sq / paths as f64 - empirical_mean * empirical_mean;
    let horizon = dt * steps as f64;
    let trace = sample_steps
        .iter()
        .enumerate()
        .map(|(i, step_idx)| {
            let t = *step_idx as f64 * dt;
            GbmTracePoint {
                t,
                representative_path: representative_path[i],
                empirical_mean: mean_sum[i] / paths as f64,
                analytic_mean: gbm.mean_at(x0, t),
            }
        })
        .collect();
    GbmEngineReport {
        horizon,
        empirical_mean,
        analytic_mean: gbm.mean_at(x0, horizon),
        empirical_var,
        analytic_var: gbm.var_at(x0, horizon),
        paths,
        trace,
    }
}

fn build_mle_report() -> MleReport {
    let true_mu = 0.12;
    let true_sigma = 0.3;
    let gbm = GeometricBrownianMotion::new(true_mu, true_sigma);
    let mut rng = Mulberry32::new(77);
    let sim = EulerMaruyamaIntegrator::new().simulate(&gbm, &[1.0], 0.004, 6000, &mut rng);
    let est = SdeMaximumLikelihoodEstimator::new(SdeMleOptions {
        iterations: Some(1500),
        learning_rate: Some(0.05),
        fd_eps: None,
    });
    let fit = est.fit(&GbmFamily, &sim.times, &sim.path);
    MleReport {
        true_mu,
        true_sigma,
        learned_mu: fit.params["mu"],
        learned_sigma: fit.params["sigma"],
        final_neg_log_lik: fit.final_neg_log_lik,
        iterations: fit.iterations,
    }
}

fn build_enkf_report() -> EnkfReport {
    let spec = StochasticDcMotorSpec {
        resistance: 2.0,
        inductance: 0.5,
        back_emf_constant: 0.1,
        torque_constant: 0.1,
        inertia: 0.02,
        friction: 0.002,
        voltage: 12.0,
        load_torque: None,
        current_noise: 0.4,
        speed_noise: 0.5,
    };
    let dt = 0.01;
    let steps = 500usize;
    let h: Vec<Vec<f64>> = vec![vec![0.0, 1.0]];
    let plant = Rc::new(RefCell::new(SdePlantStation::new(
        "motor-plant",
        SdePlantOptions {
            system: Box::new(StochasticDcMotor::new(spec.clone())),
            x0: vec![0.0, 0.0],
            dt,
            steps,
            observation_matrix: Some(h.clone()),
            observation_noise_std: Some(vec![0.6]),
            seed: Some(5),
        },
    )));
    let filter = EnsembleKalmanFilter::new(
        Box::new(StochasticDcMotor::new(spec.clone())),
        dt,
        EnkfOptions {
            ensemble_size: Some(150),
            observation_matrix: h.clone(),
            observation_noise_var: vec![0.36],
            initial_mean: vec![0.0, 0.0],
            initial_std: vec![2.0, 5.0],
            seed: Some(9),
        },
    );
    let enkf = Rc::new(RefCell::new(EnsembleKalmanFilterStation::new(
        "enkf", filter,
    )));
    let sink = Rc::new(RefCell::new(SdeEstimateSinkStation::new("sink")));

    let plant_ref: StationRef = plant.clone();
    let enkf_ref: StationRef = enkf.clone();
    let sink_ref: StationRef = sink.clone();

    plant.borrow_mut().core_mut().pipe(
        enkf_ref.clone(),
        SdeChannels::OBSERVATION,
        SdeChannels::OBSERVATION,
    );
    plant
        .borrow_mut()
        .core_mut()
        .pipe(sink_ref.clone(), SdeChannels::STATE, SdeChannels::STATE);
    enkf.borrow_mut().core_mut().pipe(
        sink_ref.clone(),
        SdeChannels::ESTIMATE,
        SdeChannels::ESTIMATE,
    );

    run_iterative_des(
        vec![plant_ref, enkf_ref, sink_ref],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(steps + 5),
            ..Default::default()
        },
    );

    let sink_b = sink.borrow();
    let rmse = sink_b.rmse_by_dimension();
    let n = sink_b.truth.len() as f64;
    let mean_i = sink_b.truth.iter().map(|t| t.state[0]).sum::<f64>() / n;
    let baseline_current_rmse = (sink_b
        .truth
        .iter()
        .map(|t| {
            let d = t.state[0] - mean_i;
            d * d
        })
        .sum::<f64>()
        / n)
        .sqrt();
    let mut truth_by_step = BTreeMap::new();
    for t in &sink_b.truth {
        truth_by_step.insert(t.step, t.clone());
    }
    let mut trace = Vec::new();
    for e in &sink_b.estimates {
        if e.step != 1 && e.step != steps && e.step % 5 != 0 {
            continue;
        }
        if let Some(t) = truth_by_step.get(&e.step) {
            trace.push(EnkfTracePoint {
                t: e.time,
                true_current: t.state[0],
                estimated_current: e.mean[0],
                true_speed: t.state[1],
                estimated_speed: e.mean[1],
            });
        }
    }
    EnkfReport {
        current_rmse: rmse[0],
        speed_rmse: rmse[1],
        baseline_current_rmse,
        trace,
    }
}

fn build_diffusion_report() -> DiffusionReport {
    let mut rng = Mulberry32::new(2024);
    let mut data: Vec<f64> = Vec::new();
    for _ in 0..3000 {
        let mode = if rng.next() < 0.5 { -2.0 } else { 2.0 };
        data.push(mode + rng.normal() * 0.4);
    }
    let mut model = DenoisingDiffusionModel::new(DiffusionOptions {
        steps: Some(100),
        beta_min: None,
        beta_max: Some(0.2),
        hidden: Some(128),
        seed: Some(3),
    });
    let final_loss = model.train(
        &data,
        DiffusionTrainOptions {
            iterations: Some(60000),
            learning_rate: Some(0.004),
        },
    );
    let samples = model.sample(3000);
    let data_stats = DenoisingDiffusionModel::summarise(&data);
    let sample_stats = DenoisingDiffusionModel::summarise(&samples);
    let near_negative_mode =
        samples.iter().filter(|s| **s < 0.0).count() as f64 / samples.len() as f64;
    DiffusionReport {
        data_mean: data_stats.mean,
        data_std: data_stats.std,
        sample_mean: sample_stats.mean,
        sample_std: sample_stats.std,
        final_loss,
        near_negative_mode,
        histogram: build_histogram(&data, &samples, -3.6, 3.6, 36),
    }
}

fn build_histogram(
    target: &[f64],
    learned: &[f64],
    min: f64,
    max: f64,
    bins: usize,
) -> Vec<HistogramBin> {
    let width = (max - min) / bins as f64;
    let target_counts = histogram_counts(target, min, width, bins);
    let learned_counts = histogram_counts(learned, min, width, bins);
    (0..bins)
        .map(|i| HistogramBin {
            center: min + (i as f64 + 0.5) * width,
            target_density: target_counts[i] / (target.len() as f64 * width),
            learned_density: learned_counts[i] / (learned.len() as f64 * width),
        })
        .collect()
}

fn histogram_counts(values: &[f64], min: f64, width: f64, bins: usize) -> Vec<f64> {
    let mut counts = vec![0.0; bins];
    for v in values {
        if !v.is_finite() {
            continue;
        }
        let mut idx = ((*v - min) / width).floor() as isize;
        if idx == bins as isize {
            idx -= 1;
        }
        if idx >= 0 && idx < bins as isize {
            counts[idx as usize] += 1.0;
        }
    }
    counts
}

fn render_run_output(data: &ReportData) -> String {
    format!(
        "================ 0. SDE engine: GBM Euler–Maruyama vs analytic ================\n\
           T={:.2}  E[X_T]: empirical {:.4} vs analytic {:.4}\n\
                    Var[X_T]: empirical {:.4} vs analytic {:.4}\n\n\
         ================ 1. ML system-id: maximum-likelihood SDE fit ================\n\
           true   : mu={}, sigma={}\n\
           learned: mu={:.4}, sigma={:.4}   (NLL={:.1}, {} Adam steps)\n\n\
         ================ 2. ML filtering: Ensemble Kalman Filter (DES pipeline) ================\n\
           observed: speed ω (noisy, σ=0.6);  hidden: current i\n\
           EnKF RMSE  → current i = {:.4},  speed ω = {:.4}\n\
           baseline   → current i (guess mean) = {:.4}   ⇒ filter recovers the hidden state\n\n\
         ================ 3. ML generative: score-based diffusion (reverse SDE) ================\n\
           target  : bimodal N(±2, 0.4²)   data mean/std = {:.3} / {:.3}\n\
           learned : sample mean/std = {:.3} / {:.3}   (final DSM loss {:.4})\n\
           modes   : {:.0}% near −2, {:.0}% near +2  (target ≈ 50/50)",
        data.gbm.horizon,
        data.gbm.empirical_mean,
        data.gbm.analytic_mean,
        data.gbm.empirical_var,
        data.gbm.analytic_var,
        num_str(data.mle.true_mu),
        num_str(data.mle.true_sigma),
        data.mle.learned_mu,
        data.mle.learned_sigma,
        data.mle.final_neg_log_lik,
        data.mle.iterations,
        data.enkf.current_rmse,
        data.enkf.speed_rmse,
        data.enkf.baseline_current_rmse,
        data.diffusion.data_mean,
        data.diffusion.data_std,
        data.diffusion.sample_mean,
        data.diffusion.sample_std,
        data.diffusion.final_loss,
        data.diffusion.near_negative_mode * 100.0,
        (1.0 - data.diffusion.near_negative_mode) * 100.0,
    )
}

fn render_report_html(data: &ReportData, log: &str) -> String {
    REPORT_TEMPLATE
        .replace("__TITLE__", &RunReportPage::escape(TITLE))
        .replace("__SUBTITLE__", &RunReportPage::escape(SUBTITLE))
        .replace("__CSS__", REPORT_CSS)
        .replace("__METRICS__", &render_metrics(data))
        .replace("__RUN_OUTPUT__", &RunReportPage::escape(log))
        .replace("__CHART_DATA__", &chart_data_json(data))
        .replace("__JS__", REPORT_JS)
}

fn write_report_html(
    out: &std::path::Path,
    data: &ReportData,
    log: &str,
) -> std::io::Result<String> {
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    std::fs::write(out, render_report_html(data, log))?;
    Ok(std::fs::canonicalize(out)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| out.to_string_lossy().into_owned()))
}

fn render_metrics(data: &ReportData) -> String {
    let rows = [
        (
            "GBM terminal mean",
            format!("{:.4}", data.gbm.empirical_mean),
        ),
        (
            "GBM analytic mean",
            format!("{:.4}", data.gbm.analytic_mean),
        ),
        (
            "GBM terminal variance",
            format!("{:.4}", data.gbm.empirical_var),
        ),
        ("GBM paths", data.gbm.paths.to_string()),
        ("MLE μ", format!("{:.4}", data.mle.learned_mu)),
        ("MLE σ", format!("{:.4}", data.mle.learned_sigma)),
        (
            "EnKF current RMSE",
            format!("{:.4}", data.enkf.current_rmse),
        ),
        (
            "Baseline current RMSE",
            format!("{:.4}", data.enkf.baseline_current_rmse),
        ),
        (
            "Diffusion sample mean/std",
            format!(
                "{:.3} / {:.3}",
                data.diffusion.sample_mean, data.diffusion.sample_std
            ),
        ),
        (
            "Diffusion mode split",
            format!(
                "{:.0}% / {:.0}%",
                data.diffusion.near_negative_mode * 100.0,
                (1.0 - data.diffusion.near_negative_mode) * 100.0
            ),
        ),
    ];
    rows.iter()
        .map(|(label, value)| {
            format!(
                "<div class=\"metric\"><span>{}</span><strong>{}</strong></div>",
                RunReportPage::escape(label),
                RunReportPage::escape(value)
            )
        })
        .collect::<Vec<_>>()
        .join("")
}

fn chart_data_json(data: &ReportData) -> String {
    let gbm: Vec<Value> = data
        .gbm
        .trace
        .iter()
        .map(|p| {
            json!({
                "t": p.t,
                "representativePath": p.representative_path,
                "empiricalMean": p.empirical_mean,
                "analyticMean": p.analytic_mean,
            })
        })
        .collect();
    let enkf: Vec<Value> = data
        .enkf
        .trace
        .iter()
        .map(|p| {
            json!({
                "t": p.t,
                "trueCurrent": p.true_current,
                "estimatedCurrent": p.estimated_current,
                "trueSpeed": p.true_speed,
                "estimatedSpeed": p.estimated_speed,
            })
        })
        .collect();
    let diffusion: Vec<Value> = data
        .diffusion
        .histogram
        .iter()
        .map(|b| {
            json!({
                "x": b.center,
                "target": b.target_density,
                "learned": b.learned_density,
            })
        })
        .collect();
    json!({
        "gbm": gbm,
        "enkf": enkf,
        "diffusion": diffusion,
    })
    .to_string()
}

fn num_str(x: f64) -> String {
    if x.fract() == 0.0 && x.is_finite() {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

const REPORT_TEMPLATE: &str = r#"<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
__CSS__
</style></head><body><main>
<a class="back" href="../index.html">&larr; all simulations</a>
<h1>__TITLE__</h1>
<p class="sub">__SUBTITLE__</p>
<section><h2>What This Run Covers</h2><p class="desc">Models dX = f(X,t)dt + g(X,t)dW where the solution is a random process. The report renders three seeded traces from the run: geometric Brownian motion against its analytic mean, a DES-wired Ensemble Kalman Filter estimating hidden motor current, and the learned diffusion sampler against the bimodal target distribution.</p></section>
<section><h2>Run Metrics</h2><div class="metric-grid">__METRICS__</div></section>
<section><h2>Visual Diagnostics</h2>
<div class="viz-grid">
<figure><figcaption>GBM Euler–Maruyama</figcaption><svg id="gbm-chart" viewBox="0 0 760 300" role="img" aria-label="GBM path and mean chart"></svg></figure>
<figure><figcaption>EnKF Hidden-State Recovery</figcaption><svg id="enkf-chart" viewBox="0 0 760 300" role="img" aria-label="EnKF truth and estimate chart"></svg></figure>
<figure class="wide"><figcaption>Diffusion Target vs Generated Samples</figcaption><svg id="diffusion-chart" viewBox="0 0 760 300" role="img" aria-label="Diffusion histogram chart"></svg></figure>
</div></section>
<section><h2>Run Output</h2><pre class="log">__RUN_OUTPUT__</pre></section>
<script>
const REPORT_DATA = __CHART_DATA__;
__JS__
</script>
</main></body></html>"#;

const REPORT_CSS: &str = r#":root{color-scheme:dark;}
body{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;margin:0;background:#0b1021;color:#e6edf3;}
main{max-width:1120px;margin:0 auto;padding:28px 20px 72px;}
a.back{color:#58a6ff;text-decoration:none;font-size:.9rem;}
a.back:hover{text-decoration:underline;}
h1{font-size:1.7rem;margin:14px 0 4px;}
p.sub{color:#8b949e;margin:0 0 26px;font-size:.95rem;}
section{background:#161d33;border:1px solid #21262d;border-radius:10px;padding:18px 20px;margin:0 0 20px;}
h2{font-size:1.15rem;margin:0 0 8px;color:#f0f6fc;}
p.desc{color:#9aa5b1;margin:0 0 14px;font-size:.92rem;line-height:1.55;}
.metric-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;}
.metric{background:#0d1117;border:1px solid #283246;border-radius:8px;padding:10px 12px;}
.metric span{display:block;color:#8b949e;font-size:.78rem;margin:0 0 4px;}
.metric strong{display:block;color:#f0f6fc;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.98rem;font-weight:650;}
.viz-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px;align-items:start;}
figure{margin:0;background:#0d1117;border:1px solid #283246;border-radius:8px;padding:12px;}
figure.wide{grid-column:1/-1;}
figcaption{font-size:.88rem;color:#f0f6fc;font-weight:650;margin:0 0 8px;}
svg{display:block;width:100%;height:auto;overflow:visible;}
.axis{stroke:#526173;stroke-width:1;}
.grid-line{stroke:#253148;stroke-width:1;}
.tick-label{fill:#8b949e;font-size:11px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.axis-label{fill:#9aa5b1;font-size:12px;font-weight:600;}
.legend text{fill:#c9d1d9;font-size:12px;}
pre.log{background:#0d1117;border:1px solid #21262d;border-radius:8px;padding:14px 16px;overflow:auto;
font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.8rem;line-height:1.5;color:#c9d1d9;}
@media(max-width:820px){.viz-grid{grid-template-columns:1fr;}figure.wide{grid-column:auto;}main{padding-inline:14px;}}"#;

const REPORT_JS: &str = r#"(function () {
  const SVG_NS = 'http://www.w3.org/2000/svg';
  function el(name, attrs, text) {
    const node = document.createElementNS(SVG_NS, name);
    for (const [key, value] of Object.entries(attrs || {})) node.setAttribute(key, String(value));
    if (text !== undefined) node.textContent = text;
    return node;
  }
  function clear(svg) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
  }
  function extent(rows, keys) {
    let min = Infinity;
    let max = -Infinity;
    for (const row of rows) {
      for (const key of keys) {
        const value = row[key];
        if (Number.isFinite(value)) {
          min = Math.min(min, value);
          max = Math.max(max, value);
        }
      }
    }
    if (!Number.isFinite(min) || !Number.isFinite(max)) return [0, 1];
    if (Math.abs(max - min) < 1e-12) return [min - 1, max + 1];
    const pad = (max - min) * 0.08;
    return [min - pad, max + pad];
  }
  function scale(value, fromMin, fromMax, toMin, toMax) {
    return toMin + (value - fromMin) / (fromMax - fromMin) * (toMax - toMin);
  }
  function drawAxes(svg, bounds, xRange, yRange, xLabel, yLabel) {
    const {x, y, w, h} = bounds;
    for (let i = 0; i <= 4; i++) {
      const gx = x + w * i / 4;
      const gy = y + h * i / 4;
      svg.appendChild(el('line', {x1: gx, y1: y, x2: gx, y2: y + h, class: 'grid-line'}));
      svg.appendChild(el('line', {x1: x, y1: gy, x2: x + w, y2: gy, class: 'grid-line'}));
      const xv = xRange[0] + (xRange[1] - xRange[0]) * i / 4;
      const yv = yRange[1] - (yRange[1] - yRange[0]) * i / 4;
      svg.appendChild(el('text', {x: gx, y: y + h + 22, 'text-anchor': 'middle', class: 'tick-label'}, xv.toFixed(2)));
      svg.appendChild(el('text', {x: x - 10, y: gy + 4, 'text-anchor': 'end', class: 'tick-label'}, yv.toFixed(2)));
    }
    svg.appendChild(el('line', {x1: x, y1: y + h, x2: x + w, y2: y + h, class: 'axis'}));
    svg.appendChild(el('line', {x1: x, y1: y, x2: x, y2: y + h, class: 'axis'}));
    svg.appendChild(el('text', {x: x + w, y: y + h + 42, 'text-anchor': 'end', class: 'axis-label'}, xLabel));
    svg.appendChild(el('text', {x: 14, y: y - 12, class: 'axis-label'}, yLabel));
  }
  function drawLegend(svg, items, x, y) {
    const group = el('g', {class: 'legend'});
    let dx = 0;
    for (const item of items) {
      group.appendChild(el('line', {x1: x + dx, y1: y, x2: x + dx + 22, y2: y, stroke: item.color, 'stroke-width': 3, 'stroke-linecap': 'round'}));
      group.appendChild(el('text', {x: x + dx + 28, y: y + 4}, item.label));
      dx += item.label.length * 7 + 54;
    }
    svg.appendChild(group);
  }
  function linePath(rows, key, bounds, xRange, yRange) {
    const {x, y, w, h} = bounds;
    return rows.map((row, i) => {
      const px = scale(row.t, xRange[0], xRange[1], x, x + w);
      const py = scale(row[key], yRange[0], yRange[1], y + h, y);
      return `${i === 0 ? 'M' : 'L'}${px.toFixed(1)},${py.toFixed(1)}`;
    }).join(' ');
  }
  function drawLineChart(id, rows, series, labels) {
    const svg = document.getElementById(id);
    clear(svg);
    const bounds = {x: 64, y: 30, w: 660, h: 205};
    const xRange = [rows[0].t, rows[rows.length - 1].t];
    const yRange = extent(rows, series.map(s => s.key));
    drawAxes(svg, bounds, xRange, yRange, labels.x, labels.y);
    for (const s of series) {
      svg.appendChild(el('path', {d: linePath(rows, s.key, bounds, xRange, yRange), fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round', opacity: s.opacity || 1}));
    }
    drawLegend(svg, series, bounds.x, 22);
  }
  function drawHistogram(id, rows) {
    const svg = document.getElementById(id);
    clear(svg);
    const bounds = {x: 64, y: 30, w: 660, h: 205};
    const xRange = [rows[0].x, rows[rows.length - 1].x];
    const yRange = [0, Math.max(...rows.flatMap(r => [r.target, r.learned])) * 1.12];
    drawAxes(svg, bounds, xRange, yRange, 'sample value', 'density');
    const barSpan = bounds.w / rows.length;
    for (const row of rows) {
      const cx = scale(row.x, xRange[0], xRange[1], bounds.x, bounds.x + bounds.w);
      const targetH = scale(row.target, yRange[0], yRange[1], 0, bounds.h);
      const learnedH = scale(row.learned, yRange[0], yRange[1], 0, bounds.h);
      svg.appendChild(el('rect', {x: cx - barSpan * 0.38, y: bounds.y + bounds.h - targetH, width: barSpan * 0.34, height: targetH, fill: '#58a6ff', opacity: 0.76}));
      svg.appendChild(el('rect', {x: cx + barSpan * 0.04, y: bounds.y + bounds.h - learnedH, width: barSpan * 0.34, height: learnedH, fill: '#f59e0b', opacity: 0.78}));
    }
    drawLegend(svg, [{label: 'target', color: '#58a6ff'}, {label: 'generated', color: '#f59e0b'}], bounds.x, 22);
  }
  drawLineChart('gbm-chart', REPORT_DATA.gbm, [
    {key: 'representativePath', label: 'seeded path', color: '#f59e0b', width: 1.8, opacity: 0.86},
    {key: 'empiricalMean', label: 'empirical mean', color: '#2dd4bf', width: 3},
    {key: 'analyticMean', label: 'analytic mean', color: '#e6edf3', width: 2.2}
  ], {x: 'time', y: 'X_t'});
  drawLineChart('enkf-chart', REPORT_DATA.enkf, [
    {key: 'trueCurrent', label: 'true current', color: '#58a6ff', width: 2.2},
    {key: 'estimatedCurrent', label: 'estimated current', color: '#f59e0b', width: 2.2},
    {key: 'trueSpeed', label: 'true speed', color: '#2dd4bf', width: 1.7, opacity: 0.7},
    {key: 'estimatedSpeed', label: 'estimated speed', color: '#c084fc', width: 1.7, opacity: 0.7}
  ], {x: 'time', y: 'state'});
  drawHistogram('diffusion-chart', REPORT_DATA.diffusion);
})();"#;

/// Entry point (TS top-level script).
pub fn run() {
    StochasticSdeReport.run();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_data() -> ReportData {
        ReportData {
            gbm: GbmEngineReport {
                horizon: 1.0,
                empirical_mean: 1.1,
                analytic_mean: 1.1,
                empirical_var: 0.2,
                analytic_var: 0.2,
                paths: 2,
                trace: vec![GbmTracePoint {
                    t: 0.0,
                    representative_path: 1.0,
                    empirical_mean: 1.0,
                    analytic_mean: 1.0,
                }],
            },
            mle: MleReport {
                true_mu: 0.1,
                true_sigma: 0.3,
                learned_mu: 0.1,
                learned_sigma: 0.3,
                final_neg_log_lik: 12.0,
                iterations: 4,
            },
            enkf: EnkfReport {
                current_rmse: 0.2,
                speed_rmse: 0.3,
                baseline_current_rmse: 1.0,
                trace: vec![EnkfTracePoint {
                    t: 0.0,
                    true_current: 0.0,
                    estimated_current: 0.0,
                    true_speed: 0.0,
                    estimated_speed: 0.0,
                }],
            },
            diffusion: DiffusionReport {
                data_mean: 0.0,
                data_std: 1.0,
                sample_mean: 0.0,
                sample_std: 1.0,
                final_loss: 0.1,
                near_negative_mode: 0.5,
                histogram: vec![HistogramBin {
                    center: 0.0,
                    target_density: 0.5,
                    learned_density: 0.5,
                }],
            },
        }
    }

    #[test]
    fn report_html_escapes_run_output() {
        let html = render_report_html(&sample_data(), "</script><b>x");
        assert!(html.contains("&lt;/script&gt;&lt;b&gt;x"));
    }
}
