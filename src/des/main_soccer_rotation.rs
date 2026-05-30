//! Port of `src/des/main-soccer-rotation.ts`.
//!
//! 7v7 youth-soccer player rotation solved every which way — random,
//! per-period Hungarian, multi-period LP relaxation, time-boxed IP/MIP
//! branch-and-cut, and exact MDP backward induction — with the match outcome
//! simulated by the DES engine and (optionally) animated as a pitch + bench
//! scene plus an IP/MIP solver-entity scene.
//!
//! Conversion notes:
//!   - `process.argv` / `process.env` → `std::env`; `Date.now()` timing →
//!     `std::time::Instant`.
//!   - the layered solvers, evaluation, and match DES all delegate to
//!     `general::soccer_rotation`; the LP-relaxation backend selector maps to
//!     `general::ip_mip_des::LpRelaxationAlgorithm`.
//!   - both animation scenes (`soccer_scene`, `soccer_ipmip_solver_scene`) are
//!     ported, so the render path is wired (not stubbed).

use std::time::Instant;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::scenes::soccer_ipmip_solver_scene as solver_scene;
use crate::des::animation::scenes::soccer_scene as scene;
use crate::des::general::ip_mip_des::{ConcreteLpRelaxationAlgorithm, LpRelaxationAlgorithm};
use crate::des::general::soccer_rotation::{
    build_sample_soccer_problem, evaluate_schedule, evaluate_soccer_pomdp_features,
    policy_greedy_hungarian, policy_ipmip_feasible, policy_lp_relaxed, policy_mdp_vi,
    policy_mdp_vi_memoryless, policy_random_schedule, run_many_matches, simulate_match_des,
    validate_schedule_structure, welch_t, AffinityBuilderOptions, GoalSide, GreedyHungarianOptions,
    MatchAggregate, MatchSimOptions, Schedule, SoccerIPMIPPolicyOptions, SoccerIPMIPPolicyResult,
    SoccerPOMDPFeatureOptions, SoccerProblem,
};

// -----------------------------------------------------------------------------
// Small formatting helpers.
// -----------------------------------------------------------------------------

/// `Number.prototype.toExponential(digits)` (signed exponent, no leading zeros).
fn to_exponential(x: f64, digits: usize) -> String {
    if x.is_nan() {
        return "NaN".to_string();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-Infinity" } else { "Infinity" }.to_string();
    }
    let s = format!("{:.*e}", digits, x);
    let (mant, exp) = s.split_once('e').unwrap_or((s.as_str(), "0"));
    let exp_num: i32 = exp.parse().unwrap_or(0);
    let sign = if exp_num < 0 { '-' } else { '+' };
    format!("{}e{}{}", mant, sign, exp_num.abs())
}

/// Convert a `general::ip_mip_des::IPMIPSolution` to the subset mirror the
/// solver scene reads (`soccer_ipmip_solver_scene::IPMIPSolution`).
fn to_scene_solution(
    mip: &crate::des::general::ip_mip_des::IPMIPSolution,
) -> solver_scene::IPMIPSolution {
    use crate::des::general::ip_mip_des::TraceAction;
    let action_str = |a: TraceAction| -> String {
        match a {
            TraceAction::Branch => "branch",
            TraceAction::Cut => "cut",
            TraceAction::Prune => "prune",
            TraceAction::Incumbent => "incumbent",
            TraceAction::Unbounded => "unbounded",
        }
        .to_string()
    };
    let trace = mip
        .trace
        .iter()
        .map(|e| solver_scene::IPMIPTraceEvent {
            node_id: e.node_id.to_string(),
            depth: e.depth as f64,
            action: action_str(e.action),
            lp_z: e.lp_z,
            fractional: e.fractional.iter().map(|&v| v as f64).collect(),
            reason: e.reason.clone(),
        })
        .collect();
    let mut usage: Vec<(String, f64)> =
        mip.lp_algorithm_usage.iter().map(|(k, v)| (k.as_str().to_string(), *v as f64)).collect();
    usage.sort_by(|a, b| a.0.cmp(&b.0));
    solver_scene::IPMIPSolution {
        trace,
        status: mip.status.as_str().to_string(),
        z: mip.z,
        gap: mip.gap,
        best_bound: mip.best_bound,
        lp_algorithm: mip.lp_algorithm.as_str().to_string(),
        lp_solves: mip.lp_solves as f64,
        elapsed_ms: mip.elapsed_ms,
        nodes_explored: mip.nodes_explored as f64,
        cuts_added: mip.cuts_added as f64,
        candidates_tried: mip.candidates_tried as f64,
        lp_algorithm_usage: usage,
        incumbent_source: mip.incumbent_source.clone(),
    }
}

/// Integer-valued floats without a trailing `.0` (mirrors JS `${number}`).
fn jn(x: f64) -> String {
    if x.is_finite() && x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

fn env_f64(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// `process.env.MIP_LP_ALGO` (a `LPRelaxationAlgorithm` string) → enum.
fn parse_lp_algorithm(s: &str) -> LpRelaxationAlgorithm {
    use ConcreteLpRelaxationAlgorithm as C;
    match s {
        "auto" => LpRelaxationAlgorithm::Auto,
        "incremental-primal-dual" => LpRelaxationAlgorithm::Concrete(C::IncrementalPrimalDual),
        "des-simplex-dantzig" => LpRelaxationAlgorithm::Concrete(C::DesSimplexDantzig),
        "des-simplex-bland" => LpRelaxationAlgorithm::Concrete(C::DesSimplexBland),
        "external-highs" => LpRelaxationAlgorithm::Concrete(C::ExternalHighs),
        "external-highs-ds" => LpRelaxationAlgorithm::Concrete(C::ExternalHighsDs),
        "external-highs-ipm" => LpRelaxationAlgorithm::Concrete(C::ExternalHighsIpm),
        // "internal-simplex" and any unknown spelling default to the internal simplex.
        _ => LpRelaxationAlgorithm::Concrete(C::InternalSimplex),
    }
}

/// `name.replace(/[^a-z0-9._-]+/gi, '-').replace(/^-+|-+$/g, '')`.
fn safe_policy_name(name: &str) -> String {
    let mut out = String::new();
    let mut prev_dash = false;
    for ch in name.chars() {
        if ch.is_ascii_alphanumeric() || ch == '.' || ch == '_' || ch == '-' {
            out.push(ch);
            prev_dash = false;
        } else if !prev_dash {
            out.push('-');
            prev_dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

struct PolicyDesc {
    name: &'static str,
    note: &'static str,
}

/// Build a schedule for `name` (mirrors the per-policy `build` closures). The
/// IP/MIP build records its full result in `mip_latest`.
fn build_schedule(
    name: &str,
    problem: &SoccerProblem,
    seed: u32,
    mip_opts: &SoccerIPMIPPolicyOptions,
    mip_latest: &mut Option<SoccerIPMIPPolicyResult>,
) -> Schedule {
    match name {
        "random" => policy_random_schedule(problem, seed + 1),
        "MDP-memoryless" => policy_mdp_vi_memoryless(problem).schedule,
        "greedy-Hungarian" => {
            policy_greedy_hungarian(problem, &GreedyHungarianOptions { fairness_aware: Some(true) })
        }
        "LP-relaxation" => policy_lp_relaxed(problem).schedule,
        "IP/MIP-feasible" => {
            let r = policy_ipmip_feasible(problem, mip_opts);
            let schedule = r.schedule.clone();
            *mip_latest = Some(r);
            schedule
        }
        "MDP-VI-exact" => policy_mdp_vi(problem).to_schedule(),
        _ => unreachable!("unknown policy {name}"),
    }
}

/// Entry point (`main()` in the TS source).
pub fn run() {
    let seed = env_usize("SEED", 4242) as u32;
    let num_matches = env_usize("N_MATCHES", 100);
    let animate = std::env::var("ANIMATE").as_deref() == Ok("1");
    let policy_filter = std::env::var("POLICY").unwrap_or_default().to_lowercase().trim().to_string();
    let mip_time_limit_ms = env_f64("MIP_TIME_LIMIT_MS", 30_000.0);
    let mip_max_nodes = env_usize("MIP_MAX_NODES", 5_000);
    let mip_lp_algorithm =
        parse_lp_algorithm(&std::env::var("MIP_LP_ALGO").unwrap_or_else(|_| "internal-simplex".to_string()));

    let problem = build_sample_soccer_problem(&AffinityBuilderOptions { seed: Some(seed), ..Default::default() });
    let mip_opts = SoccerIPMIPPolicyOptions {
        time_limit_ms: Some(mip_time_limit_ms),
        max_nodes: Some(mip_max_nodes),
        max_ticks: Some(100.max(mip_max_nodes * 8)),
        lp_algorithm: Some(mip_lp_algorithm),
        ..Default::default()
    };
    let mut mip_latest: Option<SoccerIPMIPPolicyResult> = None;

    // ─── Banner ───────────────────────────────────────────────────────────
    println!("# 7v7 youth soccer player rotation as combinatorial optimisation");
    println!(
        "# {} players, {} positions, {} periods of 20 min each",
        problem.num_players, problem.num_positions, problem.num_periods
    );
    println!("# fairness constraint: no player benched two consecutive periods");
    println!("# affinity tensor seed = {}", seed);
    println!();
    println!("# Per-player best position and period peak (sample of affinity tensor):");
    for p in 0..problem.num_players {
        let mut best_pos = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        let mut best_t = 0usize;
        for pos in 0..problem.num_positions {
            for t in 0..problem.num_periods {
                if problem.affinity[p][pos][t] > best_val {
                    best_val = problem.affinity[p][pos][t];
                    best_pos = pos;
                    best_t = t;
                }
            }
        }
        let pos_name = problem
            .position_names
            .as_ref()
            .and_then(|n| n.get(best_pos))
            .cloned()
            .unwrap_or_else(|| best_pos.to_string());
        let player_name = problem
            .player_names
            .as_ref()
            .and_then(|n| n.get(p))
            .cloned()
            .unwrap_or_else(|| format!("P{p}"));
        println!(
            "#   {}  best @ pos {}, period {}, affinity {:.2}",
            player_name,
            pos_name,
            best_t + 1,
            best_val
        );
    }
    println!();

    // ─── Policies ───────────────────────────────────────────────────────────
    let policies = [PolicyDesc { name: "random", note: "L3: trivial baseline, ignores fairness" },
        PolicyDesc {
            name: "MDP-memoryless",
            note: "L2 with state=(t,) — Markov but NO history → cannot express fairness",
        },
        PolicyDesc {
            name: "greedy-Hungarian",
            note: "L3: per-period bipartite assignment with fairness pre-fill",
        },
        PolicyDesc {
            name: "LP-relaxation",
            note: "L3: simplex / interior-point on the LP relaxation, Hungarian-rounded",
        },
        PolicyDesc {
            name: "IP/MIP-feasible",
            note: "L3: exact 0/1 IP/MIP branch-and-cut DES; time-boxed, returns best feasible incumbent",
        },
        PolicyDesc {
            name: "MDP-VI-exact",
            note: "L2 with state=(t, prev-bench) — Markov + 1-period memory ⇒ fairness",
        }];
    let filtered: Vec<&PolicyDesc> = if policy_filter.is_empty() {
        policies.iter().collect()
    } else {
        policies.iter().filter(|p| p.name.to_lowercase().contains(&policy_filter)).collect()
    };

    // LP upper bound for context.
    let lp = policy_lp_relaxed(&problem);
    println!(
        "# LP relaxation upper bound on total deterministic affinity = {:.4}",
        lp.lp_value
    );
    println!("# (solved via {}, {} iterations)", lp.solver, lp.iters);
    println!();

    let mut aggs: Vec<MatchAggregate> = Vec::new();
    for p in &filtered {
        print!("# Building schedule '{}' ... ", p.name);
        let t0 = Instant::now();
        let schedule = build_schedule(p.name, &problem, seed, &mip_opts, &mut mip_latest);
        let build_ms = t0.elapsed().as_millis();
        if let Some(err) = validate_schedule_structure(&problem, &schedule) {
            println!("FAIL: {}", err);
            continue;
        }
        let eval_res = evaluate_schedule(&problem, &schedule);
        let belief = evaluate_soccer_pomdp_features(&problem, &schedule, &SoccerPOMDPFeatureOptions::default());
        println!(
            "affinity={:.2}, fairness={}, beliefFresh={:.3}, build={}ms",
            eval_res.affinity_sum,
            if eval_res.fairness_ok { "OK" } else { "VIOLATED" },
            belief.mean_expected_fresh_on_field,
            build_ms
        );
        if p.name == "IP/MIP-feasible" {
            if let Some(latest) = &mip_latest {
                // PORT NOTE: `lpAlgorithmUsage` is a `HashMap` in the Rust model
                // (no insertion order), so the comma-joined pairs are emitted in a
                // stable key-sorted order rather than `Object.entries` order.
                let usage_str = if latest.mip.lp_algorithm_usage.is_empty() {
                    "none".to_string()
                } else {
                    let mut entries: Vec<(String, u64)> = latest
                        .mip
                        .lp_algorithm_usage
                        .iter()
                        .map(|(k, v)| (k.as_str().to_string(), *v))
                        .collect();
                    entries.sort_by(|a, b| a.0.cmp(&b.0));
                    entries.iter().map(|(k, v)| format!("{}={}", k, v)).collect::<Vec<_>>().join(",")
                };
                let fallback = if latest.used_fallback {
                    format!(", fallback={}", latest.fallback_reason.as_deref().unwrap_or(""))
                } else {
                    String::new()
                };
                println!(
                    "#   IP/MIP status={}, gap={}, nodes={}, lpSolves={}, lpUsage={}, elapsed={}ms, incumbent={}{}",
                    latest.mip.status.as_str(),
                    to_exponential(latest.mip.gap, 2),
                    latest.mip.nodes_explored,
                    latest.mip.lp_solves,
                    usage_str,
                    jn(latest.mip.elapsed_ms),
                    latest.mip.incumbent_source.as_deref().unwrap_or("none"),
                    fallback
                );
            }
        }
        print!("#   simulating {} matches ... ", num_matches);
        let t_sim = Instant::now();
        let agg = run_many_matches(
            &problem,
            &schedule,
            p.name,
            num_matches,
            seed + 1000,
            &MatchSimOptions::default(),
        );
        println!("done in {}ms", t_sim.elapsed().as_millis());
        aggs.push(agg);
    }
    println!();

    // ─── Comparison table ────────────────────────────────────────────────
    let rule = "─".repeat(94);
    println!("# {}", rule);
    println!("# Policy comparison: deterministic affinity (offline) + simulated match outcome (DES)");
    println!("# {}", rule);
    println!(
        "#   {}{}{}{}{}{}{}{}",
        format!("{:<20}", "policy"),
        format!("{:>11}", "affinity"),
        format!("{:>11}", "goal diff"),
        format!("{:>8}", "sd"),
        format!("{:>7}", "gF"),
        format!("{:>7}", "gA"),
        format!("{:>11}", "fairness"),
        format!("{:>13}", "t vs random"),
    );
    let random = aggs.iter().find(|a| a.policy_name == "random");
    for a in &aggs {
        let t = match random {
            Some(r) => welch_t(&a.raw_goal_diffs, &r.raw_goal_diffs),
            None => f64::NAN,
        };
        let tstr = if t.is_nan() { "   —".to_string() } else { format!("{:.2}", t) };
        println!(
            "#   {}{}{}{}{}{}{}{}",
            format!("{:<20}", a.policy_name),
            format!("{:>11}", format!("{:.3}", a.affinity_sum_deterministic)),
            format!("{:>11}", format!("{:.3}", a.mean_goal_diff)),
            format!("{:>8}", format!("{:.3}", a.sd_goal_diff)),
            format!("{:>7}", format!("{:.2}", a.mean_goals_for)),
            format!("{:>7}", format!("{:.2}", a.mean_goals_against)),
            format!("{:>11}", if a.fairness_ok { "OK" } else { "VIOLATED" }),
            format!("{:>13}", tstr),
        );
    }
    println!();

    // ─── Player-fairness audit ──────────────────────────────────────────
    println!("# Per-player periods on bench (out of {}):", problem.num_periods);
    for a in &aggs {
        let counts = a
            .bench_counts
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let name = problem
                    .player_names
                    .as_ref()
                    .and_then(|n| n.get(i))
                    .cloned()
                    .unwrap_or_else(|| format!("P{i}"));
                format!("{}={}", name, c)
            })
            .collect::<Vec<_>>()
            .join(" ");
        let flag = if a.fairness_ok { "" } else { "  ← VIOLATES fairness" };
        println!("#   {}  {}{}", format!("{:<20}", a.policy_name), counts, flag);
    }
    println!();

    // ─── Architectural recap ────────────────────────────────────────────
    println!("# Architectural recap:");
    for p in &filtered {
        println!("#   {} → {}", format!("{:<20}", p.name), p.note);
    }
    println!("#");
    println!("#   Layer 1 (DES): simulateMatchDES runs 80 game-minutes per match,");
    println!("#                  samples Poisson goal events from on-field affinity,");
    println!("#                  triggers a substitution event at every period boundary.");
    println!("#   Layer 2 (MDP): exact backward induction; |S| = 4 periods × C(12,5) = 3168,");
    println!("#                  reward at each (s, a) is the Hungarian-optimal assignment.");
    println!("#   POMDP feature: hidden fatigue belief is carried across periods for audit metrics.");
    println!("#   Layer 3:       random / greedy-Hungarian / LP-relaxation / IP-MIP / MDP-VI.");
    println!();

    // ─── Optional animation of the best policy ──────────────────────────
    if animate {
        let best = aggs
            .iter()
            .fold(aggs[0].clone(), |acc, x| if x.mean_goal_diff > acc.mean_goal_diff { x.clone() } else { acc });
        println!(
            "# Animating policy '{}' (best mean goal diff = {:.3})",
            best.policy_name, best.mean_goal_diff
        );
        let out_dir = std::path::Path::new("out");
        let _ = std::fs::create_dir_all(out_dir);
        let safe = safe_policy_name(&best.policy_name);
        let frames_path = out_dir.join(format!("soccer-{}.frames.jsonl", safe));
        let html_path = out_dir.join(format!("soccer-{}.html", safe));
        let mut rec = FrameRecorder::new(FrameRecorderOpts {
            frames_path: frames_path.to_string_lossy().into_owned(),
            html_path: Some(html_path.to_string_lossy().into_owned()),
            width: scene::STAGE_W,
            height: scene::STAGE_H,
            fps: Some(6.0),
            title: Some(format!("7v7 — {}", best.policy_name)),
            subtitle: Some(format!(
                "affinity {:.2}, goal diff {:.2}",
                best.affinity_sum_deterministic, best.mean_goal_diff
            )),
            background: Some("#0b1220".to_string()),
            ..Default::default()
        })
        .expect("create frame recorder");

        let scene_problem = scene::SoccerProblem {
            num_positions: problem.num_positions,
            num_periods: problem.num_periods as f64,
            player_names: problem.player_names.clone(),
            position_names: problem.position_names.clone(),
        };
        let m = simulate_match_des(
            &problem,
            &best.schedule,
            &MatchSimOptions { seed: Some(seed + 1000), ..Default::default() },
        );
        let mut ts: Vec<f64> = Vec::new();
        let mut affs: Vec<f64> = Vec::new();
        let mut g_fs: Vec<f64> = Vec::new();
        let mut g_as: Vec<f64> = Vec::new();
        for (i, tr) in m.trace.iter().enumerate() {
            let mut goal_this_tick: Option<scene::GoalSide> = None;
            for ev in &m.goal_events {
                if ev.t == tr.t {
                    goal_this_tick = Some(match ev.side {
                        GoalSide::Us => scene::GoalSide::Us,
                        GoalSide::Them => scene::GoalSide::Them,
                    });
                    break;
                }
            }
            ts.push(tr.t as f64);
            affs.push(tr.affinity_now);
            g_fs.push(tr.goals_for_cum as f64);
            g_as.push(tr.goals_against_cum as f64);
            let positions: Vec<usize> = tr.positions.iter().map(|&x| x as usize).collect();
            let bench: Vec<usize> = tr.bench.clone();
            let t_f = tr.t as f64;
            let i_f = i as f64;
            let period_f = tr.period as f64;
            let gf = tr.goals_for_cum as f64;
            let ga = tr.goals_against_cum as f64;
            let aff = tr.affinity_now;
            let sp = &scene_problem;
            rec.frame(t_f, i_f, || {
                scene::build_soccer_frame(
                    t_f,
                    i_f,
                    &scene::SoccerFrameInput {
                        t: t_f,
                        period: period_f,
                        positions,
                        bench,
                        goals_for: gf,
                        goals_against: ga,
                        affinity_now: aff,
                        goal_this_tick,
                        problem: sp,
                    },
                )
            });
        }
        rec.set_charts(scene::build_soccer_charts(&ts, &affs, &g_fs, &g_as));
        rec.finish().expect("finish recorder");
        println!("# Animation written to {}", html_path.display());

        if let Some(latest) = &mip_latest {
            let solver_solution = to_scene_solution(&latest.mip);
            let solver_frames_path = out_dir.join("soccer-IP-MIP-feasible-solver.frames.jsonl");
            let solver_html_path = out_dir.join("soccer-IP-MIP-feasible-solver.html");
            let mut solver_rec = FrameRecorder::new(FrameRecorderOpts {
                frames_path: solver_frames_path.to_string_lossy().into_owned(),
                html_path: Some(solver_html_path.to_string_lossy().into_owned()),
                width: solver_scene::SOCCER_IPMIP_SOLVER_W,
                height: solver_scene::SOCCER_IPMIP_SOLVER_H,
                fps: Some(5.0),
                title: Some("7v7 IP/MIP Solver Entities".to_string()),
                subtitle: Some(format!(
                    "status {}, LP {}, nodes {}",
                    latest.mip.status.as_str(),
                    latest.mip.lp_algorithm.as_str(),
                    latest.mip.nodes_explored
                )),
                background: Some("#f8fafc".to_string()),
                ..Default::default()
            })
            .expect("create solver frame recorder");
            let total = solver_scene::soccer_ipmip_solver_frame_count(&solver_solution);
            for i in 0..total {
                let i_f = i as f64;
                solver_rec.frame(i_f, i_f, || {
                    solver_scene::build_soccer_ipmip_solver_frame(&solver_solution, i)
                });
            }
            solver_rec.set_charts(solver_scene::build_soccer_ipmip_solver_charts(&solver_solution));
            solver_rec.finish().expect("finish solver recorder");
            println!("# Solver entity animation written to {}", solver_html_path.display());
        }
    }
}
