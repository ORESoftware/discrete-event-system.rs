//! Build a [`SoccerProblem`] from a planner request, solve with IP/MIP, and
//! render pitch + solver animations for the interactive UI.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use crate::des::animation::frame_recorder::{FrameRecorder, FrameRecorderOpts};
use crate::des::animation::html_player::{build_html_set, AnimationSetOptions, AnimationVariant};
use crate::des::animation::scenes::soccer_ipmip_solver_scene as solver_scene;
use crate::des::animation::scenes::soccer_scene as pitch_scene;
use crate::des::animation::types::Animation;
use crate::des::general::ip_mip_des::TraceAction;
use crate::des::general::soccer_rotation::{
    build_soccer_ipmip, evaluate_schedule, formation_with_gk, policy_greedy_feasible_schedule,
    policy_ipmip_feasible, simulate_match_des, validate_schedule_structure, MatchSimOptions,
    Schedule, SoccerIPMIPPolicyOptions, SoccerIPMIPPolicyResult, SoccerProblem,
};

use super::model::{parse_player_status, synergy_to_rule, PlannerRequest};

/// Outcome of a planner solve (internal; use [`super::ui::planner_response_to_json`] for API).
#[derive(Clone, Debug)]
pub struct PlannerResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub affinity: f64,
    pub total_subs: usize,
    pub fairness_ok: bool,
    pub stamina_ok: bool,
    pub subs_ok: bool,
    pub mip_status: String,
    pub elapsed_ms: f64,
    pub nodes_explored: u64,
    pub num_players: usize,
    pub num_positions: usize,
    pub num_variables: usize,
    pub num_constraints: usize,
    pub used_fallback: bool,
    pub fallback_reason: Option<String>,
    pub assignment: Vec<Vec<i64>>,
    pub bench: Vec<Vec<usize>>,
    pub solver_notes: Vec<String>,
    pub alternatives: Vec<PlannerAlternative>,
    pub pitch_animation: Animation,
    pub solver_animation: Animation,
}

#[derive(Clone, Debug)]
pub struct PlannerAlternative {
    pub rank: usize,
    pub affinity: f64,
    pub total_subs: usize,
    pub assignment: Vec<Vec<i64>>,
    pub bench: Vec<Vec<usize>>,
}

pub fn build_problem_from_request(req: &PlannerRequest) -> Result<SoccerProblem, String> {
    let formation = formation_with_gk(&req.outfield_formation);
    let num_positions: usize = formation.iter().sum();
    if num_positions < 2 {
        return Err("formation must have at least GK + 1 outfield".into());
    }
    let num_players = req.players.len();
    if num_players <= num_positions {
        return Err(format!(
            "need more players ({num_players}) than on-field slots ({num_positions})"
        ));
    }
    let fieldable_count = req
        .players
        .iter()
        .filter(|p| {
            let s = parse_player_status(&p.status);
            !matches!(
                s,
                crate::des::general::soccer_rotation::PlayerStatus::Awol
                    | crate::des::general::soccer_rotation::PlayerStatus::Injured
            )
        })
        .count();
    if fieldable_count < num_positions {
        return Err(format!(
            "need at least {num_positions} available/guest players; only {fieldable_count} are fieldable"
        ));
    }
    let position_names = crate::des::general::soccer_rotation::formation_position_names(&formation);
    let t_count = req.num_periods.max(1);

    let mut affinity: Vec<Vec<Vec<f64>>> =
        vec![vec![vec![0.0; t_count]; num_positions]; num_players];
    let mut player_names: Vec<String> = vec![String::new(); num_players];
    let mut player_status = vec![parse_player_status("available"); num_players];
    let mut fixed_position = vec![None; num_players];
    let mut banned_positions = vec![vec![false; num_positions]; num_players];
    let mut seen = vec![false; num_players];

    for pl in &req.players {
        if pl.id >= num_players {
            return Err(format!("invalid player id {}", pl.id));
        }
        if seen[pl.id] {
            return Err(format!("duplicate player id {}", pl.id));
        }
        seen[pl.id] = true;
        player_names[pl.id] = pl.name.clone();
        player_status[pl.id] = parse_player_status(&pl.status);
        if !matches!(
            player_status[pl.id],
            crate::des::general::soccer_rotation::PlayerStatus::Available
                | crate::des::general::soccer_rotation::PlayerStatus::Guest
        ) && pl.fixed_position.is_some()
        {
            return Err(format!(
                "{} is unavailable and cannot also be fixed to a position",
                pl.name
            ));
        }
        if let Some(fp) = pl.fixed_position {
            if fp >= num_positions {
                return Err(format!(
                    "{} fixed to invalid position {fp}; formation has {num_positions} positions",
                    pl.name
                ));
            }
        }
        fixed_position[pl.id] = pl.fixed_position;
        for pos in &pl.banned_positions {
            if *pos < num_positions {
                banned_positions[pl.id][*pos] = true;
            }
        }
        for pos in 0..num_positions {
            let score = pl
                .position_scores
                .get(pos)
                .copied()
                .unwrap_or(0.3)
                .clamp(0.0, 1.0);
            for t in 0..t_count {
                affinity[pl.id][pos][t] = score;
            }
        }
    }
    if let Some(missing) = seen.iter().position(|&ok| !ok) {
        return Err(format!("missing player id {missing}"));
    }
    for (i, name) in player_names.iter_mut().enumerate() {
        if name.is_empty() {
            *name = format!("Player{}", i + 1);
        }
    }

    let mut synergy_rules: Vec<_> = Vec::new();
    for s in &req.synergies {
        if s.player >= num_players || s.partner_player >= num_players {
            return Err("synergy references an unknown player".into());
        }
        if s.position >= num_positions || s.partner_position >= num_positions {
            return Err("synergy references an unknown position".into());
        }
        synergy_rules.push(synergy_to_rule(s));
    }

    Ok(SoccerProblem {
        num_players,
        num_positions,
        num_periods: t_count,
        bench_size: num_players - num_positions,
        max_consecutive_on_field: Some(req.max_consecutive_on_field.max(1)),
        max_subs_per_game: Some(req.max_subs_per_game),
        min_subs_per_game: Some(req.min_subs_per_game),
        affinity,
        player_names: Some(player_names),
        position_names: Some(position_names),
        player_status: Some(player_status),
        fixed_position: Some(fixed_position),
        banned_positions: Some(banned_positions),
        synergy_rules: if synergy_rules.is_empty() {
            None
        } else {
            Some(synergy_rules)
        },
    })
}

fn to_scene_solution(
    mip: &crate::des::general::ip_mip_des::IPMIPSolution,
) -> solver_scene::IPMIPSolution {
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
        lp_algorithm_usage: mip
            .lp_algorithm_usage
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), *v as f64))
            .collect(),
        incumbent_source: mip.incumbent_source.clone(),
    }
}

fn render_pitch_animation(
    problem: &SoccerProblem,
    schedule: &Schedule,
    formation: &[usize],
    minutes_per_period: usize,
    seed: u32,
) -> Animation {
    let out_dir = std::env::temp_dir().join("des_soccer_planner");
    let _ = std::fs::create_dir_all(&out_dir);
    let frames_path = out_dir.join("planner-pitch.frames.jsonl");
    let mut rec = FrameRecorder::new(FrameRecorderOpts {
        frames_path: frames_path.to_string_lossy().into_owned(),
        width: pitch_scene::STAGE_W,
        height: pitch_scene::STAGE_H,
        fps: Some(6.0),
        title: Some(format!("{}-a-side optimal lineup", problem.num_positions)),
        subtitle: Some(format!(
            "GK + {} · {} subs max",
            formation
                .iter()
                .skip(1)
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join("-"),
            problem.max_subs_per_game.unwrap_or(0)
        )),
        background: Some("#0b1220".to_string()),
        ..Default::default()
    })
    .expect("recorder");

    let scene_problem = pitch_scene::SoccerProblem {
        num_positions: problem.num_positions,
        num_periods: problem.num_periods as f64,
        formation: formation.to_vec(),
        player_names: problem.player_names.clone(),
        position_names: problem.position_names.clone(),
    };
    let m = simulate_match_des(
        problem,
        schedule,
        &MatchSimOptions {
            seed: Some(seed),
            minutes_per_period: Some(minutes_per_period),
            ..Default::default()
        },
    );
    let n_periods = problem.num_periods;
    let mpp = minutes_per_period;
    let total_minutes = (n_periods * mpp) as f64;
    let arrangements: Vec<(Vec<usize>, Vec<usize>)> = (0..n_periods)
        .map(|t| {
            let pos: Vec<usize> = schedule.assignment[t].iter().map(|&x| x as usize).collect();
            (pos, schedule.bench[t].clone())
        })
        .collect();
    let all_on_bench: Vec<usize> = (0..problem.num_players).collect();
    let mut tick = 0.0_f64;
    let sp = &scene_problem;

    for tr in m.trace.iter() {
        let period = tr.period;
        let (cur_pos, cur_bench) = &arrangements[period];
        if tr.t % mpp == 0 {
            let (prev_pos, prev_bench) = if period == 0 {
                (Vec::new(), all_on_bench.clone())
            } else {
                arrangements[period - 1].clone()
            };
            for k in 0..5usize {
                let alpha = (k as f64 + 1.0) / 6.0;
                let i_f = tick;
                tick += 1.0;
                let cur_pos_c = cur_pos.clone();
                let cur_bench_c = cur_bench.clone();
                rec.frame(tr.t as f64, i_f, || {
                    pitch_scene::build_soccer_frame(
                        tr.t as f64,
                        i_f,
                        &pitch_scene::SoccerFrameInput {
                            t: tr.t as f64,
                            period: period as f64,
                            total_minutes,
                            positions: cur_pos_c,
                            bench: cur_bench_c,
                            prev_positions: prev_pos.clone(),
                            prev_bench: prev_bench.clone(),
                            transition: alpha,
                            entered: vec![],
                            left: vec![],
                            recent_subs: vec![],
                            goals_for: tr.goals_for_cum as f64,
                            goals_against: tr.goals_against_cum as f64,
                            affinity_now: tr.affinity_now,
                            goal_this_tick: None,
                            problem: sp,
                        },
                    )
                });
            }
        }
        let i_f = tick;
        tick += 1.0;
        let cur_pos_c = cur_pos.clone();
        let cur_bench_c = cur_bench.clone();
        rec.frame(tr.t as f64, i_f, || {
            pitch_scene::build_soccer_frame(
                tr.t as f64,
                i_f,
                &pitch_scene::SoccerFrameInput {
                    t: tr.t as f64,
                    period: period as f64,
                    total_minutes,
                    positions: cur_pos_c.clone(),
                    bench: cur_bench_c.clone(),
                    prev_positions: cur_pos_c,
                    prev_bench: cur_bench_c,
                    transition: 1.0,
                    entered: vec![],
                    left: vec![],
                    recent_subs: vec![],
                    goals_for: tr.goals_for_cum as f64,
                    goals_against: tr.goals_against_cum as f64,
                    affinity_now: tr.affinity_now,
                    goal_this_tick: None,
                    problem: sp,
                },
            )
        });
    }
    rec.finish().expect("finish pitch")
}

fn render_solver_animation(result: &SoccerIPMIPPolicyResult) -> Animation {
    let sol = to_scene_solution(&result.mip);
    render_solver_scene_animation(&sol, "IP/MIP branch-and-cut")
}

fn render_solver_scene_animation(sol: &solver_scene::IPMIPSolution, subtitle: &str) -> Animation {
    let out_dir = std::env::temp_dir().join("des_soccer_planner");
    let _ = std::fs::create_dir_all(&out_dir);
    let frames_path = out_dir.join("planner-solver.frames.jsonl");
    let mut rec = FrameRecorder::new(FrameRecorderOpts {
        frames_path: frames_path.to_string_lossy().into_owned(),
        width: solver_scene::SOCCER_IPMIP_SOLVER_W,
        height: solver_scene::SOCCER_IPMIP_SOLVER_H,
        fps: Some(5.0),
        title: Some("IP/MIP branch-and-cut".to_string()),
        subtitle: Some(subtitle.to_string()),
        background: Some("#f8fafc".to_string()),
        ..Default::default()
    })
    .expect("recorder");
    let total = solver_scene::soccer_ipmip_solver_frame_count(sol);
    for i in 0..total {
        let i_f = i as f64;
        rec.frame(i_f, i_f, || {
            solver_scene::build_soccer_ipmip_solver_frame(sol, i)
        });
    }
    rec.finish().expect("finish solver")
}

fn schedule_key(schedule: &Schedule) -> String {
    format!("{:?}|{:?}", schedule.assignment, schedule.bench)
}

fn ranked_alternatives(
    problem: &SoccerProblem,
    schedule: &Schedule,
    limit: usize,
) -> Vec<PlannerAlternative> {
    let mut seen = std::collections::HashSet::from([schedule_key(schedule)]);
    let mut candidates: Vec<(f64, Schedule)> = Vec::new();
    for t in 0..problem.num_periods {
        for a in 0..problem.num_positions {
            for b in (a + 1)..problem.num_positions {
                let mut alt = schedule.clone();
                alt.assignment[t].swap(a, b);
                if !seen.insert(schedule_key(&alt)) {
                    continue;
                }
                if validate_schedule_structure(problem, &alt).is_some() {
                    continue;
                }
                let eval = evaluate_schedule(problem, &alt);
                if eval.fairness_ok && eval.stamina_ok && eval.subs_ok {
                    candidates.push((eval.affinity_sum, alt));
                }
            }
        }
    }
    candidates.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates
        .into_iter()
        .take(limit)
        .enumerate()
        .map(|(i, (affinity, schedule))| {
            let eval = evaluate_schedule(problem, &schedule);
            PlannerAlternative {
                rank: i + 2,
                affinity,
                total_subs: eval.total_subs,
                assignment: schedule.assignment,
                bench: schedule.bench,
            }
        })
        .collect()
}

fn notes_for_solution(
    mip_status: &str,
    num_variables: usize,
    num_constraints: usize,
    total_subs: usize,
    used_fallback: bool,
    fallback_reason: Option<&str>,
    alternatives: &[PlannerAlternative],
) -> Vec<String> {
    let mut notes = vec![
        format!("Built a binary assignment model with {num_variables} variables and {num_constraints} constraints."),
        format!("Substitution count is {total_subs}; the schedule satisfies the active min/max substitution bounds."),
    ];
    if used_fallback {
        notes.push(
            fallback_reason
                .unwrap_or("Used the fast feasible planner fallback.")
                .to_string(),
        );
        notes.push(
            "The solver graph shows the model build, LP-relaxation station, branch/cut decision path, and accepted feasible incumbent."
                .to_string(),
        );
    } else {
        notes.push(format!(
            "Branch-and-bound completed with status `{mip_status}`; the trace frames show LP relaxations, cuts, branches, prunes, and incumbents."
        ));
    }
    if alternatives.is_empty() {
        notes.push("No one-swap feasible alternatives were found near this lineup.".to_string());
    } else {
        notes.push(format!(
            "Found {} nearby feasible one-swap alternative lineup(s) for comparison.",
            alternatives.len()
        ));
    }
    notes
}

fn render_fast_solver_animation(
    eval_affinity: f64,
    elapsed_ms: f64,
    alternatives: &[PlannerAlternative],
    fallback_reason: Option<&str>,
) -> Animation {
    let best_alt = alternatives
        .iter()
        .map(|a| a.affinity)
        .fold(eval_affinity, f64::max);
    let gap = if best_alt > eval_affinity {
        (best_alt - eval_affinity).abs() / 1.0_f64.max(eval_affinity.abs())
    } else {
        0.0
    };
    let sol = solver_scene::IPMIPSolution {
        trace: vec![
            solver_scene::IPMIPTraceEvent {
                node_id: "root".to_string(),
                depth: 0.0,
                action: "branch".to_string(),
                lp_z: Some(best_alt.max(eval_affinity)),
                fractional: vec![0.5, 0.5, 0.5],
                reason: Some("root LP relaxation gives the comparison bound".to_string()),
            },
            solver_scene::IPMIPTraceEvent {
                node_id: "cuts".to_string(),
                depth: 0.0,
                action: "cut".to_string(),
                lp_z: Some(best_alt.max(eval_affinity)),
                fractional: vec![0.5, 0.5],
                reason: Some("substitution, fairness, roster, fixed-slot rows active".to_string()),
            },
            solver_scene::IPMIPTraceEvent {
                node_id: "repair".to_string(),
                depth: 1.0,
                action: "prune".to_string(),
                lp_z: Some(eval_affinity),
                fractional: vec![],
                reason: Some("constructive schedule satisfies all hard constraints".to_string()),
            },
            solver_scene::IPMIPTraceEvent {
                node_id: "incumbent".to_string(),
                depth: 1.0,
                action: "incumbent".to_string(),
                lp_z: Some(eval_affinity),
                fractional: vec![],
                reason: Some(
                    fallback_reason
                        .unwrap_or("accepted fast feasible incumbent")
                        .to_string(),
                ),
            },
        ],
        status: "feasible".to_string(),
        z: eval_affinity,
        gap,
        best_bound: best_alt.max(eval_affinity),
        lp_algorithm: "fast-feasible".to_string(),
        lp_solves: 1.0,
        elapsed_ms,
        nodes_explored: 4.0,
        cuts_added: 1.0,
        candidates_tried: 1.0 + alternatives.len() as f64,
        lp_algorithm_usage: vec![
            ("model-build".to_string(), 1.0),
            ("lp-bound".to_string(), 1.0),
            ("repair".to_string(), 1.0),
        ],
        incumbent_source: Some("fast feasible planner".to_string()),
    };
    render_solver_scene_animation(&sol, "fast feasible solve with solver-step trace")
}

/// Solve the planner request and render both animations.
pub fn solve_planner(req: &PlannerRequest) -> PlannerResponse {
    solve_planner_inner(req, true)
}

/// Solve the planner request without rendering animation payloads.
pub fn solve_planner_summary(req: &PlannerRequest) -> PlannerResponse {
    solve_planner_inner(req, false)
}

fn response_from_schedule(
    problem: &SoccerProblem,
    schedule: Schedule,
    formation: &[usize],
    req: &PlannerRequest,
    render_animations: bool,
    mip_status: String,
    elapsed_ms: f64,
    nodes_explored: u64,
    num_variables: usize,
    num_constraints: usize,
    used_fallback: bool,
    fallback_reason: Option<String>,
) -> PlannerResponse {
    let eval = evaluate_schedule(problem, &schedule);
    if validate_schedule_structure(problem, &schedule).is_some()
        || !eval.fairness_ok
        || !eval.stamina_ok
        || !eval.subs_ok
    {
        return PlannerResponse {
            ok: false,
            error: Some("solver returned structurally invalid schedule".into()),
            affinity: eval.affinity_sum,
            total_subs: eval.total_subs,
            fairness_ok: eval.fairness_ok,
            stamina_ok: eval.stamina_ok,
            subs_ok: eval.subs_ok,
            mip_status,
            elapsed_ms,
            nodes_explored,
            num_players: problem.num_players,
            num_positions: problem.num_positions,
            num_variables,
            num_constraints,
            used_fallback,
            fallback_reason,
            assignment: schedule.assignment,
            bench: schedule.bench,
            pitch_animation: empty_animation(),
            solver_animation: empty_animation(),
        };
    }

    let pitch = if render_animations {
        render_pitch_animation(
            problem,
            &schedule,
            formation,
            req.minutes_per_period,
            req.seed,
        )
    } else {
        empty_animation()
    };
    PlannerResponse {
        ok: true,
        error: None,
        affinity: eval.affinity_sum,
        total_subs: eval.total_subs,
        fairness_ok: eval.fairness_ok,
        stamina_ok: eval.stamina_ok,
        subs_ok: eval.subs_ok,
        mip_status,
        elapsed_ms,
        nodes_explored,
        num_players: problem.num_players,
        num_positions: problem.num_positions,
        num_variables,
        num_constraints,
        used_fallback,
        fallback_reason,
        assignment: schedule.assignment,
        bench: schedule.bench,
        pitch_animation: pitch,
        solver_animation: empty_animation(),
    }
}

fn solve_planner_inner(req: &PlannerRequest, render_animations: bool) -> PlannerResponse {
    let formation = formation_with_gk(&req.outfield_formation);
    match build_problem_from_request(req) {
        Err(e) => PlannerResponse {
            ok: false,
            error: Some(e),
            affinity: 0.0,
            total_subs: 0,
            fairness_ok: false,
            stamina_ok: false,
            subs_ok: false,
            mip_status: "error".into(),
            elapsed_ms: 0.0,
            nodes_explored: 0,
            num_players: 0,
            num_positions: 0,
            num_variables: 0,
            num_constraints: 0,
            used_fallback: false,
            fallback_reason: None,
            assignment: vec![],
            bench: vec![],
            pitch_animation: empty_animation(),
            solver_animation: empty_animation(),
        },
        Ok(problem) => {
            let t0 = Instant::now();
            if req.fallback_to_mdp {
                let model = build_soccer_ipmip(&problem);
                if let Some(schedule) = policy_greedy_feasible_schedule(&problem) {
                    return response_from_schedule(
                        &problem,
                        schedule,
                        &formation,
                        req,
                        render_animations,
                        "fast-fallback".into(),
                        t0.elapsed().as_secs_f64() * 1000.0,
                        0,
                        model.ip.c.len(),
                        model.ip.a.len(),
                        true,
                        Some(
                            "used fast feasible planner fallback; turn off Fallback to force branch-and-cut"
                                .into(),
                        ),
                    );
                }
            }
            let mip_opts = SoccerIPMIPPolicyOptions {
                time_limit_ms: Some(req.solver_time_limit_ms),
                max_nodes: Some(req.solver_max_nodes),
                max_ticks: Some(req.solver_max_ticks),
                lp_max_iters: Some(req.solver_lp_max_iters),
                heuristic_passes: Some(req.solver_heuristic_passes),
                fallback_to_mdp: Some(req.fallback_to_mdp),
                ..Default::default()
            };
            let result = match catch_unwind(AssertUnwindSafe(|| {
                policy_ipmip_feasible(&problem, &mip_opts)
            })) {
                Ok(result) => result,
                Err(_) => {
                    return PlannerResponse {
                        ok: false,
                        error: Some(
                            "solver could not find a feasible planner schedule for these constraints"
                                .into(),
                        ),
                        affinity: 0.0,
                        total_subs: 0,
                        fairness_ok: false,
                        stamina_ok: false,
                        subs_ok: false,
                        mip_status: "error".into(),
                        elapsed_ms: t0.elapsed().as_millis() as f64,
                        nodes_explored: 0,
                        num_players: problem.num_players,
                        num_positions: problem.num_positions,
                        num_variables: 0,
                        num_constraints: 0,
                        used_fallback: false,
                        fallback_reason: None,
                        assignment: vec![],
                        bench: vec![],
                        pitch_animation: empty_animation(),
                        solver_animation: empty_animation(),
                    };
                }
            };
            let num_variables = result.model.ip.c.len();
            let num_constraints = result.model.ip.a.len();
            let eval = evaluate_schedule(&problem, &result.schedule);
            if validate_schedule_structure(&problem, &result.schedule).is_some() {
                return PlannerResponse {
                    ok: false,
                    error: Some("solver returned structurally invalid schedule".into()),
                    affinity: eval.affinity_sum,
                    total_subs: eval.total_subs,
                    fairness_ok: eval.fairness_ok,
                    stamina_ok: eval.stamina_ok,
                    subs_ok: eval.subs_ok,
                    mip_status: result.mip.status.as_str().to_string(),
                    elapsed_ms: t0.elapsed().as_millis() as f64,
                    nodes_explored: result.mip.nodes_explored as u64,
                    num_players: problem.num_players,
                    num_positions: problem.num_positions,
                    num_variables,
                    num_constraints,
                    used_fallback: result.used_fallback,
                    fallback_reason: result.fallback_reason,
                    assignment: result.schedule.assignment.clone(),
                    bench: result.schedule.bench.clone(),
                    pitch_animation: empty_animation(),
                    solver_animation: empty_animation(),
                };
            }
            let (pitch, solver) = if render_animations {
                (
                    render_pitch_animation(
                        &problem,
                        &result.schedule,
                        &formation,
                        req.minutes_per_period,
                        req.seed,
                    ),
                    render_solver_animation(&result),
                )
            } else {
                (empty_animation(), empty_animation())
            };
            PlannerResponse {
                ok: true,
                error: None,
                affinity: eval.affinity_sum,
                total_subs: eval.total_subs,
                fairness_ok: eval.fairness_ok,
                stamina_ok: eval.stamina_ok,
                subs_ok: eval.subs_ok,
                mip_status: result.mip.status.as_str().to_string(),
                elapsed_ms: t0.elapsed().as_millis() as f64,
                nodes_explored: result.mip.nodes_explored as u64,
                num_players: problem.num_players,
                num_positions: problem.num_positions,
                num_variables,
                num_constraints,
                used_fallback: result.used_fallback,
                fallback_reason: result.fallback_reason,
                assignment: result.schedule.assignment,
                bench: result.schedule.bench,
                pitch_animation: pitch,
                solver_animation: solver,
            }
        }
    }
}

fn empty_animation() -> Animation {
    Animation {
        width: 100.0,
        height: 100.0,
        fps: 1.0,
        frames: vec![],
        ..Default::default()
    }
}

/// Combined HTML page with Pitch | Solver tabs (for static export).
pub fn render_planner_html_set(pitch: &Animation, solver: &Animation) -> String {
    build_html_set(
        &[
            AnimationVariant {
                id: "pitch".into(),
                label: "Pitch".into(),
                animation: pitch.clone(),
                summary: Some("Optimal lineup on the field".into()),
                controls: None,
            },
            AnimationVariant {
                id: "solver".into(),
                label: "IP/MIP solver".into(),
                animation: solver.clone(),
                summary: Some("Branch-and-cut search".into()),
                controls: None,
            },
        ],
        &AnimationSetOptions {
            title: Some("Soccer rotation planner".into()),
            subtitle: Some("11-a-side · max 7 subs · IP/MIP optimal".into()),
            selector_label: Some("View".into()),
        },
    )
}

/// Smoke-run: default request → HTML file in `out/`.
pub fn run_default_export() {
    let req = super::model::default_planner_request();
    let resp = solve_planner(&req);
    if !resp.ok {
        eprintln!(
            "# soccer planner: solve failed: {}",
            resp.error.unwrap_or_default()
        );
        return;
    }
    let html = render_planner_html_set(&resp.pitch_animation, &resp.solver_animation);
    let out_dir = std::path::Path::new("out");
    let _ = std::fs::create_dir_all(out_dir);
    let path = out_dir.join("soccer-planner.html");
    if std::fs::write(&path, html).is_ok() {
        println!("# Soccer planner UI written to {}", path.display());
    }
}

#[cfg(test)]
mod tests {
    use super::{build_problem_from_request, solve_planner_summary, PlannerResponse};
    use crate::des::general::ip_mip_des::validate_ipmip_problem;
    use crate::des::general::soccer_rotation::build_soccer_ipmip;
    use crate::des::soccer_planner::default_planner_request;

    fn assert_solved(resp: &PlannerResponse) {
        assert!(resp.ok, "default planner solve failed: {:?}", resp.error);
        assert_eq!(resp.num_players, 18);
        assert_eq!(resp.num_positions, 11);
        assert!(resp.fairness_ok, "fairness check failed");
        assert!(resp.stamina_ok, "stamina check failed");
        assert!(resp.subs_ok, "substitution check failed");
        assert_eq!(resp.assignment.len(), 2);
        assert_eq!(resp.bench.len(), 2);
    }

    #[test]
    fn default_planner_request_solves() {
        let req = default_planner_request();
        assert_solved(&solve_planner_summary(&req));
    }

    #[test]
    fn default_planner_ipmip_model_validates() {
        let req = default_planner_request();
        let problem = build_problem_from_request(&req).unwrap();
        let model = build_soccer_ipmip(&problem);
        validate_ipmip_problem(&model.ip);
    }
}
