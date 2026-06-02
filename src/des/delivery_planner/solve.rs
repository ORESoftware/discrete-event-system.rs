//! Time-windowed delivery routing solved through the in-house IP/MIP stack.

use std::collections::HashSet;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::des::animation::types::{
    Anchor, Animation, CircleShape, FontWeight, Frame, FrameParts, LineShape, RectShape, Shape,
    TextShape,
};
use crate::des::general::ip_mip_des::{
    solve_ipmip_with_des, BranchRule, IPMIPProblem, IPMIPSolveOptions, LpRelaxationAlgorithm,
    NodeSelection, TraceAction,
};
use crate::des::general::lp::Sense;

use super::model::{
    format_minutes, normalize_delivery_request, DeliveryObjectiveMode, DeliveryPlannerRequest,
};

const EARTH_RADIUS_MILES: f64 = 3958.7613;
const STAGE_W: f64 = 1160.0;
const STAGE_H: f64 = 680.0;
const WINDOW_CENTER_EDGE_PENALTY: f64 = 48.0;
const EDGE_SOFT_THRESHOLD_MINUTES: f64 = 30.0;
const EDGE_HARD_THRESHOLD_MINUTES: f64 = 10.0;
const EDGE_SOFT_PENALTY: f64 = 8.0;
const EDGE_HARD_PENALTY: f64 = 48.0;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeliveryVisit {
    pub stop_index: usize,
    pub label: String,
    pub address: String,
    pub arrival: u32,
    pub depart: u32,
    pub window_start: u32,
    pub window_end: u32,
    pub arrival_text: String,
    pub depart_text: String,
    pub window_text: String,
    pub distance_from_previous: f64,
    pub travel_minutes_from_previous: f64,
    pub wait_minutes: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeliveryLeg {
    pub from: String,
    pub to: String,
    pub distance: f64,
    pub travel_minutes: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DeliverySolverTrace {
    pub node_id: String,
    pub depth: usize,
    pub action: String,
    pub lp_z: Option<f64>,
    pub reason: Option<String>,
    pub fractional: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct DeliveryPlannerResponse {
    pub ok: bool,
    pub error: Option<String>,
    pub solver_status: String,
    pub solver_kind: String,
    pub used_fallback: bool,
    pub fallback_reason: Option<String>,
    pub in_house_only: bool,
    pub uses_external_solvers: bool,
    pub elapsed_ms: f64,
    pub nodes_explored: usize,
    pub lp_solves: usize,
    pub num_variables: usize,
    pub num_constraints: usize,
    pub objective_mode: DeliveryObjectiveMode,
    pub objective_value: f64,
    pub objective_distance: f64,
    pub window_edge_penalty: f64,
    pub window_center_penalty: f64,
    pub total_distance: f64,
    pub total_travel_minutes: f64,
    pub total_wait_minutes: f64,
    pub route: Vec<usize>,
    pub visits: Vec<DeliveryVisit>,
    pub legs: Vec<DeliveryLeg>,
    pub itinerary_text: String,
    pub solver_notes: Vec<String>,
    pub solver_trace: Vec<DeliverySolverTrace>,
    pub route_animation: Animation,
}

#[derive(Clone, Debug)]
struct StopNode {
    label: String,
    address: String,
    lat: f64,
    lon: f64,
    window_start: u32,
    window_end: u32,
    service_minutes: f64,
}

#[derive(Clone, Debug)]
struct ModelBuild {
    problem: IPMIPProblem,
    x_index: Vec<Vec<Option<usize>>>,
    num_constraints: usize,
    distance: Vec<Vec<f64>>,
    travel_minutes: Vec<Vec<f64>>,
}

pub fn solve_delivery_planner(req: &DeliveryPlannerRequest) -> DeliveryPlannerResponse {
    solve_delivery_planner_inner(req, true)
}

pub fn solve_delivery_planner_summary(req: &DeliveryPlannerRequest) -> DeliveryPlannerResponse {
    solve_delivery_planner_inner(req, false)
}

fn solve_delivery_planner_inner(
    req: &DeliveryPlannerRequest,
    render_animation: bool,
) -> DeliveryPlannerResponse {
    let mut req = req.clone();
    normalize_delivery_request(&mut req);
    if let Err(err) = validate_request(&req) {
        return error_response(err);
    }
    let nodes = build_nodes(&req);
    let build = build_ip_model(&req, &nodes);
    let t0 = Instant::now();
    match position_locked_route(&req, &nodes, &build) {
        Ok(Some(route)) => {
            return response_from_route(
                &req,
                &nodes,
                &build,
                route,
                "position-locked".to_string(),
                "position-locked-route-search".to_string(),
                false,
                None,
                true,
                false,
                t0.elapsed().as_secs_f64() * 1000.0,
                0,
                0,
                build.problem.c.len(),
                build.num_constraints,
                0.0,
                vec![DeliverySolverTrace {
                    node_id: "position-locked".to_string(),
                    depth: 0,
                    action: "incumbent".to_string(),
                    lp_z: None,
                    reason: Some(
                        "locked stop rows were fixed while unlocked rows were optimized"
                            .to_string(),
                    ),
                    fractional: Vec::new(),
                }],
                render_animation,
            )
            .unwrap_or_else(|| {
                error_response("locked stop positions leave no feasible route".to_string())
            });
        }
        Ok(None) => {}
        Err(err) => return error_response(err),
    }
    let mip_result = catch_unwind(AssertUnwindSafe(|| {
        solve_ipmip_with_des(
            build.problem.clone(),
            IPMIPSolveOptions {
                max_nodes: Some(req.solver_max_nodes),
                max_ticks: Some(req.solver_max_ticks),
                time_limit_ms: Some(req.solver_time_limit_ms),
                lp_max_iters: Some(req.solver_lp_max_iters),
                int_tol: Some(1e-6),
                branch_rule: Some(BranchRule::MostFractional),
                node_selection: Some(NodeSelection::BestBound),
                lp_algorithm: Some(LpRelaxationAlgorithm::Auto),
                allow_external_solvers: Some(false),
                max_cut_rounds: Some(3),
                max_cuts_per_node: Some(16),
                heuristic_passes: Some(32),
                verbose: Some(false),
                mip_start: None,
            },
        )
    }));

    match mip_result {
        Ok(sol) => {
            let solver_status = sol.status.as_str().to_string();
            if !sol.x.is_empty() {
                if let Some(route) = extract_route(&build, &sol.x, req.stops.len()) {
                    if let Some(resp) = response_from_route(
                        &req,
                        &nodes,
                        &build,
                        route,
                        solver_status.clone(),
                        "in-house-ip-mip".to_string(),
                        false,
                        None,
                        sol.in_house_only,
                        sol.uses_external_solvers,
                        sol.elapsed_ms,
                        sol.nodes_explored,
                        sol.lp_solves,
                        build.problem.c.len(),
                        build.num_constraints,
                        sol.z,
                        trace_from_ipmip(&sol.trace),
                        render_animation,
                    ) {
                        return resp;
                    }
                }
            }
            let reason = format!(
                "IP/MIP returned `{}` without a usable time-window route",
                solver_status
            );
            fallback_response(&req, &nodes, &build, t0, reason, render_animation)
        }
        Err(_) => fallback_response(
            &req,
            &nodes,
            &build,
            t0,
            "IP/MIP solver rejected the model; used route repair fallback".to_string(),
            render_animation,
        ),
    }
}

fn validate_request(req: &DeliveryPlannerRequest) -> Result<(), String> {
    if req.stops.is_empty() {
        return Err("add at least one delivery stop".to_string());
    }
    if req.stops.len() > 14 {
        return Err("the in-house planner currently accepts up to 14 stops per route".to_string());
    }
    for stop in &req.stops {
        if !stop.lat.is_finite()
            || !stop.lon.is_finite()
            || stop.lat.abs() > 90.0
            || stop.lon.abs() > 180.0
        {
            return Err(format!("{} has invalid coordinates", stop.label));
        }
        if stop.window_end < stop.window_start {
            return Err(format!("{} has an inverted time window", stop.label));
        }
    }
    Ok(())
}

fn locked_order_route(req: &DeliveryPlannerRequest) -> Result<Vec<usize>, String> {
    let ordered_ids: Vec<&str> = if req.route_rules.ordered_stop_ids.is_empty() {
        req.stops.iter().map(|s| s.id.as_str()).collect()
    } else {
        req.route_rules
            .ordered_stop_ids
            .iter()
            .map(String::as_str)
            .collect()
    };
    let mut seen = HashSet::new();
    let mut route = Vec::with_capacity(req.stops.len());
    for id in ordered_ids {
        let idx = req
            .stops
            .iter()
            .position(|stop| stop.id == id)
            .ok_or_else(|| format!("locked route references unknown stop `{id}`"))?;
        if seen.insert(idx) {
            route.push(idx);
        }
    }
    for idx in 0..req.stops.len() {
        if seen.insert(idx) {
            route.push(idx);
        }
    }
    if route.len() == req.stops.len() {
        Ok(route)
    } else {
        Err("locked route order could not cover every stop".to_string())
    }
}

fn position_locked_route(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
) -> Result<Option<Vec<usize>>, String> {
    if req.route_rules.locked_order {
        return locked_order_route(req).map(Some);
    }
    if req.route_rules.locked_positions.is_empty() {
        return Ok(None);
    }
    let locked_slots = locked_slots(req)?;
    let route = if req.stops.len() <= 9 {
        exact_position_locked_route(req, nodes, build, &locked_slots)
    } else {
        greedy_position_locked_route(req, nodes, build, &locked_slots)
    };
    route
        .ok_or_else(|| "locked stop positions leave no feasible route".to_string())
        .map(Some)
}

fn locked_slots(req: &DeliveryPlannerRequest) -> Result<Vec<Option<usize>>, String> {
    let mut slots: Vec<Option<usize>> = vec![None; req.stops.len()];
    let mut seen_stops = HashSet::new();
    for lock in &req.route_rules.locked_positions {
        if lock.position >= req.stops.len() {
            return Err(format!(
                "locked position {} is outside the {}-stop route",
                lock.position + 1,
                req.stops.len()
            ));
        }
        let stop_idx = req
            .stops
            .iter()
            .position(|stop| stop.id == lock.stop_id)
            .ok_or_else(|| format!("locked position references unknown stop `{}`", lock.stop_id))?;
        if !seen_stops.insert(stop_idx) {
            return Err(format!(
                "{} is locked more than once",
                req.stops[stop_idx].label
            ));
        }
        if let Some(existing) = slots[lock.position] {
            return Err(format!(
                "route position {} locks both {} and {}",
                lock.position + 1,
                req.stops[existing].label,
                req.stops[stop_idx].label
            ));
        }
        slots[lock.position] = Some(stop_idx);
    }
    Ok(slots)
}

fn exact_position_locked_route(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    locked_slots: &[Option<usize>],
) -> Option<Vec<usize>> {
    let mut best_route = None;
    let mut best_score = f64::INFINITY;
    let mut used = vec![false; req.stops.len()];
    let mut route = Vec::new();
    dfs_position_locked_route(
        req,
        nodes,
        build,
        locked_slots,
        0,
        req.depart_time as f64,
        &mut used,
        &mut route,
        &mut best_route,
        &mut best_score,
    );
    best_route
}

#[allow(clippy::too_many_arguments)]
fn dfs_position_locked_route(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    locked_slots: &[Option<usize>],
    cur_node: usize,
    clock: f64,
    used: &mut [bool],
    route: &mut Vec<usize>,
    best_route: &mut Option<Vec<usize>>,
    best_score: &mut f64,
) {
    if route.len() == req.stops.len() {
        if let Some(score) = route_score(req, nodes, build, route) {
            if score < *best_score {
                *best_score = score;
                *best_route = Some(route.clone());
            }
        }
        return;
    }
    let position = route.len();
    if let Some(stop_idx) = locked_slots[position] {
        if !used[stop_idx] {
            try_position_locked_step(
                req,
                nodes,
                build,
                locked_slots,
                cur_node,
                clock,
                stop_idx,
                used,
                route,
                best_route,
                best_score,
            );
        }
        return;
    }
    for stop_idx in 0..req.stops.len() {
        if used[stop_idx] {
            continue;
        }
        try_position_locked_step(
            req,
            nodes,
            build,
            locked_slots,
            cur_node,
            clock,
            stop_idx,
            used,
            route,
            best_route,
            best_score,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn try_position_locked_step(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    locked_slots: &[Option<usize>],
    cur_node: usize,
    clock: f64,
    stop_idx: usize,
    used: &mut [bool],
    route: &mut Vec<usize>,
    best_route: &mut Option<Vec<usize>>,
    best_score: &mut f64,
) {
    let next_node = stop_idx + 1;
    let stop = &nodes[next_node];
    let raw_arrival = clock + build.travel_minutes[cur_node][next_node];
    let arrival = raw_arrival.max(stop.window_start as f64);
    if arrival > stop.window_end as f64 + 1e-6 {
        return;
    }
    used[stop_idx] = true;
    route.push(stop_idx);
    dfs_position_locked_route(
        req,
        nodes,
        build,
        locked_slots,
        next_node,
        arrival + stop.service_minutes,
        used,
        route,
        best_route,
        best_score,
    );
    route.pop();
    used[stop_idx] = false;
}

fn greedy_position_locked_route(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    locked_slots: &[Option<usize>],
) -> Option<Vec<usize>> {
    let mut remaining: HashSet<usize> = (0..req.stops.len()).collect();
    let mut route = Vec::new();
    let mut cur_node = 0usize;
    let mut clock = req.depart_time as f64;
    for &locked in locked_slots.iter().take(req.stops.len()) {
        let next = match locked {
            Some(stop_idx) => {
                if !remaining.contains(&stop_idx) {
                    return None;
                }
                stop_idx
            }
            None => pick_greedy_stop(req, nodes, build, cur_node, clock, &remaining)?,
        };
        let next_node = next + 1;
        let arrival = (clock + build.travel_minutes[cur_node][next_node])
            .max(nodes[next_node].window_start as f64);
        if arrival > nodes[next_node].window_end as f64 {
            return None;
        }
        remaining.remove(&next);
        route.push(next);
        clock = arrival + nodes[next_node].service_minutes;
        cur_node = next_node;
    }
    Some(route)
}

fn error_response(message: String) -> DeliveryPlannerResponse {
    DeliveryPlannerResponse {
        ok: false,
        error: Some(message),
        solver_status: "error".to_string(),
        solver_kind: "none".to_string(),
        used_fallback: false,
        fallback_reason: None,
        in_house_only: true,
        uses_external_solvers: false,
        elapsed_ms: 0.0,
        nodes_explored: 0,
        lp_solves: 0,
        num_variables: 0,
        num_constraints: 0,
        objective_mode: DeliveryObjectiveMode::Distance,
        objective_value: 0.0,
        objective_distance: 0.0,
        window_edge_penalty: 0.0,
        window_center_penalty: 0.0,
        total_distance: 0.0,
        total_travel_minutes: 0.0,
        total_wait_minutes: 0.0,
        route: Vec::new(),
        visits: Vec::new(),
        legs: Vec::new(),
        itinerary_text: String::new(),
        solver_notes: Vec::new(),
        solver_trace: Vec::new(),
        route_animation: empty_animation(),
    }
}

fn build_nodes(req: &DeliveryPlannerRequest) -> Vec<StopNode> {
    let mut nodes = vec![StopNode {
        label: req.depot_label.clone(),
        address: req.depot_address.clone(),
        lat: req.depot_lat,
        lon: req.depot_lon,
        window_start: req.depart_time,
        window_end: 24 * 60,
        service_minutes: 0.0,
    }];
    nodes.extend(req.stops.iter().map(|s| StopNode {
        label: s.label.clone(),
        address: s.address.clone(),
        lat: s.lat,
        lon: s.lon,
        window_start: s.window_start,
        window_end: s.window_end,
        service_minutes: s.service_minutes,
    }));
    nodes
}

fn window_center_dev_coeff(node: &StopNode) -> f64 {
    let half_width = ((node.window_end.saturating_sub(node.window_start)) as f64 / 2.0).max(1.0);
    WINDOW_CENTER_EDGE_PENALTY / half_width
}

fn window_center_node_penalty(node: &StopNode, arrival: f64) -> f64 {
    let midpoint = (node.window_start as f64 + node.window_end as f64) / 2.0;
    let half_width = ((node.window_end.saturating_sub(node.window_start)) as f64 / 2.0).max(1.0);
    (arrival - midpoint).abs() / half_width * WINDOW_CENTER_EDGE_PENALTY
}

fn window_edge_penalty_for_arrival(node: &StopNode, arrival: f64) -> f64 {
    let start_slack = arrival - node.window_start as f64;
    let end_slack = node.window_end as f64 - arrival;
    let edge_slack = start_slack.min(end_slack).max(0.0);
    let soft = ((EDGE_SOFT_THRESHOLD_MINUTES - edge_slack).max(0.0) / EDGE_SOFT_THRESHOLD_MINUTES)
        * EDGE_SOFT_PENALTY;
    let hard = ((EDGE_HARD_THRESHOLD_MINUTES - edge_slack).max(0.0) / EDGE_HARD_THRESHOLD_MINUTES)
        * EDGE_HARD_PENALTY;
    soft + hard
}

fn build_ip_model(req: &DeliveryPlannerRequest, nodes: &[StopNode]) -> ModelBuild {
    let n_nodes = nodes.len();
    let mut distance = vec![vec![0.0; n_nodes]; n_nodes];
    let mut travel_minutes = vec![vec![0.0; n_nodes]; n_nodes];
    for i in 0..n_nodes {
        for j in 0..n_nodes {
            if i == j {
                continue;
            }
            let d = haversine_miles(nodes[i].lat, nodes[i].lon, nodes[j].lat, nodes[j].lon);
            distance[i][j] = d;
            travel_minutes[i][j] = d / req.average_speed_mph.max(1.0) * 60.0;
        }
    }

    let mut x_index = vec![vec![None; n_nodes]; n_nodes];
    let mut c = Vec::new();
    let mut integer_vars = Vec::new();
    let mut ub = Vec::new();
    let mut var_names = Vec::new();
    for i in 0..n_nodes {
        for j in 0..n_nodes {
            if i == j {
                continue;
            }
            let idx = c.len();
            x_index[i][j] = Some(idx);
            c.push(match req.objective_mode {
                DeliveryObjectiveMode::Distance => distance[i][j],
                DeliveryObjectiveMode::TravelTime | DeliveryObjectiveMode::WindowCenter => {
                    travel_minutes[i][j]
                }
            });
            integer_vars.push(true);
            ub.push(1.0);
            var_names.push(format!("x_{}_{}", i, j));
        }
    }
    let mut t_index = vec![None; n_nodes];
    let latest_end = nodes
        .iter()
        .skip(1)
        .map(|s| s.window_end)
        .max()
        .unwrap_or(req.depart_time)
        .max(req.depart_time) as f64;
    let max_travel = travel_minutes
        .iter()
        .flatten()
        .copied()
        .fold(0.0_f64, f64::max);
    let horizon = latest_end
        + nodes.iter().map(|s| s.service_minutes).sum::<f64>()
        + max_travel * (n_nodes as f64 + 1.0)
        + 60.0;
    for (node, slot) in t_index.iter_mut().enumerate().skip(1) {
        let idx = c.len();
        *slot = Some(idx);
        c.push(0.0);
        integer_vars.push(false);
        ub.push(horizon);
        var_names.push(format!("arrival_{}", node));
    }
    let mut center_dev_index = vec![None; n_nodes];
    let mut edge_start_soft_index = vec![None; n_nodes];
    let mut edge_end_soft_index = vec![None; n_nodes];
    let mut edge_start_hard_index = vec![None; n_nodes];
    let mut edge_end_hard_index = vec![None; n_nodes];
    if req.objective_mode == DeliveryObjectiveMode::WindowCenter {
        for (node, slot) in center_dev_index.iter_mut().enumerate().skip(1) {
            let idx = c.len();
            *slot = Some(idx);
            c.push(window_center_dev_coeff(&nodes[node]));
            integer_vars.push(false);
            ub.push(horizon);
            var_names.push(format!("center_deviation_{}", node));
        }
    } else {
        for node in 1..n_nodes {
            let idx = c.len();
            edge_start_soft_index[node] = Some(idx);
            c.push(EDGE_SOFT_PENALTY / EDGE_SOFT_THRESHOLD_MINUTES);
            integer_vars.push(false);
            ub.push(EDGE_SOFT_THRESHOLD_MINUTES);
            var_names.push(format!("edge_start_soft_{}", node));

            let idx = c.len();
            edge_end_soft_index[node] = Some(idx);
            c.push(EDGE_SOFT_PENALTY / EDGE_SOFT_THRESHOLD_MINUTES);
            integer_vars.push(false);
            ub.push(EDGE_SOFT_THRESHOLD_MINUTES);
            var_names.push(format!("edge_end_soft_{}", node));

            let idx = c.len();
            edge_start_hard_index[node] = Some(idx);
            c.push(EDGE_HARD_PENALTY / EDGE_HARD_THRESHOLD_MINUTES);
            integer_vars.push(false);
            ub.push(EDGE_HARD_THRESHOLD_MINUTES);
            var_names.push(format!("edge_start_hard_{}", node));

            let idx = c.len();
            edge_end_hard_index[node] = Some(idx);
            c.push(EDGE_HARD_PENALTY / EDGE_HARD_THRESHOLD_MINUTES);
            integer_vars.push(false);
            ub.push(EDGE_HARD_THRESHOLD_MINUTES);
            var_names.push(format!("edge_end_hard_{}", node));
        }
    }

    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut con_names = Vec::new();
    let var_count = c.len();

    for customer in 1..n_nodes {
        let incoming = (0..n_nodes)
            .filter(|&i| i != customer)
            .map(|i| (x_index[i][customer].unwrap(), 1.0))
            .collect();
        add_eq_rows(
            &mut a,
            &mut b,
            &mut con_names,
            var_count,
            incoming,
            1.0,
            &format!("in_{customer}"),
        );
        let outgoing = (0..n_nodes)
            .filter(|&j| j != customer)
            .map(|j| (x_index[customer][j].unwrap(), 1.0))
            .collect();
        add_eq_rows(
            &mut a,
            &mut b,
            &mut con_names,
            var_count,
            outgoing,
            1.0,
            &format!("out_{customer}"),
        );
    }
    add_eq_rows(
        &mut a,
        &mut b,
        &mut con_names,
        var_count,
        (1..n_nodes)
            .map(|j| (x_index[0][j].unwrap(), 1.0))
            .collect(),
        1.0,
        "depot_out",
    );
    add_eq_rows(
        &mut a,
        &mut b,
        &mut con_names,
        var_count,
        (1..n_nodes)
            .map(|i| (x_index[i][0].unwrap(), 1.0))
            .collect(),
        1.0,
        "depot_in",
    );

    for customer in 1..n_nodes {
        let t = t_index[customer].unwrap();
        let mut row = vec![0.0; var_count];
        row[t] = 1.0;
        add_le_row(
            &mut a,
            &mut b,
            &mut con_names,
            row,
            nodes[customer].window_end as f64,
            format!("window_end_{customer}"),
        );
        let mut row = vec![0.0; var_count];
        row[t] = -1.0;
        add_le_row(
            &mut a,
            &mut b,
            &mut con_names,
            row,
            -(nodes[customer].window_start as f64),
            format!("window_start_{customer}"),
        );
        if let Some(dev) = center_dev_index[customer] {
            let midpoint =
                (nodes[customer].window_start as f64 + nodes[customer].window_end as f64) / 2.0;
            let mut row = vec![0.0; var_count];
            row[t] = 1.0;
            row[dev] = -1.0;
            add_le_row(
                &mut a,
                &mut b,
                &mut con_names,
                row,
                midpoint,
                format!("center_hi_{customer}"),
            );
            let mut row = vec![0.0; var_count];
            row[t] = -1.0;
            row[dev] = -1.0;
            add_le_row(
                &mut a,
                &mut b,
                &mut con_names,
                row,
                -midpoint,
                format!("center_lo_{customer}"),
            );
        }
        if let (Some(start_soft), Some(end_soft), Some(start_hard), Some(end_hard)) = (
            edge_start_soft_index[customer],
            edge_end_soft_index[customer],
            edge_start_hard_index[customer],
            edge_end_hard_index[customer],
        ) {
            add_edge_penalty_rows(
                &mut a,
                &mut b,
                &mut con_names,
                var_count,
                t,
                start_soft,
                end_soft,
                nodes[customer].window_start as f64,
                nodes[customer].window_end as f64,
                EDGE_SOFT_THRESHOLD_MINUTES,
                format!("edge_soft_{customer}"),
            );
            add_edge_penalty_rows(
                &mut a,
                &mut b,
                &mut con_names,
                var_count,
                t,
                start_hard,
                end_hard,
                nodes[customer].window_start as f64,
                nodes[customer].window_end as f64,
                EDGE_HARD_THRESHOLD_MINUTES,
                format!("edge_hard_{customer}"),
            );
        }
    }

    let big_m = horizon + max_travel + 24.0 * 60.0;
    for j in 1..n_nodes {
        let mut row = vec![0.0; var_count];
        row[t_index[j].unwrap()] = -1.0;
        row[x_index[0][j].unwrap()] = big_m;
        add_le_row(
            &mut a,
            &mut b,
            &mut con_names,
            row,
            big_m - req.depart_time as f64 - travel_minutes[0][j],
            format!("time_0_{j}"),
        );
    }
    for i in 1..n_nodes {
        for j in 1..n_nodes {
            if i == j {
                continue;
            }
            let mut row = vec![0.0; var_count];
            row[t_index[i].unwrap()] = 1.0;
            row[t_index[j].unwrap()] = -1.0;
            row[x_index[i][j].unwrap()] = big_m;
            add_le_row(
                &mut a,
                &mut b,
                &mut con_names,
                row,
                big_m - nodes[i].service_minutes - travel_minutes[i][j],
                format!("time_{i}_{j}"),
            );
        }
    }
    let num_constraints = a.len();

    ModelBuild {
        problem: IPMIPProblem {
            sense: Sense::Min,
            c,
            a,
            b,
            integer_vars,
            ub: Some(ub),
            var_names: Some(var_names),
            con_names: Some(con_names),
            lazy_constraints: None,
            variable_nodes: None,
            constraint_nodes: None,
        },
        x_index,
        num_constraints,
        distance,
        travel_minutes,
    }
}

fn add_le_row(
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    con_names: &mut Vec<String>,
    row: Vec<f64>,
    rhs: f64,
    name: String,
) {
    a.push(row);
    b.push(rhs);
    con_names.push(name);
}

fn add_eq_rows(
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    con_names: &mut Vec<String>,
    var_count: usize,
    terms: Vec<(usize, f64)>,
    rhs: f64,
    name: &str,
) {
    let mut row = vec![0.0; var_count];
    for (idx, coef) in terms {
        row[idx] = coef;
    }
    add_le_row(a, b, con_names, row.clone(), rhs, format!("{name}_le"));
    for x in &mut row {
        *x = -*x;
    }
    add_le_row(a, b, con_names, row, -rhs, format!("{name}_ge"));
}

#[allow(clippy::too_many_arguments)]
fn add_edge_penalty_rows(
    a: &mut Vec<Vec<f64>>,
    b: &mut Vec<f64>,
    con_names: &mut Vec<String>,
    var_count: usize,
    t_idx: usize,
    start_var: usize,
    end_var: usize,
    window_start: f64,
    window_end: f64,
    threshold: f64,
    name: String,
) {
    let mut row = vec![0.0; var_count];
    row[start_var] = -1.0;
    row[t_idx] = -1.0;
    add_le_row(
        a,
        b,
        con_names,
        row,
        -(window_start + threshold),
        format!("{name}_start"),
    );

    let mut row = vec![0.0; var_count];
    row[end_var] = -1.0;
    row[t_idx] = 1.0;
    add_le_row(
        a,
        b,
        con_names,
        row,
        window_end - threshold,
        format!("{name}_end"),
    );
}

fn extract_route(build: &ModelBuild, x: &[f64], n_customers: usize) -> Option<Vec<usize>> {
    let mut route = Vec::with_capacity(n_customers);
    let mut seen: HashSet<usize> = HashSet::new();
    let mut cur = 0usize;
    for _ in 0..=n_customers {
        let next = build.x_index[cur].iter().enumerate().find_map(|(j, idx)| {
            idx.and_then(|k| (x.get(k).copied().unwrap_or(0.0) > 0.5).then_some(j))
        })?;
        if next == 0 {
            return (route.len() == n_customers).then_some(route);
        }
        if !seen.insert(next) {
            return None;
        }
        route.push(next - 1);
        cur = next;
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn response_from_route(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    route: Vec<usize>,
    solver_status: String,
    solver_kind: String,
    used_fallback: bool,
    fallback_reason: Option<String>,
    in_house_only: bool,
    uses_external_solvers: bool,
    elapsed_ms: f64,
    nodes_explored: usize,
    lp_solves: usize,
    num_variables: usize,
    num_constraints: usize,
    _solver_objective: f64,
    solver_trace: Vec<DeliverySolverTrace>,
    render_animation: bool,
) -> Option<DeliveryPlannerResponse> {
    let itinerary = build_itinerary(req, nodes, build, &route)?;
    let objective_value = objective_score(req.objective_mode, &itinerary);
    let notes = delivery_notes(
        req.objective_mode,
        &solver_status,
        &solver_kind,
        used_fallback,
        fallback_reason.as_deref(),
        req.route_rules.locked_order,
        req.route_rules.locked_positions.len(),
        num_variables,
        num_constraints,
        itinerary.total_distance,
        objective_value,
        itinerary.window_edge_penalty,
        itinerary.window_center_penalty,
    );
    let animation = if render_animation {
        render_delivery_animation(req, nodes, &route, &itinerary.visits, &itinerary.legs)
    } else {
        empty_animation()
    };
    Some(DeliveryPlannerResponse {
        ok: true,
        error: None,
        solver_status,
        solver_kind,
        used_fallback,
        fallback_reason,
        in_house_only,
        uses_external_solvers,
        elapsed_ms,
        nodes_explored,
        lp_solves,
        num_variables,
        num_constraints,
        objective_mode: req.objective_mode,
        objective_value,
        objective_distance: itinerary.total_distance,
        window_edge_penalty: itinerary.window_edge_penalty,
        window_center_penalty: itinerary.window_center_penalty,
        total_distance: itinerary.total_distance,
        total_travel_minutes: itinerary.total_travel_minutes,
        total_wait_minutes: itinerary.total_wait_minutes,
        route,
        visits: itinerary.visits,
        legs: itinerary.legs,
        itinerary_text: itinerary.text,
        solver_notes: notes,
        solver_trace,
        route_animation: animation,
    })
}

struct ItineraryBuild {
    visits: Vec<DeliveryVisit>,
    legs: Vec<DeliveryLeg>,
    text: String,
    total_distance: f64,
    total_travel_minutes: f64,
    total_wait_minutes: f64,
    window_edge_penalty: f64,
    window_center_penalty: f64,
}

fn build_itinerary(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    route: &[usize],
) -> Option<ItineraryBuild> {
    let mut visits = Vec::new();
    let mut legs = Vec::new();
    let mut total_distance = 0.0;
    let mut total_travel_minutes = 0.0;
    let mut total_wait_minutes = 0.0;
    let mut cur_node = 0usize;
    let mut clock = req.depart_time as f64;
    for &stop_idx in route {
        let next_node = stop_idx + 1;
        let travel = build.travel_minutes[cur_node][next_node];
        let distance = build.distance[cur_node][next_node];
        let raw_arrival = clock + travel;
        let stop = &nodes[next_node];
        let arrival = raw_arrival.max(stop.window_start as f64);
        let wait = (arrival - raw_arrival).max(0.0);
        if arrival > stop.window_end as f64 + 1e-6 {
            return None;
        }
        let depart = arrival + stop.service_minutes;
        legs.push(DeliveryLeg {
            from: nodes[cur_node].label.clone(),
            to: stop.label.clone(),
            distance,
            travel_minutes: travel,
        });
        visits.push(DeliveryVisit {
            stop_index: stop_idx,
            label: stop.label.clone(),
            address: stop.address.clone(),
            arrival: arrival.round() as u32,
            depart: depart.round() as u32,
            window_start: stop.window_start,
            window_end: stop.window_end,
            arrival_text: format_minutes(arrival.round() as u32),
            depart_text: format_minutes(depart.round() as u32),
            window_text: format!(
                "{}-{}",
                format_minutes(stop.window_start),
                format_minutes(stop.window_end)
            ),
            distance_from_previous: distance,
            travel_minutes_from_previous: travel,
            wait_minutes: wait,
        });
        total_distance += distance;
        total_travel_minutes += travel;
        total_wait_minutes += wait;
        clock = depart;
        cur_node = next_node;
    }
    let back_distance = build.distance[cur_node][0];
    let back_travel = build.travel_minutes[cur_node][0];
    total_distance += back_distance;
    total_travel_minutes += back_travel;
    legs.push(DeliveryLeg {
        from: nodes[cur_node].label.clone(),
        to: nodes[0].label.clone(),
        distance: back_distance,
        travel_minutes: back_travel,
    });
    let return_time = clock + back_travel;
    let edge_penalty = window_edge_penalty(&visits);
    let center_penalty = window_center_penalty(&visits);
    let text = itinerary_text(
        req,
        nodes,
        &visits,
        total_distance,
        total_travel_minutes,
        total_wait_minutes,
        return_time.round() as u32,
    );
    Some(ItineraryBuild {
        visits,
        legs,
        text,
        total_distance,
        total_travel_minutes,
        total_wait_minutes,
        window_edge_penalty: edge_penalty,
        window_center_penalty: center_penalty,
    })
}

fn objective_score(mode: DeliveryObjectiveMode, itinerary: &ItineraryBuild) -> f64 {
    match mode {
        DeliveryObjectiveMode::Distance => itinerary.total_distance + itinerary.window_edge_penalty,
        DeliveryObjectiveMode::TravelTime => {
            itinerary.total_travel_minutes + itinerary.window_edge_penalty
        }
        DeliveryObjectiveMode::WindowCenter => {
            itinerary.total_travel_minutes + itinerary.window_center_penalty
        }
    }
}

fn window_edge_penalty(visits: &[DeliveryVisit]) -> f64 {
    visits
        .iter()
        .map(|visit| {
            let node = StopNode {
                label: visit.label.clone(),
                address: visit.address.clone(),
                lat: 0.0,
                lon: 0.0,
                window_start: visit.window_start,
                window_end: visit.window_end,
                service_minutes: 0.0,
            };
            window_edge_penalty_for_arrival(&node, visit.arrival as f64)
        })
        .sum()
}

fn window_center_penalty(visits: &[DeliveryVisit]) -> f64 {
    visits.iter().map(window_center_visit_penalty).sum()
}

fn window_center_visit_penalty(visit: &DeliveryVisit) -> f64 {
    let midpoint = (visit.window_start as f64 + visit.window_end as f64) / 2.0;
    let half_width = ((visit.window_end.saturating_sub(visit.window_start)) as f64 / 2.0).max(1.0);
    let deviation = (visit.arrival as f64 - midpoint).abs();
    (deviation / half_width) * WINDOW_CENTER_EDGE_PENALTY
}

fn itinerary_text(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    visits: &[DeliveryVisit],
    total_distance: f64,
    total_travel_minutes: f64,
    total_wait_minutes: f64,
    return_time: u32,
) -> String {
    let mut lines = Vec::new();
    lines.push(format!("Delivery itinerary for {}", req.user_id));
    lines.push(format!(
        "Depart {} at {}",
        nodes[0].label,
        format_minutes(req.depart_time)
    ));
    for (i, visit) in visits.iter().enumerate() {
        lines.push(format!(
            "{}. {} arrive {} depart {} window {}  ({:.2} mi, {:.0} min travel, {:.0} min wait)",
            i + 1,
            visit.label,
            visit.arrival_text,
            visit.depart_text,
            visit.window_text,
            visit.distance_from_previous,
            visit.travel_minutes_from_previous,
            visit.wait_minutes
        ));
        lines.push(format!("   {}", visit.address));
    }
    lines.push(format!(
        "Return to {} at {}",
        nodes[0].label,
        format_minutes(return_time)
    ));
    lines.push(format!(
        "Totals: {:.2} mi, {:.0} min drive, {:.0} min wait",
        total_distance, total_travel_minutes, total_wait_minutes
    ));
    lines.join("\n")
}

fn fallback_response(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    t0: Instant,
    reason: String,
    render_animation: bool,
) -> DeliveryPlannerResponse {
    match fallback_route(req, nodes, build) {
        Some(route) => response_from_route(
            req,
            nodes,
            build,
            route,
            "feasible-fallback".to_string(),
            "route-repair-dp".to_string(),
            true,
            Some(reason),
            true,
            false,
            t0.elapsed().as_secs_f64() * 1000.0,
            0,
            0,
            build.problem.c.len(),
            build.num_constraints,
            0.0,
            vec![DeliverySolverTrace {
                node_id: "fallback".to_string(),
                depth: 0,
                action: "incumbent".to_string(),
                lp_z: None,
                reason: Some("constructive route satisfied all delivery windows".to_string()),
                fractional: Vec::new(),
            }],
            render_animation,
        )
        .unwrap_or_else(|| error_response("no route can satisfy these time windows".to_string())),
        None => error_response("no route can satisfy these time windows".to_string()),
    }
}

fn fallback_route(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
) -> Option<Vec<usize>> {
    if req.stops.len() <= 9 {
        let mut best_route = None;
        let mut best_score = f64::INFINITY;
        let mut used = vec![false; req.stops.len()];
        let mut route = Vec::new();
        dfs_route(
            req,
            nodes,
            build,
            0,
            req.depart_time as f64,
            &mut used,
            &mut route,
            &mut best_route,
            &mut best_score,
        );
        return best_route;
    }
    greedy_route(req, nodes, build)
}

#[allow(clippy::too_many_arguments)]
fn dfs_route(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    cur_node: usize,
    clock: f64,
    used: &mut [bool],
    route: &mut Vec<usize>,
    best_route: &mut Option<Vec<usize>>,
    best_score: &mut f64,
) {
    if route.len() == req.stops.len() {
        if let Some(score) = route_score(req, nodes, build, route) {
            if score < *best_score {
                *best_score = score;
                *best_route = Some(route.clone());
            }
        }
        return;
    }
    for stop_idx in 0..req.stops.len() {
        if used[stop_idx] {
            continue;
        }
        let next_node = stop_idx + 1;
        let stop = &nodes[next_node];
        let raw_arrival = clock + build.travel_minutes[cur_node][next_node];
        let arrival = raw_arrival.max(stop.window_start as f64);
        if arrival > stop.window_end as f64 + 1e-6 {
            continue;
        }
        used[stop_idx] = true;
        route.push(stop_idx);
        dfs_route(
            req,
            nodes,
            build,
            next_node,
            arrival + stop.service_minutes,
            used,
            route,
            best_route,
            best_score,
        );
        route.pop();
        used[stop_idx] = false;
    }
}

fn route_score(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    route: &[usize],
) -> Option<f64> {
    build_itinerary(req, nodes, build, route)
        .map(|itinerary| objective_score(req.objective_mode, &itinerary))
}

fn greedy_leg_score(
    mode: DeliveryObjectiveMode,
    build: &ModelBuild,
    nodes: &[StopNode],
    cur_node: usize,
    next_node: usize,
    arrival: f64,
) -> f64 {
    match mode {
        DeliveryObjectiveMode::Distance => {
            build.distance[cur_node][next_node]
                + window_edge_penalty_for_arrival(&nodes[next_node], arrival)
        }
        DeliveryObjectiveMode::TravelTime => {
            build.travel_minutes[cur_node][next_node]
                + window_edge_penalty_for_arrival(&nodes[next_node], arrival)
        }
        DeliveryObjectiveMode::WindowCenter => {
            build.travel_minutes[cur_node][next_node]
                + window_center_node_penalty(&nodes[next_node], arrival)
        }
    }
}

fn objective_label(mode: DeliveryObjectiveMode) -> &'static str {
    match mode {
        DeliveryObjectiveMode::Distance => "route miles",
        DeliveryObjectiveMode::TravelTime => "drive minutes",
        DeliveryObjectiveMode::WindowCenter => "drive minutes plus window-center penalty",
    }
}

fn objective_note(mode: DeliveryObjectiveMode) -> &'static str {
    match mode {
        DeliveryObjectiveMode::Distance => {
            "Objective minimized route miles plus edge-window penalties; arrival-time rows enforce every customer delivery window."
        }
        DeliveryObjectiveMode::TravelTime => {
            "Objective minimized computed drive minutes plus edge-window penalties; arrival-time rows enforce every customer delivery window."
        }
        DeliveryObjectiveMode::WindowCenter => {
            "Objective minimized computed drive minutes plus a strong linear penalty for arriving away from each delivery-window midpoint."
        }
    }
}

fn delivery_notes(
    mode: DeliveryObjectiveMode,
    solver_status: &str,
    solver_kind: &str,
    used_fallback: bool,
    fallback_reason: Option<&str>,
    locked_order: bool,
    locked_position_count: usize,
    num_variables: usize,
    num_constraints: usize,
    total_distance: f64,
    objective_value: f64,
    window_edge_penalty: f64,
    window_center_penalty: f64,
) -> Vec<String> {
    let mut notes = vec![
        format!(
            "Built a directed binary route model with {num_variables} variables and {num_constraints} constraint rows."
        ),
        objective_note(mode).to_string(),
        format!("Optimized {} = {:.2}.", objective_label(mode), objective_value),
        format!("Planned total route distance is {:.2} miles.", total_distance),
    ];
    if mode != DeliveryObjectiveMode::WindowCenter {
        notes.push(format!(
            "Edge-window penalty contribution is {:.2}; arrivals inside 10 minutes of either edge receive the largest penalty, and arrivals inside 30 minutes receive a smaller penalty.",
            window_edge_penalty
        ));
    }
    if mode == DeliveryObjectiveMode::WindowCenter {
        notes.push(format!(
            "Window-center penalty contribution is {:.2}; zero is centered in every availability window.",
            window_center_penalty
        ));
    }
    if locked_order {
        notes.push(
            "Locked stop order was treated as a hard rule, so the vertical list order fixed the route sequence."
                .to_string(),
        );
    }
    if locked_position_count > 0 {
        notes.push(format!(
            "{locked_position_count} stop position lock(s) were treated as hard row constraints; unlocked rows remained optimizable."
        ));
    }
    if used_fallback {
        notes.push(
            fallback_reason
                .unwrap_or("Used the constructive route repair fallback.")
                .to_string(),
        );
    } else {
        notes.push(format!(
            "{solver_kind} completed with status `{solver_status}` using in-house LP relaxations."
        ));
    }
    notes
}

fn greedy_route(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
) -> Option<Vec<usize>> {
    let mut remaining: HashSet<usize> = (0..req.stops.len()).collect();
    let mut route = Vec::new();
    let mut cur_node = 0usize;
    let mut clock = req.depart_time as f64;
    while !remaining.is_empty() {
        let next = pick_greedy_stop(req, nodes, build, cur_node, clock, &remaining)?;
        let next_node = next + 1;
        let arrival = (clock + build.travel_minutes[cur_node][next_node])
            .max(nodes[next_node].window_start as f64);
        if arrival > nodes[next_node].window_end as f64 {
            return None;
        }
        remaining.remove(&next);
        route.push(next);
        clock = arrival + nodes[next_node].service_minutes;
        cur_node = next_node;
    }
    Some(route)
}

fn pick_greedy_stop(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    build: &ModelBuild,
    cur_node: usize,
    clock: f64,
    remaining: &HashSet<usize>,
) -> Option<usize> {
    remaining.iter().copied().min_by(|&a, &b| {
        let score = |idx: usize| {
            let node = idx + 1;
            let raw_arrival = clock + build.travel_minutes[cur_node][node];
            let arrival = raw_arrival.max(nodes[node].window_start as f64);
            if arrival > nodes[node].window_end as f64 {
                f64::INFINITY
            } else {
                greedy_leg_score(req.objective_mode, build, nodes, cur_node, node, arrival)
                    + (nodes[node].window_end as f64 - arrival) * 0.001
            }
        };
        score(a).total_cmp(&score(b))
    })
}

fn trace_from_ipmip(
    trace: &[crate::des::general::ip_mip_des::IPMIPTraceEvent],
) -> Vec<DeliverySolverTrace> {
    trace
        .iter()
        .take(80)
        .map(|e| DeliverySolverTrace {
            node_id: e.node_id.to_string(),
            depth: e.depth,
            action: match e.action {
                TraceAction::Branch => "branch",
                TraceAction::Cut => "cut",
                TraceAction::Prune => "prune",
                TraceAction::Incumbent => "incumbent",
                TraceAction::Unbounded => "unbounded",
            }
            .to_string(),
            lp_z: e.lp_z,
            reason: e.reason.clone(),
            fractional: e.fractional.clone(),
        })
        .collect()
}

fn haversine_miles(a_lat: f64, a_lon: f64, b_lat: f64, b_lon: f64) -> f64 {
    let d_lat = (b_lat - a_lat).to_radians();
    let d_lon = (b_lon - a_lon).to_radians();
    let lat1 = a_lat.to_radians();
    let lat2 = b_lat.to_radians();
    let h = (d_lat / 2.0).sin().powi(2) + lat1.cos() * lat2.cos() * (d_lon / 2.0).sin().powi(2);
    2.0 * EARTH_RADIUS_MILES * h.sqrt().asin()
}

fn render_delivery_animation(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    route: &[usize],
    visits: &[DeliveryVisit],
    legs: &[DeliveryLeg],
) -> Animation {
    let mut ordered_nodes = vec![0usize];
    ordered_nodes.extend(route.iter().map(|idx| idx + 1));
    ordered_nodes.push(0);
    let bounds = map_bounds(nodes);
    let mut frames = Vec::new();
    let mut tick = 0.0;
    for leg_idx in 0..ordered_nodes.len().saturating_sub(1) {
        let from = ordered_nodes[leg_idx];
        let to = ordered_nodes[leg_idx + 1];
        for step in 0..8 {
            let alpha = step as f64 / 8.0;
            let x = nodes[from].lon + (nodes[to].lon - nodes[from].lon) * alpha;
            let y = nodes[from].lat + (nodes[to].lat - nodes[from].lat) * alpha;
            let caption = if to == 0 {
                format!("returning to {}", nodes[0].label)
            } else {
                format!(
                    "leg {}: {} -> {}",
                    leg_idx + 1,
                    nodes[from].label,
                    nodes[to].label
                )
            };
            frames.push(
                build_delivery_frame(
                    req,
                    nodes,
                    route,
                    visits,
                    legs,
                    bounds,
                    leg_idx,
                    (x, y),
                    caption,
                )
                .into_frame(tick, tick),
            );
            tick += 1.0;
        }
    }
    Animation {
        width: STAGE_W,
        height: STAGE_H,
        fps: 6.0,
        title: Some("Delivery time-window route".to_string()),
        subtitle: Some(format!("{} stops · {}", route.len(), req.user_id)),
        frames,
        charts: None,
        background: Some("#eef3f6".to_string()),
    }
}

#[allow(clippy::too_many_arguments)]
fn build_delivery_frame(
    req: &DeliveryPlannerRequest,
    nodes: &[StopNode],
    route: &[usize],
    visits: &[DeliveryVisit],
    legs: &[DeliveryLeg],
    bounds: (f64, f64, f64, f64),
    active_leg: usize,
    vehicle_lon_lat: (f64, f64),
    caption: String,
) -> FrameParts {
    let mut shapes = Vec::new();
    shapes.push(Shape::Rect(RectShape {
        x: 0.0,
        y: 0.0,
        w: STAGE_W,
        h: STAGE_H,
        fill: "#eef3f6".to_string(),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: 28.0,
        y: 28.0,
        w: 760.0,
        h: 624.0,
        fill: "#fbfdff".to_string(),
        stroke: Some("#c6d3dd".to_string()),
        stroke_width: Some(1.0),
        rx: Some(8.0),
        ..Default::default()
    }));
    shapes.push(Shape::Rect(RectShape {
        x: 812.0,
        y: 28.0,
        w: 320.0,
        h: 624.0,
        fill: "#ffffff".to_string(),
        stroke: Some("#c6d3dd".to_string()),
        stroke_width: Some(1.0),
        rx: Some(8.0),
        ..Default::default()
    }));
    let mut ordered_nodes = vec![0usize];
    ordered_nodes.extend(route.iter().map(|idx| idx + 1));
    ordered_nodes.push(0);
    for leg in 0..ordered_nodes.len().saturating_sub(1) {
        let a = project(
            nodes[ordered_nodes[leg]].lon,
            nodes[ordered_nodes[leg]].lat,
            bounds,
        );
        let b = project(
            nodes[ordered_nodes[leg + 1]].lon,
            nodes[ordered_nodes[leg + 1]].lat,
            bounds,
        );
        shapes.push(Shape::Line(LineShape {
            x1: a.0,
            y1: a.1,
            x2: b.0,
            y2: b.1,
            stroke: if leg < active_leg {
                "#2e7d59".to_string()
            } else if leg == active_leg {
                "#1f6feb".to_string()
            } else {
                "#99a9b5".to_string()
            },
            stroke_width: Some(if leg == active_leg { 4.0 } else { 2.2 }),
            opacity: Some(if leg <= active_leg { 0.95 } else { 0.55 }),
            dasharray: (leg > active_leg).then(|| "7,5".to_string()),
            ..Default::default()
        }));
    }
    for (idx, node) in nodes.iter().enumerate() {
        let p = project(node.lon, node.lat, bounds);
        let is_depot = idx == 0;
        let done = idx > 0
            && route
                .iter()
                .position(|&r| r + 1 == idx)
                .map(|pos| pos < active_leg)
                .unwrap_or(false);
        shapes.push(Shape::Circle(CircleShape {
            x: p.0,
            y: p.1,
            r: if is_depot { 12.0 } else { 9.0 },
            fill: if is_depot {
                "#243447".to_string()
            } else if done {
                "#2e7d59".to_string()
            } else {
                "#f3a712".to_string()
            },
            stroke: Some("#ffffff".to_string()),
            stroke_width: Some(2.0),
            title: Some(node.label.clone()),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: p.0 + 13.0,
            y: p.1 - 10.0,
            text: node.label.clone(),
            font_size: Some(12.0),
            fill: Some("#243447".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
    }
    let vehicle = project(vehicle_lon_lat.0, vehicle_lon_lat.1, bounds);
    shapes.push(Shape::Circle(CircleShape {
        x: vehicle.0,
        y: vehicle.1,
        r: 13.0,
        fill: "#1f6feb".to_string(),
        stroke: Some("#ffffff".to_string()),
        stroke_width: Some(3.0),
        title: Some("vehicle".to_string()),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 50.0,
        y: 58.0,
        text: format!(
            "Depart {} · {:.1} mph · {} stops",
            format_minutes(req.depart_time),
            req.average_speed_mph,
            route.len()
        ),
        font_size: Some(16.0),
        fill: Some("#243447".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    shapes.push(Shape::Text(TextShape {
        x: 832.0,
        y: 60.0,
        text: "Itinerary".to_string(),
        font_size: Some(16.0),
        fill: Some("#243447".to_string()),
        font_weight: Some(FontWeight::Bold),
        ..Default::default()
    }));
    let mut y = 92.0;
    for (i, visit) in visits.iter().take(10).enumerate() {
        let active = i == active_leg && active_leg < visits.len();
        shapes.push(Shape::Rect(RectShape {
            x: 832.0,
            y: y - 18.0,
            w: 280.0,
            h: 44.0,
            fill: if active { "#eaf2ff" } else { "#f7fafc" }.to_string(),
            stroke: Some(if active { "#1f6feb" } else { "#d8e1e8" }.to_string()),
            stroke_width: Some(1.0),
            rx: Some(6.0),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: 846.0,
            y,
            text: format!("{}. {}  {}", i + 1, visit.label, visit.arrival_text),
            font_size: Some(12.0),
            fill: Some("#243447".to_string()),
            font_weight: Some(FontWeight::Bold),
            ..Default::default()
        }));
        shapes.push(Shape::Text(TextShape {
            x: 846.0,
            y: y + 17.0,
            text: format!(
                "{}  {:.1} mi",
                visit.window_text, visit.distance_from_previous
            ),
            font_size: Some(11.0),
            fill: Some("#62727f".to_string()),
            ..Default::default()
        }));
        y += 54.0;
    }
    if let Some(leg) = legs.get(active_leg) {
        shapes.push(Shape::Text(TextShape {
            x: 50.0,
            y: 626.0,
            text: format!(
                "{} -> {}   {:.2} mi / {:.0} min",
                leg.from, leg.to, leg.distance, leg.travel_minutes
            ),
            font_size: Some(13.0),
            fill: Some("#334456".to_string()),
            ..Default::default()
        }));
    }
    shapes.push(Shape::Text(TextShape {
        x: 1112.0,
        y: 632.0,
        text: caption.clone(),
        font_size: Some(11.0),
        fill: Some("#62727f".to_string()),
        anchor: Some(Anchor::End),
        ..Default::default()
    }));
    FrameParts::with_caption(shapes, caption)
}

fn map_bounds(nodes: &[StopNode]) -> (f64, f64, f64, f64) {
    let min_lat = nodes.iter().map(|n| n.lat).fold(f64::INFINITY, f64::min);
    let max_lat = nodes
        .iter()
        .map(|n| n.lat)
        .fold(f64::NEG_INFINITY, f64::max);
    let min_lon = nodes.iter().map(|n| n.lon).fold(f64::INFINITY, f64::min);
    let max_lon = nodes
        .iter()
        .map(|n| n.lon)
        .fold(f64::NEG_INFINITY, f64::max);
    let lat_pad = ((max_lat - min_lat).abs() * 0.16).max(0.01);
    let lon_pad = ((max_lon - min_lon).abs() * 0.16).max(0.01);
    (
        min_lon - lon_pad,
        max_lon + lon_pad,
        min_lat - lat_pad,
        max_lat + lat_pad,
    )
}

fn project(lon: f64, lat: f64, bounds: (f64, f64, f64, f64)) -> (f64, f64) {
    let (min_lon, max_lon, min_lat, max_lat) = bounds;
    let x = 64.0 + (lon - min_lon) / (max_lon - min_lon).max(1e-9) * 688.0;
    let y = 612.0 - (lat - min_lat) / (max_lat - min_lat).max(1e-9) * 512.0;
    (x, y)
}

pub fn empty_animation() -> Animation {
    Animation {
        width: STAGE_W,
        height: STAGE_H,
        fps: 6.0,
        title: Some("Delivery time-window route".to_string()),
        subtitle: None,
        frames: vec![Frame {
            t: 0.0,
            tick: 0.0,
            shapes: vec![Shape::Text(TextShape {
                x: STAGE_W / 2.0,
                y: STAGE_H / 2.0,
                text: "No route".to_string(),
                font_size: Some(18.0),
                fill: Some("#62727f".to_string()),
                anchor: Some(Anchor::Middle),
                ..Default::default()
            })],
            caption: Some("No route".to_string()),
        }],
        charts: None,
        background: Some("#eef3f6".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::delivery_planner::model::{
        default_delivery_request, DeliveryLockedPosition, DeliveryObjectiveMode,
    };
    use crate::des::general::ip_mip_des::IPMIPStatus;

    #[test]
    fn default_delivery_solves_with_in_house_model_or_fallback() {
        let req = default_delivery_request();
        let resp = solve_delivery_planner_summary(&req);
        assert!(resp.ok, "{:?}", resp.error);
        assert_eq!(resp.route.len(), req.stops.len());
        assert!(resp.num_variables > req.stops.len());
        assert!(resp.num_constraints > req.stops.len());
        assert!(resp.in_house_only);
        assert!(resp.itinerary_text.contains("Delivery itinerary"));
    }

    #[test]
    fn compact_two_stop_case_is_ip_mip_optimal() {
        let mut req = default_delivery_request();
        req.stops.truncate(2);
        req.stops[0].window_start = 8 * 60;
        req.stops[0].window_end = 12 * 60;
        req.stops[1].window_start = 8 * 60;
        req.stops[1].window_end = 12 * 60;
        let resp = solve_delivery_planner_summary(&req);
        assert!(resp.ok, "{:?}", resp.error);
        assert!(!resp.used_fallback, "{}", resp.solver_status);
        assert_eq!(resp.solver_status, IPMIPStatus::Optimal.as_str());
    }

    #[test]
    fn impossible_window_reports_error() {
        let mut req = default_delivery_request();
        req.stops.truncate(1);
        req.stops[0].window_start = 1;
        req.stops[0].window_end = 2;
        let resp = solve_delivery_planner_summary(&req);
        assert!(!resp.ok);
    }

    #[test]
    fn locked_order_uses_ordered_stop_ids_as_hard_rule() {
        let mut req = default_delivery_request();
        req.stops.truncate(3);
        for stop in &mut req.stops {
            stop.window_start = 8 * 60;
            stop.window_end = 18 * 60;
        }
        req.route_rules.locked_order = true;
        req.route_rules.ordered_stop_ids =
            vec!["S3".to_string(), "S1".to_string(), "S2".to_string()];

        let resp = solve_delivery_planner_summary(&req);

        assert!(resp.ok, "{:?}", resp.error);
        assert_eq!(resp.route, vec![2, 0, 1]);
        assert_eq!(resp.solver_kind, "position-locked-route-search");
        assert!(resp
            .solver_notes
            .iter()
            .any(|n| n.contains("Locked stop order")));
    }

    #[test]
    fn locked_position_fixes_only_that_route_slot() {
        let mut req = default_delivery_request();
        req.stops.truncate(3);
        for stop in &mut req.stops {
            stop.window_start = 8 * 60;
            stop.window_end = 18 * 60;
        }
        req.route_rules.locked_positions = vec![DeliveryLockedPosition {
            stop_id: "S3".to_string(),
            position: 1,
        }];

        let resp = solve_delivery_planner_summary(&req);

        assert!(resp.ok, "{:?}", resp.error);
        assert_eq!(resp.route.len(), 3);
        assert_eq!(resp.route[1], 2);
        assert_eq!(resp.solver_kind, "position-locked-route-search");
        assert!(resp
            .solver_notes
            .iter()
            .any(|n| n.contains("unlocked rows remained optimizable")));
    }

    #[test]
    fn objective_modes_are_reported_and_scored() {
        let mut req = default_delivery_request();
        req.stops.truncate(2);
        for stop in &mut req.stops {
            stop.window_start = 8 * 60;
            stop.window_end = 12 * 60;
        }

        req.objective_mode = DeliveryObjectiveMode::TravelTime;
        let travel = solve_delivery_planner_summary(&req);
        assert!(travel.ok, "{:?}", travel.error);
        assert_eq!(travel.objective_mode, DeliveryObjectiveMode::TravelTime);
        assert!(
            (travel.objective_value - (travel.total_travel_minutes + travel.window_edge_penalty))
                .abs()
                < 1e-6
        );

        req.objective_mode = DeliveryObjectiveMode::WindowCenter;
        let centered = solve_delivery_planner_summary(&req);
        assert!(centered.ok, "{:?}", centered.error);
        assert_eq!(centered.objective_mode, DeliveryObjectiveMode::WindowCenter);
        assert!(centered.window_center_penalty >= 0.0);
        assert!(
            (centered.objective_value
                - (centered.total_travel_minutes + centered.window_center_penalty))
                .abs()
                < 1e-6
        );
    }
}
