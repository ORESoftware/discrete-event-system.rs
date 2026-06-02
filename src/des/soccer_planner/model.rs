//! JSON request/response types for the interactive soccer rotation planner.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::des::general::soccer_rotation::{
    default_outfield_formation, formation_position_names, formation_with_gk,
    parse_outfield_formation, PlayerStatus, PositionSynergyRule,
};

/// One player row in the planner UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlannerPlayer {
    pub id: usize,
    pub name: String,
    /// `"available" | "awol" | "injured" | "guest"`.
    pub status: String,
    /// Per-position score in `[0, 1]` (index = position slot).
    pub position_scores: Vec<f64>,
    /// Positions this player must never play (by index).
    #[serde(default)]
    pub banned_positions: Vec<usize>,
    /// If set, player is locked to this position index for the whole match.
    pub fixed_position: Option<usize>,
}

/// Chemistry rule: conditional position score.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlannerSynergy {
    pub player: usize,
    pub position: usize,
    pub partner_player: usize,
    pub partner_position: usize,
    pub score_with: f64,
    pub score_without: f64,
}

/// Full planner configuration POSTed from the UI.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlannerRequest {
    /// Outfield formation excluding GK, e.g. `[4, 4, 2]` for 11-a-side.
    #[serde(default = "default_outfield")]
    pub outfield_formation: Vec<usize>,
    #[serde(default = "default_periods")]
    pub num_periods: usize,
    #[serde(default = "default_minutes")]
    pub minutes_per_period: usize,
    #[serde(default = "default_max_subs")]
    pub max_subs_per_game: usize,
    #[serde(default = "default_min_subs")]
    pub min_subs_per_game: usize,
    #[serde(default = "default_stamina")]
    pub max_consecutive_on_field: usize,
    pub players: Vec<PlannerPlayer>,
    #[serde(default)]
    pub synergies: Vec<PlannerSynergy>,
    #[serde(default = "default_seed")]
    pub seed: u32,
    #[serde(default = "default_solver_time_limit_ms")]
    pub solver_time_limit_ms: f64,
    #[serde(default = "default_solver_max_nodes")]
    pub solver_max_nodes: usize,
    #[serde(default = "default_solver_max_ticks")]
    pub solver_max_ticks: usize,
    #[serde(default = "default_solver_lp_max_iters")]
    pub solver_lp_max_iters: usize,
    #[serde(default = "default_solver_heuristic_passes")]
    pub solver_heuristic_passes: usize,
    #[serde(default = "default_fallback_to_mdp")]
    pub fallback_to_mdp: bool,
}

fn default_outfield() -> Vec<usize> {
    default_outfield_formation(11)
}
fn default_periods() -> usize {
    2
}
fn default_minutes() -> usize {
    45
}
fn default_max_subs() -> usize {
    7
}
fn default_min_subs() -> usize {
    0
}
fn default_stamina() -> usize {
    3
}
fn default_seed() -> u32 {
    4242
}
fn default_solver_time_limit_ms() -> f64 {
    120_000.0
}
fn default_solver_max_nodes() -> usize {
    20_000
}
fn default_solver_max_ticks() -> usize {
    200_000
}
fn default_solver_lp_max_iters() -> usize {
    6_000
}
fn default_solver_heuristic_passes() -> usize {
    120
}
fn default_fallback_to_mdp() -> bool {
    true
}

impl Default for PlannerRequest {
    fn default() -> Self {
        default_planner_request()
    }
}

/// Default 11-a-side squad (18 players = 11 on field + 7 bench slots).
pub fn default_planner_request() -> PlannerRequest {
    let outfield = default_outfield_formation(11);
    let formation = formation_with_gk(&outfield);
    let num_positions = formation.iter().sum::<usize>();
    let position_names = formation_position_names(&formation);
    let num_players = 18;

    let mut players: Vec<PlannerPlayer> = Vec::new();
    for p in 0..num_players {
        let mut scores = vec![0.35; num_positions];
        // Give each player a plausible best position.
        let best = p % num_positions;
        scores[best] = 0.88;
        if best + 1 < num_positions {
            scores[best + 1] = 0.62;
        }
        if best > 0 {
            scores[best - 1] = 0.58;
        }
        // GK slot only for player 0 by default.
        if p == 0 {
            scores.fill(0.2);
            scores[0] = 0.95;
        }
        players.push(PlannerPlayer {
            id: p,
            name: format!("Player{}", p + 1),
            status: "available".to_string(),
            position_scores: scores,
            banned_positions: if p == 0 {
                (1..num_positions).collect()
            } else {
                vec![0]
            },
            fixed_position: if p == 0 { Some(0) } else { None },
        });
    }

    // Example chemistry: ST (Player10 @ pos 10) better with creative mid (Player8 @ CM).
    let st_pos = position_names.iter().position(|n| n == "ST").unwrap_or(10);
    let cm_pos = position_names.iter().position(|n| n == "CM").unwrap_or(6);
    let synergies = vec![PlannerSynergy {
        player: 9,
        position: st_pos,
        partner_player: 7,
        partner_position: cm_pos,
        score_with: 0.92,
        score_without: 0.78,
    }];

    PlannerRequest {
        outfield_formation: outfield,
        num_periods: 2,
        minutes_per_period: 45,
        max_subs_per_game: 7,
        min_subs_per_game: 0,
        max_consecutive_on_field: 3,
        players,
        synergies,
        seed: 4242,
        solver_time_limit_ms: default_solver_time_limit_ms(),
        solver_max_nodes: default_solver_max_nodes(),
        solver_max_ticks: default_solver_max_ticks(),
        solver_lp_max_iters: default_solver_lp_max_iters(),
        solver_heuristic_passes: default_solver_heuristic_passes(),
        fallback_to_mdp: default_fallback_to_mdp(),
    }
}

pub fn parse_player_status(s: &str) -> PlayerStatus {
    match s.to_lowercase().as_str() {
        "awol" => PlayerStatus::Awol,
        "injured" => PlayerStatus::Injured,
        "guest" => PlayerStatus::Guest,
        _ => PlayerStatus::Available,
    }
}

pub fn synergy_to_rule(s: &PlannerSynergy) -> PositionSynergyRule {
    PositionSynergyRule {
        player: s.player,
        position: s.position,
        partner_player: s.partner_player,
        partner_position: s.partner_position,
        score_with: s.score_with.clamp(0.0, 1.0),
        score_without: s.score_without.clamp(0.0, 1.0),
    }
}

pub fn planner_position_count(req: &PlannerRequest) -> usize {
    formation_with_gk(&req.outfield_formation).iter().sum()
}

pub fn normalize_planner_request(req: &mut PlannerRequest) {
    if req.outfield_formation.is_empty() || req.outfield_formation.iter().any(|&n| n == 0) {
        req.outfield_formation = default_outfield_formation(11);
    }
    req.num_periods = req.num_periods.max(1);
    req.minutes_per_period = req.minutes_per_period.max(1);
    if req.min_subs_per_game > req.max_subs_per_game {
        req.min_subs_per_game = req.max_subs_per_game;
    }
    req.max_consecutive_on_field = req.max_consecutive_on_field.max(1);
    req.solver_time_limit_ms = req.solver_time_limit_ms.max(250.0);
    req.solver_max_nodes = req.solver_max_nodes.max(1);
    req.solver_max_ticks = req.solver_max_ticks.max(1);
    req.solver_lp_max_iters = req.solver_lp_max_iters.max(1);
    req.solver_heuristic_passes = req.solver_heuristic_passes.max(1);

    let n_pos = planner_position_count(req);
    for (i, player) in req.players.iter_mut().enumerate() {
        player.id = i;
        while player.position_scores.len() < n_pos {
            player.position_scores.push(0.35);
        }
        player.position_scores.truncate(n_pos);
        player
            .position_scores
            .iter_mut()
            .for_each(|s| *s = s.clamp(0.0, 1.0));
        player.banned_positions.sort_unstable();
        player.banned_positions.dedup();
        player.banned_positions.retain(|&p| p < n_pos);
        if player.fixed_position.map(|p| p >= n_pos).unwrap_or(false) {
            player.fixed_position = None;
        }
    }
    req.synergies.retain(|s| {
        s.player < req.players.len()
            && s.partner_player < req.players.len()
            && s.position < n_pos
            && s.partner_position < n_pos
    });
    for s in &mut req.synergies {
        s.score_with = s.score_with.clamp(0.0, 1.0);
        s.score_without = s.score_without.clamp(0.0, 1.0);
    }
}

fn value_usize(command: &Value, key: &str) -> Option<usize> {
    command.get(key).and_then(Value::as_u64).map(|n| n as usize)
}

fn value_f64(command: &Value, key: &str) -> Option<f64> {
    command.get(key).and_then(Value::as_f64)
}

fn value_string(command: &Value, key: &str) -> Option<String> {
    command.get(key).and_then(Value::as_str).map(str::to_string)
}

fn value_usize_vec(command: &Value, key: &str) -> Option<Vec<usize>> {
    command.get(key)?.as_array().map(|items| {
        items
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect()
    })
}

fn value_f64_vec(command: &Value, key: &str) -> Option<Vec<f64>> {
    command.get(key)?.as_array().map(|items| {
        items
            .iter()
            .map(|x| x.as_f64().unwrap_or(0.35).clamp(0.0, 1.0))
            .collect()
    })
}

fn parse_formation_value(command: &Value) -> Option<Vec<usize>> {
    if let Some(v) = value_usize_vec(command, "outfieldFormation") {
        return Some(v);
    }
    value_string(command, "formation").and_then(|s| parse_outfield_formation(&s))
}

fn player_mut(req: &mut PlannerRequest, id: usize) -> Result<&mut PlannerPlayer, String> {
    req.players
        .iter_mut()
        .find(|p| p.id == id)
        .ok_or_else(|| format!("player id {id} not found"))
}

fn default_new_player(req: &PlannerRequest, name: String, status: String) -> PlannerPlayer {
    let n_pos = planner_position_count(req);
    PlannerPlayer {
        id: req.players.len(),
        name,
        status,
        position_scores: vec![0.5; n_pos],
        banned_positions: Vec::new(),
        fixed_position: None,
    }
}

fn remove_player_and_rewrite_refs(req: &mut PlannerRequest, id: usize) -> Result<(), String> {
    let pos = req
        .players
        .iter()
        .position(|p| p.id == id)
        .ok_or_else(|| format!("player id {id} not found"))?;
    req.players.remove(pos);
    req.synergies
        .retain(|s| s.player != id && s.partner_player != id);
    for s in &mut req.synergies {
        if s.player > id {
            s.player -= 1;
        }
        if s.partner_player > id {
            s.partner_player -= 1;
        }
    }
    Ok(())
}

/// Apply one JSON command to a planner request. This is the planner-specific
/// stream-edit layer: the command stream mutates roster variables and
/// constraint rows before a solver endpoint or streaming model asks for `solve`.
pub fn apply_planner_stream_command(
    req: &mut PlannerRequest,
    command: &Value,
) -> Result<Value, String> {
    let op = command.get("op").and_then(Value::as_str).unwrap_or("");
    match op {
        "reset" => {
            *req = default_planner_request();
        }
        "set_match" => {
            if let Some(v) = value_usize(command, "numPeriods") {
                req.num_periods = v;
            }
            if let Some(v) = value_usize(command, "minutesPerPeriod") {
                req.minutes_per_period = v;
            }
            if let Some(v) = value_usize(command, "maxSubsPerGame") {
                req.max_subs_per_game = v;
            }
            if let Some(v) = value_usize(command, "minSubsPerGame") {
                req.min_subs_per_game = v;
            }
            if let Some(v) = value_usize(command, "maxConsecutiveOnField") {
                req.max_consecutive_on_field = v;
            }
            if let Some(v) = value_usize(command, "seed") {
                req.seed = v as u32;
            }
            if let Some(v) = value_f64(command, "solverTimeLimitMs") {
                req.solver_time_limit_ms = v;
            }
            if let Some(v) = value_usize(command, "solverMaxNodes") {
                req.solver_max_nodes = v;
            }
            if let Some(v) = value_usize(command, "solverMaxTicks") {
                req.solver_max_ticks = v;
            }
            if let Some(v) = value_usize(command, "solverLpMaxIters") {
                req.solver_lp_max_iters = v;
            }
            if let Some(v) = value_usize(command, "solverHeuristicPasses") {
                req.solver_heuristic_passes = v;
            }
            if let Some(v) = command.get("fallbackToMdp").and_then(Value::as_bool) {
                req.fallback_to_mdp = v;
            }
        }
        "set_formation" => {
            req.outfield_formation = parse_formation_value(command).ok_or_else(|| {
                "set_formation needs `formation` or `outfieldFormation`".to_string()
            })?;
        }
        "add_player" | "add_variable" => {
            let mut player = if let Some(raw) = command.get("player") {
                serde_json::from_value::<PlannerPlayer>(raw.clone())
                    .map_err(|e| format!("invalid player: {e}"))?
            } else {
                default_new_player(
                    req,
                    value_string(command, "name")
                        .unwrap_or_else(|| format!("Player{}", req.players.len() + 1)),
                    value_string(command, "status").unwrap_or_else(|| "guest".to_string()),
                )
            };
            player.id = req.players.len();
            if let Some(scores) = value_f64_vec(command, "positionScores") {
                player.position_scores = scores;
            }
            if let Some(banned) = value_usize_vec(command, "bannedPositions") {
                player.banned_positions = banned;
            }
            if command.get("fixedPosition").is_some() {
                player.fixed_position = command
                    .get("fixedPosition")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize);
            }
            req.players.push(player);
        }
        "remove_player" | "remove_variable" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "remove_player needs `id`".to_string())?;
            remove_player_and_rewrite_refs(req, id)?;
        }
        "set_player" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "set_player needs `id`".to_string())?;
            let player = player_mut(req, id)?;
            if let Some(name) = value_string(command, "name") {
                player.name = name;
            }
            if let Some(status) = value_string(command, "status") {
                player.status = status;
            }
            if let Some(scores) = value_f64_vec(command, "positionScores") {
                player.position_scores = scores;
            }
            if let Some(banned) = value_usize_vec(command, "bannedPositions") {
                player.banned_positions = banned;
            }
            if command.get("fixedPosition").is_some() {
                player.fixed_position = command
                    .get("fixedPosition")
                    .and_then(Value::as_u64)
                    .map(|n| n as usize);
            }
        }
        "set_player_status" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "set_player_status needs `id`".to_string())?;
            let status = value_string(command, "status")
                .ok_or_else(|| "set_player_status needs `status`".to_string())?;
            player_mut(req, id)?.status = status;
        }
        "rename_player" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "rename_player needs `id`".to_string())?;
            let name = value_string(command, "name")
                .ok_or_else(|| "rename_player needs `name`".to_string())?;
            player_mut(req, id)?.name = name;
        }
        "set_position_score" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "set_position_score needs `id`".to_string())?;
            let pos = value_usize(command, "position")
                .ok_or_else(|| "set_position_score needs `position`".to_string())?;
            let score = value_f64(command, "score")
                .ok_or_else(|| "set_position_score needs `score`".to_string())?;
            let player = player_mut(req, id)?;
            if pos >= player.position_scores.len() {
                return Err(format!("position {pos} out of range"));
            }
            player.position_scores[pos] = score.clamp(0.0, 1.0);
        }
        "set_position_scores" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "set_position_scores needs `id`".to_string())?;
            let scores = value_f64_vec(command, "scores")
                .or_else(|| value_f64_vec(command, "positionScores"))
                .ok_or_else(|| "set_position_scores needs `scores`".to_string())?;
            player_mut(req, id)?.position_scores = scores;
        }
        "ban_position" | "add_constraint" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "ban_position needs `id`/`player`".to_string())?;
            let pos = value_usize(command, "position")
                .ok_or_else(|| "ban_position needs `position`".to_string())?;
            let player = player_mut(req, id)?;
            if !player.banned_positions.contains(&pos) {
                player.banned_positions.push(pos);
            }
        }
        "allow_position" | "remove_constraint" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "allow_position needs `id`/`player`".to_string())?;
            let pos = value_usize(command, "position")
                .ok_or_else(|| "allow_position needs `position`".to_string())?;
            let player = player_mut(req, id)?;
            player.banned_positions.retain(|&p| p != pos);
        }
        "set_fixed_position" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "set_fixed_position needs `id`/`player`".to_string())?;
            let fixed = command
                .get("position")
                .and_then(Value::as_u64)
                .map(|n| n as usize);
            player_mut(req, id)?.fixed_position = fixed;
        }
        "clear_fixed_position" => {
            let id = value_usize(command, "id")
                .or_else(|| value_usize(command, "player"))
                .ok_or_else(|| "clear_fixed_position needs `id`/`player`".to_string())?;
            player_mut(req, id)?.fixed_position = None;
        }
        "add_synergy" => {
            let synergy = if let Some(raw) = command.get("synergy") {
                serde_json::from_value::<PlannerSynergy>(raw.clone())
                    .map_err(|e| format!("invalid synergy: {e}"))?
            } else {
                PlannerSynergy {
                    player: value_usize(command, "player")
                        .ok_or_else(|| "add_synergy needs `player`".to_string())?,
                    position: value_usize(command, "position")
                        .ok_or_else(|| "add_synergy needs `position`".to_string())?,
                    partner_player: value_usize(command, "partnerPlayer")
                        .ok_or_else(|| "add_synergy needs `partnerPlayer`".to_string())?,
                    partner_position: value_usize(command, "partnerPosition")
                        .ok_or_else(|| "add_synergy needs `partnerPosition`".to_string())?,
                    score_with: value_f64(command, "scoreWith").unwrap_or(0.9),
                    score_without: value_f64(command, "scoreWithout").unwrap_or(0.7),
                }
            };
            req.synergies.push(synergy);
        }
        "remove_synergy" => {
            let index = value_usize(command, "index")
                .ok_or_else(|| "remove_synergy needs `index`".to_string())?;
            if index >= req.synergies.len() {
                return Err(format!("synergy index {index} out of range"));
            }
            req.synergies.remove(index);
        }
        "clear_synergies" => {
            req.synergies.clear();
        }
        "solve" | "snapshot" => {}
        other => return Err(format!("unknown planner stream op `{other}`")),
    }
    normalize_planner_request(req);
    Ok(json!({
        "event": "applied",
        "op": op,
        "players": req.players.len(),
        "positions": planner_position_count(req),
        "periods": req.num_periods,
        "synergies": req.synergies.len()
    }))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{
        apply_planner_stream_command, default_planner_request, normalize_planner_request,
        planner_position_count,
    };

    #[test]
    fn streamed_new_players_default_to_all_positions_allowed() {
        let mut req = default_planner_request();
        req.outfield_formation = vec![2, 3, 1];
        req.players.clear();
        req.synergies.clear();
        normalize_planner_request(&mut req);

        for i in 0..11 {
            apply_planner_stream_command(
                &mut req,
                &json!({
                    "op": "add_player",
                    "name": format!("Player{}", i + 1),
                    "status": "available"
                }),
            )
            .unwrap();
        }

        assert_eq!(planner_position_count(&req), 7);
        assert_eq!(req.players.len(), 11);
        assert!(req
            .players
            .iter()
            .all(|p| p.fixed_position.is_none() && p.banned_positions.is_empty()));

        let problem = crate::des::soccer_planner::solve::build_problem_from_request(&req).unwrap();
        let banned = problem.banned_positions.unwrap();
        assert!(banned.iter().all(|row| !row[0]));
    }
}
