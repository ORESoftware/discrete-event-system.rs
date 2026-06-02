//! 2D 11v11 soccer simulation prototype.
//!
//! This module is the live-match counterpart to [`soccer_rotation`]: it models a
//! full pitch, 22 player agents, three officials, soft real-time human input
//! queues, simple ball physics, and MDP/POMDP-shaped decision surfaces. The
//! simulation itself remains single threaded; external controller threads can
//! push [`HumanInputFrame`] values into [`SharedHumanInputs`] between ticks.

#![allow(dead_code)]

use std::collections::{HashMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::des::general::general::fisher_yates_shuffle;
use crate::des::general::prng::{mulberry32, SeededRandom};
use crate::des::shared::capabilities::RandomSource;

pub const DEFAULT_DT_SECONDS: f64 = 0.1;
pub const DEFAULT_DURATION_SECONDS: f64 = 10.0 * 60.0;
pub const DEFAULT_FIELD_LENGTH_YARDS: f64 = 120.0;
pub const DEFAULT_FIELD_WIDTH_YARDS: f64 = 80.0;
pub const DEFAULT_GOAL_WIDTH_YARDS: f64 = 8.0;
pub const DEFAULT_BALL_DRAG_PER_TICK: f64 = 0.015;
pub const DEFAULT_BALL_STOP_SPEED_YPS: f64 = 0.35;
const PLAYER_CONTROL_RADIUS_YARDS: f64 = 1.55;
const PLAYER_BODY_RADIUS_YARDS: f64 = 0.78;
const PLAYER_COLLISION_DAMPING: f64 = 0.34;
const SHOT_SAVE_DEPTH_YARDS: f64 = 1.6;
const BALL_AGENT_ID: usize = 25;
const PLAYER_POSITION_HISTORY_LIMIT: usize = 50;
const BALL_POSITION_HISTORY_LIMIT: usize = 50;

fn default_ball_drag_per_tick() -> f64 {
    DEFAULT_BALL_DRAG_PER_TICK
}

fn default_ball_stop_speed_yps() -> f64 {
    DEFAULT_BALL_STOP_SPEED_YPS
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

impl Vec2 {
    pub fn new(x: f64, y: f64) -> Self {
        Vec2 { x, y }
    }

    pub fn zero() -> Self {
        Vec2 { x: 0.0, y: 0.0 }
    }

    pub fn len(self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    pub fn distance(self, other: Vec2) -> f64 {
        (self - other).len()
    }

    pub fn normalized(self) -> Vec2 {
        let l = self.len();
        if l <= 1e-9 {
            Vec2::zero()
        } else {
            self / l
        }
    }

    pub fn clamp_to_pitch(self, field_width: f64, field_length: f64) -> Vec2 {
        Vec2 {
            x: self.x.max(0.0).min(field_width),
            y: self.y.max(0.0).min(field_length),
        }
    }
}

impl std::ops::Add for Vec2 {
    type Output = Vec2;

    fn add(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl std::ops::AddAssign for Vec2 {
    fn add_assign(&mut self, rhs: Vec2) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl std::ops::Sub for Vec2 {
    type Output = Vec2;

    fn sub(self, rhs: Vec2) -> Self::Output {
        Vec2::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl std::ops::Mul<f64> for Vec2 {
    type Output = Vec2;

    fn mul(self, rhs: f64) -> Self::Output {
        Vec2::new(self.x * rhs, self.y * rhs)
    }
}

impl std::ops::Div<f64> for Vec2 {
    type Output = Vec2;

    fn div(self, rhs: f64) -> Self::Output {
        Vec2::new(self.x / rhs, self.y / rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Team {
    Home,
    Away,
}

impl Team {
    pub fn other(self) -> Team {
        match self {
            Team::Home => Team::Away,
            Team::Away => Team::Home,
        }
    }

    pub fn attack_dir(self) -> f64 {
        match self {
            Team::Home => 1.0,
            Team::Away => -1.0,
        }
    }

    pub fn goal_y(self, field_length: f64) -> f64 {
        match self {
            Team::Home => field_length,
            Team::Away => 0.0,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Team::Home => "Home",
            Team::Away => "Away",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PlayerRole {
    Goalkeeper,
    Defender,
    Midfielder,
    Forward,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OfficialKind {
    CenterReferee,
    AssistantRefereeNear,
    AssistantRefereeFar,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillProfile {
    pub top_speed_yps: f64,
    pub acceleration_yps2: f64,
    pub shooting: f64,
    pub passing: f64,
    pub dribbling: f64,
    pub first_touch: f64,
    pub defending: f64,
    pub stamina: f64,
    pub decision_noise: f64,
    pub aggression: f64,
}

impl SkillProfile {
    fn blended(seed: usize, role: PlayerRole, rng: &mut SeededRandom) -> Self {
        let role_bias = match role {
            PlayerRole::Goalkeeper => (6.4, 6.1, 0.42, 0.64, 0.38, 0.74, 0.84, 0.78, 0.38),
            PlayerRole::Defender => (7.2, 7.0, 0.52, 0.70, 0.56, 0.66, 0.80, 0.82, 0.66),
            PlayerRole::Midfielder => (7.7, 7.6, 0.65, 0.82, 0.76, 0.76, 0.67, 0.88, 0.58),
            PlayerRole::Forward => (8.0, 7.8, 0.82, 0.72, 0.82, 0.75, 0.48, 0.80, 0.72),
        };
        let jitter = |rng: &mut SeededRandom, scale: f64| (rng.next_float() - 0.5) * scale;
        SkillProfile {
            top_speed_yps: role_bias.0 + jitter(rng, 0.9) + (seed % 3) as f64 * 0.08,
            acceleration_yps2: role_bias.1 + jitter(rng, 0.8),
            shooting: (role_bias.2 + jitter(rng, 0.24)).clamp(0.1, 0.98),
            passing: (role_bias.3 + jitter(rng, 0.22)).clamp(0.1, 0.98),
            dribbling: (role_bias.4 + jitter(rng, 0.22)).clamp(0.1, 0.98),
            first_touch: (role_bias.5 + jitter(rng, 0.20)).clamp(0.1, 0.98),
            defending: (role_bias.6 + jitter(rng, 0.22)).clamp(0.1, 0.98),
            stamina: (role_bias.7 + jitter(rng, 0.18)).clamp(0.2, 0.98),
            decision_noise: (0.08 + jitter(rng, 0.08)).clamp(0.01, 0.18),
            aggression: (role_bias.8 + jitter(rng, 0.22)).clamp(0.1, 0.95),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentPreferences {
    pub shoot_bias: f64,
    pub pass_bias: f64,
    pub dribble_bias: f64,
    pub open_space_bias: f64,
}

impl Default for AgentPreferences {
    fn default() -> Self {
        AgentPreferences {
            shoot_bias: 0.40,
            pass_bias: 0.55,
            dribble_bias: 0.45,
            open_space_bias: 0.70,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerMdpState {
    pub tick: u64,
    pub ball_zone_x: usize,
    pub ball_zone_y: usize,
    pub possession_team: Option<Team>,
    pub score_diff_for_home: i32,
    pub phase: TacticalPhase,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TacticalPhase {
    Kickoff,
    HomeBuildUp,
    AwayBuildUp,
    HomeAttack,
    AwayAttack,
    Transition,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerPomdpObservation {
    pub player_id: usize,
    pub has_ball: bool,
    pub ball_distance: f64,
    pub nearest_opponent_distance: f64,
    pub nearest_teammate_distance: f64,
    pub shot_lane_open: bool,
    pub yards_to_goal: f64,
    pub open_space_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BeliefState {
    pub possession_confidence: f64,
    pub pressure: f64,
    pub pass_lane_open: f64,
    pub shot_quality: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentDecisionTrace {
    pub mdp_state: SoccerMdpState,
    pub observation: SoccerPomdpObservation,
    pub belief: BeliefState,
    pub operation_order: Vec<String>,
    pub action: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningTransition {
    pub tick: u64,
    pub player_id: usize,
    pub team: Team,
    pub role: PlayerRole,
    pub state: SoccerMdpState,
    pub observation: SoccerPomdpObservation,
    pub belief: BeliefState,
    pub action: String,
    pub reward: f64,
    pub next_state: SoccerMdpState,
    pub next_observation: SoccerPomdpObservation,
    pub done: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerQStateKey {
    pub phase: TacticalPhase,
    pub role: PlayerRole,
    pub possession_relative: i8,
    pub ball_zone_x: usize,
    pub ball_zone_y: usize,
    pub score_diff_bucket: i8,
    pub has_ball: bool,
    pub shot_lane_open: bool,
    pub ball_distance_bin: u8,
    pub yards_to_goal_bin: u8,
    pub pressure_bin: u8,
    pub open_space_bin: u8,
}

impl SoccerQStateKey {
    pub fn from_parts(
        state: &SoccerMdpState,
        observation: &SoccerPomdpObservation,
        team: Team,
        role: PlayerRole,
    ) -> Self {
        let possession_relative = match state.possession_team {
            Some(t) if t == team => 1,
            Some(_) => -1,
            None => 0,
        };
        let score_diff_for_team = match team {
            Team::Home => state.score_diff_for_home,
            Team::Away => -state.score_diff_for_home,
        };
        SoccerQStateKey {
            phase: state.phase,
            role,
            possession_relative,
            ball_zone_x: state.ball_zone_x,
            ball_zone_y: state.ball_zone_y,
            score_diff_bucket: score_diff_for_team.clamp(-2, 2) as i8,
            has_ball: observation.has_ball,
            shot_lane_open: observation.shot_lane_open,
            ball_distance_bin: distance_bucket(observation.ball_distance, &[3.0, 8.0, 18.0, 36.0]),
            yards_to_goal_bin: distance_bucket(
                observation.yards_to_goal,
                &[12.0, 20.0, 35.0, 55.0],
            ),
            pressure_bin: pressure_bucket(observation.nearest_opponent_distance),
            open_space_bin: distance_bucket(observation.open_space_score, &[8.0, 14.0, 22.0, 32.0]),
        }
    }

    pub fn from_transition(transition: &SoccerLearningTransition) -> Self {
        Self::from_parts(
            &transition.state,
            &transition.observation,
            transition.team,
            transition.role,
        )
    }

    pub fn from_next_transition(transition: &SoccerLearningTransition) -> Self {
        Self::from_parts(
            &transition.next_state,
            &transition.next_observation,
            transition.team,
            transition.role,
        )
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerQActionKey {
    pub state: SoccerQStateKey,
    pub action: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerQPolicyOptions {
    pub alpha: f64,
    pub gamma: f64,
}

impl Default for SoccerQPolicyOptions {
    fn default() -> Self {
        SoccerQPolicyOptions {
            alpha: 0.24,
            gamma: 0.94,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerQEntry {
    pub state: SoccerQStateKey,
    pub action: String,
    pub value: f64,
    pub visits: u32,
}

#[derive(Clone, Debug)]
pub struct SoccerQPolicy {
    pub q_values: HashMap<SoccerQActionKey, f64>,
    pub visits: HashMap<SoccerQActionKey, u32>,
    pub options: SoccerQPolicyOptions,
}

impl Default for SoccerQPolicy {
    fn default() -> Self {
        Self::new(SoccerQPolicyOptions::default())
    }
}

impl SoccerQPolicy {
    pub fn new(options: SoccerQPolicyOptions) -> Self {
        SoccerQPolicy {
            q_values: HashMap::new(),
            visits: HashMap::new(),
            options,
        }
    }

    pub fn train(&mut self, transitions: &[SoccerLearningTransition]) {
        for transition in transitions {
            self.update(transition);
        }
    }

    pub fn update(&mut self, transition: &SoccerLearningTransition) {
        let state = SoccerQStateKey::from_transition(transition);
        let next_state = SoccerQStateKey::from_next_transition(transition);
        let action = normalize_soccer_action_label(&transition.action).to_string();
        let key = SoccerQActionKey { state, action };
        let old = self.q_values.get(&key).copied().unwrap_or(0.0);
        let max_next = if transition.done {
            0.0
        } else {
            self.best_value(&next_state).unwrap_or(0.0)
        };
        let alpha = self.options.alpha.clamp(0.0, 1.0);
        let gamma = self.options.gamma.clamp(0.0, 0.999);
        let target = transition.reward + gamma * max_next;
        let updated = old + alpha * (target - old);
        self.q_values.insert(key.clone(), updated);
        *self.visits.entry(key).or_insert(0) += 1;
    }

    pub fn best_action(&self, state: &SoccerQStateKey) -> Option<String> {
        self.best_action_filtered(state, |_| true)
    }

    pub fn best_action_for_snapshot(
        &self,
        snapshot: &WorldSnapshot,
        player_id: usize,
    ) -> Option<String> {
        let player = snapshot.players.iter().find(|p| p.id == player_id)?;
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state(),
            &snapshot.observation_for(player_id),
            player.team,
            player.role,
        );
        self.best_action_filtered(&state, |action| {
            learned_action_label_is_legal(action, snapshot, player_id)
        })
    }

    pub fn best_value(&self, state: &SoccerQStateKey) -> Option<f64> {
        self.q_values
            .iter()
            .filter(|(key, _)| &key.state == state)
            .map(|(_, value)| *value)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn q_value(&self, state: &SoccerQStateKey, action: &str) -> Option<f64> {
        let key = SoccerQActionKey {
            state: state.clone(),
            action: normalize_soccer_action_label(action).to_string(),
        };
        self.q_values.get(&key).copied()
    }

    pub fn set_action_value(&mut self, state: SoccerQStateKey, action: &str, value: f64) {
        let key = SoccerQActionKey {
            state,
            action: normalize_soccer_action_label(action).to_string(),
        };
        self.q_values.insert(key.clone(), value);
        self.visits.entry(key).or_insert(1);
    }

    pub fn set_action_value_for_snapshot(
        &mut self,
        snapshot: &WorldSnapshot,
        player_id: usize,
        action: &str,
        value: f64,
    ) -> bool {
        let Some(player) = snapshot.players.iter().find(|p| p.id == player_id) else {
            return false;
        };
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state(),
            &snapshot.observation_for(player_id),
            player.team,
            player.role,
        );
        self.set_action_value(state, action, value);
        true
    }

    pub fn entries(&self) -> Vec<SoccerQEntry> {
        let mut entries = self
            .q_values
            .iter()
            .map(|(key, value)| SoccerQEntry {
                state: key.state.clone(),
                action: key.action.clone(),
                value: *value,
                visits: self.visits.get(key).copied().unwrap_or(0),
            })
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| {
            a.action
                .cmp(&b.action)
                .then_with(|| a.visits.cmp(&b.visits))
                .then_with(|| {
                    a.value
                        .partial_cmp(&b.value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        entries
    }

    fn best_action_filtered<F>(&self, state: &SoccerQStateKey, is_legal: F) -> Option<String>
    where
        F: Fn(&str) -> bool,
    {
        self.q_values
            .iter()
            .filter(|(key, _)| &key.state == state && is_legal(&key.action))
            .max_by(|(a_key, a_value), (b_key, b_value)| {
                a_value
                    .partial_cmp(b_value)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| {
                        self.visits
                            .get(a_key)
                            .copied()
                            .unwrap_or(0)
                            .cmp(&self.visits.get(b_key).copied().unwrap_or(0))
                    })
            })
            .map(|(key, _)| key.action.clone())
    }
}

pub fn train_soccer_q_policy(
    dataset: &SoccerLearningDataset,
    options: SoccerQPolicyOptions,
) -> SoccerQPolicy {
    let mut policy = SoccerQPolicy::new(options);
    policy.train(&dataset.transitions);
    policy
}

fn normalize_soccer_action_label(action: &str) -> &str {
    match action {
        "move" => "space",
        other => other,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerAgent {
    pub id: usize,
    pub name: String,
    pub team: Team,
    pub role: PlayerRole,
    pub shirt: u8,
    pub home_position: Vec2,
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub position_history: VecDeque<Vec2>,
    pub skills: SkillProfile,
    pub fatigue: f64,
    pub controller_slot: Option<usize>,
    pub preferences: AgentPreferences,
    pub last_decision: Option<AgentDecisionTrace>,
}

impl PlayerAgent {
    fn record_position_history(&mut self) {
        self.position_history.push_back(self.position);
        while self.position_history.len() > PLAYER_POSITION_HISTORY_LIMIT {
            self.position_history.pop_front();
        }
    }

    pub fn history_velocity_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 2 {
            return self.velocity;
        }
        let last = self.position_history.len() - 1;
        (self.position_history[last] - self.position_history[last - 1]) / dt_seconds
    }

    pub fn history_acceleration_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 3 {
            return self.acceleration;
        }
        let last = self.position_history.len() - 1;
        let v0 = (self.position_history[last - 1] - self.position_history[last - 2]) / dt_seconds;
        let v1 = (self.position_history[last] - self.position_history[last - 1]) / dt_seconds;
        (v1 - v0) / dt_seconds
    }

    pub fn run_time_step(
        &mut self,
        snapshot: &WorldSnapshot,
        human_input: Option<&HumanInputFrame>,
        learned_action: Option<&str>,
        rng: &mut SeededRandom,
    ) -> PlayerIntent {
        let mdp_state = snapshot.mdp_state();
        let observation = snapshot.observation_for(self.id);
        let belief = belief_from_observation(&observation);
        let directive = snapshot.tactical_directive(self.team);
        let has_ball = observation.has_ball;

        if let Some(input) = human_input {
            let (action, action_label) = if input.shoot {
                (SoccerAction::Shoot { power: 1.0 }, "shoot")
            } else if input.pass {
                (
                    SoccerAction::Pass {
                        target_player: input.target_player,
                        power: 0.78,
                    },
                    "pass",
                )
            } else {
                let dir = input.axis.normalized();
                (
                    SoccerAction::MoveTo(
                        self.position + dir * if input.sprint { 7.0 } else { 4.5 },
                    ),
                    "human-move",
                )
            };
            self.last_decision = Some(AgentDecisionTrace {
                mdp_state,
                observation,
                belief,
                operation_order: vec!["human-input".to_string()],
                action: action_label.to_string(),
            });
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: input.sprint,
            };
        }

        if has_ball
            && observation.yards_to_goal <= directive.shot_threshold_yards
            && observation.shot_lane_open
            && rng.next_float() > self.skills.decision_noise
        {
            let action = SoccerAction::Shoot {
                power: 0.72 + 0.28 * self.skills.shooting,
            };
            self.last_decision = Some(AgentDecisionTrace {
                mdp_state,
                observation,
                belief,
                operation_order: vec!["finish".to_string()],
                action: action.label(),
            });
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: false,
            };
        }

        if let Some(label) = learned_action {
            if let Some((action, action_label)) =
                self.action_from_learned_label(label, snapshot, &observation)
            {
                self.last_decision = Some(AgentDecisionTrace {
                    mdp_state,
                    observation,
                    belief,
                    operation_order: vec!["learned-policy".to_string(), label.to_string()],
                    action: action_label,
                });
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }
        }

        let mut ops = vec![0usize, 1, 2, 3, 4];
        fisher_yates_shuffle(&mut ops, rng);
        let mut order_names = Vec::with_capacity(ops.len());
        let mut chosen = None;
        for op in ops {
            match op {
                0 => {
                    order_names.push("pass".to_string());
                    let pass_chance =
                        (self.preferences.pass_bias * directive.pass_priority).clamp(0.04, 0.96);
                    if has_ball && rng.next_float() < pass_chance {
                        if let Some(target) = snapshot.best_pass_target(self.id) {
                            chosen = Some((
                                SoccerAction::Pass {
                                    target_player: Some(target),
                                    power: 0.58 + 0.32 * self.skills.passing,
                                },
                                "pass".to_string(),
                            ));
                            break;
                        }
                    }
                }
                1 => {
                    order_names.push("dribble".to_string());
                    let dribble_chance = (self.preferences.dribble_bias
                        * (0.62 + self.skills.dribbling * 0.48)
                        * directive.carry_priority)
                        .clamp(0.02, 0.94);
                    if has_ball && rng.next_float() < dribble_chance {
                        let target = snapshot.forward_space_for(self.id, self.home_position);
                        chosen = Some((SoccerAction::Dribble(target), "dribble".to_string()));
                        break;
                    }
                }
                2 => {
                    order_names.push("tackle".to_string());
                    if let Some(holder) = snapshot.ball.holder {
                        let holder_is_opponent = snapshot
                            .players
                            .iter()
                            .find(|p| p.id == holder)
                            .is_some_and(|p| p.team == self.team.other());
                        if !has_ball
                            && holder_is_opponent
                            && self.position.distance(snapshot.ball.position) < 3.1
                            && rng.next_float()
                                < ((self.skills.defending * 0.6 + self.skills.aggression * 0.4)
                                    * directive.press_intensity)
                                    .clamp(0.02, 0.92)
                        {
                            chosen = Some((
                                SoccerAction::Tackle {
                                    target_player: holder,
                                },
                                "tackle".to_string(),
                            ));
                            break;
                        }
                    }
                }
                3 => {
                    order_names.push("defend".to_string());
                    if !has_ball && snapshot.possession_team() == Some(self.team.other()) {
                        let dist = self.position.distance(snapshot.ball.position);
                        let defend_radius = 10.0 + directive.press_intensity * 12.0;
                        if dist < defend_radius
                            || rng.next_float()
                                < (self.skills.aggression * directive.press_intensity)
                                    .clamp(0.02, 0.95)
                        {
                            chosen = Some((
                                SoccerAction::MoveTo(snapshot.ball.position),
                                "defend".to_string(),
                            ));
                            break;
                        }
                    }
                }
                4 => {
                    order_names.push("space".to_string());
                    if !has_ball {
                        let target = if snapshot.possession_team() == Some(self.team) {
                            snapshot.open_space_for(self.id, self.home_position)
                        } else {
                            snapshot.defensive_shape_for(self.id, self.home_position)
                        };
                        chosen = Some((SoccerAction::MoveTo(target), "space".to_string()));
                        break;
                    }
                }
                _ => {}
            }
        }

        let (action, action_label) = chosen
            .unwrap_or_else(|| (SoccerAction::MoveTo(self.home_position), "hold".to_string()));
        self.last_decision = Some(AgentDecisionTrace {
            mdp_state,
            observation,
            belief,
            operation_order: order_names,
            action: action_label,
        });
        PlayerIntent {
            player_id: self.id,
            action,
            sprint: false,
        }
    }

    fn action_from_learned_label(
        &self,
        label: &str,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> Option<(SoccerAction, String)> {
        match label {
            "shoot"
                if observation.has_ball
                    && observation.shot_lane_open
                    && observation.yards_to_goal
                        <= snapshot.tactical_directive(self.team).shot_threshold_yards =>
            {
                Some((
                    SoccerAction::Shoot {
                        power: 0.72 + 0.28 * self.skills.shooting,
                    },
                    "shoot".to_string(),
                ))
            }
            "pass" if observation.has_ball => snapshot.best_pass_target(self.id).map(|target| {
                (
                    SoccerAction::Pass {
                        target_player: Some(target),
                        power: 0.58 + 0.32 * self.skills.passing,
                    },
                    "pass".to_string(),
                )
            }),
            "dribble" if observation.has_ball => Some((
                SoccerAction::Dribble(snapshot.forward_space_for(self.id, self.home_position)),
                "dribble".to_string(),
            )),
            "defend" if snapshot.possession_team() == Some(self.team.other()) => Some((
                SoccerAction::MoveTo(snapshot.ball.position),
                "defend".to_string(),
            )),
            "tackle" => snapshot.ball.holder.and_then(|holder| {
                let holder_player = snapshot.players.iter().find(|p| p.id == holder)?;
                if holder_player.team == self.team.other()
                    && self.position.distance(holder_player.position) < 3.2
                {
                    Some((
                        SoccerAction::Tackle {
                            target_player: holder,
                        },
                        "tackle".to_string(),
                    ))
                } else {
                    None
                }
            }),
            "space" if !observation.has_ball => {
                let target = if snapshot.possession_team() == Some(self.team) {
                    snapshot.open_space_for(self.id, self.home_position)
                } else {
                    snapshot.defensive_shape_for(self.id, self.home_position)
                };
                Some((SoccerAction::MoveTo(target), "space".to_string()))
            }
            "hold" => Some((SoccerAction::MoveTo(self.home_position), "hold".to_string())),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SoccerAction {
    HoldShape,
    MoveTo(Vec2),
    Dribble(Vec2),
    Pass {
        target_player: Option<usize>,
        power: f64,
    },
    Shoot {
        power: f64,
    },
    Tackle {
        target_player: usize,
    },
}

impl SoccerAction {
    fn label(&self) -> String {
        match self {
            SoccerAction::HoldShape => "hold".to_string(),
            SoccerAction::MoveTo(_) => "move".to_string(),
            SoccerAction::Dribble(_) => "dribble".to_string(),
            SoccerAction::Pass { .. } => "pass".to_string(),
            SoccerAction::Shoot { .. } => "shoot".to_string(),
            SoccerAction::Tackle { .. } => "tackle".to_string(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerIntent {
    pub player_id: usize,
    pub action: SoccerAction,
    pub sprint: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HumanInputFrame {
    pub controller_slot: usize,
    pub player_id: Option<usize>,
    pub seq: u64,
    pub axis: Vec2,
    pub sprint: bool,
    pub pass: bool,
    pub shoot: bool,
    pub target_player: Option<usize>,
}

const HUMAN_INPUT_QUEUE_LIMIT: usize = 256;

#[derive(Clone, Debug, Default)]
struct SharedHumanInputStore {
    pending: VecDeque<HumanInputFrame>,
    latest_seq_by_slot: HashMap<usize, u64>,
}

impl SharedHumanInputStore {
    fn push(&mut self, input: HumanInputFrame) -> bool {
        if self
            .latest_seq_by_slot
            .get(&input.controller_slot)
            .is_some_and(|last_seq| input.seq <= *last_seq)
        {
            return false;
        }
        self.latest_seq_by_slot
            .insert(input.controller_slot, input.seq);
        self.pending.push_back(input);
        while self.pending.len() > HUMAN_INPUT_QUEUE_LIMIT {
            self.pending.pop_front();
        }
        true
    }

    fn drain_latest_by_slot(&mut self) -> HashMap<usize, HumanInputFrame> {
        let mut latest = HashMap::new();
        for input in self.pending.drain(..) {
            latest
                .entry(input.controller_slot)
                .and_modify(|current: &mut HumanInputFrame| {
                    if input.seq > current.seq {
                        *current = input.clone();
                    }
                })
                .or_insert(input);
        }
        latest
    }
}

#[derive(Clone, Default)]
pub struct SharedHumanInputs {
    inner: Arc<RwLock<SharedHumanInputStore>>,
}

impl SharedHumanInputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, input: HumanInputFrame) -> bool {
        self.inner
            .write()
            .expect("human input queue lock poisoned")
            .push(input)
    }

    pub fn drain_latest_by_slot(&self) -> HashMap<usize, HumanInputFrame> {
        self.inner
            .write()
            .expect("human input queue lock poisoned")
            .drain_latest_by_slot()
    }

    pub fn queued_len(&self) -> usize {
        self.inner
            .read()
            .expect("human input queue lock poisoned")
            .pending
            .len()
    }

    pub fn last_seq_for_slot(&self, controller_slot: usize) -> Option<u64> {
        self.inner
            .read()
            .expect("human input queue lock poisoned")
            .latest_seq_by_slot
            .get(&controller_slot)
            .copied()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerPositionSample {
    pub player_id: usize,
    pub tick: u64,
    pub clock_seconds: f64,
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SharedPlayerPositionSnapshot {
    pub latest: Vec<PlayerPositionSample>,
    pub histories: HashMap<usize, Vec<PlayerPositionSample>>,
}

impl SharedPlayerPositionSnapshot {
    pub fn latest_for(&self, player_id: usize) -> Option<&PlayerPositionSample> {
        self.latest
            .iter()
            .find(|sample| sample.player_id == player_id)
    }

    pub fn history_for(&self, player_id: usize) -> Option<&[PlayerPositionSample]> {
        self.histories
            .get(&player_id)
            .map(|history| history.as_slice())
    }

    fn from_player_snapshots(players: &[PlayerSnapshot], tick: u64, clock_seconds: f64) -> Self {
        let latest = players
            .iter()
            .map(|p| PlayerPositionSample {
                player_id: p.id,
                tick,
                clock_seconds,
                position: p.position,
                velocity: p.velocity,
                acceleration: p.acceleration,
            })
            .collect::<Vec<_>>();
        let histories = latest
            .iter()
            .cloned()
            .map(|sample| (sample.player_id, vec![sample]))
            .collect::<HashMap<_, _>>();
        SharedPlayerPositionSnapshot { latest, histories }
    }
}

#[derive(Clone, Debug)]
struct SharedPlayerPositionStore {
    capacity: usize,
    latest: Vec<PlayerPositionSample>,
    histories: HashMap<usize, VecDeque<PlayerPositionSample>>,
}

impl SharedPlayerPositionStore {
    fn with_capacity(capacity: usize) -> Self {
        SharedPlayerPositionStore {
            capacity: capacity.max(1),
            latest: Vec::new(),
            histories: HashMap::new(),
        }
    }

    fn latest_for(&self, player_id: usize) -> Option<&PlayerPositionSample> {
        self.latest
            .iter()
            .find(|sample| sample.player_id == player_id)
    }

    fn sync_from_players(&mut self, players: &[PlayerAgent], tick: u64, clock_seconds: f64) {
        self.latest = players
            .iter()
            .map(|p| PlayerPositionSample {
                player_id: p.id,
                tick,
                clock_seconds,
                position: p.position,
                velocity: p.velocity,
                acceleration: p.acceleration,
            })
            .collect();
        for sample in &self.latest {
            let history = self.histories.entry(sample.player_id).or_default();
            if history
                .back()
                .is_some_and(|last| last.tick == tick && last.clock_seconds == clock_seconds)
            {
                if let Some(last) = history.back_mut() {
                    *last = sample.clone();
                }
            } else {
                history.push_back(sample.clone());
            }
            while history.len() > self.capacity {
                history.pop_front();
            }
        }
    }

    fn snapshot_with_current_players(
        &self,
        players: &[PlayerAgent],
        tick: u64,
        clock_seconds: f64,
        dt_seconds: f64,
    ) -> SharedPlayerPositionSnapshot {
        let latest = players
            .iter()
            .map(|p| {
                let prev_velocity = self
                    .latest_for(p.id)
                    .map(|sample| sample.velocity)
                    .unwrap_or(p.velocity);
                let acceleration = if dt_seconds > 0.0 {
                    (p.velocity - prev_velocity) / dt_seconds
                } else {
                    p.acceleration
                };
                PlayerPositionSample {
                    player_id: p.id,
                    tick,
                    clock_seconds,
                    position: p.position,
                    velocity: p.velocity,
                    acceleration,
                }
            })
            .collect::<Vec<_>>();

        let mut histories = self
            .histories
            .iter()
            .map(|(id, history)| (*id, history.iter().cloned().collect::<Vec<_>>()))
            .collect::<HashMap<_, _>>();

        for sample in &latest {
            let history = histories.entry(sample.player_id).or_default();
            if history.last().is_some_and(|last| last.tick == tick) {
                if let Some(last) = history.last_mut() {
                    *last = sample.clone();
                }
            } else {
                history.push(sample.clone());
            }
            while history.len() > self.capacity {
                history.remove(0);
            }
        }

        SharedPlayerPositionSnapshot { latest, histories }
    }

    fn snapshot(&self) -> SharedPlayerPositionSnapshot {
        SharedPlayerPositionSnapshot {
            latest: self.latest.clone(),
            histories: self
                .histories
                .iter()
                .map(|(id, history)| (*id, history.iter().cloned().collect()))
                .collect(),
        }
    }
}

#[derive(Clone, Debug)]
pub struct SharedPlayerPositions {
    inner: Arc<RwLock<SharedPlayerPositionStore>>,
}

impl SharedPlayerPositions {
    pub fn with_capacity(capacity: usize) -> Self {
        SharedPlayerPositions {
            inner: Arc::new(RwLock::new(SharedPlayerPositionStore::with_capacity(
                capacity,
            ))),
        }
    }

    pub fn snapshot(&self) -> SharedPlayerPositionSnapshot {
        self.inner
            .read()
            .expect("shared player position lock poisoned")
            .snapshot()
    }

    pub fn latest_for(&self, player_id: usize) -> Option<PlayerPositionSample> {
        self.inner
            .read()
            .expect("shared player position lock poisoned")
            .latest_for(player_id)
            .cloned()
    }

    pub fn history_for(&self, player_id: usize) -> Option<Vec<PlayerPositionSample>> {
        self.inner
            .read()
            .expect("shared player position lock poisoned")
            .histories
            .get(&player_id)
            .map(|history| history.iter().cloned().collect())
    }

    fn sync_from_players(&self, players: &[PlayerAgent], tick: u64, clock_seconds: f64) {
        self.inner
            .write()
            .expect("shared player position lock poisoned")
            .sync_from_players(players, tick, clock_seconds);
    }

    fn snapshot_with_current_players(
        &self,
        players: &[PlayerAgent],
        tick: u64,
        clock_seconds: f64,
        dt_seconds: f64,
    ) -> SharedPlayerPositionSnapshot {
        self.inner
            .read()
            .expect("shared player position lock poisoned")
            .snapshot_with_current_players(players, tick, clock_seconds, dt_seconds)
    }
}

impl Default for SharedPlayerPositions {
    fn default() -> Self {
        SharedPlayerPositions::with_capacity(PLAYER_POSITION_HISTORY_LIMIT)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BallState {
    pub position: Vec2,
    pub velocity: Vec2,
    #[serde(default)]
    pub acceleration: Vec2,
    pub holder: Option<usize>,
    pub last_touch_team: Option<Team>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BallPositionSample {
    pub tick: u64,
    pub clock_seconds: f64,
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub holder: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BallDecisionTrace {
    pub tick: u64,
    pub action: String,
    pub position: Vec2,
    pub holder: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BallAgent {
    pub id: usize,
    pub position: Vec2,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub position_history: VecDeque<BallPositionSample>,
    pub holder: Option<usize>,
    pub last_touch_team: Option<Team>,
    pub last_decision: Option<BallDecisionTrace>,
}

impl BallAgent {
    pub fn new(id: usize, state: BallState) -> Self {
        BallAgent {
            id,
            position: state.position,
            velocity: state.velocity,
            acceleration: state.acceleration,
            position_history: VecDeque::from([BallPositionSample {
                tick: 0,
                clock_seconds: 0.0,
                position: state.position,
                velocity: state.velocity,
                acceleration: state.acceleration,
                holder: state.holder,
            }]),
            holder: state.holder,
            last_touch_team: state.last_touch_team,
            last_decision: None,
        }
    }

    pub fn to_state(&self) -> BallState {
        BallState {
            position: self.position,
            velocity: self.velocity,
            acceleration: self.acceleration,
            holder: self.holder,
            last_touch_team: self.last_touch_team,
        }
    }

    fn update_acceleration_from(&mut self, previous_velocity: Vec2, dt_seconds: f64) {
        self.acceleration = if dt_seconds > 0.0 {
            (self.velocity - previous_velocity) / dt_seconds
        } else {
            Vec2::zero()
        };
    }

    fn record_position_history(&mut self, tick: u64, clock_seconds: f64) {
        self.position_history.push_back(BallPositionSample {
            tick,
            clock_seconds,
            position: self.position,
            velocity: self.velocity,
            acceleration: self.acceleration,
            holder: self.holder,
        });
        while self.position_history.len() > BALL_POSITION_HISTORY_LIMIT {
            self.position_history.pop_front();
        }
    }

    pub fn history_velocity_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 2 {
            return self.velocity;
        }
        let last = self.position_history.len() - 1;
        (self.position_history[last].position - self.position_history[last - 1].position)
            / dt_seconds
    }

    pub fn history_acceleration_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 3 {
            return self.acceleration;
        }
        let last = self.position_history.len() - 1;
        let v0 = (self.position_history[last - 1].position
            - self.position_history[last - 2].position)
            / dt_seconds;
        let v1 = (self.position_history[last].position - self.position_history[last - 1].position)
            / dt_seconds;
        (v1 - v0) / dt_seconds
    }

    fn run_time_step(
        &mut self,
        context: BallStepContext<'_>,
        rng: &mut SeededRandom,
    ) -> BallStepOutcome {
        if let Some(holder) = self.holder {
            if let Some(player) = context.players.iter().find(|player| player.id == holder) {
                let lead = Vec2::new(0.0, player.team.attack_dir()) * 0.75;
                self.position = (player.position + lead)
                    .clamp_to_pitch(context.field_width, context.field_length);
                self.velocity = player.velocity;
                self.last_touch_team = Some(player.team);
                self.record_decision(context.tick, "held");
            }
            return BallStepOutcome::None;
        }

        self.position += self.velocity * context.dt_seconds;
        let drag_multiplier = (1.0 - context.ball_drag_per_tick).clamp(0.0, 1.0);
        self.velocity = self.velocity * drag_multiplier;
        if self.velocity.len() < context.ball_stop_speed_yps {
            self.velocity = Vec2::zero();
        }

        if self.position.x < 0.0 || self.position.x > context.field_width {
            let awarded_team = self.last_touch_team.map(Team::other).unwrap_or(Team::Home);
            self.position = Vec2::new(
                self.position.x.max(0.0).min(context.field_width),
                self.position.y.max(0.0).min(context.field_length),
            );
            self.velocity = Vec2::zero();
            self.holder = None;
            self.last_touch_team = Some(awarded_team);
            self.record_decision(context.tick, "throw-in");
            return BallStepOutcome::OutOfPlay {
                restart: BallRestart {
                    kind: BallRestartKind::ThrowIn,
                    awarded_team,
                    position: self.position,
                },
                shot: context.pending_shot,
            };
        }

        if self.position.y < 0.0 || self.position.y > context.field_length {
            let goal_x = context.field_width * 0.5;
            let in_goal = (self.position.x - goal_x).abs() <= context.goal_width * 0.5;
            if in_goal {
                let scoring_team = if self.position.y > context.field_length {
                    Team::Home
                } else {
                    Team::Away
                };
                if let Some(shot) = context.pending_shot {
                    let defending_team = scoring_team.other();
                    if let Some(keeper_id) = goalkeeper_for_players(context.players, defending_team)
                    {
                        let Some(keeper) =
                            context.players.iter().find(|player| player.id == keeper_id)
                        else {
                            self.record_decision(context.tick, "goal");
                            return BallStepOutcome::Goal {
                                scoring_team,
                                shot: Some(shot),
                            };
                        };
                        let save_probability = goalkeeper_save_probability(
                            keeper,
                            self.position,
                            self.velocity.len(),
                            context.goal_width,
                        );
                        if rng.next_float() < save_probability {
                            let save_y = match defending_team {
                                Team::Home => SHOT_SAVE_DEPTH_YARDS,
                                Team::Away => context.field_length - SHOT_SAVE_DEPTH_YARDS,
                            };
                            let save_position = Vec2::new(
                                self.position.x.clamp(
                                    context.field_width * 0.5 - context.goal_width * 0.55,
                                    context.field_width * 0.5 + context.goal_width * 0.55,
                                ),
                                save_y,
                            );
                            self.holder = Some(keeper_id);
                            self.position = save_position;
                            self.velocity = Vec2::zero();
                            self.last_touch_team = Some(defending_team);
                            self.record_decision(context.tick, "save");
                            return BallStepOutcome::Save {
                                shot,
                                defending_team,
                                keeper_id,
                                save_position,
                            };
                        }
                    }
                    self.record_decision(context.tick, "goal");
                    return BallStepOutcome::Goal {
                        scoring_team,
                        shot: Some(shot),
                    };
                }
                self.record_decision(context.tick, "goal");
                return BallStepOutcome::Goal {
                    scoring_team,
                    shot: None,
                };
            }
            let defending_team = if self.position.y > context.field_length {
                Team::Away
            } else {
                Team::Home
            };
            let attacking_team = defending_team.other();
            let last_touch = self.last_touch_team;
            let kind = if last_touch == Some(defending_team) {
                BallRestartKind::CornerKick
            } else {
                BallRestartKind::GoalKick
            };
            let awarded_team = match kind {
                BallRestartKind::CornerKick => attacking_team,
                BallRestartKind::GoalKick => defending_team,
                BallRestartKind::ThrowIn => unreachable!("endline cannot award throw-in"),
                BallRestartKind::FreeKick => unreachable!("endline cannot award free kick"),
            };
            let end_y = self.position.y.max(0.0).min(context.field_length);
            let restart_position = match kind {
                BallRestartKind::GoalKick => {
                    let y = match defending_team {
                        Team::Home => SHOT_SAVE_DEPTH_YARDS + 4.4,
                        Team::Away => context.field_length - SHOT_SAVE_DEPTH_YARDS - 4.4,
                    };
                    Vec2::new(context.field_width * 0.5, y)
                }
                BallRestartKind::CornerKick => {
                    let corner_x = if self.position.x <= context.field_width * 0.5 {
                        0.0
                    } else {
                        context.field_width
                    };
                    Vec2::new(corner_x, end_y)
                }
                BallRestartKind::ThrowIn => unreachable!("endline cannot award throw-in"),
                BallRestartKind::FreeKick => unreachable!("endline cannot award free kick"),
            };
            self.position = restart_position;
            self.velocity = Vec2::zero();
            self.holder = None;
            self.last_touch_team = Some(awarded_team);
            let action = match kind {
                BallRestartKind::GoalKick => "goal-kick",
                BallRestartKind::CornerKick => "corner-kick",
                BallRestartKind::ThrowIn => "throw-in",
                BallRestartKind::FreeKick => "free-kick",
            };
            self.record_decision(context.tick, action);
            return BallStepOutcome::OutOfPlay {
                restart: BallRestart {
                    kind,
                    awarded_team,
                    position: restart_position,
                },
                shot: context.pending_shot,
            };
        }

        if let Some((holder, holder_team)) =
            nearest_ball_controller_for(self.position, self.velocity, context.players, rng)
        {
            self.holder = Some(holder);
            self.last_touch_team = Some(holder_team);
            self.record_decision(context.tick, "controlled");
            let possession_result = match context.pending_pass {
                Some(pass) if pass.team == holder_team => {
                    BallPossessionResult::PassCompleted(holder_team)
                }
                Some(pass) if pass.team != holder_team => {
                    BallPossessionResult::Interception(holder_team)
                }
                _ => BallPossessionResult::LooseBallRecovery(holder_team),
            };
            return BallStepOutcome::Controlled {
                holder,
                holder_team,
                possession_result,
            };
        }

        self.record_decision(context.tick, "roll");
        BallStepOutcome::None
    }

    fn record_decision(&mut self, tick: u64, action: &str) {
        self.last_decision = Some(BallDecisionTrace {
            tick,
            action: action.to_string(),
            position: self.position,
            holder: self.holder,
        });
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfficialAgent {
    pub id: usize,
    pub kind: OfficialKind,
    pub position: Vec2,
    pub velocity: Vec2,
}

impl OfficialAgent {
    fn run_time_step(&mut self, snapshot: &WorldSnapshot, rng: &mut SeededRandom) {
        let target = match self.kind {
            OfficialKind::CenterReferee => Vec2::new(
                snapshot.field_width * 0.5,
                snapshot.ball.position.y * 0.72 + snapshot.field_length * 0.14,
            ),
            OfficialKind::AssistantRefereeNear => Vec2::new(1.5, snapshot.ball.position.y),
            OfficialKind::AssistantRefereeFar => {
                Vec2::new(snapshot.field_width - 1.5, snapshot.ball.position.y)
            }
        };
        let jitter = Vec2::new(rng.next_float() - 0.5, rng.next_float() - 0.5) * 0.25;
        let desired = (target + jitter - self.position).normalized() * 6.1;
        self.velocity = approach_velocity(self.velocity, desired, 5.2, snapshot.dt_seconds);
        self.position += self.velocity * snapshot.dt_seconds;
        self.position = self
            .position
            .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchConfig {
    pub dt_seconds: f64,
    pub duration_seconds: f64,
    pub field_length_yards: f64,
    pub field_width_yards: f64,
    pub goal_width_yards: f64,
    #[serde(default = "default_ball_drag_per_tick")]
    pub ball_drag_per_tick: f64,
    #[serde(default = "default_ball_stop_speed_yps")]
    pub ball_stop_speed_yps: f64,
    pub max_human_players: usize,
    pub seed: u32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        MatchConfig {
            dt_seconds: DEFAULT_DT_SECONDS,
            duration_seconds: DEFAULT_DURATION_SECONDS,
            field_length_yards: DEFAULT_FIELD_LENGTH_YARDS,
            field_width_yards: DEFAULT_FIELD_WIDTH_YARDS,
            goal_width_yards: DEFAULT_GOAL_WIDTH_YARDS,
            ball_drag_per_tick: DEFAULT_BALL_DRAG_PER_TICK,
            ball_stop_speed_yps: DEFAULT_BALL_STOP_SPEED_YPS,
            max_human_players: 4,
            seed: 2026,
        }
    }
}

impl MatchConfig {
    pub fn total_ticks(&self) -> u64 {
        (self.duration_seconds / self.dt_seconds).round() as u64
    }

    pub fn human_slots(&self) -> usize {
        self.max_human_players.min(4)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchStats {
    pub shots_home: u32,
    pub shots_away: u32,
    pub shots_on_target_home: u32,
    pub shots_on_target_away: u32,
    pub saves_home: u32,
    pub saves_away: u32,
    pub passes_attempted_home: u32,
    pub passes_attempted_away: u32,
    pub passes_completed_home: u32,
    pub passes_completed_away: u32,
    pub interceptions_home: u32,
    pub interceptions_away: u32,
    pub loose_ball_recoveries_home: u32,
    pub loose_ball_recoveries_away: u32,
    pub offsides_home: u32,
    pub offsides_away: u32,
    pub throw_ins_home: u32,
    pub throw_ins_away: u32,
    pub goal_kicks_home: u32,
    pub goal_kicks_away: u32,
    pub corner_kicks_home: u32,
    pub corner_kicks_away: u32,
    pub free_kicks_home: u32,
    pub free_kicks_away: u32,
    pub fouls_home: u32,
    pub fouls_away: u32,
    pub tackles: u32,
}

#[derive(Clone, Debug)]
struct PendingPass {
    team: Team,
    from: usize,
    target: Option<usize>,
    offside: Option<PendingOffside>,
}

#[derive(Clone, Debug)]
struct PendingShot {
    team: Team,
    shooter: usize,
}

#[derive(Clone, Debug)]
struct PendingOffside {
    team: Team,
    passer: usize,
    target: usize,
    position: Vec2,
    ball_y: f64,
    second_last_defender_y: f64,
}

#[derive(Clone, Debug)]
struct BallStepContext<'a> {
    tick: u64,
    clock_seconds: f64,
    dt_seconds: f64,
    ball_drag_per_tick: f64,
    ball_stop_speed_yps: f64,
    field_length: f64,
    field_width: f64,
    goal_width: f64,
    players: &'a [PlayerAgent],
    pending_pass: Option<PendingPass>,
    pending_shot: Option<PendingShot>,
}

#[derive(Clone, Debug)]
enum BallPossessionResult {
    PassCompleted(Team),
    Interception(Team),
    LooseBallRecovery(Team),
}

#[derive(Clone, Debug)]
enum BallStepOutcome {
    None,
    Controlled {
        holder: usize,
        holder_team: Team,
        possession_result: BallPossessionResult,
    },
    Save {
        shot: PendingShot,
        defending_team: Team,
        keeper_id: usize,
        save_position: Vec2,
    },
    Goal {
        scoring_team: Team,
        shot: Option<PendingShot>,
    },
    Miss {
        shot: PendingShot,
    },
    OutOfPlay {
        restart: BallRestart,
        shot: Option<PendingShot>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BallRestartKind {
    ThrowIn,
    GoalKick,
    CornerKick,
    FreeKick,
}

#[derive(Clone, Debug)]
struct BallRestart {
    kind: BallRestartKind,
    awarded_team: Team,
    position: Vec2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TeamTacticalDirective {
    pub team: Team,
    pub defensive_line_y: f64,
    pub support_depth_yards: f64,
    pub width_yards: f64,
    pub press_intensity: f64,
    pub pass_priority: f64,
    pub carry_priority: f64,
    pub shot_threshold_yards: f64,
    pub risk_tolerance: f64,
}

impl TeamTacticalDirective {
    fn neutral(team: Team, field_width: f64, field_length: f64) -> Self {
        TeamTacticalDirective {
            team,
            defensive_line_y: match team {
                Team::Home => field_length * 0.40,
                Team::Away => field_length * 0.60,
            },
            support_depth_yards: 11.0,
            width_yards: field_width * 0.62,
            press_intensity: 0.48,
            pass_priority: 1.0,
            carry_priority: 1.0,
            shot_threshold_yards: 20.0,
            risk_tolerance: 0.50,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CentralBrain {
    pub phase: TacticalPhase,
    pub possession_team: Option<Team>,
    pub pressure_line_home: f64,
    pub pressure_line_away: f64,
    pub home_directive: TeamTacticalDirective,
    pub away_directive: TeamTacticalDirective,
}

impl Default for CentralBrain {
    fn default() -> Self {
        CentralBrain {
            phase: TacticalPhase::Kickoff,
            possession_team: None,
            pressure_line_home: DEFAULT_FIELD_LENGTH_YARDS * 0.55,
            pressure_line_away: DEFAULT_FIELD_LENGTH_YARDS * 0.45,
            home_directive: TeamTacticalDirective::neutral(
                Team::Home,
                DEFAULT_FIELD_WIDTH_YARDS,
                DEFAULT_FIELD_LENGTH_YARDS,
            ),
            away_directive: TeamTacticalDirective::neutral(
                Team::Away,
                DEFAULT_FIELD_WIDTH_YARDS,
                DEFAULT_FIELD_LENGTH_YARDS,
            ),
        }
    }
}

impl CentralBrain {
    pub fn run_time_step(&mut self, snapshot: &WorldSnapshot) {
        self.possession_team = snapshot.possession_team();
        let y = snapshot.ball.position.y;
        self.phase = match self.possession_team {
            Some(Team::Home) if y > snapshot.field_length * 0.68 => TacticalPhase::HomeAttack,
            Some(Team::Home) => TacticalPhase::HomeBuildUp,
            Some(Team::Away) if y < snapshot.field_length * 0.32 => TacticalPhase::AwayAttack,
            Some(Team::Away) => TacticalPhase::AwayBuildUp,
            None if snapshot.tick < 5 => TacticalPhase::Kickoff,
            None => TacticalPhase::Transition,
        };
        let score_diff_home = snapshot.score_home as i32 - snapshot.score_away as i32;
        self.home_directive = tactical_directive_for_team(
            Team::Home,
            self.phase,
            self.possession_team,
            snapshot.ball.position,
            score_diff_home,
            snapshot.field_width,
            snapshot.field_length,
        );
        self.away_directive = tactical_directive_for_team(
            Team::Away,
            self.phase,
            self.possession_team,
            snapshot.ball.position,
            -score_diff_home,
            snapshot.field_width,
            snapshot.field_length,
        );
        self.pressure_line_home = self.home_directive.defensive_line_y;
        self.pressure_line_away = self.away_directive.defensive_line_y;
    }

    pub fn directive_for(&self, team: Team) -> &TeamTacticalDirective {
        match team {
            Team::Home => &self.home_directive,
            Team::Away => &self.away_directive,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerSnapshot {
    pub id: usize,
    pub name: String,
    pub team: Team,
    pub role: PlayerRole,
    pub shirt: u8,
    pub position: Vec2,
    pub velocity: Vec2,
    pub home_position: Vec2,
    pub controller_slot: Option<usize>,
    pub acceleration: Vec2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorldSnapshot {
    pub tick: u64,
    pub clock_seconds: f64,
    pub dt_seconds: f64,
    pub field_length: f64,
    pub field_width: f64,
    pub goal_width: f64,
    pub ball: BallState,
    #[serde(default)]
    pub ball_history: Vec<BallPositionSample>,
    pub players: Vec<PlayerSnapshot>,
    pub shared_positions: SharedPlayerPositionSnapshot,
    pub score_home: u32,
    pub score_away: u32,
    pub phase: TacticalPhase,
    pub home_directive: TeamTacticalDirective,
    pub away_directive: TeamTacticalDirective,
}

impl WorldSnapshot {
    fn from_match(m: &SoccerMatch) -> Self {
        let shared_positions = m.shared_positions.snapshot_with_current_players(
            &m.players,
            m.tick,
            m.clock_seconds,
            m.config.dt_seconds,
        );
        let players = m
            .players
            .iter()
            .map(|p| PlayerSnapshot {
                id: p.id,
                name: p.name.clone(),
                team: p.team,
                role: p.role,
                shirt: p.shirt,
                position: p.position,
                velocity: p.velocity,
                home_position: p.home_position,
                controller_slot: p.controller_slot,
                acceleration: shared_positions
                    .latest_for(p.id)
                    .map(|sample| sample.acceleration)
                    .unwrap_or(p.acceleration),
            })
            .collect::<Vec<_>>();
        WorldSnapshot {
            tick: m.tick,
            clock_seconds: m.clock_seconds,
            dt_seconds: m.config.dt_seconds,
            field_length: m.config.field_length_yards,
            field_width: m.config.field_width_yards,
            goal_width: m.config.goal_width_yards,
            ball: m.ball.to_state(),
            ball_history: m.ball.position_history.iter().cloned().collect(),
            players,
            shared_positions,
            score_home: m.score_home,
            score_away: m.score_away,
            phase: m.central_brain.phase,
            home_directive: m.central_brain.home_directive.clone(),
            away_directive: m.central_brain.away_directive.clone(),
        }
    }

    pub fn possession_team(&self) -> Option<Team> {
        self.ball
            .holder
            .and_then(|id| self.players.iter().find(|p| p.id == id))
            .map(|p| p.team)
            .or(self.ball.last_touch_team)
    }

    pub fn player_position(&self, player_id: usize) -> Option<Vec2> {
        self.shared_positions
            .latest_for(player_id)
            .map(|sample| sample.position)
            .or_else(|| {
                self.players
                    .iter()
                    .find(|p| p.id == player_id)
                    .map(|p| p.position)
            })
    }

    pub fn player_velocity(&self, player_id: usize) -> Option<Vec2> {
        self.shared_positions
            .latest_for(player_id)
            .map(|sample| sample.velocity)
            .or_else(|| {
                self.players
                    .iter()
                    .find(|p| p.id == player_id)
                    .map(|p| p.velocity)
            })
    }

    pub fn player_acceleration(&self, player_id: usize) -> Option<Vec2> {
        self.shared_positions
            .latest_for(player_id)
            .map(|sample| sample.acceleration)
            .or_else(|| {
                self.players
                    .iter()
                    .find(|p| p.id == player_id)
                    .map(|p| p.acceleration)
            })
    }

    pub fn player_position_history(&self, player_id: usize) -> Option<&[PlayerPositionSample]> {
        self.shared_positions.history_for(player_id)
    }

    pub fn ball_position_history(&self) -> &[BallPositionSample] {
        &self.ball_history
    }

    pub fn tactical_directive(&self, team: Team) -> &TeamTacticalDirective {
        match team {
            Team::Home => &self.home_directive,
            Team::Away => &self.away_directive,
        }
    }

    pub fn mdp_state(&self) -> SoccerMdpState {
        SoccerMdpState {
            tick: self.tick,
            ball_zone_x: zone(self.ball.position.x, self.field_width, 6),
            ball_zone_y: zone(self.ball.position.y, self.field_length, 8),
            possession_team: self.possession_team(),
            score_diff_for_home: self.score_home as i32 - self.score_away as i32,
            phase: self.phase,
        }
    }

    pub fn observation_for(&self, player_id: usize) -> SoccerPomdpObservation {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return SoccerPomdpObservation {
                player_id,
                has_ball: false,
                ball_distance: 0.0,
                nearest_opponent_distance: 0.0,
                nearest_teammate_distance: 0.0,
                shot_lane_open: false,
                yards_to_goal: 0.0,
                open_space_score: 0.0,
            };
        };
        let me_position = self.player_position(me.id).unwrap_or(me.position);
        let opponents = self
            .players
            .iter()
            .filter(|p| p.team != me.team)
            .collect::<Vec<_>>();
        let teammates = self
            .players
            .iter()
            .filter(|p| p.team == me.team && p.id != me.id)
            .collect::<Vec<_>>();
        let nearest_opponent_distance = opponents
            .iter()
            .map(|p| {
                self.player_position(p.id)
                    .unwrap_or(p.position)
                    .distance(me_position)
            })
            .fold(f64::INFINITY, f64::min);
        let nearest_teammate_distance = teammates
            .iter()
            .map(|p| {
                self.player_position(p.id)
                    .unwrap_or(p.position)
                    .distance(me_position)
            })
            .fold(f64::INFINITY, f64::min);
        let goal = Vec2::new(self.field_width * 0.5, me.team.goal_y(self.field_length));
        SoccerPomdpObservation {
            player_id,
            has_ball: self.ball.holder == Some(player_id),
            ball_distance: me_position.distance(self.ball.position),
            nearest_opponent_distance,
            nearest_teammate_distance,
            shot_lane_open: self.clear_line(me_position, goal, me.team.other(), 3.0),
            yards_to_goal: (goal.y - me_position.y).abs(),
            open_space_score: self.space_score_at(me_position, me.team),
        }
    }

    pub fn best_pass_target(&self, player_id: usize) -> Option<usize> {
        let me = self.players.iter().find(|p| p.id == player_id)?;
        let me_position = self.player_position(me.id).unwrap_or(me.position);
        let directive = self.tactical_directive(me.team);
        self.players
            .iter()
            .filter(|p| p.team == me.team && p.id != me.id)
            .filter_map(|p| {
                let position = self.player_position(p.id).unwrap_or(p.position);
                (self.clear_line(me_position, position, me.team.other(), 2.5)
                    && self.pending_offside_for_pass(me.id, p.id).is_none())
                .then_some((p, position))
            })
            .map(|p| {
                let (p, position) = p;
                let forward = (position.y - me_position.y) * me.team.attack_dir();
                let dist = me_position.distance(position);
                let support_fit = (dist - directive.support_depth_yards).abs();
                let forward_weight = 0.08 + directive.risk_tolerance * 0.15;
                let score = forward * forward_weight + self.space_score_at(position, me.team)
                    - dist * 0.010
                    - support_fit * 0.020;
                (p.id, score)
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    fn pending_offside_for_pass(
        &self,
        passer_id: usize,
        target_id: usize,
    ) -> Option<PendingOffside> {
        let passer = self.players.iter().find(|p| p.id == passer_id)?;
        let target = self.players.iter().find(|p| p.id == target_id)?;
        if passer.team != target.team || passer.id == target.id {
            return None;
        }
        let target_position = self.player_position(target.id).unwrap_or(target.position);
        let half_line = self.field_length * 0.5;
        match passer.team {
            Team::Home if target_position.y <= half_line => return None,
            Team::Away if target_position.y >= half_line => return None,
            _ => {}
        }

        let mut defender_ys = self
            .players
            .iter()
            .filter(|p| p.team == passer.team.other())
            .filter_map(|p| self.player_position(p.id).map(|position| position.y))
            .collect::<Vec<_>>();
        if defender_ys.len() < 2 {
            return None;
        }

        let is_offside = match passer.team {
            Team::Home => {
                defender_ys.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                let second_last_defender_y = defender_ys[1];
                (target_position.y > self.ball.position.y)
                    && (target_position.y > second_last_defender_y)
                    && (target_position.y > half_line)
            }
            Team::Away => {
                defender_ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let second_last_defender_y = defender_ys[1];
                (target_position.y < self.ball.position.y)
                    && (target_position.y < second_last_defender_y)
                    && (target_position.y < half_line)
            }
        };
        if !is_offside {
            return None;
        }

        let second_last_defender_y = match passer.team {
            Team::Home => defender_ys[1],
            Team::Away => defender_ys[1],
        };
        Some(PendingOffside {
            team: passer.team,
            passer: passer.id,
            target: target.id,
            position: target_position,
            ball_y: self.ball.position.y,
            second_last_defender_y,
        })
    }

    pub fn open_space_for(&self, player_id: usize, home: Vec2) -> Vec2 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return home;
        };
        let me_position = self.player_position(me.id).unwrap_or(me.position);
        let directive = self.tactical_directive(me.team);
        let width_scale = (directive.width_yards / (self.field_width * 0.62)).clamp(0.65, 1.35);
        let depth_scale = (directive.support_depth_yards / 11.0).clamp(0.65, 1.65);
        let mut best = home;
        let mut best_score = f64::NEG_INFINITY;
        for dx in [-16.0, -8.0, 0.0, 8.0, 16.0] {
            for dy in [-6.0, 4.0, 12.0, 20.0] {
                let p = Vec2::new(
                    home.x + dx * width_scale,
                    home.y + dy * depth_scale * me.team.attack_dir() + self.ball.position.y * 0.08,
                )
                .clamp_to_pitch(self.field_width, self.field_length);
                let forward = (p.y - me_position.y) * me.team.attack_dir();
                let score = self.space_score_at(p, me.team)
                    + forward.max(-4.0) * (0.03 + directive.risk_tolerance * 0.05)
                    - p.distance(home) * 0.025;
                if score > best_score {
                    best = p;
                    best_score = score;
                }
            }
        }
        best
    }

    pub fn forward_space_for(&self, player_id: usize, home: Vec2) -> Vec2 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return home;
        };
        let me_position = self.player_position(me.id).unwrap_or(me.position);
        let ahead = Vec2::new(
            me_position.x + (self.field_width * 0.5 - me_position.x) * 0.08,
            me_position.y + 9.0 * me.team.attack_dir(),
        );
        self.open_space_for(player_id, ahead)
            .clamp_to_pitch(self.field_width, self.field_length)
    }

    pub fn defensive_shape_for(&self, player_id: usize, home: Vec2) -> Vec2 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return home;
        };
        let ball_y = self.ball.position.y;
        let directive = self.tactical_directive(me.team);
        let role_line_bias = match me.role {
            PlayerRole::Goalkeeper => 0.10,
            PlayerRole::Defender => 0.70,
            PlayerRole::Midfielder => 0.48,
            PlayerRole::Forward => 0.28,
        };
        let compact_y = home.y * (1.0 - role_line_bias)
            + directive.defensive_line_y * role_line_bias
            + ball_y * 0.18
            - 3.0 * me.team.attack_dir();
        let width_factor = (directive.width_yards / self.field_width).clamp(0.42, 0.88);
        let mid_x = self.field_width * 0.5;
        let compact_x = mid_x
            + (home.x - mid_x) * width_factor
            + (self.ball.position.x - mid_x) * (0.18 + directive.press_intensity * 0.12);
        Vec2::new(compact_x, compact_y).clamp_to_pitch(self.field_width, self.field_length)
    }

    fn space_score_at(&self, p: Vec2, team: Team) -> f64 {
        let opponent_dist = self
            .players
            .iter()
            .filter(|other| other.team != team)
            .map(|other| {
                self.player_position(other.id)
                    .unwrap_or(other.position)
                    .distance(p)
            })
            .fold(35.0, f64::min)
            .min(35.0);
        let teammate_crowding = self
            .players
            .iter()
            .filter(|other| other.team == team)
            .map(|other| {
                self.player_position(other.id)
                    .unwrap_or(other.position)
                    .distance(p)
            })
            .filter(|&d| d > 0.1)
            .fold(25.0, f64::min)
            .min(25.0);
        opponent_dist * 0.75 + teammate_crowding * 0.25
    }

    fn clear_line(&self, from: Vec2, to: Vec2, defending_team: Team, radius: f64) -> bool {
        self.players
            .iter()
            .filter(|p| p.team == defending_team)
            .all(|p| {
                let position = self.player_position(p.id).unwrap_or(p.position);
                segment_distance_to_point(from, to, position) > radius
            })
    }
}

fn tactical_directive_for_team(
    team: Team,
    phase: TacticalPhase,
    possession_team: Option<Team>,
    ball_position: Vec2,
    score_diff_for_team: i32,
    field_width: f64,
    field_length: f64,
) -> TeamTacticalDirective {
    let has_ball = possession_team == Some(team);
    let defending = possession_team == Some(team.other());
    let attacking_phase = matches!(
        (team, phase),
        (Team::Home, TacticalPhase::HomeAttack) | (Team::Away, TacticalPhase::AwayAttack)
    );
    let build_up_phase = matches!(
        (team, phase),
        (Team::Home, TacticalPhase::HomeBuildUp) | (Team::Away, TacticalPhase::AwayBuildUp)
    );
    let trailing = score_diff_for_team < 0;
    let leading = score_diff_for_team > 0;
    let urgency: f64 = if trailing {
        0.16
    } else if leading {
        -0.08
    } else {
        0.0
    };
    let attack_dir = team.attack_dir();
    let line_seed = if has_ball {
        let holding_distance = if attacking_phase { 18.0 } else { 25.0 };
        ball_position.y - attack_dir * holding_distance
    } else if defending {
        let press_step = 8.0 + (urgency.max(0.0) * 18.0);
        ball_position.y - attack_dir * press_step
    } else {
        match team {
            Team::Home => field_length * 0.42,
            Team::Away => field_length * 0.58,
        }
    };
    let defensive_line_y: f64 = line_seed.clamp(field_length * 0.08, field_length * 0.92);

    let risk_tolerance: f64 = (0.46
        + if attacking_phase { 0.15 } else { 0.0 }
        + if build_up_phase { 0.04 } else { 0.0 }
        + urgency)
        .clamp(0.28, 0.82);
    let press_intensity: f64 = (if defending {
        0.58 + risk_tolerance * 0.45
    } else if possession_team.is_none() {
        0.54
    } else {
        0.30 + risk_tolerance * 0.12
    })
    .clamp(0.22, 1.0);
    let pass_base: f64 = if has_ball {
        if build_up_phase {
            1.18
        } else if attacking_phase {
            0.96
        } else {
            1.04
        }
    } else {
        0.84
    };
    let pass_priority: f64 = (pass_base + if leading { 0.06 } else { 0.0 }
        - if trailing && attacking_phase {
            0.04
        } else {
            0.0
        })
    .clamp(0.62, 1.32);
    let carry_priority: f64 = (if has_ball {
        if attacking_phase {
            1.20
        } else if build_up_phase {
            0.92
        } else {
            1.02
        }
    } else {
        0.76
    } + urgency * 0.55)
        .clamp(0.55, 1.35);
    let support_depth_yards: f64 = (if attacking_phase {
        15.5
    } else if build_up_phase {
        11.5
    } else if defending {
        7.5
    } else {
        10.0
    } + risk_tolerance * 3.0)
        .clamp(6.5, 20.0);
    let width_yards: f64 = (field_width
        * if has_ball {
            0.66 + risk_tolerance * 0.16
        } else {
            0.48 + press_intensity * 0.10
        })
    .clamp(field_width * 0.42, field_width * 0.88);
    let shot_base: f64 =
        20.0 + if attacking_phase { 2.4 } else { 0.0 } + if trailing { 2.0 } else { 0.0 }
            - if leading { 1.2 } else { 0.0 };
    let shot_threshold_yards: f64 = shot_base.clamp(16.0, 25.0);

    TeamTacticalDirective {
        team,
        defensive_line_y,
        support_depth_yards,
        width_yards,
        press_intensity,
        pass_priority,
        carry_priority,
        shot_threshold_yards,
        risk_tolerance,
    }
}

fn belief_from_observation(obs: &SoccerPomdpObservation) -> BeliefState {
    let pressure = (1.0 - obs.nearest_opponent_distance / 18.0).clamp(0.0, 1.0);
    let shot_quality = if obs.shot_lane_open {
        (1.0 - obs.yards_to_goal / 45.0).clamp(0.0, 1.0)
    } else {
        0.0
    };
    BeliefState {
        possession_confidence: if obs.has_ball { 0.98 } else { 0.45 },
        pressure,
        pass_lane_open: (obs.nearest_teammate_distance / 24.0).clamp(0.0, 1.0),
        shot_quality,
    }
}

fn soccer_transition_reward(
    player: &PlayerAgent,
    decision: &AgentDecisionTrace,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    score_home_before: u32,
    score_away_before: u32,
    score_home_after: u32,
    score_away_after: u32,
) -> f64 {
    let (before_for, before_against, after_for, after_against) = match player.team {
        Team::Home => (
            score_home_before,
            score_away_before,
            score_home_after,
            score_away_after,
        ),
        Team::Away => (
            score_away_before,
            score_home_before,
            score_away_after,
            score_home_after,
        ),
    };
    let mut reward = (after_for as f64 - before_for as f64) * 10.0
        - (after_against as f64 - before_against as f64) * 8.0;

    let progress = (after.ball.position.y - before.ball.position.y) * player.team.attack_dir();
    reward += progress.clamp(-12.0, 12.0) * 0.025;

    match (before.possession_team(), after.possession_team()) {
        (Some(t0), Some(t1)) if t0 != player.team && t1 == player.team => reward += 0.65,
        (Some(t0), Some(t1)) if t0 == player.team && t1 != player.team => reward -= 0.55,
        (_, Some(t1)) if t1 == player.team => reward += 0.04,
        (_, Some(_)) => reward -= 0.03,
        _ => {}
    }

    if after.ball.holder == Some(player.id) {
        reward += 0.08;
    }
    if decision.action == "pass" && after.possession_team() == Some(player.team) {
        reward += 0.08;
    }
    if decision.action == "shoot"
        && decision.observation.shot_lane_open
        && decision.observation.yards_to_goal <= 20.0
    {
        reward += 0.15;
    }

    let next_obs = after.observation_for(player.id);
    reward += (next_obs.open_space_score - decision.observation.open_space_score)
        .clamp(-15.0, 15.0)
        * 0.004;
    reward -= next_obs.pressure_like_penalty() * 0.015;
    reward
}

impl SoccerPomdpObservation {
    fn pressure_like_penalty(&self) -> f64 {
        (18.0 - self.nearest_opponent_distance).max(0.0)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchEvent {
    pub tick: u64,
    pub clock_seconds: f64,
    pub kind: String,
    pub team: Option<Team>,
    pub player_id: Option<usize>,
    pub description: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OfficialSnapshot {
    pub id: usize,
    pub kind: OfficialKind,
    pub position: Vec2,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchFrame {
    pub tick: u64,
    pub clock_seconds: f64,
    pub ball: BallState,
    pub players: Vec<PlayerSnapshot>,
    pub officials: Vec<OfficialSnapshot>,
    pub score_home: u32,
    pub score_away: u32,
    pub phase: TacticalPhase,
    pub home_directive: TeamTacticalDirective,
    pub away_directive: TeamTacticalDirective,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SimulationTrace {
    pub config: MatchConfig,
    pub summary: MatchSummary,
    pub frames: Vec<MatchFrame>,
    pub events: Vec<MatchEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningDataset {
    pub config: MatchConfig,
    pub summary: MatchSummary,
    pub transitions: Vec<SoccerLearningTransition>,
    pub events: Vec<MatchEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerTrackingDataset {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub config: MatchConfig,
    #[serde(default)]
    pub frames: Vec<SoccerTrackingFrame>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerTrackingFrame {
    #[serde(default)]
    pub tick: u64,
    #[serde(default)]
    pub clock_seconds: f64,
    #[serde(default)]
    pub ball_position: Vec2,
    #[serde(default)]
    pub ball_velocity: Option<Vec2>,
    #[serde(default)]
    pub ball_holder: Option<usize>,
    #[serde(default)]
    pub last_touch_team: Option<Team>,
    #[serde(default)]
    pub score_home: Option<u32>,
    #[serde(default)]
    pub score_away: Option<u32>,
    #[serde(default)]
    pub players: Vec<SoccerTrackingPlayerSample>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerTrackingPlayerSample {
    pub id: usize,
    #[serde(default)]
    pub name: Option<String>,
    pub team: Team,
    pub role: PlayerRole,
    #[serde(default)]
    pub shirt: Option<u8>,
    pub position: Vec2,
    #[serde(default)]
    pub velocity: Option<Vec2>,
    #[serde(default)]
    pub home_position: Option<Vec2>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerPolicyArtifact {
    pub config: MatchConfig,
    pub summary: MatchSummary,
    pub transition_count: usize,
    pub options: SoccerQPolicyOptions,
    pub entries: Vec<SoccerQEntry>,
    pub events: Vec<MatchEvent>,
}

impl SoccerTrackingDataset {
    pub fn validate(&self) -> Result<(), String> {
        if self.config.dt_seconds <= 0.0 {
            return Err("tracking dataset config.dtSeconds must be positive".to_string());
        }
        if self.config.field_length_yards <= 0.0 || self.config.field_width_yards <= 0.0 {
            return Err("tracking dataset field dimensions must be positive".to_string());
        }
        if self.frames.len() < 2 {
            return Err("tracking dataset needs at least two frames".to_string());
        }
        for (idx, frame) in self.frames.iter().enumerate() {
            if frame.players.is_empty() {
                return Err(format!("tracking frame {idx} has no players"));
            }
            if let Some(holder) = frame.ball_holder {
                if !frame.players.iter().any(|p| p.id == holder) {
                    return Err(format!(
                        "tracking frame {idx} ballHolder {holder} is missing from players"
                    ));
                }
            }
        }
        Ok(())
    }

    pub fn to_learning_dataset(&self) -> Result<SoccerLearningDataset, String> {
        soccer_tracking_dataset_to_learning_dataset(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchSummary {
    pub score_home: u32,
    pub score_away: u32,
    pub ticks: u64,
    pub simulated_seconds: f64,
    pub stats: MatchStats,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ControllerAssignment {
    pub controller_slot: usize,
    pub player_id: usize,
    pub player_name: String,
    pub team: Team,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerControllerAssignmentRequest {
    pub controller_slot: usize,
    pub player_id: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerControllerAssignmentResponse {
    pub controller_assignments: Vec<ControllerAssignment>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerStepRequest {
    #[serde(default)]
    pub inputs: Vec<HumanInputFrame>,
    #[serde(default)]
    pub ticks: u64,
    #[serde(default)]
    pub record_every_ticks: Option<u64>,
}

impl Default for SoccerStepRequest {
    fn default() -> Self {
        SoccerStepRequest {
            inputs: Vec::new(),
            ticks: 1,
            record_every_ticks: Some(1),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerStepResponse {
    pub frame: MatchFrame,
    pub frames: Vec<MatchFrame>,
    pub events: Vec<MatchEvent>,
    pub learning_transitions: Vec<SoccerLearningTransition>,
    pub summary: MatchSummary,
    pub controller_assignments: Vec<ControllerAssignment>,
    pub accepted_inputs: usize,
    pub done: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLiveStateResponse {
    pub config: MatchConfig,
    pub frame: MatchFrame,
    pub summary: MatchSummary,
    pub controller_assignments: Vec<ControllerAssignment>,
    pub done: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerInputAck {
    pub accepted_inputs: usize,
    pub queued: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLiveServerConfig {
    pub host: String,
    pub port: u16,
    pub match_config: MatchConfig,
}

impl Default for SoccerLiveServerConfig {
    fn default() -> Self {
        SoccerLiveServerConfig {
            host: "127.0.0.1".to_string(),
            port: 5055,
            match_config: MatchConfig::default(),
        }
    }
}

pub struct SoccerMatch {
    pub config: MatchConfig,
    pub tick: u64,
    pub clock_seconds: f64,
    pub players: Vec<PlayerAgent>,
    pub officials: Vec<OfficialAgent>,
    pub ball: BallAgent,
    pub shared_positions: SharedPlayerPositions,
    pub score_home: u32,
    pub score_away: u32,
    pub stats: MatchStats,
    pub events: Vec<MatchEvent>,
    pub learning_transitions: Vec<SoccerLearningTransition>,
    pub learned_policy: Option<SoccerQPolicy>,
    pub human_inputs: SharedHumanInputs,
    pub central_brain: CentralBrain,
    rng: SeededRandom,
    pending_pass: Option<PendingPass>,
    pending_shot: Option<PendingShot>,
}

impl SoccerMatch {
    pub fn default_11v11(config: MatchConfig) -> Self {
        let mut rng = mulberry32(config.seed);
        let players = default_players(&config, &mut rng);
        let officials = vec![
            OfficialAgent {
                id: 22,
                kind: OfficialKind::CenterReferee,
                position: Vec2::new(
                    config.field_width_yards * 0.5,
                    config.field_length_yards * 0.5,
                ),
                velocity: Vec2::zero(),
            },
            OfficialAgent {
                id: 23,
                kind: OfficialKind::AssistantRefereeNear,
                position: Vec2::new(1.5, config.field_length_yards * 0.5),
                velocity: Vec2::zero(),
            },
            OfficialAgent {
                id: 24,
                kind: OfficialKind::AssistantRefereeFar,
                position: Vec2::new(
                    config.field_width_yards - 1.5,
                    config.field_length_yards * 0.5,
                ),
                velocity: Vec2::zero(),
            },
        ];
        let kickoff = players
            .iter()
            .find(|p| p.team == Team::Home && p.role == PlayerRole::Midfielder)
            .map(|p| p.id);
        let shared_positions = SharedPlayerPositions::default();
        shared_positions.sync_from_players(&players, 0, 0.0);
        SoccerMatch {
            config: config.clone(),
            tick: 0,
            clock_seconds: 0.0,
            players,
            officials,
            ball: BallAgent::new(
                BALL_AGENT_ID,
                BallState {
                    position: Vec2::new(
                        config.field_width_yards * 0.5,
                        config.field_length_yards * 0.5,
                    ),
                    velocity: Vec2::zero(),
                    acceleration: Vec2::zero(),
                    holder: kickoff,
                    last_touch_team: Some(Team::Home),
                },
            ),
            shared_positions,
            score_home: 0,
            score_away: 0,
            stats: MatchStats::default(),
            events: Vec::new(),
            learning_transitions: Vec::new(),
            learned_policy: None,
            human_inputs: SharedHumanInputs::new(),
            central_brain: CentralBrain::default(),
            rng,
            pending_pass: None,
            pending_shot: None,
        }
    }

    pub fn with_human_inputs(mut self, human_inputs: SharedHumanInputs) -> Self {
        self.human_inputs = human_inputs;
        self
    }

    pub fn with_learned_policy(mut self, learned_policy: SoccerQPolicy) -> Self {
        self.learned_policy = Some(learned_policy);
        self
    }

    pub fn set_learned_policy(&mut self, learned_policy: SoccerQPolicy) {
        self.learned_policy = Some(learned_policy);
    }

    pub fn learned_policy(&self) -> Option<&SoccerQPolicy> {
        self.learned_policy.as_ref()
    }

    pub fn learned_policy_mut(&mut self) -> Option<&mut SoccerQPolicy> {
        self.learned_policy.as_mut()
    }

    pub fn summary(&self) -> MatchSummary {
        MatchSummary {
            score_home: self.score_home,
            score_away: self.score_away,
            ticks: self.tick,
            simulated_seconds: self.clock_seconds,
            stats: self.stats.clone(),
        }
    }

    pub fn is_done(&self) -> bool {
        self.tick >= self.config.total_ticks()
    }

    pub fn controller_assignments(&self) -> Vec<ControllerAssignment> {
        let mut assignments = self
            .players
            .iter()
            .filter_map(|p| {
                p.controller_slot
                    .map(|controller_slot| ControllerAssignment {
                        controller_slot,
                        player_id: p.id,
                        player_name: p.name.clone(),
                        team: p.team,
                    })
            })
            .collect::<Vec<_>>();
        assignments.sort_by_key(|a| a.controller_slot);
        assignments
    }

    pub fn assign_controller_slot(
        &mut self,
        controller_slot: usize,
        player_id: Option<usize>,
    ) -> Result<(), String> {
        let slot_count = self.config.human_slots();
        if controller_slot >= slot_count {
            let range = if slot_count == 0 {
                "no configured slots".to_string()
            } else {
                format!("configured slots 0..{}", slot_count - 1)
            };
            return Err(format!(
                "controller slot {controller_slot} is outside {range}"
            ));
        }

        let target_idx = if let Some(player_id) = player_id {
            Some(
                self.players
                    .iter()
                    .position(|p| p.id == player_id)
                    .ok_or_else(|| format!("player {player_id} does not exist"))?,
            )
        } else {
            None
        };

        for player in &mut self.players {
            if player.controller_slot == Some(controller_slot) {
                player.controller_slot = None;
            }
        }

        if let Some(target_idx) = target_idx {
            self.players[target_idx].controller_slot = Some(controller_slot);
        }

        Ok(())
    }

    pub fn run_time_step(&mut self) {
        if self.is_done() {
            return;
        }
        let brain_input_snapshot = WorldSnapshot::from_match(self);
        let score_home_before = self.score_home;
        let score_away_before = self.score_away;
        self.central_brain.run_time_step(&brain_input_snapshot);
        let snapshot = WorldSnapshot::from_match(self);
        let latest_inputs = self.human_inputs.drain_latest_by_slot();
        let ball_velocity_before = self.ball.velocity;

        let mut actor_order: Vec<usize> = (0..self.players.len() + self.officials.len()).collect();
        fisher_yates_shuffle(&mut actor_order, &mut self.rng);

        let mut intents = Vec::new();
        for actor in actor_order {
            if actor < self.players.len() {
                let input = self.players[actor]
                    .controller_slot
                    .and_then(|slot| latest_inputs.get(&slot))
                    .filter(|frame| frame.player_id.is_none() || frame.player_id == Some(actor));
                let learned_action = self
                    .learned_policy
                    .as_ref()
                    .and_then(|policy| policy.best_action_for_snapshot(&snapshot, actor));
                let intent = self.players[actor].run_time_step(
                    &snapshot,
                    input,
                    learned_action.as_deref(),
                    &mut self.rng,
                );
                intents.push(intent);
            } else {
                let official_idx = actor - self.players.len();
                self.officials[official_idx].run_time_step(&snapshot, &mut self.rng);
            }
        }

        for intent in intents {
            self.apply_player_intent(intent);
        }
        self.resolve_player_collisions();
        self.integrate_ball();
        self.ball
            .update_acceleration_from(ball_velocity_before, self.config.dt_seconds);
        self.clock_seconds += self.config.dt_seconds;
        self.tick += 1;
        self.record_player_position_histories();
        self.record_ball_position_history();
        self.shared_positions
            .sync_from_players(&self.players, self.tick, self.clock_seconds);
        let next_snapshot = WorldSnapshot::from_match(self);
        let learning_start = self.learning_transitions.len();
        self.record_learning_transitions(
            &snapshot,
            &next_snapshot,
            score_home_before,
            score_away_before,
        );
        if self.learned_policy.is_some() {
            let new_transitions = self.learning_transitions[learning_start..].to_vec();
            if let Some(policy) = &mut self.learned_policy {
                policy.train(&new_transitions);
            }
        }
    }

    pub fn to_frame(&self) -> MatchFrame {
        let snapshot = WorldSnapshot::from_match(self);
        MatchFrame {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            ball: self.ball.to_state(),
            players: snapshot.players,
            officials: self
                .officials
                .iter()
                .map(|o| OfficialSnapshot {
                    id: o.id,
                    kind: o.kind,
                    position: o.position,
                })
                .collect(),
            score_home: self.score_home,
            score_away: self.score_away,
            phase: self.central_brain.phase,
            home_directive: self.central_brain.home_directive.clone(),
            away_directive: self.central_brain.away_directive.clone(),
        }
    }

    fn apply_player_intent(&mut self, intent: PlayerIntent) {
        if intent.player_id >= self.players.len() {
            return;
        }
        let player_id = intent.player_id;
        let player_pos = self.players[player_id].position;
        let player_team = self.players[player_id].team;
        match intent.action {
            SoccerAction::HoldShape => {
                let target = self.players[player_id].home_position;
                self.move_player_towards(player_id, target, intent.sprint);
            }
            SoccerAction::MoveTo(target) => {
                self.move_player_towards(player_id, target, intent.sprint);
            }
            SoccerAction::Dribble(target) => {
                self.move_player_towards(player_id, target, true);
                if self.ball.holder == Some(player_id) {
                    let dir = (target - player_pos).normalized();
                    self.ball.position = self.players[player_id].position + dir * 0.9;
                    self.ball.velocity = self.players[player_id].velocity;
                }
            }
            SoccerAction::Pass {
                target_player,
                power,
            } => {
                if self.ball.holder == Some(player_id) {
                    let snapshot = WorldSnapshot::from_match(self);
                    let observation = snapshot.observation_for(player_id);
                    let target_id = target_player.or_else(|| snapshot.best_pass_target(player_id));
                    let target = target_id
                        .and_then(|id| self.players.iter().find(|p| p.id == id).map(|p| p.position))
                        .unwrap_or_else(|| {
                            Vec2::new(player_pos.x, player_pos.y + 18.0 * player_team.attack_dir())
                                .clamp_to_pitch(
                                    self.config.field_width_yards,
                                    self.config.field_length_yards,
                                )
                        });
                    let distance = player_pos.distance(target);
                    let pressure = pressure_from_observation(&observation);
                    let aimed_target = noisy_pass_target(
                        player_pos,
                        target,
                        self.players[player_id].skills.passing,
                        pressure,
                        distance,
                        &mut self.rng,
                    )
                    .clamp_to_pitch(
                        self.config.field_width_yards,
                        self.config.field_length_yards,
                    );
                    let speed = 16.0 + 16.0 * power.clamp(0.0, 1.0);
                    self.ball.holder = None;
                    self.ball.position = player_pos;
                    self.ball.velocity = (aimed_target - player_pos).normalized() * speed;
                    self.ball.last_touch_team = Some(player_team);
                    let offside = target_id
                        .and_then(|target| snapshot.pending_offside_for_pass(player_id, target));
                    self.pending_pass = Some(PendingPass {
                        team: player_team,
                        from: player_id,
                        target: target_id,
                        offside,
                    });
                    self.pending_shot = None;
                    self.stat_pass_attempt(player_team);
                }
                self.move_player_towards(player_id, self.players[player_id].home_position, false);
            }
            SoccerAction::Shoot { power } => {
                if self.ball.holder == Some(player_id) {
                    let snapshot = WorldSnapshot::from_match(self);
                    let observation = snapshot.observation_for(player_id);
                    let goal = Vec2::new(
                        noisy_shot_target_x(
                            self.config.field_width_yards * 0.5,
                            self.config.goal_width_yards,
                            self.players[player_id].skills.shooting,
                            pressure_from_observation(&observation),
                            observation.yards_to_goal,
                            &mut self.rng,
                        ),
                        player_team.goal_y(self.config.field_length_yards),
                    );
                    let speed = 28.0 + 18.0 * power.clamp(0.0, 1.0);
                    self.ball.holder = None;
                    self.ball.position = player_pos;
                    self.ball.velocity = (goal - player_pos).normalized() * speed;
                    self.ball.last_touch_team = Some(player_team);
                    self.pending_pass = None;
                    self.pending_shot = Some(PendingShot {
                        team: player_team,
                        shooter: player_id,
                    });
                    self.stat_shot(player_team);
                    self.events.push(MatchEvent {
                        tick: self.tick,
                        clock_seconds: self.clock_seconds,
                        kind: "shot".to_string(),
                        team: Some(player_team),
                        player_id: Some(player_id),
                        description: format!("{} shot", self.players[player_id].name),
                    });
                }
                self.move_player_towards(player_id, self.players[player_id].home_position, false);
            }
            SoccerAction::Tackle { target_player } => {
                self.stats.tackles += 1;
                if self.ball.holder == Some(target_player)
                    && self.players[target_player].position.distance(player_pos) < 2.2
                {
                    let contact_distance =
                        self.players[target_player].position.distance(player_pos);
                    let contact_speed = (self.players[player_id].velocity
                        - self.players[target_player].velocity)
                        .len();
                    let foul_probability = tackle_foul_probability(
                        &self.players[player_id].skills,
                        &self.players[target_player].skills,
                        contact_distance,
                        contact_speed,
                    );
                    if self.rng.next_float() < foul_probability {
                        self.call_foul(
                            player_team,
                            player_id,
                            target_player,
                            self.players[target_player].position,
                        );
                        return;
                    }
                    let success_probability = tackle_success_probability(
                        &self.players[player_id].skills,
                        &self.players[target_player].skills,
                    );
                    if self.rng.next_float() < success_probability {
                        self.ball.holder = Some(player_id);
                        self.ball.last_touch_team = Some(player_team);
                        self.events.push(MatchEvent {
                            tick: self.tick,
                            clock_seconds: self.clock_seconds,
                            kind: "tackle".to_string(),
                            team: Some(player_team),
                            player_id: Some(player_id),
                            description: format!("{} won a tackle", self.players[player_id].name),
                        });
                    }
                }
                self.move_player_towards(player_id, self.ball.position, true);
            }
        }
    }

    fn move_player_towards(&mut self, player_id: usize, target: Vec2, sprint: bool) {
        let dt = self.config.dt_seconds;
        let p = &mut self.players[player_id];
        let previous_velocity = p.velocity;
        let fatigue_factor = (0.72 + 0.28 * p.skills.stamina - p.fatigue * 0.12).clamp(0.58, 1.05);
        let speed = p.skills.top_speed_yps * fatigue_factor * if sprint { 1.08 } else { 0.86 };
        let desired = (target - p.position).normalized() * speed;
        p.velocity = approach_velocity(p.velocity, desired, p.skills.acceleration_yps2, dt);
        p.acceleration = if dt > 0.0 {
            (p.velocity - previous_velocity) / dt
        } else {
            Vec2::zero()
        };
        p.position += p.velocity * dt;
        p.position = p.position.clamp_to_pitch(
            self.config.field_width_yards,
            self.config.field_length_yards,
        );
        p.fatigue = (p.fatigue + if sprint { 0.0018 } else { -0.0007 }).clamp(0.0, 1.0);
    }

    fn record_player_position_histories(&mut self) {
        for player in &mut self.players {
            player.record_position_history();
        }
    }

    fn record_ball_position_history(&mut self) {
        self.ball
            .record_position_history(self.tick, self.clock_seconds);
    }

    fn resolve_player_collisions(&mut self) {
        let min_sep = PLAYER_BODY_RADIUS_YARDS * 2.0;
        let width = self.config.field_width_yards;
        let length = self.config.field_length_yards;
        for i in 0..self.players.len() {
            for j in i + 1..self.players.len() {
                let (left, right) = self.players.split_at_mut(j);
                let a = &mut left[i];
                let b = &mut right[0];
                let delta = b.position - a.position;
                let dist = delta.len();
                if dist >= min_sep {
                    continue;
                }

                let normal = if dist <= 1e-9 {
                    deterministic_separation_normal(a.id, b.id)
                } else {
                    delta / dist
                };
                let overlap = min_sep - dist.max(1e-9);
                let push = normal * (overlap * 0.5);
                a.position = (a.position - push).clamp_to_pitch(width, length);
                b.position = (b.position + push).clamp_to_pitch(width, length);

                let a_normal_speed = dot(a.velocity, normal);
                let b_normal_speed = dot(b.velocity, normal);
                if a_normal_speed > b_normal_speed {
                    let impulse = (a_normal_speed - b_normal_speed) * PLAYER_COLLISION_DAMPING;
                    a.velocity = a.velocity - normal * impulse;
                    b.velocity += normal * impulse;
                }
            }
        }
    }

    fn integrate_ball(&mut self) {
        let previous_velocity = self.ball.velocity;
        let context = BallStepContext {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            dt_seconds: self.config.dt_seconds,
            ball_drag_per_tick: self.config.ball_drag_per_tick,
            ball_stop_speed_yps: self.config.ball_stop_speed_yps,
            field_length: self.config.field_length_yards,
            field_width: self.config.field_width_yards,
            goal_width: self.config.goal_width_yards,
            players: &self.players,
            pending_pass: self.pending_pass.clone(),
            pending_shot: self.pending_shot.clone(),
        };
        let outcome = self.ball.run_time_step(context, &mut self.rng);
        self.apply_ball_outcome(outcome);
        self.ball
            .update_acceleration_from(previous_velocity, self.config.dt_seconds);
    }

    fn nearest_ball_controller(&mut self) -> Option<(usize, Team)> {
        nearest_ball_controller_for(
            self.ball.position,
            self.ball.velocity,
            &self.players,
            &mut self.rng,
        )
    }

    fn apply_ball_outcome(&mut self, outcome: BallStepOutcome) {
        match outcome {
            BallStepOutcome::None => {}
            BallStepOutcome::Controlled {
                holder,
                holder_team,
                possession_result,
            } => {
                if let Some(offside) = self
                    .pending_pass
                    .as_ref()
                    .and_then(|pass| pass.offside.clone())
                {
                    if offside.target == holder && offside.team == holder_team {
                        self.pending_pass = None;
                        self.call_offside(offside);
                        return;
                    }
                }
                match possession_result {
                    BallPossessionResult::PassCompleted(team) => {
                        self.pending_pass = None;
                        self.stat_pass_completed(team);
                    }
                    BallPossessionResult::Interception(team) => {
                        self.pending_pass = None;
                        self.stat_interception(team);
                    }
                    BallPossessionResult::LooseBallRecovery(team) => {
                        self.stat_loose_ball_recovery(team);
                    }
                }
            }
            BallStepOutcome::Save {
                shot,
                defending_team,
                keeper_id,
                save_position,
            } => {
                self.pending_shot = None;
                self.stat_shot_on_target(shot.team);
                self.stat_save(defending_team);
                if let Some(keeper) = self
                    .players
                    .iter_mut()
                    .find(|player| player.id == keeper_id)
                {
                    keeper.position = save_position;
                    keeper.velocity = Vec2::zero();
                    keeper.acceleration = Vec2::zero();
                }
                let keeper_name = self
                    .players
                    .iter()
                    .find(|player| player.id == keeper_id)
                    .map(|player| player.name.as_str())
                    .unwrap_or("Keeper");
                let shooter_name = self
                    .players
                    .iter()
                    .find(|player| player.id == shot.shooter)
                    .map(|player| player.name.as_str())
                    .unwrap_or("shooter");
                self.events.push(MatchEvent {
                    tick: self.tick,
                    clock_seconds: self.clock_seconds,
                    kind: "save".to_string(),
                    team: Some(defending_team),
                    player_id: Some(keeper_id),
                    description: format!("{keeper_name} saved a shot by {shooter_name}"),
                });
            }
            BallStepOutcome::Goal { scoring_team, shot } => {
                if let Some(shot) = shot {
                    self.pending_shot = None;
                    self.stat_shot_on_target(shot.team);
                }
                self.score_goal(scoring_team);
            }
            BallStepOutcome::Miss { shot } => {
                self.pending_shot = None;
                let shooter_name = self
                    .players
                    .iter()
                    .find(|player| player.id == shot.shooter)
                    .map(|player| player.name.as_str())
                    .unwrap_or("Player");
                self.events.push(MatchEvent {
                    tick: self.tick,
                    clock_seconds: self.clock_seconds,
                    kind: "miss".to_string(),
                    team: Some(shot.team),
                    player_id: Some(shot.shooter),
                    description: format!("{shooter_name} missed"),
                });
            }
            BallStepOutcome::OutOfPlay { restart, shot } => {
                self.pending_pass = None;
                self.pending_shot = None;
                if let Some(shot) = shot {
                    self.record_miss_event(shot);
                }
                self.apply_restart(restart);
            }
        }
    }

    fn record_miss_event(&mut self, shot: PendingShot) {
        let shooter_name = self
            .players
            .iter()
            .find(|player| player.id == shot.shooter)
            .map(|player| player.name.as_str())
            .unwrap_or("Player");
        self.events.push(MatchEvent {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            kind: "miss".to_string(),
            team: Some(shot.team),
            player_id: Some(shot.shooter),
            description: format!("{shooter_name} missed"),
        });
    }

    fn apply_restart(&mut self, restart: BallRestart) {
        self.stat_restart(restart.kind, restart.awarded_team);
        let restart_holder = self.nearest_player_on_team(restart.awarded_team, restart.position);
        if let Some(holder_id) = restart_holder {
            if let Some(holder) = self.players.iter_mut().find(|p| p.id == holder_id) {
                holder.position = restart.position;
                holder.velocity = Vec2::zero();
                holder.acceleration = Vec2::zero();
                holder.record_position_history();
            }
        }

        match restart.kind {
            BallRestartKind::FreeKick => {
                if let Some(center_ref) = self
                    .officials
                    .iter_mut()
                    .find(|official| official.kind == OfficialKind::CenterReferee)
                {
                    center_ref.position = restart.position;
                    center_ref.velocity = Vec2::zero();
                }
            }
            _ => {
                if let Some(assistant) = self.officials.iter_mut().find(|official| {
                    matches!(
                        official.kind,
                        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar
                    )
                }) {
                    assistant.position.y = restart.position.y;
                    assistant.velocity = Vec2::zero();
                }
            }
        }

        self.ball.position = restart.position;
        self.ball.velocity = Vec2::zero();
        self.ball.holder = restart_holder;
        self.ball.last_touch_team = Some(restart.awarded_team);
        self.ball
            .record_decision(self.tick, restart_kind_action(restart.kind));
        self.shared_positions
            .sync_from_players(&self.players, self.tick, self.clock_seconds);

        let taker = restart_holder
            .and_then(|id| self.players.iter().find(|player| player.id == id))
            .map(|player| player.name.as_str())
            .unwrap_or("Restart");
        self.events.push(MatchEvent {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            kind: restart_kind_action(restart.kind).to_string(),
            team: Some(restart.awarded_team),
            player_id: restart_holder,
            description: format!(
                "{} {} for {}",
                taker,
                restart_kind_label(restart.kind),
                restart.awarded_team.label()
            ),
        });
    }

    fn call_foul(&mut self, fouling_team: Team, fouler_id: usize, fouled_id: usize, spot: Vec2) {
        let awarded_team = fouling_team.other();
        let restart_spot = spot.clamp_to_pitch(
            self.config.field_width_yards,
            self.config.field_length_yards,
        );
        self.stat_foul(fouling_team);
        self.pending_pass = None;
        self.pending_shot = None;
        self.events.push(MatchEvent {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            kind: "foul".to_string(),
            team: Some(fouling_team),
            player_id: Some(fouler_id),
            description: format!(
                "{} fouled {}",
                self.players[fouler_id].name, self.players[fouled_id].name
            ),
        });
        self.apply_restart(BallRestart {
            kind: BallRestartKind::FreeKick,
            awarded_team,
            position: restart_spot,
        });
    }

    fn call_offside(&mut self, offside: PendingOffside) {
        let defending_team = offside.team.other();
        let restart_spot = offside.position.clamp_to_pitch(
            self.config.field_width_yards,
            self.config.field_length_yards,
        );
        let restart_holder = self.nearest_player_on_team(defending_team, restart_spot);

        if let Some(holder_id) = restart_holder {
            if let Some(holder) = self.players.iter_mut().find(|p| p.id == holder_id) {
                holder.position = restart_spot;
                holder.velocity = Vec2::zero();
                holder.acceleration = Vec2::zero();
                holder.record_position_history();
            }
        }

        let assistant_kind = if restart_spot.x < self.config.field_width_yards * 0.5 {
            OfficialKind::AssistantRefereeNear
        } else {
            OfficialKind::AssistantRefereeFar
        };
        if let Some(assistant) = self
            .officials
            .iter_mut()
            .find(|official| official.kind == assistant_kind)
        {
            assistant.position.y = restart_spot.y;
            assistant.velocity = Vec2::zero();
        }

        self.stat_offside(offside.team);
        self.ball.position = restart_spot;
        self.ball.velocity = Vec2::zero();
        self.ball.holder = restart_holder;
        self.ball.last_touch_team = Some(defending_team);
        self.ball.record_decision(self.tick, "offside");
        self.pending_shot = None;
        self.shared_positions
            .sync_from_players(&self.players, self.tick, self.clock_seconds);

        let target_name = self
            .players
            .iter()
            .find(|player| player.id == offside.target)
            .map(|player| player.name.as_str())
            .unwrap_or("runner");
        let passer_name = self
            .players
            .iter()
            .find(|player| player.id == offside.passer)
            .map(|player| player.name.as_str())
            .unwrap_or("passer");
        self.events.push(MatchEvent {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            kind: "offside".to_string(),
            team: Some(offside.team),
            player_id: Some(offside.target),
            description: format!(
                "{target_name} flagged offside on {passer_name}'s pass ({:.1} beyond line {:.1}, ball {:.1})",
                restart_spot.y, offside.second_last_defender_y, offside.ball_y
            ),
        });
    }

    fn nearest_player_on_team(&self, team: Team, point: Vec2) -> Option<usize> {
        self.players
            .iter()
            .filter(|player| player.team == team)
            .min_by(|a, b| {
                a.position
                    .distance(point)
                    .partial_cmp(&b.position.distance(point))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|player| player.id)
    }

    fn score_goal(&mut self, scoring_team: Team) {
        match scoring_team {
            Team::Home => self.score_home += 1,
            Team::Away => self.score_away += 1,
        }
        self.events.push(MatchEvent {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            kind: "goal".to_string(),
            team: Some(scoring_team),
            player_id: None,
            description: format!("{} goal", scoring_team.label()),
        });
        self.reset_after_goal(scoring_team.other());
    }

    fn reset_after_goal(&mut self, kickoff_team: Team) {
        for p in &mut self.players {
            p.position = p.home_position;
            p.velocity = Vec2::zero();
            p.acceleration = Vec2::zero();
            p.record_position_history();
        }
        let kickoff = self
            .players
            .iter()
            .filter(|p| p.team == kickoff_team)
            .min_by(|a, b| {
                a.position
                    .distance(Vec2::new(
                        self.config.field_width_yards * 0.5,
                        self.config.field_length_yards * 0.5,
                    ))
                    .partial_cmp(&b.position.distance(Vec2::new(
                        self.config.field_width_yards * 0.5,
                        self.config.field_length_yards * 0.5,
                    )))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id);
        self.ball.position = Vec2::new(
            self.config.field_width_yards * 0.5,
            self.config.field_length_yards * 0.5,
        );
        self.ball.velocity = Vec2::zero();
        self.ball.holder = kickoff;
        self.ball.last_touch_team = Some(kickoff_team);
        self.ball.record_decision(self.tick, "reset");
        self.pending_pass = None;
        self.pending_shot = None;
        self.shared_positions
            .sync_from_players(&self.players, self.tick, self.clock_seconds);
    }

    fn stat_shot(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.shots_home += 1,
            Team::Away => self.stats.shots_away += 1,
        }
    }

    fn stat_shot_on_target(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.shots_on_target_home += 1,
            Team::Away => self.stats.shots_on_target_away += 1,
        }
    }

    fn stat_save(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.saves_home += 1,
            Team::Away => self.stats.saves_away += 1,
        }
    }

    fn stat_pass_attempt(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.passes_attempted_home += 1,
            Team::Away => self.stats.passes_attempted_away += 1,
        }
    }

    fn stat_pass_completed(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.passes_completed_home += 1,
            Team::Away => self.stats.passes_completed_away += 1,
        }
    }

    fn stat_interception(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.interceptions_home += 1,
            Team::Away => self.stats.interceptions_away += 1,
        }
    }

    fn stat_loose_ball_recovery(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.loose_ball_recoveries_home += 1,
            Team::Away => self.stats.loose_ball_recoveries_away += 1,
        }
    }

    fn stat_offside(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.offsides_home += 1,
            Team::Away => self.stats.offsides_away += 1,
        }
    }

    fn stat_restart(&mut self, kind: BallRestartKind, team: Team) {
        match (kind, team) {
            (BallRestartKind::ThrowIn, Team::Home) => self.stats.throw_ins_home += 1,
            (BallRestartKind::ThrowIn, Team::Away) => self.stats.throw_ins_away += 1,
            (BallRestartKind::GoalKick, Team::Home) => self.stats.goal_kicks_home += 1,
            (BallRestartKind::GoalKick, Team::Away) => self.stats.goal_kicks_away += 1,
            (BallRestartKind::CornerKick, Team::Home) => self.stats.corner_kicks_home += 1,
            (BallRestartKind::CornerKick, Team::Away) => self.stats.corner_kicks_away += 1,
            (BallRestartKind::FreeKick, Team::Home) => self.stats.free_kicks_home += 1,
            (BallRestartKind::FreeKick, Team::Away) => self.stats.free_kicks_away += 1,
        }
    }

    fn stat_foul(&mut self, team: Team) {
        match team {
            Team::Home => self.stats.fouls_home += 1,
            Team::Away => self.stats.fouls_away += 1,
        }
    }

    fn goalkeeper_for(&self, team: Team) -> Option<usize> {
        goalkeeper_for_players(&self.players, team)
    }

    fn record_learning_transitions(
        &mut self,
        before: &WorldSnapshot,
        after: &WorldSnapshot,
        score_home_before: u32,
        score_away_before: u32,
    ) {
        let done = self.is_done();
        for player in &self.players {
            let Some(decision) = &player.last_decision else {
                continue;
            };
            let reward = soccer_transition_reward(
                player,
                decision,
                before,
                after,
                score_home_before,
                score_away_before,
                self.score_home,
                self.score_away,
            );
            self.learning_transitions.push(SoccerLearningTransition {
                tick: before.tick,
                player_id: player.id,
                team: player.team,
                role: player.role,
                state: decision.mdp_state.clone(),
                observation: decision.observation.clone(),
                belief: decision.belief.clone(),
                action: decision.action.clone(),
                reward,
                next_state: after.mdp_state(),
                next_observation: after.observation_for(player.id),
                done,
            });
        }
    }
}

pub struct SoccerRealtimeSession {
    sim: SoccerMatch,
    input_queue: SharedHumanInputs,
    emitted_event_cursor: usize,
    emitted_learning_cursor: usize,
}

impl SoccerRealtimeSession {
    pub fn new(config: MatchConfig) -> Self {
        let input_queue = SharedHumanInputs::new();
        let sim = SoccerMatch::default_11v11(config).with_human_inputs(input_queue.clone());
        SoccerRealtimeSession {
            sim,
            input_queue,
            emitted_event_cursor: 0,
            emitted_learning_cursor: 0,
        }
    }

    pub fn from_match(sim: SoccerMatch) -> Self {
        let input_queue = sim.human_inputs.clone();
        SoccerRealtimeSession {
            sim,
            input_queue,
            emitted_event_cursor: 0,
            emitted_learning_cursor: 0,
        }
    }

    pub fn input_queue(&self) -> SharedHumanInputs {
        self.input_queue.clone()
    }

    pub fn shared_positions(&self) -> SharedPlayerPositions {
        self.sim.shared_positions.clone()
    }

    pub fn push_human_input(&self, input: HumanInputFrame) -> bool {
        self.input_queue.push(input)
    }

    pub fn match_ref(&self) -> &SoccerMatch {
        &self.sim
    }

    pub fn match_mut(&mut self) -> &mut SoccerMatch {
        &mut self.sim
    }

    pub fn snapshot(&self) -> MatchFrame {
        self.sim.to_frame()
    }

    pub fn is_done(&self) -> bool {
        self.sim.is_done()
    }

    pub fn step_once(&mut self) -> SoccerStepResponse {
        self.step(SoccerStepRequest::default())
    }

    pub fn step(&mut self, request: SoccerStepRequest) -> SoccerStepResponse {
        let mut accepted_inputs = 0;
        for input in request.inputs {
            if self.input_queue.push(input) {
                accepted_inputs += 1;
            }
        }

        let ticks = request.ticks.max(1);
        let record_every = request.record_every_ticks.unwrap_or(1).max(1);
        let mut frames = Vec::new();
        for i in 0..ticks {
            if self.sim.is_done() {
                break;
            }
            self.sim.run_time_step();
            if (i + 1) % record_every == 0 || self.sim.is_done() {
                frames.push(self.sim.to_frame());
            }
        }
        if frames.is_empty() {
            frames.push(self.sim.to_frame());
        }

        let events = self.sim.events[self.emitted_event_cursor..].to_vec();
        self.emitted_event_cursor = self.sim.events.len();
        let learning_transitions =
            self.sim.learning_transitions[self.emitted_learning_cursor..].to_vec();
        self.emitted_learning_cursor = self.sim.learning_transitions.len();

        SoccerStepResponse {
            frame: self.sim.to_frame(),
            frames,
            events,
            learning_transitions,
            summary: self.sim.summary(),
            controller_assignments: self.sim.controller_assignments(),
            accepted_inputs,
            done: self.sim.is_done(),
        }
    }

    pub fn step_json(&mut self, request_json: &str) -> Result<String, String> {
        let req: SoccerStepRequest =
            serde_json::from_str(request_json).map_err(|e| format!("parse step request: {e}"))?;
        let resp = self.step(req);
        serde_json::to_string(&resp).map_err(|e| format!("serialize step response: {e}"))
    }

    pub fn assign_controller_slot(
        &mut self,
        request: SoccerControllerAssignmentRequest,
    ) -> Result<SoccerControllerAssignmentResponse, String> {
        self.sim
            .assign_controller_slot(request.controller_slot, request.player_id)?;
        Ok(SoccerControllerAssignmentResponse {
            controller_assignments: self.sim.controller_assignments(),
        })
    }

    pub fn state_response(&self) -> SoccerLiveStateResponse {
        SoccerLiveStateResponse {
            config: self.sim.config.clone(),
            frame: self.sim.to_frame(),
            summary: self.sim.summary(),
            controller_assignments: self.sim.controller_assignments(),
            done: self.sim.is_done(),
        }
    }
}

pub struct SoccerLiveServer {
    config: SoccerLiveServerConfig,
    session: Arc<Mutex<SoccerRealtimeSession>>,
    input_queue: SharedHumanInputs,
}

impl SoccerLiveServer {
    pub fn new(config: SoccerLiveServerConfig) -> Self {
        let session = SoccerRealtimeSession::new(config.match_config.clone());
        let input_queue = session.input_queue();
        SoccerLiveServer {
            config,
            session: Arc::new(Mutex::new(session)),
            input_queue,
        }
    }

    pub fn local_url(&self) -> String {
        format!("http://{}:{}/", self.config.host, self.config.port)
    }

    pub fn run(self) -> std::io::Result<()> {
        let listener = TcpListener::bind((self.config.host.as_str(), self.config.port))?;
        println!("# Live soccer UI: {}", self.local_url());
        for stream in listener.incoming() {
            let stream = stream?;
            let session = Arc::clone(&self.session);
            let input_queue = self.input_queue.clone();
            thread::spawn(move || {
                let _ = handle_live_soccer_stream(stream, session, input_queue);
            });
        }
        Ok(())
    }
}

pub fn run_live_soccer_server(config: SoccerLiveServerConfig) -> std::io::Result<()> {
    SoccerLiveServer::new(config).run()
}

fn handle_live_soccer_stream(
    mut stream: TcpStream,
    session: Arc<Mutex<SoccerRealtimeSession>>,
    input_queue: SharedHumanInputs,
) -> std::io::Result<()> {
    let raw = read_http_request(&mut stream)?;
    let response = handle_live_soccer_request(&raw, &session, &input_queue);
    stream.write_all(&response.to_bytes())?;
    stream.flush()
}

fn read_http_request(stream: &mut TcpStream) -> std::io::Result<String> {
    stream.set_read_timeout(Some(Duration::from_millis(250)))?;
    let mut data = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        match stream.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => {
                data.extend_from_slice(&buf[..n]);
                if request_body_complete(&data) || data.len() > 1_048_576 {
                    break;
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock || e.kind() == ErrorKind::TimedOut => {
                break;
            }
            Err(e) => return Err(e),
        }
    }
    Ok(String::from_utf8_lossy(&data).into_owned())
}

fn request_body_complete(data: &[u8]) -> bool {
    let Some(split) = find_header_body_split(data) else {
        return false;
    };
    let header = String::from_utf8_lossy(&data[..split]);
    let content_len = http_content_length(&header).unwrap_or(0);
    data.len().saturating_sub(split + 4) >= content_len
}

fn find_header_body_split(data: &[u8]) -> Option<usize> {
    data.windows(4).position(|w| w == b"\r\n\r\n")
}

fn http_content_length(header: &str) -> Option<usize> {
    header.lines().find_map(|line| {
        let (name, value) = line.split_once(':')?;
        if name.trim().eq_ignore_ascii_case("content-length") {
            value.trim().parse().ok()
        } else {
            None
        }
    })
}

#[derive(Clone, Debug)]
struct LiveHttpRequest<'a> {
    method: &'a str,
    path: &'a str,
    body: &'a str,
}

fn parse_live_http_request(raw: &str) -> Result<LiveHttpRequest<'_>, String> {
    let (head, body) = raw
        .split_once("\r\n\r\n")
        .ok_or_else(|| "missing HTTP header terminator".to_string())?;
    let first = head
        .lines()
        .next()
        .ok_or_else(|| "missing request line".to_string())?;
    let mut parts = first.split_whitespace();
    let method = parts.next().ok_or_else(|| "missing method".to_string())?;
    let path = parts.next().ok_or_else(|| "missing path".to_string())?;
    Ok(LiveHttpRequest { method, path, body })
}

#[derive(Clone, Debug)]
struct LiveHttpResponse {
    status: u16,
    reason: &'static str,
    content_type: &'static str,
    body: String,
}

impl LiveHttpResponse {
    fn html(body: String) -> Self {
        LiveHttpResponse {
            status: 200,
            reason: "OK",
            content_type: "text/html; charset=utf-8",
            body,
        }
    }

    fn json<T: Serialize>(value: &T) -> Self {
        match serde_json::to_string(value) {
            Ok(body) => LiveHttpResponse {
                status: 200,
                reason: "OK",
                content_type: "application/json; charset=utf-8",
                body,
            },
            Err(e) => Self::error(
                500,
                "Internal Server Error",
                &format!("serialize json: {e}"),
            ),
        }
    }

    fn error(status: u16, reason: &'static str, message: &str) -> Self {
        let body = serde_json::json!({ "ok": false, "error": message }).to_string();
        LiveHttpResponse {
            status,
            reason,
            content_type: "application/json; charset=utf-8",
            body,
        }
    }

    fn options() -> Self {
        LiveHttpResponse {
            status: 204,
            reason: "No Content",
            content_type: "text/plain; charset=utf-8",
            body: String::new(),
        }
    }

    fn to_bytes(&self) -> Vec<u8> {
        let headers = format!(
            "HTTP/1.1 {} {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nCache-Control: no-store\r\nAccess-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: content-type\r\nAccess-Control-Allow-Methods: GET,POST,OPTIONS\r\nConnection: close\r\n\r\n",
            self.status,
            self.reason,
            self.content_type,
            self.body.as_bytes().len()
        );
        let mut out = headers.into_bytes();
        out.extend_from_slice(self.body.as_bytes());
        out
    }
}

fn handle_live_soccer_request(
    raw: &str,
    session: &Arc<Mutex<SoccerRealtimeSession>>,
    input_queue: &SharedHumanInputs,
) -> LiveHttpResponse {
    let req = match parse_live_http_request(raw) {
        Ok(req) => req,
        Err(e) => return LiveHttpResponse::error(400, "Bad Request", &e),
    };
    let path = req.path.split('?').next().unwrap_or(req.path);
    match (req.method, path) {
        ("OPTIONS", _) => LiveHttpResponse::options(),
        ("GET", "/") | ("GET", "/soccer/live") => LiveHttpResponse::html(soccer_live_page_html()),
        ("GET", "/api/state") => {
            let guard = match session.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    return LiveHttpResponse::error(
                        500,
                        "Internal Server Error",
                        "soccer session lock poisoned",
                    )
                }
            };
            LiveHttpResponse::json(&guard.state_response())
        }
        ("POST", "/api/input") => match parse_human_input_payload(req.body) {
            Ok(inputs) => {
                let mut count = 0;
                for input in inputs {
                    if input_queue.push(input) {
                        count += 1;
                    }
                }
                LiveHttpResponse::json(&SoccerInputAck {
                    accepted_inputs: count,
                    queued: true,
                })
            }
            Err(e) => LiveHttpResponse::error(400, "Bad Request", &e),
        },
        ("POST", "/api/step") => {
            let step_req = match serde_json::from_str::<SoccerStepRequest>(req.body) {
                Ok(req) => req,
                Err(e) => {
                    return LiveHttpResponse::error(
                        400,
                        "Bad Request",
                        &format!("parse step request: {e}"),
                    )
                }
            };
            let mut guard = match session.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    return LiveHttpResponse::error(
                        500,
                        "Internal Server Error",
                        "soccer session lock poisoned",
                    )
                }
            };
            LiveHttpResponse::json(&guard.step(step_req))
        }
        ("POST", "/api/assign") => {
            let assignment_req =
                match serde_json::from_str::<SoccerControllerAssignmentRequest>(req.body) {
                    Ok(req) => req,
                    Err(e) => {
                        return LiveHttpResponse::error(
                            400,
                            "Bad Request",
                            &format!("parse controller assignment request: {e}"),
                        )
                    }
                };
            let mut guard = match session.lock() {
                Ok(guard) => guard,
                Err(_) => {
                    return LiveHttpResponse::error(
                        500,
                        "Internal Server Error",
                        "soccer session lock poisoned",
                    )
                }
            };
            match guard.assign_controller_slot(assignment_req) {
                Ok(response) => LiveHttpResponse::json(&response),
                Err(e) => LiveHttpResponse::error(400, "Bad Request", &e),
            }
        }
        _ => LiveHttpResponse::error(404, "Not Found", "unknown soccer live route"),
    }
}

fn parse_human_input_payload(body: &str) -> Result<Vec<HumanInputFrame>, String> {
    if let Ok(input) = serde_json::from_str::<HumanInputFrame>(body) {
        return Ok(vec![input]);
    }
    serde_json::from_str::<Vec<HumanInputFrame>>(body)
        .map_err(|e| format!("parse human input payload: {e}"))
}

pub fn run_default_simulation() -> SimulationTrace {
    run_simulation(MatchConfig::default(), 5)
}

pub fn run_simulation(config: MatchConfig, record_every_ticks: u64) -> SimulationTrace {
    let mut sim = SoccerMatch::default_11v11(config.clone());
    let mut frames = vec![sim.to_frame()];
    let total_ticks = config.total_ticks();
    let record_every_ticks = record_every_ticks.max(1);
    for _ in 0..total_ticks {
        sim.run_time_step();
        if sim.tick % record_every_ticks == 0 || sim.tick == total_ticks {
            frames.push(sim.to_frame());
        }
    }
    SimulationTrace {
        config,
        summary: sim.summary(),
        frames,
        events: sim.events,
    }
}

pub fn run_learning_episode(config: MatchConfig) -> SoccerLearningDataset {
    let mut sim = SoccerMatch::default_11v11(config.clone());
    let total_ticks = config.total_ticks();
    for _ in 0..total_ticks {
        sim.run_time_step();
    }
    SoccerLearningDataset {
        config,
        summary: sim.summary(),
        transitions: sim.learning_transitions,
        events: sim.events,
    }
}

pub fn soccer_tracking_dataset_from_json(raw: &str) -> Result<SoccerTrackingDataset, String> {
    serde_json::from_str(raw).map_err(|e| format!("parse soccer tracking dataset: {e}"))
}

pub fn soccer_tracking_dataset_to_learning_dataset(
    tracking: &SoccerTrackingDataset,
) -> Result<SoccerLearningDataset, String> {
    tracking.validate()?;
    let home_positions = tracking_home_positions(tracking);
    let mut transitions = Vec::new();
    let mut events = Vec::new();

    for pair in tracking.frames.windows(2) {
        let before = tracking_frame_to_world_snapshot(&tracking.config, &pair[0], &home_positions);
        let after = tracking_frame_to_world_snapshot(&tracking.config, &pair[1], &home_positions);
        tracking_goal_events(&before, &after, &mut events);

        for player in &before.players {
            if !after.players.iter().any(|p| p.id == player.id) {
                continue;
            }
            let action = infer_tracking_action(player, &before, &after);
            let observation = before.observation_for(player.id);
            let decision = AgentDecisionTrace {
                mdp_state: before.mdp_state(),
                observation: observation.clone(),
                belief: belief_from_observation(&observation),
                operation_order: vec!["tracking-imitation".to_string()],
                action,
            };
            let player_agent = player_agent_from_snapshot(player);
            let reward = soccer_transition_reward(
                &player_agent,
                &decision,
                &before,
                &after,
                before.score_home,
                before.score_away,
                after.score_home,
                after.score_away,
            );
            transitions.push(SoccerLearningTransition {
                tick: before.tick,
                player_id: player.id,
                team: player.team,
                role: player.role,
                state: decision.mdp_state,
                observation: decision.observation,
                belief: decision.belief,
                action: decision.action,
                reward,
                next_state: after.mdp_state(),
                next_observation: after.observation_for(player.id),
                done: pair[1].tick
                    == tracking
                        .frames
                        .last()
                        .map(|f| f.tick)
                        .unwrap_or(pair[1].tick),
            });
        }
    }

    let last = tracking.frames.last().expect("validated tracking frames");
    Ok(SoccerLearningDataset {
        config: tracking.config.clone(),
        summary: MatchSummary {
            score_home: last.score_home.unwrap_or(0),
            score_away: last.score_away.unwrap_or(0),
            ticks: last.tick,
            simulated_seconds: last.clock_seconds,
            stats: MatchStats::default(),
        },
        transitions,
        events,
    })
}

pub fn train_soccer_q_policy_from_tracking(
    tracking: &SoccerTrackingDataset,
    options: SoccerQPolicyOptions,
) -> Result<SoccerQPolicy, String> {
    let dataset = soccer_tracking_dataset_to_learning_dataset(tracking)?;
    Ok(train_soccer_q_policy(&dataset, options))
}

pub fn soccer_policy_artifact_from_learning_dataset(
    dataset: &SoccerLearningDataset,
    options: SoccerQPolicyOptions,
) -> SoccerPolicyArtifact {
    let policy = train_soccer_q_policy(dataset, options.clone());
    SoccerPolicyArtifact {
        config: dataset.config.clone(),
        summary: dataset.summary.clone(),
        transition_count: dataset.transitions.len(),
        options,
        entries: policy.entries(),
        events: dataset.events.clone(),
    }
}

pub fn soccer_simulation_page_html(trace: &SimulationTrace) -> String {
    let json = serde_json::to_string(trace)
        .unwrap_or_else(|_| "{}".to_string())
        .replace("</script", "<\\/script");
    include_str!("soccer_ui.html").replace("__SOCCER_TRACE__", &json)
}

pub fn soccer_live_page_html() -> String {
    include_str!("soccer_live_ui.html").to_string()
}

pub fn write_soccer_artifacts() {
    let trace = run_default_simulation();
    let ui_path = std::path::Path::new("out/soccer-sim.html");
    let _ = std::fs::create_dir_all("out");
    let _ = std::fs::write(ui_path, soccer_simulation_page_html(&trace));
    let learning_config = MatchConfig {
        duration_seconds: 12.0,
        seed: MatchConfig::default().seed + 1,
        ..MatchConfig::default()
    };
    let learning = run_learning_episode(learning_config);
    let policy_artifact =
        soccer_policy_artifact_from_learning_dataset(&learning, SoccerQPolicyOptions::default());
    let policy_path = std::path::Path::new("out/soccer-q-policy.json");
    if let Ok(json) = serde_json::to_string_pretty(&policy_artifact) {
        let _ = std::fs::write(policy_path, json);
    }
    println!("# Soccer simulation UI: {}", ui_path.display());
    println!("# Soccer Q-policy artifact: {}", policy_path.display());
}

fn tracking_home_positions(tracking: &SoccerTrackingDataset) -> HashMap<usize, Vec2> {
    let mut positions = HashMap::new();
    for frame in &tracking.frames {
        for player in &frame.players {
            positions
                .entry(player.id)
                .or_insert(player.home_position.unwrap_or(player.position));
        }
    }
    positions
}

fn tracking_frame_to_world_snapshot(
    config: &MatchConfig,
    frame: &SoccerTrackingFrame,
    home_positions: &HashMap<usize, Vec2>,
) -> WorldSnapshot {
    let holder_team = frame
        .ball_holder
        .and_then(|holder| frame.players.iter().find(|p| p.id == holder))
        .map(|p| p.team);
    let last_touch_team = frame.last_touch_team.or(holder_team);
    let players = frame
        .players
        .iter()
        .map(|p| PlayerSnapshot {
            id: p.id,
            name: p
                .name
                .clone()
                .unwrap_or_else(|| format!("Tracking {}", p.id)),
            team: p.team,
            role: p.role,
            shirt: p.shirt.unwrap_or((p.id % 100).max(1) as u8),
            position: p.position,
            velocity: p.velocity.unwrap_or_default(),
            home_position: home_positions
                .get(&p.id)
                .copied()
                .unwrap_or(p.home_position.unwrap_or(p.position)),
            controller_slot: None,
            acceleration: Vec2::zero(),
        })
        .collect::<Vec<_>>();
    let shared_positions = SharedPlayerPositionSnapshot::from_player_snapshots(
        &players,
        frame.tick,
        frame.clock_seconds,
    );
    let phase = tracking_phase(config, frame, last_touch_team);
    let score_home = frame.score_home.unwrap_or(0);
    let score_away = frame.score_away.unwrap_or(0);
    let score_diff_home = score_home as i32 - score_away as i32;
    WorldSnapshot {
        tick: frame.tick,
        clock_seconds: frame.clock_seconds,
        dt_seconds: config.dt_seconds,
        field_length: config.field_length_yards,
        field_width: config.field_width_yards,
        goal_width: config.goal_width_yards,
        ball: BallState {
            position: frame.ball_position,
            velocity: frame.ball_velocity.unwrap_or_default(),
            acceleration: Vec2::zero(),
            holder: frame.ball_holder,
            last_touch_team,
        },
        ball_history: vec![BallPositionSample {
            tick: frame.tick,
            clock_seconds: frame.clock_seconds,
            position: frame.ball_position,
            velocity: frame.ball_velocity.unwrap_or_default(),
            acceleration: Vec2::zero(),
            holder: frame.ball_holder,
        }],
        players,
        shared_positions,
        score_home,
        score_away,
        phase,
        home_directive: tactical_directive_for_team(
            Team::Home,
            phase,
            last_touch_team,
            frame.ball_position,
            score_diff_home,
            config.field_width_yards,
            config.field_length_yards,
        ),
        away_directive: tactical_directive_for_team(
            Team::Away,
            phase,
            last_touch_team,
            frame.ball_position,
            -score_diff_home,
            config.field_width_yards,
            config.field_length_yards,
        ),
    }
}

fn tracking_phase(
    config: &MatchConfig,
    frame: &SoccerTrackingFrame,
    possession_team: Option<Team>,
) -> TacticalPhase {
    let y = frame.ball_position.y;
    match possession_team {
        Some(Team::Home) if y > config.field_length_yards * 0.68 => TacticalPhase::HomeAttack,
        Some(Team::Home) => TacticalPhase::HomeBuildUp,
        Some(Team::Away) if y < config.field_length_yards * 0.32 => TacticalPhase::AwayAttack,
        Some(Team::Away) => TacticalPhase::AwayBuildUp,
        None if frame.tick < 5 => TacticalPhase::Kickoff,
        None => TacticalPhase::Transition,
    }
}

fn infer_tracking_action(
    player: &PlayerSnapshot,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
) -> String {
    let Some(next_player) = after.players.iter().find(|p| p.id == player.id) else {
        return "hold".to_string();
    };
    let before_obs = before.observation_for(player.id);
    let after_obs = after.observation_for(player.id);
    let moved = player.position.distance(next_player.position);

    if before.ball.holder == Some(player.id) {
        if tracking_team_scored(player.team, before, after) {
            return "shoot".to_string();
        }
        if let Some(holder) = after.ball.holder {
            if holder != player.id
                && after
                    .players
                    .iter()
                    .find(|p| p.id == holder)
                    .is_some_and(|p| p.team == player.team)
            {
                return "pass".to_string();
            }
        }
        if tracking_ball_near_teammate(after, player.id, player.team)
            && after.ball.holder != Some(player.id)
        {
            return "pass".to_string();
        }
        if before_obs.shot_lane_open
            && before_obs.yards_to_goal <= 25.0
            && tracking_ball_moved_toward_goal(before, after, player.team)
        {
            return "shoot".to_string();
        }
        if moved > before.dt_seconds * 1.2 {
            return "dribble".to_string();
        }
        return "hold".to_string();
    }

    if before.possession_team() == Some(player.team.other())
        && after.ball.holder == Some(player.id)
        && player.position.distance(before.ball.position) < 3.8
    {
        return "tackle".to_string();
    }

    let before_ball_distance = player.position.distance(before.ball.position);
    let after_ball_distance = next_player.position.distance(after.ball.position);
    if before.possession_team() == Some(player.team.other())
        && after_ball_distance + 0.35 < before_ball_distance
    {
        return "defend".to_string();
    }

    if before.possession_team() == Some(player.team)
        && (after_obs.open_space_score > before_obs.open_space_score + 0.25
            || moved > before.dt_seconds * 1.2)
    {
        return "space".to_string();
    }

    if moved > before.dt_seconds * 1.2 {
        if before.possession_team() == Some(player.team.other()) {
            "defend".to_string()
        } else {
            "space".to_string()
        }
    } else {
        "hold".to_string()
    }
}

fn tracking_team_scored(team: Team, before: &WorldSnapshot, after: &WorldSnapshot) -> bool {
    match team {
        Team::Home => after.score_home > before.score_home,
        Team::Away => after.score_away > before.score_away,
    }
}

fn tracking_ball_moved_toward_goal(
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    team: Team,
) -> bool {
    (after.ball.position.y - before.ball.position.y) * team.attack_dir() > 3.0
}

fn tracking_ball_near_teammate(after: &WorldSnapshot, player_id: usize, team: Team) -> bool {
    after
        .players
        .iter()
        .filter(|p| p.team == team && p.id != player_id)
        .any(|p| p.position.distance(after.ball.position) < 4.0)
}

fn tracking_goal_events(
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    events: &mut Vec<MatchEvent>,
) {
    if after.score_home > before.score_home {
        events.push(MatchEvent {
            tick: after.tick,
            clock_seconds: after.clock_seconds,
            kind: "goal".to_string(),
            team: Some(Team::Home),
            player_id: before.ball.holder,
            description: "Home goal from tracking data".to_string(),
        });
    }
    if after.score_away > before.score_away {
        events.push(MatchEvent {
            tick: after.tick,
            clock_seconds: after.clock_seconds,
            kind: "goal".to_string(),
            team: Some(Team::Away),
            player_id: before.ball.holder,
            description: "Away goal from tracking data".to_string(),
        });
    }
}

fn player_agent_from_snapshot(player: &PlayerSnapshot) -> PlayerAgent {
    let mut position_history = VecDeque::new();
    position_history.push_back(player.position);
    PlayerAgent {
        id: player.id,
        name: player.name.clone(),
        team: player.team,
        role: player.role,
        shirt: player.shirt,
        home_position: player.home_position,
        position: player.position,
        velocity: player.velocity,
        acceleration: player.acceleration,
        position_history,
        skills: neutral_tracking_skill_profile(player.role),
        fatigue: 0.0,
        controller_slot: None,
        preferences: AgentPreferences::default(),
        last_decision: None,
    }
}

fn neutral_tracking_skill_profile(role: PlayerRole) -> SkillProfile {
    let shooting = match role {
        PlayerRole::Forward => 0.78,
        PlayerRole::Midfielder => 0.66,
        PlayerRole::Defender => 0.50,
        PlayerRole::Goalkeeper => 0.40,
    };
    SkillProfile {
        top_speed_yps: 7.4,
        acceleration_yps2: 7.1,
        shooting,
        passing: 0.72,
        dribbling: 0.68,
        first_touch: 0.70,
        defending: 0.68,
        stamina: 0.82,
        decision_noise: 0.05,
        aggression: 0.58,
    }
}

fn tackle_success_probability(defender: &SkillProfile, attacker: &SkillProfile) -> f64 {
    let ball_control = attacker.dribbling * 0.70 + attacker.first_touch * 0.30;
    let defensive_pressure = defender.defending * 0.82 + defender.aggression * 0.18;
    (defensive_pressure / (defensive_pressure + ball_control)).clamp(0.18, 0.82)
}

fn tackle_foul_probability(
    defender: &SkillProfile,
    attacker: &SkillProfile,
    contact_distance: f64,
    contact_speed: f64,
) -> f64 {
    let timing_risk = (1.0 - defender.defending).clamp(0.0, 1.0) * 0.34;
    let aggression_risk = defender.aggression.clamp(0.0, 1.0) * 0.24;
    let control_risk = (attacker.dribbling * 0.55 + attacker.first_touch * 0.45) * 0.12;
    let speed_risk = (contact_speed / 10.0).clamp(0.0, 1.0) * 0.18;
    let reach_risk = (contact_distance / 2.2).clamp(0.0, 1.0) * 0.10;
    (0.02 + timing_risk + aggression_risk + control_risk + speed_risk + reach_risk)
        .clamp(0.03, 0.78)
}

fn pressure_from_observation(observation: &SoccerPomdpObservation) -> f64 {
    (1.0 - observation.nearest_opponent_distance / 18.0).clamp(0.0, 1.0)
}

fn noisy_pass_target(
    from: Vec2,
    target: Vec2,
    passing_skill: f64,
    pressure: f64,
    distance: f64,
    rng: &mut SeededRandom,
) -> Vec2 {
    let dir = (target - from).normalized();
    let lateral = Vec2::new(-dir.y, dir.x);
    let skill_error = 1.0 - passing_skill.clamp(0.05, 0.99);
    let error_scale = (0.35 + distance * 0.020) * (0.35 + skill_error * 1.45 + pressure * 0.75);
    let lateral_error = triangular_sample(rng) * error_scale;
    let weight_error = triangular_sample(rng) * error_scale * 0.42;
    target + lateral * lateral_error + dir * weight_error
}

fn noisy_shot_target_x(
    goal_center_x: f64,
    goal_width: f64,
    shooting_skill: f64,
    pressure: f64,
    yards_to_goal: f64,
    rng: &mut SeededRandom,
) -> f64 {
    let skill_error = 1.0 - shooting_skill.clamp(0.05, 0.99);
    let miss_window = goal_width * (0.18 + skill_error * 0.95 + pressure * 0.38)
        + yards_to_goal.max(0.0) * 0.018 * (0.35 + skill_error);
    goal_center_x + triangular_sample(rng) * miss_window
}

fn goalkeeper_save_probability(
    keeper: &PlayerAgent,
    shot_crossing: Vec2,
    shot_speed: f64,
    goal_width: f64,
) -> f64 {
    let reaction = keeper.skills.defending * 0.50
        + keeper.skills.first_touch * 0.32
        + keeper.skills.acceleration_yps2.min(9.5) / 9.5 * 0.18;
    let distance_to_shot = keeper.position.distance(shot_crossing);
    let reach_penalty = (distance_to_shot / (goal_width * 0.72)).clamp(0.0, 1.5);
    let speed_penalty = (shot_speed / 48.0).clamp(0.0, 1.0) * 0.28;
    (0.18 + reaction * 0.72 - reach_penalty * 0.34 - speed_penalty).clamp(0.04, 0.86)
}

fn goalkeeper_for_players(players: &[PlayerAgent], team: Team) -> Option<usize> {
    players
        .iter()
        .find(|player| player.team == team && player.role == PlayerRole::Goalkeeper)
        .map(|player| player.id)
}

fn nearest_ball_controller_for(
    ball_pos: Vec2,
    ball_velocity: Vec2,
    players: &[PlayerAgent],
    rng: &mut SeededRandom,
) -> Option<(usize, Team)> {
    let ball_speed = ball_velocity.len();
    let mut candidates = Vec::new();
    for p in players {
        let control_radius = PLAYER_CONTROL_RADIUS_YARDS
            + p.skills.first_touch * 0.48
            + (1.0 - (ball_speed / 18.0).clamp(0.0, 1.0)) * 0.24;
        let dist = p.position.distance(ball_pos);
        if dist > control_radius {
            continue;
        }
        let to_ball = (ball_pos - p.position).normalized();
        let closing_speed = dot(p.velocity - ball_velocity, to_ball).clamp(-8.0, 8.0);
        let score = -dist * 1.45
            + p.skills.first_touch * 0.72
            + p.skills.aggression * 0.18
            + closing_speed * 0.055;
        candidates.push((p.id, p.team, score));
    }
    sample_control_candidate(&candidates, rng)
}

fn triangular_sample(rng: &mut SeededRandom) -> f64 {
    rng.next_float() + rng.next_float() - 1.0
}

pub fn segment_distance_to_point(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    let ab = b - a;
    let denom = ab.x * ab.x + ab.y * ab.y;
    if denom <= 1e-12 {
        return p.distance(a);
    }
    let ap = p - a;
    let t = ((ap.x * ab.x + ap.y * ab.y) / denom).clamp(0.0, 1.0);
    let projection = a + ab * t;
    p.distance(projection)
}

fn dot(a: Vec2, b: Vec2) -> f64 {
    a.x * b.x + a.y * b.y
}

fn deterministic_separation_normal(a_id: usize, b_id: usize) -> Vec2 {
    let seed = ((a_id as u64 + 1) * 1_103_515_245) ^ ((b_id as u64 + 7) * 12_345);
    let angle = (seed % 6283) as f64 / 1000.0;
    Vec2::new(angle.cos(), angle.sin()).normalized()
}

fn sample_control_candidate(
    candidates: &[(usize, Team, f64)],
    rng: &mut SeededRandom,
) -> Option<(usize, Team)> {
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        let (id, team, _) = candidates[0];
        return Some((id, team));
    }

    let max_score = candidates
        .iter()
        .map(|(_, _, score)| *score)
        .fold(f64::NEG_INFINITY, f64::max);
    let temperature = 0.48;
    let weights = candidates
        .iter()
        .map(|(_, _, score)| ((*score - max_score) / temperature).exp())
        .collect::<Vec<_>>();
    let total = weights.iter().sum::<f64>();
    if total <= 1e-12 || !total.is_finite() {
        let (id, team, _) = candidates[0];
        return Some((id, team));
    }

    let mut draw = rng.next_float() * total;
    for ((id, team, _), weight) in candidates.iter().zip(weights.iter()) {
        draw -= *weight;
        if draw <= 0.0 {
            return Some((*id, *team));
        }
    }
    candidates.last().map(|(id, team, _)| (*id, *team))
}

fn restart_kind_action(kind: BallRestartKind) -> &'static str {
    match kind {
        BallRestartKind::ThrowIn => "throw-in",
        BallRestartKind::GoalKick => "goal-kick",
        BallRestartKind::CornerKick => "corner-kick",
        BallRestartKind::FreeKick => "free-kick",
    }
}

fn restart_kind_label(kind: BallRestartKind) -> &'static str {
    match kind {
        BallRestartKind::ThrowIn => "throw-in",
        BallRestartKind::GoalKick => "goal kick",
        BallRestartKind::CornerKick => "corner",
        BallRestartKind::FreeKick => "free kick",
    }
}

fn learned_action_label_is_legal(action: &str, snapshot: &WorldSnapshot, player_id: usize) -> bool {
    let action = normalize_soccer_action_label(action);
    let Some(player) = snapshot.players.iter().find(|p| p.id == player_id) else {
        return false;
    };
    let observation = snapshot.observation_for(player_id);
    match action {
        "shoot" => {
            observation.has_ball && observation.shot_lane_open && observation.yards_to_goal <= 20.0
        }
        "pass" => observation.has_ball && snapshot.best_pass_target(player_id).is_some(),
        "dribble" => observation.has_ball,
        "defend" => snapshot.possession_team() == Some(player.team.other()),
        "tackle" => snapshot.ball.holder.is_some_and(|holder| {
            snapshot
                .players
                .iter()
                .find(|p| p.id == holder)
                .is_some_and(|holder_player| {
                    holder_player.team == player.team.other()
                        && player.position.distance(holder_player.position) < 3.2
                })
        }),
        "space" => !observation.has_ball,
        "hold" => true,
        "human-move" => false,
        _ => false,
    }
}

pub fn shot_lane_is_clear(snapshot: &WorldSnapshot, player_id: usize) -> bool {
    let Some(player) = snapshot.players.iter().find(|p| p.id == player_id) else {
        return false;
    };
    let goal = Vec2::new(
        snapshot.field_width * 0.5,
        player.team.goal_y(snapshot.field_length),
    );
    snapshot.clear_line(player.position, goal, player.team.other(), 3.0)
}

fn approach_velocity(current: Vec2, desired: Vec2, accel: f64, dt: f64) -> Vec2 {
    let delta = desired - current;
    let max_delta = accel * dt;
    if delta.len() <= max_delta {
        desired
    } else {
        current + delta.normalized() * max_delta
    }
}

fn zone(v: f64, max: f64, buckets: usize) -> usize {
    ((v / max).clamp(0.0, 0.999_999) * buckets as f64).floor() as usize
}

fn distance_bucket(value: f64, edges: &[f64]) -> u8 {
    edges
        .iter()
        .position(|edge| value <= *edge)
        .unwrap_or(edges.len()) as u8
}

fn pressure_bucket(nearest_opponent_distance: f64) -> u8 {
    let pressure = (1.0 - nearest_opponent_distance / 18.0).clamp(0.0, 1.0);
    distance_bucket(pressure, &[0.15, 0.35, 0.60, 0.82])
}

fn default_players(config: &MatchConfig, rng: &mut SeededRandom) -> Vec<PlayerAgent> {
    let home_layout = vec![
        (
            "Home GK".to_string(),
            PlayerRole::Goalkeeper,
            1,
            Vec2::new(40.0, 7.0),
        ),
        (
            "Home LB".to_string(),
            PlayerRole::Defender,
            2,
            Vec2::new(14.0, 24.0),
        ),
        (
            "Home LCB".to_string(),
            PlayerRole::Defender,
            4,
            Vec2::new(31.0, 23.0),
        ),
        (
            "Home RCB".to_string(),
            PlayerRole::Defender,
            5,
            Vec2::new(49.0, 23.0),
        ),
        (
            "Home RB".to_string(),
            PlayerRole::Defender,
            3,
            Vec2::new(66.0, 24.0),
        ),
        (
            "Home LM".to_string(),
            PlayerRole::Midfielder,
            11,
            Vec2::new(17.0, 52.0),
        ),
        (
            "Home CM1".to_string(),
            PlayerRole::Midfielder,
            6,
            Vec2::new(33.0, 50.0),
        ),
        (
            "Home CM2".to_string(),
            PlayerRole::Midfielder,
            8,
            Vec2::new(47.0, 50.0),
        ),
        (
            "Home RM".to_string(),
            PlayerRole::Midfielder,
            7,
            Vec2::new(63.0, 52.0),
        ),
        (
            "Home ST1".to_string(),
            PlayerRole::Forward,
            9,
            Vec2::new(31.0, 82.0),
        ),
        (
            "Home ST2".to_string(),
            PlayerRole::Forward,
            10,
            Vec2::new(49.0, 82.0),
        ),
    ];
    let away_layout = home_layout
        .iter()
        .map(|(name, role, shirt, pos)| {
            (
                name.replacen("Home", "Away", 1),
                *role,
                *shirt,
                Vec2::new(pos.x, config.field_length_yards - pos.y),
            )
        })
        .collect::<Vec<_>>();

    let mut players = Vec::with_capacity(22);
    for (team, layout) in [(Team::Home, home_layout), (Team::Away, away_layout)] {
        for (local_idx, (name, role, shirt, pos)) in layout.into_iter().enumerate() {
            let id = players.len();
            let controller_slot = if team == Team::Home && local_idx < config.human_slots() {
                Some(local_idx)
            } else {
                None
            };
            let mut preferences = AgentPreferences::default();
            if role == PlayerRole::Forward {
                preferences.shoot_bias = 0.78;
                preferences.dribble_bias = 0.58;
            } else if role == PlayerRole::Midfielder {
                preferences.pass_bias = 0.76;
                preferences.open_space_bias = 0.82;
            }
            players.push(PlayerAgent {
                id,
                name,
                team,
                role,
                shirt,
                home_position: pos,
                position: pos,
                velocity: Vec2::zero(),
                acceleration: Vec2::zero(),
                position_history: VecDeque::from([pos]),
                skills: SkillProfile::blended(id, role, rng),
                fatigue: 0.0,
                controller_slot,
                preferences,
                last_decision: None,
            });
        }
    }
    players
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_match_has_22_players_and_3_officials() {
        let sim = SoccerMatch::default_11v11(MatchConfig::default());
        assert_eq!(sim.players.len(), 22);
        assert_eq!(sim.officials.len(), 3);
        assert_eq!(sim.config.dt_seconds, 0.1);
    }

    #[test]
    fn ball_agent_records_own_run_time_step() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.2,
            seed: 201,
            ..Default::default()
        });

        sim.run_time_step();

        assert_eq!(sim.ball.id, BALL_AGENT_ID);
        let decision = sim.ball.last_decision.as_ref().expect("ball decision");
        assert_eq!(decision.tick, 0);
        assert!(!decision.action.is_empty());
    }

    #[test]
    fn ball_agent_tracks_acceleration_and_rolling_history() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            seed: 204,
            ..Default::default()
        });
        for player in &mut sim.players {
            player.position = Vec2::new(2.0, 2.0);
            player.velocity = Vec2::zero();
        }
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(20.0, 60.0);
        sim.ball.velocity = Vec2::new(10.0, 0.0);
        sim.ball.acceleration = Vec2::zero();
        sim.ball.last_touch_team = Some(Team::Home);
        sim.pending_pass = None;
        sim.pending_shot = None;
        sim.ball.position_history.clear();
        sim.ball.record_position_history(0, 0.0);

        for _ in 0..60 {
            sim.integrate_ball();
            sim.tick += 1;
            sim.clock_seconds += sim.config.dt_seconds;
            sim.record_ball_position_history();
        }

        assert_eq!(sim.ball.position_history.len(), BALL_POSITION_HISTORY_LIMIT);
        assert!(sim.ball.acceleration.x < 0.0);
        assert!(sim
            .ball
            .history_velocity_estimate(sim.config.dt_seconds)
            .len()
            .is_finite());
        assert!(sim
            .ball
            .history_acceleration_estimate(sim.config.dt_seconds)
            .len()
            .is_finite());

        let snapshot = WorldSnapshot::from_match(&sim);
        assert_eq!(
            snapshot.ball_position_history().len(),
            BALL_POSITION_HISTORY_LIMIT
        );
        assert_eq!(snapshot.ball.acceleration, sim.ball.acceleration);
        assert_eq!(sim.to_frame().ball.acceleration, sim.ball.acceleration);
    }

    #[test]
    fn central_brain_updates_before_player_decisions() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 203,
            ..Default::default()
        });
        sim.players[9].position = Vec2::new(40.0, 95.0);
        sim.ball.holder = Some(9);
        sim.ball.position = sim.players[9].position;
        sim.ball.last_touch_team = Some(Team::Home);

        sim.run_time_step();

        assert_eq!(sim.central_brain.phase, TacticalPhase::HomeAttack);
        let decision = sim.players[9]
            .last_decision
            .as_ref()
            .expect("striker decision");
        assert_eq!(decision.mdp_state.phase, TacticalPhase::HomeAttack);
        assert_eq!(decision.mdp_state.possession_team, Some(Team::Home));
        assert!(sim.central_brain.home_directive.shot_threshold_yards > 20.0);
    }

    #[test]
    fn central_brain_directives_shape_team_behavior() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.players[9].position = Vec2::new(40.0, 94.0);
        sim.ball.holder = Some(9);
        sim.ball.position = sim.players[9].position;
        sim.ball.last_touch_team = Some(Team::Home);
        let before = WorldSnapshot::from_match(&sim);
        sim.central_brain.run_time_step(&before);
        let snapshot = WorldSnapshot::from_match(&sim);

        let home = snapshot.tactical_directive(Team::Home);
        let away = snapshot.tactical_directive(Team::Away);
        assert_eq!(snapshot.phase, TacticalPhase::HomeAttack);
        assert!(home.risk_tolerance > away.risk_tolerance);
        assert!(away.press_intensity > home.press_intensity);
        assert!(home.shot_threshold_yards > 20.0);

        let away_def_home = sim.players[12].home_position;
        let defensive_shape = snapshot.defensive_shape_for(12, away_def_home);
        let mut neutral_snapshot = snapshot.clone();
        neutral_snapshot.away_directive =
            TeamTacticalDirective::neutral(Team::Away, snapshot.field_width, snapshot.field_length);
        let neutral_shape = neutral_snapshot.defensive_shape_for(12, away_def_home);
        assert!(defensive_shape.y > neutral_shape.y);
    }

    #[test]
    fn shared_position_board_tracks_rolling_player_history() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 6.0,
            seed: 202,
            ..Default::default()
        });

        for _ in 0..60 {
            sim.run_time_step();
        }

        assert_eq!(
            sim.players[0].position_history.len(),
            PLAYER_POSITION_HISTORY_LIMIT
        );
        assert!(sim.players[0]
            .history_velocity_estimate(sim.config.dt_seconds)
            .len()
            .is_finite());
        assert!(sim.players[0]
            .history_acceleration_estimate(sim.config.dt_seconds)
            .len()
            .is_finite());

        let shared = sim.shared_positions.clone();
        let history_len = std::thread::spawn(move || {
            let snapshot = shared.snapshot();
            assert_eq!(snapshot.latest.len(), 22);
            snapshot.history_for(0).expect("player history").len()
        })
        .join()
        .expect("reader thread joins");
        assert_eq!(history_len, PLAYER_POSITION_HISTORY_LIMIT);

        let snapshot = WorldSnapshot::from_match(&sim);
        assert_eq!(
            snapshot
                .player_position_history(0)
                .expect("snapshot history")
                .len(),
            PLAYER_POSITION_HISTORY_LIMIT
        );
        assert_eq!(
            snapshot.player_position(0).expect("snapshot position"),
            sim.players[0].position
        );
    }

    #[test]
    fn short_simulation_advances_ticks_and_records_frames() {
        let trace = run_simulation(
            MatchConfig {
                duration_seconds: 3.0,
                seed: 99,
                ..Default::default()
            },
            2,
        );
        assert_eq!(trace.summary.ticks, 30);
        assert!(trace.frames.len() >= 15);
        assert_eq!(trace.frames[0].players.len(), 22);
    }

    #[test]
    fn human_input_queue_keeps_latest_by_slot() {
        let q = SharedHumanInputs::new();
        assert!(q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(1),
            seq: 1,
            axis: Vec2::new(1.0, 0.0),
            sprint: false,
            pass: false,
            shoot: false,
            target_player: None,
        }));
        assert!(q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(1),
            seq: 2,
            axis: Vec2::new(0.0, 1.0),
            sprint: true,
            pass: false,
            shoot: false,
            target_player: None,
        }));
        let latest = q.drain_latest_by_slot();
        assert_eq!(latest.get(&0).unwrap().seq, 2);
        assert!(latest.get(&0).unwrap().sprint);
    }

    #[test]
    fn human_input_queue_rejects_stale_frames_and_allows_readers() {
        let q = SharedHumanInputs::new();
        assert!(q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(0),
            seq: 5,
            axis: Vec2::new(1.0, 0.0),
            sprint: false,
            pass: false,
            shoot: false,
            target_player: None,
        }));
        assert!(!q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(0),
            seq: 4,
            axis: Vec2::new(-1.0, 0.0),
            sprint: true,
            pass: false,
            shoot: false,
            target_player: None,
        }));
        assert_eq!(q.queued_len(), 1);
        assert_eq!(q.last_seq_for_slot(0), Some(5));

        let reader_queue = q.clone();
        let reader = std::thread::spawn(move || {
            assert_eq!(reader_queue.queued_len(), 1);
            assert_eq!(reader_queue.last_seq_for_slot(0), Some(5));
        });
        reader.join().expect("reader joins");

        assert!(q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(0),
            seq: 6,
            axis: Vec2::new(0.0, 1.0),
            sprint: true,
            pass: false,
            shoot: false,
            target_player: None,
        }));
        let latest = q.drain_latest_by_slot();
        assert_eq!(latest.get(&0).unwrap().seq, 6);
        assert!(latest.get(&0).unwrap().sprint);
        assert_eq!(q.queued_len(), 0);
        assert!(!q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(0),
            seq: 5,
            axis: Vec2::zero(),
            sprint: false,
            pass: false,
            shoot: false,
            target_player: None,
        }));
    }

    #[test]
    fn human_input_queue_handles_multiple_controller_threads() {
        let q = SharedHumanInputs::new();
        let mut handles = Vec::new();
        for slot in 0..4 {
            let q = q.clone();
            handles.push(std::thread::spawn(move || {
                for seq in [1, 3, 2] {
                    let _ = q.push(HumanInputFrame {
                        controller_slot: slot,
                        player_id: Some(slot),
                        seq,
                        axis: Vec2::new(slot as f64, seq as f64),
                        sprint: seq == 3,
                        pass: false,
                        shoot: false,
                        target_player: None,
                    });
                }
            }));
        }
        for handle in handles {
            handle.join().expect("controller writer joins");
        }

        let latest = q.drain_latest_by_slot();
        assert_eq!(latest.len(), 4);
        for slot in 0..4 {
            let input = latest.get(&slot).expect("slot input");
            assert_eq!(input.seq, 3);
            assert!(input.sprint);
            assert_eq!(q.last_seq_for_slot(slot), Some(3));
        }
    }

    #[test]
    fn realtime_session_consumes_input_from_another_thread() {
        let mut session = SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 1,
            seed: 77,
            ..Default::default()
        });
        let start_x = session.match_ref().players[0].position.x;
        let queue = session.input_queue();

        std::thread::spawn(move || {
            queue.push(HumanInputFrame {
                controller_slot: 0,
                player_id: Some(0),
                seq: 1,
                axis: Vec2::new(1.0, 0.0),
                sprint: true,
                pass: false,
                shoot: false,
                target_player: None,
            });
        })
        .join()
        .expect("controller input thread joins");

        let resp = session.step_once();
        assert_eq!(resp.summary.ticks, 1);
        assert_eq!(resp.accepted_inputs, 0);
        assert!(session.match_ref().players[0].position.x > start_x);
    }

    #[test]
    fn human_input_target_player_drives_targeted_pass() {
        let mut session = SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 1,
            seed: 78,
            ..Default::default()
        });
        {
            let sim = session.match_mut();
            sim.players[5].controller_slot = Some(0);
            sim.ball.holder = Some(5);
            sim.ball.position = sim.players[5].position;
            sim.ball.velocity = Vec2::zero();
            sim.ball.last_touch_team = Some(Team::Home);
        }

        let response = session.step(SoccerStepRequest {
            inputs: vec![HumanInputFrame {
                controller_slot: 0,
                player_id: Some(5),
                seq: 1,
                axis: Vec2::zero(),
                sprint: false,
                pass: true,
                shoot: false,
                target_player: Some(8),
            }],
            ticks: 1,
            record_every_ticks: Some(1),
        });

        assert_eq!(response.accepted_inputs, 1);
        assert_eq!(session.match_ref().stats.passes_attempted_home, 1);
        let pass = session
            .match_ref()
            .pending_pass
            .as_ref()
            .expect("pending targeted pass");
        assert_eq!(pass.from, 5);
        assert_eq!(pass.target, Some(8));
        assert_eq!(
            session.match_ref().players[5]
                .last_decision
                .as_ref()
                .expect("human pass decision")
                .action,
            "pass"
        );
    }

    #[test]
    fn controller_slot_reassignment_moves_human_control_between_players() {
        let mut session = SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 2,
            seed: 79,
            ..Default::default()
        });

        let response = session
            .assign_controller_slot(SoccerControllerAssignmentRequest {
                controller_slot: 0,
                player_id: Some(5),
            })
            .expect("controller reassignment");
        assert!(response
            .controller_assignments
            .iter()
            .any(|a| a.controller_slot == 0 && a.player_id == 5));
        assert_eq!(session.match_ref().players[0].controller_slot, None);
        assert_eq!(session.match_ref().players[5].controller_slot, Some(0));

        let start_x = session.match_ref().players[5].position.x;
        let response = session.step(SoccerStepRequest {
            inputs: vec![HumanInputFrame {
                controller_slot: 0,
                player_id: Some(5),
                seq: 1,
                axis: Vec2::new(1.0, 0.0),
                sprint: true,
                pass: false,
                shoot: false,
                target_player: None,
            }],
            ticks: 1,
            record_every_ticks: Some(1),
        });

        assert_eq!(response.accepted_inputs, 1);
        assert!(session.match_ref().players[5].position.x > start_x);
    }

    #[test]
    fn realtime_session_step_json_round_trips() {
        let mut session = SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 2,
            seed: 12,
            ..Default::default()
        });
        let json = session
            .step_json(r#"{"ticks":3,"recordEveryTicks":1}"#)
            .expect("step json");
        let value: serde_json::Value = serde_json::from_str(&json).expect("response json");
        assert_eq!(value["summary"]["ticks"], 3);
        assert_eq!(value["frames"].as_array().unwrap().len(), 3);
        assert_eq!(value["controllerAssignments"].as_array().unwrap().len(), 2);
        assert_eq!(value["learningTransitions"].as_array().unwrap().len(), 66);
    }

    #[test]
    fn learning_episode_records_transition_per_player_per_tick() {
        let dataset = run_learning_episode(MatchConfig {
            duration_seconds: 0.2,
            seed: 13,
            ..Default::default()
        });
        assert_eq!(dataset.summary.ticks, 2);
        assert_eq!(dataset.transitions.len(), 44);
        let first = &dataset.transitions[0];
        assert_eq!(first.tick, 0);
        assert_eq!(first.next_state.tick, 1);
        assert!(!first.action.is_empty());
        assert!(first.reward.is_finite());
        assert_eq!(first.observation.player_id, first.player_id);
        assert_eq!(first.next_observation.player_id, first.player_id);
    }

    #[test]
    fn q_policy_trains_from_learning_episode() {
        let dataset = run_learning_episode(MatchConfig {
            duration_seconds: 0.2,
            seed: 14,
            ..Default::default()
        });
        let policy = train_soccer_q_policy(&dataset, SoccerQPolicyOptions::default());
        assert!(!policy.q_values.is_empty());

        let first = &dataset.transitions[0];
        let state = SoccerQStateKey::from_transition(first);
        assert!(policy.best_action(&state).is_some());
        assert!(policy
            .q_value(&state, &first.action)
            .expect("transition q-value")
            .is_finite());
        assert_eq!(policy.entries().len(), policy.q_values.len());
    }

    #[test]
    fn learned_policy_biases_agent_decision_when_legal() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 15,
            ..Default::default()
        });
        let snapshot = WorldSnapshot::from_match(&sim);
        assert_eq!(snapshot.ball.holder, Some(5));
        assert!(snapshot.best_pass_target(5).is_some());

        let mut policy = SoccerQPolicy::default();
        assert!(policy.set_action_value_for_snapshot(&snapshot, 5, "pass", 5.0));
        sim.set_learned_policy(policy);
        sim.run_time_step();

        assert_eq!(
            sim.players[5]
                .last_decision
                .as_ref()
                .expect("player decision")
                .action,
            "pass"
        );
        assert!(sim.stats.passes_attempted_home > 0);
        assert!(!sim
            .learned_policy()
            .expect("online policy")
            .q_values
            .is_empty());
    }

    #[test]
    fn semantic_move_labels_are_recorded_for_learning() {
        let dataset = run_learning_episode(MatchConfig {
            duration_seconds: 0.1,
            seed: 16,
            ..Default::default()
        });
        assert!(dataset
            .transitions
            .iter()
            .any(|t| t.action == "space" || t.action == "defend"));
        assert!(!dataset.transitions.iter().any(|t| t.action == "move"));
    }

    #[test]
    fn tracking_dataset_converts_pass_to_learning_transition_and_policy() {
        let tracking = sample_tracking_pass_dataset();
        let dataset = tracking.to_learning_dataset().expect("tracking conversion");
        assert_eq!(dataset.transitions.len(), 3);

        let passer = dataset
            .transitions
            .iter()
            .find(|t| t.player_id == 0)
            .expect("passer transition");
        assert_eq!(passer.action, "pass");
        assert!(passer.reward.is_finite());

        let policy =
            train_soccer_q_policy_from_tracking(&tracking, SoccerQPolicyOptions::default())
                .expect("tracking policy");
        let state = SoccerQStateKey::from_transition(passer);
        assert!(policy.q_value(&state, "pass").is_some());

        let artifact =
            soccer_policy_artifact_from_learning_dataset(&dataset, SoccerQPolicyOptions::default());
        assert_eq!(artifact.transition_count, 3);
        assert!(!artifact.entries.is_empty());
    }

    #[test]
    fn tracking_dataset_json_round_trips_and_validates() {
        let tracking = sample_tracking_pass_dataset();
        let json = serde_json::to_string(&tracking).expect("tracking json");
        let parsed = soccer_tracking_dataset_from_json(&json).expect("parse tracking");
        assert_eq!(parsed.frames.len(), 2);
        assert!(parsed.validate().is_ok());

        let invalid = SoccerTrackingDataset {
            frames: parsed.frames[..1].to_vec(),
            ..parsed
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn tracking_dataset_infers_tackle() {
        let tracking = sample_tracking_tackle_dataset();
        let dataset = tracking.to_learning_dataset().expect("tracking conversion");
        let defender = dataset
            .transitions
            .iter()
            .find(|t| t.player_id == 0)
            .expect("defender transition");
        assert_eq!(defender.action, "tackle");
    }

    #[test]
    fn player_collision_resolution_separates_body_overlap() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.players[0].position = Vec2::new(40.0, 60.0);
        sim.players[1].position = Vec2::new(40.0, 60.0);
        sim.players[0].velocity = Vec2::new(1.0, 0.0);
        sim.players[1].velocity = Vec2::new(-1.0, 0.0);

        sim.resolve_player_collisions();

        let dist = sim.players[0].position.distance(sim.players[1].position);
        assert!(dist >= PLAYER_BODY_RADIUS_YARDS * 2.0 - 1e-6);
    }

    #[test]
    fn loose_ball_contest_is_probabilistic_but_skill_weighted() {
        let mut stronger_wins = 0;
        let mut weaker_wins = 0;
        for seed in 0..180 {
            let mut sim = SoccerMatch::default_11v11(MatchConfig {
                seed,
                ..Default::default()
            });
            sim.ball.holder = None;
            sim.ball.position = Vec2::new(40.0, 60.0);
            sim.ball.velocity = Vec2::zero();
            sim.players[0].position = Vec2::new(39.1, 60.0);
            sim.players[0].velocity = Vec2::zero();
            sim.players[0].skills.first_touch = 0.10;
            sim.players[0].skills.dribbling = 0.10;
            sim.players[0].skills.aggression = 0.10;
            sim.players[11].position = Vec2::new(41.15, 60.0);
            sim.players[11].velocity = Vec2::new(-4.0, 0.0);
            sim.players[11].skills.first_touch = 0.98;
            sim.players[11].skills.dribbling = 0.98;
            sim.players[11].skills.aggression = 0.95;

            match sim.nearest_ball_controller().expect("loose-ball winner").0 {
                11 => stronger_wins += 1,
                0 => weaker_wins += 1,
                other => panic!("unexpected loose-ball winner {other}"),
            }
        }

        assert!(stronger_wins > weaker_wins);
        assert!(weaker_wins > 0);
    }

    #[test]
    fn loose_ball_recovery_updates_holder_and_stats() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.ball.holder = None;
        sim.ball.position = sim.players[5].position;
        sim.ball.velocity = Vec2::zero();
        sim.pending_pass = None;

        sim.integrate_ball();

        assert_eq!(sim.ball.holder, Some(5));
        assert_eq!(sim.stats.loose_ball_recoveries_home, 1);
        assert_eq!(sim.stats.loose_ball_recoveries_away, 0);
    }

    #[test]
    fn offside_geometry_uses_ball_second_last_defender_and_halfway_line() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.players[5].position = Vec2::new(40.0, 70.0);
        sim.players[9].position = Vec2::new(40.0, 108.0);
        for away in 11..22 {
            sim.players[away].position = Vec2::new(8.0 + away as f64, 82.0);
        }
        sim.players[11].position = Vec2::new(40.0, 118.0);
        sim.players[12].position = Vec2::new(42.0, 96.0);
        sim.ball.position = sim.players[5].position;
        sim.ball.holder = Some(5);

        let snapshot = WorldSnapshot::from_match(&sim);
        let offside = snapshot
            .pending_offside_for_pass(5, 9)
            .expect("runner should be offside");
        assert_eq!(offside.target, 9);
        assert_eq!(offside.second_last_defender_y, 96.0);

        sim.players[9].position = Vec2::new(40.0, 48.0);
        let snapshot = WorldSnapshot::from_match(&sim);
        assert!(snapshot.pending_offside_for_pass(5, 9).is_none());
    }

    #[test]
    fn completed_pass_to_offside_runner_awards_defensive_restart() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.players[5].position = Vec2::new(40.0, 70.0);
        sim.players[9].position = Vec2::new(40.0, 108.0);
        for away in 11..22 {
            sim.players[away].position = Vec2::new(8.0 + away as f64, 82.0);
        }
        sim.players[11].position = Vec2::new(40.0, 118.0);
        sim.players[12].position = Vec2::new(42.0, 96.0);
        sim.ball.position = sim.players[5].position;
        sim.ball.holder = Some(5);

        sim.apply_player_intent(PlayerIntent {
            player_id: 5,
            action: SoccerAction::Pass {
                target_player: Some(9),
                power: 1.0,
            },
            sprint: false,
        });
        assert!(sim
            .pending_pass
            .as_ref()
            .and_then(|pass| pass.offside.as_ref())
            .is_some());

        sim.ball.holder = None;
        sim.ball.position = sim.players[9].position;
        sim.ball.velocity = Vec2::zero();
        sim.integrate_ball();

        assert_eq!(sim.stats.offsides_home, 1);
        assert_eq!(sim.stats.passes_completed_home, 0);
        assert_eq!(sim.ball.last_touch_team, Some(Team::Away));
        let holder = sim.ball.holder.expect("defensive restart holder");
        assert_eq!(sim.players[holder].team, Team::Away);
        assert!(sim.events.iter().any(|event| event.kind == "offside"));
        assert_eq!(
            sim.ball
                .last_decision
                .as_ref()
                .expect("ball decision")
                .action,
            "offside"
        );
    }

    #[test]
    fn ai_best_pass_target_avoids_offside_runner() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.players[5].position = Vec2::new(40.0, 70.0);
        sim.players[8].position = Vec2::new(30.0, 88.0);
        sim.players[9].position = Vec2::new(40.0, 108.0);
        for id in [6, 7, 10] {
            sim.players[id].position = Vec2::new(14.0 + id as f64, 58.0);
        }
        for away in 11..22 {
            sim.players[away].position = Vec2::new(74.0, 82.0);
        }
        sim.players[11].position = Vec2::new(40.0, 118.0);
        sim.players[12].position = Vec2::new(42.0, 96.0);
        sim.ball.position = sim.players[5].position;
        sim.ball.holder = Some(5);

        let snapshot = WorldSnapshot::from_match(&sim);
        assert!(snapshot.pending_offside_for_pass(5, 9).is_some());
        assert_ne!(snapshot.best_pass_target(5), Some(9));
    }

    #[test]
    fn configurable_ball_drag_controls_roll_down_speed() {
        let mut short_grass = SoccerMatch::default_11v11(MatchConfig {
            ball_drag_per_tick: 0.005,
            seed: 301,
            ..Default::default()
        });
        let mut long_grass = SoccerMatch::default_11v11(MatchConfig {
            ball_drag_per_tick: 0.20,
            seed: 301,
            ..Default::default()
        });
        for sim in [&mut short_grass, &mut long_grass] {
            sim.ball.holder = None;
            sim.ball.position = Vec2::new(40.0, 60.0);
            sim.ball.velocity = Vec2::new(10.0, 0.0);
            sim.ball.last_touch_team = Some(Team::Home);
            sim.pending_pass = None;
            sim.pending_shot = None;
        }

        short_grass.integrate_ball();
        long_grass.integrate_ball();

        assert!(long_grass.ball.velocity.len() < short_grass.ball.velocity.len());
        assert!((short_grass.ball.velocity.len() - 9.95).abs() < 1e-9);
        assert!((long_grass.ball.velocity.len() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn touchline_out_awards_throw_in_instead_of_bouncing() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(-1.0, 64.0);
        sim.ball.velocity = Vec2::new(-8.0, 0.0);
        sim.ball.last_touch_team = Some(Team::Home);

        sim.integrate_ball();

        assert_eq!(sim.stats.throw_ins_away, 1);
        assert_eq!(sim.ball.position.x, 0.0);
        assert_eq!(sim.ball.velocity, Vec2::zero());
        let holder = sim.ball.holder.expect("throw-in taker");
        assert_eq!(sim.players[holder].team, Team::Away);
        assert_eq!(
            sim.ball
                .last_decision
                .as_ref()
                .expect("ball decision")
                .action,
            "throw-in"
        );
        assert!(sim.events.iter().any(|event| event.kind == "throw-in"));
    }

    #[test]
    fn endline_out_awards_goal_kick_or_corner() {
        let mut goal_kick = SoccerMatch::default_11v11(MatchConfig::default());
        goal_kick.ball.holder = None;
        goal_kick.ball.position = Vec2::new(58.0, 121.0);
        goal_kick.ball.velocity = Vec2::new(0.0, 16.0);
        goal_kick.ball.last_touch_team = Some(Team::Home);
        goal_kick.pending_shot = Some(PendingShot {
            team: Team::Home,
            shooter: 9,
        });

        goal_kick.integrate_ball();

        assert_eq!(goal_kick.stats.goal_kicks_away, 1);
        assert_eq!(goal_kick.ball.last_touch_team, Some(Team::Away));
        assert!(goal_kick.events.iter().any(|event| event.kind == "miss"));
        assert!(goal_kick
            .events
            .iter()
            .any(|event| event.kind == "goal-kick"));

        let mut corner = SoccerMatch::default_11v11(MatchConfig::default());
        corner.ball.holder = None;
        corner.ball.position = Vec2::new(10.0, 121.0);
        corner.ball.velocity = Vec2::new(0.0, 16.0);
        corner.ball.last_touch_team = Some(Team::Away);

        corner.integrate_ball();

        assert_eq!(corner.stats.corner_kicks_home, 1);
        assert_eq!(corner.ball.last_touch_team, Some(Team::Home));
        assert_eq!(
            corner
                .ball
                .last_decision
                .as_ref()
                .expect("ball decision")
                .action,
            "corner-kick"
        );
        assert!(corner
            .events
            .iter()
            .any(|event| event.kind == "corner-kick"));
    }

    #[test]
    fn tackle_resolution_is_probabilistic_between_dribbling_and_defense() {
        let mut tackle_wins = 0;
        let mut dribble_survives = 0;
        for seed in 0..220 {
            let mut sim = SoccerMatch::default_11v11(MatchConfig {
                seed,
                ..Default::default()
            });
            sim.players[0].position = Vec2::new(40.0, 60.0);
            sim.players[0].skills.defending = 0.80;
            sim.players[0].skills.aggression = 0.55;
            sim.players[11].position = Vec2::new(41.2, 60.0);
            sim.players[11].skills.dribbling = 0.70;
            sim.players[11].skills.first_touch = 0.70;
            sim.ball.holder = Some(11);
            sim.ball.position = sim.players[11].position;

            let p = tackle_success_probability(&sim.players[0].skills, &sim.players[11].skills);
            assert!(p > 0.5 && p < 1.0);

            sim.apply_player_intent(PlayerIntent {
                player_id: 0,
                action: SoccerAction::Tackle { target_player: 11 },
                sprint: true,
            });

            if sim.ball.holder == Some(0) {
                tackle_wins += 1;
            } else if sim.ball.holder == Some(11) {
                dribble_survives += 1;
            }
        }

        assert!(tackle_wins > 0);
        assert!(dribble_survives > 0);
    }

    #[test]
    fn tackle_foul_probability_rises_with_poor_timing_and_aggression() {
        let clean_defender = SkillProfile {
            defending: 0.96,
            aggression: 0.18,
            ..neutral_tracking_skill_profile(PlayerRole::Defender)
        };
        let reckless_defender = SkillProfile {
            defending: 0.20,
            aggression: 0.96,
            ..neutral_tracking_skill_profile(PlayerRole::Defender)
        };
        let attacker = SkillProfile {
            dribbling: 0.88,
            first_touch: 0.86,
            ..neutral_tracking_skill_profile(PlayerRole::Forward)
        };

        let clean = tackle_foul_probability(&clean_defender, &attacker, 0.4, 1.0);
        let reckless = tackle_foul_probability(&reckless_defender, &attacker, 1.8, 8.0);

        assert!(reckless > clean * 2.5);
        assert!(clean > 0.0 && reckless < 1.0);
    }

    #[test]
    fn center_ref_foul_call_awards_free_kick_restart() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let spot = Vec2::new(42.0, 63.0);
        sim.players[0].position = Vec2::new(41.0, 63.0);
        sim.players[11].position = spot;
        sim.ball.holder = Some(11);
        sim.ball.position = spot;

        sim.call_foul(Team::Home, 0, 11, spot);

        assert_eq!(sim.stats.fouls_home, 1);
        assert_eq!(sim.stats.free_kicks_away, 1);
        assert_eq!(sim.ball.position, spot);
        assert_eq!(sim.ball.velocity, Vec2::zero());
        assert_eq!(sim.ball.last_touch_team, Some(Team::Away));
        let holder = sim.ball.holder.expect("free kick taker");
        assert_eq!(sim.players[holder].team, Team::Away);
        let center_ref = sim
            .officials
            .iter()
            .find(|official| official.kind == OfficialKind::CenterReferee)
            .expect("center ref");
        assert_eq!(center_ref.position, spot);
        assert!(sim.events.iter().any(|event| event.kind == "foul"));
        assert!(sim.events.iter().any(|event| event.kind == "free-kick"));
        assert_eq!(
            sim.ball
                .last_decision
                .as_ref()
                .expect("ball decision")
                .action,
            "free-kick"
        );
    }

    #[test]
    fn pass_execution_error_grows_with_pressure_and_lower_skill() {
        let from = Vec2::new(20.0, 45.0);
        let target = Vec2::new(55.0, 70.0);
        let distance = from.distance(target);
        let mut clean_rng = mulberry32(501);
        let mut pressured_rng = mulberry32(501);
        let mut clean_error = 0.0;
        let mut pressured_error = 0.0;
        for _ in 0..80 {
            clean_error += noisy_pass_target(from, target, 0.92, 0.05, distance, &mut clean_rng)
                .distance(target);
            pressured_error +=
                noisy_pass_target(from, target, 0.35, 0.95, distance, &mut pressured_rng)
                    .distance(target);
        }

        assert!(pressured_error > clean_error * 2.0);
    }

    #[test]
    fn keeper_save_is_probabilistic_and_updates_shot_stats() {
        let mut saw_save = false;
        for seed in 0..60 {
            let mut sim = SoccerMatch::default_11v11(MatchConfig {
                seed,
                ..Default::default()
            });
            let keeper_id = sim.goalkeeper_for(Team::Away).expect("away keeper");
            sim.players[keeper_id].position = Vec2::new(40.0, 118.4);
            sim.players[keeper_id].skills.defending = 0.98;
            sim.players[keeper_id].skills.first_touch = 0.98;
            sim.players[keeper_id].skills.acceleration_yps2 = 9.5;
            sim.ball.holder = None;
            sim.ball.position = Vec2::new(40.0, 121.0);
            sim.ball.velocity = Vec2::new(0.0, 22.0);
            sim.pending_shot = Some(PendingShot {
                team: Team::Home,
                shooter: 9,
            });

            sim.integrate_ball();

            assert_eq!(sim.stats.shots_on_target_home, 1);
            if sim.stats.saves_away == 1 {
                assert_eq!(sim.ball.holder, Some(keeper_id));
                assert_eq!(sim.score_home, 0);
                saw_save = true;
                break;
            }
        }

        assert!(saw_save);
    }

    #[test]
    fn live_http_routes_state_and_step_json() {
        let session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 2,
            seed: 55,
            ..Default::default()
        })));
        let input_queue = session.lock().unwrap().input_queue();

        let state = handle_live_soccer_request(
            "GET /api/state HTTP/1.1\r\nHost: local\r\n\r\n",
            &session,
            &input_queue,
        );
        assert_eq!(state.status, 200);
        assert!(state.body.contains("\"controllerAssignments\""));

        let body = r#"{"ticks":2,"recordEveryTicks":1}"#;
        let step = handle_live_soccer_request(
            &format!(
                "POST /api/step HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
            &session,
            &input_queue,
        );
        assert_eq!(step.status, 200);
        let value: serde_json::Value = serde_json::from_str(&step.body).expect("step json");
        assert_eq!(value["summary"]["ticks"], 2);
        assert_eq!(value["frames"].as_array().unwrap().len(), 2);
        assert_eq!(value["learningTransitions"].as_array().unwrap().len(), 44);
        assert!(value["frame"].get("homeDirective").is_some());
        assert!(value["frame"].get("awayDirective").is_some());
        assert!(value["frame"]["homeDirective"]
            .get("pressIntensity")
            .is_some());
        assert!(value["frame"]["ball"].get("acceleration").is_some());
        assert!(value["summary"]["stats"]
            .get("looseBallRecoveriesHome")
            .is_some());
        assert!(value["summary"]["stats"].get("shotsOnTargetHome").is_some());
        assert!(value["summary"]["stats"].get("savesAway").is_some());
        assert!(value["summary"]["stats"].get("offsidesHome").is_some());
        assert!(value["summary"]["stats"].get("throwInsHome").is_some());
        assert!(value["summary"]["stats"].get("goalKicksAway").is_some());
        assert!(value["summary"]["stats"].get("cornerKicksHome").is_some());
        assert!(value["summary"]["stats"].get("foulsHome").is_some());
        assert!(value["summary"]["stats"].get("freeKicksAway").is_some());
    }

    #[test]
    fn live_http_assign_route_updates_controller_assignments() {
        let session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 2,
            seed: 58,
            ..Default::default()
        })));
        let input_queue = session.lock().unwrap().input_queue();

        let body = r#"{"controllerSlot":0,"playerId":5}"#;
        let assign = handle_live_soccer_request(
            &format!(
                "POST /api/assign HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
            &session,
            &input_queue,
        );
        assert_eq!(assign.status, 200);
        let assign_value: serde_json::Value =
            serde_json::from_str(&assign.body).expect("assignment json");
        assert!(assign_value["controllerAssignments"]
            .as_array()
            .unwrap()
            .iter()
            .any(|a| a["controllerSlot"] == 0 && a["playerId"] == 5));
        assert_eq!(
            session.lock().unwrap().match_ref().players[5].controller_slot,
            Some(0)
        );

        let start_x = session.lock().unwrap().match_ref().players[5].position.x;
        let step_body = r#"{"ticks":1,"inputs":[{"controllerSlot":0,"playerId":5,"seq":1,"axis":{"x":1.0,"y":0.0},"sprint":true,"pass":false,"shoot":false,"targetPlayer":null}]}"#;
        let step = handle_live_soccer_request(
            &format!(
                "POST /api/step HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                step_body.len(),
                step_body
            ),
            &session,
            &input_queue,
        );
        assert_eq!(step.status, 200);
        assert!(session.lock().unwrap().match_ref().players[5].position.x > start_x);
    }

    #[test]
    fn live_http_input_route_feeds_next_step() {
        let session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 1,
            seed: 56,
            ..Default::default()
        })));
        let input_queue = session.lock().unwrap().input_queue();
        let start_x = session.lock().unwrap().match_ref().players[0].position.x;

        let input = r#"{"controllerSlot":0,"playerId":0,"seq":1,"axis":{"x":1.0,"y":0.0},"sprint":true,"pass":false,"shoot":false,"targetPlayer":null}"#;
        let ack = handle_live_soccer_request(
            &format!(
                "POST /api/input HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                input.len(),
                input
            ),
            &session,
            &input_queue,
        );
        assert_eq!(ack.status, 200);
        let ack_value: serde_json::Value = serde_json::from_str(&ack.body).expect("ack json");
        assert_eq!(ack_value["acceptedInputs"], 1);

        let step_body = r#"{"ticks":1}"#;
        let step = handle_live_soccer_request(
            &format!(
                "POST /api/step HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                step_body.len(),
                step_body
            ),
            &session,
            &input_queue,
        );
        assert_eq!(step.status, 200);
        assert!(session.lock().unwrap().match_ref().players[0].position.x > start_x);
    }

    #[test]
    fn live_http_input_route_counts_only_newer_controller_frames() {
        let session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 1,
            seed: 57,
            ..Default::default()
        })));
        let input_queue = session.lock().unwrap().input_queue();
        let body = r#"[
            {"controllerSlot":0,"playerId":0,"seq":2,"axis":{"x":1.0,"y":0.0},"sprint":true,"pass":false,"shoot":false,"targetPlayer":null},
            {"controllerSlot":0,"playerId":0,"seq":1,"axis":{"x":-1.0,"y":0.0},"sprint":false,"pass":false,"shoot":false,"targetPlayer":null}
        ]"#;

        let ack = handle_live_soccer_request(
            &format!(
                "POST /api/input HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
            &session,
            &input_queue,
        );
        assert_eq!(ack.status, 200);
        let value: serde_json::Value = serde_json::from_str(&ack.body).expect("ack json");
        assert_eq!(value["acceptedInputs"], 1);
        assert_eq!(input_queue.last_seq_for_slot(0), Some(2));

        let latest = input_queue.drain_latest_by_slot();
        assert_eq!(latest.get(&0).expect("slot input").seq, 2);
    }

    #[test]
    fn shot_lane_detects_defender_between_ball_and_goal() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.players[9].position = Vec2::new(40.0, 101.0);
        sim.players[11 + 2].position = Vec2::new(40.0, 110.0);
        sim.ball.holder = Some(9);
        let snapshot = WorldSnapshot::from_match(&sim);
        assert!(!shot_lane_is_clear(&snapshot, 9));
    }

    fn sample_tracking_pass_dataset() -> SoccerTrackingDataset {
        let config = MatchConfig {
            duration_seconds: 0.2,
            seed: 101,
            ..Default::default()
        };
        SoccerTrackingDataset {
            source: "unit-pass".to_string(),
            config,
            frames: vec![
                SoccerTrackingFrame {
                    tick: 0,
                    clock_seconds: 0.0,
                    ball_position: Vec2::new(40.0, 70.0),
                    ball_velocity: Some(Vec2::zero()),
                    ball_holder: Some(0),
                    last_touch_team: Some(Team::Home),
                    score_home: Some(0),
                    score_away: Some(0),
                    players: vec![
                        SoccerTrackingPlayerSample {
                            id: 0,
                            name: Some("Home passer".to_string()),
                            team: Team::Home,
                            role: PlayerRole::Midfielder,
                            shirt: Some(8),
                            position: Vec2::new(40.0, 70.0),
                            velocity: None,
                            home_position: Some(Vec2::new(40.0, 65.0)),
                        },
                        SoccerTrackingPlayerSample {
                            id: 1,
                            name: Some("Home runner".to_string()),
                            team: Team::Home,
                            role: PlayerRole::Forward,
                            shirt: Some(9),
                            position: Vec2::new(44.0, 82.0),
                            velocity: None,
                            home_position: Some(Vec2::new(44.0, 80.0)),
                        },
                        SoccerTrackingPlayerSample {
                            id: 2,
                            name: Some("Away defender".to_string()),
                            team: Team::Away,
                            role: PlayerRole::Defender,
                            shirt: Some(4),
                            position: Vec2::new(58.0, 78.0),
                            velocity: None,
                            home_position: Some(Vec2::new(58.0, 78.0)),
                        },
                    ],
                },
                SoccerTrackingFrame {
                    tick: 1,
                    clock_seconds: 0.1,
                    ball_position: Vec2::new(44.0, 82.0),
                    ball_velocity: Some(Vec2::new(8.0, 16.0)),
                    ball_holder: Some(1),
                    last_touch_team: Some(Team::Home),
                    score_home: Some(0),
                    score_away: Some(0),
                    players: vec![
                        SoccerTrackingPlayerSample {
                            id: 0,
                            name: Some("Home passer".to_string()),
                            team: Team::Home,
                            role: PlayerRole::Midfielder,
                            shirt: Some(8),
                            position: Vec2::new(40.2, 70.4),
                            velocity: None,
                            home_position: Some(Vec2::new(40.0, 65.0)),
                        },
                        SoccerTrackingPlayerSample {
                            id: 1,
                            name: Some("Home runner".to_string()),
                            team: Team::Home,
                            role: PlayerRole::Forward,
                            shirt: Some(9),
                            position: Vec2::new(44.0, 82.0),
                            velocity: None,
                            home_position: Some(Vec2::new(44.0, 80.0)),
                        },
                        SoccerTrackingPlayerSample {
                            id: 2,
                            name: Some("Away defender".to_string()),
                            team: Team::Away,
                            role: PlayerRole::Defender,
                            shirt: Some(4),
                            position: Vec2::new(56.5, 78.5),
                            velocity: None,
                            home_position: Some(Vec2::new(58.0, 78.0)),
                        },
                    ],
                },
            ],
        }
    }

    fn sample_tracking_tackle_dataset() -> SoccerTrackingDataset {
        let mut tracking = sample_tracking_pass_dataset();
        tracking.source = "unit-tackle".to_string();
        tracking.frames[0].ball_position = Vec2::new(41.0, 60.0);
        tracking.frames[0].ball_holder = Some(2);
        tracking.frames[0].last_touch_team = Some(Team::Away);
        tracking.frames[0].players[0].position = Vec2::new(40.4, 60.0);
        tracking.frames[0].players[0].role = PlayerRole::Defender;
        tracking.frames[0].players[1].position = Vec2::new(50.0, 80.0);
        tracking.frames[0].players[2].position = Vec2::new(41.0, 60.0);
        tracking.frames[1].ball_position = Vec2::new(40.4, 60.0);
        tracking.frames[1].ball_holder = Some(0);
        tracking.frames[1].last_touch_team = Some(Team::Home);
        tracking.frames[1].players[0].position = Vec2::new(40.4, 60.0);
        tracking.frames[1].players[0].role = PlayerRole::Defender;
        tracking.frames[1].players[1].position = Vec2::new(50.0, 80.0);
        tracking.frames[1].players[2].position = Vec2::new(41.4, 60.2);
        tracking
    }
}
