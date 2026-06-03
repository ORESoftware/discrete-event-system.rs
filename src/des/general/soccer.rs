//! 2D 11v11 soccer simulation prototype.
//!
//! This module is the live-match counterpart to [`soccer_rotation`]: it models a
//! full pitch, 22 player agents, three officials, soft real-time human input
//! queues, simple ball physics, and MDP/POMDP-shaped decision surfaces. The
//! simulation itself remains single threaded; external controller threads can
//! push [`HumanInputFrame`] values into [`SharedHumanInputs`] between ticks.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::io::{ErrorKind, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Condvar, Mutex, RwLock};
use std::thread;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

use crate::des::general::general::fisher_yates_shuffle;
use crate::des::general::prng::{mulberry32, SeededRandom};
use crate::des::shared::capabilities::RandomSource;

pub const DEFAULT_DT_SECONDS: f64 = 0.1;
pub const DEFAULT_DURATION_SECONDS: f64 = 10.0 * 60.0;
pub const DEFAULT_FIELD_LENGTH_YARDS: f64 = 120.0;
pub const DEFAULT_FIELD_WIDTH_YARDS: f64 = 80.0;
pub const DEFAULT_GOAL_WIDTH_YARDS: f64 = 8.0;
pub const DEFAULT_BALL_DRAG_PER_TICK: f64 = 0.0225;
pub const DEFAULT_BALL_AIR_RESISTANCE: f64 = 0.0075;
pub const DEFAULT_BALL_GRASS_RESISTANCE_YPS2: f64 = 0.72;
pub const DEFAULT_BALL_STOP_SPEED_YPS: f64 = 0.45;
pub const DEFAULT_PLAYER_VISION_SKILL: f64 = 7.6;
pub const DEFAULT_CONTROLLER_DEBOUNCE_MS: u64 = 4;
const PLAYER_CONTROL_RADIUS_YARDS: f64 = 1.55;
const PLAYER_BODY_RADIUS_YARDS: f64 = 0.78;
const PLAYER_COLLISION_DAMPING: f64 = 0.34;
const SHOT_SAVE_DEPTH_YARDS: f64 = 1.6;
const BALL_AGENT_ID: usize = 25;
const PLAYER_POSITION_HISTORY_LIMIT: usize = 50;
const BALL_POSITION_HISTORY_LIMIT: usize = 50;
const CONTROLLER_INPUT_YIELD_MS: u64 = DEFAULT_CONTROLLER_DEBOUNCE_MS + 2;
const FIRST_TOUCH_WINDOW_TICKS: u64 = 3;
const PITCH_FINE_GRID_COLUMNS: usize = 12;
const PITCH_FINE_GRID_ROWS: usize = 16;
const PITCH_TACTICAL_GRID_COLUMNS: usize = 6;
const PITCH_TACTICAL_GRID_ROWS: usize = 8;
const PITCH_MACRO_GRID_COLUMNS: usize = 3;
const PITCH_MACRO_GRID_ROWS: usize = 4;
const PITCH_GRID_BACKOFF_LEVELS: [PitchGridLevel; 4] = [
    PitchGridLevel::Fine,
    PitchGridLevel::Tactical,
    PitchGridLevel::Macro,
    PitchGridLevel::WholePitch,
];
const PLAYER_BASE_VISION_RANGE_YARDS: f64 = 28.0;
const PLAYER_VISION_RANGE_BONUS_YARDS: f64 = 28.0;
const PLAYER_BASE_FIELD_OF_VIEW_DEGREES: f64 = 168.0;
const PLAYER_FIELD_OF_VIEW_BONUS_DEGREES: f64 = 64.0;
const PROBABILITY_REFERENCE_DT_SECONDS: f64 = 1.0;
const DRIBBLE_TOUCH_LEAD_YARDS: f64 = 0.92;
const DRIBBLE_HEAVY_TOUCH_MIN_YARDS: f64 = 2.25;
const SHOT_ON_FRAME_MIN_PROBABILITY: f64 = 0.60;
const SHOT_KEEPER_BEAT_MIN_PROBABILITY: f64 = 0.30;
const SHOT_BAILOUT_NEAR_GOAL_YARDS: f64 = 12.0;
const SHOT_BAILOUT_DISPOSSESSION_RISK: f64 = 0.80;
const SHOT_BAILOUT_ON_FRAME_PROBABILITY: f64 = 0.20;
const SHOT_BLOCK_DIRECT_PROBABILITY: f64 = 0.80;
const SHOT_BLOCK_LANE_RADIUS_YARDS: f64 = 3.25;
const SHOT_BLOCK_DECISION_MAX_PROBABILITY: f64 = 0.58;
const SHOT_BLOCK_BAILOUT_MAX_PROBABILITY: f64 = 0.86;
const SHOT_BLOCK_QUICK_RELEASE_MULTIPLIER: f64 = 0.68;
const SHOT_SCREEN_IDEAL_MIN_YARDS: f64 = 1.0;
const SHOT_SCREEN_IDEAL_MAX_YARDS: f64 = 3.0;
const BALL_CURL_DECAY_PER_SECOND: f64 = 1.10;
const MAX_BALL_CURL_YPS2: f64 = 7.6;
const POSSESSION_CHASE_MIN_BALL_RELOCATION_YARDS: f64 = 0.90;
const POSSESSION_CHASE_MIN_ACTIVE_DEFENDERS: usize = 2;
const POSSESSION_CHASE_MIN_CREDIT: f64 = 0.035;
const GOAL_REWARD_POINTS: f64 = 100.0;
const SHOT_ON_TARGET_REWARD_POINTS: f64 = 50.0;
const DEFENSIVE_RELAXATION_THREAT_YARDS: f64 = 48.0;
const NO_PRESSURE_BACK_PASS_THRESHOLD_YARDS: f64 = 10.0;
const SETTLED_POSSESSION_SECONDS: f64 = 5.0;
const ATTACK_SPACING_MIN_YARDS: f64 = 5.0;
const ATTACK_SPACING_IDEAL_YARDS: f64 = 10.0;
const ATTACK_SPACING_MAX_YARDS: f64 = 15.0;
const DEFENSE_SPACING_MIN_YARDS: f64 = 2.0;
const DEFENSE_SPACING_IDEAL_YARDS: f64 = 4.0;
const DEFENSE_SPACING_MAX_YARDS: f64 = 8.0;
const DEFENSIVE_GOAL_LINE_BUFFER_YARDS: f64 = 6.0;
const DEFENSIVE_MAX_BEHIND_BALL_YARDS: f64 = 30.0;
const STRIKER_ONSIDE_BUFFER_YARDS: f64 = 1.25;
const CENTER_REF_BALL_CLEARANCE_YARDS: f64 = 7.0;
const ASSISTANT_REF_BALL_CLEARANCE_YARDS: f64 = 4.0;
const LIVE_HTTP_MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;

fn default_ball_drag_per_tick() -> f64 {
    DEFAULT_BALL_DRAG_PER_TICK
}

fn default_ball_air_resistance() -> f64 {
    DEFAULT_BALL_AIR_RESISTANCE
}

fn default_ball_grass_resistance_yps2() -> f64 {
    DEFAULT_BALL_GRASS_RESISTANCE_YPS2
}

fn default_ball_stop_speed_yps() -> f64 {
    DEFAULT_BALL_STOP_SPEED_YPS
}

fn default_player_vision_skill() -> f64 {
    7.6
}

fn default_skill_score() -> f64 {
    7.0
}

fn default_skill_strength() -> f64 {
    7.0
}

fn default_skill_height() -> f64 {
    7.0
}

fn default_goalkeeping_skill() -> f64 {
    2.0
}

fn default_learning_enabled() -> bool {
    true
}

fn default_learning_logging_enabled() -> bool {
    true
}

fn default_learning_interval_ticks() -> usize {
    1
}

fn default_period_count() -> usize {
    1
}

fn default_period_break_recovery_seconds() -> f64 {
    0.0
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

    pub fn dot(self, other: Vec2) -> f64 {
        self.x * other.x + self.y * other.y
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IncomingBallKind {
    #[default]
    None,
    GroundPass,
    AerialPass,
    Cross,
    AerialCross,
    LooseBall,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum PassFlight {
    #[default]
    Floor,
    Aerial,
}

impl PassFlight {
    fn is_aerial(self) -> bool {
        matches!(self, PassFlight::Aerial)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct IncomingBallContext {
    #[serde(default)]
    pub from_player: Option<usize>,
    #[serde(default)]
    pub target_player: Option<usize>,
    #[serde(default)]
    pub team: Option<Team>,
    #[serde(default)]
    pub kind: IncomingBallKind,
    #[serde(default)]
    pub origin: Option<Vec2>,
    #[serde(default)]
    pub intended_target: Option<Vec2>,
    #[serde(default)]
    pub speed_yps: f64,
    #[serde(default)]
    pub distance_yards: f64,
    #[serde(default)]
    pub received_tick: u64,
    #[serde(default)]
    pub is_cross: bool,
    #[serde(default)]
    pub is_aerial: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum MovementGait {
    #[default]
    Stand,
    Walk,
    BackWalk,
    Skip,
    BackSkip,
    SideStep,
    Jog,
    Run,
    Sprint,
}

impl MovementGait {
    fn speed_multiplier(self) -> f64 {
        match self {
            MovementGait::Stand => 0.0,
            MovementGait::Walk => 0.30,
            MovementGait::BackWalk => 0.24,
            MovementGait::Skip => 0.45,
            MovementGait::BackSkip => 0.38,
            MovementGait::SideStep => 0.42,
            MovementGait::Jog => 0.62,
            MovementGait::Run => 0.84,
            MovementGait::Sprint => 1.08,
        }
    }

    fn fatigue_delta(self, stamina: f64, dt_seconds: f64) -> f64 {
        let cardio = ability01(stamina);
        let dt = dt_seconds.max(0.0);
        match self {
            MovementGait::Stand => -0.012 * (0.75 + cardio * 0.75) * dt,
            MovementGait::Walk | MovementGait::BackWalk => -0.008 * (0.70 + cardio * 0.60) * dt,
            MovementGait::Skip | MovementGait::BackSkip | MovementGait::SideStep => {
                -0.003 * (0.65 + cardio * 0.45) * dt
            }
            MovementGait::Jog => 0.003 * (1.15 - cardio * 0.55) * dt,
            MovementGait::Run => 0.010 * (1.35 - cardio * 0.65) * dt,
            MovementGait::Sprint => 0.025 * (1.55 - cardio * 0.80) * dt,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum OfficialKind {
    CenterReferee,
    AssistantRefereeNear,
    AssistantRefereeFar,
}

impl OfficialKind {
    fn label(self) -> &'static str {
        match self {
            OfficialKind::CenterReferee => "Center ref",
            OfficialKind::AssistantRefereeNear => "Near assistant",
            OfficialKind::AssistantRefereeFar => "Far assistant",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AssistantFlank {
    Near,
    Far,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SkillProfile {
    #[serde(alias = "topSpeedYps")]
    pub top_speed: f64,
    #[serde(alias = "accelerationYps2")]
    pub acceleration: f64,
    #[serde(default = "default_skill_strength")]
    pub strength: f64,
    #[serde(default = "default_skill_height", alias = "heightInches")]
    pub height: f64,
    #[serde(default = "default_skill_score")]
    pub shooting: f64,
    #[serde(default = "default_skill_score")]
    pub right_foot_shot_power: f64,
    #[serde(default = "default_skill_score")]
    pub left_foot_shot_power: f64,
    #[serde(default = "default_skill_score")]
    pub passing: f64,
    #[serde(default = "default_skill_score")]
    pub passing_completion_rate: f64,
    #[serde(default = "default_skill_score")]
    pub flair_passing: f64,
    #[serde(
        default = "default_skill_score",
        alias = "crossingAbilityWithLeftRoot",
        alias = "leftFootCrossingAbility"
    )]
    pub crossing_left: f64,
    #[serde(default = "default_skill_score", alias = "rightFootCrossingAbility")]
    pub crossing_right: f64,
    #[serde(default = "default_skill_score")]
    pub dribbling: f64,
    #[serde(default = "default_skill_score")]
    pub first_touch: f64,
    #[serde(default = "default_skill_score", alias = "defensiveAbility")]
    pub defending: f64,
    #[serde(default = "default_goalkeeping_skill", alias = "abilityInGoal")]
    pub goalkeeping: f64,
    #[serde(default = "default_skill_score", alias = "trackingBack")]
    pub defensive_tracking: f64,
    #[serde(default = "default_skill_score")]
    pub stamina: f64,
    #[serde(default = "default_player_vision_skill")]
    pub vision: f64,
    pub decision_noise: f64,
    pub aggression: f64,
}

impl SkillProfile {
    fn blended(seed: usize, role: PlayerRole, rng: &mut SeededRandom) -> Self {
        let role_bias = match role {
            PlayerRole::Goalkeeper => (
                5.8, 6.0, 8.0, 8.6, 3.6, 6.2, 4.0, 7.4, 7.2, 7.8, 7.0, 4.2, 9.2, 7.2,
            ),
            PlayerRole::Defender => (
                6.9, 6.8, 8.0, 7.6, 5.0, 6.8, 5.6, 6.5, 8.2, 8.2, 7.4, 7.0, 2.0, 8.6,
            ),
            PlayerRole::Midfielder => (
                7.4, 7.5, 6.8, 6.0, 6.7, 8.1, 7.7, 7.7, 6.8, 8.7, 8.4, 6.2, 1.7, 7.4,
            ),
            PlayerRole::Forward => (
                8.1, 8.0, 7.5, 7.0, 8.4, 7.1, 8.3, 7.5, 4.9, 8.0, 7.8, 7.4, 1.5, 5.3,
            ),
        };
        let jitter = |rng: &mut SeededRandom, scale: f64| (rng.next_float() - 0.5) * scale;
        let shooting = (role_bias.4 + jitter(rng, 1.1)).clamp(1.0, 10.0);
        let passing = (role_bias.5 + jitter(rng, 1.0)).clamp(1.0, 10.0);
        let dribbling = (role_bias.6 + jitter(rng, 1.0)).clamp(1.0, 10.0);
        let first_touch = (role_bias.7 + jitter(rng, 0.9)).clamp(1.0, 10.0);
        let defending = (role_bias.8 + jitter(rng, 1.0)).clamp(1.0, 10.0);
        let dominant_right = seed % 5 != 0;
        let strong_foot = (shooting + jitter(rng, 0.6)).clamp(1.0, 10.0);
        let weak_foot = (shooting - 1.1 + jitter(rng, 0.8)).clamp(1.0, 10.0);
        let (right_foot_shot_power, left_foot_shot_power) = if dominant_right {
            (strong_foot, weak_foot)
        } else {
            (weak_foot, strong_foot)
        };
        let crossing_left =
            (passing * 0.72 + first_touch * 0.18 + jitter(rng, 0.9)).clamp(1.0, 10.0);
        let crossing_right =
            (passing * 0.76 + first_touch * 0.16 + jitter(rng, 0.9)).clamp(1.0, 10.0);
        SkillProfile {
            top_speed: role_bias.0 + jitter(rng, 0.9) + (seed % 3) as f64 * 0.08,
            acceleration: role_bias.1 + jitter(rng, 0.8),
            strength: (role_bias.2 + jitter(rng, 0.9)).clamp(1.0, 10.0),
            height: (role_bias.3 + jitter(rng, 0.9)).clamp(1.0, 10.0),
            shooting,
            right_foot_shot_power,
            left_foot_shot_power,
            passing,
            passing_completion_rate: (passing * 0.86 + first_touch * 0.12 + jitter(rng, 0.45))
                .clamp(1.0, 10.0),
            flair_passing: (passing * 0.58 + dribbling * 0.30 + jitter(rng, 0.9)).clamp(1.0, 10.0),
            crossing_left,
            crossing_right,
            dribbling,
            first_touch,
            defending,
            goalkeeping: (role_bias.12 + jitter(rng, 0.8)).clamp(1.0, 10.0),
            defensive_tracking: (role_bias.13 * 0.78 + defending * 0.22 + jitter(rng, 0.7))
                .clamp(1.0, 10.0),
            stamina: (role_bias.9 + jitter(rng, 0.8)).clamp(1.0, 10.0),
            vision: (role_bias.10 + jitter(rng, 0.8)).clamp(1.0, 10.0),
            decision_noise: (0.08 + jitter(rng, 0.08)).clamp(0.01, 0.18),
            aggression: (role_bias.11 + jitter(rng, 0.9)).clamp(1.0, 10.0),
        }
    }
}

impl Default for SkillProfile {
    fn default() -> Self {
        SkillProfile {
            top_speed: 7.4,
            acceleration: 7.2,
            strength: 6.8,
            height: 6.6,
            shooting: 6.2,
            right_foot_shot_power: 6.6,
            left_foot_shot_power: 5.4,
            passing: 7.2,
            passing_completion_rate: 7.2,
            flair_passing: 4.2,
            crossing_left: 6.6,
            crossing_right: 7.0,
            dribbling: 6.8,
            first_touch: 7.0,
            defending: 6.2,
            goalkeeping: 2.0,
            defensive_tracking: 6.2,
            stamina: 8.2,
            vision: DEFAULT_PLAYER_VISION_SKILL,
            decision_noise: 0.08,
            aggression: 5.5,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PitchGridLevel {
    #[default]
    WholePitch,
    Macro,
    Tactical,
    Fine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchGridCell {
    pub level: PitchGridLevel,
    pub columns: usize,
    pub rows: usize,
    pub x: usize,
    pub y: usize,
    pub id: usize,
    pub parent_id: Option<usize>,
}

impl Default for PitchGridCell {
    fn default() -> Self {
        PitchGridCell {
            level: PitchGridLevel::WholePitch,
            columns: 1,
            rows: 1,
            x: 0,
            y: 0,
            id: 0,
            parent_id: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PitchGridAddress {
    pub fine: PitchGridCell,
    pub tactical: PitchGridCell,
    pub macro_zone: PitchGridCell,
    pub whole_pitch: PitchGridCell,
}

impl Default for PitchGridAddress {
    fn default() -> Self {
        let whole_pitch = PitchGridCell::default();
        PitchGridAddress {
            fine: whole_pitch,
            tactical: whole_pitch,
            macro_zone: whole_pitch,
            whole_pitch,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FacingBucket {
    #[default]
    Unknown,
    North,
    NorthEast,
    East,
    SouthEast,
    South,
    SouthWest,
    West,
    NorthWest,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerMdpState {
    pub tick: u64,
    pub ball_zone_x: usize,
    pub ball_zone_y: usize,
    #[serde(default)]
    pub player_grid: PitchGridAddress,
    #[serde(default)]
    pub receive_facing: FacingBucket,
    #[serde(default)]
    pub action_facing: FacingBucket,
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
    #[serde(default)]
    pub player_grid: PitchGridAddress,
    #[serde(default)]
    pub receive_facing: FacingBucket,
    #[serde(default)]
    pub action_facing: FacingBucket,
    pub has_ball: bool,
    #[serde(default)]
    pub visible_ball: bool,
    #[serde(default)]
    pub visible_teammates: usize,
    #[serde(default)]
    pub visible_opponents: usize,
    #[serde(default)]
    pub visible_pass_options: usize,
    #[serde(default)]
    pub visible_aerial_pass_options: usize,
    #[serde(default)]
    pub floor_pass_lane_score: f64,
    #[serde(default)]
    pub aerial_pass_bypass_score: f64,
    #[serde(default)]
    pub aerial_pass_interception_risk: f64,
    #[serde(default)]
    pub ball_position_confidence: f64,
    #[serde(default)]
    pub teammate_position_confidence: f64,
    #[serde(default)]
    pub opponent_position_confidence: f64,
    #[serde(default)]
    pub player_position_confidences: Vec<PlayerPositionConfidence>,
    pub ball_distance: f64,
    pub nearest_opponent_distance: f64,
    pub nearest_teammate_distance: f64,
    #[serde(default)]
    pub team_spacing_score: f64,
    #[serde(default)]
    pub preferred_team_spacing_yards: f64,
    pub shot_lane_open: bool,
    #[serde(default)]
    pub shot_block_probability: f64,
    #[serde(default)]
    pub shot_blocker_distance_yards: f64,
    #[serde(default)]
    pub shot_on_frame_probability: f64,
    #[serde(default)]
    pub shot_beat_goalkeeper_probability: f64,
    #[serde(default)]
    pub shot_curl_probability: f64,
    #[serde(default)]
    pub pass_curl_probability: f64,
    #[serde(default)]
    pub immediate_dispossession_risk: f64,
    pub yards_to_goal: f64,
    #[serde(default)]
    pub yards_to_own_goal: f64,
    #[serde(default)]
    pub opponent_goal_angle_degrees: f64,
    #[serde(default)]
    pub opposing_goalkeeper_distance: f64,
    #[serde(default)]
    pub opposing_goalkeeper_angle_degrees: f64,
    #[serde(default)]
    pub forward_dribble_space_yards: f64,
    #[serde(default)]
    pub real_pressure: f64,
    #[serde(default)]
    pub perceived_pressure: f64,
    #[serde(default)]
    pub real_time_on_ball_seconds: f64,
    #[serde(default)]
    pub perceived_time_on_ball_seconds: f64,
    #[serde(default)]
    pub fatigue: f64,
    #[serde(default)]
    pub nearest_defender_fatigue: f64,
    #[serde(default)]
    pub perceived_nearest_defender_fatigue: f64,
    #[serde(default)]
    pub nearest_defender_fatigue_confidence: f64,
    #[serde(default)]
    pub perceived_fatigue_advantage: f64,
    #[serde(default)]
    pub first_touch_available: bool,
    #[serde(default)]
    pub incoming_ball_kind: IncomingBallKind,
    #[serde(default)]
    pub incoming_ball_speed_yps: f64,
    #[serde(default)]
    pub incoming_ball_distance_yards: f64,
    #[serde(default)]
    pub receiving_pending_pass: bool,
    #[serde(default)]
    pub pending_pass_off_target_yards: f64,
    #[serde(default)]
    pub pending_pass_receiver_urgency: f64,
    #[serde(default)]
    pub first_time_shot_score: f64,
    #[serde(default)]
    pub first_time_pass_score: f64,
    #[serde(default)]
    pub control_touch_score: f64,
    #[serde(default)]
    pub skill_top_speed: f64,
    #[serde(default)]
    pub skill_acceleration: f64,
    #[serde(default)]
    pub skill_stamina: f64,
    #[serde(default)]
    pub skill_strength: f64,
    #[serde(default)]
    pub skill_height: f64,
    #[serde(default)]
    pub skill_dribbling: f64,
    #[serde(default)]
    pub skill_aggression: f64,
    #[serde(default)]
    pub skill_defending: f64,
    #[serde(default)]
    pub skill_right_foot_shot_power: f64,
    #[serde(default)]
    pub skill_left_foot_shot_power: f64,
    #[serde(default)]
    pub skill_passing_completion_rate: f64,
    #[serde(default)]
    pub skill_flair_passing: f64,
    #[serde(default)]
    pub skill_crossing_left: f64,
    #[serde(default)]
    pub skill_crossing_right: f64,
    #[serde(default)]
    pub skill_goalkeeping: f64,
    #[serde(default)]
    pub skill_defensive_tracking: f64,
    pub open_space_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PlayerPositionConfidence {
    pub observer_id: usize,
    pub player_id: usize,
    pub team: Team,
    pub distance_yards: f64,
    pub in_front: bool,
    pub confidence: f64,
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
pub struct AgentActionOptionTrace {
    pub label: String,
    pub score: f64,
    pub probability: f64,
    pub legal: bool,
}

impl AgentActionOptionTrace {
    fn new(label: impl Into<String>, score: f64, legal: bool) -> Self {
        AgentActionOptionTrace {
            label: label.into(),
            score: if score.is_finite() {
                score.max(0.0)
            } else {
                0.0
            },
            probability: 0.0,
            legal,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentActionTargetTrace {
    #[serde(default)]
    pub point: Option<Vec2>,
    #[serde(default)]
    pub player_id: Option<usize>,
    #[serde(default)]
    pub grid: Option<PitchGridAddress>,
    #[serde(default)]
    pub facing: FacingBucket,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentDecisionTrace {
    pub mdp_state: SoccerMdpState,
    pub observation: SoccerPomdpObservation,
    pub belief: BeliefState,
    pub operation_order: Vec<String>,
    #[serde(default)]
    pub action_options: Vec<AgentActionOptionTrace>,
    #[serde(default)]
    pub action_target: Option<AgentActionTargetTrace>,
    pub action: String,
}

fn normalize_action_options(
    mut options: Vec<AgentActionOptionTrace>,
) -> Vec<AgentActionOptionTrace> {
    let total: f64 = options
        .iter()
        .filter(|option| option.legal)
        .map(|option| option.score.max(0.0))
        .sum();
    let legal_count = options.iter().filter(|option| option.legal).count();
    for option in &mut options {
        option.probability = if !option.legal {
            0.0
        } else if total > 1e-9 {
            option.score.max(0.0) / total
        } else if legal_count > 0 {
            1.0 / legal_count as f64
        } else {
            0.0
        };
    }
    options
}

fn single_action_option(label: &str) -> Vec<AgentActionOptionTrace> {
    vec![AgentActionOptionTrace {
        label: label.to_string(),
        score: 1.0,
        probability: 1.0,
        legal: true,
    }]
}

fn action_option_score(options: &[AgentActionOptionTrace], label: &str) -> f64 {
    options
        .iter()
        .find(|option| option.label == label)
        .filter(|option| option.legal)
        .map(|option| option.score.max(0.0))
        .unwrap_or(0.0)
}

fn weighted_fisher_yates_order<T>(mut items: Vec<(T, f64)>, rng: &mut SeededRandom) -> Vec<T> {
    let mut ordered = Vec::with_capacity(items.len());
    while !items.is_empty() {
        let total: f64 = items.iter().map(|(_, weight)| weight.max(0.0)).sum();
        let chosen_idx = if total > 1e-9 {
            let mut draw = rng.next_float() * total;
            let mut idx = items.len() - 1;
            for (candidate_idx, (_, weight)) in items.iter().enumerate() {
                draw -= weight.max(0.0);
                if draw <= 0.0 {
                    idx = candidate_idx;
                    break;
                }
            }
            idx
        } else {
            ((rng.next_float() * items.len() as f64).floor() as usize).min(items.len() - 1)
        };
        ordered.push(items.swap_remove(chosen_idx).0);
    }
    ordered
}

fn time_window_probability(probability_at_reference_dt: f64, dt_seconds: f64) -> f64 {
    let p = probability_at_reference_dt.clamp(0.0, 1.0);
    if p <= 0.0 || dt_seconds <= 0.0 {
        return 0.0;
    }
    if p >= 1.0 {
        return 1.0;
    }
    let scale = (dt_seconds / PROBABILITY_REFERENCE_DT_SECONDS).max(0.0);
    (1.0 - (1.0 - p).powf(scale)).clamp(0.0, 1.0)
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
    #[serde(default)]
    pub action_target: Option<AgentActionTargetTrace>,
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
    #[serde(default)]
    pub player_fine_cell_id: usize,
    #[serde(default)]
    pub player_tactical_cell_id: usize,
    #[serde(default)]
    pub player_macro_cell_id: usize,
    #[serde(default)]
    pub player_root_cell_id: usize,
    #[serde(default)]
    pub receive_facing: FacingBucket,
    #[serde(default)]
    pub action_facing: FacingBucket,
    pub score_diff_bucket: i8,
    pub has_ball: bool,
    pub visible_ball: bool,
    pub shot_lane_open: bool,
    #[serde(default)]
    pub shot_on_frame_probability_bin: u8,
    #[serde(default)]
    pub shot_beat_goalkeeper_probability_bin: u8,
    #[serde(default)]
    pub shot_block_probability_bin: u8,
    #[serde(default)]
    pub shot_blocker_distance_bin: u8,
    #[serde(default)]
    pub shot_curl_probability_bin: u8,
    #[serde(default)]
    pub pass_curl_probability_bin: u8,
    #[serde(default)]
    pub immediate_dispossession_risk_bin: u8,
    pub visible_pass_options_bin: u8,
    #[serde(default)]
    pub visible_aerial_pass_options_bin: u8,
    #[serde(default)]
    pub floor_pass_lane_score_bin: u8,
    #[serde(default)]
    pub aerial_pass_bypass_score_bin: u8,
    #[serde(default)]
    pub aerial_pass_interception_risk_bin: u8,
    #[serde(default)]
    pub ball_position_confidence_bin: u8,
    #[serde(default)]
    pub teammate_position_confidence_bin: u8,
    #[serde(default)]
    pub opponent_position_confidence_bin: u8,
    pub ball_distance_bin: u8,
    pub yards_to_goal_bin: u8,
    #[serde(default)]
    pub yards_to_own_goal_bin: u8,
    #[serde(default)]
    pub opponent_goal_angle_bin: u8,
    #[serde(default)]
    pub opposing_goalkeeper_distance_bin: u8,
    #[serde(default)]
    pub forward_dribble_space_bin: u8,
    #[serde(default)]
    pub team_spacing_score_bin: u8,
    #[serde(default)]
    pub preferred_team_spacing_bin: u8,
    #[serde(default)]
    pub perceived_time_on_ball_bin: u8,
    #[serde(default)]
    pub fatigue_bin: u8,
    #[serde(default)]
    pub nearest_defender_fatigue_bin: u8,
    #[serde(default)]
    pub nearest_defender_fatigue_confidence_bin: u8,
    #[serde(default)]
    pub perceived_fatigue_advantage_bin: u8,
    pub pressure_bin: u8,
    #[serde(default)]
    pub perceived_pressure_bin: u8,
    #[serde(default)]
    pub first_touch_kind: IncomingBallKind,
    #[serde(default)]
    pub incoming_ball_speed_bin: u8,
    #[serde(default)]
    pub receiving_pending_pass: bool,
    #[serde(default)]
    pub pending_pass_off_target_bin: u8,
    #[serde(default)]
    pub pending_pass_receiver_urgency_bin: u8,
    #[serde(default)]
    pub first_time_shot_bin: u8,
    #[serde(default)]
    pub control_touch_bin: u8,
    #[serde(default)]
    pub skill_top_speed_bin: u8,
    #[serde(default)]
    pub skill_acceleration_bin: u8,
    #[serde(default)]
    pub skill_stamina_bin: u8,
    #[serde(default)]
    pub skill_strength_bin: u8,
    #[serde(default)]
    pub skill_height_bin: u8,
    #[serde(default)]
    pub skill_dribbling_bin: u8,
    #[serde(default)]
    pub skill_aggression_bin: u8,
    #[serde(default)]
    pub skill_right_foot_shot_bin: u8,
    #[serde(default)]
    pub skill_left_foot_shot_bin: u8,
    #[serde(default)]
    pub skill_passing_completion_bin: u8,
    #[serde(default)]
    pub skill_flair_passing_bin: u8,
    #[serde(default)]
    pub skill_crossing_bin: u8,
    #[serde(default)]
    pub skill_crossing_left_bin: u8,
    #[serde(default)]
    pub skill_crossing_right_bin: u8,
    #[serde(default)]
    pub skill_goalkeeping_bin: u8,
    #[serde(default)]
    pub skill_defending_bin: u8,
    #[serde(default)]
    pub skill_defensive_tracking_bin: u8,
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
        let player_grid = if state.player_grid.fine.level == PitchGridLevel::WholePitch
            && observation.player_grid.fine.level != PitchGridLevel::WholePitch
        {
            observation.player_grid
        } else {
            state.player_grid
        };
        let receive_facing = if state.receive_facing == FacingBucket::Unknown {
            observation.receive_facing
        } else {
            state.receive_facing
        };
        let action_facing = if state.action_facing == FacingBucket::Unknown {
            observation.action_facing
        } else {
            state.action_facing
        };
        SoccerQStateKey {
            phase: state.phase,
            role,
            possession_relative,
            ball_zone_x: state.ball_zone_x,
            ball_zone_y: state.ball_zone_y,
            player_fine_cell_id: player_grid.fine.id,
            player_tactical_cell_id: player_grid.tactical.id,
            player_macro_cell_id: player_grid.macro_zone.id,
            player_root_cell_id: player_grid.whole_pitch.id,
            receive_facing,
            action_facing,
            score_diff_bucket: score_diff_for_team.clamp(-2, 2) as i8,
            has_ball: observation.has_ball,
            visible_ball: observation.visible_ball,
            shot_lane_open: observation.shot_lane_open,
            shot_on_frame_probability_bin: distance_bucket(
                observation.shot_on_frame_probability,
                &[0.20, 0.40, 0.60, 0.78],
            ),
            shot_beat_goalkeeper_probability_bin: distance_bucket(
                observation.shot_beat_goalkeeper_probability,
                &[0.15, 0.25, 0.35, 0.50],
            ),
            shot_block_probability_bin: distance_bucket(
                observation.shot_block_probability,
                &[0.20, 0.40, 0.62, 0.82],
            ),
            shot_blocker_distance_bin: distance_bucket(
                observation.shot_blocker_distance_yards,
                &[3.0, 7.0, 14.0, 28.0],
            ),
            shot_curl_probability_bin: distance_bucket(
                observation.shot_curl_probability,
                &[0.10, 0.24, 0.42, 0.62],
            ),
            pass_curl_probability_bin: distance_bucket(
                observation.pass_curl_probability,
                &[0.10, 0.24, 0.42, 0.62],
            ),
            immediate_dispossession_risk_bin: distance_bucket(
                observation.immediate_dispossession_risk,
                &[0.20, 0.45, 0.70, 0.85],
            ),
            visible_pass_options_bin: observation.visible_pass_options.min(3) as u8,
            visible_aerial_pass_options_bin: observation.visible_aerial_pass_options.min(3) as u8,
            floor_pass_lane_score_bin: distance_bucket(
                observation.floor_pass_lane_score,
                &[0.15, 0.35, 0.60, 0.82],
            ),
            aerial_pass_bypass_score_bin: distance_bucket(
                observation.aerial_pass_bypass_score,
                &[0.15, 0.35, 0.60, 0.82],
            ),
            aerial_pass_interception_risk_bin: distance_bucket(
                observation.aerial_pass_interception_risk,
                &[0.15, 0.35, 0.60, 0.82],
            ),
            ball_position_confidence_bin: confidence_bucket(observation.ball_position_confidence),
            teammate_position_confidence_bin: confidence_bucket(
                observation.teammate_position_confidence,
            ),
            opponent_position_confidence_bin: confidence_bucket(
                observation.opponent_position_confidence,
            ),
            ball_distance_bin: distance_bucket(observation.ball_distance, &[3.0, 8.0, 18.0, 36.0]),
            yards_to_goal_bin: distance_bucket(
                observation.yards_to_goal,
                &[12.0, 20.0, 35.0, 55.0],
            ),
            yards_to_own_goal_bin: distance_bucket(
                observation.yards_to_own_goal,
                &[12.0, 24.0, 45.0, 70.0],
            ),
            opponent_goal_angle_bin: distance_bucket(
                observation.opponent_goal_angle_degrees,
                &[8.0, 16.0, 28.0, 42.0],
            ),
            opposing_goalkeeper_distance_bin: distance_bucket(
                observation.opposing_goalkeeper_distance,
                &[8.0, 16.0, 28.0, 45.0],
            ),
            forward_dribble_space_bin: distance_bucket(
                observation.forward_dribble_space_yards,
                &[3.0, 8.0, 14.0, 24.0],
            ),
            team_spacing_score_bin: distance_bucket(
                observation.team_spacing_score,
                &[-0.45, 0.0, 0.45, 0.80],
            ),
            preferred_team_spacing_bin: distance_bucket(
                observation.preferred_team_spacing_yards,
                &[4.0, 8.0, 12.0, 16.0],
            ),
            perceived_time_on_ball_bin: distance_bucket(
                observation.perceived_time_on_ball_seconds,
                &[0.35, 0.75, 1.3, 2.2],
            ),
            fatigue_bin: distance_bucket(observation.fatigue, &[0.15, 0.35, 0.60, 0.82]),
            nearest_defender_fatigue_bin: distance_bucket(
                observation.perceived_nearest_defender_fatigue,
                &[0.15, 0.35, 0.60, 0.82],
            ),
            nearest_defender_fatigue_confidence_bin: confidence_bucket(
                observation.nearest_defender_fatigue_confidence,
            ),
            perceived_fatigue_advantage_bin: fatigue_advantage_bucket(
                observation.perceived_fatigue_advantage,
            ),
            pressure_bin: pressure_bucket(observation.nearest_opponent_distance),
            perceived_pressure_bin: distance_bucket(
                observation.perceived_pressure,
                &[0.15, 0.35, 0.60, 0.82],
            ),
            first_touch_kind: if observation.first_touch_available {
                observation.incoming_ball_kind
            } else {
                IncomingBallKind::None
            },
            incoming_ball_speed_bin: distance_bucket(
                observation.incoming_ball_speed_yps,
                &[8.0, 14.0, 22.0, 32.0],
            ),
            receiving_pending_pass: observation.receiving_pending_pass,
            pending_pass_off_target_bin: distance_bucket(
                observation.pending_pass_off_target_yards,
                &[0.75, 1.5, 3.0, 5.5],
            ),
            pending_pass_receiver_urgency_bin: distance_bucket(
                observation.pending_pass_receiver_urgency,
                &[0.20, 0.40, 0.60, 0.82],
            ),
            first_time_shot_bin: distance_bucket(
                observation.first_time_shot_score,
                &[0.15, 0.35, 0.60, 0.82],
            ),
            control_touch_bin: distance_bucket(
                observation.control_touch_score,
                &[0.15, 0.35, 0.60, 0.82],
            ),
            skill_top_speed_bin: skill_bucket(observation.skill_top_speed),
            skill_acceleration_bin: skill_bucket(observation.skill_acceleration),
            skill_stamina_bin: skill_bucket(observation.skill_stamina),
            skill_strength_bin: skill_bucket(observation.skill_strength),
            skill_height_bin: skill_bucket(observation.skill_height),
            skill_dribbling_bin: skill_bucket(observation.skill_dribbling),
            skill_aggression_bin: skill_bucket(observation.skill_aggression),
            skill_defending_bin: skill_bucket(observation.skill_defending),
            skill_right_foot_shot_bin: skill_bucket(observation.skill_right_foot_shot_power),
            skill_left_foot_shot_bin: skill_bucket(observation.skill_left_foot_shot_power),
            skill_passing_completion_bin: skill_bucket(observation.skill_passing_completion_rate),
            skill_flair_passing_bin: skill_bucket(observation.skill_flair_passing),
            skill_crossing_bin: skill_bucket(
                observation
                    .skill_crossing_left
                    .max(observation.skill_crossing_right),
            ),
            skill_crossing_left_bin: skill_bucket(observation.skill_crossing_left),
            skill_crossing_right_bin: skill_bucket(observation.skill_crossing_right),
            skill_goalkeeping_bin: skill_bucket(observation.skill_goalkeeping),
            skill_defensive_tracking_bin: skill_bucket(observation.skill_defensive_tracking),
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

    fn matches_learning_context(&self, other: &Self) -> bool {
        self.phase == other.phase
            && self.role == other.role
            && self.possession_relative == other.possession_relative
            && self.ball_zone_x == other.ball_zone_x
            && self.ball_zone_y == other.ball_zone_y
            && self.score_diff_bucket == other.score_diff_bucket
            && self.has_ball == other.has_ball
            && self.visible_ball == other.visible_ball
            && self.shot_lane_open == other.shot_lane_open
            && self.shot_on_frame_probability_bin == other.shot_on_frame_probability_bin
            && self.shot_beat_goalkeeper_probability_bin
                == other.shot_beat_goalkeeper_probability_bin
            && self.shot_block_probability_bin == other.shot_block_probability_bin
            && self.shot_blocker_distance_bin == other.shot_blocker_distance_bin
            && self.shot_curl_probability_bin == other.shot_curl_probability_bin
            && self.pass_curl_probability_bin == other.pass_curl_probability_bin
            && self.immediate_dispossession_risk_bin == other.immediate_dispossession_risk_bin
            && self.visible_pass_options_bin == other.visible_pass_options_bin
            && self.visible_aerial_pass_options_bin == other.visible_aerial_pass_options_bin
            && self.floor_pass_lane_score_bin == other.floor_pass_lane_score_bin
            && self.aerial_pass_bypass_score_bin == other.aerial_pass_bypass_score_bin
            && self.aerial_pass_interception_risk_bin == other.aerial_pass_interception_risk_bin
            && self.ball_position_confidence_bin == other.ball_position_confidence_bin
            && self.teammate_position_confidence_bin == other.teammate_position_confidence_bin
            && self.opponent_position_confidence_bin == other.opponent_position_confidence_bin
            && self.ball_distance_bin == other.ball_distance_bin
            && self.yards_to_goal_bin == other.yards_to_goal_bin
            && self.yards_to_own_goal_bin == other.yards_to_own_goal_bin
            && self.opponent_goal_angle_bin == other.opponent_goal_angle_bin
            && self.opposing_goalkeeper_distance_bin == other.opposing_goalkeeper_distance_bin
            && self.forward_dribble_space_bin == other.forward_dribble_space_bin
            && self.team_spacing_score_bin == other.team_spacing_score_bin
            && self.preferred_team_spacing_bin == other.preferred_team_spacing_bin
            && self.perceived_time_on_ball_bin == other.perceived_time_on_ball_bin
            && self.fatigue_bin == other.fatigue_bin
            && self.nearest_defender_fatigue_bin == other.nearest_defender_fatigue_bin
            && self.nearest_defender_fatigue_confidence_bin
                == other.nearest_defender_fatigue_confidence_bin
            && self.perceived_fatigue_advantage_bin == other.perceived_fatigue_advantage_bin
            && self.pressure_bin == other.pressure_bin
            && self.perceived_pressure_bin == other.perceived_pressure_bin
            && self.first_touch_kind == other.first_touch_kind
            && self.incoming_ball_speed_bin == other.incoming_ball_speed_bin
            && self.receiving_pending_pass == other.receiving_pending_pass
            && self.pending_pass_off_target_bin == other.pending_pass_off_target_bin
            && self.pending_pass_receiver_urgency_bin == other.pending_pass_receiver_urgency_bin
            && self.first_time_shot_bin == other.first_time_shot_bin
            && self.control_touch_bin == other.control_touch_bin
            && self.skill_top_speed_bin == other.skill_top_speed_bin
            && self.skill_acceleration_bin == other.skill_acceleration_bin
            && self.skill_stamina_bin == other.skill_stamina_bin
            && self.skill_strength_bin == other.skill_strength_bin
            && self.skill_height_bin == other.skill_height_bin
            && self.skill_dribbling_bin == other.skill_dribbling_bin
            && self.skill_aggression_bin == other.skill_aggression_bin
            && self.skill_defending_bin == other.skill_defending_bin
            && self.skill_right_foot_shot_bin == other.skill_right_foot_shot_bin
            && self.skill_left_foot_shot_bin == other.skill_left_foot_shot_bin
            && self.skill_passing_completion_bin == other.skill_passing_completion_bin
            && self.skill_flair_passing_bin == other.skill_flair_passing_bin
            && self.skill_crossing_bin == other.skill_crossing_bin
            && self.skill_crossing_left_bin == other.skill_crossing_left_bin
            && self.skill_crossing_right_bin == other.skill_crossing_right_bin
            && self.skill_goalkeeping_bin == other.skill_goalkeeping_bin
            && self.skill_defensive_tracking_bin == other.skill_defensive_tracking_bin
            && self.open_space_bin == other.open_space_bin
            && facing_bucket_matches(self.receive_facing, other.receive_facing)
            && facing_bucket_matches(self.action_facing, other.action_facing)
    }

    fn matches_spatial_level(&self, other: &Self, level: PitchGridLevel) -> bool {
        self.matches_learning_context(other)
            && match level {
                PitchGridLevel::Fine => self.player_fine_cell_id == other.player_fine_cell_id,
                PitchGridLevel::Tactical => {
                    self.player_tactical_cell_id == other.player_tactical_cell_id
                }
                PitchGridLevel::Macro => self.player_macro_cell_id == other.player_macro_cell_id,
                PitchGridLevel::WholePitch => self.player_root_cell_id == other.player_root_cell_id,
            }
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

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerQTargetKey {
    pub state: SoccerQStateKey,
    pub action: String,
    pub target_fine_cell_id: usize,
    pub target_tactical_cell_id: usize,
    pub target_macro_cell_id: usize,
    pub target_root_cell_id: usize,
}

impl SoccerQTargetKey {
    fn from_state_action_grid(
        state: SoccerQStateKey,
        action: &str,
        grid: PitchGridAddress,
    ) -> Self {
        SoccerQTargetKey {
            state,
            action: normalize_soccer_action_label(action).to_string(),
            target_fine_cell_id: grid.fine.id,
            target_tactical_cell_id: grid.tactical.id,
            target_macro_cell_id: grid.macro_zone.id,
            target_root_cell_id: grid.whole_pitch.id,
        }
    }

    fn target_matches_grid(&self, grid: &PitchGridAddress, level: PitchGridLevel) -> bool {
        match level {
            PitchGridLevel::Fine => self.target_fine_cell_id == grid.fine.id,
            PitchGridLevel::Tactical => self.target_tactical_cell_id == grid.tactical.id,
            PitchGridLevel::Macro => self.target_macro_cell_id == grid.macro_zone.id,
            PitchGridLevel::WholePitch => self.target_root_cell_id == grid.whole_pitch.id,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerQTargetEntry {
    pub state: SoccerQStateKey,
    pub action: String,
    pub target_fine_cell_id: usize,
    pub target_tactical_cell_id: usize,
    pub target_macro_cell_id: usize,
    pub target_root_cell_id: usize,
    pub value: f64,
    pub visits: u32,
}

#[derive(Clone, Debug)]
pub struct SoccerQPolicy {
    pub q_values: HashMap<SoccerQActionKey, f64>,
    pub visits: HashMap<SoccerQActionKey, u32>,
    pub target_values: HashMap<SoccerQTargetKey, f64>,
    pub target_visits: HashMap<SoccerQTargetKey, u32>,
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
            target_values: HashMap::new(),
            target_visits: HashMap::new(),
            options,
        }
    }

    pub fn from_entries(
        options: SoccerQPolicyOptions,
        entries: &[SoccerQEntry],
    ) -> Result<Self, String> {
        Self::from_entries_with_targets(options, entries, &[])
    }

    pub fn from_entries_with_targets(
        options: SoccerQPolicyOptions,
        entries: &[SoccerQEntry],
        target_entries: &[SoccerQTargetEntry],
    ) -> Result<Self, String> {
        validate_soccer_q_policy_options(&options)?;
        let mut policy = SoccerQPolicy::new(options);
        for entry in entries {
            if !entry.value.is_finite() {
                return Err("policy entry value must be finite".to_string());
            }
            let action = normalize_soccer_action_label(&entry.action).to_string();
            if action.trim().is_empty() {
                return Err("policy entry action must not be empty".to_string());
            }
            let key = SoccerQActionKey {
                state: entry.state.clone(),
                action,
            };
            policy.q_values.insert(key.clone(), entry.value);
            policy.visits.insert(key, entry.visits);
        }
        for entry in target_entries {
            if !entry.value.is_finite() {
                return Err("policy target entry value must be finite".to_string());
            }
            let action = normalize_soccer_action_label(&entry.action).to_string();
            if action.trim().is_empty() {
                return Err("policy target entry action must not be empty".to_string());
            }
            let key = SoccerQTargetKey {
                state: entry.state.clone(),
                action,
                target_fine_cell_id: entry.target_fine_cell_id,
                target_tactical_cell_id: entry.target_tactical_cell_id,
                target_macro_cell_id: entry.target_macro_cell_id,
                target_root_cell_id: entry.target_root_cell_id,
            };
            policy.target_values.insert(key.clone(), entry.value);
            policy.target_visits.insert(key, entry.visits);
        }
        Ok(policy)
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
        let key = SoccerQActionKey {
            state: state.clone(),
            action: action.clone(),
        };
        let old = self.q_values.get(&key).copied().unwrap_or(0.0);
        let max_next = if transition.done {
            0.0
        } else {
            self.best_value_hierarchical(&next_state).unwrap_or(0.0)
        };
        let alpha = self.options.alpha.clamp(0.0, 1.0);
        let gamma = self.options.gamma.clamp(0.0, 0.999);
        let target = transition.reward + gamma * max_next;
        let updated = old + alpha * (target - old);
        self.q_values.insert(key.clone(), updated);
        *self.visits.entry(key).or_insert(0) += 1;

        if let Some(grid) = transition
            .action_target
            .as_ref()
            .and_then(|target| target.grid)
        {
            let target_key = SoccerQTargetKey::from_state_action_grid(state, &action, grid);
            let old_target = self.target_values.get(&target_key).copied().unwrap_or(0.0);
            let updated_target = old_target + alpha * (target - old_target);
            self.target_values
                .insert(target_key.clone(), updated_target);
            *self.target_visits.entry(target_key).or_insert(0) += 1;
        }
    }

    pub fn best_action(&self, state: &SoccerQStateKey) -> Option<String> {
        self.best_action_filtered(state, |_| true)
    }

    pub fn best_action_hierarchical(&self, state: &SoccerQStateKey) -> Option<String> {
        self.best_action_filtered_hierarchical(state, |_| true)
    }

    pub fn best_action_for_snapshot(
        &self,
        snapshot: &WorldSnapshot,
        player_id: usize,
    ) -> Option<String> {
        let player = snapshot.players.iter().find(|p| p.id == player_id)?;
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(player_id),
            &snapshot.observation_for(player_id),
            player.team,
            player.role,
        );
        self.best_action_filtered_hierarchical(&state, |action| {
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

    pub fn best_value_hierarchical(&self, state: &SoccerQStateKey) -> Option<f64> {
        PITCH_GRID_BACKOFF_LEVELS
            .iter()
            .find_map(|level| self.best_value_at_spatial_level(state, *level))
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
            &snapshot.mdp_state_for_player(player_id),
            &snapshot.observation_for(player_id),
            player.team,
            player.role,
        );
        self.set_action_value(state, action, value);
        true
    }

    pub fn set_target_value(
        &mut self,
        state: SoccerQStateKey,
        action: &str,
        grid: PitchGridAddress,
        value: f64,
    ) {
        let key = SoccerQTargetKey::from_state_action_grid(state, action, grid);
        self.target_values.insert(key.clone(), value);
        self.target_visits.entry(key).or_insert(1);
    }

    pub fn set_target_value_for_snapshot(
        &mut self,
        snapshot: &WorldSnapshot,
        player_id: usize,
        action: &str,
        point: Vec2,
        value: f64,
    ) -> bool {
        let Some(player) = snapshot.players.iter().find(|p| p.id == player_id) else {
            return false;
        };
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(player_id),
            &snapshot.observation_for(player_id),
            player.team,
            player.role,
        );
        self.set_target_value(
            state,
            action,
            pitch_grid_address(point, snapshot.field_width, snapshot.field_length),
            value,
        );
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

    pub fn target_entries(&self) -> Vec<SoccerQTargetEntry> {
        let mut entries = self
            .target_values
            .iter()
            .map(|(key, value)| SoccerQTargetEntry {
                state: key.state.clone(),
                action: key.action.clone(),
                target_fine_cell_id: key.target_fine_cell_id,
                target_tactical_cell_id: key.target_tactical_cell_id,
                target_macro_cell_id: key.target_macro_cell_id,
                target_root_cell_id: key.target_root_cell_id,
                value: *value,
                visits: self.target_visits.get(key).copied().unwrap_or(0),
            })
            .collect::<Vec<_>>();
        entries.sort_by(|a, b| {
            a.action
                .cmp(&b.action)
                .then_with(|| a.target_root_cell_id.cmp(&b.target_root_cell_id))
                .then_with(|| a.target_macro_cell_id.cmp(&b.target_macro_cell_id))
                .then_with(|| a.target_tactical_cell_id.cmp(&b.target_tactical_cell_id))
                .then_with(|| a.target_fine_cell_id.cmp(&b.target_fine_cell_id))
                .then_with(|| a.visits.cmp(&b.visits))
                .then_with(|| {
                    a.value
                        .partial_cmp(&b.value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });
        entries
    }

    pub fn visit_count(&self) -> u64 {
        self.visits.values().map(|v| u64::from(*v)).sum()
    }

    pub fn target_visit_count(&self) -> u64 {
        self.target_visits.values().map(|v| u64::from(*v)).sum()
    }

    pub fn best_target_grid_for_state_action(
        &self,
        state: &SoccerQStateKey,
        action: &str,
    ) -> Option<SoccerQTargetEntry> {
        let action = normalize_soccer_action_label(action);
        PITCH_GRID_BACKOFF_LEVELS.iter().find_map(|level| {
            self.target_values
                .iter()
                .filter(|(key, _)| {
                    key.action == action && key.state.matches_spatial_level(state, *level)
                })
                .max_by(|(a_key, a_value), (b_key, b_value)| {
                    a_value
                        .partial_cmp(b_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| {
                            self.target_visits
                                .get(a_key)
                                .copied()
                                .unwrap_or(0)
                                .cmp(&self.target_visits.get(b_key).copied().unwrap_or(0))
                        })
                })
                .map(|(key, value)| SoccerQTargetEntry {
                    state: key.state.clone(),
                    action: key.action.clone(),
                    target_fine_cell_id: key.target_fine_cell_id,
                    target_tactical_cell_id: key.target_tactical_cell_id,
                    target_macro_cell_id: key.target_macro_cell_id,
                    target_root_cell_id: key.target_root_cell_id,
                    value: *value,
                    visits: self.target_visits.get(key).copied().unwrap_or(0),
                })
        })
    }

    pub fn target_preference_for_point(
        &self,
        state: &SoccerQStateKey,
        action: &str,
        point: Vec2,
        field_width: f64,
        field_length: f64,
    ) -> f64 {
        let grid = pitch_grid_address(point, field_width, field_length);
        self.target_preference_for_grid(state, action, &grid)
            .unwrap_or(0.0)
    }

    pub fn target_preference_for_snapshot(
        &self,
        snapshot: &WorldSnapshot,
        player_id: usize,
        action: &str,
        point: Vec2,
    ) -> f64 {
        let Some(player) = snapshot.players.iter().find(|p| p.id == player_id) else {
            return 0.0;
        };
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(player_id),
            &snapshot.observation_for(player_id),
            player.team,
            player.role,
        );
        self.target_preference_for_point(
            &state,
            action,
            point,
            snapshot.field_width,
            snapshot.field_length,
        )
    }

    pub fn best_target_player_for_snapshot(
        &self,
        snapshot: &WorldSnapshot,
        player_id: usize,
        action: &str,
        candidates: &[usize],
    ) -> Option<usize> {
        let player = snapshot.players.iter().find(|p| p.id == player_id)?;
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(player_id),
            &snapshot.observation_for(player_id),
            player.team,
            player.role,
        );
        candidates
            .iter()
            .filter_map(|candidate_id| {
                let position = snapshot.player_position(*candidate_id)?;
                let grid =
                    pitch_grid_address(position, snapshot.field_width, snapshot.field_length);
                let preference = self.target_preference_for_grid(&state, action, &grid)?;
                Some((*candidate_id, preference))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(candidate_id, _)| candidate_id)
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

    fn best_action_filtered_hierarchical<F>(
        &self,
        state: &SoccerQStateKey,
        is_legal: F,
    ) -> Option<String>
    where
        F: Fn(&str) -> bool,
    {
        PITCH_GRID_BACKOFF_LEVELS
            .iter()
            .find_map(|level| self.best_action_filtered_at_spatial_level(state, *level, &is_legal))
    }

    fn best_action_filtered_at_spatial_level<F>(
        &self,
        state: &SoccerQStateKey,
        level: PitchGridLevel,
        is_legal: &F,
    ) -> Option<String>
    where
        F: Fn(&str) -> bool,
    {
        self.q_values
            .iter()
            .filter(|(key, _)| key.state.matches_spatial_level(state, level))
            .filter(|(key, _)| is_legal(&key.action))
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

    fn best_value_at_spatial_level(
        &self,
        state: &SoccerQStateKey,
        level: PitchGridLevel,
    ) -> Option<f64> {
        self.q_values
            .iter()
            .filter(|(key, _)| key.state.matches_spatial_level(state, level))
            .map(|(_, value)| *value)
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn target_preference_for_grid(
        &self,
        state: &SoccerQStateKey,
        action: &str,
        grid: &PitchGridAddress,
    ) -> Option<f64> {
        let action = normalize_soccer_action_label(action);
        for source_level in PITCH_GRID_BACKOFF_LEVELS {
            let best = self
                .target_values
                .iter()
                .filter(|(key, _)| {
                    key.action == action && key.state.matches_spatial_level(state, source_level)
                })
                .filter_map(|(key, value)| {
                    let weight = if key.target_matches_grid(grid, PitchGridLevel::Fine) {
                        1.0
                    } else if key.target_matches_grid(grid, PitchGridLevel::Tactical) {
                        0.75
                    } else if key.target_matches_grid(grid, PitchGridLevel::Macro) {
                        0.45
                    } else if key.target_matches_grid(grid, PitchGridLevel::WholePitch) {
                        0.15
                    } else {
                        return None;
                    };
                    let visits = self.target_visits.get(key).copied().unwrap_or(0);
                    Some(((*value * weight).clamp(-5.0, 5.0), visits))
                })
                .max_by(|(a_value, a_visits), (b_value, b_visits)| {
                    a_value
                        .partial_cmp(b_value)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then_with(|| a_visits.cmp(b_visits))
                });
            if let Some((value, _)) = best {
                return Some(value);
            }
        }
        None
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

fn validate_soccer_q_policy_options(options: &SoccerQPolicyOptions) -> Result<(), String> {
    if !options.alpha.is_finite() || !(0.0..=1.0).contains(&options.alpha) {
        return Err("policy alpha must be finite and between 0.0 and 1.0".to_string());
    }
    if !options.gamma.is_finite() || !(0.0..=0.999).contains(&options.gamma) {
        return Err("policy gamma must be finite and between 0.0 and 0.999".to_string());
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct SoccerTeamQPolicies {
    pub home: SoccerQPolicy,
    pub away: SoccerQPolicy,
}

impl SoccerTeamQPolicies {
    pub fn new(options: SoccerQPolicyOptions) -> Self {
        SoccerTeamQPolicies {
            home: SoccerQPolicy::new(options.clone()),
            away: SoccerQPolicy::new(options),
        }
    }

    pub fn from_artifact(artifact: &SoccerTeamPolicyArtifact) -> Result<Self, String> {
        let home_options = artifact.home_options.clone().unwrap_or_default();
        let away_options = artifact.away_options.clone().unwrap_or_default();
        Ok(SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries_with_targets(
                home_options,
                &artifact.home_entries,
                &artifact.home_target_entries,
            )?,
            away: SoccerQPolicy::from_entries_with_targets(
                away_options,
                &artifact.away_entries,
                &artifact.away_target_entries,
            )?,
        })
    }

    pub fn from_self_play_artifact(
        artifact: &SoccerSelfPlayTrainingArtifact,
    ) -> Result<Self, String> {
        Ok(SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries_with_targets(
                artifact.options.clone(),
                &artifact.home_entries,
                &artifact.home_target_entries,
            )?,
            away: SoccerQPolicy::from_entries_with_targets(
                artifact.options.clone(),
                &artifact.away_entries,
                &artifact.away_target_entries,
            )?,
        })
    }

    pub fn policy(&self, team: Team) -> &SoccerQPolicy {
        match team {
            Team::Home => &self.home,
            Team::Away => &self.away,
        }
    }

    pub fn policy_mut(&mut self, team: Team) -> &mut SoccerQPolicy {
        match team {
            Team::Home => &mut self.home,
            Team::Away => &mut self.away,
        }
    }

    pub fn train(&mut self, transitions: &[SoccerLearningTransition]) {
        for transition in transitions {
            self.policy_mut(transition.team).update(transition);
        }
    }

    pub fn train_adversarial(&mut self, transitions: &[SoccerLearningTransition]) {
        let mut tick_rewards: BTreeMap<u64, (f64, u32, f64, u32)> = BTreeMap::new();
        for transition in transitions {
            let entry = tick_rewards.entry(transition.tick).or_default();
            match transition.team {
                Team::Home => {
                    entry.0 += transition.reward;
                    entry.1 += 1;
                }
                Team::Away => {
                    entry.2 += transition.reward;
                    entry.3 += 1;
                }
            }
        }

        for transition in transitions {
            let Some((home_sum, home_count, away_sum, away_count)) =
                tick_rewards.get(&transition.tick).copied()
            else {
                self.policy_mut(transition.team).update(transition);
                continue;
            };
            let home_avg = if home_count == 0 {
                0.0
            } else {
                home_sum / f64::from(home_count)
            };
            let away_avg = if away_count == 0 {
                0.0
            } else {
                away_sum / f64::from(away_count)
            };
            let opponent_avg = match transition.team {
                Team::Home => away_avg,
                Team::Away => home_avg,
            };
            let mut adversarial_transition = transition.clone();
            adversarial_transition.reward = transition.reward - opponent_avg;
            self.policy_mut(transition.team)
                .update(&adversarial_transition);
        }
    }

    pub fn total_entries(&self) -> usize {
        self.home.q_values.len() + self.away.q_values.len()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerSelfPlayEpisodeSummary {
    pub episode: usize,
    pub seed: u64,
    pub summary: MatchSummary,
    pub transitions: usize,
    pub home_policy_entries: usize,
    #[serde(default)]
    pub home_policy_target_entries: usize,
    pub away_policy_entries: usize,
    #[serde(default)]
    pub away_policy_target_entries: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerSelfPlayTrainingArtifact {
    pub config: MatchConfig,
    pub options: SoccerQPolicyOptions,
    #[serde(default = "default_tactical_learning_weights")]
    pub tactical_learning: SoccerTacticalLearningWeights,
    pub episodes: Vec<SoccerSelfPlayEpisodeSummary>,
    pub home_entries: Vec<SoccerQEntry>,
    #[serde(default)]
    pub home_target_entries: Vec<SoccerQTargetEntry>,
    pub away_entries: Vec<SoccerQEntry>,
    #[serde(default)]
    pub away_target_entries: Vec<SoccerQTargetEntry>,
}

fn normalize_soccer_action_label(action: &str) -> &str {
    match action {
        "move" => "space",
        "pass1" | "pass2" | "pass3" => "pass",
        "aerial-pass1" | "aerial-pass2" | "aerial-pass3" => "aerial-pass",
        "header" => "first-time-header",
        "chest-control" => "control-touch",
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
    pub jerk: Vec2,
    #[serde(default)]
    pub movement_gait: MovementGait,
    pub position_history: VecDeque<Vec2>,
    #[serde(default)]
    pub receive_facing: FacingBucket,
    #[serde(default)]
    pub action_facing: FacingBucket,
    #[serde(default)]
    pub incoming_ball: Option<IncomingBallContext>,
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

    pub fn history_jerk_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 4 {
            return self.jerk;
        }
        let last = self.position_history.len() - 1;
        let v0 = (self.position_history[last - 2] - self.position_history[last - 3]) / dt_seconds;
        let v1 = (self.position_history[last - 1] - self.position_history[last - 2]) / dt_seconds;
        let v2 = (self.position_history[last] - self.position_history[last - 1]) / dt_seconds;
        let a0 = (v1 - v0) / dt_seconds;
        let a1 = (v2 - v1) / dt_seconds;
        (a1 - a0) / dt_seconds
    }

    fn possession_action_options(
        &self,
        observation: &SoccerPomdpObservation,
        directive: &TeamTacticalDirective,
        pass_target_count: usize,
        aerial_pass_target_count: usize,
    ) -> Vec<AgentActionOptionTrace> {
        let shooting = ability01(self.skills.shooting);
        let dribbling = ability01(self.skills.dribbling);
        let passing = ability01(self.skills.passing_completion_rate);
        let crossing = ability01(self.skills.crossing_left.max(self.skills.crossing_right));
        let shot_legal = shot_decision_is_qualified(observation);
        let shot_quality_weight = (observation.shot_on_frame_probability * 0.72
            + observation.shot_beat_goalkeeper_probability * 0.48
            + observation.shot_curl_probability * 0.12)
            .clamp(0.0, 1.25);
        let shot_block_penalty =
            (1.0 - observation.shot_block_probability.clamp(0.0, 1.0) * 0.58).clamp(0.30, 1.0);
        let shot_score = (self.preferences.shoot_bias
            * (0.52 + shooting * 0.62)
            * (1.0 + directive.risk_tolerance * 0.35)
            * (0.78 + (observation.opponent_goal_angle_degrees / 42.0).clamp(0.0, 1.0) * 0.44)
            * (0.34 + shot_quality_weight)
            * shot_block_penalty
            * 0.042)
            .clamp(0.004, 0.12);
        let fatigue_dribble = fatigue_dribble_multiplier(observation);
        let shot_creation_carry = shot_creation_carry_multiplier(observation);
        let dribble_score = (self.preferences.dribble_bias
            * (0.62 + dribbling * 0.48)
            * directive.carry_priority
            * (0.70 + (observation.forward_dribble_space_yards / 18.0).clamp(0.0, 1.0) * 0.58)
            * fatigue_dribble
            * shot_creation_carry)
            .clamp(0.02, 0.94);
        let mut options = vec![
            AgentActionOptionTrace::new("shoot", shot_score, shot_legal),
            AgentActionOptionTrace::new("dribble", dribble_score, true),
        ];
        for rank in 0..pass_target_count.min(3) {
            let rank_weight = match rank {
                0 => 1.00,
                1 => 0.74,
                _ => 0.56,
            };
            let quick_release = (1.35 - observation.perceived_time_on_ball_seconds)
                .max(0.0)
                .min(1.0);
            let pass_score = (self.preferences.pass_bias
                * directive.pass_priority
                * (0.70 + passing * 0.42)
                * (1.0 + quick_release * 0.22)
                * (1.0 + observation.pass_curl_probability * 0.055)
                * rank_weight)
                .clamp(0.04, 0.97);
            options.push(AgentActionOptionTrace::new(
                format!("pass{}", rank + 1),
                pass_score,
                true,
            ));
        }
        for rank in 0..aerial_pass_target_count.min(3) {
            let rank_weight = match rank {
                0 => 0.82,
                1 => 0.62,
                _ => 0.48,
            };
            let bypass_bonus = (observation.perceived_pressure * 0.20
                + (1.0 - observation.forward_dribble_space_yards / 16.0).clamp(0.0, 1.0) * 0.16
                + observation.aerial_pass_bypass_score * 0.34)
                .clamp(0.0, 0.42);
            let interception_penalty =
                (1.0 - observation.aerial_pass_interception_risk * 0.38).clamp(0.58, 1.0);
            let aerial_score = (self.preferences.pass_bias
                * directive.pass_priority
                * (0.48
                    + passing * 0.26
                    + crossing * 0.22
                    + ability01(self.skills.flair_passing) * 0.10)
                * (1.0 + bypass_bonus)
                * interception_penalty
                * (1.0 + observation.pass_curl_probability * 0.075)
                * rank_weight)
                .clamp(0.02, 0.74);
            options.push(AgentActionOptionTrace::new(
                format!("aerial-pass{}", rank + 1),
                aerial_score,
                true,
            ));
        }
        normalize_action_options(options)
    }

    fn first_touch_action_options(
        &self,
        observation: &SoccerPomdpObservation,
        pass_target_count: usize,
    ) -> Vec<AgentActionOptionTrace> {
        let is_aerial = matches!(
            observation.incoming_ball_kind,
            IncomingBallKind::AerialCross | IncomingBallKind::AerialPass
        );
        let shot_label = if is_aerial {
            "header"
        } else {
            "first-time-shot"
        };
        let control_label = if is_aerial {
            "chest-control"
        } else {
            "control-touch"
        };
        let shot_legal = first_time_shot_decision_is_qualified(observation);
        let pass_legal = pass_target_count > 0;
        let quick_pressure_bonus = 1.0 + observation.perceived_pressure.clamp(0.0, 1.0) * 0.24;
        normalize_action_options(vec![
            AgentActionOptionTrace::new(
                shot_label,
                (observation.first_time_shot_score * quick_pressure_bonus * 0.52).clamp(0.01, 0.64),
                shot_legal,
            ),
            AgentActionOptionTrace::new(
                "first-time-pass",
                (observation.first_time_pass_score * quick_pressure_bonus * 0.86).clamp(0.02, 0.88),
                pass_legal,
            ),
            AgentActionOptionTrace::new(
                control_label,
                (observation.control_touch_score * (1.12 - observation.perceived_pressure * 0.18))
                    .clamp(0.02, 0.95),
                true,
            ),
        ])
    }

    fn support_action_options(&self) -> Vec<AgentActionOptionTrace> {
        normalize_action_options(vec![
            AgentActionOptionTrace::new(
                "support-shape",
                0.90 * self.preferences.open_space_bias,
                true,
            ),
            AgentActionOptionTrace::new(
                "support-roam",
                0.10 * self.preferences.open_space_bias,
                true,
            ),
        ])
    }

    fn defensive_action_options(
        &self,
        snapshot: &WorldSnapshot,
        directive: &TeamTacticalDirective,
    ) -> Vec<AgentActionOptionTrace> {
        let tackle_legal = snapshot.ball.holder.is_some_and(|holder| {
            snapshot
                .players
                .iter()
                .find(|p| p.id == holder)
                .is_some_and(|p| {
                    p.team == self.team.other() && self.position.distance(p.position) < 3.1
                })
        });
        let defending = ability01(self.skills.defending);
        let aggression = ability01(self.skills.aggression);
        let tackle_score =
            ((defending * 0.6 + aggression * 0.4) * directive.press_intensity).clamp(0.02, 0.92);
        normalize_action_options(vec![
            AgentActionOptionTrace::new("tackle", tackle_score, tackle_legal),
            AgentActionOptionTrace::new("defend-shape", 0.90, true),
            AgentActionOptionTrace::new("defend-roam", 0.10, true),
        ])
    }

    fn decision_trace(
        &self,
        snapshot: &WorldSnapshot,
        mdp_state: SoccerMdpState,
        observation: SoccerPomdpObservation,
        belief: BeliefState,
        operation_order: Vec<String>,
        action_options: Vec<AgentActionOptionTrace>,
        action: &SoccerAction,
        action_label: impl Into<String>,
    ) -> AgentDecisionTrace {
        AgentDecisionTrace {
            mdp_state,
            observation,
            belief,
            operation_order,
            action_options,
            action_target: self.action_target_trace(action, snapshot),
            action: action_label.into(),
        }
    }

    fn action_target_trace(
        &self,
        action: &SoccerAction,
        snapshot: &WorldSnapshot,
    ) -> Option<AgentActionTargetTrace> {
        let (point, player_id) = match action {
            SoccerAction::HoldShape => (self.home_position, None),
            SoccerAction::MoveTo(target)
            | SoccerAction::Dribble(target)
            | SoccerAction::ControlTouch { target } => (*target, None),
            SoccerAction::Pass {
                target_player,
                flight,
                ..
            } => {
                let resolved_target = target_player
                    .or_else(|| {
                        if flight.is_aerial() {
                            snapshot.best_aerial_pass_target(self.id)
                        } else {
                            snapshot.best_visible_pass_target(self.id)
                        }
                    })
                    .or_else(|| snapshot.best_pass_target(self.id));
                let point = resolved_target
                    .and_then(|id| snapshot.player_position(id))
                    .unwrap_or_else(|| {
                        Vec2::new(
                            self.position.x,
                            self.position.y + 18.0 * self.team.attack_dir(),
                        )
                        .clamp_to_pitch(snapshot.field_width, snapshot.field_length)
                    });
                (point, resolved_target)
            }
            SoccerAction::Shoot { .. } => (
                Vec2::new(
                    snapshot.field_width * 0.5,
                    self.team.goal_y(snapshot.field_length),
                ),
                None,
            ),
            SoccerAction::Tackle { target_player } => {
                let point = snapshot
                    .player_position(*target_player)
                    .unwrap_or(snapshot.ball.position);
                (point, Some(*target_player))
            }
        };
        let point = point.clamp_to_pitch(snapshot.field_width, snapshot.field_length);
        let facing = facing_bucket_from_vector(point - self.position);
        Some(AgentActionTargetTrace {
            point: Some(point),
            player_id,
            grid: Some(pitch_grid_address(
                point,
                snapshot.field_width,
                snapshot.field_length,
            )),
            facing,
        })
    }

    pub fn run_time_step(
        &mut self,
        snapshot: &WorldSnapshot,
        human_input: Option<&HumanInputFrame>,
        learned_plan: Option<&SoccerLearnedPlan>,
        rng: &mut SeededRandom,
    ) -> PlayerIntent {
        let mdp_state = snapshot.mdp_state_for_player(self.id);
        let observation = snapshot.observation_for(self.id);
        let belief = belief_from_observation(&observation);
        let directive = snapshot.tactical_directive(self.team);
        let has_ball = observation.has_ball;
        let shooting_skill = ability01(self.skills.shooting);
        let passing_skill = ability01(self.skills.passing_completion_rate);
        let dribbling_skill = ability01(self.skills.dribbling);
        let defending_skill = ability01(self.skills.defending);
        let aggression_skill = ability01(self.skills.aggression);

        if let Some(input) = human_input {
            let (action, action_label) = if input.shoot {
                (SoccerAction::Shoot { power: 1.0 }, "shoot")
            } else if input.pass {
                let pass_flight = input.pass_flight;
                (
                    SoccerAction::Pass {
                        target_player: input.target_player,
                        power: 0.78,
                        flight: pass_flight,
                    },
                    if pass_flight.is_aerial() {
                        "aerial-pass"
                    } else {
                        "pass"
                    },
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
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                vec!["human-input".to_string()],
                single_action_option(action_label),
                &action,
                action_label,
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: input.sprint,
            };
        }

        if has_ball && observation.first_touch_available {
            let pass_targets = snapshot.ranked_visible_pass_targets(self.id, 3);
            let action_options = self.first_touch_action_options(&observation, pass_targets.len());
            let weighted_ops = action_options
                .iter()
                .map(|option| {
                    (
                        option.label.clone(),
                        if option.legal { option.score } else { 0.0 },
                    )
                })
                .collect::<Vec<_>>();
            let ops = weighted_fisher_yates_order(weighted_ops, rng);
            let mut order_names = Vec::with_capacity(ops.len());
            let mut chosen = None;
            for op in ops {
                order_names.push(op.clone());
                match normalize_soccer_action_label(&op) {
                    "first-time-shot" | "first-time-header"
                        if first_time_shot_decision_is_qualified(&observation) =>
                    {
                        let finish_skill = if matches!(
                            observation.incoming_ball_kind,
                            IncomingBallKind::AerialCross | IncomingBallKind::AerialPass
                        ) {
                            aerial_duel_skill_from_agent(self)
                        } else {
                            ability01(
                                self.skills
                                    .right_foot_shot_power
                                    .max(self.skills.left_foot_shot_power),
                            )
                        };
                        chosen = Some((
                            SoccerAction::Shoot {
                                power: 0.68 + 0.28 * finish_skill,
                            },
                            normalize_soccer_action_label(&op).to_string(),
                        ));
                        break;
                    }
                    "first-time-pass" if !pass_targets.is_empty() => {
                        let target = pass_targets[0];
                        chosen = Some((
                            SoccerAction::Pass {
                                target_player: Some(target),
                                power: 0.54 + 0.30 * passing_skill,
                                flight: PassFlight::Floor,
                            },
                            "first-time-pass".to_string(),
                        ));
                        break;
                    }
                    "control-touch" => {
                        let target = (self.position + carried_ball_lead(self))
                            .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
                        chosen = Some((
                            SoccerAction::ControlTouch { target },
                            normalize_soccer_action_label(&op).to_string(),
                        ));
                        break;
                    }
                    _ => {}
                }
            }
            let (action, action_label) = chosen.unwrap_or_else(|| {
                let target = (self.position + carried_ball_lead(self))
                    .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
                (
                    SoccerAction::ControlTouch { target },
                    "control-touch".to_string(),
                )
            });
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                order_names,
                action_options,
                &action,
                action_label,
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: false,
            };
        }

        if has_ball && shot_decision_is_qualified(&observation) {
            let finish_chance = (0.018
                + shooting_skill * 0.050
                + observation.shot_on_frame_probability * 0.060
                + observation.shot_beat_goalkeeper_probability * 0.034
                + observation.shot_curl_probability * 0.012
                - observation.shot_block_probability * 0.030
                - self.skills.decision_noise * 0.12)
                .clamp(0.012, 0.12);
            if rng.next_float() < time_window_probability(finish_chance, snapshot.dt_seconds) {
                let action = SoccerAction::Shoot {
                    power: 0.72 + 0.28 * shooting_skill,
                };
                let action_label = action.label();
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["finish".to_string()],
                    single_action_option("shoot"),
                    &action,
                    action_label,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }
        }

        if has_ball {
            if let Some(target) = snapshot.long_ball_in_behind_target(self.id) {
                let crossing = ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                let long_ball_chance = (0.10
                    + passing_skill * 0.07
                    + crossing * 0.05
                    + directive.risk_tolerance * 0.03)
                    .clamp(0.05, 0.22);
                if rng.next_float() < time_window_probability(long_ball_chance, snapshot.dt_seconds)
                {
                    let action = SoccerAction::Pass {
                        target_player: Some(target),
                        power: 0.80 + 0.14 * crossing.max(passing_skill),
                        flight: PassFlight::Aerial,
                    };
                    self.last_decision = Some(self.decision_trace(
                        snapshot,
                        mdp_state,
                        observation,
                        belief,
                        vec!["long-ball-in-behind".to_string()],
                        single_action_option("aerial-pass"),
                        &action,
                        "aerial-pass",
                    ));
                    return PlayerIntent {
                        player_id: self.id,
                        action,
                        sprint: false,
                    };
                }
            }
        }

        if !has_ball {
            if let Some((target, sprint)) = snapshot.pending_pass_reception_target_for(self.id) {
                let action = SoccerAction::MoveTo(target);
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["receive-pending-pass".to_string()],
                    single_action_option("recover"),
                    &action,
                    "recover",
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint,
                };
            }
        }

        if let Some(plan) = learned_plan {
            if let Some((action, action_label)) =
                self.action_from_learned_plan(plan, snapshot, &observation)
            {
                if matches!(action, SoccerAction::Shoot { .. }) {
                    let learned_shot_chance = (0.025
                        + shooting_skill * 0.050
                        + observation.shot_curl_probability * 0.010
                        - observation.shot_block_probability * 0.026
                        - self.skills.decision_noise * 0.10)
                        .clamp(0.008, 0.08);
                    if rng.next_float()
                        >= time_window_probability(learned_shot_chance, snapshot.dt_seconds)
                    {
                        let defer_action = SoccerAction::Dribble(
                            snapshot.shot_creation_space_for(self.id, self.home_position),
                        );
                        self.last_decision = Some(self.decision_trace(
                            snapshot,
                            mdp_state,
                            observation,
                            belief,
                            vec![
                                "learned-policy".to_string(),
                                plan.action.clone(),
                                "defer-shot".to_string(),
                            ],
                            single_action_option("dribble"),
                            &defer_action,
                            "dribble",
                        ));
                        return PlayerIntent {
                            player_id: self.id,
                            action: defer_action,
                            sprint: false,
                        };
                    }
                }
                self.last_decision = Some(self.decision_trace(
                    snapshot,
                    mdp_state,
                    observation,
                    belief,
                    vec!["learned-policy".to_string(), plan.action.clone()],
                    single_action_option(&action_label),
                    &action,
                    action_label,
                ));
                return PlayerIntent {
                    player_id: self.id,
                    action,
                    sprint: false,
                };
            }
        }

        if has_ball {
            let pass_targets = snapshot.ranked_visible_pass_targets(self.id, 3);
            let aerial_pass_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 3);
            let action_options = self.possession_action_options(
                &observation,
                &directive,
                pass_targets.len(),
                aerial_pass_targets.len(),
            );
            let mut weighted_ops = vec![
                (
                    "shoot".to_string(),
                    action_option_score(&action_options, "shoot"),
                ),
                (
                    "dribble".to_string(),
                    action_option_score(&action_options, "dribble"),
                ),
            ];
            for rank in 0..pass_targets.len() {
                let label = format!("pass{}", rank + 1);
                weighted_ops.push((label.clone(), action_option_score(&action_options, &label)));
            }
            for rank in 0..aerial_pass_targets.len() {
                let label = format!("aerial-pass{}", rank + 1);
                weighted_ops.push((label.clone(), action_option_score(&action_options, &label)));
            }
            let ops = weighted_fisher_yates_order(weighted_ops, rng);
            let mut order_names = Vec::with_capacity(ops.len());
            let mut chosen = None;
            for op in ops {
                match op.as_str() {
                    "shoot" => {
                        order_names.push("shoot".to_string());
                        let shot_quality_weight = (observation.shot_on_frame_probability * 0.72
                            + observation.shot_beat_goalkeeper_probability * 0.48
                            + observation.shot_curl_probability * 0.12)
                            .clamp(0.0, 1.25);
                        let shot_block_penalty =
                            (1.0 - observation.shot_block_probability * 0.58).clamp(0.30, 1.0);
                        let shot_chance = (self.preferences.shoot_bias
                            * (0.52 + shooting_skill * 0.62)
                            * (1.0 + directive.risk_tolerance * 0.35)
                            * (0.78
                                + (observation.opponent_goal_angle_degrees / 42.0)
                                    .clamp(0.0, 1.0)
                                    * 0.44)
                            * (0.34 + shot_quality_weight)
                            * shot_block_penalty
                            * 0.042)
                            .clamp(0.004, 0.12);
                        if shot_decision_is_qualified(&observation)
                            && rng.next_float()
                                < time_window_probability(shot_chance, snapshot.dt_seconds)
                        {
                            chosen = Some((
                                SoccerAction::Shoot {
                                    power: 0.72 + 0.28 * shooting_skill,
                                },
                                "shoot".to_string(),
                            ));
                            break;
                        }
                    }
                    "dribble" => {
                        order_names.push("dribble".to_string());
                        let dribble_chance = (self.preferences.dribble_bias
                            * (0.62 + dribbling_skill * 0.48)
                            * directive.carry_priority
                            * (0.70
                                + (observation.forward_dribble_space_yards / 18.0)
                                    .clamp(0.0, 1.0)
                                    * 0.58)
                            * fatigue_dribble_multiplier(&observation)
                            * shot_creation_carry_multiplier(&observation))
                        .clamp(0.02, 0.94);
                        if rng.next_float()
                            < time_window_probability(dribble_chance, snapshot.dt_seconds)
                        {
                            let target =
                                snapshot.shot_creation_space_for(self.id, self.home_position);
                            chosen = Some((SoccerAction::Dribble(target), "dribble".to_string()));
                            break;
                        }
                    }
                    pass_label if pass_label.starts_with("pass") => {
                        let rank = pass_label
                            .trim_start_matches("pass")
                            .parse::<usize>()
                            .ok()
                            .and_then(|n| n.checked_sub(1))
                            .unwrap_or(0);
                        if rank >= pass_targets.len() {
                            continue;
                        }
                        order_names.push(format!("pass{}", rank + 1));
                        let rank_weight = match rank {
                            0 => 1.00,
                            1 => 0.74,
                            _ => 0.56,
                        };
                        let pass_chance = (self.preferences.pass_bias
                            * directive.pass_priority
                            * (0.70 + passing_skill * 0.42)
                            * (1.0
                                + (1.35 - observation.perceived_time_on_ball_seconds)
                                    .max(0.0)
                                    .min(1.0)
                                    * 0.22)
                            * (1.0 + observation.pass_curl_probability * 0.055)
                            * rank_weight)
                            .clamp(0.04, 0.97);
                        if rng.next_float()
                            < time_window_probability(pass_chance, snapshot.dt_seconds)
                        {
                            let target = pass_targets[rank];
                            chosen = Some((
                                SoccerAction::Pass {
                                    target_player: Some(target),
                                    power: 0.58 + 0.32 * passing_skill,
                                    flight: PassFlight::Floor,
                                },
                                format!("pass{}", rank + 1),
                            ));
                            break;
                        }
                    }
                    pass_label if pass_label.starts_with("aerial-pass") => {
                        let rank = pass_label
                            .trim_start_matches("aerial-pass")
                            .parse::<usize>()
                            .ok()
                            .and_then(|n| n.checked_sub(1))
                            .unwrap_or(0);
                        if rank >= aerial_pass_targets.len() {
                            continue;
                        }
                        order_names.push(format!("aerial-pass{}", rank + 1));
                        let rank_weight = match rank {
                            0 => 0.82,
                            1 => 0.62,
                            _ => 0.48,
                        };
                        let crossing =
                            ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                        let bypass_boost =
                            (1.0 + observation.aerial_pass_bypass_score * 0.28).clamp(1.0, 1.28);
                        let risk_penalty = (1.0 - observation.aerial_pass_interception_risk * 0.36)
                            .clamp(0.60, 1.0);
                        let pass_chance = (self.preferences.pass_bias
                            * directive.pass_priority
                            * (0.46 + passing_skill * 0.28 + crossing * 0.22)
                            * bypass_boost
                            * risk_penalty
                            * (1.0 + observation.pass_curl_probability * 0.075)
                            * rank_weight)
                            .clamp(0.02, 0.76);
                        if rng.next_float()
                            < time_window_probability(pass_chance, snapshot.dt_seconds)
                        {
                            let target = aerial_pass_targets[rank];
                            chosen = Some((
                                SoccerAction::Pass {
                                    target_player: Some(target),
                                    power: 0.56 + 0.28 * crossing.max(passing_skill),
                                    flight: PassFlight::Aerial,
                                },
                                format!("aerial-pass{}", rank + 1),
                            ));
                            break;
                        }
                    }
                    _ => {}
                }
            }

            let (action, action_label) = chosen.unwrap_or_else(|| {
                let carry_to_create = shot_creation_carry_multiplier(&observation) > 1.28
                    && observation.forward_dribble_space_yards > 2.2;
                if carry_to_create || pass_targets.is_empty() {
                    (
                        SoccerAction::Dribble(
                            snapshot.shot_creation_space_for(self.id, self.home_position),
                        ),
                        "dribble".to_string(),
                    )
                } else if let Some(target) = pass_targets.first() {
                    (
                        SoccerAction::Pass {
                            target_player: Some(*target),
                            power: 0.58 + 0.32 * passing_skill,
                            flight: PassFlight::Floor,
                        },
                        "pass1".to_string(),
                    )
                } else {
                    (
                        SoccerAction::Dribble(
                            snapshot.shot_creation_space_for(self.id, self.home_position),
                        ),
                        "dribble".to_string(),
                    )
                }
            });
            self.last_decision = Some(self.decision_trace(
                snapshot,
                mdp_state,
                observation,
                belief,
                order_names,
                action_options,
                &action,
                action_label,
            ));
            return PlayerIntent {
                player_id: self.id,
                action,
                sprint: false,
            };
        }

        let possession_team = snapshot.controlled_possession_team();
        let mut order_names = Vec::new();
        let action_options;
        let (action, action_label) = if possession_team == Some(self.team) {
            action_options = self.support_action_options();
            let support_order = weighted_fisher_yates_order(
                vec![
                    (
                        "support-shape",
                        action_option_score(&action_options, "support-shape"),
                    ),
                    (
                        "support-roam",
                        action_option_score(&action_options, "support-roam"),
                    ),
                ],
                rng,
            );
            let roam = support_order
                .first()
                .is_some_and(|label| *label == "support-roam");
            order_names.extend(support_order.into_iter().map(str::to_string));
            (
                SoccerAction::MoveTo(snapshot.positional_open_space_for(
                    self.id,
                    self.home_position,
                    roam,
                )),
                "space".to_string(),
            )
        } else if possession_team == Some(self.team.other()) {
            action_options = self.defensive_action_options(snapshot, &directive);
            let ops = weighted_fisher_yates_order(
                vec![
                    ("tackle", action_option_score(&action_options, "tackle")),
                    (
                        "defend-shape",
                        action_option_score(&action_options, "defend-shape"),
                    ),
                    (
                        "defend-roam",
                        action_option_score(&action_options, "defend-roam"),
                    ),
                ],
                rng,
            );
            let mut chosen = None;
            for op in ops {
                match op {
                    "tackle" => {
                        order_names.push("tackle".to_string());
                        if let Some(holder) = snapshot.ball.holder {
                            let holder_is_opponent = snapshot
                                .players
                                .iter()
                                .find(|p| p.id == holder)
                                .is_some_and(|p| p.team == self.team.other());
                            if holder_is_opponent
                                && self.position.distance(snapshot.ball.position) < 3.1
                                && rng.next_float()
                                    < time_window_probability(
                                        ((defending_skill * 0.6 + aggression_skill * 0.4)
                                            * directive.press_intensity)
                                            .clamp(0.02, 0.92),
                                        snapshot.dt_seconds,
                                    )
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
                    "defend-shape" | "defend-roam" => {
                        let roam = op == "defend-roam";
                        order_names.push(op.to_string());
                        let dist = self.position.distance(snapshot.ball.position);
                        let defend_radius = 3.0 + directive.press_intensity * 3.0;
                        let target = if roam && dist < defend_radius {
                            snapshot.ball.position
                        } else {
                            snapshot.defensive_assignment_for(self.id, self.home_position, roam)
                        };
                        chosen = Some((SoccerAction::MoveTo(target), "defend".to_string()));
                        break;
                    }
                    _ => {}
                }
            }
            chosen.unwrap_or_else(|| {
                (
                    SoccerAction::MoveTo(snapshot.defensive_assignment_for(
                        self.id,
                        self.home_position,
                        false,
                    )),
                    "defend".to_string(),
                )
            })
        } else {
            let my_distance = self.position.distance(snapshot.ball.position);
            let fifty_fifty_duel = loose_ball_fifty_fifty_duel_for(snapshot, self.id);
            let closer_teammates = snapshot
                .players
                .iter()
                .filter(|player| player.team == self.team && player.id != self.id)
                .filter(|player| {
                    snapshot
                        .player_position(player.id)
                        .unwrap_or(player.position)
                        .distance(snapshot.ball.position)
                        < my_distance
                })
                .count();
            let goalkeeper_can_recover = self.role != PlayerRole::Goalkeeper || my_distance <= 12.0;
            if fifty_fifty_duel {
                action_options = single_action_option("fifty-fifty-duel");
                order_names.push("fifty-fifty-duel".to_string());
                (
                    SoccerAction::MoveTo(snapshot.ball.position),
                    "recover".to_string(),
                )
            } else if closer_teammates < 2 && my_distance <= 46.0 && goalkeeper_can_recover {
                action_options = single_action_option("recover");
                order_names.push("recover-loose-ball".to_string());
                (
                    SoccerAction::MoveTo(snapshot.ball.position),
                    "recover".to_string(),
                )
            } else {
                action_options = single_action_option("hold");
                order_names.push("hold-shape".to_string());
                (
                    SoccerAction::MoveTo(snapshot.defensive_shape_for(self.id, self.home_position)),
                    "hold".to_string(),
                )
            }
        };

        self.last_decision = Some(self.decision_trace(
            snapshot,
            mdp_state,
            observation,
            belief,
            order_names,
            action_options,
            &action,
            action_label,
        ));
        PlayerIntent {
            player_id: self.id,
            action,
            sprint: false,
        }
    }

    fn action_from_learned_plan(
        &self,
        plan: &SoccerLearnedPlan,
        snapshot: &WorldSnapshot,
        observation: &SoccerPomdpObservation,
    ) -> Option<(SoccerAction, String)> {
        let label = normalize_soccer_action_label(&plan.action);
        match label {
            "shoot" if observation.has_ball && shot_decision_is_qualified(observation) => Some((
                SoccerAction::Shoot {
                    power: 0.72 + 0.28 * ability01(self.skills.shooting),
                },
                "shoot".to_string(),
            )),
            "pass" if observation.has_ball => {
                let visible_targets = snapshot.ranked_visible_pass_targets(self.id, 11);
                let target = plan
                    .target_player
                    .filter(|target| visible_targets.contains(target))
                    .or_else(|| {
                        plan.target_point.and_then(|point| {
                            visible_targets.iter().copied().min_by(|a, b| {
                                let a_dist = snapshot
                                    .player_position(*a)
                                    .map(|position| position.distance(point))
                                    .unwrap_or(f64::INFINITY);
                                let b_dist = snapshot
                                    .player_position(*b)
                                    .map(|position| position.distance(point))
                                    .unwrap_or(f64::INFINITY);
                                a_dist
                                    .partial_cmp(&b_dist)
                                    .unwrap_or(std::cmp::Ordering::Equal)
                            })
                        })
                    })
                    .or_else(|| visible_targets.first().copied());
                target.map(|target| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.58 + 0.32 * ability01(self.skills.passing_completion_rate),
                            flight: PassFlight::Floor,
                        },
                        "pass".to_string(),
                    )
                })
            }
            "aerial-pass" if observation.has_ball => {
                let visible_targets = snapshot.ranked_visible_aerial_pass_targets(self.id, 11);
                let target = plan
                    .target_player
                    .filter(|target| visible_targets.contains(target))
                    .or_else(|| visible_targets.first().copied());
                target.map(|target| {
                    let crossing =
                        ability01(self.skills.crossing_left.max(self.skills.crossing_right));
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.56
                                + 0.28
                                    * crossing.max(ability01(self.skills.passing_completion_rate)),
                            flight: PassFlight::Aerial,
                        },
                        "aerial-pass".to_string(),
                    )
                })
            }
            "first-time-shot" | "first-time-header"
                if observation.has_ball
                    && observation.first_touch_available
                    && first_time_shot_decision_is_qualified(observation) =>
            {
                let finish_skill = if label == "first-time-header" {
                    aerial_duel_skill_from_agent(self)
                } else {
                    ability01(
                        self.skills
                            .right_foot_shot_power
                            .max(self.skills.left_foot_shot_power),
                    )
                };
                Some((
                    SoccerAction::Shoot {
                        power: 0.68 + 0.28 * finish_skill,
                    },
                    label.to_string(),
                ))
            }
            "first-time-pass" if observation.has_ball && observation.first_touch_available => {
                let target = snapshot
                    .ranked_visible_pass_targets(self.id, 1)
                    .first()
                    .copied();
                target.map(|target| {
                    (
                        SoccerAction::Pass {
                            target_player: Some(target),
                            power: 0.54 + 0.30 * ability01(self.skills.passing_completion_rate),
                            flight: PassFlight::Floor,
                        },
                        "first-time-pass".to_string(),
                    )
                })
            }
            "control-touch" if observation.has_ball && observation.first_touch_available => {
                let target = (self.position + carried_ball_lead(self))
                    .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
                Some((
                    SoccerAction::ControlTouch { target },
                    "control-touch".to_string(),
                ))
            }
            "dribble" if observation.has_ball => Some((
                SoccerAction::Dribble(plan.target_point.unwrap_or_else(|| {
                    snapshot.shot_creation_space_for(self.id, self.home_position)
                })),
                "dribble".to_string(),
            )),
            "defend" if snapshot.controlled_possession_team() == Some(self.team.other()) => Some((
                SoccerAction::MoveTo(snapshot.defensive_assignment_for(
                    self.id,
                    self.home_position,
                    false,
                )),
                "defend".to_string(),
            )),
            "recover" if snapshot.controlled_possession_team().is_none() => Some((
                SoccerAction::MoveTo(snapshot.ball.position),
                "recover".to_string(),
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
                let target = if snapshot.controlled_possession_team() == Some(self.team) {
                    snapshot.positional_open_space_for(self.id, self.home_position, false)
                } else if snapshot.controlled_possession_team() == Some(self.team.other()) {
                    snapshot.defensive_assignment_for(self.id, self.home_position, false)
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
    ControlTouch {
        target: Vec2,
    },
    Pass {
        target_player: Option<usize>,
        power: f64,
        #[serde(default)]
        flight: PassFlight,
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
            SoccerAction::ControlTouch { .. } => "control-touch".to_string(),
            SoccerAction::Pass { flight, .. } => {
                if flight.is_aerial() {
                    "aerial-pass".to_string()
                } else {
                    "pass".to_string()
                }
            }
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

#[derive(Clone, Debug, Default)]
pub struct SoccerLearnedPlan {
    pub action: String,
    pub target_player: Option<usize>,
    pub target_point: Option<Vec2>,
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
    #[serde(default, alias = "flight")]
    pub pass_flight: PassFlight,
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

    fn drain_latest_for_slot(&mut self, controller_slot: usize) -> Option<HumanInputFrame> {
        let mut latest = None;
        let mut retained = VecDeque::with_capacity(self.pending.len());
        while let Some(input) = self.pending.pop_front() {
            if input.controller_slot == controller_slot {
                if latest
                    .as_ref()
                    .map_or(true, |current: &HumanInputFrame| input.seq > current.seq)
                {
                    latest = Some(input);
                }
            } else {
                retained.push_back(input);
            }
        }
        self.pending = retained;
        latest
    }
}

#[derive(Clone)]
pub struct SharedHumanInputs {
    inner: Arc<RwLock<SharedHumanInputStore>>,
    notifier: Arc<(Mutex<u64>, Condvar)>,
}

impl Default for SharedHumanInputs {
    fn default() -> Self {
        SharedHumanInputs {
            inner: Arc::new(RwLock::new(SharedHumanInputStore::default())),
            notifier: Arc::new((Mutex::new(0), Condvar::new())),
        }
    }
}

impl SharedHumanInputs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&self, input: HumanInputFrame) -> bool {
        let accepted = self
            .inner
            .write()
            .expect("human input queue lock poisoned")
            .push(input);
        if accepted {
            let (lock, condvar) = &*self.notifier;
            let mut version = lock.lock().expect("human input notifier lock poisoned");
            *version = version.saturating_add(1);
            condvar.notify_all();
        }
        accepted
    }

    pub fn drain_latest_by_slot(&self) -> HashMap<usize, HumanInputFrame> {
        self.inner
            .write()
            .expect("human input queue lock poisoned")
            .drain_latest_by_slot()
    }

    pub fn drain_latest_for_slot(&self, controller_slot: usize) -> Option<HumanInputFrame> {
        self.inner
            .write()
            .expect("human input queue lock poisoned")
            .drain_latest_for_slot(controller_slot)
    }

    pub fn notification_version(&self) -> u64 {
        let (lock, _) = &*self.notifier;
        *lock.lock().expect("human input notifier lock poisoned")
    }

    pub fn wait_for_change_since(&self, version: u64, timeout: Duration) -> u64 {
        let (lock, condvar) = &*self.notifier;
        let guard = lock.lock().expect("human input notifier lock poisoned");
        let (guard, _) = condvar
            .wait_timeout_while(guard, timeout, |current| *current <= version)
            .expect("human input notifier wait poisoned");
        *guard
    }

    pub fn wait_for_pending_input(&self, timeout: Duration) -> bool {
        let version = self.notification_version();
        if self.queued_len() > 0 {
            return true;
        }
        let next_version = self.wait_for_change_since(version, timeout);
        next_version > version || self.queued_len() > 0
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HumanControllerThreadStats {
    pub controller_slot: usize,
    pub pending: bool,
    pub closed: bool,
    pub accepted_frames: u64,
    pub pushed_frames: u64,
    pub overwritten_frames: u64,
    pub rejected_stale_frames: u64,
    pub latest_seq_seen: Option<u64>,
}

#[derive(Default)]
struct HumanControllerMailboxState {
    latest: Option<HumanInputFrame>,
    closed: bool,
    accepted_frames: u64,
    pushed_frames: u64,
    overwritten_frames: u64,
    rejected_stale_frames: u64,
    latest_seq_seen: Option<u64>,
}

#[derive(Clone, Default)]
struct HumanControllerMailbox {
    inner: Arc<(Mutex<HumanControllerMailboxState>, Condvar)>,
}

impl HumanControllerMailbox {
    fn send(&self, input: HumanInputFrame) -> Result<bool, String> {
        let (lock, condvar) = &*self.inner;
        let mut state = lock.lock().expect("human controller mailbox lock poisoned");
        if state.closed {
            return Err("controller mailbox is closed".to_string());
        }
        if state
            .latest_seq_seen
            .is_some_and(|latest_seq| input.seq <= latest_seq)
        {
            state.rejected_stale_frames = state.rejected_stale_frames.saturating_add(1);
            return Ok(false);
        }

        state.latest_seq_seen = Some(input.seq);
        state.accepted_frames = state.accepted_frames.saturating_add(1);
        if state.latest.is_some() {
            state.overwritten_frames = state.overwritten_frames.saturating_add(1);
        }
        state.latest = Some(input);
        condvar.notify_one();
        Ok(true)
    }

    fn close(&self) {
        let (lock, condvar) = &*self.inner;
        let mut state = lock.lock().expect("human controller mailbox lock poisoned");
        state.closed = true;
        condvar.notify_all();
    }

    fn wait_for_debounced_input(&self, debounce_interval: Duration) -> Option<HumanInputFrame> {
        let (lock, condvar) = &*self.inner;
        let mut state = lock.lock().expect("human controller mailbox lock poisoned");
        while state.latest.is_none() && !state.closed {
            state = condvar
                .wait(state)
                .expect("human controller mailbox wait poisoned");
        }
        state.latest.as_ref()?;

        if debounce_interval > Duration::from_millis(0) && !state.closed {
            let deadline = Instant::now() + debounce_interval;
            loop {
                let now = Instant::now();
                if now >= deadline || state.closed {
                    break;
                }
                let wait_for = deadline.saturating_duration_since(now);
                let (next_state, _) = condvar
                    .wait_timeout(state, wait_for)
                    .expect("human controller mailbox debounce wait poisoned");
                state = next_state;
            }
        }

        state.latest.take()
    }

    fn record_push(&self, accepted: bool) {
        if !accepted {
            return;
        }
        let (lock, _) = &*self.inner;
        let mut state = lock.lock().expect("human controller mailbox lock poisoned");
        state.pushed_frames = state.pushed_frames.saturating_add(1);
    }

    fn stats(&self, controller_slot: usize) -> HumanControllerThreadStats {
        let (lock, _) = &*self.inner;
        let state = lock.lock().expect("human controller mailbox lock poisoned");
        HumanControllerThreadStats {
            controller_slot,
            pending: state.latest.is_some(),
            closed: state.closed,
            accepted_frames: state.accepted_frames,
            pushed_frames: state.pushed_frames,
            overwritten_frames: state.overwritten_frames,
            rejected_stale_frames: state.rejected_stale_frames,
            latest_seq_seen: state.latest_seq_seen,
        }
    }
}

pub struct HumanControllerThread {
    controller_slot: usize,
    mailbox: HumanControllerMailbox,
    handle: Option<JoinHandle<()>>,
}

impl HumanControllerThread {
    pub fn spawn(
        input_queue: SharedHumanInputs,
        controller_slot: usize,
        debounce_interval: Duration,
    ) -> Result<Self, String> {
        let mailbox = HumanControllerMailbox::default();
        let worker_mailbox = mailbox.clone();
        let handle = thread::Builder::new()
            .name(format!("soccer-human-controller-{controller_slot}"))
            .spawn(move || {
                run_human_controller_thread(
                    input_queue,
                    controller_slot,
                    debounce_interval,
                    worker_mailbox,
                );
            })
            .map_err(|err| format!("failed to spawn controller thread {controller_slot}: {err}"))?;
        Ok(HumanControllerThread {
            controller_slot,
            mailbox,
            handle: Some(handle),
        })
    }

    pub fn controller_slot(&self) -> usize {
        self.controller_slot
    }

    pub fn send_input(&self, mut input: HumanInputFrame) -> Result<bool, String> {
        input.controller_slot = self.controller_slot;
        self.mailbox.send(input).map_err(|err| {
            format!(
                "controller thread {} send failed: {err}",
                self.controller_slot
            )
        })
    }

    pub fn stats(&self) -> HumanControllerThreadStats {
        self.mailbox.stats(self.controller_slot)
    }

    pub fn stop(mut self) -> Result<(), String> {
        self.mailbox.close();
        if let Some(handle) = self.handle.take() {
            handle
                .join()
                .map_err(|_| format!("controller thread {} panicked", self.controller_slot))?;
        }
        Ok(())
    }
}

impl Drop for HumanControllerThread {
    fn drop(&mut self) {
        self.mailbox.close();
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub fn spawn_human_controller_threads(
    input_queue: SharedHumanInputs,
    controller_slots: usize,
    debounce_interval: Duration,
) -> Result<Vec<HumanControllerThread>, String> {
    (0..controller_slots.min(4))
        .map(|slot| HumanControllerThread::spawn(input_queue.clone(), slot, debounce_interval))
        .collect()
}

fn run_human_controller_thread(
    input_queue: SharedHumanInputs,
    controller_slot: usize,
    debounce_interval: Duration,
    mailbox: HumanControllerMailbox,
) {
    while let Some(mut input) = mailbox.wait_for_debounced_input(debounce_interval) {
        input.controller_slot = controller_slot;
        let accepted = input_queue.push(input);
        mailbox.record_push(accepted);
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
    pub jerk: Vec2,
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
                jerk: p.jerk,
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
                jerk: p.jerk,
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
                let prev_acceleration = self
                    .latest_for(p.id)
                    .map(|sample| sample.acceleration)
                    .unwrap_or(p.acceleration);
                let jerk = if dt_seconds > 0.0 {
                    (acceleration - prev_acceleration) / dt_seconds
                } else {
                    p.jerk
                };
                PlayerPositionSample {
                    player_id: p.id,
                    tick,
                    clock_seconds,
                    position: p.position,
                    velocity: p.velocity,
                    acceleration,
                    jerk,
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
    #[serde(default)]
    pub curl_acceleration: Vec2,
    #[serde(default)]
    pub altitude_yards: f64,
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
    #[serde(default)]
    pub curl_acceleration: Vec2,
    #[serde(default)]
    pub altitude_yards: f64,
    pub holder: Option<usize>,
    #[serde(default)]
    pub last_touch_team: Option<Team>,
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
    pub curl_acceleration: Vec2,
    pub altitude_yards: f64,
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
            curl_acceleration: state.curl_acceleration,
            altitude_yards: state.altitude_yards,
            position_history: VecDeque::from([BallPositionSample {
                tick: 0,
                clock_seconds: 0.0,
                position: state.position,
                velocity: state.velocity,
                acceleration: state.acceleration,
                curl_acceleration: state.curl_acceleration,
                altitude_yards: state.altitude_yards,
                holder: state.holder,
                last_touch_team: state.last_touch_team,
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
            curl_acceleration: self.curl_acceleration,
            altitude_yards: self.altitude_yards,
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
            curl_acceleration: self.curl_acceleration,
            altitude_yards: self.altitude_yards,
            holder: self.holder,
            last_touch_team: self.last_touch_team,
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
                let lead = carried_ball_lead(player);
                self.position = (player.position + lead)
                    .clamp_to_pitch(context.field_width, context.field_length);
                self.velocity = player.velocity;
                self.altitude_yards = 0.0;
                self.last_touch_team = Some(player.team);
                self.record_decision(context.tick, "held");
            }
            return BallStepOutcome::None;
        }

        let previous_position = self.position;
        let previous_velocity = self.velocity;
        self.velocity = ball_velocity_after_resistance(
            self.velocity,
            context.dt_seconds,
            context.ball_drag_per_tick,
            context.ball_air_resistance,
            context.ball_grass_resistance_yps2,
        );
        self.position += (previous_velocity + self.velocity) * 0.5 * context.dt_seconds;
        self.altitude_yards = context
            .pending_pass
            .as_ref()
            .map(|pass| pass_ball_altitude_yards(pass, self.position))
            .unwrap_or(0.0);
        if self.velocity.len() < context.ball_stop_speed_yps {
            self.velocity = Vec2::zero();
            self.altitude_yards = 0.0;
        }

        let x_crossing_fraction = boundary_crossing_fraction(
            previous_position.x,
            self.position.x,
            0.0,
            context.field_width,
        );
        let y_crossing_fraction = boundary_crossing_fraction(
            previous_position.y,
            self.position.y,
            0.0,
            context.field_length,
        );
        if self.position.x < 0.0 || self.position.x > context.field_width {
            let endline_crossed_first =
                matches!((x_crossing_fraction, y_crossing_fraction), (Some(x), Some(y)) if y <= x);
            if !endline_crossed_first {
                let awarded_team = self.last_touch_team.map(Team::other).unwrap_or(Team::Home);
                self.position = Vec2::new(
                    self.position.x.max(0.0).min(context.field_width),
                    self.position.y.max(0.0).min(context.field_length),
                );
                self.velocity = Vec2::zero();
                self.altitude_yards = 0.0;
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
        }

        if self.position.y < 0.0 || self.position.y > context.field_length {
            let goal_x = context.field_width * 0.5;
            let goal_line_y = if self.position.y > context.field_length {
                context.field_length
            } else {
                0.0
            };
            let crossing_fraction = y_crossing_fraction.unwrap_or(1.0);
            let crossing_x =
                previous_position.x + (self.position.x - previous_position.x) * crossing_fraction;
            let crossing_position =
                Vec2::new(crossing_x.clamp(0.0, context.field_width), goal_line_y);
            let in_goal = (crossing_x - goal_x).abs() <= context.goal_width * 0.5;
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
                            crossing_position,
                            self.velocity.len(),
                            context.goal_width,
                        );
                        if rng.next_float() < save_probability {
                            let save_y = match defending_team {
                                Team::Home => SHOT_SAVE_DEPTH_YARDS,
                                Team::Away => context.field_length - SHOT_SAVE_DEPTH_YARDS,
                            };
                            let save_position = Vec2::new(
                                crossing_position.x.clamp(
                                    context.field_width * 0.5 - context.goal_width * 0.55,
                                    context.field_width * 0.5 + context.goal_width * 0.55,
                                ),
                                save_y,
                            );
                            self.holder = Some(keeper_id);
                            self.position = save_position;
                            self.velocity = Vec2::zero();
                            self.altitude_yards = 0.0;
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
                    let corner_x = if crossing_position.x <= context.field_width * 0.5 {
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
            self.altitude_yards = 0.0;
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

        if let Some((holder, holder_team)) = nearest_ball_controller_for(
            self.position,
            self.velocity,
            context.players,
            context.pending_pass.as_ref(),
            rng,
        ) {
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
    pub acceleration: Vec2,
    pub jerk: Vec2,
    pub position_history: VecDeque<Vec2>,
}

impl OfficialAgent {
    fn new(id: usize, kind: OfficialKind, position: Vec2) -> Self {
        OfficialAgent {
            id,
            kind,
            position,
            velocity: Vec2::zero(),
            acceleration: Vec2::zero(),
            jerk: Vec2::zero(),
            position_history: VecDeque::from([position]),
        }
    }

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

    pub fn history_jerk_estimate(&self, dt_seconds: f64) -> Vec2 {
        if dt_seconds <= 0.0 || self.position_history.len() < 4 {
            return self.jerk;
        }
        let last = self.position_history.len() - 1;
        let v0 = (self.position_history[last - 2] - self.position_history[last - 3]) / dt_seconds;
        let v1 = (self.position_history[last - 1] - self.position_history[last - 2]) / dt_seconds;
        let v2 = (self.position_history[last] - self.position_history[last - 1]) / dt_seconds;
        let a0 = (v1 - v0) / dt_seconds;
        let a1 = (v2 - v1) / dt_seconds;
        (a1 - a0) / dt_seconds
    }

    fn run_time_step(&mut self, snapshot: &WorldSnapshot, rng: &mut SeededRandom) {
        let previous_velocity = self.velocity;
        let previous_acceleration = self.acceleration;
        let offside_line = assistant_offside_line_snapshot(snapshot, self.kind);
        let base_target = match self.kind {
            OfficialKind::CenterReferee => Vec2::new(
                snapshot.field_width * 0.5,
                snapshot.ball.position.y * 0.72 + snapshot.field_length * 0.14,
            ),
            OfficialKind::AssistantRefereeNear => Vec2::new(
                1.5,
                offside_line
                    .as_ref()
                    .map(|line| line.effective_line_y)
                    .unwrap_or(snapshot.ball.position.y),
            ),
            OfficialKind::AssistantRefereeFar => Vec2::new(
                snapshot.field_width - 1.5,
                offside_line
                    .as_ref()
                    .map(|line| line.effective_line_y)
                    .unwrap_or(snapshot.ball.position.y),
            ),
        };
        let jitter = Vec2::new(rng.next_float() - 0.5, rng.next_float() - 0.5) * 0.25;
        let target =
            official_clearance_target(self.kind, base_target + jitter, self.position, snapshot);
        let desired = (target - self.position).normalized() * 6.1;
        self.velocity = approach_velocity(self.velocity, desired, 5.2, snapshot.dt_seconds);
        self.acceleration = if snapshot.dt_seconds > 0.0 {
            (self.velocity - previous_velocity) / snapshot.dt_seconds
        } else {
            Vec2::zero()
        };
        self.jerk = if snapshot.dt_seconds > 0.0 {
            (self.acceleration - previous_acceleration) / snapshot.dt_seconds
        } else {
            Vec2::zero()
        };
        self.position += self.velocity * snapshot.dt_seconds;
        self.position = self
            .position
            .clamp_to_pitch(snapshot.field_width, snapshot.field_length);
        self.record_position_history();
    }
}

fn official_clearance_target(
    kind: OfficialKind,
    target: Vec2,
    position: Vec2,
    snapshot: &WorldSnapshot,
) -> Vec2 {
    let clearance = match kind {
        OfficialKind::CenterReferee => CENTER_REF_BALL_CLEARANCE_YARDS,
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => {
            ASSISTANT_REF_BALL_CLEARANCE_YARDS
        }
    };
    let fallback = official_clearance_fallback(kind, snapshot.ball.position, snapshot);
    let mut adjusted = point_with_clearance(target, snapshot.ball.position, fallback, clearance);

    if let Some(holder_position) = snapshot
        .ball
        .holder
        .and_then(|holder| snapshot.players.iter().find(|player| player.id == holder))
        .map(|player| player.position)
    {
        adjusted = point_with_clearance(adjusted, holder_position, fallback, clearance * 0.78);
    }

    let current_distance = position.distance(snapshot.ball.position);
    if current_distance < clearance {
        let away = if current_distance > 1e-6 {
            (position - snapshot.ball.position).normalized()
        } else {
            fallback
        };
        adjusted += away * (clearance - current_distance + 0.7);
    }

    adjusted.clamp_to_pitch(snapshot.field_width, snapshot.field_length)
}

fn point_with_clearance(point: Vec2, blocked: Vec2, fallback: Vec2, clearance: f64) -> Vec2 {
    let delta = point - blocked;
    let distance = delta.len();
    if distance >= clearance {
        point
    } else {
        let away = if distance > 1e-6 {
            delta.normalized()
        } else {
            fallback
        };
        blocked + away * clearance
    }
}

fn official_clearance_fallback(
    kind: OfficialKind,
    ball_position: Vec2,
    snapshot: &WorldSnapshot,
) -> Vec2 {
    match kind {
        OfficialKind::CenterReferee => {
            if ball_position.x <= snapshot.field_width * 0.5 {
                Vec2::new(1.0, 0.0)
            } else {
                Vec2::new(-1.0, 0.0)
            }
        }
        OfficialKind::AssistantRefereeNear | OfficialKind::AssistantRefereeFar => {
            if ball_position.y <= snapshot.field_length * 0.5 {
                Vec2::new(0.0, -1.0)
            } else {
                Vec2::new(0.0, 1.0)
            }
        }
    }
}

fn assistant_offside_line_snapshot(
    snapshot: &WorldSnapshot,
    kind: OfficialKind,
) -> Option<AssistantOffsideLineSnapshot> {
    let flank = match kind {
        OfficialKind::AssistantRefereeNear => AssistantFlank::Near,
        OfficialKind::AssistantRefereeFar => AssistantFlank::Far,
        OfficialKind::CenterReferee => return None,
    };
    let attacking_team = snapshot.possession_team()?;
    let defending_team = attacking_team.other();
    let mut defender_ys = snapshot
        .players
        .iter()
        .filter(|player| player.team == defending_team)
        .filter_map(|player| {
            snapshot
                .player_position(player.id)
                .map(|position| position.y)
        })
        .collect::<Vec<_>>();
    if defender_ys.len() < 2 {
        return None;
    }

    let halfway_y = snapshot.field_length * 0.5;
    let second_last_defender_y = match attacking_team {
        Team::Home => {
            defender_ys.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            defender_ys[1]
        }
        Team::Away => {
            defender_ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            defender_ys[1]
        }
    };
    let effective_line_y = match attacking_team {
        Team::Home => snapshot
            .ball
            .position
            .y
            .max(second_last_defender_y)
            .max(halfway_y),
        Team::Away => snapshot
            .ball
            .position
            .y
            .min(second_last_defender_y)
            .min(halfway_y),
    };
    let players_beyond_line = snapshot
        .players
        .iter()
        .filter(|player| player.team == attacking_team)
        .filter_map(|player| {
            let position = snapshot.player_position(player.id)?;
            if !assistant_flank_contains(flank, position, snapshot.field_width) {
                return None;
            }
            let beyond_line = match attacking_team {
                Team::Home => position.y > effective_line_y,
                Team::Away => position.y < effective_line_y,
            };
            beyond_line.then_some(player.id)
        })
        .collect();

    Some(AssistantOffsideLineSnapshot {
        flank,
        attacking_team,
        defending_team,
        second_last_defender_y,
        ball_y: snapshot.ball.position.y,
        halfway_y,
        effective_line_y,
        players_beyond_line,
    })
}

fn assistant_flank_contains(flank: AssistantFlank, position: Vec2, field_width: f64) -> bool {
    match flank {
        AssistantFlank::Near => position.x <= field_width * 0.5,
        AssistantFlank::Far => position.x > field_width * 0.5,
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SoccerTacticalLearningWeights {
    pub attack_spacing_delta_weight: f64,
    pub attack_spacing_score_weight: f64,
    pub attack_width_delta_weight: f64,
    pub attack_width_score_weight: f64,
    pub attack_flank_lane_weight: f64,
    pub defense_spacing_delta_weight: f64,
    pub defense_spacing_score_weight: f64,
    pub defense_contract_delta_weight: f64,
    pub defense_compactness_score_weight: f64,
}

impl Default for SoccerTacticalLearningWeights {
    fn default() -> Self {
        SoccerTacticalLearningWeights {
            attack_spacing_delta_weight: 0.22,
            attack_spacing_score_weight: 0.06,
            attack_width_delta_weight: 0.52,
            attack_width_score_weight: 0.14,
            attack_flank_lane_weight: 0.28,
            defense_spacing_delta_weight: 0.08,
            defense_spacing_score_weight: 0.04,
            defense_contract_delta_weight: 0.42,
            defense_compactness_score_weight: 0.14,
        }
    }
}

impl SoccerTacticalLearningWeights {
    fn validate(&self) -> Result<(), String> {
        for (name, value) in [
            ("attackSpacingDeltaWeight", self.attack_spacing_delta_weight),
            ("attackSpacingScoreWeight", self.attack_spacing_score_weight),
            ("attackWidthDeltaWeight", self.attack_width_delta_weight),
            ("attackWidthScoreWeight", self.attack_width_score_weight),
            ("attackFlankLaneWeight", self.attack_flank_lane_weight),
            (
                "defenseSpacingDeltaWeight",
                self.defense_spacing_delta_weight,
            ),
            (
                "defenseSpacingScoreWeight",
                self.defense_spacing_score_weight,
            ),
            (
                "defenseContractDeltaWeight",
                self.defense_contract_delta_weight,
            ),
            (
                "defenseCompactnessScoreWeight",
                self.defense_compactness_score_weight,
            ),
        ] {
            if !value.is_finite() {
                return Err(format!("{name} must be finite"));
            }
            if !(-5.0..=5.0).contains(&value) {
                return Err(format!("{name} must be between -5.0 and 5.0"));
            }
        }
        Ok(())
    }
}

fn default_tactical_learning_weights() -> SoccerTacticalLearningWeights {
    SoccerTacticalLearningWeights::default()
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchConfig {
    pub dt_seconds: f64,
    pub duration_seconds: f64,
    #[serde(default = "default_period_count")]
    pub period_count: usize,
    #[serde(default = "default_period_break_recovery_seconds")]
    pub period_break_recovery_seconds: f64,
    pub field_length_yards: f64,
    pub field_width_yards: f64,
    pub goal_width_yards: f64,
    #[serde(default = "default_ball_drag_per_tick")]
    pub ball_drag_per_tick: f64,
    #[serde(default = "default_ball_air_resistance")]
    pub ball_air_resistance: f64,
    #[serde(default = "default_ball_grass_resistance_yps2")]
    pub ball_grass_resistance_yps2: f64,
    #[serde(default = "default_ball_stop_speed_yps")]
    pub ball_stop_speed_yps: f64,
    #[serde(default = "default_learning_enabled")]
    pub learning_enabled: bool,
    #[serde(default = "default_learning_logging_enabled")]
    pub learning_logging_enabled: bool,
    #[serde(default = "default_learning_interval_ticks")]
    pub learning_interval_ticks: usize,
    #[serde(default = "default_tactical_learning_weights")]
    pub tactical_learning: SoccerTacticalLearningWeights,
    pub max_human_players: usize,
    pub seed: u32,
}

impl Default for MatchConfig {
    fn default() -> Self {
        MatchConfig {
            dt_seconds: DEFAULT_DT_SECONDS,
            duration_seconds: DEFAULT_DURATION_SECONDS,
            period_count: 1,
            period_break_recovery_seconds: 0.0,
            field_length_yards: DEFAULT_FIELD_LENGTH_YARDS,
            field_width_yards: DEFAULT_FIELD_WIDTH_YARDS,
            goal_width_yards: DEFAULT_GOAL_WIDTH_YARDS,
            ball_drag_per_tick: DEFAULT_BALL_DRAG_PER_TICK,
            ball_air_resistance: DEFAULT_BALL_AIR_RESISTANCE,
            ball_grass_resistance_yps2: DEFAULT_BALL_GRASS_RESISTANCE_YPS2,
            ball_stop_speed_yps: DEFAULT_BALL_STOP_SPEED_YPS,
            learning_enabled: true,
            learning_logging_enabled: true,
            learning_interval_ticks: 1,
            tactical_learning: SoccerTacticalLearningWeights::default(),
            max_human_players: 4,
            seed: 2026,
        }
    }
}

impl MatchConfig {
    pub fn total_ticks(&self) -> u64 {
        (self.duration_seconds / self.dt_seconds).round() as u64
    }

    pub fn periods(&self) -> usize {
        self.period_count.max(1)
    }

    pub fn period_start_after_tick(&self, tick: u64) -> Option<usize> {
        let periods = self.periods();
        let total_ticks = self.total_ticks();
        for completed_periods in 1..periods {
            let boundary_tick =
                total_ticks.saturating_mul(completed_periods as u64) / periods as u64;
            if boundary_tick > 0 && boundary_tick < total_ticks && boundary_tick == tick {
                return Some(completed_periods + 1);
            }
        }
        None
    }

    pub fn human_slots(&self) -> usize {
        self.max_human_players.min(4)
    }
}

fn kickoff_team_for_period(period_number: usize) -> Team {
    if period_number % 2 == 0 {
        Team::Away
    } else {
        Team::Home
    }
}

fn validate_ball_surface(
    ball_drag_per_tick: f64,
    ball_air_resistance: f64,
    ball_grass_resistance_yps2: f64,
    ball_stop_speed_yps: f64,
) -> Result<(), String> {
    if !ball_drag_per_tick.is_finite() {
        return Err("ballDragPerTick must be finite".to_string());
    }
    if !(0.0..=0.95).contains(&ball_drag_per_tick) {
        return Err("ballDragPerTick must be between 0.0 and 0.95".to_string());
    }
    if !ball_air_resistance.is_finite() {
        return Err("ballAirResistance must be finite".to_string());
    }
    if !(0.0..=0.10).contains(&ball_air_resistance) {
        return Err("ballAirResistance must be between 0.0 and 0.10".to_string());
    }
    if !ball_grass_resistance_yps2.is_finite() {
        return Err("ballGrassResistanceYps2 must be finite".to_string());
    }
    if !(0.0..=5.0).contains(&ball_grass_resistance_yps2) {
        return Err("ballGrassResistanceYps2 must be between 0.0 and 5.0".to_string());
    }
    if !ball_stop_speed_yps.is_finite() {
        return Err("ballStopSpeedYps must be finite".to_string());
    }
    if !(0.0..=20.0).contains(&ball_stop_speed_yps) {
        return Err("ballStopSpeedYps must be between 0.0 and 20.0".to_string());
    }
    Ok(())
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
    #[serde(default)]
    pub defensive_chase_load_home: f64,
    #[serde(default)]
    pub defensive_chase_load_away: f64,
    #[serde(default)]
    pub possession_chase_advantage_home: f64,
    #[serde(default)]
    pub possession_chase_advantage_away: f64,
}

#[derive(Clone, Debug)]
struct PendingPass {
    team: Team,
    from: usize,
    target: Option<usize>,
    flight: PassFlight,
    is_cross: bool,
    origin: Vec2,
    intended_target: Vec2,
    distance_yards: f64,
    offside: Option<PendingOffside>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PendingPassSnapshot {
    pub team: Team,
    pub from: usize,
    pub target: Option<usize>,
    pub flight: PassFlight,
    pub is_cross: bool,
    pub origin: Vec2,
    pub intended_target: Vec2,
    pub distance_yards: f64,
    pub off_target_yards: f64,
    pub receiver_urgency: f64,
    pub nearest_receiver: Option<usize>,
}

#[derive(Clone, Debug)]
struct PendingShot {
    team: Team,
    shooter: usize,
}

#[derive(Clone, Debug)]
struct ShotBlockAssessment {
    blocker_id: usize,
    defending_team: Team,
    block_position: Vec2,
    probability: f64,
    distance_to_ball: f64,
    lateral_distance: f64,
    screen_score: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShotDeflectionKind {
    CornerKick,
    GoalBound,
    Rebound,
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
    ball_air_resistance: f64,
    ball_grass_resistance_yps2: f64,
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
    ShotBlocked {
        shot: PendingShot,
        blocker_id: usize,
        defending_team: Team,
        position: Vec2,
        deflection_kind: ShotDeflectionKind,
        restart: Option<BallRestart>,
    },
    OutOfPlay {
        restart: BallRestart,
        shot: Option<PendingShot>,
    },
}

#[derive(Clone, Debug)]
struct SoccerRewardEvent {
    tick: u64,
    player_id: usize,
    amount: f64,
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
    pub defensive_cover_target: usize,
    pub defensive_cover_actual: usize,
    pub foremost_attacker_y: Option<f64>,
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
            defensive_cover_target: 2,
            defensive_cover_actual: 0,
            foremost_attacker_y: None,
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

#[derive(Clone, Copy, Debug)]
struct DefensiveCoverProfile {
    target: usize,
    actual: usize,
    foremost_attacker_y: Option<f64>,
}

impl Default for DefensiveCoverProfile {
    fn default() -> Self {
        DefensiveCoverProfile {
            target: 2,
            actual: 2,
            foremost_attacker_y: None,
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
    pub fn run_time_step(&mut self, snapshot: &WorldSnapshot, rng: &mut SeededRandom) {
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
        let home_cover =
            defensive_cover_profile(snapshot, Team::Home, sample_defensive_cover_target(rng));
        let away_cover =
            defensive_cover_profile(snapshot, Team::Away, sample_defensive_cover_target(rng));
        self.home_directive = tactical_directive_for_team(
            Team::Home,
            self.phase,
            self.possession_team,
            snapshot.ball.position,
            score_diff_home,
            snapshot.field_width,
            snapshot.field_length,
            home_cover,
        );
        self.away_directive = tactical_directive_for_team(
            Team::Away,
            self.phase,
            self.possession_team,
            snapshot.ball.position,
            -score_diff_home,
            snapshot.field_width,
            snapshot.field_length,
            away_cover,
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

    pub fn to_snapshot(
        &self,
        snapshot: &WorldSnapshot,
        tracked_officials: usize,
    ) -> CentralBrainSnapshot {
        CentralBrainSnapshot {
            phase: self.phase,
            possession_team: self.possession_team.or_else(|| snapshot.possession_team()),
            ball_position: snapshot.ball.position,
            ball_velocity: snapshot.ball.velocity,
            ball_holder: snapshot.ball.holder,
            pressure_line_home: self.pressure_line_home,
            pressure_line_away: self.pressure_line_away,
            tracked_players: snapshot
                .players
                .iter()
                .map(|player| CentralBrainPlayerAwareness {
                    id: player.id,
                    team: player.team,
                    position: player.position,
                    velocity: player.velocity,
                    controller_slot: player.controller_slot,
                })
                .collect(),
            tracked_officials,
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
    #[serde(default)]
    pub position_history: Vec<Vec2>,
    pub velocity: Vec2,
    #[serde(default)]
    pub movement_gait: MovementGait,
    #[serde(default)]
    pub receive_facing: FacingBucket,
    #[serde(default)]
    pub action_facing: FacingBucket,
    #[serde(default)]
    pub incoming_ball: Option<IncomingBallContext>,
    #[serde(default)]
    pub skills: SkillProfile,
    #[serde(default)]
    pub fatigue: f64,
    pub home_position: Vec2,
    pub controller_slot: Option<usize>,
    #[serde(default)]
    pub vision_range_yards: f64,
    #[serde(default)]
    pub field_of_view_degrees: f64,
    pub acceleration: Vec2,
    pub jerk: Vec2,
    pub last_decision: Option<AgentDecisionTrace>,
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
    #[serde(default)]
    pub pending_pass: Option<PendingPassSnapshot>,
    pub players: Vec<PlayerSnapshot>,
    pub shared_positions: SharedPlayerPositionSnapshot,
    pub score_home: u32,
    pub score_away: u32,
    pub phase: TacticalPhase,
    pub home_directive: TeamTacticalDirective,
    pub away_directive: TeamTacticalDirective,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TeamSpacingMode {
    InPossession,
    Defending,
}

impl TeamSpacingMode {
    fn band(self) -> (f64, f64, f64) {
        match self {
            TeamSpacingMode::InPossession => (
                ATTACK_SPACING_MIN_YARDS,
                ATTACK_SPACING_IDEAL_YARDS,
                ATTACK_SPACING_MAX_YARDS,
            ),
            TeamSpacingMode::Defending => (
                DEFENSE_SPACING_MIN_YARDS,
                DEFENSE_SPACING_IDEAL_YARDS,
                DEFENSE_SPACING_MAX_YARDS,
            ),
        }
    }

    fn ideal(self) -> f64 {
        self.band().1
    }
}

fn spacing_score_from_distance(distance_yards: f64, mode: TeamSpacingMode) -> f64 {
    if !distance_yards.is_finite() {
        return -1.0;
    }
    let (min, ideal, max) = mode.band();
    if distance_yards < min {
        -((min - distance_yards) / min.max(1.0)).clamp(0.0, 1.0)
    } else if distance_yards > max {
        -((distance_yards - max) / max.max(1.0)).clamp(0.0, 1.0)
    } else {
        let span = if distance_yards <= ideal {
            (ideal - min).max(1e-6)
        } else {
            (max - ideal).max(1e-6)
        };
        (1.0 - ((distance_yards - ideal).abs() / span) * 0.45).clamp(0.55, 1.0)
    }
}

fn pending_pass_snapshot_from(
    pass: &PendingPass,
    ball_position: Vec2,
    ball_velocity: Vec2,
    players: &[PlayerSnapshot],
) -> PendingPassSnapshot {
    let off_target_yards =
        segment_distance_to_point(pass.origin, pass.intended_target, ball_position);
    let intended_receiver = pass.target.or_else(|| {
        players
            .iter()
            .filter(|player| player.team == pass.team && player.id != pass.from)
            .min_by(|a, b| {
                a.position
                    .distance(pass.intended_target)
                    .partial_cmp(&b.position.distance(pass.intended_target))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|player| player.id)
    });
    let nearest_same_team_to_ball = players
        .iter()
        .filter(|player| player.team == pass.team && player.id != pass.from)
        .min_by(|a, b| {
            a.position
                .distance(ball_position)
                .partial_cmp(&b.position.distance(ball_position))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|player| player.id);
    let nearest_receiver = if off_target_yards > 1.1 {
        match (intended_receiver, nearest_same_team_to_ball) {
            (Some(intended), Some(nearest)) => {
                let intended_distance = players
                    .iter()
                    .find(|player| player.id == intended)
                    .map(|player| player.position.distance(ball_position))
                    .unwrap_or(f64::INFINITY);
                let nearest_distance = players
                    .iter()
                    .find(|player| player.id == nearest)
                    .map(|player| player.position.distance(ball_position))
                    .unwrap_or(f64::INFINITY);
                if nearest_distance + 4.0 < intended_distance {
                    Some(nearest)
                } else {
                    Some(intended)
                }
            }
            (Some(intended), None) => Some(intended),
            (None, Some(nearest)) => Some(nearest),
            (None, None) => None,
        }
    } else {
        intended_receiver.or(nearest_same_team_to_ball)
    };
    let receiver_distance = nearest_receiver
        .and_then(|id| {
            players
                .iter()
                .find(|player| player.id == id)
                .map(|player| player.position.distance(ball_position))
        })
        .unwrap_or(36.0);
    let nearest_opponent_distance = players
        .iter()
        .filter(|player| player.team == pass.team.other())
        .map(|player| player.position.distance(ball_position))
        .fold(36.0, f64::min)
        .min(36.0);
    let pressure = (1.0 - nearest_opponent_distance / receiver_distance.max(1.0)).clamp(0.0, 1.0);
    let receiver_urgency = ((off_target_yards / 5.5).clamp(0.0, 1.0) * 0.36
        + (receiver_distance / 18.0).clamp(0.0, 1.0) * 0.34
        + pressure * 0.20
        + (ball_velocity.len() / 28.0).clamp(0.0, 1.0) * 0.10)
        .clamp(0.0, 1.0);

    PendingPassSnapshot {
        team: pass.team,
        from: pass.from,
        target: pass.target,
        flight: pass.flight,
        is_cross: pass.is_cross,
        origin: pass.origin,
        intended_target: pass.intended_target,
        distance_yards: pass.distance_yards,
        off_target_yards,
        receiver_urgency,
        nearest_receiver,
    }
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
                position_history: p.position_history.iter().cloned().collect(),
                velocity: p.velocity,
                movement_gait: p.movement_gait,
                receive_facing: p.receive_facing,
                action_facing: p.action_facing,
                incoming_ball: p.incoming_ball.clone(),
                skills: p.skills.clone(),
                fatigue: p.fatigue,
                home_position: p.home_position,
                controller_slot: p.controller_slot,
                vision_range_yards: vision_range_yards(p.skills.vision),
                field_of_view_degrees: field_of_view_degrees(p.skills.vision),
                acceleration: shared_positions
                    .latest_for(p.id)
                    .map(|sample| sample.acceleration)
                    .unwrap_or(p.acceleration),
                jerk: shared_positions
                    .latest_for(p.id)
                    .map(|sample| sample.jerk)
                    .unwrap_or(p.jerk),
                last_decision: p.last_decision.clone(),
            })
            .collect::<Vec<_>>();
        let pending_pass = m.pending_pass.as_ref().map(|pass| {
            pending_pass_snapshot_from(pass, m.ball.position, m.ball.velocity, &players)
        });
        WorldSnapshot {
            tick: m.tick,
            clock_seconds: m.clock_seconds,
            dt_seconds: m.config.dt_seconds,
            field_length: m.config.field_length_yards,
            field_width: m.config.field_width_yards,
            goal_width: m.config.goal_width_yards,
            ball: m.ball.to_state(),
            ball_history: m.ball.position_history.iter().cloned().collect(),
            pending_pass,
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

    pub fn controlled_possession_team(&self) -> Option<Team> {
        self.ball
            .holder
            .and_then(|id| self.players.iter().find(|p| p.id == id))
            .map(|p| p.team)
    }

    fn possession_team_for_ball_sample(&self, sample: &BallPositionSample) -> Option<Team> {
        sample
            .holder
            .and_then(|id| self.players.iter().find(|p| p.id == id))
            .map(|p| p.team)
            .or(sample.last_touch_team)
    }

    fn possession_elapsed_seconds_for_team(&self, team: Team) -> f64 {
        if self.possession_team() != Some(team) {
            return 0.0;
        }
        let mut oldest_same_team = self.clock_seconds;
        for sample in self.ball_history.iter().rev() {
            if self.possession_team_for_ball_sample(sample) == Some(team) {
                oldest_same_team = sample.clock_seconds;
            } else {
                break;
            }
        }
        (self.clock_seconds - oldest_same_team).max(0.0)
    }

    fn possession_spacing_weight(&self, team: Team) -> f64 {
        let settled = (self.possession_elapsed_seconds_for_team(team) / SETTLED_POSSESSION_SECONDS)
            .clamp(0.0, 1.0);
        0.35 + settled * 0.65
    }

    fn team_spacing_mode_for(&self, team: Team) -> Option<TeamSpacingMode> {
        match self.possession_team() {
            Some(t) if t == team => Some(TeamSpacingMode::InPossession),
            Some(_) => Some(TeamSpacingMode::Defending),
            None => None,
        }
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

    pub fn player_jerk(&self, player_id: usize) -> Option<Vec2> {
        self.shared_positions
            .latest_for(player_id)
            .map(|sample| sample.jerk)
            .or_else(|| {
                self.players
                    .iter()
                    .find(|p| p.id == player_id)
                    .map(|p| p.jerk)
            })
    }

    pub fn player_position_history(&self, player_id: usize) -> Option<&[PlayerPositionSample]> {
        self.shared_positions.history_for(player_id)
    }

    pub fn ball_position_history(&self) -> &[BallPositionSample] {
        &self.ball_history
    }

    fn nearest_opponent_at(&self, team: Team, position: Vec2) -> Option<(usize, Vec2, f64)> {
        self.players
            .iter()
            .filter(|player| player.team == team.other())
            .filter_map(|player| {
                let player_position = self.player_position(player.id).unwrap_or(player.position);
                Some((
                    player.id,
                    player_position,
                    player_position.distance(position),
                ))
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
    }

    fn nearest_opponent_distance_at(&self, team: Team, position: Vec2) -> f64 {
        self.nearest_opponent_at(team, position)
            .map(|(_, _, distance)| distance)
            .unwrap_or(f64::INFINITY)
    }

    fn no_pressure_at(&self, team: Team, position: Vec2) -> bool {
        self.nearest_opponent_distance_at(team, position) > NO_PRESSURE_BACK_PASS_THRESHOLD_YARDS
    }

    fn nearest_teammate_distance_at(
        &self,
        team: Team,
        position: Vec2,
        exclude_player_id: Option<usize>,
    ) -> f64 {
        self.players
            .iter()
            .filter(|player| player.team == team)
            .filter(|player| exclude_player_id != Some(player.id))
            .map(|player| {
                self.player_position(player.id)
                    .unwrap_or(player.position)
                    .distance(position)
            })
            .fold(f64::INFINITY, f64::min)
    }

    fn team_spacing_score_for_candidate(
        &self,
        team: Team,
        exclude_player_id: Option<usize>,
        position: Vec2,
        mode: TeamSpacingMode,
    ) -> f64 {
        spacing_score_from_distance(
            self.nearest_teammate_distance_at(team, position, exclude_player_id),
            mode,
        )
    }

    fn own_goal_y_for(&self, team: Team) -> f64 {
        team.other().goal_y(self.field_length)
    }

    fn ball_near_own_goal_line(&self, team: Team) -> bool {
        (self.ball.position.y - self.own_goal_y_for(team)).abs() <= DEFENSIVE_GOAL_LINE_BUFFER_YARDS
    }

    fn clamp_defensive_goal_line_and_ball_gap(&self, team: Team, mut target: Vec2) -> Vec2 {
        let dir = team.attack_dir();
        let own_goal_y = self.own_goal_y_for(team);
        if !self.ball_near_own_goal_line(team) {
            if own_goal_y <= self.field_length * 0.5 {
                target.y = target.y.max(own_goal_y + DEFENSIVE_GOAL_LINE_BUFFER_YARDS);
            } else {
                target.y = target.y.min(own_goal_y - DEFENSIVE_GOAL_LINE_BUFFER_YARDS);
            }

            let deepest_connected_y = self.ball.position.y - dir * DEFENSIVE_MAX_BEHIND_BALL_YARDS;
            if dir > 0.0 {
                target.y = target.y.max(deepest_connected_y);
            } else {
                target.y = target.y.min(deepest_connected_y);
            }
        }
        target.clamp_to_pitch(self.field_width, self.field_length)
    }

    fn clamp_forward_onside_support(&self, player: &PlayerSnapshot, mut target: Vec2) -> Vec2 {
        if self.possession_team() != Some(player.team)
            || player.role != PlayerRole::Forward
            || self.in_behind_run_target_for(player.id).is_some()
        {
            return target;
        }
        let Some(line_y) = self.second_last_defender_line_for(player.team) else {
            return target;
        };
        let half_line = self.field_length * 0.5;
        match player.team {
            Team::Home
                if target.y > half_line && target.y > line_y - STRIKER_ONSIDE_BUFFER_YARDS =>
            {
                target.y = line_y - STRIKER_ONSIDE_BUFFER_YARDS;
            }
            Team::Away
                if target.y < half_line && target.y < line_y + STRIKER_ONSIDE_BUFFER_YARDS =>
            {
                target.y = line_y + STRIKER_ONSIDE_BUFFER_YARDS;
            }
            _ => {}
        }
        target.clamp_to_pitch(self.field_width, self.field_length)
    }

    pub fn tactical_directive(&self, team: Team) -> &TeamTacticalDirective {
        match team {
            Team::Home => &self.home_directive,
            Team::Away => &self.away_directive,
        }
    }

    pub fn goalkeeper_for(&self, team: Team) -> Option<usize> {
        self.players
            .iter()
            .find(|player| player.team == team && player.role == PlayerRole::Goalkeeper)
            .map(|player| player.id)
    }

    pub fn mdp_state(&self) -> SoccerMdpState {
        SoccerMdpState {
            tick: self.tick,
            ball_zone_x: zone(self.ball.position.x, self.field_width, 6),
            ball_zone_y: zone(self.ball.position.y, self.field_length, 8),
            player_grid: PitchGridAddress::default(),
            receive_facing: FacingBucket::Unknown,
            action_facing: FacingBucket::Unknown,
            possession_team: self.possession_team(),
            score_diff_for_home: self.score_home as i32 - self.score_away as i32,
            phase: self.phase,
        }
    }

    pub fn mdp_state_for_player(&self, player_id: usize) -> SoccerMdpState {
        let mut state = self.mdp_state();
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return state;
        };
        let position = self.player_position(me.id).unwrap_or(me.position);
        state.player_grid = pitch_grid_address(position, self.field_width, self.field_length);
        state.receive_facing = me.receive_facing;
        let facing = self
            .player_facing_direction(me)
            .map(facing_bucket_from_vector)
            .unwrap_or(me.action_facing);
        state.action_facing = if facing == FacingBucket::Unknown {
            me.action_facing
        } else {
            facing
        };
        state
    }

    pub fn observation_for(&self, player_id: usize) -> SoccerPomdpObservation {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return SoccerPomdpObservation {
                player_id,
                player_grid: PitchGridAddress::default(),
                receive_facing: FacingBucket::Unknown,
                action_facing: FacingBucket::Unknown,
                has_ball: false,
                visible_ball: false,
                visible_teammates: 0,
                visible_opponents: 0,
                visible_pass_options: 0,
                visible_aerial_pass_options: 0,
                floor_pass_lane_score: 0.0,
                aerial_pass_bypass_score: 0.0,
                aerial_pass_interception_risk: 0.0,
                ball_position_confidence: 0.0,
                teammate_position_confidence: 0.0,
                opponent_position_confidence: 0.0,
                player_position_confidences: Vec::new(),
                ball_distance: 0.0,
                nearest_opponent_distance: 0.0,
                nearest_teammate_distance: 0.0,
                team_spacing_score: 0.0,
                preferred_team_spacing_yards: 0.0,
                shot_lane_open: false,
                shot_block_probability: 1.0,
                shot_blocker_distance_yards: 0.0,
                shot_on_frame_probability: 0.0,
                shot_beat_goalkeeper_probability: 0.0,
                shot_curl_probability: 0.0,
                pass_curl_probability: 0.0,
                immediate_dispossession_risk: 0.0,
                yards_to_goal: 0.0,
                yards_to_own_goal: 0.0,
                opponent_goal_angle_degrees: 0.0,
                opposing_goalkeeper_distance: 0.0,
                opposing_goalkeeper_angle_degrees: 0.0,
                forward_dribble_space_yards: 0.0,
                real_pressure: 0.0,
                perceived_pressure: 0.0,
                real_time_on_ball_seconds: 0.0,
                perceived_time_on_ball_seconds: 0.0,
                fatigue: 0.0,
                nearest_defender_fatigue: 0.0,
                perceived_nearest_defender_fatigue: 0.5,
                nearest_defender_fatigue_confidence: 0.0,
                perceived_fatigue_advantage: 0.0,
                first_touch_available: false,
                incoming_ball_kind: IncomingBallKind::None,
                incoming_ball_speed_yps: 0.0,
                incoming_ball_distance_yards: 0.0,
                receiving_pending_pass: false,
                pending_pass_off_target_yards: 0.0,
                pending_pass_receiver_urgency: 0.0,
                first_time_shot_score: 0.0,
                first_time_pass_score: 0.0,
                control_touch_score: 0.0,
                skill_top_speed: 0.0,
                skill_acceleration: 0.0,
                skill_stamina: 0.0,
                skill_strength: 0.0,
                skill_height: 0.0,
                skill_dribbling: 0.0,
                skill_aggression: 0.0,
                skill_defending: 0.0,
                skill_right_foot_shot_power: 0.0,
                skill_left_foot_shot_power: 0.0,
                skill_passing_completion_rate: 0.0,
                skill_flair_passing: 0.0,
                skill_crossing_left: 0.0,
                skill_crossing_right: 0.0,
                skill_goalkeeping: 0.0,
                skill_defensive_tracking: 0.0,
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
        let visible_opponents = opponents
            .iter()
            .filter(|p| self.player_can_see_player(me.id, p.id))
            .count();
        let visible_teammates = teammates
            .iter()
            .filter(|p| self.player_can_see_player(me.id, p.id))
            .count();
        let vision_range = player_vision_range(me);
        let nearest_opponent_distance = opponents
            .iter()
            .filter(|p| self.player_can_see_player(me.id, p.id))
            .map(|p| {
                self.player_position(p.id)
                    .unwrap_or(p.position)
                    .distance(me_position)
            })
            .fold(f64::INFINITY, f64::min)
            .min(vision_range + 12.0);
        let nearest_teammate_distance = teammates
            .iter()
            .filter(|p| self.player_can_see_player(me.id, p.id))
            .map(|p| {
                self.player_position(p.id)
                    .unwrap_or(p.position)
                    .distance(me_position)
            })
            .fold(f64::INFINITY, f64::min)
            .min(vision_range + 12.0);
        let (team_spacing_score, preferred_team_spacing_yards) = self
            .team_spacing_mode_for(me.team)
            .map(|mode| {
                (
                    spacing_score_from_distance(nearest_teammate_distance, mode),
                    mode.ideal(),
                )
            })
            .unwrap_or((0.0, 0.0));
        let goal = Vec2::new(self.field_width * 0.5, me.team.goal_y(self.field_length));
        let own_goal = Vec2::new(
            self.field_width * 0.5,
            me.team.other().goal_y(self.field_length),
        );
        let has_ball = self.ball.holder == Some(player_id);
        let visible_ball = has_ball || self.player_can_see_point(me.id, self.ball.position);
        let ball_position_confidence = if has_ball {
            1.0
        } else {
            self.player_position_confidence_for_point(me.id, self.ball.position)
                .unwrap_or(0.0)
        };
        let teammate_position_confidence =
            average_player_position_confidence(self, me.id, teammates.iter().map(|p| p.id));
        let opponent_position_confidence =
            average_player_position_confidence(self, me.id, opponents.iter().map(|p| p.id));
        let player_position_confidences = self
            .players
            .iter()
            .filter(|player| player.id != me.id)
            .filter_map(|player| self.player_position_confidence_entry(me.id, player))
            .collect::<Vec<_>>();
        let visible_pass_targets = self.ranked_visible_pass_targets(player_id, 3);
        let visible_aerial_pass_targets = self.ranked_visible_aerial_pass_targets(player_id, 3);
        let floor_pass_lane_score =
            floor_pass_lane_score_for_snapshot(self, me, me_position, &visible_pass_targets);
        let aerial_pass_bypass_score = aerial_pass_bypass_score_for_snapshot(
            self,
            me,
            me_position,
            &visible_aerial_pass_targets,
        );
        let aerial_pass_interception_risk = aerial_pass_interception_risk_for_snapshot(
            self,
            me,
            me_position,
            &visible_aerial_pass_targets,
        );
        let player_grid = pitch_grid_address(me_position, self.field_width, self.field_length);
        let action_facing = self
            .player_facing_direction(me)
            .map(facing_bucket_from_vector)
            .unwrap_or(me.action_facing);
        let real_pressure = pressure_from_nearest_distance(nearest_opponent_distance);
        let perceived_pressure =
            perceived_pressure_for_player(me, real_pressure, visible_opponents);
        let real_time_on_ball_seconds = time_on_ball_seconds(real_pressure);
        let perceived_time_on_ball_seconds = time_on_ball_seconds(perceived_pressure);
        let expected_shot_power = 0.72 + 0.28 * ability01(me.skills.shooting);
        let expected_shot_speed_yps = shot_speed_yps_from_power(expected_shot_power, &me.skills);
        let shot_block_assessment = shot_block_assessment_for_snapshot(
            self,
            me_position,
            me.team,
            expected_shot_speed_yps,
            false,
        );
        let shot_block_probability = shot_block_assessment
            .as_ref()
            .map(|assessment| assessment.probability)
            .unwrap_or(0.0);
        let shot_blocker_distance_yards = shot_block_assessment
            .as_ref()
            .map(|assessment| assessment.distance_to_ball)
            .unwrap_or(self.field_length);
        let first_time_shot_block_probability = shot_block_assessment_for_snapshot(
            self,
            me_position,
            me.team,
            expected_shot_speed_yps,
            true,
        )
        .map(|assessment| assessment.probability)
        .unwrap_or(0.0);
        let shot_curl_probability = shot_curl_probability_for_player(
            &me.skills,
            perceived_pressure,
            (goal.y - me_position.y).abs(),
            self.goal_angle_degrees(me_position, me.team),
        );
        let pass_curl_probability = pass_curl_probability_for_snapshot(
            self,
            me,
            me_position,
            &visible_pass_targets,
            &visible_aerial_pass_targets,
            perceived_pressure,
        );
        let nearest_defender = opponents
            .iter()
            .filter_map(|p| {
                let position = self.player_position(p.id).unwrap_or(p.position);
                Some((*p, position, position.distance(me_position)))
            })
            .min_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));
        let (nearest_defender_fatigue, nearest_defender_fatigue_confidence) =
            if let Some((defender, defender_position, _)) = nearest_defender {
                let position_confidence = self
                    .player_position_confidence_for_point(me.id, defender_position)
                    .unwrap_or(0.0);
                (
                    defender.fatigue.clamp(0.0, 1.0),
                    (position_confidence * 0.85).clamp(0.0, 1.0),
                )
            } else {
                (0.5, 0.0)
            };
        let perceived_nearest_defender_fatigue = nearest_defender_fatigue
            * nearest_defender_fatigue_confidence
            + 0.5 * (1.0 - nearest_defender_fatigue_confidence);
        let perceived_fatigue_advantage =
            perceived_nearest_defender_fatigue - me.fatigue.clamp(0.0, 1.0);
        let opposing_keeper_position = self
            .goalkeeper_for(me.team.other())
            .and_then(|keeper_id| self.player_position(keeper_id));
        let incoming = me.incoming_ball.as_ref().filter(|context| {
            self.tick.saturating_sub(context.received_tick) <= FIRST_TOUCH_WINDOW_TICKS
        });
        let first_touch_available = has_ball && incoming.is_some();
        let incoming_kind = incoming
            .map(|context| context.kind)
            .unwrap_or(IncomingBallKind::None);
        let incoming_speed_yps = incoming.map(|context| context.speed_yps).unwrap_or(0.0);
        let incoming_distance_yards = incoming
            .map(|context| context.distance_yards)
            .unwrap_or(0.0);
        let receiving_pending_pass = self
            .pending_pass
            .as_ref()
            .is_some_and(|pass| pass.team == me.team && pass.nearest_receiver == Some(me.id));
        let pending_pass_off_target_yards = if receiving_pending_pass {
            self.pending_pass
                .as_ref()
                .map(|pass| pass.off_target_yards)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let pending_pass_receiver_urgency = if receiving_pending_pass {
            self.pending_pass
                .as_ref()
                .map(|pass| pass.receiver_urgency)
                .unwrap_or(0.0)
        } else {
            0.0
        };
        let first_time_shot_score = if first_touch_available {
            first_time_shot_score_for_player(
                me,
                incoming_kind,
                first_time_shot_block_probability,
                (goal.y - me_position.y).abs(),
                self.goal_angle_degrees(me_position, me.team),
                perceived_pressure,
            )
        } else {
            0.0
        };
        let first_time_pass_score = if first_touch_available {
            first_time_pass_score_for_player(me, incoming_kind, perceived_pressure)
        } else {
            0.0
        };
        let control_touch_score = if first_touch_available {
            control_touch_score_for_player(
                me,
                incoming_kind,
                incoming_speed_yps,
                perceived_pressure,
            )
        } else {
            0.0
        };
        let shot_lane_open = shot_block_probability <= 0.34;
        let yards_to_goal = (goal.y - me_position.y).abs();
        let opponent_goal_angle_degrees = self.goal_angle_degrees(me_position, me.team);
        let shot_on_frame_probability = shot_on_frame_probability(
            self.goal_width,
            ability01(me.skills.shooting),
            perceived_pressure,
            yards_to_goal,
            opponent_goal_angle_degrees,
            shot_block_probability,
        );
        let shot_beat_goalkeeper_probability =
            (shot_beat_goalkeeper_probability_for_snapshot(self, me, me_position)
                * (1.0 - shot_block_probability * 0.70).clamp(0.20, 1.0)
                + shot_curl_probability * 0.045)
                .clamp(0.0, 1.0);
        let immediate_dispossession_risk = if has_ball {
            immediate_dispossession_risk_for_player(
                me,
                nearest_opponent_distance,
                perceived_pressure,
                perceived_time_on_ball_seconds,
            )
        } else {
            0.0
        };
        SoccerPomdpObservation {
            player_id,
            player_grid,
            receive_facing: me.receive_facing,
            action_facing: if action_facing == FacingBucket::Unknown {
                me.action_facing
            } else {
                action_facing
            },
            has_ball,
            visible_ball,
            visible_teammates,
            visible_opponents,
            visible_pass_options: visible_pass_targets.len(),
            visible_aerial_pass_options: visible_aerial_pass_targets.len(),
            floor_pass_lane_score,
            aerial_pass_bypass_score,
            aerial_pass_interception_risk,
            ball_position_confidence,
            teammate_position_confidence,
            opponent_position_confidence,
            player_position_confidences,
            ball_distance: if visible_ball {
                me_position.distance(self.ball.position)
            } else {
                vision_range + 12.0
            },
            nearest_opponent_distance,
            nearest_teammate_distance,
            team_spacing_score,
            preferred_team_spacing_yards,
            shot_lane_open,
            shot_block_probability,
            shot_blocker_distance_yards,
            shot_on_frame_probability,
            shot_beat_goalkeeper_probability,
            shot_curl_probability,
            pass_curl_probability,
            immediate_dispossession_risk,
            yards_to_goal,
            yards_to_own_goal: (own_goal.y - me_position.y).abs(),
            opponent_goal_angle_degrees,
            opposing_goalkeeper_distance: opposing_keeper_position
                .map(|position| me_position.distance(position))
                .unwrap_or(self.field_length),
            opposing_goalkeeper_angle_degrees: opposing_keeper_position
                .map(|position| {
                    angle_between_vectors_degrees(goal - me_position, position - me_position)
                })
                .unwrap_or(0.0),
            forward_dribble_space_yards: self.forward_dribble_space_yards(player_id),
            real_pressure,
            perceived_pressure,
            real_time_on_ball_seconds,
            perceived_time_on_ball_seconds,
            fatigue: me.fatigue.clamp(0.0, 1.0),
            nearest_defender_fatigue,
            perceived_nearest_defender_fatigue,
            nearest_defender_fatigue_confidence,
            perceived_fatigue_advantage,
            first_touch_available,
            incoming_ball_kind: incoming_kind,
            incoming_ball_speed_yps: incoming_speed_yps,
            incoming_ball_distance_yards: incoming_distance_yards,
            receiving_pending_pass,
            pending_pass_off_target_yards,
            pending_pass_receiver_urgency,
            first_time_shot_score,
            first_time_pass_score,
            control_touch_score,
            skill_top_speed: me.skills.top_speed,
            skill_acceleration: me.skills.acceleration,
            skill_stamina: me.skills.stamina,
            skill_strength: me.skills.strength,
            skill_height: me.skills.height,
            skill_dribbling: me.skills.dribbling,
            skill_aggression: me.skills.aggression,
            skill_defending: me.skills.defending,
            skill_right_foot_shot_power: me.skills.right_foot_shot_power,
            skill_left_foot_shot_power: me.skills.left_foot_shot_power,
            skill_passing_completion_rate: me.skills.passing_completion_rate,
            skill_flair_passing: me.skills.flair_passing,
            skill_crossing_left: me.skills.crossing_left,
            skill_crossing_right: me.skills.crossing_right,
            skill_goalkeeping: me.skills.goalkeeping,
            skill_defensive_tracking: me.skills.defensive_tracking,
            open_space_score: self.space_score_at(me_position, me.team),
        }
    }

    pub fn best_pass_target(&self, player_id: usize) -> Option<usize> {
        self.ranked_pass_targets(player_id, 1).into_iter().next()
    }

    pub fn best_visible_pass_target(&self, player_id: usize) -> Option<usize> {
        self.ranked_visible_pass_targets(player_id, 1)
            .into_iter()
            .next()
    }

    pub fn best_aerial_pass_target(&self, player_id: usize) -> Option<usize> {
        self.ranked_visible_aerial_pass_targets(player_id, 1)
            .into_iter()
            .next()
            .or_else(|| self.best_pass_target(player_id))
    }

    pub fn ranked_pass_targets(&self, player_id: usize, limit: usize) -> Vec<usize> {
        self.ranked_pass_targets_filtered(player_id, limit, false)
    }

    pub fn ranked_visible_pass_targets(&self, player_id: usize, limit: usize) -> Vec<usize> {
        self.ranked_pass_targets_filtered(player_id, limit, true)
    }

    pub fn ranked_visible_aerial_pass_targets(&self, player_id: usize, limit: usize) -> Vec<usize> {
        self.ranked_aerial_pass_targets_filtered(player_id, limit, true)
    }

    fn ranked_pass_targets_filtered(
        &self,
        player_id: usize,
        limit: usize,
        visible_only: bool,
    ) -> Vec<usize> {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return Vec::new();
        };
        let me_position = self.player_position(me.id).unwrap_or(me.position);
        let directive = self.tactical_directive(me.team);
        let no_pressure = self.no_pressure_at(me.team, me_position);
        let mut ranked = self
            .players
            .iter()
            .filter(|p| p.team == me.team && p.id != me.id)
            .filter_map(|p| {
                let position = self.player_position(p.id).unwrap_or(p.position);
                let forward = (position.y - me_position.y) * me.team.attack_dir();
                if no_pressure && forward < -1.25 {
                    return None;
                }
                ((!visible_only || self.player_can_see_player(me.id, p.id))
                    && self.clear_line(me_position, position, me.team.other(), 2.5)
                    && self.pending_offside_for_pass(me.id, p.id).is_none())
                .then_some((p, position))
            })
            .map(|p| {
                let (p, position) = p;
                let forward = (position.y - me_position.y) * me.team.attack_dir();
                let dist = me_position.distance(position);
                let support_fit = (dist - directive.support_depth_yards).abs();
                let confidence = self
                    .player_position_confidence_for_point(me.id, position)
                    .unwrap_or(0.0);
                let own_half = pass_origin_in_own_half(me.team, me_position, self.field_length);
                let backward = forward < -1.25;
                let blind_backward_penalty = if backward && own_half {
                    2.4 + (1.0 - confidence) * 3.4
                } else if backward {
                    0.8 + (1.0 - confidence) * 1.2
                } else {
                    0.0
                };
                let forward_weight = 0.08 + directive.risk_tolerance * 0.15;
                let role_bonus = match p.role {
                    PlayerRole::Forward => 1.4,
                    PlayerRole::Midfielder => 0.8,
                    PlayerRole::Defender => 0.1,
                    PlayerRole::Goalkeeper => -3.0,
                };
                let finishing_window_bonus = self.shooting_window_score_at(p, position)
                    * (5.4 + directive.risk_tolerance * 2.4);
                let score = forward * forward_weight + self.space_score_at(position, me.team)
                    - dist * 0.010
                    - support_fit * 0.020
                    + confidence * 0.65
                    + role_bonus
                    + finishing_window_bonus;
                let score = score - blind_backward_penalty;
                (p.id, score)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.into_iter().take(limit).map(|(id, _)| id).collect()
    }

    fn ranked_aerial_pass_targets_filtered(
        &self,
        player_id: usize,
        limit: usize,
        visible_only: bool,
    ) -> Vec<usize> {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return Vec::new();
        };
        let me_position = self.player_position(me.id).unwrap_or(me.position);
        let directive = self.tactical_directive(me.team);
        let no_pressure = self.no_pressure_at(me.team, me_position);
        let mut ranked = self
            .players
            .iter()
            .filter(|p| p.team == me.team && p.id != me.id)
            .filter_map(|p| {
                let position = self.player_position(p.id).unwrap_or(p.position);
                let forward = (position.y - me_position.y) * me.team.attack_dir();
                if no_pressure && forward < -1.25 {
                    return None;
                }
                ((!visible_only || self.player_can_see_player(me.id, p.id))
                    && self.pending_offside_for_pass(me.id, p.id).is_none())
                .then_some((p, position))
            })
            .map(|(p, position)| {
                let forward = (position.y - me_position.y) * me.team.attack_dir();
                let dist = me_position.distance(position);
                let confidence = self
                    .player_position_confidence_for_point(me.id, position)
                    .unwrap_or(0.0);
                let is_cross = pass_would_be_cross(
                    me_position,
                    position,
                    me.team,
                    self.field_width,
                    self.field_length,
                );
                let aerial_target_bonus = ability01(p.skills.height) * 1.4
                    + ability01(p.skills.strength) * 0.8
                    + ability01(p.skills.first_touch) * 0.6;
                let lane_bonus = if self.clear_line(me_position, position, me.team.other(), 2.5) {
                    0.4
                } else {
                    1.8
                };
                let cross_bonus = if is_cross { 1.3 } else { 0.0 };
                let finishing_window_bonus = self.shooting_window_score_at(p, position)
                    * (3.4 + directive.risk_tolerance * 1.8);
                let in_behind_bonus = self
                    .projected_in_behind_pass_point(me.id, p.id)
                    .map(|pass_point| 2.4 + self.space_score_at(pass_point, me.team) * 0.045)
                    .unwrap_or(0.0);
                let blind_backward_penalty = if forward < -1.25
                    && pass_origin_in_own_half(me.team, me_position, self.field_length)
                {
                    1.6 + (1.0 - confidence) * 2.0
                } else if forward < -1.25 {
                    0.5 + (1.0 - confidence) * 0.8
                } else {
                    0.0
                };
                let score = forward * (0.10 + directive.risk_tolerance * 0.18)
                    + self.space_score_at(position, me.team) * 0.65
                    - dist * 0.018
                    + confidence * 0.42
                    + aerial_target_bonus
                    + lane_bonus
                    + cross_bonus
                    + finishing_window_bonus
                    + in_behind_bonus
                    - blind_backward_penalty;
                (p.id, score)
            })
            .collect::<Vec<_>>();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.into_iter().take(limit).map(|(id, _)| id).collect()
    }

    pub fn player_can_see_player(&self, observer_id: usize, target_id: usize) -> bool {
        if observer_id == target_id {
            return true;
        }
        self.player_position(target_id)
            .map(|position| self.player_can_see_point(observer_id, position))
            .unwrap_or(false)
    }

    pub fn player_can_see_point(&self, observer_id: usize, point: Vec2) -> bool {
        let Some(observer) = self.players.iter().find(|p| p.id == observer_id) else {
            return false;
        };
        let observer_position = self
            .player_position(observer.id)
            .unwrap_or(observer.position);
        let to_point = point - observer_position;
        let distance = to_point.len();
        if distance <= PLAYER_CONTROL_RADIUS_YARDS * 3.0 {
            return true;
        }
        let vision_range = player_vision_range(observer);
        if distance > vision_range {
            return false;
        }
        let facing = self.player_facing_direction(observer);
        let Some(facing) = facing else {
            return true;
        };
        let fov = player_field_of_view(observer);
        let half_fov_cos = (fov.to_radians() * 0.5).cos();
        facing.dot(to_point.normalized()) >= half_fov_cos
    }

    pub fn player_position_confidence_for_point(
        &self,
        observer_id: usize,
        point: Vec2,
    ) -> Option<f64> {
        let observer = self.players.iter().find(|p| p.id == observer_id)?;
        let observer_position = self
            .player_position(observer.id)
            .unwrap_or(observer.position);
        Some(position_confidence_for_observer(
            observer,
            observer_position,
            point,
            self.player_facing_direction(observer),
        ))
    }

    pub fn player_position_confidence_entry(
        &self,
        observer_id: usize,
        target: &PlayerSnapshot,
    ) -> Option<PlayerPositionConfidence> {
        let observer = self.players.iter().find(|p| p.id == observer_id)?;
        let observer_position = self
            .player_position(observer.id)
            .unwrap_or(observer.position);
        let target_position = self.player_position(target.id).unwrap_or(target.position);
        let to_target = target_position - observer_position;
        let facing = self.player_facing_direction(observer);
        let in_front = facing
            .map(|facing| facing.normalized().dot(to_target.normalized()) >= 0.0)
            .unwrap_or(false);
        Some(PlayerPositionConfidence {
            observer_id,
            player_id: target.id,
            team: target.team,
            distance_yards: to_target.len(),
            in_front,
            confidence: position_confidence_for_observer(
                observer,
                observer_position,
                target_position,
                facing,
            ),
        })
    }

    fn player_facing_direction(&self, player: &PlayerSnapshot) -> Option<Vec2> {
        if player.velocity.len() > 0.35 {
            return Some(player.velocity.normalized());
        }
        if self.ball.holder == Some(player.id) {
            return Some(Vec2::new(0.0, player.team.attack_dir()));
        }
        let player_position = self.player_position(player.id).unwrap_or(player.position);
        let to_ball = self.ball.position - player_position;
        if to_ball.len() > PLAYER_CONTROL_RADIUS_YARDS {
            Some(to_ball.normalized())
        } else {
            Some(Vec2::new(0.0, player.team.attack_dir()))
        }
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

    fn second_last_defender_line_for(&self, attacking_team: Team) -> Option<f64> {
        let mut defender_ys = self
            .players
            .iter()
            .filter(|p| p.team == attacking_team.other())
            .filter_map(|p| self.player_position(p.id).map(|position| position.y))
            .collect::<Vec<_>>();
        if defender_ys.len() < 2 {
            return None;
        }
        match attacking_team {
            Team::Home => {
                defender_ys.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            }
            Team::Away => {
                defender_ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            }
        }
        defender_ys.get(1).copied()
    }

    fn neutral_in_behind_window(&self, team: Team) -> bool {
        let progress_from_midfield =
            (self.ball.position.y - self.field_length * 0.5) * team.attack_dir();
        (-18.0..=28.0).contains(&progress_from_midfield)
    }

    fn in_behind_run_target_for(&self, player_id: usize) -> Option<Vec2> {
        let me = self.players.iter().find(|p| p.id == player_id)?;
        if self.possession_team() != Some(me.team)
            || self.ball.holder == Some(player_id)
            || !matches!(me.role, PlayerRole::Forward | PlayerRole::Midfielder)
            || !self.neutral_in_behind_window(me.team)
        {
            return None;
        }
        let cadence = (self.tick + player_id as u64 * 17) % 41;
        if cadence > 5 {
            return None;
        }
        let current = self.player_position(me.id).unwrap_or(me.position);
        let line_y = self.second_last_defender_line_for(me.team)?;
        let staging_y = line_y - me.team.attack_dir() * 20.0;
        if (current.y - staging_y).abs() > 14.0 {
            return None;
        }
        let holder_position = self
            .ball
            .holder
            .and_then(|holder| self.player_position(holder))
            .unwrap_or(self.ball.position);
        let run_y = line_y + me.team.attack_dir() * 9.0;
        let target = Vec2::new(
            (current.x * 0.76 + holder_position.x * 0.24).clamp(4.0, self.field_width - 4.0),
            run_y,
        )
        .clamp_to_pitch(self.field_width, self.field_length);
        Some(target)
    }

    fn projected_in_behind_pass_point(&self, passer_id: usize, target_id: usize) -> Option<Vec2> {
        let passer = self.players.iter().find(|p| p.id == passer_id)?;
        let target = self.players.iter().find(|p| p.id == target_id)?;
        if passer.team != target.team
            || passer.id == target.id
            || self
                .pending_offside_for_pass(passer_id, target_id)
                .is_some()
            || !self.neutral_in_behind_window(passer.team)
        {
            return None;
        }
        let target_position = self.player_position(target.id).unwrap_or(target.position);
        let line_y = self.second_last_defender_line_for(passer.team)?;
        let staging_y = line_y - passer.team.attack_dir() * 20.0;
        if (target_position.y - staging_y).abs() > 16.0 {
            return None;
        }
        let projected_y = line_y + passer.team.attack_dir() * 11.0;
        Some(
            Vec2::new(target_position.x, projected_y)
                .clamp_to_pitch(self.field_width, self.field_length),
        )
    }

    fn long_ball_in_behind_target(&self, passer_id: usize) -> Option<usize> {
        let passer = self.players.iter().find(|p| p.id == passer_id)?;
        if self.ball.holder != Some(passer_id)
            || self.possession_team() != Some(passer.team)
            || !self.neutral_in_behind_window(passer.team)
        {
            return None;
        }
        let passer_position = self.player_position(passer.id).unwrap_or(passer.position);
        self.players
            .iter()
            .filter(|p| p.team == passer.team && p.id != passer.id)
            .filter(|p| matches!(p.role, PlayerRole::Forward | PlayerRole::Midfielder))
            .filter_map(|target| {
                let pass_point = self.projected_in_behind_pass_point(passer.id, target.id)?;
                let target_position = self.player_position(target.id).unwrap_or(target.position);
                let forward = (pass_point.y - passer_position.y) * passer.team.attack_dir();
                if forward <= 12.0 {
                    return None;
                }
                let line_bonus = if self.clear_line(
                    passer_position,
                    target_position,
                    passer.team.other(),
                    2.5,
                ) {
                    0.0
                } else {
                    1.2
                };
                let score = forward * 0.10 + self.space_score_at(pass_point, passer.team) * 0.06
                    - passer_position.distance(pass_point) * 0.010
                    + line_bonus;
                Some((target.id, score))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(id, _)| id)
    }

    fn check_to_ball_target_for(&self, player_id: usize, home: Vec2) -> Option<Vec2> {
        let me = self.players.iter().find(|p| p.id == player_id)?;
        if self.possession_team() != Some(me.team)
            || self.ball.holder == Some(player_id)
            || me.role == PlayerRole::Goalkeeper
        {
            return None;
        }
        let current = self.player_position(me.id).unwrap_or(me.position);
        let marker = self.nearest_opponent_at(me.team, current)?;
        if marker.2 > 4.5 {
            return None;
        }
        let toward_ball = (self.ball.position - current).normalized();
        let toward_own_goal = (Vec2::new(
            self.field_width * 0.5,
            me.team.other().goal_y(self.field_length),
        ) - current)
            .normalized();
        let lateral_away = (current - marker.1).normalized();
        let candidate = (current + toward_ball * 4.8 + toward_own_goal * 3.8 + lateral_away * 1.6)
            .clamp_to_pitch(self.field_width, self.field_length);
        if candidate.distance(self.ball.position) >= current.distance(self.ball.position) {
            return None;
        }
        let candidate_pressure = self.nearest_opponent_distance_at(me.team, candidate);
        if candidate_pressure <= marker.2 + 0.8 && candidate_pressure < 5.2 {
            return None;
        }
        if self.position_would_be_offside(me.team, candidate)
            || !self.clear_line(self.ball.position, candidate, me.team.other(), 2.0)
        {
            return None;
        }
        Some(self.clamp_to_role_position(player_id, candidate, home, false))
    }

    fn pending_pass_reception_target_for(&self, player_id: usize) -> Option<(Vec2, bool)> {
        let me = self.players.iter().find(|p| p.id == player_id)?;
        let pass = self.pending_pass.as_ref()?;
        if self.ball.holder.is_some()
            || pass.team != me.team
            || pass.nearest_receiver != Some(player_id)
        {
            return None;
        }
        let current = self.player_position(me.id).unwrap_or(me.position);
        let distance_to_ball = current.distance(self.ball.position);
        if distance_to_ball > 42.0 {
            return None;
        }
        let lead_seconds = (0.18 + distance_to_ball / 42.0 * 0.42).clamp(0.18, 0.60);
        let target = (self.ball.position + self.ball.velocity * lead_seconds)
            .clamp_to_pitch(self.field_width, self.field_length);
        let sprint = pass.receiver_urgency >= 0.42
            || pass.off_target_yards > 1.25
            || distance_to_ball > 7.0
            || self.nearest_opponent_distance_at(me.team, self.ball.position)
                <= distance_to_ball + 2.0;
        Some((target, sprint))
    }

    pub fn open_space_for(&self, player_id: usize, home: Vec2) -> Vec2 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return home;
        };
        let me_position = self.player_position(me.id).unwrap_or(me.position);
        let directive = self.tactical_directive(me.team);
        let width_scale = (directive.width_yards / (self.field_width * 0.62)).clamp(0.65, 1.35);
        let depth_scale = (directive.support_depth_yards / 11.0).clamp(0.65, 1.65);
        let possession = self.possession_team() == Some(me.team);
        let own_half_possession =
            possession && pass_origin_in_own_half(me.team, self.ball.position, self.field_length);
        let role_depth = match me.role {
            PlayerRole::Goalkeeper => -8.0,
            PlayerRole::Defender if own_half_possession => 7.0,
            PlayerRole::Defender => 4.0,
            PlayerRole::Midfielder if own_half_possession => directive.support_depth_yards + 4.0,
            PlayerRole::Midfielder => directive.support_depth_yards,
            PlayerRole::Forward if own_half_possession => directive.support_depth_yards + 10.0,
            PlayerRole::Forward => directive.support_depth_yards + 7.0,
        };
        let base = if possession {
            Vec2::new(
                home.x * 0.58 + self.ball.position.x * 0.42,
                self.ball.position.y + role_depth * me.team.attack_dir(),
            )
        } else {
            home
        };
        let mut best = base.clamp_to_pitch(self.field_width, self.field_length);
        let mut best_score = f64::NEG_INFINITY;
        for dx in [-22.0, -13.0, -6.0, 0.0, 6.0, 13.0, 22.0] {
            for dy in [-8.0, 0.0, 7.0, 14.0, 22.0, 30.0] {
                let p = Vec2::new(
                    base.x + dx * width_scale,
                    base.y + dy * depth_scale * me.team.attack_dir(),
                )
                .clamp_to_pitch(self.field_width, self.field_length);
                let forward = (p.y - me_position.y) * me.team.attack_dir();
                let forward_from_ball = (p.y - self.ball.position.y) * me.team.attack_dir();
                let lane_bonus = if self.clear_line(self.ball.position, p, me.team.other(), 2.0) {
                    1.8
                } else {
                    -1.2
                };
                let offside_penalty = if self.position_would_be_offside(me.team, p) {
                    14.0
                } else {
                    0.0
                };
                let spacing_bonus = if possession {
                    self.team_spacing_score_for_candidate(
                        me.team,
                        Some(me.id),
                        p,
                        TeamSpacingMode::InPossession,
                    ) * (1.0 + self.possession_spacing_weight(me.team) * 1.5)
                } else {
                    0.0
                };
                let score = self.space_score_at(p, me.team)
                    + forward.max(-4.0) * (0.04 + directive.risk_tolerance * 0.08)
                    + forward_from_ball.clamp(-6.0, 28.0)
                        * (0.08 + directive.risk_tolerance * 0.09)
                    - (forward_from_ball - role_depth).abs() * 0.035
                    - p.distance(home) * 0.010
                    + lane_bonus
                    + spacing_bonus
                    - offside_penalty;
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

    pub fn shot_creation_space_for(&self, player_id: usize, home: Vec2) -> Vec2 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return home;
        };
        let me_position = self.player_position(me.id).unwrap_or(me.position);
        let center_x = self.field_width * 0.5;
        let dir = me.team.attack_dir();
        let base = self.forward_space_for(player_id, home);
        let mut best = base;
        let mut best_score = self.shot_creation_score_for(player_id, me, best, home);

        for dx in [-18.0, -12.0, -7.0, -3.0, 0.0, 3.0, 7.0, 12.0, 18.0] {
            for dy in [4.0, 8.0, 12.0, 16.0, 22.0, 28.0] {
                let raw = Vec2::new(me_position.x + dx, me_position.y + dy * dir);
                let central_pull = if (raw.y - self.field_length * 0.5) * dir > 8.0 {
                    0.34
                } else {
                    0.16
                };
                let candidate = Vec2::new(
                    raw.x * (1.0 - central_pull) + center_x * central_pull,
                    raw.y,
                )
                .clamp_to_pitch(self.field_width, self.field_length);
                let score = self.shot_creation_score_for(player_id, me, candidate, home);
                if score > best_score {
                    best = candidate;
                    best_score = score;
                }
            }
        }
        best.clamp_to_pitch(self.field_width, self.field_length)
    }

    fn shot_creation_score_for(
        &self,
        player_id: usize,
        player: &PlayerSnapshot,
        candidate: Vec2,
        home: Vec2,
    ) -> f64 {
        let current = self.player_position(player.id).unwrap_or(player.position);
        let forward = (candidate.y - current.y) * player.team.attack_dir();
        let pass_lane = if self.clear_line(self.ball.position, candidate, player.team.other(), 2.2)
        {
            0.62
        } else {
            -0.72
        };
        let offside_penalty = if self.ball.holder != Some(player_id)
            && self.position_would_be_offside(player.team, candidate)
        {
            12.0
        } else {
            0.0
        };
        let centrality = (1.0
            - ((candidate.x - self.field_width * 0.5).abs() / (self.field_width * 0.5)))
            .clamp(0.0, 1.0);
        self.shooting_window_score_at(player, candidate) * 5.4
            + self.space_score_at(candidate, player.team) * 0.060
            + forward.clamp(-4.0, 24.0) * 0.090
            + centrality * 0.70
            + pass_lane
            - candidate.distance(current) * 0.022
            - candidate.distance(home) * 0.006
            - offside_penalty
    }

    fn shooting_window_score_at(&self, player: &PlayerSnapshot, position: Vec2) -> f64 {
        let goal = Vec2::new(
            self.field_width * 0.5,
            player.team.goal_y(self.field_length),
        );
        let shot_speed = shot_speed_yps_from_power(
            0.72 + 0.28 * ability01(player.skills.shooting),
            &player.skills,
        );
        let shot_block_probability =
            shot_block_assessment_for_snapshot(self, position, player.team, shot_speed, false)
                .map(|assessment| assessment.probability)
                .unwrap_or(0.0);
        let yards_to_goal = (goal.y - position.y).abs();
        let nearest_opponent_distance = self
            .players
            .iter()
            .filter(|other| other.team == player.team.other())
            .map(|other| {
                self.player_position(other.id)
                    .unwrap_or(other.position)
                    .distance(position)
            })
            .fold(36.0, f64::min)
            .min(36.0);
        let pressure = pressure_from_nearest_distance(nearest_opponent_distance);
        let on_frame = shot_on_frame_probability(
            self.goal_width,
            ability01(player.skills.shooting),
            pressure,
            yards_to_goal,
            self.goal_angle_degrees(position, player.team),
            shot_block_probability,
        );
        let beat_keeper = shot_beat_goalkeeper_probability_for_snapshot(self, player, position)
            * (1.0 - shot_block_probability * 0.70).clamp(0.20, 1.0);
        let centrality = (1.0
            - ((position.x - self.field_width * 0.5).abs() / (self.field_width * 0.5)))
            .clamp(0.0, 1.0);
        let range_shape = (1.0 - (yards_to_goal - 14.0).abs() / 24.0).clamp(0.0, 1.0);
        (on_frame * 0.66 + beat_keeper * 0.38 + centrality * 0.08 + range_shape * 0.07)
            .clamp(0.0, 1.25)
    }

    pub fn forward_dribble_space_yards(&self, player_id: usize) -> f64 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return 0.0;
        };
        let start = self.player_position(me.id).unwrap_or(me.position);
        let dir = Vec2::new(0.0, me.team.attack_dir());
        let max_space = 30.0_f64
            .min(match me.team {
                Team::Home => self.field_length - start.y,
                Team::Away => start.y,
            })
            .max(0.0);
        let mut nearest_block = max_space;
        for opponent in self.players.iter().filter(|p| p.team == me.team.other()) {
            let position = self
                .player_position(opponent.id)
                .unwrap_or(opponent.position);
            let to_opponent = position - start;
            let forward = to_opponent.dot(dir);
            if forward <= 0.0 || forward > max_space {
                continue;
            }
            let lateral = (to_opponent - dir * forward).len();
            if lateral <= 4.0 {
                nearest_block = nearest_block.min((forward - PLAYER_CONTROL_RADIUS_YARDS).max(0.0));
            }
        }
        nearest_block
    }

    pub fn goal_angle_degrees(&self, position: Vec2, attacking_team: Team) -> f64 {
        let goal_y = attacking_team.goal_y(self.field_length);
        let left_post = Vec2::new(self.field_width * 0.5 - self.goal_width * 0.5, goal_y);
        let right_post = Vec2::new(self.field_width * 0.5 + self.goal_width * 0.5, goal_y);
        angle_between_vectors_degrees(left_post - position, right_post - position)
    }

    pub fn positional_open_space_for(&self, player_id: usize, home: Vec2, roam: bool) -> Vec2 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return home;
        };
        if self.possession_team() == Some(me.team) && !roam {
            if let Some(target) = self.check_to_ball_target_for(player_id, home) {
                return target;
            }
            if let Some(target) = self.in_behind_run_target_for(player_id) {
                return self.clamp_to_role_position(player_id, target, home, false);
            }
        }
        let open = self.open_space_for(player_id, home);
        if roam {
            return open;
        }
        let own_half_possession = self.possession_team() == Some(me.team)
            && pass_origin_in_own_half(me.team, self.ball.position, self.field_length);
        let shape = if self.possession_team() == Some(me.team) {
            let support_y = self.ball.position.y
                - me.team.attack_dir()
                    * match me.role {
                        PlayerRole::Goalkeeper if own_half_possession => 24.0,
                        PlayerRole::Goalkeeper => 30.0,
                        PlayerRole::Defender if own_half_possession => 9.0,
                        PlayerRole::Defender => 18.0,
                        PlayerRole::Midfielder if own_half_possession => -4.0,
                        PlayerRole::Midfielder => 8.0,
                        PlayerRole::Forward if own_half_possession => -14.0,
                        PlayerRole::Forward => -8.0,
                    };
            Vec2::new(home.x * 0.72 + self.ball.position.x * 0.28, support_y)
                .clamp_to_pitch(self.field_width, self.field_length)
        } else {
            self.defensive_shape_for(player_id, home)
        };
        let attack_depth = (self.ball.position.y - self.field_length * 0.5) * me.team.attack_dir();
        let target = if self.possession_team() == Some(me.team)
            && attack_depth > -4.0
            && matches!(me.role, PlayerRole::Forward | PlayerRole::Midfielder)
        {
            let finish = self.shot_creation_space_for(player_id, home);
            open * 0.34 + shape * 0.20 + finish * 0.36 + home * 0.10
        } else if own_half_possession && me.role != PlayerRole::Goalkeeper {
            open * 0.64 + shape * 0.26 + home * 0.10
        } else {
            open * 0.55 + shape * 0.30 + home * 0.15
        };
        self.clamp_to_role_position(player_id, target, home, false)
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

    pub fn mark_or_zone_for(&self, player_id: usize, home: Vec2) -> Vec2 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return home;
        };
        if me.role == PlayerRole::Goalkeeper {
            return self.defensive_shape_for(player_id, home);
        }

        let zone = self.defensive_shape_for(player_id, home);
        let own_goal = Vec2::new(
            self.field_width * 0.5,
            me.team.other().goal_y(self.field_length),
        );
        let mut best_mark = None;
        let mut best_score = f64::NEG_INFINITY;
        for opponent in self.players.iter().filter(|p| p.team == me.team.other()) {
            let opponent_position = self
                .player_position(opponent.id)
                .unwrap_or(opponent.position);
            let distance_to_goal = opponent_position.distance(own_goal);
            let ball_distance = opponent_position.distance(self.ball.position);
            let zone_distance = opponent_position.distance(zone);
            let role_threat = match opponent.role {
                PlayerRole::Forward => 7.0,
                PlayerRole::Midfielder => 4.0,
                PlayerRole::Defender => 1.5,
                PlayerRole::Goalkeeper => -8.0,
            };
            let holder_bonus = if self.ball.holder == Some(opponent.id) {
                9.0
            } else {
                0.0
            };
            let lane_threat = if self.clear_line(opponent_position, own_goal, me.team, 2.4) {
                4.0
            } else {
                0.0
            };
            let score = (70.0 - distance_to_goal).max(0.0) * 0.18
                + (32.0 - ball_distance).max(0.0) * 0.15
                - zone_distance * 0.055
                + role_threat
                + holder_bonus
                + lane_threat;
            if score > best_score {
                best_mark = Some(opponent_position);
                best_score = score;
            }
        }

        let Some(mark) = best_mark else {
            return zone;
        };
        let goal_side = (own_goal - mark).normalized();
        let mark_distance = match me.role {
            PlayerRole::Defender => 2.4,
            PlayerRole::Midfielder => 3.4,
            PlayerRole::Forward => 4.8,
            PlayerRole::Goalkeeper => 1.0,
        };
        let mark_target =
            (mark + goal_side * mark_distance).clamp_to_pitch(self.field_width, self.field_length);
        (mark_target * 0.72 + zone * 0.28).clamp_to_pitch(self.field_width, self.field_length)
    }

    pub fn defensive_assignment_for(&self, player_id: usize, home: Vec2, roam: bool) -> Vec2 {
        let zone = self.defensive_shape_for(player_id, home);
        let mark = self.mark_or_zone_for(player_id, home);
        let blended = if roam {
            mark * 0.78 + zone * 0.22
        } else {
            mark * 0.46 + zone * 0.54
        };
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return self.clamp_to_role_position(player_id, blended, home, roam);
        };
        let mut best = blended;
        let mut best_score = f64::NEG_INFINITY;
        for dx in [-5.0, -2.5, 0.0, 2.5, 5.0] {
            for dy in [-4.0, -2.0, 0.0, 2.0, 4.0] {
                let candidate = (blended + Vec2::new(dx, dy))
                    .clamp_to_pitch(self.field_width, self.field_length);
                let compact_score = self.team_spacing_score_for_candidate(
                    me.team,
                    Some(player_id),
                    candidate,
                    TeamSpacingMode::Defending,
                );
                let score = compact_score * 0.45
                    - candidate.distance(blended) * 0.055
                    - candidate.distance(mark) * 0.025;
                if score > best_score {
                    best = candidate;
                    best_score = score;
                }
            }
        }
        self.clamp_to_role_position(player_id, best, home, roam)
    }

    fn clamp_to_role_position(
        &self,
        player_id: usize,
        target: Vec2,
        home: Vec2,
        roam: bool,
    ) -> Vec2 {
        let Some(me) = self.players.iter().find(|p| p.id == player_id) else {
            return home;
        };
        if roam {
            return target.clamp_to_pitch(self.field_width, self.field_length);
        }
        let radius = match me.role {
            PlayerRole::Goalkeeper => 7.0,
            PlayerRole::Defender => 13.0,
            PlayerRole::Midfielder => 18.0,
            PlayerRole::Forward => 20.0,
        };
        let delta = target - home;
        let bounded = if delta.len() <= radius {
            target.clamp_to_pitch(self.field_width, self.field_length)
        } else {
            (home + delta.normalized() * radius).clamp_to_pitch(self.field_width, self.field_length)
        };
        let bounded = self.clamp_forward_onside_support(me, bounded);
        if self.possession_team() == Some(me.team.other()) && me.role != PlayerRole::Goalkeeper {
            self.clamp_defensive_goal_line_and_ball_gap(me.team, bounded)
        } else {
            bounded
        }
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

    fn shot_lane_clear(&self, from: Vec2, attacking_team: Team, radius: f64) -> bool {
        let speed = mph_to_yps(60.0);
        let threshold = if radius <= 2.0 { 0.18 } else { 0.34 };
        shot_block_assessment_for_snapshot(self, from, attacking_team, speed, false)
            .map(|assessment| assessment.probability <= threshold)
            .unwrap_or(true)
    }

    fn position_would_be_offside(&self, team: Team, position: Vec2) -> bool {
        let half_line = self.field_length * 0.5;
        match team {
            Team::Home if position.y <= half_line => return false,
            Team::Away if position.y >= half_line => return false,
            _ => {}
        }

        let mut defender_ys = self
            .players
            .iter()
            .filter(|p| p.team == team.other())
            .filter_map(|p| self.player_position(p.id).map(|position| position.y))
            .collect::<Vec<_>>();
        if defender_ys.len() < 2 {
            return false;
        }

        match team {
            Team::Home => {
                defender_ys.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                position.y > self.ball.position.y && position.y > defender_ys[1]
            }
            Team::Away => {
                defender_ys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                position.y < self.ball.position.y && position.y < defender_ys[1]
            }
        }
    }
}

fn sample_defensive_cover_target(rng: &mut SeededRandom) -> usize {
    let roll = rng.next_float();
    if roll < 0.10 {
        0
    } else if roll < 0.20 {
        1
    } else if roll < 0.50 {
        2
    } else if roll < 0.80 {
        3
    } else {
        4
    }
}

fn defensive_cover_profile(
    snapshot: &WorldSnapshot,
    team: Team,
    target: usize,
) -> DefensiveCoverProfile {
    let foremost_attacker_y = snapshot
        .players
        .iter()
        .filter(|player| player.team == team.other() && player.role != PlayerRole::Goalkeeper)
        .filter_map(|player| {
            snapshot
                .player_position(player.id)
                .map(|position| position.y)
        })
        .reduce(|a, b| match team {
            Team::Home => a.min(b),
            Team::Away => a.max(b),
        });

    let actual = foremost_attacker_y
        .map(|attacker_y| {
            snapshot
                .players
                .iter()
                .filter(|player| player.team == team && player.role != PlayerRole::Goalkeeper)
                .filter_map(|player| snapshot.player_position(player.id))
                .filter(|position| match team {
                    Team::Home => position.y <= attacker_y,
                    Team::Away => position.y >= attacker_y,
                })
                .count()
        })
        .unwrap_or(0);

    DefensiveCoverProfile {
        target,
        actual,
        foremost_attacker_y,
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
    defensive_cover: DefensiveCoverProfile,
) -> TeamTacticalDirective {
    let has_ball = possession_team == Some(team);
    let defending = possession_team == Some(team.other());
    let attacking_phase = matches!(
        (team, phase),
        (Team::Home, TacticalPhase::HomeAttack) | (Team::Away, TacticalPhase::AwayAttack)
    );
    let own_half_possession =
        has_ball && pass_origin_in_own_half(team, ball_position, field_length);
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
    let mut line_seed = if has_ball {
        let holding_distance = if own_half_possession {
            14.0
        } else if attacking_phase {
            18.0
        } else {
            25.0
        };
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
    if defending {
        if let Some(foremost_attacker_y) = defensive_cover.foremost_attacker_y {
            let goal_side_dir = -attack_dir;
            let cover_gap = match defensive_cover.target {
                0 => -2.0,
                1 => 3.5,
                2 => 6.5,
                3 => 9.5,
                _ => 12.5,
            };
            let shortage = defensive_cover
                .target
                .saturating_sub(defensive_cover.actual) as f64;
            let surplus = defensive_cover
                .actual
                .saturating_sub(defensive_cover.target + 1) as f64;
            let adjusted_gap = cover_gap + shortage * 2.0 - surplus * 1.0;
            let cover_line_seed = foremost_attacker_y + goal_side_dir * adjusted_gap;
            line_seed = line_seed * 0.32 + cover_line_seed * 0.68;
        }
    }
    let defensive_line_y: f64 = line_seed.clamp(field_length * 0.08, field_length * 0.92);

    let risk_tolerance: f64 = (0.46
        + if attacking_phase { 0.15 } else { 0.0 }
        + if build_up_phase { 0.04 } else { 0.0 }
        + if own_half_possession { 0.11 } else { 0.0 }
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
        } else if own_half_possession {
            1.16
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
    } + risk_tolerance * 3.0
        + if own_half_possession { 4.5 } else { 0.0 })
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
        defensive_cover_target: defensive_cover.target,
        defensive_cover_actual: defensive_cover.actual,
        foremost_attacker_y: defensive_cover.foremost_attacker_y,
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
    let pressure = obs.perceived_pressure;
    let shot_quality =
        if obs.visible_ball && obs.shot_block_probability <= SHOT_BLOCK_BAILOUT_MAX_PROBABILITY {
            let block_factor = (1.0 - obs.shot_block_probability * 0.55).clamp(0.25, 1.0);
            ((obs.shot_on_frame_probability * 0.64
                + obs.shot_beat_goalkeeper_probability * 0.30
                + obs.shot_curl_probability * 0.08
                + (1.0 - obs.immediate_dispossession_risk) * 0.06)
                * block_factor)
                .clamp(0.0, 1.0)
        } else {
            0.0
        };
    let visible_option_confidence = (obs.visible_pass_options as f64 / 3.0).clamp(0.0, 1.0);
    BeliefState {
        possession_confidence: if obs.has_ball {
            0.98
        } else if obs.visible_ball {
            0.62
        } else {
            0.28
        },
        pressure,
        pass_lane_open: (visible_option_confidence * 0.70
            + (obs.nearest_teammate_distance / 24.0).clamp(0.0, 1.0) * 0.30)
            .clamp(0.0, 1.0),
        shot_quality,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PassDirectionBucket {
    Forward,
    Lateral,
    Backward,
}

fn pass_origin_in_own_half(team: Team, origin: Vec2, field_length: f64) -> bool {
    match team {
        Team::Home => origin.y < field_length * 0.5,
        Team::Away => origin.y > field_length * 0.5,
    }
}

fn pass_direction_bucket(team: Team, origin: Vec2, target: Vec2) -> PassDirectionBucket {
    let forward = (target.y - origin.y) * team.attack_dir();
    if forward > 1.25 {
        PassDirectionBucket::Forward
    } else if forward < -1.25 {
        PassDirectionBucket::Backward
    } else {
        PassDirectionBucket::Lateral
    }
}

fn completed_pass_reward(team: Team, origin: Vec2, target: Vec2, field_length: f64) -> f64 {
    match (
        pass_direction_bucket(team, origin, target),
        pass_origin_in_own_half(team, origin, field_length),
    ) {
        (PassDirectionBucket::Forward, true) => 5.0,
        (PassDirectionBucket::Forward, false) => 6.0,
        (PassDirectionBucket::Lateral, _) => 3.0,
        (PassDirectionBucket::Backward, true) => 0.2,
        (PassDirectionBucket::Backward, false) => 1.4,
    }
}

#[derive(Clone, Debug)]
struct PlayerRewardLoad {
    player_id: usize,
    amount: f64,
    load: f64,
}

#[derive(Clone, Debug)]
struct PossessionChaseSignal {
    possession_team: Team,
    defending_team: Team,
    attacking_load: f64,
    defensive_load: f64,
    attacking_credit: f64,
    defender_penalties: Vec<PlayerRewardLoad>,
}

#[derive(Clone, Debug)]
struct DefensiveRelaxationSignal {
    possession_team: Team,
    defending_team: Team,
    attacking_credit: f64,
    defender_penalties: Vec<PlayerRewardLoad>,
}

fn snapshot_player<'a>(
    snapshot: &'a WorldSnapshot,
    player_id: usize,
) -> Option<&'a PlayerSnapshot> {
    snapshot
        .players
        .iter()
        .find(|player| player.id == player_id)
}

fn player_motion_distance(before: &WorldSnapshot, after: &WorldSnapshot, player_id: usize) -> f64 {
    let Some(before_player) = snapshot_player(before, player_id) else {
        return 0.0;
    };
    let before_position = before
        .player_position(player_id)
        .unwrap_or(before_player.position);
    let after_position = after
        .player_position(player_id)
        .or_else(|| snapshot_player(after, player_id).map(|player| player.position))
        .unwrap_or(before_position);
    before_position.distance(after_position)
}

fn player_motion_load(before: &WorldSnapshot, after: &WorldSnapshot, player_id: usize) -> f64 {
    let distance = player_motion_distance(before, after, player_id);
    if distance <= 1e-9 {
        return 0.0;
    }
    let dt = before.dt_seconds.max(1e-6);
    let speed_yps = distance / dt;
    let acceleration = after
        .player_acceleration(player_id)
        .or_else(|| snapshot_player(after, player_id).map(|player| player.acceleration))
        .unwrap_or_else(Vec2::zero)
        .len();
    let gait_multiplier = snapshot_player(after, player_id)
        .map(|player| match player.movement_gait {
            MovementGait::Sprint => 1.28,
            MovementGait::Run => 1.16,
            MovementGait::Jog => 1.08,
            MovementGait::SideStep | MovementGait::BackSkip | MovementGait::Skip => 1.04,
            MovementGait::Walk | MovementGait::BackWalk => 0.92,
            MovementGait::Stand => 0.70,
        })
        .unwrap_or(1.0);
    let speed_cost = 1.0 + (speed_yps / 8.0).powi(2).clamp(0.0, 1.15) * 0.32;
    distance * gait_multiplier * speed_cost + acceleration * dt * 0.035
}

fn player_normalized_last_action(player: &PlayerSnapshot) -> &str {
    player
        .last_decision
        .as_ref()
        .map(|decision| normalize_soccer_action_label(&decision.action))
        .unwrap_or("hold")
}

fn team_average_defensive_depth(snapshot: &WorldSnapshot, team: Team) -> f64 {
    let own_goal_y = team.other().goal_y(snapshot.field_length);
    let mut total = 0.0;
    let mut count = 0.0;
    for player in snapshot.players.iter().filter(|player| player.team == team) {
        let position = snapshot
            .player_position(player.id)
            .unwrap_or(player.position);
        total += (position.y - own_goal_y).abs();
        count += 1.0;
    }
    if count > 0.0 {
        total / count
    } else {
        snapshot.field_length
    }
}

fn possession_chase_signal(
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    possession_team: Team,
) -> Option<PossessionChaseSignal> {
    if before.controlled_possession_team() != Some(possession_team)
        || after.controlled_possession_team() != Some(possession_team)
    {
        return None;
    }

    let ball_relocation = before.ball.position.distance(after.ball.position);
    if ball_relocation < POSSESSION_CHASE_MIN_BALL_RELOCATION_YARDS {
        return None;
    }

    let defending_team = possession_team.other();
    let mut attacking_load = 0.0;
    let mut defensive_load = 0.0;
    let mut active_defender_loads = Vec::new();
    let mut defender_speed_total = 0.0;
    let mut defender_count = 0.0;

    for player in &before.players {
        let load = player_motion_load(before, after, player.id);
        if player.team == possession_team {
            attacking_load += load;
            continue;
        }
        defensive_load += load;
        defender_count += 1.0;
        defender_speed_total += load / before.dt_seconds.max(1e-6);

        let Some(after_player) = snapshot_player(after, player.id) else {
            continue;
        };
        let action = player_normalized_last_action(after_player);
        let before_position = before.player_position(player.id).unwrap_or(player.position);
        let distance_to_ball = before_position.distance(before.ball.position);
        let moved = player_motion_distance(before, after, player.id);
        let speed_yps = moved / before.dt_seconds.max(1e-6);
        let active_chase = matches!(action, "defend" | "tackle")
            && distance_to_ball <= 38.0
            && (speed_yps >= 1.15 || moved >= 0.45);
        if active_chase {
            active_defender_loads.push((player.id, load));
        }
    }

    if active_defender_loads.len() < POSSESSION_CHASE_MIN_ACTIVE_DEFENDERS {
        return None;
    }

    let average_defender_load_rate = if defender_count > 0.0 {
        defender_speed_total / defender_count
    } else {
        0.0
    };
    let average_defensive_depth = team_average_defensive_depth(before, defending_team);
    let compact_low_block = average_defensive_depth <= 34.0 && average_defender_load_rate < 1.25;
    if compact_low_block {
        return None;
    }

    let load_advantage = defensive_load - attacking_load * 1.08;
    if load_advantage <= 0.0 {
        return None;
    }

    let lateral_switch_bonus =
        1.0 + ((after.ball.position.x - before.ball.position.x).abs() / 26.0).clamp(0.0, 0.45);
    let attacking_credit = (load_advantage / 11.0 * 0.070 * lateral_switch_bonus).clamp(0.0, 0.48);
    if attacking_credit < POSSESSION_CHASE_MIN_CREDIT {
        return None;
    }

    let active_load_total = active_defender_loads
        .iter()
        .map(|(_, load)| *load)
        .sum::<f64>()
        .max(1e-6);
    let defender_penalties = active_defender_loads
        .into_iter()
        .filter_map(|(player_id, load)| {
            let share = load / active_load_total;
            let amount = -(attacking_credit * 0.78 * share).clamp(0.015, 0.16);
            (amount.abs() >= 0.015).then_some(PlayerRewardLoad {
                player_id,
                amount,
                load,
            })
        })
        .collect::<Vec<_>>();

    Some(PossessionChaseSignal {
        possession_team,
        defending_team,
        attacking_load,
        defensive_load,
        attacking_credit,
        defender_penalties,
    })
}

fn defensive_relaxation_signal(
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    possession_team: Team,
) -> Option<DefensiveRelaxationSignal> {
    if before.controlled_possession_team() != Some(possession_team)
        || after.controlled_possession_team() != Some(possession_team)
    {
        return None;
    }

    let ball_relocation = before.ball.position.distance(after.ball.position);
    if ball_relocation < POSSESSION_CHASE_MIN_BALL_RELOCATION_YARDS {
        return None;
    }

    let defending_team = possession_team.other();
    let attacking_goal_y = possession_team.goal_y(before.field_length);
    let yards_to_goal = (attacking_goal_y - after.ball.position.y).abs();
    let ball_progress =
        (after.ball.position.y - before.ball.position.y) * possession_team.attack_dir();
    let lateral_shift = (after.ball.position.x - before.ball.position.x).abs();
    let threat_increased = yards_to_goal <= DEFENSIVE_RELAXATION_THREAT_YARDS
        || ball_progress > 0.80
        || lateral_shift > 5.0;
    if !threat_increased {
        return None;
    }

    let average_defensive_depth = team_average_defensive_depth(before, defending_team);
    if average_defensive_depth <= 34.0 && yards_to_goal > 24.0 {
        return None;
    }

    let dt = before.dt_seconds.max(1e-6);
    let mut defender_penalties = Vec::new();
    let mut opening_pressure = 0.0;
    for defender in before
        .players
        .iter()
        .filter(|player| player.team == defending_team)
    {
        let Some(after_defender) = snapshot_player(after, defender.id) else {
            continue;
        };
        let before_position = before
            .player_position(defender.id)
            .unwrap_or(defender.position);
        let after_position = after
            .player_position(defender.id)
            .unwrap_or(after_defender.position);
        let moved = before_position.distance(after_position);
        let speed_yps = moved / dt;
        let distance_to_after_ball = after_position.distance(after.ball.position);
        if distance_to_after_ball > 36.0 && yards_to_goal > 30.0 {
            continue;
        }

        let action = player_normalized_last_action(after_defender);
        let passive = speed_yps < 0.75
            && !matches!(action, "tackle")
            && (!matches!(action, "defend") || moved < 0.18);
        if !passive {
            continue;
        }

        let fatigue = after_defender.fatigue.clamp(0.0, 1.0);
        let stamina = ability01(after_defender.skills.stamina);
        let energy_reserve = ((1.0 - fatigue) * 0.62 + stamina * 0.38).clamp(0.0, 1.0);
        let proximity_weight = ((36.0 - distance_to_after_ball) / 36.0).clamp(0.0, 1.0);
        let danger_weight =
            ((DEFENSIVE_RELAXATION_THREAT_YARDS - yards_to_goal) / 36.0).clamp(0.15, 1.0);
        let movement_weight =
            ((ball_progress.max(0.0) + lateral_shift * 0.14) / 4.5).clamp(0.0, 1.0);
        let opening = proximity_weight * danger_weight * (0.58 + movement_weight * 0.42);
        if opening < 0.10 {
            continue;
        }

        let penalty_scale = 0.55 + energy_reserve * 0.75;
        let amount = -(0.026 + opening * 0.095 * penalty_scale).clamp(0.025, 0.18);
        opening_pressure += opening * (0.70 + fatigue * 0.30);
        defender_penalties.push(PlayerRewardLoad {
            player_id: defender.id,
            amount,
            load: 0.0,
        });
    }

    let min_passive_defenders = if yards_to_goal <= 24.0 { 1 } else { 2 };
    if defender_penalties.len() < min_passive_defenders {
        return None;
    }

    let attacking_credit = (0.030 + opening_pressure * 0.040).clamp(0.035, 0.34);
    Some(DefensiveRelaxationSignal {
        possession_team,
        defending_team,
        attacking_credit,
        defender_penalties,
    })
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
    infer_discrete_events: bool,
) -> f64 {
    soccer_transition_reward_with_tactics(
        player,
        decision,
        before,
        after,
        score_home_before,
        score_away_before,
        score_home_after,
        score_away_after,
        infer_discrete_events,
        &SoccerTacticalLearningWeights::default(),
    )
}

fn soccer_transition_reward_with_tactics(
    player: &PlayerAgent,
    decision: &AgentDecisionTrace,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    score_home_before: u32,
    score_away_before: u32,
    score_home_after: u32,
    score_away_after: u32,
    infer_discrete_events: bool,
    tactical_learning: &SoccerTacticalLearningWeights,
) -> f64 {
    let action = normalize_soccer_action_label(&decision.action);
    let mut reward =
        dense_soccer_transition_reward(player, decision, before, after, action, tactical_learning);

    if !infer_discrete_events {
        return reward;
    }

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
    reward += (after_for as f64 - before_for as f64) * 100.0;
    if after_against > before_against {
        reward -= if matches!(player.role, PlayerRole::Goalkeeper | PlayerRole::Defender) {
            5.0
        } else {
            2.0
        };
    }

    if matches!(action, "pass" | "aerial-pass" | "first-time-pass") {
        if let Some(holder) = after.ball.holder {
            if holder != player.id
                && after
                    .players
                    .iter()
                    .find(|candidate| candidate.id == holder)
                    .is_some_and(|candidate| candidate.team == player.team)
            {
                let target = after.player_position(holder).unwrap_or(after.ball.position);
                let origin = before
                    .player_position(player.id)
                    .unwrap_or(before.ball.position);
                reward += completed_pass_reward(player.team, origin, target, before.field_length);
            }
        }
    }

    if before.possession_team() == Some(player.team.other()) && after.ball.holder == Some(player.id)
    {
        reward += 10.0;
    }

    reward
}

fn dense_soccer_transition_reward(
    player: &PlayerAgent,
    decision: &AgentDecisionTrace,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    action: &str,
    tactical_learning: &SoccerTacticalLearningWeights,
) -> f64 {
    let before_pos = before.player_position(player.id).unwrap_or(player.position);
    let after_pos = after.player_position(player.id).unwrap_or(before_pos);
    let moved_yards = before_pos.distance(after_pos);
    let before_obs = &decision.observation;
    let after_obs = after.observation_for(player.id);
    let before_possession = before.controlled_possession_team();
    let after_possession = after.controlled_possession_team();
    let attack_dir = player.team.attack_dir();
    let ball_forward = (after.ball.position.y - before.ball.position.y) * attack_dir;
    let player_forward = (after_pos.y - before_pos.y) * attack_dir;
    let mut reward = 0.0;
    let spacing_mode = before.team_spacing_mode_for(player.team);
    let spacing_delta = spacing_mode
        .map(|mode| {
            let before_score = before.team_spacing_score_for_candidate(
                player.team,
                Some(player.id),
                before_pos,
                mode,
            );
            let after_score = after.team_spacing_score_for_candidate(
                player.team,
                Some(player.id),
                after_pos,
                mode,
            );
            (before_score, after_score, after_score - before_score)
        })
        .unwrap_or((0.0, 0.0, 0.0));

    if before.ball.holder == Some(player.id) {
        let own_half_holder =
            pass_origin_in_own_half(player.team, before.ball.position, before.field_length);
        if after_possession == Some(player.team) {
            reward += 0.18;
        } else if after_possession == Some(player.team.other()) {
            reward -= if own_half_holder { 3.5 } else { 2.2 };
        } else {
            reward -= if own_half_holder { 0.85 } else { 0.55 };
        }
        reward += ball_forward.clamp(-8.0, 12.0) * 0.11;
        if own_half_holder {
            reward += ball_forward.clamp(-8.0, 14.0) * 0.08;
            if ball_forward < -1.25 && before_obs.perceived_pressure < 0.25 {
                reward -= 1.4;
            }
            if matches!(action, "pass" | "aerial-pass" | "first-time-pass")
                && ball_forward < -1.25
                && before_obs.perceived_pressure < 0.35
            {
                reward -= 1.1;
            }
        }
        reward += (before_obs.yards_to_goal - after_obs.yards_to_goal).clamp(-8.0, 8.0) * 0.07;
        if matches!(action, "dribble" | "pass" | "aerial-pass" | "shoot") {
            reward += 0.08;
        }
        if let Some(TeamSpacingMode::InPossession) = spacing_mode {
            let weight = before.possession_spacing_weight(player.team);
            reward += spacing_delta.2.clamp(-1.0, 1.0) * 0.05 * weight;
        }
        reward += tactical_shape_reward(
            player,
            before,
            after,
            before_pos,
            after_pos,
            spacing_mode,
            spacing_delta,
            tactical_learning,
        );
        if moved_yards < 0.08 && ball_forward.abs() < 0.08 {
            reward -= 0.55;
        }
    } else if before_possession == Some(player.team) {
        let own_half_team_possession =
            pass_origin_in_own_half(player.team, before.ball.position, before.field_length);
        reward +=
            (after_obs.open_space_score - before_obs.open_space_score).clamp(-5.0, 5.0) * 0.12;
        reward += match player.role {
            PlayerRole::Goalkeeper => player_forward.clamp(-4.0, 4.0) * 0.01,
            PlayerRole::Defender => player_forward.clamp(-5.0, 8.0) * 0.025,
            PlayerRole::Midfielder => player_forward.clamp(-6.0, 10.0) * 0.045,
            PlayerRole::Forward => player_forward.clamp(-6.0, 10.0) * 0.055,
        };
        if own_half_team_possession && player.role != PlayerRole::Goalkeeper {
            reward += player_forward.clamp(-5.0, 12.0) * 0.055;
            if player_forward < -0.35 {
                reward -= 0.16;
            }
        }
        if let Some(TeamSpacingMode::InPossession) = spacing_mode {
            let weight = before.possession_spacing_weight(player.team);
            reward += spacing_delta.2.clamp(-1.0, 1.0) * 0.20 * weight;
            reward += spacing_delta.1.clamp(-1.0, 1.0) * 0.045 * weight;
        }
        reward += tactical_shape_reward(
            player,
            before,
            after,
            before_pos,
            after_pos,
            spacing_mode,
            spacing_delta,
            tactical_learning,
        );
        reward += ball_forward.clamp(-8.0, 12.0) * 0.025;
        if matches!(action, "space") && moved_yards > 0.15 {
            reward += 0.07;
        }
        if moved_yards < 0.06 && !matches!(player.role, PlayerRole::Goalkeeper) {
            reward -= 0.18;
        }
    } else if before_possession == Some(player.team.other()) {
        let before_target = before
            .ball
            .holder
            .and_then(|holder| before.player_position(holder))
            .unwrap_or(before.ball.position);
        let after_target = before
            .ball
            .holder
            .and_then(|holder| after.player_position(holder))
            .unwrap_or(after.ball.position);
        let before_distance = before_pos.distance(before_target);
        let after_distance = after_pos.distance(after_target);
        let tracking_skill = ability01(player.skills.defensive_tracking);
        reward +=
            (before_distance - after_distance).clamp(-6.0, 6.0) * (0.055 + tracking_skill * 0.045);
        reward += defensive_goal_side_reward(player.team, before_pos, before);
        let before_goal_line_shape =
            defensive_goal_line_spacing_score(player.team, before_pos, before);
        let after_goal_line_shape =
            defensive_goal_line_spacing_score(player.team, after_pos, after);
        reward += after_goal_line_shape * 0.08
            + (after_goal_line_shape - before_goal_line_shape).clamp(-1.0, 1.0) * 0.18;
        if let Some(TeamSpacingMode::Defending) = spacing_mode {
            reward += spacing_delta.2.clamp(-1.0, 1.0) * 0.09;
            reward += spacing_delta.1.clamp(-1.0, 1.0) * 0.025;
        }
        reward += tactical_shape_reward(
            player,
            before,
            after,
            before_pos,
            after_pos,
            spacing_mode,
            spacing_delta,
            tactical_learning,
        );
        let opponent_forward =
            (after.ball.position.y - before.ball.position.y) * player.team.other().attack_dir();
        reward -= opponent_forward.clamp(-8.0, 12.0) * 0.035;
        if matches!(action, "defend" | "tackle") && moved_yards > 0.10 {
            reward += 0.06;
        }
        if after_possession == Some(player.team) {
            reward += if after.ball.holder == Some(player.id) {
                1.2
            } else {
                0.28
            };
        }
        if moved_yards < 0.06 && before_distance > 7.0 {
            reward -= 0.22;
        }
    } else {
        let before_distance = before_pos.distance(before.ball.position);
        let after_distance = after_pos.distance(after.ball.position);
        reward += (before_distance - after_distance).clamp(-8.0, 8.0) * 0.10;
        if loose_ball_fifty_fifty_duel_for(before, player.id) {
            reward += if after_distance < before_distance {
                0.75
            } else {
                -0.35
            };
        }
        if after_possession == Some(player.team) {
            reward += if after.ball.holder == Some(player.id) {
                0.95
            } else {
                0.24
            };
        }
        if moved_yards < 0.06 && before_distance > 9.0 {
            reward -= 0.20;
        }
    }

    if action == "hold" {
        reward -= if before.ball.holder == Some(player.id) || before_possession == Some(player.team)
        {
            0.55
        } else {
            0.20
        };
    }

    reward.clamp(-4.0, 4.0)
}

fn tactical_shape_reward(
    player: &PlayerAgent,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    before_pos: Vec2,
    after_pos: Vec2,
    spacing_mode: Option<TeamSpacingMode>,
    spacing_delta: (f64, f64, f64),
    weights: &SoccerTacticalLearningWeights,
) -> f64 {
    if matches!(player.role, PlayerRole::Goalkeeper) {
        return 0.0;
    }

    let field_width = before.field_width.max(1.0);
    let before_possession = before.controlled_possession_team();
    if before_possession == Some(player.team) {
        let spacing_weight = before.possession_spacing_weight(player.team).max(0.35);
        let before_width = team_field_player_lateral_width_for_candidate(
            before,
            player.team,
            player.id,
            before_pos,
        );
        let after_width =
            team_field_player_lateral_width_for_candidate(after, player.team, player.id, after_pos);
        let width_delta = ((after_width - before_width) / field_width).clamp(-1.0, 1.0);
        let flank_delta = (flank_lane_score(after_pos, field_width)
            - flank_lane_score(before_pos, field_width))
        .clamp(-1.0, 1.0);
        let mut reward = width_delta * weights.attack_width_delta_weight * spacing_weight;
        reward += attack_width_score(after_width, field_width)
            * weights.attack_width_score_weight
            * spacing_weight;
        reward += flank_delta * weights.attack_flank_lane_weight;
        reward +=
            flank_lane_score(after_pos, field_width) * weights.attack_flank_lane_weight * 0.18;
        if let Some(TeamSpacingMode::InPossession) = spacing_mode {
            reward += spacing_delta.2.clamp(-1.0, 1.0)
                * weights.attack_spacing_delta_weight
                * spacing_weight;
            reward += spacing_delta.1.clamp(-1.0, 1.0)
                * weights.attack_spacing_score_weight
                * spacing_weight;
        }
        return reward;
    }

    if before_possession == Some(player.team.other()) {
        let before_width = team_field_player_lateral_width_for_candidate(
            before,
            player.team,
            player.id,
            before_pos,
        );
        let after_width =
            team_field_player_lateral_width_for_candidate(after, player.team, player.id, after_pos);
        let contract_delta = ((before_width - after_width) / field_width).clamp(-1.0, 1.0);
        let mut reward = contract_delta * weights.defense_contract_delta_weight;
        reward += defense_contract_width_score(after_width, field_width)
            * weights.defense_compactness_score_weight;
        if let Some(TeamSpacingMode::Defending) = spacing_mode {
            reward += spacing_delta.2.clamp(-1.0, 1.0) * weights.defense_spacing_delta_weight;
            reward += spacing_delta.1.clamp(-1.0, 1.0) * weights.defense_spacing_score_weight;
        }
        return reward;
    }

    0.0
}

fn team_field_player_lateral_width_for_candidate(
    snapshot: &WorldSnapshot,
    team: Team,
    replace_player_id: usize,
    candidate_position: Vec2,
) -> f64 {
    let mut min_x = f64::INFINITY;
    let mut max_x = f64::NEG_INFINITY;
    let mut count = 0;
    for player in snapshot
        .players
        .iter()
        .filter(|player| player.team == team && player.role != PlayerRole::Goalkeeper)
    {
        let position = if player.id == replace_player_id {
            candidate_position
        } else {
            snapshot
                .player_position(player.id)
                .unwrap_or(player.position)
        };
        min_x = min_x.min(position.x);
        max_x = max_x.max(position.x);
        count += 1;
    }
    if count < 2 {
        0.0
    } else {
        (max_x - min_x).clamp(0.0, snapshot.field_width.max(0.0))
    }
}

fn attack_width_score(width_yards: f64, field_width_yards: f64) -> f64 {
    width_band_score(
        width_yards,
        field_width_yards,
        field_width_yards * 0.52,
        field_width_yards * 0.78,
        field_width_yards * 0.96,
    )
}

fn defense_contract_width_score(width_yards: f64, field_width_yards: f64) -> f64 {
    width_band_score(
        width_yards,
        field_width_yards,
        field_width_yards * 0.22,
        field_width_yards * 0.38,
        field_width_yards * 0.58,
    )
}

fn width_band_score(
    width_yards: f64,
    field_width_yards: f64,
    min_yards: f64,
    ideal_yards: f64,
    max_yards: f64,
) -> f64 {
    if !width_yards.is_finite() || field_width_yards <= 0.0 {
        return -1.0;
    }
    if width_yards < min_yards {
        -((min_yards - width_yards) / min_yards.max(1.0)).clamp(0.0, 1.0)
    } else if width_yards > max_yards {
        -((width_yards - max_yards) / (field_width_yards - max_yards).max(1.0)).clamp(0.0, 1.0)
    } else {
        let span = if width_yards <= ideal_yards {
            (ideal_yards - min_yards).max(1e-6)
        } else {
            (max_yards - ideal_yards).max(1e-6)
        };
        (1.0 - ((width_yards - ideal_yards).abs() / span) * 0.45).clamp(0.55, 1.0)
    }
}

fn flank_lane_score(position: Vec2, field_width_yards: f64) -> f64 {
    if !position.x.is_finite() || field_width_yards <= 0.0 {
        return 0.0;
    }
    let half_width = (field_width_yards * 0.5).max(1.0);
    let lateral = ((position.x - field_width_yards * 0.5).abs() / half_width).clamp(0.0, 1.0);
    if lateral < 0.35 {
        -((0.35 - lateral) / 0.35).clamp(0.0, 1.0)
    } else {
        ((lateral - 0.35) / 0.55).clamp(0.0, 1.0)
    }
}

fn defensive_goal_side_reward(team: Team, player_position: Vec2, snapshot: &WorldSnapshot) -> f64 {
    let Some(attacker_position) = snapshot
        .ball
        .holder
        .and_then(|holder| snapshot.players.iter().find(|p| p.id == holder))
        .filter(|holder| holder.team == team.other())
        .and_then(|holder| snapshot.player_position(holder.id))
    else {
        return 0.0;
    };
    let own_goal_y = team.other().goal_y(snapshot.field_length);
    let goal_side_of_ball =
        goal_side_between_y(player_position.y, snapshot.ball.position.y, own_goal_y);
    let goal_side_of_attacker =
        goal_side_between_y(player_position.y, attacker_position.y, own_goal_y);
    match (goal_side_of_ball, goal_side_of_attacker) {
        (true, true) => 0.18,
        (true, false) | (false, true) => -0.10,
        (false, false) => -0.34,
    }
}

fn defensive_goal_line_spacing_score(
    team: Team,
    player_position: Vec2,
    snapshot: &WorldSnapshot,
) -> f64 {
    let own_goal_y = team.other().goal_y(snapshot.field_length);
    let depth_from_goal_line = (player_position.y - own_goal_y).abs();
    let goal_line_buffer_penalty = if !snapshot.ball_near_own_goal_line(team)
        && depth_from_goal_line < DEFENSIVE_GOAL_LINE_BUFFER_YARDS
    {
        -((DEFENSIVE_GOAL_LINE_BUFFER_YARDS - depth_from_goal_line)
            / DEFENSIVE_GOAL_LINE_BUFFER_YARDS)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let behind_ball = ((snapshot.ball.position.y - player_position.y) * team.attack_dir()).max(0.0);
    let disconnected_penalty = if !snapshot.ball_near_own_goal_line(team)
        && behind_ball > DEFENSIVE_MAX_BEHIND_BALL_YARDS
    {
        -((behind_ball - DEFENSIVE_MAX_BEHIND_BALL_YARDS) / DEFENSIVE_MAX_BEHIND_BALL_YARDS)
            .clamp(0.0, 1.0)
    } else {
        0.0
    };
    let healthy_line_bonus = if goal_line_buffer_penalty == 0.0 && disconnected_penalty == 0.0 {
        0.35
    } else {
        0.0
    };
    (healthy_line_bonus + goal_line_buffer_penalty + disconnected_penalty).clamp(-1.0, 0.35)
}

fn goal_side_between_y(player_y: f64, threat_y: f64, own_goal_y: f64) -> bool {
    if own_goal_y <= threat_y {
        player_y <= threat_y + 0.75
    } else {
        player_y >= threat_y - 0.75
    }
}

fn loose_ball_fifty_fifty_duel_for(snapshot: &WorldSnapshot, player_id: usize) -> bool {
    let Some(me) = snapshot.players.iter().find(|p| p.id == player_id) else {
        return false;
    };
    if snapshot.ball.holder.is_some() || snapshot.ball.velocity.len() > 10.0 {
        return false;
    }
    let Some((home, away)) = loose_ball_fifty_fifty_duel(snapshot) else {
        return false;
    };
    let contender_id = match me.team {
        Team::Home => home,
        Team::Away => away,
    };
    contender_id == player_id
}

fn loose_ball_fifty_fifty_duel(snapshot: &WorldSnapshot) -> Option<(usize, usize)> {
    if snapshot.ball.holder.is_some() {
        return None;
    }
    let nearest = |team| {
        snapshot
            .players
            .iter()
            .filter(|player| player.team == team)
            .filter_map(|player| {
                let position = snapshot
                    .player_position(player.id)
                    .unwrap_or(player.position);
                let velocity_toward_ball = player
                    .velocity
                    .normalized()
                    .dot((snapshot.ball.position - position).normalized());
                Some((
                    player.id,
                    position.distance(snapshot.ball.position),
                    velocity_toward_ball,
                ))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    };
    let home = nearest(Team::Home)?;
    let away = nearest(Team::Away)?;
    let distance_close = home.1.min(away.1) <= 18.0;
    let distance_even = (home.1 - away.1).abs() <= 2.8;
    let velocity_even = (home.2 - away.2).abs() <= 0.55;
    (distance_close && distance_even && velocity_even).then_some((home.0, away.0))
}

impl SoccerPomdpObservation {
    fn pressure_like_penalty(&self) -> f64 {
        (self.perceived_pressure * 18.0).max(0.0)
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
    #[serde(default)]
    pub position_history: Vec<Vec2>,
    pub velocity: Vec2,
    pub acceleration: Vec2,
    pub jerk: Vec2,
    #[serde(default)]
    pub offside_line: Option<AssistantOffsideLineSnapshot>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AssistantOffsideLineSnapshot {
    pub flank: AssistantFlank,
    pub attacking_team: Team,
    pub defending_team: Team,
    pub second_last_defender_y: f64,
    pub ball_y: f64,
    pub halfway_y: f64,
    pub effective_line_y: f64,
    pub players_beyond_line: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum AgentScheduleKind {
    Player,
    Official,
    Ball,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentScheduleEntry {
    pub kind: AgentScheduleKind,
    pub id: usize,
    pub label: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CentralBrainPlayerAwareness {
    pub id: usize,
    pub team: Team,
    pub position: Vec2,
    pub velocity: Vec2,
    pub controller_slot: Option<usize>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CentralBrainSnapshot {
    pub phase: TacticalPhase,
    pub possession_team: Option<Team>,
    pub ball_position: Vec2,
    pub ball_velocity: Vec2,
    pub ball_holder: Option<usize>,
    pub pressure_line_home: f64,
    pub pressure_line_away: f64,
    pub tracked_players: Vec<CentralBrainPlayerAwareness>,
    pub tracked_officials: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchFrame {
    pub tick: u64,
    pub clock_seconds: f64,
    pub ball: BallState,
    #[serde(default)]
    pub ball_history: Vec<BallPositionSample>,
    pub players: Vec<PlayerSnapshot>,
    pub officials: Vec<OfficialSnapshot>,
    #[serde(default)]
    pub agent_schedule: Vec<AgentScheduleEntry>,
    pub score_home: u32,
    pub score_away: u32,
    pub phase: TacticalPhase,
    pub central_brain: CentralBrainSnapshot,
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
    #[serde(default, alias = "ballAltitude", alias = "ballZ")]
    pub ball_altitude_yards: Option<f64>,
    #[serde(default, alias = "flight", alias = "ballFlight")]
    pub pass_flight: Option<PassFlight>,
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
    #[serde(default)]
    pub skills: Option<SkillProfile>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerPolicyArtifact {
    pub config: MatchConfig,
    pub summary: MatchSummary,
    pub transition_count: usize,
    pub options: SoccerQPolicyOptions,
    pub entries: Vec<SoccerQEntry>,
    #[serde(default)]
    pub target_entries: Vec<SoccerQTargetEntry>,
    pub events: Vec<MatchEvent>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerTeamPolicyArtifact {
    pub config: MatchConfig,
    pub summary: MatchSummary,
    pub learning: SoccerLearningSnapshot,
    pub adversarial: bool,
    pub home_options: Option<SoccerQPolicyOptions>,
    pub away_options: Option<SoccerQPolicyOptions>,
    pub home_entries: Vec<SoccerQEntry>,
    #[serde(default)]
    pub home_target_entries: Vec<SoccerQTargetEntry>,
    pub away_entries: Vec<SoccerQEntry>,
    #[serde(default)]
    pub away_target_entries: Vec<SoccerQTargetEntry>,
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
pub struct SoccerBallSurfaceRequest {
    #[serde(default = "default_ball_drag_per_tick")]
    pub ball_drag_per_tick: f64,
    #[serde(default = "default_ball_air_resistance")]
    pub ball_air_resistance: f64,
    #[serde(default = "default_ball_grass_resistance_yps2")]
    pub ball_grass_resistance_yps2: f64,
    #[serde(default = "default_ball_stop_speed_yps")]
    pub ball_stop_speed_yps: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerBallSurfaceResponse {
    pub config: MatchConfig,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningRuntimeRequest {
    pub learning_enabled: Option<bool>,
    pub learning_logging_enabled: Option<bool>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningRuntimeResponse {
    pub config: MatchConfig,
    pub learning: SoccerLearningSnapshot,
}

fn default_self_play_training_episodes() -> usize {
    100
}

fn default_import_trained_policy() -> bool {
    true
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerSelfPlayTrainingRequest {
    #[serde(default = "default_self_play_training_episodes")]
    pub episodes: usize,
    pub minutes: Option<f64>,
    #[serde(default, alias = "halves")]
    pub period_count: Option<usize>,
    #[serde(default, alias = "halftimeRecoverySeconds")]
    pub period_break_recovery_seconds: Option<f64>,
    pub dt_seconds: Option<f64>,
    pub learning_interval_ticks: Option<usize>,
    pub seed: Option<u32>,
    pub options: Option<SoccerQPolicyOptions>,
    pub tactical_learning: Option<SoccerTacticalLearningWeights>,
    pub artifact_path: Option<String>,
    #[serde(default = "default_import_trained_policy")]
    pub import_into_session: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerSelfPlayTrainingResponse {
    pub artifact_path: Option<String>,
    pub imported_home_entries: usize,
    pub imported_away_entries: usize,
    pub learning: SoccerLearningSnapshot,
    pub artifact: SoccerSelfPlayTrainingArtifact,
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
    pub learning: SoccerLearningSnapshot,
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
    pub learning: SoccerLearningSnapshot,
    pub summary: MatchSummary,
    pub controller_assignments: Vec<ControllerAssignment>,
    pub done: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerTeamPolicyImportResponse {
    pub learning: SoccerLearningSnapshot,
    pub imported_home_entries: usize,
    pub imported_away_entries: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerTrackingImportRequest {
    #[serde(default)]
    pub source: String,
    #[serde(default)]
    pub format: Option<String>,
    pub content: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerTrackingImportResponse {
    pub learning: SoccerLearningSnapshot,
    pub source: String,
    pub format: String,
    pub frames: usize,
    pub imported_transitions: usize,
    pub imported_home_entries: usize,
    pub imported_away_entries: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SoccerLearningSnapshot {
    pub total_transitions: usize,
    #[serde(default = "default_learning_enabled")]
    pub learning_enabled: bool,
    #[serde(default = "default_learning_logging_enabled")]
    pub learning_logging_enabled: bool,
    pub shared_policy_enabled: bool,
    pub shared_policy_entries: usize,
    pub shared_policy_visits: u64,
    #[serde(default)]
    pub shared_policy_target_entries: usize,
    #[serde(default)]
    pub shared_policy_target_visits: u64,
    pub team_policies_enabled: bool,
    pub adversarial_learning_enabled: bool,
    pub home_policy_entries: usize,
    pub home_policy_visits: u64,
    #[serde(default)]
    pub home_policy_target_entries: usize,
    #[serde(default)]
    pub home_policy_target_visits: u64,
    pub away_policy_entries: usize,
    pub away_policy_visits: u64,
    #[serde(default)]
    pub away_policy_target_entries: usize,
    #[serde(default)]
    pub away_policy_target_visits: u64,
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
    pub team_policies: Option<SoccerTeamQPolicies>,
    pub human_inputs: SharedHumanInputs,
    pub central_brain: CentralBrain,
    rng: SeededRandom,
    pending_pass: Option<PendingPass>,
    pending_shot: Option<PendingShot>,
    reward_events: Vec<SoccerRewardEvent>,
    possession_chain: VecDeque<usize>,
    defensive_delay_clocks: HashMap<usize, f64>,
    defensive_beat_clocks: HashMap<usize, f64>,
    last_agent_schedule: Vec<AgentScheduleEntry>,
}

impl SoccerMatch {
    pub fn default_11v11(config: MatchConfig) -> Self {
        let mut rng = mulberry32(config.seed);
        let mut players = default_players(&config, &mut rng);
        let officials = vec![
            OfficialAgent::new(
                22,
                OfficialKind::CenterReferee,
                Vec2::new(
                    config.field_width_yards * 0.5,
                    config.field_length_yards * 0.5,
                ),
            ),
            OfficialAgent::new(
                23,
                OfficialKind::AssistantRefereeNear,
                Vec2::new(1.5, config.field_length_yards * 0.5),
            ),
            OfficialAgent::new(
                24,
                OfficialKind::AssistantRefereeFar,
                Vec2::new(
                    config.field_width_yards - 1.5,
                    config.field_length_yards * 0.5,
                ),
            ),
        ];
        let kickoff = players
            .iter()
            .find(|p| p.team == Team::Home && p.role == PlayerRole::Midfielder)
            .map(|p| p.id);
        if let Some(holder_id) = kickoff {
            mark_player_receive_facing(&mut players, holder_id);
        }
        let shared_positions = SharedPlayerPositions::default();
        shared_positions.sync_from_players(&players, 0, 0.0);
        let mut possession_chain = VecDeque::new();
        if let Some(holder_id) = kickoff {
            possession_chain.push_back(holder_id);
        }
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
                    curl_acceleration: Vec2::zero(),
                    altitude_yards: 0.0,
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
            team_policies: None,
            human_inputs: SharedHumanInputs::new(),
            central_brain: CentralBrain::default(),
            rng,
            pending_pass: None,
            pending_shot: None,
            reward_events: Vec::new(),
            possession_chain,
            defensive_delay_clocks: HashMap::new(),
            defensive_beat_clocks: HashMap::new(),
            last_agent_schedule: Vec::new(),
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

    pub fn with_team_policies(mut self, team_policies: SoccerTeamQPolicies) -> Self {
        self.team_policies = Some(team_policies);
        self
    }

    pub fn set_team_policies(&mut self, team_policies: SoccerTeamQPolicies) {
        self.team_policies = Some(team_policies);
    }

    pub fn team_policies(&self) -> Option<&SoccerTeamQPolicies> {
        self.team_policies.as_ref()
    }

    pub fn team_policies_mut(&mut self) -> Option<&mut SoccerTeamQPolicies> {
        self.team_policies.as_mut()
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

    pub fn learning_snapshot(&self) -> SoccerLearningSnapshot {
        let (
            shared_policy_entries,
            shared_policy_visits,
            shared_policy_target_entries,
            shared_policy_target_visits,
        ) = self
            .learned_policy
            .as_ref()
            .map(|policy| {
                (
                    policy.q_values.len(),
                    policy.visit_count(),
                    policy.target_values.len(),
                    policy.target_visit_count(),
                )
            })
            .unwrap_or((0, 0, 0, 0));
        let (
            home_policy_entries,
            home_policy_visits,
            home_policy_target_entries,
            home_policy_target_visits,
            away_policy_entries,
            away_policy_visits,
            away_policy_target_entries,
            away_policy_target_visits,
        ) = self
            .team_policies
            .as_ref()
            .map(|policies| {
                (
                    policies.home.q_values.len(),
                    policies.home.visit_count(),
                    policies.home.target_values.len(),
                    policies.home.target_visit_count(),
                    policies.away.q_values.len(),
                    policies.away.visit_count(),
                    policies.away.target_values.len(),
                    policies.away.target_visit_count(),
                )
            })
            .unwrap_or((0, 0, 0, 0, 0, 0, 0, 0));

        SoccerLearningSnapshot {
            total_transitions: self.learning_transitions.len(),
            learning_enabled: self.config.learning_enabled,
            learning_logging_enabled: self.config.learning_logging_enabled,
            shared_policy_enabled: self.learned_policy.is_some(),
            shared_policy_entries,
            shared_policy_visits,
            shared_policy_target_entries,
            shared_policy_target_visits,
            team_policies_enabled: self.team_policies.is_some(),
            adversarial_learning_enabled: self.team_policies.is_some(),
            home_policy_entries,
            home_policy_visits,
            home_policy_target_entries,
            home_policy_target_visits,
            away_policy_entries,
            away_policy_visits,
            away_policy_target_entries,
            away_policy_target_visits,
        }
    }

    pub fn team_policy_artifact(&self) -> SoccerTeamPolicyArtifact {
        let (
            home_options,
            away_options,
            home_entries,
            home_target_entries,
            away_entries,
            away_target_entries,
        ) = if let Some(policies) = &self.team_policies {
            (
                Some(policies.home.options.clone()),
                Some(policies.away.options.clone()),
                policies.home.entries(),
                policies.home.target_entries(),
                policies.away.entries(),
                policies.away.target_entries(),
            )
        } else {
            (None, None, Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };

        SoccerTeamPolicyArtifact {
            config: self.config.clone(),
            summary: self.summary(),
            learning: self.learning_snapshot(),
            adversarial: self.team_policies.is_some(),
            home_options,
            away_options,
            home_entries,
            home_target_entries,
            away_entries,
            away_target_entries,
            events: self.events.clone(),
        }
    }

    pub fn import_team_policy_artifact(
        &mut self,
        artifact: SoccerTeamPolicyArtifact,
    ) -> Result<SoccerTeamPolicyImportResponse, String> {
        let policies = SoccerTeamQPolicies::from_artifact(&artifact)?;
        let imported_home_entries = policies.home.q_values.len();
        let imported_away_entries = policies.away.q_values.len();
        self.team_policies = Some(policies);
        Ok(SoccerTeamPolicyImportResponse {
            learning: self.learning_snapshot(),
            imported_home_entries,
            imported_away_entries,
        })
    }

    pub fn import_self_play_training_artifact(
        &mut self,
        artifact: SoccerSelfPlayTrainingArtifact,
    ) -> Result<SoccerTeamPolicyImportResponse, String> {
        let policies = SoccerTeamQPolicies::from_self_play_artifact(&artifact)?;
        let imported_home_entries = policies.home.q_values.len();
        let imported_away_entries = policies.away.q_values.len();
        self.team_policies = Some(policies);
        Ok(SoccerTeamPolicyImportResponse {
            learning: self.learning_snapshot(),
            imported_home_entries,
            imported_away_entries,
        })
    }

    pub fn import_tracking_for_team_policy(
        &mut self,
        request: SoccerTrackingImportRequest,
    ) -> Result<SoccerTrackingImportResponse, String> {
        let source = if request.source.trim().is_empty() {
            "live-upload".to_string()
        } else {
            request.source.clone()
        };
        let format = tracking_import_format(&request);
        let tracking = match format.as_str() {
            "json" => soccer_tracking_dataset_from_json(&request.content),
            "csv" => {
                soccer_tracking_dataset_from_csv(&request.content, self.config.clone(), &source)
            }
            other => Err(format!("unsupported tracking import format {other}")),
        }?;
        let dataset = tracking.to_learning_dataset()?;
        let policies = self
            .team_policies
            .get_or_insert_with(|| SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
        policies.train_adversarial(&dataset.transitions);
        let imported_home_entries = policies.home.q_values.len();
        let imported_away_entries = policies.away.q_values.len();
        let learning = self.learning_snapshot();

        Ok(SoccerTrackingImportResponse {
            learning,
            source: tracking.source.clone(),
            format,
            frames: tracking.frames.len(),
            imported_transitions: dataset.transitions.len(),
            imported_home_entries,
            imported_away_entries,
        })
    }

    pub fn is_done(&self) -> bool {
        self.tick >= self.config.total_ticks()
    }

    pub fn start_new_period(&mut self, period_number: usize, kickoff_team: Team) {
        let recovery_seconds = self.config.period_break_recovery_seconds.max(0.0);
        if recovery_seconds > 0.0 {
            for player in &mut self.players {
                player.fatigue = (player.fatigue
                    + MovementGait::Stand.fatigue_delta(player.skills.stamina, recovery_seconds))
                .clamp(0.0, 1.0);
            }
        }
        self.events.push(MatchEvent {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            kind: "period-break".to_string(),
            team: None,
            player_id: None,
            description: format!("period {} complete", period_number.saturating_sub(1)),
        });
        self.reset_after_goal(kickoff_team);
        self.events.push(MatchEvent {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            kind: "period-start".to_string(),
            team: Some(kickoff_team),
            player_id: self.ball.holder,
            description: format!(
                "period {period_number} kickoff for {}",
                kickoff_team.label()
            ),
        });
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

    pub fn clear_controller_assignments(&mut self) {
        for player in &mut self.players {
            player.controller_slot = None;
        }
    }

    pub fn update_ball_surface(&mut self, request: SoccerBallSurfaceRequest) -> Result<(), String> {
        validate_ball_surface(
            request.ball_drag_per_tick,
            request.ball_air_resistance,
            request.ball_grass_resistance_yps2,
            request.ball_stop_speed_yps,
        )?;
        self.config.ball_drag_per_tick = request.ball_drag_per_tick;
        self.config.ball_air_resistance = request.ball_air_resistance;
        self.config.ball_grass_resistance_yps2 = request.ball_grass_resistance_yps2;
        self.config.ball_stop_speed_yps = request.ball_stop_speed_yps;
        Ok(())
    }

    pub fn update_learning_runtime(
        &mut self,
        request: SoccerLearningRuntimeRequest,
    ) -> SoccerLearningRuntimeResponse {
        if let Some(enabled) = request.learning_enabled {
            self.config.learning_enabled = enabled;
        }
        if let Some(enabled) = request.learning_logging_enabled {
            self.config.learning_logging_enabled = enabled;
        }
        SoccerLearningRuntimeResponse {
            config: self.config.clone(),
            learning: self.learning_snapshot(),
        }
    }

    fn learned_action_for_player(
        &self,
        snapshot: &WorldSnapshot,
        player_id: usize,
    ) -> Option<SoccerLearnedPlan> {
        let player = snapshot.players.iter().find(|p| p.id == player_id)?;
        if let Some(team_policies) = &self.team_policies {
            let policy = team_policies.policy(player.team);
            if let Some(action) = policy.best_action_for_snapshot(snapshot, player_id) {
                return Some(Self::learned_plan_for_policy(
                    policy, snapshot, player_id, action,
                ));
            }
        }
        self.learned_policy.as_ref().and_then(|policy| {
            policy
                .best_action_for_snapshot(snapshot, player_id)
                .map(|action| Self::learned_plan_for_policy(policy, snapshot, player_id, action))
        })
    }

    fn learned_plan_for_policy(
        policy: &SoccerQPolicy,
        snapshot: &WorldSnapshot,
        player_id: usize,
        action: String,
    ) -> SoccerLearnedPlan {
        let normalized_action = normalize_soccer_action_label(&action).to_string();
        let is_pass = matches!(
            normalized_action.as_str(),
            "pass" | "aerial-pass" | "first-time-pass"
        );
        let mut plan = SoccerLearnedPlan {
            action,
            target_player: None,
            target_point: None,
        };
        if is_pass {
            let candidates = if normalized_action == "aerial-pass" {
                snapshot.ranked_visible_aerial_pass_targets(player_id, 11)
            } else {
                snapshot.ranked_visible_pass_targets(player_id, 11)
            };
            plan.target_player = policy.best_target_player_for_snapshot(
                snapshot,
                player_id,
                &normalized_action,
                &candidates,
            );
            plan.target_point = plan
                .target_player
                .and_then(|target| snapshot.player_position(target));
        }
        plan
    }

    fn yield_for_controller_threads(&self) {
        if !self
            .players
            .iter()
            .any(|player| player.controller_slot.is_some())
        {
            return;
        }
        let _ = self
            .human_inputs
            .wait_for_pending_input(Duration::from_millis(CONTROLLER_INPUT_YIELD_MS));
        thread::yield_now();
    }

    pub fn run_time_step(&mut self) {
        if self.is_done() {
            return;
        }
        let brain_input_snapshot = WorldSnapshot::from_match(self);
        let score_home_before = self.score_home;
        let score_away_before = self.score_away;
        let reward_event_start = self.reward_events.len();
        self.central_brain
            .run_time_step(&brain_input_snapshot, &mut self.rng);
        self.yield_for_controller_threads();
        let snapshot = WorldSnapshot::from_match(self);
        let ball_velocity_before = self.ball.velocity;

        let mut actor_order: Vec<usize> = (0..self.players.len() + self.officials.len()).collect();
        fisher_yates_shuffle(&mut actor_order, &mut self.rng);
        self.last_agent_schedule = self.agent_schedule_for_actor_order(&actor_order);
        self.last_agent_schedule.push(AgentScheduleEntry {
            kind: AgentScheduleKind::Ball,
            id: self.ball.id,
            label: "ball".to_string(),
        });

        let mut intents = Vec::new();
        for actor in actor_order.iter().copied() {
            if actor < self.players.len() {
                let input_frame = self.players[actor]
                    .controller_slot
                    .and_then(|slot| self.human_inputs.drain_latest_for_slot(slot))
                    .filter(|frame| frame.player_id.is_none() || frame.player_id == Some(actor));
                let learned_plan = if self.config.learning_enabled {
                    self.learned_action_for_player(&snapshot, actor)
                } else {
                    None
                };
                let intent = self.players[actor].run_time_step(
                    &snapshot,
                    input_frame.as_ref(),
                    learned_plan.as_ref(),
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
        self.update_possession_chase_trackers(
            &snapshot,
            &next_snapshot,
            self.config.learning_enabled || self.config.learning_logging_enabled,
        );
        if self.config.learning_enabled || self.config.learning_logging_enabled {
            let next_snapshot = WorldSnapshot::from_match(self);
            self.update_defensive_reward_trackers(&snapshot, &next_snapshot);
            let has_tick_reward_events = self.reward_events.len() > reward_event_start;
            let period_start = self.config.period_start_after_tick(self.tick);
            let interval = self.config.learning_interval_ticks.max(1);
            let learning_due = interval <= 1
                || has_tick_reward_events
                || period_start.is_some()
                || self.tick as usize % interval == 0;
            if learning_due {
                let new_transitions = self.learning_transitions_for(
                    &snapshot,
                    &next_snapshot,
                    score_home_before,
                    score_away_before,
                    &self.reward_events[reward_event_start..],
                );
                if self.config.learning_enabled {
                    if let Some(team_policies) = &mut self.team_policies {
                        team_policies.train_adversarial(&new_transitions);
                    }
                    if let Some(policy) = &mut self.learned_policy {
                        policy.train(&new_transitions);
                    }
                }
                if self.config.learning_logging_enabled {
                    self.learning_transitions.extend(new_transitions);
                }
            }
        }
        if let Some(period_number) = self.config.period_start_after_tick(self.tick) {
            self.start_new_period(period_number, kickoff_team_for_period(period_number));
        }
    }

    pub fn to_frame(&self) -> MatchFrame {
        let snapshot = WorldSnapshot::from_match(self);
        let central_brain = self
            .central_brain
            .to_snapshot(&snapshot, self.officials.len());
        let officials = self
            .officials
            .iter()
            .map(|o| OfficialSnapshot {
                id: o.id,
                kind: o.kind,
                position: o.position,
                position_history: o.position_history.iter().cloned().collect(),
                velocity: o.velocity,
                acceleration: o.acceleration,
                jerk: o.jerk,
                offside_line: assistant_offside_line_snapshot(&snapshot, o.kind),
            })
            .collect();
        MatchFrame {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            ball: self.ball.to_state(),
            ball_history: snapshot.ball_history,
            players: snapshot.players,
            officials,
            agent_schedule: self.last_agent_schedule.clone(),
            score_home: self.score_home,
            score_away: self.score_away,
            phase: self.central_brain.phase,
            central_brain,
            home_directive: self.central_brain.home_directive.clone(),
            away_directive: self.central_brain.away_directive.clone(),
        }
    }

    fn agent_schedule_for_actor_order(&self, actor_order: &[usize]) -> Vec<AgentScheduleEntry> {
        actor_order
            .iter()
            .filter_map(|actor| {
                if *actor < self.players.len() {
                    let player = &self.players[*actor];
                    Some(AgentScheduleEntry {
                        kind: AgentScheduleKind::Player,
                        id: player.id,
                        label: format!("{} #{} {}", player.team.label(), player.shirt, player.name),
                    })
                } else {
                    let official_idx = *actor - self.players.len();
                    self.officials
                        .get(official_idx)
                        .map(|official| AgentScheduleEntry {
                            kind: AgentScheduleKind::Official,
                            id: official.id,
                            label: official.kind.label().to_string(),
                        })
                }
            })
            .collect()
    }

    fn mark_ball_received(&mut self, holder_id: usize) {
        mark_player_receive_facing(&mut self.players, holder_id);
    }

    fn record_reward_event(&mut self, player_id: usize, amount: f64) {
        self.record_reward_event_at(self.tick, player_id, amount);
    }

    fn record_reward_event_at(&mut self, tick: u64, player_id: usize, amount: f64) {
        if amount.abs() <= 1e-9 || player_id >= self.players.len() {
            return;
        }
        self.reward_events.push(SoccerRewardEvent {
            tick,
            player_id,
            amount,
        });
    }

    fn record_possession_touch(&mut self, player_id: usize) {
        if player_id >= self.players.len() {
            return;
        }
        if self
            .possession_chain
            .back()
            .is_some_and(|last_player| *last_player == player_id)
        {
            return;
        }
        self.possession_chain.push_back(player_id);
        while self.possession_chain.len() > 10 {
            self.possession_chain.pop_front();
        }
    }

    fn record_completed_pass_reward(&mut self, pass: &PendingPass, receiver: usize) {
        let amount = completed_pass_reward(
            pass.team,
            pass.origin,
            pass.intended_target,
            self.config.field_length_yards,
        );
        self.record_reward_event(pass.from, amount);
        self.record_possession_touch(pass.from);
        self.record_possession_touch(receiver);
    }

    fn record_interception_reward(
        &mut self,
        interceptor: usize,
        intercepted_pass: Option<&PendingPass>,
    ) {
        self.record_reward_event(interceptor, 10.0);
        self.record_possession_touch(interceptor);
        if let Some(pass) = intercepted_pass {
            let own_half =
                pass_origin_in_own_half(pass.team, pass.origin, self.config.field_length_yards);
            let backward = pass_direction_bucket(pass.team, pass.origin, pass.intended_target)
                == PassDirectionBucket::Backward;
            let penalty = if own_half && backward { -6.0 } else { -2.0 };
            self.record_reward_event(pass.from, penalty);
        }
    }

    fn record_duel_rewards(&mut self, winner: usize) {
        if winner >= self.players.len() {
            return;
        }
        let winner_team = self.players[winner].team;
        let ball_position = self.ball.position;
        let contenders = self
            .players
            .iter()
            .filter(|player| {
                player.position.distance(ball_position) <= PLAYER_CONTROL_RADIUS_YARDS + 1.35
            })
            .map(|player| (player.id, player.team))
            .collect::<Vec<_>>();
        let has_teammate = contenders.iter().any(|(_, team)| *team == winner_team);
        let has_opponent = contenders.iter().any(|(_, team)| *team != winner_team);
        if !has_teammate || !has_opponent {
            return;
        }
        self.record_reward_event(winner, 6.0);
        for (player_id, team) in contenders {
            if team != winner_team {
                self.record_reward_event(player_id, -4.0);
            }
        }
    }

    fn recent_possession_reward_weights(
        &self,
        team: Team,
        primary_player: Option<usize>,
        max_recipients: usize,
    ) -> Vec<(usize, f64)> {
        fn add_weight(weights: &mut Vec<(usize, f64)>, player_id: usize, amount: f64) {
            if let Some((_, weight)) = weights.iter_mut().find(|(id, _)| *id == player_id) {
                *weight += amount;
            } else {
                weights.push((player_id, amount));
            }
        }

        let mut weights = Vec::new();
        for (rank, player_id) in self
            .possession_chain
            .iter()
            .rev()
            .copied()
            .take(max_recipients.max(1))
            .enumerate()
        {
            if self
                .players
                .get(player_id)
                .is_some_and(|player| player.team == team)
            {
                let recency_weight = match rank {
                    0 => 5.0,
                    1 => 3.0,
                    2 => 2.0,
                    _ => 1.0 / (rank as f64),
                };
                add_weight(&mut weights, player_id, recency_weight);
            }
        }
        if let Some(primary_player) = primary_player {
            if self
                .players
                .get(primary_player)
                .is_some_and(|player| player.team == team)
            {
                add_weight(&mut weights, primary_player, 4.0);
            }
        }
        weights
    }

    fn record_possession_reward_pool(
        &mut self,
        team: Team,
        primary_player: Option<usize>,
        total_points: f64,
        max_recipients: usize,
    ) {
        if total_points <= 0.0 {
            return;
        }
        let weights = self.recent_possession_reward_weights(team, primary_player, max_recipients);
        let total_weight = weights.iter().map(|(_, weight)| *weight).sum::<f64>();
        if total_weight <= 1e-9 {
            return;
        }
        for (player_id, weight) in weights {
            self.record_reward_event(player_id, total_points * weight / total_weight);
        }
    }

    fn record_shot_on_target_rewards(&mut self, shooting_team: Team, shooter: usize) {
        self.record_possession_touch(shooter);
        self.record_possession_reward_pool(
            shooting_team,
            Some(shooter),
            SHOT_ON_TARGET_REWARD_POINTS,
            8,
        );
    }

    fn record_goal_rewards(&mut self, scoring_team: Team, shooter: Option<usize>) {
        if let Some(shooter) = shooter {
            self.record_possession_touch(shooter);
        }
        self.record_possession_reward_pool(scoring_team, shooter, GOAL_REWARD_POINTS, 10);

        for player in self
            .players
            .iter()
            .filter(|player| player.team == scoring_team.other())
            .filter(|player| matches!(player.role, PlayerRole::Goalkeeper | PlayerRole::Defender))
            .filter(|player| {
                let own_goal_y = scoring_team.goal_y(self.config.field_length_yards);
                (player.position.y - own_goal_y).abs() <= 32.0
            })
            .map(|player| player.id)
            .collect::<Vec<_>>()
        {
            self.record_reward_event(player, -5.0);
        }
    }

    fn record_possession_team_reward_at(&mut self, tick: u64, team: Team, amount: f64) {
        if amount <= 1e-9 {
            return;
        }
        let mut recipients = Vec::new();
        if let Some(holder) = self.ball.holder {
            if self
                .players
                .get(holder)
                .is_some_and(|player| player.team == team)
            {
                recipients.push(holder);
            }
        }
        for player_id in self.possession_chain.iter().rev().copied().take(6) {
            if recipients.contains(&player_id) {
                continue;
            }
            if self
                .players
                .get(player_id)
                .is_some_and(|player| player.team == team)
            {
                recipients.push(player_id);
            }
        }
        if recipients.is_empty() {
            return;
        }

        if recipients.len() == 1 {
            self.record_reward_event_at(tick, recipients[0], amount);
            return;
        }

        let holder_amount = amount * 0.60;
        self.record_reward_event_at(tick, recipients[0], holder_amount);
        let support_amount = amount * 0.40 / (recipients.len() - 1) as f64;
        for player_id in recipients.into_iter().skip(1) {
            self.record_reward_event_at(tick, player_id, support_amount);
        }
    }

    fn add_extra_chase_fatigue(&mut self, player_id: usize, load: f64, reward_cost: f64) {
        let Some(player) = self.players.get_mut(player_id) else {
            return;
        };
        let stamina_resistance = (1.18 - ability01(player.skills.stamina) * 0.58).clamp(0.58, 1.18);
        let fatigue_multiplier = 1.0 + player.fatigue.clamp(0.0, 1.0) * 0.45;
        let dt = self.config.dt_seconds.max(1e-6);
        let load_rate = load / dt;
        let load_delta = load_rate.max(0.0) / 12.0 * 0.006 * dt;
        let reward_delta = reward_cost.max(0.0) * 0.006;
        let delta = ((load_delta + reward_delta) * stamina_resistance * fatigue_multiplier)
            .clamp(0.0, 0.018);
        player.fatigue = (player.fatigue + delta).clamp(0.0, 1.0);
    }

    fn record_possession_chase_stats(
        &mut self,
        defending_team: Team,
        possession_team: Team,
        defensive_load: f64,
        advantage: f64,
    ) {
        match defending_team {
            Team::Home => self.stats.defensive_chase_load_home += defensive_load.max(0.0),
            Team::Away => self.stats.defensive_chase_load_away += defensive_load.max(0.0),
        }
        match possession_team {
            Team::Home => self.stats.possession_chase_advantage_home += advantage.max(0.0),
            Team::Away => self.stats.possession_chase_advantage_away += advantage.max(0.0),
        }
    }

    fn apply_possession_chase_signal(
        &mut self,
        tick: u64,
        signal: PossessionChaseSignal,
        record_rewards: bool,
    ) {
        let advantage = (signal.defensive_load - signal.attacking_load).max(0.0);
        self.record_possession_chase_stats(
            signal.defending_team,
            signal.possession_team,
            signal.defensive_load,
            advantage,
        );
        if record_rewards {
            self.record_possession_team_reward_at(
                tick,
                signal.possession_team,
                signal.attacking_credit,
            );
        }
        for penalty in signal.defender_penalties {
            self.add_extra_chase_fatigue(penalty.player_id, penalty.load, penalty.amount.abs());
            if record_rewards {
                self.record_reward_event_at(tick, penalty.player_id, penalty.amount);
            }
        }
    }

    fn apply_defensive_relaxation_signal(
        &mut self,
        tick: u64,
        signal: DefensiveRelaxationSignal,
        record_rewards: bool,
    ) {
        self.record_possession_chase_stats(
            signal.defending_team,
            signal.possession_team,
            0.0,
            signal.attacking_credit * 11.0,
        );
        if !record_rewards {
            return;
        }
        self.record_possession_team_reward_at(
            tick,
            signal.possession_team,
            signal.attacking_credit,
        );
        for penalty in signal.defender_penalties {
            self.record_reward_event_at(tick, penalty.player_id, penalty.amount);
        }
    }

    fn update_possession_chase_trackers(
        &mut self,
        before: &WorldSnapshot,
        after: &WorldSnapshot,
        record_rewards: bool,
    ) {
        let Some(possession_team) = before.controlled_possession_team() else {
            return;
        };
        if let Some(signal) = possession_chase_signal(before, after, possession_team) {
            self.apply_possession_chase_signal(before.tick, signal, record_rewards);
        }
        if let Some(signal) = defensive_relaxation_signal(before, after, possession_team) {
            self.apply_defensive_relaxation_signal(before.tick, signal, record_rewards);
        }
    }

    fn facing_for_player_action(&self, player_id: usize, action: &SoccerAction) -> FacingBucket {
        let Some(player) = self.players.get(player_id) else {
            return FacingBucket::Unknown;
        };
        let target = match action {
            SoccerAction::HoldShape => Some(player.home_position),
            SoccerAction::MoveTo(target)
            | SoccerAction::Dribble(target)
            | SoccerAction::ControlTouch { target } => Some(*target),
            SoccerAction::Pass { target_player, .. } => target_player
                .and_then(|target_id| self.players.get(target_id).map(|target| target.position)),
            SoccerAction::Shoot { .. } => Some(Vec2::new(
                self.config.field_width_yards * 0.5,
                player.team.goal_y(self.config.field_length_yards),
            )),
            SoccerAction::Tackle { target_player } => self
                .players
                .get(*target_player)
                .map(|target| target.position),
        };
        let facing = target
            .map(|target| facing_bucket_from_vector(target - player.position))
            .unwrap_or(FacingBucket::Unknown);
        if facing == FacingBucket::Unknown {
            player.action_facing
        } else {
            facing
        }
    }

    fn apply_player_intent(&mut self, intent: PlayerIntent) {
        if intent.player_id >= self.players.len() {
            return;
        }
        let player_id = intent.player_id;
        let action_facing = self.facing_for_player_action(player_id, &intent.action);
        self.players[player_id].action_facing = action_facing;
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
                let snapshot = WorldSnapshot::from_match(self);
                let observation = snapshot.observation_for(player_id);
                let pressure = pressure_from_observation(&observation);
                let dribble_dir = (target - player_pos).normalized();
                self.move_player_towards(player_id, target, true);
                if self.ball.holder == Some(player_id) {
                    let touch_probability =
                        dribble_heavy_touch_probability(&self.players[player_id], pressure);
                    let player = &self.players[player_id];
                    let dir = if dribble_dir.len() > 0.0 {
                        dribble_dir
                    } else {
                        carried_ball_lead(player).normalized()
                    };
                    if self.rng.next_float()
                        < time_window_probability(touch_probability, self.config.dt_seconds)
                    {
                        let touch_distance = DRIBBLE_HEAVY_TOUCH_MIN_YARDS
                            + pressure * 0.9
                            + (1.0 - ability01(player.skills.dribbling)) * 0.65;
                        self.ball.holder = None;
                        self.ball.position = (player.position + dir * touch_distance)
                            .clamp_to_pitch(
                                self.config.field_width_yards,
                                self.config.field_length_yards,
                            );
                        self.ball.velocity =
                            player.velocity + dir * (4.0 + pressure * 8.0 + touch_distance);
                        self.ball.altitude_yards = 0.0;
                        self.ball.last_touch_team = Some(player_team);
                        self.pending_pass = None;
                        self.pending_shot = None;
                        self.events.push(MatchEvent {
                            tick: self.tick,
                            clock_seconds: self.clock_seconds,
                            kind: "heavy-touch".to_string(),
                            team: Some(player_team),
                            player_id: Some(player_id),
                            description: format!("{} heavy touch", player.name),
                        });
                    } else {
                        self.ball.position = (player.position + carried_ball_lead(player))
                            .clamp_to_pitch(
                                self.config.field_width_yards,
                                self.config.field_length_yards,
                            );
                        self.ball.velocity = player.velocity;
                        self.ball.altitude_yards = 0.0;
                        self.ball.last_touch_team = Some(player_team);
                    }
                }
            }
            SoccerAction::ControlTouch { target } => {
                self.move_player_towards(player_id, target, false);
                if self.ball.holder == Some(player_id) {
                    let player = &mut self.players[player_id];
                    self.ball.position = (player.position + carried_ball_lead(player) * 0.45)
                        .clamp_to_pitch(
                            self.config.field_width_yards,
                            self.config.field_length_yards,
                        );
                    self.ball.velocity = player.velocity * 0.35;
                    self.ball.altitude_yards = 0.0;
                    self.ball.last_touch_team = Some(player_team);
                    player.incoming_ball = None;
                    self.pending_pass = None;
                    self.pending_shot = None;
                }
            }
            SoccerAction::Pass {
                target_player,
                power,
                flight,
            } => {
                if self.ball.holder == Some(player_id) {
                    let snapshot = WorldSnapshot::from_match(self);
                    let observation = snapshot.observation_for(player_id);
                    let target_id = target_player.or_else(|| {
                        if flight.is_aerial() {
                            snapshot.best_aerial_pass_target(player_id)
                        } else {
                            snapshot.best_pass_target(player_id)
                        }
                    });
                    let target = target_id
                        .and_then(|id| {
                            if flight.is_aerial() {
                                snapshot.projected_in_behind_pass_point(player_id, id)
                            } else {
                                None
                            }
                            .or_else(|| {
                                self.players.iter().find(|p| p.id == id).map(|p| p.position)
                            })
                        })
                        .unwrap_or_else(|| {
                            Vec2::new(player_pos.x, player_pos.y + 18.0 * player_team.attack_dir())
                                .clamp_to_pitch(
                                    self.config.field_width_yards,
                                    self.config.field_length_yards,
                                )
                        });
                    let distance = player_pos.distance(target);
                    let pressure = pressure_from_observation(&observation);
                    let is_cross = pass_would_be_cross(
                        player_pos,
                        target,
                        player_team,
                        self.config.field_width_yards,
                        self.config.field_length_yards,
                    );
                    let pass_skill =
                        pass_execution_skill(&self.players[player_id].skills, flight, is_cross);
                    let aimed_target = noisy_pass_target(
                        player_pos,
                        target,
                        pass_skill,
                        pressure,
                        distance,
                        &mut self.rng,
                    )
                    .clamp_to_pitch(
                        self.config.field_width_yards,
                        self.config.field_length_yards,
                    );
                    let speed = (16.0 + 16.0 * power.clamp(0.0, 1.0))
                        * if flight.is_aerial() { 0.92 } else { 1.0 };
                    self.ball.holder = None;
                    self.ball.position = player_pos;
                    self.ball.velocity = (aimed_target - player_pos).normalized() * speed;
                    self.ball.altitude_yards = if flight.is_aerial() { 0.05 } else { 0.0 };
                    self.ball.last_touch_team = Some(player_team);
                    self.players[player_id].incoming_ball = None;
                    let offside = target_id
                        .and_then(|target| snapshot.pending_offside_for_pass(player_id, target));
                    self.pending_pass = Some(PendingPass {
                        team: player_team,
                        from: player_id,
                        target: target_id,
                        flight,
                        is_cross,
                        origin: player_pos,
                        intended_target: target,
                        distance_yards: distance,
                        offside,
                    });
                    self.pending_shot = None;
                    self.record_possession_touch(player_id);
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
                            ability01(self.players[player_id].skills.shooting),
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
                    self.ball.altitude_yards = 0.0;
                    self.ball.last_touch_team = Some(player_team);
                    self.players[player_id].incoming_ball = None;
                    self.pending_pass = None;
                    self.pending_shot = Some(PendingShot {
                        team: player_team,
                        shooter: player_id,
                    });
                    self.record_possession_touch(player_id);
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
                        self.ball.altitude_yards = 0.0;
                        self.ball.last_touch_team = Some(player_team);
                        self.mark_ball_received(player_id);
                        self.record_reward_event(player_id, 10.0);
                        self.record_possession_touch(player_id);
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
        let previous_acceleration = p.acceleration;
        let to_target = target - p.position;
        let gait = classify_movement_gait(p.team, to_target, sprint);
        p.movement_gait = gait;
        let fatigue_factor = fatigue_speed_factor(p.skills.stamina, p.fatigue);
        let speed =
            top_speed_yps_from_score(p.skills.top_speed) * fatigue_factor * gait.speed_multiplier();
        let desired = to_target.normalized() * speed;
        let acceleration_factor = (0.62 + fatigue_factor * 0.38).clamp(0.45, 1.05);
        p.velocity = approach_velocity(
            p.velocity,
            desired,
            acceleration_yps2_from_score(p.skills.acceleration) * acceleration_factor,
            dt,
        );
        let movement_facing = facing_bucket_from_vector(p.velocity);
        if movement_facing != FacingBucket::Unknown {
            p.action_facing = movement_facing;
        }
        p.acceleration = if dt > 0.0 {
            (p.velocity - previous_velocity) / dt
        } else {
            Vec2::zero()
        };
        p.jerk = if dt > 0.0 {
            (p.acceleration - previous_acceleration) / dt
        } else {
            Vec2::zero()
        };
        p.position += p.velocity * dt;
        p.position = p.position.clamp_to_pitch(
            self.config.field_width_yards,
            self.config.field_length_yards,
        );
        p.fatigue = (p.fatigue + gait.fatigue_delta(p.skills.stamina, dt)).clamp(0.0, 1.0);
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
            ball_air_resistance: self.config.ball_air_resistance,
            ball_grass_resistance_yps2: self.config.ball_grass_resistance_yps2,
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
            self.pending_pass.as_ref(),
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
                let pending_pass_for_reward = self.pending_pass.clone();
                let incoming_context = self.pending_pass.as_ref().map(|pass| {
                    incoming_context_from_pass(pass, holder, self.ball.velocity.len(), self.tick)
                });
                self.ball.holder = Some(holder);
                self.ball.altitude_yards = 0.0;
                self.ball.last_touch_team = Some(holder_team);
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
                self.mark_ball_received(holder);
                self.record_possession_touch(holder);
                self.record_duel_rewards(holder);
                if let Some(player) = self.players.iter_mut().find(|player| player.id == holder) {
                    player.incoming_ball = incoming_context;
                }
                match possession_result {
                    BallPossessionResult::PassCompleted(team) => {
                        if let Some(pass) = pending_pass_for_reward.as_ref() {
                            self.record_completed_pass_reward(pass, holder);
                        }
                        self.pending_pass = None;
                        self.stat_pass_completed(team);
                    }
                    BallPossessionResult::Interception(team) => {
                        self.record_interception_reward(holder, pending_pass_for_reward.as_ref());
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
                self.ball.altitude_yards = 0.0;
                self.stat_shot_on_target(shot.team);
                self.record_shot_on_target_rewards(shot.team, shot.shooter);
                self.stat_save(defending_team);
                self.mark_ball_received(keeper_id);
                if let Some(keeper) = self
                    .players
                    .iter_mut()
                    .find(|player| player.id == keeper_id)
                {
                    keeper.position = save_position;
                    keeper.velocity = Vec2::zero();
                    keeper.acceleration = Vec2::zero();
                    keeper.jerk = Vec2::zero();
                    keeper.movement_gait = MovementGait::Stand;
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
                if let Some(shot) = shot.as_ref() {
                    self.record_shot_on_target_rewards(shot.team, shot.shooter);
                }
                self.record_goal_rewards(scoring_team, shot.as_ref().map(|shot| shot.shooter));
                self.ball.altitude_yards = 0.0;
                if let Some(shot) = shot {
                    self.pending_shot = None;
                    self.stat_shot_on_target(shot.team);
                }
                self.score_goal(scoring_team);
            }
            BallStepOutcome::Miss { shot } => {
                self.pending_shot = None;
                self.ball.altitude_yards = 0.0;
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
            BallStepOutcome::ShotBlocked {
                shot,
                blocker_id,
                defending_team,
                position,
                deflection_kind,
                restart,
            } => {
                self.pending_pass = None;
                self.pending_shot = None;
                self.ball.position = position.clamp_to_pitch(
                    self.config.field_width_yards,
                    self.config.field_length_yards,
                );
                self.ball.holder = None;
                self.ball.altitude_yards = 0.0;
                self.ball.last_touch_team = Some(defending_team);
                self.record_reward_event(blocker_id, 12.0);
                self.record_reward_event(shot.shooter, 2.0);
                let blocker_name = self
                    .players
                    .iter()
                    .find(|player| player.id == blocker_id)
                    .map(|player| player.name.as_str())
                    .unwrap_or("Defender");
                let shooter_name = self
                    .players
                    .iter()
                    .find(|player| player.id == shot.shooter)
                    .map(|player| player.name.as_str())
                    .unwrap_or("shooter");
                let detail = match deflection_kind {
                    ShotDeflectionKind::CornerKick => "behind for a corner",
                    ShotDeflectionKind::GoalBound => "off the goal-bound line",
                    ShotDeflectionKind::Rebound => "back into play",
                };
                self.events.push(MatchEvent {
                    tick: self.tick,
                    clock_seconds: self.clock_seconds,
                    kind: "shot-blocked".to_string(),
                    team: Some(defending_team),
                    player_id: Some(blocker_id),
                    description: format!("{blocker_name} blocked {shooter_name}'s shot {detail}"),
                });
                if let Some(restart) = restart {
                    self.apply_restart(restart);
                }
            }
            BallStepOutcome::OutOfPlay { restart, shot } => {
                self.pending_pass = None;
                self.pending_shot = None;
                self.ball.altitude_yards = 0.0;
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
                holder.jerk = Vec2::zero();
                holder.movement_gait = MovementGait::Stand;
                holder.record_position_history();
            }
        }

        self.arrange_restart_shape(
            restart.kind,
            restart.awarded_team,
            restart.position,
            restart_holder,
        );

        match restart.kind {
            BallRestartKind::FreeKick => {
                if let Some(center_ref) = self
                    .officials
                    .iter_mut()
                    .find(|official| official.kind == OfficialKind::CenterReferee)
                {
                    center_ref.position = restart.position;
                    center_ref.velocity = Vec2::zero();
                    center_ref.acceleration = Vec2::zero();
                    center_ref.jerk = Vec2::zero();
                    center_ref.record_position_history();
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
                    assistant.acceleration = Vec2::zero();
                    assistant.jerk = Vec2::zero();
                    assistant.record_position_history();
                }
            }
        }

        self.ball.position = restart.position;
        self.ball.velocity = Vec2::zero();
        self.ball.holder = restart_holder;
        self.ball.last_touch_team = Some(restart.awarded_team);
        if let Some(holder_id) = restart_holder {
            self.mark_ball_received(holder_id);
            self.record_possession_touch(holder_id);
        }
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

    fn restart_team_ids(
        &self,
        team: Team,
        include_goalkeeper: bool,
        exclude: Option<usize>,
    ) -> Vec<usize> {
        let mut ids = self
            .players
            .iter()
            .filter(|player| player.team == team)
            .filter(|player| include_goalkeeper || player.role != PlayerRole::Goalkeeper)
            .filter(|player| exclude != Some(player.id))
            .map(|player| {
                let role_rank = match player.role {
                    PlayerRole::Forward => 0,
                    PlayerRole::Midfielder => 1,
                    PlayerRole::Defender => 2,
                    PlayerRole::Goalkeeper => 3,
                };
                (player.id, role_rank, player.home_position.x)
            })
            .collect::<Vec<_>>();
        ids.sort_by(|a, b| {
            a.1.cmp(&b.1)
                .then_with(|| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal))
        });
        ids.into_iter().map(|(id, _, _)| id).collect()
    }

    fn set_dead_ball_player_position(&mut self, player_id: usize, position: Vec2) {
        let width = self.config.field_width_yards;
        let length = self.config.field_length_yards;
        if let Some(player) = self.players.iter_mut().find(|p| p.id == player_id) {
            player.position = position.clamp_to_pitch(width, length);
            player.velocity = Vec2::zero();
            player.acceleration = Vec2::zero();
            player.jerk = Vec2::zero();
            player.movement_gait = MovementGait::Stand;
            player.record_position_history();
        }
    }

    fn arrange_restart_shape(
        &mut self,
        kind: BallRestartKind,
        awarded_team: Team,
        spot: Vec2,
        taker: Option<usize>,
    ) {
        match kind {
            BallRestartKind::CornerKick => {
                self.arrange_corner_shape(awarded_team, spot, taker);
            }
            BallRestartKind::GoalKick => {
                self.arrange_goal_kick_shape(awarded_team, spot, taker);
            }
            BallRestartKind::ThrowIn => {
                self.arrange_throw_in_shape(awarded_team, spot, taker);
            }
            BallRestartKind::FreeKick => {
                self.arrange_free_kick_shape(awarded_team, spot, taker);
            }
        }
        if let Some(taker_id) = taker {
            self.set_dead_ball_player_position(taker_id, spot);
        }
    }

    fn arrange_corner_shape(&mut self, attacking_team: Team, spot: Vec2, taker: Option<usize>) {
        let width = self.config.field_width_yards;
        let length = self.config.field_length_yards;
        let dir = attacking_team.attack_dir();
        let goal_y = attacking_team.goal_y(length);
        let goal_center = Vec2::new(width * 0.5, goal_y);
        let attacking_slots = [
            Vec2::new(width * 0.42, goal_y - dir * 7.0),
            Vec2::new(width * 0.55, goal_y - dir * 8.5),
            Vec2::new(width * 0.34, goal_y - dir * 12.0),
            Vec2::new(width * 0.64, goal_y - dir * 13.5),
            Vec2::new(width * 0.50, goal_y - dir * 18.0),
            Vec2::new(width * 0.28, goal_y - dir * 22.0),
            Vec2::new(width * 0.72, goal_y - dir * 24.0),
        ];
        for (player_id, slot) in self
            .restart_team_ids(attacking_team, false, taker)
            .into_iter()
            .zip(attacking_slots)
        {
            self.set_dead_ball_player_position(player_id, slot);
        }

        let defending_team = attacking_team.other();
        if let Some(keeper_id) = self.goalkeeper_for(defending_team) {
            self.set_dead_ball_player_position(keeper_id, goal_center - Vec2::new(0.0, dir * 2.0));
        }
        let defending_slots = [
            Vec2::new(width * 0.40, goal_y - dir * 4.5),
            Vec2::new(width * 0.56, goal_y - dir * 5.5),
            Vec2::new(width * 0.32, goal_y - dir * 8.5),
            Vec2::new(width * 0.68, goal_y - dir * 9.0),
            Vec2::new(width * 0.48, goal_y - dir * 13.0),
            Vec2::new(width * 0.22, goal_y - dir * 17.0),
            Vec2::new(width * 0.78, goal_y - dir * 18.0),
        ];
        for (player_id, slot) in self
            .restart_team_ids(defending_team, false, None)
            .into_iter()
            .zip(defending_slots)
        {
            self.set_dead_ball_player_position(player_id, slot);
        }

        if let Some(taker_id) = taker {
            self.set_dead_ball_player_position(taker_id, spot);
        }
    }

    fn arrange_goal_kick_shape(&mut self, team: Team, spot: Vec2, taker: Option<usize>) {
        let width = self.config.field_width_yards;
        let length = self.config.field_length_yards;
        let dir = team.attack_dir();
        let own_goal_y = team.other().goal_y(length);
        let outlet_slots = [
            Vec2::new(width * 0.20, own_goal_y + dir * 18.0),
            Vec2::new(width * 0.36, own_goal_y + dir * 22.0),
            Vec2::new(width * 0.64, own_goal_y + dir * 22.0),
            Vec2::new(width * 0.80, own_goal_y + dir * 18.0),
            Vec2::new(width * 0.34, own_goal_y + dir * 36.0),
            Vec2::new(width * 0.66, own_goal_y + dir * 36.0),
        ];
        for (player_id, slot) in self
            .restart_team_ids(team, false, taker)
            .into_iter()
            .rev()
            .zip(outlet_slots)
        {
            self.set_dead_ball_player_position(player_id, slot);
        }
        for (player_id, slot) in self
            .restart_team_ids(team.other(), false, None)
            .into_iter()
            .zip([
                Vec2::new(width * 0.28, own_goal_y + dir * 45.0),
                Vec2::new(width * 0.50, own_goal_y + dir * 50.0),
                Vec2::new(width * 0.72, own_goal_y + dir * 45.0),
                Vec2::new(width * 0.38, own_goal_y + dir * 58.0),
            ])
        {
            self.set_dead_ball_player_position(player_id, slot);
        }
        if let Some(taker_id) = taker.or_else(|| self.goalkeeper_for(team)) {
            self.set_dead_ball_player_position(taker_id, spot);
        }
    }

    fn arrange_throw_in_shape(&mut self, team: Team, spot: Vec2, taker: Option<usize>) {
        let width = self.config.field_width_yards;
        let length = self.config.field_length_yards;
        let dir = team.attack_dir();
        let side_sign = if spot.x <= width * 0.5 { 1.0 } else { -1.0 };
        let inside_x = (spot.x + side_sign * 8.0).clamp(4.0, width - 4.0);
        let receiver_slots = [
            Vec2::new(inside_x, spot.y - dir * 7.0),
            Vec2::new(inside_x + side_sign * 8.0, spot.y + dir * 6.0),
            Vec2::new(inside_x, spot.y + dir * 15.0),
            Vec2::new(width * 0.5, spot.y + dir * 10.0),
        ];
        for (player_id, slot) in self
            .restart_team_ids(team, false, taker)
            .into_iter()
            .zip(receiver_slots)
        {
            self.set_dead_ball_player_position(player_id, slot.clamp_to_pitch(width, length));
        }

        let defender_slots = receiver_slots.map(|slot| {
            Vec2::new(slot.x + side_sign * 2.0, slot.y + dir * 2.8).clamp_to_pitch(width, length)
        });
        for (player_id, slot) in self
            .restart_team_ids(team.other(), false, None)
            .into_iter()
            .zip(defender_slots)
        {
            self.set_dead_ball_player_position(player_id, slot);
        }
    }

    fn arrange_free_kick_shape(&mut self, team: Team, spot: Vec2, taker: Option<usize>) {
        let width = self.config.field_width_yards;
        let length = self.config.field_length_yards;
        let dir = team.attack_dir();
        let goal = Vec2::new(width * 0.5, team.goal_y(length));
        let distance_to_goal = spot.distance(goal);
        if distance_to_goal <= 32.0 {
            let defending_team = team.other();
            if let Some(keeper_id) = self.goalkeeper_for(defending_team) {
                self.set_dead_ball_player_position(
                    keeper_id,
                    Vec2::new(width * 0.5, goal.y - dir * 2.0),
                );
            }
            let to_goal = (goal - spot).normalized();
            let wall_center = (spot + to_goal * 10.0).clamp_to_pitch(width, length);
            let lateral = Vec2::new(-to_goal.y, to_goal.x).normalized();
            let wall_ids = self.restart_team_ids(defending_team, false, None);
            for (player_id, offset) in wall_ids.into_iter().take(4).zip([-3.0, -1.0, 1.0, 3.0]) {
                self.set_dead_ball_player_position(player_id, wall_center + lateral * offset);
            }
            let attack_slots = [
                Vec2::new(spot.x + 5.0, spot.y - dir * 2.0),
                Vec2::new(width * 0.38, goal.y - dir * 9.0),
                Vec2::new(width * 0.60, goal.y - dir * 10.0),
                Vec2::new(width * 0.50, goal.y - dir * 16.0),
            ];
            for (player_id, slot) in self
                .restart_team_ids(team, false, taker)
                .into_iter()
                .zip(attack_slots)
            {
                self.set_dead_ball_player_position(player_id, slot);
            }
        } else {
            let support_slots = [
                Vec2::new(spot.x - 8.0, spot.y - dir * 5.0),
                Vec2::new(spot.x + 8.0, spot.y - dir * 4.0),
                Vec2::new(spot.x, spot.y + dir * 10.0),
                Vec2::new(width * 0.34, spot.y + dir * 18.0),
                Vec2::new(width * 0.66, spot.y + dir * 18.0),
            ];
            for (player_id, slot) in self
                .restart_team_ids(team, false, taker)
                .into_iter()
                .zip(support_slots)
            {
                self.set_dead_ball_player_position(player_id, slot.clamp_to_pitch(width, length));
            }
        }
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
                holder.jerk = Vec2::zero();
                holder.movement_gait = MovementGait::Stand;
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
            assistant.acceleration = Vec2::zero();
            assistant.jerk = Vec2::zero();
            assistant.record_position_history();
        }

        self.stat_offside(offside.team);
        self.ball.position = restart_spot;
        self.ball.velocity = Vec2::zero();
        self.ball.holder = restart_holder;
        self.ball.last_touch_team = Some(defending_team);
        if let Some(holder_id) = restart_holder {
            self.mark_ball_received(holder_id);
            self.record_possession_touch(holder_id);
        }
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

    fn update_defensive_reward_trackers(&mut self, before: &WorldSnapshot, after: &WorldSnapshot) {
        let Some(holder_id) = before.ball.holder else {
            self.defensive_delay_clocks.clear();
            self.defensive_beat_clocks.clear();
            return;
        };
        let Some(holder_before) = before.players.iter().find(|player| player.id == holder_id)
        else {
            return;
        };
        let holder_team = holder_before.team;
        let holder_before_position = before
            .player_position(holder_id)
            .unwrap_or(holder_before.position);
        let holder_after_position = after
            .player_position(holder_id)
            .or_else(|| after.ball.holder.and_then(|id| after.player_position(id)))
            .unwrap_or(after.ball.position);
        let attacker_progress =
            (holder_after_position.y - holder_before_position.y) * holder_team.attack_dir();
        let retained_attack = after.possession_team() == Some(holder_team);
        let dt = self.config.dt_seconds.max(0.0);

        let mut delay_rewards = Vec::new();
        let mut beat_penalties = Vec::new();
        for defender in before
            .players
            .iter()
            .filter(|player| player.team != holder_team)
        {
            let current_action = self
                .players
                .get(defender.id)
                .and_then(|player| player.last_decision.as_ref())
                .map(|decision| normalize_soccer_action_label(&decision.action))
                .unwrap_or("hold");
            let defender_before_position = before
                .player_position(defender.id)
                .unwrap_or(defender.position);
            let defender_after_position = after
                .player_position(defender.id)
                .unwrap_or(defender_before_position);
            let distance_to_holder = defender_before_position.distance(holder_before_position);
            let active_defense = matches!(current_action, "defend" | "tackle");
            let delayed = active_defense
                && retained_attack
                && distance_to_holder <= 6.0
                && attacker_progress <= 0.8;
            let delay_clock = self
                .defensive_delay_clocks
                .entry(defender.id)
                .or_insert(0.0);
            if delayed {
                *delay_clock += dt;
                if *delay_clock >= 2.0 {
                    delay_rewards.push(defender.id);
                    *delay_clock -= 2.0;
                }
            } else {
                *delay_clock = 0.0;
            }

            let lateral_gap = (holder_after_position.x - defender_after_position.x).abs();
            let attacker_ahead = (holder_after_position.y - defender_after_position.y)
                * holder_team.attack_dir()
                > 0.35;
            let beaten = active_defense
                && retained_attack
                && distance_to_holder <= 5.2
                && lateral_gap <= 5.0
                && attacker_ahead
                && attacker_progress > 0.10;
            let beat_clock = self.defensive_beat_clocks.entry(defender.id).or_insert(0.0);
            if beaten {
                *beat_clock += dt;
                if *beat_clock >= 0.25 {
                    beat_penalties.push(defender.id);
                    *beat_clock = 0.0;
                }
            } else {
                *beat_clock = 0.0;
            }
        }

        for defender_id in delay_rewards {
            self.record_reward_event_at(before.tick, defender_id, 2.0);
        }
        for defender_id in beat_penalties {
            self.record_reward_event_at(before.tick, defender_id, -3.0);
        }
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
        let center = Vec2::new(
            self.config.field_width_yards * 0.5,
            self.config.field_length_yards * 0.5,
        );
        for p in &mut self.players {
            p.position = p.home_position;
            p.velocity = Vec2::zero();
            p.acceleration = Vec2::zero();
            p.jerk = Vec2::zero();
            p.movement_gait = MovementGait::Stand;
            p.receive_facing = FacingBucket::Unknown;
            p.action_facing = default_team_facing(p.team);
            p.record_position_history();
        }
        let kickoff = self
            .players
            .iter()
            .filter(|p| p.team == kickoff_team)
            .min_by(|a, b| {
                a.position
                    .distance(center)
                    .partial_cmp(&b.position.distance(center))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|p| p.id);
        if let Some(holder_id) = kickoff {
            if let Some(holder) = self.players.iter_mut().find(|p| p.id == holder_id) {
                holder.position = center;
                holder.velocity = Vec2::zero();
                holder.acceleration = Vec2::zero();
                holder.jerk = Vec2::zero();
                holder.movement_gait = MovementGait::Stand;
                holder.record_position_history();
            }
        }
        self.ball.position = center;
        self.ball.velocity = Vec2::zero();
        self.ball.holder = kickoff;
        self.ball.last_touch_team = Some(kickoff_team);
        self.possession_chain.clear();
        if let Some(holder_id) = kickoff {
            self.mark_ball_received(holder_id);
            self.record_possession_touch(holder_id);
        }
        self.ball.record_decision(self.tick, "kickoff");
        self.pending_pass = None;
        self.pending_shot = None;
        self.shared_positions
            .sync_from_players(&self.players, self.tick, self.clock_seconds);
        let taker = kickoff
            .and_then(|id| self.players.iter().find(|player| player.id == id))
            .map(|player| player.name.as_str())
            .unwrap_or("Kickoff");
        self.events.push(MatchEvent {
            tick: self.tick,
            clock_seconds: self.clock_seconds,
            kind: "kickoff".to_string(),
            team: Some(kickoff_team),
            player_id: kickoff,
            description: format!("{taker} kickoff for {}", kickoff_team.label()),
        });
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
        let transitions =
            self.learning_transitions_for(before, after, score_home_before, score_away_before, &[]);
        self.learning_transitions.extend(transitions);
    }

    fn learning_transitions_for(
        &self,
        before: &WorldSnapshot,
        after: &WorldSnapshot,
        score_home_before: u32,
        score_away_before: u32,
        tick_reward_events: &[SoccerRewardEvent],
    ) -> Vec<SoccerLearningTransition> {
        let done = self.is_done();
        let mut transitions = Vec::new();
        for player in &self.players {
            let Some(decision) = &player.last_decision else {
                continue;
            };
            let reward = soccer_transition_reward_with_tactics(
                player,
                decision,
                before,
                after,
                score_home_before,
                score_away_before,
                self.score_home,
                self.score_away,
                false,
                &self.config.tactical_learning,
            );
            let reward = reward
                + tick_reward_events
                    .iter()
                    .filter(|event| event.tick == before.tick && event.player_id == player.id)
                    .map(|event| event.amount)
                    .sum::<f64>();
            transitions.push(SoccerLearningTransition {
                tick: before.tick,
                player_id: player.id,
                team: player.team,
                role: player.role,
                state: decision.mdp_state.clone(),
                observation: decision.observation.clone(),
                belief: decision.belief.clone(),
                action: decision.action.clone(),
                action_target: decision.action_target.clone(),
                reward,
                next_state: after.mdp_state_for_player(player.id),
                next_observation: after.observation_for(player.id),
                done,
            });
        }
        transitions
    }
}

pub struct SoccerRealtimeSession {
    sim: SoccerMatch,
    input_queue: SharedHumanInputs,
    controller_threads: Vec<HumanControllerThread>,
    emitted_event_cursor: usize,
    emitted_learning_cursor: usize,
    tracking_frames: Vec<SoccerTrackingFrame>,
}

impl SoccerRealtimeSession {
    pub fn new(config: MatchConfig) -> Self {
        Self::new_with_controller_threads(config, true)
    }

    pub fn new_without_controller_threads(config: MatchConfig) -> Self {
        Self::new_with_controller_threads(config, false)
    }

    fn new_with_controller_threads(config: MatchConfig, threaded_controllers: bool) -> Self {
        let input_queue = SharedHumanInputs::new();
        let controller_threads = if threaded_controllers {
            spawn_human_controller_threads(
                input_queue.clone(),
                config.human_slots(),
                Duration::from_millis(DEFAULT_CONTROLLER_DEBOUNCE_MS),
            )
            .unwrap_or_default()
        } else {
            Vec::new()
        };
        let mut sim = SoccerMatch::default_11v11(config)
            .with_human_inputs(input_queue.clone())
            .with_team_policies(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));
        sim.clear_controller_assignments();
        let tracking_frames = vec![tracking_frame_from_match(&sim)];
        SoccerRealtimeSession {
            sim,
            input_queue,
            controller_threads,
            emitted_event_cursor: 0,
            emitted_learning_cursor: 0,
            tracking_frames,
        }
    }

    pub fn from_match(sim: SoccerMatch) -> Self {
        Self::from_match_with_controller_threads(sim, true)
    }

    pub fn from_match_without_controller_threads(sim: SoccerMatch) -> Self {
        Self::from_match_with_controller_threads(sim, false)
    }

    fn from_match_with_controller_threads(sim: SoccerMatch, threaded_controllers: bool) -> Self {
        let input_queue = sim.human_inputs.clone();
        let controller_threads = if threaded_controllers {
            spawn_human_controller_threads(
                input_queue.clone(),
                sim.config.human_slots(),
                Duration::from_millis(DEFAULT_CONTROLLER_DEBOUNCE_MS),
            )
            .unwrap_or_default()
        } else {
            Vec::new()
        };
        let tracking_frames = vec![tracking_frame_from_match(&sim)];
        SoccerRealtimeSession {
            sim,
            input_queue,
            controller_threads,
            emitted_event_cursor: 0,
            emitted_learning_cursor: 0,
            tracking_frames,
        }
    }

    pub fn input_queue(&self) -> SharedHumanInputs {
        self.input_queue.clone()
    }

    pub fn spawn_human_controller_threads(
        &self,
        debounce_interval: Duration,
    ) -> Result<Vec<HumanControllerThread>, String> {
        spawn_human_controller_threads(
            self.input_queue.clone(),
            self.sim.config.human_slots(),
            debounce_interval,
        )
    }

    pub fn owned_controller_thread_count(&self) -> usize {
        self.controller_threads.len()
    }

    pub fn shared_positions(&self) -> SharedPlayerPositions {
        self.sim.shared_positions.clone()
    }

    pub fn push_human_input(&self, input: HumanInputFrame) -> bool {
        self.dispatch_human_input(input)
    }

    pub fn push_human_inputs<I>(&self, inputs: I) -> usize
    where
        I: IntoIterator<Item = HumanInputFrame>,
    {
        let mut latest_by_slot = HashMap::<usize, HumanInputFrame>::new();
        for input in inputs {
            if self
                .input_queue
                .last_seq_for_slot(input.controller_slot)
                .is_some_and(|last_seq| input.seq <= last_seq)
            {
                continue;
            }
            latest_by_slot
                .entry(input.controller_slot)
                .and_modify(|current| {
                    if input.seq > current.seq {
                        *current = input.clone();
                    }
                })
                .or_insert(input);
        }
        latest_by_slot
            .into_values()
            .filter(|input| self.dispatch_human_input(input.clone()))
            .count()
    }

    fn dispatch_human_input(&self, input: HumanInputFrame) -> bool {
        if let Some(controller) = self
            .controller_threads
            .iter()
            .find(|controller| controller.controller_slot() == input.controller_slot)
        {
            controller.send_input(input).unwrap_or(false)
        } else {
            self.input_queue.push(input)
        }
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
        let accepted_inputs = self.push_human_inputs(request.inputs);

        let ticks = request.ticks.max(1);
        let record_every = request.record_every_ticks.unwrap_or(1).max(1);
        let mut frames = Vec::new();
        for i in 0..ticks {
            if self.sim.is_done() {
                break;
            }
            self.sim.run_time_step();
            self.tracking_frames
                .push(tracking_frame_from_match(&self.sim));
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
            learning: self.sim.learning_snapshot(),
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

    pub fn update_ball_surface(
        &mut self,
        request: SoccerBallSurfaceRequest,
    ) -> Result<SoccerBallSurfaceResponse, String> {
        self.sim.update_ball_surface(request)?;
        Ok(SoccerBallSurfaceResponse {
            config: self.sim.config.clone(),
        })
    }

    pub fn update_learning_runtime(
        &mut self,
        request: SoccerLearningRuntimeRequest,
    ) -> SoccerLearningRuntimeResponse {
        self.sim.update_learning_runtime(request)
    }

    pub fn team_policy_artifact(&self) -> SoccerTeamPolicyArtifact {
        self.sim.team_policy_artifact()
    }

    pub fn tracking_dataset(&self) -> SoccerTrackingDataset {
        SoccerTrackingDataset {
            source: "live-session".to_string(),
            config: self.sim.config.clone(),
            frames: self.tracking_frames.clone(),
        }
    }

    pub fn import_team_policy_artifact(
        &mut self,
        artifact: SoccerTeamPolicyArtifact,
    ) -> Result<SoccerTeamPolicyImportResponse, String> {
        self.sim.import_team_policy_artifact(artifact)
    }

    pub fn import_tracking_for_team_policy(
        &mut self,
        request: SoccerTrackingImportRequest,
    ) -> Result<SoccerTrackingImportResponse, String> {
        self.sim.import_tracking_for_team_policy(request)
    }

    pub fn train_self_play_team_policy(
        &mut self,
        request: SoccerSelfPlayTrainingRequest,
    ) -> Result<SoccerSelfPlayTrainingResponse, String> {
        if request.episodes == 0 {
            return Err("self-play training episodes must be at least 1".to_string());
        }

        let mut config = self.sim.config.clone();
        if let Some(minutes) = request.minutes {
            if !minutes.is_finite() || minutes <= 0.0 {
                return Err("self-play training minutes must be positive and finite".to_string());
            }
            config.duration_seconds = minutes * 60.0;
        }
        if let Some(period_count) = request.period_count {
            if period_count == 0 {
                return Err("self-play training periodCount must be at least 1".to_string());
            }
            config.period_count = period_count;
        }
        if let Some(period_break_recovery_seconds) = request.period_break_recovery_seconds {
            if !period_break_recovery_seconds.is_finite() || period_break_recovery_seconds < 0.0 {
                return Err(
                    "self-play training periodBreakRecoverySeconds must be non-negative and finite"
                        .to_string(),
                );
            }
            config.period_break_recovery_seconds = period_break_recovery_seconds;
        }
        if let Some(dt_seconds) = request.dt_seconds {
            if !dt_seconds.is_finite() || dt_seconds <= 0.0 {
                return Err("self-play training dtSeconds must be positive and finite".to_string());
            }
            config.dt_seconds = dt_seconds;
        }
        if let Some(learning_interval_ticks) = request.learning_interval_ticks {
            if learning_interval_ticks == 0 {
                return Err(
                    "self-play training learningIntervalTicks must be at least 1".to_string(),
                );
            }
            config.learning_interval_ticks = learning_interval_ticks;
        }
        if let Some(seed) = request.seed {
            config.seed = seed;
        }
        if let Some(tactical_learning) = request.tactical_learning {
            tactical_learning.validate()?;
            config.tactical_learning = tactical_learning;
        }
        config.tactical_learning.validate()?;
        config.learning_enabled = true;
        config.learning_logging_enabled = false;
        config.max_human_players = 0;

        let options = request.options.unwrap_or_default();
        validate_soccer_q_policy_options(&options)?;
        let artifact = train_soccer_team_policies_from_self_play(config, request.episodes, options);
        let artifact_path =
            write_self_play_training_artifact(request.artifact_path.as_deref(), &artifact)?;

        let (imported_home_entries, imported_away_entries, learning) =
            if request.import_into_session {
                self.sim.config.tactical_learning = artifact.tactical_learning.clone();
                let import = self
                    .sim
                    .import_self_play_training_artifact(artifact.clone())?;
                (
                    import.imported_home_entries,
                    import.imported_away_entries,
                    import.learning,
                )
            } else {
                (0, 0, self.sim.learning_snapshot())
            };

        Ok(SoccerSelfPlayTrainingResponse {
            artifact_path,
            imported_home_entries,
            imported_away_entries,
            learning,
            artifact,
        })
    }

    pub fn state_response(&self) -> SoccerLiveStateResponse {
        SoccerLiveStateResponse {
            config: self.sim.config.clone(),
            frame: self.sim.to_frame(),
            learning: self.sim.learning_snapshot(),
            summary: self.sim.summary(),
            controller_assignments: self.sim.controller_assignments(),
            done: self.sim.is_done(),
        }
    }
}

fn tracking_frame_from_match(sim: &SoccerMatch) -> SoccerTrackingFrame {
    SoccerTrackingFrame {
        tick: sim.tick,
        clock_seconds: sim.clock_seconds,
        ball_position: sim.ball.position,
        ball_velocity: Some(sim.ball.velocity),
        ball_altitude_yards: Some(sim.ball.altitude_yards),
        pass_flight: sim.pending_pass.as_ref().map(|pass| pass.flight),
        ball_holder: sim.ball.holder,
        last_touch_team: sim.ball.last_touch_team,
        score_home: Some(sim.score_home),
        score_away: Some(sim.score_away),
        players: sim
            .players
            .iter()
            .map(|player| SoccerTrackingPlayerSample {
                id: player.id,
                name: Some(player.name.clone()),
                team: player.team,
                role: player.role,
                shirt: Some(player.shirt),
                position: player.position,
                velocity: Some(player.velocity),
                home_position: Some(player.home_position),
                skills: Some(player.skills.clone()),
            })
            .collect(),
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
                if request_body_complete(&data) || data.len() > LIVE_HTTP_MAX_REQUEST_BYTES {
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

fn write_self_play_training_artifact(
    artifact_path: Option<&str>,
    artifact: &SoccerSelfPlayTrainingArtifact,
) -> Result<Option<String>, String> {
    let Some(path) = artifact_path.map(str::trim).filter(|path| !path.is_empty()) else {
        return Ok(None);
    };
    let json = serde_json::to_string_pretty(artifact)
        .map_err(|e| format!("serialize self-play training artifact: {e}"))?;
    let path_ref = std::path::Path::new(path);
    if let Some(parent) = path_ref
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("create artifact directory {}: {e}", parent.display()))?;
    }
    std::fs::write(path_ref, json).map_err(|e| {
        format!(
            "write self-play training artifact {}: {e}",
            path_ref.display()
        )
    })?;
    Ok(Some(path.to_string()))
}

fn handle_live_soccer_request(
    raw: &str,
    session: &Arc<Mutex<SoccerRealtimeSession>>,
    _input_queue: &SharedHumanInputs,
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
        ("GET", "/api/team-policy") | ("GET", "/api/policy") => {
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
            LiveHttpResponse::json(&guard.team_policy_artifact())
        }
        ("GET", "/api/tracking-dataset") | ("GET", "/api/tracking-export") => {
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
            LiveHttpResponse::json(&guard.tracking_dataset())
        }
        ("GET", "/api/tracking-template") | ("GET", "/api/tracking-schema") => {
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
            LiveHttpResponse::json(&soccer_tracking_template_dataset(&guard.match_ref().config))
        }
        ("POST", "/api/team-policy") | ("POST", "/api/policy") => {
            let artifact = match serde_json::from_str::<SoccerTeamPolicyArtifact>(req.body) {
                Ok(artifact) => artifact,
                Err(e) => {
                    return LiveHttpResponse::error(
                        400,
                        "Bad Request",
                        &format!("parse team policy artifact: {e}"),
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
            match guard.import_team_policy_artifact(artifact) {
                Ok(response) => LiveHttpResponse::json(&response),
                Err(e) => LiveHttpResponse::error(400, "Bad Request", &e),
            }
        }
        ("POST", "/api/tracking-policy") | ("POST", "/api/tracking") => {
            let tracking_req = match serde_json::from_str::<SoccerTrackingImportRequest>(req.body) {
                Ok(req) => req,
                Err(e) => {
                    return LiveHttpResponse::error(
                        400,
                        "Bad Request",
                        &format!("parse tracking import request: {e}"),
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
            match guard.import_tracking_for_team_policy(tracking_req) {
                Ok(response) => LiveHttpResponse::json(&response),
                Err(e) => LiveHttpResponse::error(400, "Bad Request", &e),
            }
        }
        ("POST", "/api/train-self-play") | ("POST", "/api/self-play-training") => {
            let training_req = match serde_json::from_str::<SoccerSelfPlayTrainingRequest>(req.body)
            {
                Ok(req) => req,
                Err(e) => {
                    return LiveHttpResponse::error(
                        400,
                        "Bad Request",
                        &format!("parse self-play training request: {e}"),
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
            match guard.train_self_play_team_policy(training_req) {
                Ok(response) => LiveHttpResponse::json(&response),
                Err(e) => LiveHttpResponse::error(400, "Bad Request", &e),
            }
        }
        ("POST", "/api/input") => match parse_human_input_payload(req.body) {
            Ok(inputs) => {
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
                let count = guard.push_human_inputs(inputs);
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
        ("POST", "/api/surface") => {
            let surface_req = match serde_json::from_str::<SoccerBallSurfaceRequest>(req.body) {
                Ok(req) => req,
                Err(e) => {
                    return LiveHttpResponse::error(
                        400,
                        "Bad Request",
                        &format!("parse ball surface request: {e}"),
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
            match guard.update_ball_surface(surface_req) {
                Ok(response) => LiveHttpResponse::json(&response),
                Err(e) => LiveHttpResponse::error(400, "Bad Request", &e),
            }
        }
        ("POST", "/api/learning") => {
            let learning_req = match serde_json::from_str::<SoccerLearningRuntimeRequest>(req.body)
            {
                Ok(req) => req,
                Err(e) => {
                    return LiveHttpResponse::error(
                        400,
                        "Bad Request",
                        &format!("parse learning runtime request: {e}"),
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
            LiveHttpResponse::json(&guard.update_learning_runtime(learning_req))
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

fn run_site_simulation() -> SimulationTrace {
    run_simulation(
        MatchConfig {
            duration_seconds: 60.0,
            learning_enabled: false,
            learning_logging_enabled: false,
            ..MatchConfig::default()
        },
        2,
    )
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

pub fn soccer_tracking_template_dataset(config: &MatchConfig) -> SoccerTrackingDataset {
    SoccerTrackingDataset {
        source: "tracking-template".to_string(),
        config: config.clone(),
        frames: vec![
            SoccerTrackingFrame {
                tick: 0,
                clock_seconds: 0.0,
                ball_position: Vec2::new(40.0, 70.0),
                ball_velocity: Some(Vec2::zero()),
                ball_altitude_yards: Some(0.0),
                pass_flight: None,
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
                        velocity: Some(Vec2::zero()),
                        home_position: Some(Vec2::new(40.0, 65.0)),
                        skills: None,
                    },
                    SoccerTrackingPlayerSample {
                        id: 1,
                        name: Some("Home runner".to_string()),
                        team: Team::Home,
                        role: PlayerRole::Forward,
                        shirt: Some(9),
                        position: Vec2::new(44.0, 82.0),
                        velocity: Some(Vec2::new(0.0, 1.8)),
                        home_position: Some(Vec2::new(44.0, 80.0)),
                        skills: None,
                    },
                    SoccerTrackingPlayerSample {
                        id: 2,
                        name: Some("Away defender".to_string()),
                        team: Team::Away,
                        role: PlayerRole::Defender,
                        shirt: Some(4),
                        position: Vec2::new(58.0, 78.0),
                        velocity: Some(Vec2::new(-0.8, 0.2)),
                        home_position: Some(Vec2::new(58.0, 78.0)),
                        skills: None,
                    },
                ],
            },
            SoccerTrackingFrame {
                tick: 1,
                clock_seconds: config.dt_seconds,
                ball_position: Vec2::new(44.0, 82.0),
                ball_velocity: Some(Vec2::new(8.0, 16.0)),
                ball_altitude_yards: Some(0.0),
                pass_flight: Some(PassFlight::Floor),
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
                        velocity: Some(Vec2::new(2.0, 4.0)),
                        home_position: Some(Vec2::new(40.0, 65.0)),
                        skills: None,
                    },
                    SoccerTrackingPlayerSample {
                        id: 1,
                        name: Some("Home runner".to_string()),
                        team: Team::Home,
                        role: PlayerRole::Forward,
                        shirt: Some(9),
                        position: Vec2::new(44.0, 82.0),
                        velocity: Some(Vec2::new(0.0, 2.2)),
                        home_position: Some(Vec2::new(44.0, 80.0)),
                        skills: None,
                    },
                    SoccerTrackingPlayerSample {
                        id: 2,
                        name: Some("Away defender".to_string()),
                        team: Team::Away,
                        role: PlayerRole::Defender,
                        shirt: Some(4),
                        position: Vec2::new(56.5, 78.5),
                        velocity: Some(Vec2::new(-1.5, 0.5)),
                        home_position: Some(Vec2::new(58.0, 78.0)),
                        skills: None,
                    },
                ],
            },
        ],
    }
}

fn tracking_import_format(request: &SoccerTrackingImportRequest) -> String {
    if let Some(format) = request.format.as_deref() {
        let normalized = format.trim().to_ascii_lowercase();
        if normalized == "json" || normalized == "csv" {
            return normalized;
        }
    }
    let source = request.source.trim().to_ascii_lowercase();
    if source.ends_with(".csv") || source.ends_with(".tsv") {
        "csv".to_string()
    } else if source.ends_with(".json") || request.content.trim_start().starts_with('{') {
        "json".to_string()
    } else {
        "csv".to_string()
    }
}

/// Parse player tracking rows into a [`SoccerTrackingDataset`].
///
/// Expected headers are one row per player sample with required `tick`,
/// `player_id`, `team`, `role`, and player position columns. Position can be
/// provided as pitch yards (`x`, `y`), normalized footage coordinates
/// (`x_norm`, `y_norm`), or pixels (`pixel_x`, `pixel_y`) with image dimensions.
/// Optional columns include `clock_seconds`, `name`, `shirt`, `vx`, `vy`,
/// `home_x`, `home_y`, `home_x_norm`, `home_y_norm`, skill columns such as
/// `top_speed`, `dribbling`, `passing_completion_rate`, `crossing_left`, and
/// `defending`, plus `ball_x`, `ball_y`, `ball_x_norm`, `ball_y_norm`,
/// `ball_pixel_x`, `ball_pixel_y`, `ball_vx`, `ball_vy`,
/// `ball_altitude_yards`, `pass_flight`, `ball_holder`, `last_touch_team`,
/// `score_home`, and `score_away`.
pub fn soccer_tracking_dataset_from_csv(
    raw: &str,
    config: MatchConfig,
    source: &str,
) -> Result<SoccerTrackingDataset, String> {
    let records = parse_csv_records(raw)?;
    let Some((header, rows)) = records.split_first() else {
        return Err("tracking csv is empty".to_string());
    };
    let header_map = csv_header_map(header);
    let mut builders: BTreeMap<u64, TrackingCsvFrameBuilder> = BTreeMap::new();

    for (row_idx, row) in rows.iter().enumerate() {
        let line_no = row_idx + 2;
        if row.iter().all(|field| field.trim().is_empty()) {
            continue;
        }
        let tick = csv_required_u64(row, &header_map, &["tick"], line_no)?;
        let builder = builders
            .entry(tick)
            .or_insert_with(|| TrackingCsvFrameBuilder::new(tick));
        if let Some(clock) = csv_optional_f64(
            row,
            &header_map,
            &["clock_seconds", "clock", "time"],
            line_no,
        )? {
            builder.clock_seconds = Some(clock);
        }
        if let Some(ball_position) =
            csv_optional_pitch_point(row, &header_map, BALL_PITCH_POINT_ALIASES, &config, line_no)?
        {
            builder.ball_position = Some(ball_position);
        }
        if let Some(ball_velocity) = csv_optional_vec2(
            row,
            &header_map,
            &["ball_vx", "ballvx"],
            &["ball_vy", "ballvy"],
            line_no,
        )? {
            builder.ball_velocity = Some(ball_velocity);
        }
        if let Some(altitude) = csv_optional_f64(
            row,
            &header_map,
            &[
                "ball_altitude_yards",
                "ball_altitude",
                "ball_z",
                "ballz",
                "altitude_yards",
                "altitude",
            ],
            line_no,
        )? {
            builder.ball_altitude_yards = Some(altitude.max(0.0));
        }
        if let Some(flight_raw) = csv_optional(
            row,
            &header_map,
            &[
                "pass_flight",
                "passflight",
                "ball_flight",
                "ballflight",
                "flight",
            ],
        ) {
            builder.pass_flight = Some(parse_tracking_pass_flight(flight_raw, line_no)?);
        }
        if let Some(holder) =
            csv_optional_usize(row, &header_map, &["ball_holder", "ballholder"], line_no)?
        {
            builder.ball_holder = Some(holder);
        }
        if let Some(team_raw) =
            csv_optional(row, &header_map, &["last_touch_team", "lasttouchteam"])
        {
            builder.last_touch_team = Some(parse_tracking_team(team_raw, line_no)?);
        }
        if let Some(score_home) =
            csv_optional_u32(row, &header_map, &["score_home", "scorehome"], line_no)?
        {
            builder.score_home = Some(score_home);
        }
        if let Some(score_away) =
            csv_optional_u32(row, &header_map, &["score_away", "scoreaway"], line_no)?
        {
            builder.score_away = Some(score_away);
        }

        let id = csv_required_usize(row, &header_map, &["player_id", "playerid", "id"], line_no)?;
        let team_raw = csv_required(row, &header_map, &["team"], line_no)?;
        let role_raw = csv_required(row, &header_map, &["role", "position_role"], line_no)?;
        let role = parse_tracking_role(role_raw, line_no)?;
        let skills = tracking_csv_skill_profile(row, &header_map, role, line_no)?;
        let position = csv_required_pitch_point(
            row,
            &header_map,
            PLAYER_PITCH_POINT_ALIASES,
            &config,
            line_no,
        );
        let position = position?;
        builder.players.push(SoccerTrackingPlayerSample {
            id,
            name: csv_optional(row, &header_map, &["name", "player_name", "playername"])
                .map(ToString::to_string),
            team: parse_tracking_team(team_raw, line_no)?,
            role,
            shirt: csv_optional_u8(
                row,
                &header_map,
                &["shirt", "shirt_number", "number"],
                line_no,
            )?,
            position,
            velocity: csv_optional_vec2(
                row,
                &header_map,
                &["vx", "player_vx", "playervx"],
                &["vy", "player_vy", "playervy"],
                line_no,
            )?,
            home_position: csv_optional_pitch_point(
                row,
                &header_map,
                HOME_PITCH_POINT_ALIASES,
                &config,
                line_no,
            )?,
            skills,
        });
    }

    if builders.is_empty() {
        return Err("tracking csv has no player rows".to_string());
    }

    let center = Vec2::new(
        config.field_width_yards * 0.5,
        config.field_length_yards * 0.5,
    );
    let frames = builders
        .into_values()
        .map(|builder| SoccerTrackingFrame {
            tick: builder.tick,
            clock_seconds: builder
                .clock_seconds
                .unwrap_or(builder.tick as f64 * config.dt_seconds),
            ball_position: builder.ball_position.unwrap_or(center),
            ball_velocity: builder.ball_velocity,
            ball_altitude_yards: builder.ball_altitude_yards,
            pass_flight: builder.pass_flight,
            ball_holder: builder.ball_holder,
            last_touch_team: builder.last_touch_team,
            score_home: builder.score_home,
            score_away: builder.score_away,
            players: builder.players,
        })
        .collect::<Vec<_>>();

    let dataset = SoccerTrackingDataset {
        source: source.to_string(),
        config,
        frames,
    };
    dataset.validate()?;
    Ok(dataset)
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
            let action = infer_tracking_action(player, &before, &after, &pair[0], &pair[1]);
            let observation = before.observation_for(player.id);
            let action_target = tracking_action_target_trace(player, &before, &after, &action);
            let decision = AgentDecisionTrace {
                mdp_state: before.mdp_state_for_player(player.id),
                observation: observation.clone(),
                belief: belief_from_observation(&observation),
                operation_order: vec!["tracking-imitation".to_string()],
                action_options: single_action_option(&action),
                action_target,
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
                true,
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
                action_target: decision.action_target,
                reward,
                next_state: after.mdp_state_for_player(player.id),
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

pub fn train_soccer_team_policies_from_self_play(
    config: MatchConfig,
    episodes: usize,
    options: SoccerQPolicyOptions,
) -> SoccerSelfPlayTrainingArtifact {
    let initial_policies = SoccerTeamQPolicies::new(options.clone());
    train_soccer_team_policies_from_self_play_with_initial_policies_and_progress(
        config,
        episodes,
        options,
        initial_policies,
        |_| {},
        |_, _, _, _| {},
    )
}

pub fn train_soccer_team_policies_from_self_play_with_progress<F, G>(
    config: MatchConfig,
    episodes: usize,
    options: SoccerQPolicyOptions,
    on_episode: F,
    on_progress: G,
) -> SoccerSelfPlayTrainingArtifact
where
    F: FnMut(&SoccerSelfPlayEpisodeSummary),
    G: FnMut(usize, u64, u64, u64),
{
    let initial_policies = SoccerTeamQPolicies::new(options.clone());
    train_soccer_team_policies_from_self_play_with_initial_policies_and_progress(
        config,
        episodes,
        options,
        initial_policies,
        on_episode,
        on_progress,
    )
}

pub fn train_soccer_team_policies_from_self_play_with_initial_policies_and_progress<F, G>(
    config: MatchConfig,
    episodes: usize,
    options: SoccerQPolicyOptions,
    policies: SoccerTeamQPolicies,
    on_episode: F,
    on_progress: G,
) -> SoccerSelfPlayTrainingArtifact
where
    F: FnMut(&SoccerSelfPlayEpisodeSummary),
    G: FnMut(usize, u64, u64, u64),
{
    train_soccer_team_policies_from_self_play_with_initial_policies_progress_and_checkpoints(
        config,
        episodes,
        options,
        policies,
        on_episode,
        on_progress,
        |_| {},
    )
}

pub fn train_soccer_team_policies_from_self_play_with_initial_policies_progress_and_checkpoints<
    F,
    G,
    H,
>(
    config: MatchConfig,
    episodes: usize,
    options: SoccerQPolicyOptions,
    mut policies: SoccerTeamQPolicies,
    mut on_episode: F,
    mut on_progress: G,
    mut on_checkpoint: H,
) -> SoccerSelfPlayTrainingArtifact
where
    F: FnMut(&SoccerSelfPlayEpisodeSummary),
    G: FnMut(usize, u64, u64, u64),
    H: FnMut(&SoccerSelfPlayTrainingArtifact),
{
    policies.home.options = options.clone();
    policies.away.options = options.clone();
    let mut episode_summaries = Vec::new();
    let base_seed = config.seed;

    for episode in 0..episodes {
        let episode_seed = base_seed.wrapping_add(episode as u32);
        let mut episode_config = config.clone();
        episode_config.seed = episode_seed;
        let total_ticks = episode_config.total_ticks();
        let mut sim = SoccerMatch::default_11v11(episode_config).with_team_policies(policies);
        let progress_interval = (total_ticks / 9).max(1);
        for tick_idx in 0..total_ticks {
            sim.run_time_step();
            let completed_ticks = tick_idx + 1;
            if completed_ticks == total_ticks || completed_ticks % progress_interval == 0 {
                on_progress(
                    episode,
                    episode_seed as u64,
                    completed_ticks as u64,
                    total_ticks as u64,
                );
            }
        }
        policies = sim
            .team_policies
            .take()
            .unwrap_or_else(|| SoccerTeamQPolicies::new(options.clone()));
        let summary = SoccerSelfPlayEpisodeSummary {
            episode,
            seed: episode_seed as u64,
            summary: sim.summary(),
            transitions: sim.learning_transitions.len(),
            home_policy_entries: policies.home.q_values.len(),
            home_policy_target_entries: policies.home.target_values.len(),
            away_policy_entries: policies.away.q_values.len(),
            away_policy_target_entries: policies.away.target_values.len(),
        };
        on_episode(&summary);
        episode_summaries.push(summary);
        let checkpoint_artifact =
            soccer_self_play_training_artifact(&config, &options, &episode_summaries, &policies);
        on_checkpoint(&checkpoint_artifact);
    }

    soccer_self_play_training_artifact(&config, &options, &episode_summaries, &policies)
}

fn soccer_self_play_training_artifact(
    config: &MatchConfig,
    options: &SoccerQPolicyOptions,
    episode_summaries: &[SoccerSelfPlayEpisodeSummary],
    policies: &SoccerTeamQPolicies,
) -> SoccerSelfPlayTrainingArtifact {
    SoccerSelfPlayTrainingArtifact {
        tactical_learning: config.tactical_learning.clone(),
        config: config.clone(),
        options: options.clone(),
        episodes: episode_summaries.to_vec(),
        home_entries: policies.home.entries(),
        home_target_entries: policies.home.target_entries(),
        away_entries: policies.away.entries(),
        away_target_entries: policies.away.target_entries(),
    }
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
        target_entries: policy.target_entries(),
        events: dataset.events.clone(),
    }
}

pub fn soccer_simulation_page_html(trace: &SimulationTrace) -> String {
    let json = serde_json::to_string(trace)
        .unwrap_or_else(|_| "{}".to_string())
        .replace("</script", "<\\/script");
    include_str!("soccer_ui.html").replace("__SOCCER_TRACE__", &json)
}

fn match_frames_jsonl(frames: &[MatchFrame]) -> Result<String, serde_json::Error> {
    let mut jsonl = String::new();
    for frame in frames {
        jsonl.push_str(&serde_json::to_string(frame)?);
        jsonl.push('\n');
    }
    Ok(jsonl)
}

pub fn soccer_live_page_html() -> String {
    include_str!("soccer_live_ui.html").to_string()
}

pub fn write_soccer_artifacts() {
    let trace = run_site_simulation();
    let ui_path = std::path::Path::new("out/soccer-sim.html");
    let frames_path = std::path::Path::new("out/soccer-sim.frames.jsonl");
    let _ = std::fs::create_dir_all("out");
    let _ = std::fs::write(ui_path, soccer_simulation_page_html(&trace));
    if let Ok(jsonl) = match_frames_jsonl(&trace.frames) {
        let _ = std::fs::write(frames_path, jsonl);
    }
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
    let self_play_config = MatchConfig {
        duration_seconds: 4.0,
        seed: MatchConfig::default().seed + 20,
        ..MatchConfig::default()
    };
    let self_play_artifact = train_soccer_team_policies_from_self_play(
        self_play_config,
        3,
        SoccerQPolicyOptions::default(),
    );
    let self_play_path = std::path::Path::new("out/soccer-self-play-team-policies.json");
    if let Ok(json) = serde_json::to_string_pretty(&self_play_artifact) {
        let _ = std::fs::write(self_play_path, json);
    }
    println!("# Soccer simulation UI: {}", ui_path.display());
    println!("# Soccer simulation frames: {}", frames_path.display());
    println!("# Soccer Q-policy artifact: {}", policy_path.display());
    println!(
        "# Soccer self-play team-policy artifact: {}",
        self_play_path.display()
    );
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

#[derive(Clone, Debug)]
struct TrackingCsvFrameBuilder {
    tick: u64,
    clock_seconds: Option<f64>,
    ball_position: Option<Vec2>,
    ball_velocity: Option<Vec2>,
    ball_altitude_yards: Option<f64>,
    pass_flight: Option<PassFlight>,
    ball_holder: Option<usize>,
    last_touch_team: Option<Team>,
    score_home: Option<u32>,
    score_away: Option<u32>,
    players: Vec<SoccerTrackingPlayerSample>,
}

impl TrackingCsvFrameBuilder {
    fn new(tick: u64) -> Self {
        TrackingCsvFrameBuilder {
            tick,
            clock_seconds: None,
            ball_position: None,
            ball_velocity: None,
            ball_altitude_yards: None,
            pass_flight: None,
            ball_holder: None,
            last_touch_team: None,
            score_home: None,
            score_away: None,
            players: Vec::new(),
        }
    }
}

fn parse_csv_records(raw: &str) -> Result<Vec<Vec<String>>, String> {
    let mut records = Vec::new();
    for (idx, line) in raw.lines().enumerate() {
        let line = line.trim_end_matches('\r');
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        records.push(parse_csv_record(line).map_err(|e| format!("csv line {}: {e}", idx + 1))?);
    }
    Ok(records)
}

fn parse_csv_record(line: &str) -> Result<Vec<String>, String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(ch) = chars.next() {
        match ch {
            '"' if quoted && chars.peek() == Some(&'"') => {
                field.push('"');
                chars.next();
            }
            '"' => quoted = !quoted,
            ',' if !quoted => {
                fields.push(field.trim().to_string());
                field.clear();
            }
            _ => field.push(ch),
        }
    }
    if quoted {
        return Err("unterminated quoted field".to_string());
    }
    fields.push(field.trim().to_string());
    Ok(fields)
}

fn normalize_csv_header(value: &str) -> String {
    value
        .chars()
        .filter(|ch| *ch != '_' && *ch != '-' && !ch.is_whitespace())
        .flat_map(|ch| ch.to_lowercase())
        .collect()
}

fn csv_header_map(header: &[String]) -> HashMap<String, usize> {
    header
        .iter()
        .enumerate()
        .map(|(idx, name)| (normalize_csv_header(name), idx))
        .collect()
}

fn csv_optional<'a>(
    row: &'a [String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
) -> Option<&'a str> {
    aliases.iter().find_map(|alias| {
        let idx = header.get(&normalize_csv_header(alias))?;
        let value = row.get(*idx)?.trim();
        if value.is_empty() {
            None
        } else {
            Some(value)
        }
    })
}

fn csv_required<'a>(
    row: &'a [String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<&'a str, String> {
    csv_optional(row, header, aliases)
        .ok_or_else(|| format!("csv line {line_no} missing required column {}", aliases[0]))
}

fn csv_required_f64(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<f64, String> {
    let raw = csv_required(row, header, aliases, line_no)?;
    raw.parse::<f64>()
        .map_err(|e| format!("csv line {line_no} parse {}={raw}: {e}", aliases[0]))
}

fn csv_required_u64(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<u64, String> {
    let raw = csv_required(row, header, aliases, line_no)?;
    raw.parse::<u64>()
        .map_err(|e| format!("csv line {line_no} parse {}={raw}: {e}", aliases[0]))
}

fn csv_required_usize(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<usize, String> {
    let raw = csv_required(row, header, aliases, line_no)?;
    raw.parse::<usize>()
        .map_err(|e| format!("csv line {line_no} parse {}={raw}: {e}", aliases[0]))
}

fn csv_optional_f64(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<Option<f64>, String> {
    let Some(raw) = csv_optional(row, header, aliases) else {
        return Ok(None);
    };
    raw.parse::<f64>()
        .map(Some)
        .map_err(|e| format!("csv line {line_no} parse {}={raw}: {e}", aliases[0]))
}

fn csv_optional_usize(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<Option<usize>, String> {
    let Some(raw) = csv_optional(row, header, aliases) else {
        return Ok(None);
    };
    raw.parse::<usize>()
        .map(Some)
        .map_err(|e| format!("csv line {line_no} parse {}={raw}: {e}", aliases[0]))
}

fn csv_optional_u32(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<Option<u32>, String> {
    let Some(raw) = csv_optional(row, header, aliases) else {
        return Ok(None);
    };
    raw.parse::<u32>()
        .map(Some)
        .map_err(|e| format!("csv line {line_no} parse {}={raw}: {e}", aliases[0]))
}

fn csv_optional_u8(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<Option<u8>, String> {
    let Some(raw) = csv_optional(row, header, aliases) else {
        return Ok(None);
    };
    raw.parse::<u8>()
        .map(Some)
        .map_err(|e| format!("csv line {line_no} parse {}={raw}: {e}", aliases[0]))
}

fn csv_optional_vec2(
    row: &[String],
    header: &HashMap<String, usize>,
    x_aliases: &[&str],
    y_aliases: &[&str],
    line_no: usize,
) -> Result<Option<Vec2>, String> {
    let x = csv_optional_f64(row, header, x_aliases, line_no)?;
    let y = csv_optional_f64(row, header, y_aliases, line_no)?;
    match (x, y) {
        (Some(x), Some(y)) => Ok(Some(Vec2::new(x, y))),
        (None, None) => Ok(None),
        _ => Err(format!(
            "csv line {line_no} must provide both {} and {}",
            x_aliases[0], y_aliases[0]
        )),
    }
}

#[derive(Clone, Copy)]
struct TrackingCsvPitchPointAliases {
    direct_x: &'static [&'static str],
    direct_y: &'static [&'static str],
    normalized_x: &'static [&'static str],
    normalized_y: &'static [&'static str],
    pixel_x: &'static [&'static str],
    pixel_y: &'static [&'static str],
}

const PLAYER_PITCH_POINT_ALIASES: TrackingCsvPitchPointAliases = TrackingCsvPitchPointAliases {
    direct_x: &[
        "x",
        "player_x",
        "playerx",
        "position_x",
        "positionx",
        "pitch_x",
        "pitchx",
        "yard_x",
        "yardx",
        "yards_x",
        "yardsx",
    ],
    direct_y: &[
        "y",
        "player_y",
        "playery",
        "position_y",
        "positiony",
        "pitch_y",
        "pitchy",
        "yard_y",
        "yardy",
        "yards_y",
        "yardsy",
    ],
    normalized_x: &[
        "x_norm",
        "xnorm",
        "norm_x",
        "normx",
        "normalized_x",
        "normalizedx",
        "x_normalized",
        "xnormalized",
        "player_x_norm",
        "playerxnorm",
        "player_norm_x",
        "playernormx",
        "player_x_normalized",
        "playerxnormalized",
    ],
    normalized_y: &[
        "y_norm",
        "ynorm",
        "norm_y",
        "normy",
        "normalized_y",
        "normalizedy",
        "y_normalized",
        "ynormalized",
        "player_y_norm",
        "playerynorm",
        "player_norm_y",
        "playernormy",
        "player_y_normalized",
        "playerynormalized",
    ],
    pixel_x: &[
        "pixel_x",
        "pixelx",
        "px",
        "x_px",
        "xpx",
        "player_pixel_x",
        "playerpixelx",
        "player_px",
        "playerpx",
        "player_x_px",
        "playerxpx",
        "bbox_center_x",
        "bboxcenterx",
        "center_x",
        "centerx",
    ],
    pixel_y: &[
        "pixel_y",
        "pixely",
        "py",
        "y_px",
        "ypx",
        "player_pixel_y",
        "playerpixely",
        "player_py",
        "playerpy",
        "player_y_px",
        "playerypx",
        "bbox_center_y",
        "bboxcentery",
        "center_y",
        "centery",
    ],
};

const HOME_PITCH_POINT_ALIASES: TrackingCsvPitchPointAliases = TrackingCsvPitchPointAliases {
    direct_x: &["home_x", "homex", "home_position_x", "homepositionx"],
    direct_y: &["home_y", "homey", "home_position_y", "homepositiony"],
    normalized_x: &[
        "home_x_norm",
        "homexnorm",
        "home_norm_x",
        "homenormx",
        "home_x_normalized",
        "homexnormalized",
    ],
    normalized_y: &[
        "home_y_norm",
        "homeynorm",
        "home_norm_y",
        "homenormy",
        "home_y_normalized",
        "homeynormalized",
    ],
    pixel_x: &[
        "home_pixel_x",
        "homepixelx",
        "home_px",
        "homepx",
        "home_x_px",
        "homexpx",
    ],
    pixel_y: &[
        "home_pixel_y",
        "homepixely",
        "home_py",
        "homepy",
        "home_y_px",
        "homeypx",
    ],
};

const BALL_PITCH_POINT_ALIASES: TrackingCsvPitchPointAliases = TrackingCsvPitchPointAliases {
    direct_x: &["ball_x", "ballx", "ball_position_x", "ballpositionx"],
    direct_y: &["ball_y", "bally", "ball_position_y", "ballpositiony"],
    normalized_x: &[
        "ball_x_norm",
        "ballxnorm",
        "ball_norm_x",
        "ballnormx",
        "ball_normalized_x",
        "ballnormalizedx",
        "ball_x_normalized",
        "ballxnormalized",
    ],
    normalized_y: &[
        "ball_y_norm",
        "ballynorm",
        "ball_norm_y",
        "ballnormy",
        "ball_normalized_y",
        "ballnormalizedy",
        "ball_y_normalized",
        "ballynormalized",
    ],
    pixel_x: &[
        "ball_pixel_x",
        "ballpixelx",
        "ball_px",
        "ballpx",
        "ball_x_px",
        "ballxpx",
        "ball_center_x",
        "ballcenterx",
    ],
    pixel_y: &[
        "ball_pixel_y",
        "ballpixely",
        "ball_py",
        "ballpy",
        "ball_y_px",
        "ballypx",
        "ball_center_y",
        "ballcentery",
    ],
};

fn csv_required_pitch_point(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: TrackingCsvPitchPointAliases,
    config: &MatchConfig,
    line_no: usize,
) -> Result<Vec2, String> {
    csv_optional_pitch_point(row, header, aliases, config, line_no)?.ok_or_else(|| {
        format!(
            "csv line {line_no} missing required position columns {}, {}, or {}",
            aliases.direct_x[0], aliases.normalized_x[0], aliases.pixel_x[0]
        )
    })
}

fn csv_optional_pitch_point(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: TrackingCsvPitchPointAliases,
    config: &MatchConfig,
    line_no: usize,
) -> Result<Option<Vec2>, String> {
    if let Some(point) =
        csv_optional_vec2(row, header, aliases.direct_x, aliases.direct_y, line_no)?
    {
        return Ok(Some(point));
    }
    if let Some(point) = csv_optional_vec2(
        row,
        header,
        aliases.normalized_x,
        aliases.normalized_y,
        line_no,
    )? {
        return Ok(Some(
            Vec2::new(
                point.x * config.field_width_yards,
                point.y * config.field_length_yards,
            )
            .clamp_to_pitch(config.field_width_yards, config.field_length_yards),
        ));
    }

    let pixel = csv_optional_vec2(row, header, aliases.pixel_x, aliases.pixel_y, line_no)?;
    let Some(pixel) = pixel else {
        return Ok(None);
    };
    let image_dimensions = csv_required_image_dimensions(row, header, line_no)?;
    Ok(Some(
        Vec2::new(
            pixel.x / image_dimensions.x * config.field_width_yards,
            pixel.y / image_dimensions.y * config.field_length_yards,
        )
        .clamp_to_pitch(config.field_width_yards, config.field_length_yards),
    ))
}

fn csv_required_image_dimensions(
    row: &[String],
    header: &HashMap<String, usize>,
    line_no: usize,
) -> Result<Vec2, String> {
    let dimensions = csv_optional_vec2(
        row,
        header,
        &[
            "image_width",
            "imagewidth",
            "frame_width",
            "framewidth",
            "video_width",
            "videowidth",
            "width",
        ],
        &[
            "image_height",
            "imageheight",
            "frame_height",
            "frameheight",
            "video_height",
            "videoheight",
            "height",
        ],
        line_no,
    )?;
    let Some(dimensions) = dimensions else {
        return Err(format!(
            "csv line {line_no} pixel coordinates require image_width and image_height"
        ));
    };
    if dimensions.x <= 0.0 || dimensions.y <= 0.0 {
        return Err(format!(
            "csv line {line_no} image dimensions must be positive"
        ));
    }
    Ok(dimensions)
}

fn csv_optional_skill_score(
    row: &[String],
    header: &HashMap<String, usize>,
    aliases: &[&str],
    line_no: usize,
) -> Result<Option<f64>, String> {
    csv_optional_f64(row, header, aliases, line_no).map(|value| value.map(ability_score))
}

fn tracking_csv_skill_profile(
    row: &[String],
    header: &HashMap<String, usize>,
    role: PlayerRole,
    line_no: usize,
) -> Result<Option<SkillProfile>, String> {
    let mut skills = neutral_tracking_skill_profile(role);
    let mut found = false;

    macro_rules! apply_skill {
        ($field:ident, [$($alias:expr),+ $(,)?]) => {
            if let Some(score) = csv_optional_skill_score(row, header, &[$($alias),+], line_no)? {
                skills.$field = score;
                found = true;
            }
        };
    }

    apply_skill!(
        top_speed,
        [
            "top_speed",
            "topSpeed",
            "top_speed_yps",
            "topSpeedYps",
            "skill_top_speed",
            "speed"
        ]
    );
    apply_skill!(
        acceleration,
        [
            "acceleration",
            "acceleration_yps2",
            "accelerationYps2",
            "skill_acceleration"
        ]
    );
    apply_skill!(strength, ["strength", "skill_strength"]);
    apply_skill!(height, ["height", "skill_height"]);
    apply_skill!(shooting, ["shooting", "shooting_ability", "skill_shooting"]);
    apply_skill!(
        right_foot_shot_power,
        [
            "right_foot_shot_power",
            "rightFootShotPower",
            "right_shot",
            "skill_right_foot_shot_power"
        ]
    );
    apply_skill!(
        left_foot_shot_power,
        [
            "left_foot_shot_power",
            "leftFootShotPower",
            "left_shot",
            "skill_left_foot_shot_power"
        ]
    );
    apply_skill!(passing, ["passing", "passing_ability", "skill_passing"]);
    apply_skill!(
        passing_completion_rate,
        [
            "passing_completion_rate",
            "passingCompletionRate",
            "pass_completion",
            "pass_completion_rate",
            "skill_passing_completion_rate"
        ]
    );
    apply_skill!(
        flair_passing,
        [
            "flair_passing",
            "flairPassing",
            "flair",
            "skill_flair_passing"
        ]
    );
    apply_skill!(
        crossing_left,
        [
            "crossing_left",
            "crossingLeft",
            "left_crossing",
            "leftFootCrossingAbility",
            "crossingAbilityWithLeftRoot",
            "skill_crossing_left"
        ]
    );
    apply_skill!(
        crossing_right,
        [
            "crossing_right",
            "crossingRight",
            "right_crossing",
            "rightFootCrossingAbility",
            "skill_crossing_right"
        ]
    );
    apply_skill!(
        dribbling,
        ["dribbling", "dribbling_ability", "skill_dribbling"]
    );
    apply_skill!(
        first_touch,
        [
            "first_touch",
            "firstTouch",
            "control_touch",
            "skill_first_touch"
        ]
    );
    apply_skill!(
        defending,
        [
            "defending",
            "defensive_ability",
            "defensiveAbility",
            "skill_defending"
        ]
    );
    apply_skill!(
        goalkeeping,
        [
            "goalkeeping",
            "gk_strength",
            "ability_in_goal",
            "abilityInGoal",
            "skill_goalkeeping"
        ]
    );
    apply_skill!(
        defensive_tracking,
        [
            "defensive_tracking",
            "defensiveTracking",
            "tracking_back",
            "trackingBack",
            "skill_defensive_tracking"
        ]
    );
    apply_skill!(stamina, ["stamina", "skill_stamina"]);
    apply_skill!(vision, ["vision", "skill_vision"]);
    apply_skill!(aggression, ["aggression", "skill_aggression"]);

    if let Some(noise) = csv_optional_f64(
        row,
        header,
        &["decision_noise", "decisionNoise", "skill_decision_noise"],
        line_no,
    )? {
        skills.decision_noise = if noise.is_finite() {
            noise.clamp(0.0, 1.0)
        } else {
            skills.decision_noise
        };
        found = true;
    }

    Ok(found.then_some(skills))
}

fn parse_tracking_team(raw: &str, line_no: usize) -> Result<Team, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "home" | "h" | "1" => Ok(Team::Home),
        "away" | "a" | "2" => Ok(Team::Away),
        _ => Err(format!("csv line {line_no} unknown team {raw}")),
    }
}

fn parse_tracking_role(raw: &str, line_no: usize) -> Result<PlayerRole, String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "goalkeeper" | "keeper" | "gk" | "g" => Ok(PlayerRole::Goalkeeper),
        "defender" | "defence" | "defense" | "def" | "d" => Ok(PlayerRole::Defender),
        "midfielder" | "midfield" | "mid" | "m" => Ok(PlayerRole::Midfielder),
        "forward" | "striker" | "winger" | "fw" | "st" | "f" => Ok(PlayerRole::Forward),
        _ => Err(format!("csv line {line_no} unknown role {raw}")),
    }
}

fn parse_tracking_pass_flight(raw: &str, line_no: usize) -> Result<PassFlight, String> {
    match normalize_csv_header(raw).as_str() {
        "floor" | "ground" | "groundpass" | "grounded" | "low" | "rolling" | "roll" | "0" => {
            Ok(PassFlight::Floor)
        }
        "aerial" | "air" | "airborne" | "aerialpass" | "aerialcross" | "chipped" | "chip"
        | "high" | "lob" | "lofted" | "longball" | "1" => Ok(PassFlight::Aerial),
        _ => Err(format!("csv line {line_no} unknown pass flight {raw}")),
    }
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
        .map(|p| {
            let skills = p
                .skills
                .clone()
                .unwrap_or_else(|| neutral_tracking_skill_profile(p.role));
            PlayerSnapshot {
                id: p.id,
                name: p
                    .name
                    .clone()
                    .unwrap_or_else(|| format!("Tracking {}", p.id)),
                team: p.team,
                role: p.role,
                shirt: p.shirt.unwrap_or((p.id % 100).max(1) as u8),
                position: p.position,
                position_history: vec![p.position],
                velocity: p.velocity.unwrap_or_default(),
                movement_gait: MovementGait::Stand,
                receive_facing: FacingBucket::Unknown,
                action_facing: {
                    let facing = facing_bucket_from_vector(p.velocity.unwrap_or_default());
                    if facing == FacingBucket::Unknown {
                        default_team_facing(p.team)
                    } else {
                        facing
                    }
                },
                incoming_ball: None,
                vision_range_yards: vision_range_yards(skills.vision),
                field_of_view_degrees: field_of_view_degrees(skills.vision),
                skills,
                fatigue: 0.0,
                home_position: home_positions
                    .get(&p.id)
                    .copied()
                    .unwrap_or(p.home_position.unwrap_or(p.position)),
                controller_slot: None,
                acceleration: Vec2::zero(),
                jerk: Vec2::zero(),
                last_decision: None,
            }
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
            curl_acceleration: Vec2::zero(),
            altitude_yards: frame.ball_altitude_yards.unwrap_or(0.0).max(0.0),
            holder: frame.ball_holder,
            last_touch_team,
        },
        ball_history: vec![BallPositionSample {
            tick: frame.tick,
            clock_seconds: frame.clock_seconds,
            position: frame.ball_position,
            velocity: frame.ball_velocity.unwrap_or_default(),
            acceleration: Vec2::zero(),
            curl_acceleration: Vec2::zero(),
            altitude_yards: frame.ball_altitude_yards.unwrap_or(0.0).max(0.0),
            holder: frame.ball_holder,
            last_touch_team,
        }],
        pending_pass: None,
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
            DefensiveCoverProfile::default(),
        ),
        away_directive: tactical_directive_for_team(
            Team::Away,
            phase,
            last_touch_team,
            frame.ball_position,
            -score_diff_home,
            config.field_width_yards,
            config.field_length_yards,
            DefensiveCoverProfile::default(),
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
    before_frame: &SoccerTrackingFrame,
    after_frame: &SoccerTrackingFrame,
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
                return tracking_pass_label(before_frame, after_frame, before, after).to_string();
            }
        }
        if tracking_ball_near_teammate(after, player.id, player.team)
            && after.ball.holder != Some(player.id)
        {
            return tracking_pass_label(before_frame, after_frame, before, after).to_string();
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

fn tracking_pass_label(
    before_frame: &SoccerTrackingFrame,
    after_frame: &SoccerTrackingFrame,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
) -> &'static str {
    if tracking_pass_flight_between(before_frame, after_frame, before, after).is_aerial() {
        "aerial-pass"
    } else {
        "pass"
    }
}

fn tracking_pass_flight_between(
    before_frame: &SoccerTrackingFrame,
    after_frame: &SoccerTrackingFrame,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
) -> PassFlight {
    after_frame
        .pass_flight
        .or(before_frame.pass_flight)
        .unwrap_or_else(|| {
            let before_altitude = before_frame
                .ball_altitude_yards
                .unwrap_or(before.ball.altitude_yards);
            let after_altitude = after_frame
                .ball_altitude_yards
                .unwrap_or(after.ball.altitude_yards);
            if before_altitude.max(after_altitude) > 0.35 {
                PassFlight::Aerial
            } else {
                PassFlight::Floor
            }
        })
}

fn tracking_action_target_trace(
    player: &PlayerSnapshot,
    before: &WorldSnapshot,
    after: &WorldSnapshot,
    action: &str,
) -> Option<AgentActionTargetTrace> {
    let next_player = after.players.iter().find(|p| p.id == player.id);
    let goal = Vec2::new(
        before.field_width * 0.5,
        player.team.goal_y(before.field_length),
    );
    let (point, target_player) = match normalize_soccer_action_label(action) {
        "pass" | "aerial-pass" => {
            let holder_teammate = after.ball.holder.and_then(|holder| {
                after
                    .players
                    .iter()
                    .find(|p| p.id == holder && p.team == player.team && p.id != player.id)
                    .map(|p| (p.position, Some(p.id)))
            });
            holder_teammate.unwrap_or((after.ball.position, after.ball.holder))
        }
        "shoot" => (goal, None),
        "dribble" | "space" | "defend" => (
            next_player.map(|p| p.position).unwrap_or(player.position),
            None,
        ),
        "tackle" => {
            let target_player = before.ball.holder.or(after.ball.holder);
            let point = target_player
                .and_then(|id| {
                    before
                        .player_position(id)
                        .or_else(|| after.player_position(id))
                })
                .unwrap_or(before.ball.position);
            (point, target_player)
        }
        "hold" => (player.home_position, None),
        _ => return None,
    };
    let point = point.clamp_to_pitch(before.field_width, before.field_length);
    Some(AgentActionTargetTrace {
        point: Some(point),
        player_id: target_player,
        grid: Some(pitch_grid_address(
            point,
            before.field_width,
            before.field_length,
        )),
        facing: facing_bucket_from_vector(point - player.position),
    })
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
    let mut position_history = player
        .position_history
        .iter()
        .copied()
        .collect::<VecDeque<_>>();
    if position_history.is_empty() {
        position_history.push_back(player.position);
    }
    while position_history.len() > PLAYER_POSITION_HISTORY_LIMIT {
        position_history.pop_front();
    }
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
        jerk: player.jerk,
        movement_gait: player.movement_gait,
        position_history,
        receive_facing: player.receive_facing,
        action_facing: player.action_facing,
        incoming_ball: player.incoming_ball.clone(),
        skills: player.skills.clone(),
        fatigue: player.fatigue.clamp(0.0, 1.0),
        controller_slot: None,
        preferences: AgentPreferences::default(),
        last_decision: None,
    }
}

fn neutral_tracking_skill_profile(role: PlayerRole) -> SkillProfile {
    let shooting = match role {
        PlayerRole::Forward => 7.8,
        PlayerRole::Midfielder => 6.6,
        PlayerRole::Defender => 5.0,
        PlayerRole::Goalkeeper => 4.0,
    };
    SkillProfile {
        top_speed: 7.4,
        acceleration: 7.1,
        strength: match role {
            PlayerRole::Goalkeeper => 7.8,
            PlayerRole::Defender => 8.0,
            PlayerRole::Midfielder => 6.6,
            PlayerRole::Forward => 7.4,
        },
        height: match role {
            PlayerRole::Goalkeeper => 8.6,
            PlayerRole::Defender => 7.6,
            PlayerRole::Midfielder => 6.0,
            PlayerRole::Forward => 7.0,
        },
        shooting,
        right_foot_shot_power: (shooting + 0.6).clamp(1.0, 10.0),
        left_foot_shot_power: (shooting - 0.8).clamp(1.0, 10.0),
        passing: 7.2,
        passing_completion_rate: 7.2,
        flair_passing: 4.2,
        crossing_left: 6.6,
        crossing_right: 7.0,
        dribbling: 6.8,
        first_touch: 7.0,
        defending: 6.8,
        goalkeeping: if role == PlayerRole::Goalkeeper {
            8.8
        } else {
            1.6
        },
        defensive_tracking: 6.8,
        stamina: 8.2,
        vision: DEFAULT_PLAYER_VISION_SKILL,
        decision_noise: 0.05,
        aggression: 5.8,
    }
}

fn tackle_success_probability(defender: &SkillProfile, attacker: &SkillProfile) -> f64 {
    let ball_control =
        ability01(attacker.dribbling) * 0.70 + ability01(attacker.first_touch) * 0.30;
    let defensive_pressure =
        ability01(defender.defending) * 0.82 + ability01(defender.aggression) * 0.18;
    (defensive_pressure / (defensive_pressure + ball_control)).clamp(0.18, 0.82)
}

fn tackle_foul_probability(
    defender: &SkillProfile,
    attacker: &SkillProfile,
    contact_distance: f64,
    contact_speed: f64,
) -> f64 {
    let timing_risk = (1.0 - ability01(defender.defending)) * 0.34;
    let aggression_risk = ability01(defender.aggression) * 0.24;
    let control_risk =
        (ability01(attacker.dribbling) * 0.55 + ability01(attacker.first_touch) * 0.45) * 0.12;
    let speed_risk = (contact_speed / 10.0).clamp(0.0, 1.0) * 0.18;
    let reach_risk = (contact_distance / 2.2).clamp(0.0, 1.0) * 0.10;
    (0.02 + timing_risk + aggression_risk + control_risk + speed_risk + reach_risk)
        .clamp(0.03, 0.78)
}

fn carried_ball_lead(player: &PlayerAgent) -> Vec2 {
    let facing = if player.velocity.len() > 0.45 {
        player.velocity.normalized()
    } else {
        Vec2::new(0.0, player.team.attack_dir())
    };
    facing * DRIBBLE_TOUCH_LEAD_YARDS
}

fn dribble_heavy_touch_probability(player: &PlayerAgent, pressure: f64) -> f64 {
    let control = ability01(player.skills.dribbling) * 0.58
        + ability01(player.skills.first_touch) * 0.30
        + ability01(player.skills.stamina) * 0.12;
    let fatigue_risk = player.fatigue.clamp(0.0, 1.0) * 0.05;
    let pressure = pressure.clamp(0.0, 1.0);
    (0.004 + pressure * (1.0 - control).clamp(0.0, 1.0) * 0.42 + fatigue_risk).clamp(0.004, 0.36)
}

fn pressure_from_observation(observation: &SoccerPomdpObservation) -> f64 {
    observation.perceived_pressure.clamp(0.0, 1.0)
}

fn pressure_from_nearest_distance(nearest_opponent_distance: f64) -> f64 {
    (1.0 - nearest_opponent_distance / 18.0).clamp(0.0, 1.0)
}

fn perceived_pressure_for_player(
    player: &PlayerSnapshot,
    real_pressure: f64,
    visible_opponents: usize,
) -> f64 {
    let vision_relief = (ability01(player.skills.vision) - 0.55).clamp(-0.25, 0.35) * 0.22;
    let crowding = (visible_opponents as f64 / 8.0).clamp(0.0, 1.0) * 0.12;
    (real_pressure + crowding - vision_relief).clamp(0.0, 1.0)
}

fn time_on_ball_seconds(pressure: f64) -> f64 {
    (2.8 - pressure.clamp(0.0, 1.0) * 2.35).clamp(0.25, 3.0)
}

fn immediate_dispossession_risk_for_player(
    player: &PlayerSnapshot,
    nearest_opponent_distance: f64,
    perceived_pressure: f64,
    perceived_time_on_ball_seconds: f64,
) -> f64 {
    let close_risk =
        (1.0 - (nearest_opponent_distance - PLAYER_CONTROL_RADIUS_YARDS) / 4.4).clamp(0.0, 1.0);
    let control_skill = ability01(player.skills.dribbling) * 0.52
        + ability01(player.skills.first_touch) * 0.30
        + ability01(player.skills.strength) * 0.18;
    let skill_risk = (1.0 - control_skill).clamp(0.0, 1.0);
    let clock_risk = (1.0 - perceived_time_on_ball_seconds / 2.8).clamp(0.0, 1.0);
    (perceived_pressure.clamp(0.0, 1.0) * 0.42
        + close_risk * 0.32
        + clock_risk * 0.16
        + skill_risk * 0.16
        + player.fatigue.clamp(0.0, 1.0) * 0.10)
        .clamp(0.0, 1.0)
}

fn shot_decision_is_qualified(observation: &SoccerPomdpObservation) -> bool {
    let block_risk = observation.shot_block_probability.clamp(0.0, 1.0);
    let quality_shot = block_risk <= SHOT_BLOCK_DECISION_MAX_PROBABILITY
        && observation.shot_on_frame_probability >= SHOT_ON_FRAME_MIN_PROBABILITY
        && observation.shot_beat_goalkeeper_probability >= SHOT_KEEPER_BEAT_MIN_PROBABILITY;
    let pressure_bailout = block_risk <= SHOT_BLOCK_BAILOUT_MAX_PROBABILITY
        && observation.yards_to_goal <= SHOT_BAILOUT_NEAR_GOAL_YARDS
        && observation.immediate_dispossession_risk >= SHOT_BAILOUT_DISPOSSESSION_RISK
        && observation.shot_on_frame_probability >= SHOT_BAILOUT_ON_FRAME_PROBABILITY;
    quality_shot || pressure_bailout
}

fn first_time_shot_decision_is_qualified(observation: &SoccerPomdpObservation) -> bool {
    observation.first_time_shot_score > 0.0 && shot_decision_is_qualified(observation)
}

fn aerial_duel_skill_from_snapshot(player: &PlayerSnapshot) -> f64 {
    let height = ability01(player.skills.height);
    (height * 0.36
        + ability01(player.skills.strength) * 0.26
        + ability01(player.skills.aggression) * 0.18
        + ability01(player.skills.first_touch) * 0.20)
        .clamp(0.0, 1.0)
}

fn aerial_duel_skill_from_agent(player: &PlayerAgent) -> f64 {
    let height = ability01(player.skills.height);
    (height * 0.36
        + ability01(player.skills.strength) * 0.26
        + ability01(player.skills.aggression) * 0.18
        + ability01(player.skills.first_touch) * 0.20)
        .clamp(0.0, 1.0)
}

fn first_time_shot_score_for_player(
    player: &PlayerSnapshot,
    incoming_kind: IncomingBallKind,
    shot_block_probability: f64,
    yards_to_goal: f64,
    goal_angle_degrees: f64,
    pressure: f64,
) -> f64 {
    if shot_block_probability > SHOT_BLOCK_BAILOUT_MAX_PROBABILITY {
        return 0.0;
    }
    let foot_power = (ability01(
        player
            .skills
            .right_foot_shot_power
            .max(player.skills.left_foot_shot_power),
    ) * 0.70
        + ability01(player.skills.shooting) * 0.30)
        .clamp(0.0, 1.0);
    let aerial_finish = aerial_duel_skill_from_snapshot(player);
    let strike_skill = match incoming_kind {
        IncomingBallKind::AerialCross | IncomingBallKind::AerialPass => {
            foot_power * 0.45 + aerial_finish * 0.55
        }
        IncomingBallKind::Cross => foot_power * 0.78 + ability01(player.skills.first_touch) * 0.22,
        _ => foot_power * 0.70 + ability01(player.skills.first_touch) * 0.30,
    };
    let geometry = ((1.0 - yards_to_goal / 34.0).clamp(0.0, 1.0) * 0.58
        + (goal_angle_degrees / 42.0).clamp(0.0, 1.0) * 0.42)
        .clamp(0.0, 1.0);
    let quick_release_block_discount =
        (1.0 - shot_block_probability.clamp(0.0, 1.0) * 0.46).clamp(0.46, 1.0);
    ((strike_skill * 0.62 + geometry * 0.38 + pressure.clamp(0.0, 1.0) * 0.14)
        * quick_release_block_discount)
        .clamp(0.0, 1.0)
}

fn first_time_pass_score_for_player(
    player: &PlayerSnapshot,
    incoming_kind: IncomingBallKind,
    pressure: f64,
) -> f64 {
    let flair_bonus = match incoming_kind {
        IncomingBallKind::AerialCross | IncomingBallKind::AerialPass => 0.08,
        IncomingBallKind::Cross => 0.05,
        _ => 0.0,
    };
    (ability01(player.skills.passing_completion_rate) * 0.50
        + ability01(player.skills.first_touch) * 0.24
        + ability01(player.skills.flair_passing) * 0.18
        + pressure.clamp(0.0, 1.0) * 0.12
        + flair_bonus)
        .clamp(0.0, 1.0)
}

fn control_touch_score_for_player(
    player: &PlayerSnapshot,
    incoming_kind: IncomingBallKind,
    incoming_speed_yps: f64,
    pressure: f64,
) -> f64 {
    let aerial_control = if matches!(
        incoming_kind,
        IncomingBallKind::AerialCross | IncomingBallKind::AerialPass
    ) {
        aerial_duel_skill_from_snapshot(player) * 0.26
    } else {
        0.0
    };
    let speed_penalty = (incoming_speed_yps / 36.0).clamp(0.0, 1.0) * 0.20;
    (ability01(player.skills.first_touch) * 0.42
        + ability01(player.skills.strength) * 0.22
        + ability01(player.skills.dribbling) * 0.16
        + aerial_control
        - pressure.clamp(0.0, 1.0) * 0.18
        - speed_penalty)
        .clamp(0.0, 1.0)
}

fn angle_between_vectors_degrees(a: Vec2, b: Vec2) -> f64 {
    let al = a.len();
    let bl = b.len();
    if al <= 1e-9 || bl <= 1e-9 {
        return 0.0;
    }
    let cos = (a.dot(b) / (al * bl)).clamp(-1.0, 1.0);
    cos.acos().to_degrees()
}

fn vision_range_yards(vision: f64) -> f64 {
    PLAYER_BASE_VISION_RANGE_YARDS + ability01(vision) * PLAYER_VISION_RANGE_BONUS_YARDS
}

fn field_of_view_degrees(vision: f64) -> f64 {
    PLAYER_BASE_FIELD_OF_VIEW_DEGREES + ability01(vision) * PLAYER_FIELD_OF_VIEW_BONUS_DEGREES
}

fn player_vision_range(player: &PlayerSnapshot) -> f64 {
    if player.vision_range_yards.is_finite() && player.vision_range_yards > 0.0 {
        player.vision_range_yards
    } else {
        vision_range_yards(DEFAULT_PLAYER_VISION_SKILL)
    }
}

fn player_field_of_view(player: &PlayerSnapshot) -> f64 {
    if player.field_of_view_degrees.is_finite() && player.field_of_view_degrees > 0.0 {
        player.field_of_view_degrees
    } else {
        field_of_view_degrees(DEFAULT_PLAYER_VISION_SKILL)
    }
}

fn average_player_position_confidence<I>(
    snapshot: &WorldSnapshot,
    observer_id: usize,
    player_ids: I,
) -> f64
where
    I: IntoIterator<Item = usize>,
{
    let mut total = 0.0;
    let mut count = 0usize;
    for player_id in player_ids {
        if let Some(position) = snapshot.player_position(player_id) {
            if let Some(confidence) =
                snapshot.player_position_confidence_for_point(observer_id, position)
            {
                total += confidence;
                count += 1;
            }
        }
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn position_confidence_for_observer(
    observer: &PlayerSnapshot,
    observer_position: Vec2,
    point: Vec2,
    facing: Option<Vec2>,
) -> f64 {
    let to_point = point - observer_position;
    let distance = to_point.len();
    if distance <= PLAYER_CONTROL_RADIUS_YARDS * 3.0 {
        return 1.0;
    }

    let range = player_vision_range(observer).max(1.0);
    let distance_confidence = (1.0 - (distance / range).clamp(0.0, 1.8) * 0.52).clamp(0.06, 1.0);
    let facing_confidence = if let Some(facing) = facing {
        let facing = facing.normalized();
        if facing.len() <= 1e-9 || distance <= 1e-9 {
            0.72
        } else {
            let dot = facing.dot(to_point.normalized());
            let half_fov_cos = (player_field_of_view(observer).to_radians() * 0.5).cos();
            if dot >= half_fov_cos {
                1.0
            } else if dot >= 0.0 {
                0.74
            } else {
                0.50
            }
        }
    } else {
        0.72
    };

    (distance_confidence * facing_confidence).clamp(0.0, 1.0)
}

fn pass_would_be_cross(
    from: Vec2,
    target: Vec2,
    team: Team,
    field_width: f64,
    field_length: f64,
) -> bool {
    let attacking_third = match team {
        Team::Home => from.y > field_length * 0.58 && target.y > field_length * 0.70,
        Team::Away => from.y < field_length * 0.42 && target.y < field_length * 0.30,
    };
    let wide_channel = from.x < field_width * 0.28 || from.x > field_width * 0.72;
    let target_central = target.x > field_width * 0.28 && target.x < field_width * 0.72;
    attacking_third && wide_channel && target_central
}

fn floor_pass_lane_score_for_snapshot(
    snapshot: &WorldSnapshot,
    player: &PlayerSnapshot,
    player_position: Vec2,
    targets: &[usize],
) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for target_id in targets {
        let Some(target) = snapshot.players.iter().find(|p| p.id == *target_id) else {
            continue;
        };
        let target_position = snapshot
            .player_position(target.id)
            .unwrap_or(target.position);
        let distance = player_position.distance(target_position);
        let confidence = snapshot
            .player_position_confidence_for_point(player.id, target_position)
            .unwrap_or(0.0);
        let forward = ((target_position.y - player_position.y) * player.team.attack_dir() / 28.0)
            .clamp(-0.25, 1.0);
        let distance_fit = (1.0 - (distance - 18.0).abs() / 32.0).clamp(0.0, 1.0);
        total += (0.30 + confidence * 0.34 + distance_fit * 0.22 + forward.max(0.0) * 0.14)
            .clamp(0.0, 1.0);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (total / count as f64).clamp(0.0, 1.0)
    }
}

fn aerial_pass_bypass_score_for_snapshot(
    snapshot: &WorldSnapshot,
    player: &PlayerSnapshot,
    player_position: Vec2,
    targets: &[usize],
) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for target_id in targets {
        let Some(target) = snapshot.players.iter().find(|p| p.id == *target_id) else {
            continue;
        };
        let target_position = snapshot
            .player_position(target.id)
            .unwrap_or(target.position);
        let floor_blocked =
            !snapshot.clear_line(player_position, target_position, player.team.other(), 2.5);
        let forward = ((target_position.y - player_position.y) * player.team.attack_dir() / 30.0)
            .clamp(-0.20, 1.0);
        let cross_bonus = if pass_would_be_cross(
            player_position,
            target_position,
            player.team,
            snapshot.field_width,
            snapshot.field_length,
        ) {
            0.18
        } else {
            0.0
        };
        let receiver_air = aerial_duel_skill_from_snapshot(target) * 0.30
            + ability01(target.skills.first_touch) * 0.12;
        let blocked_bonus = if floor_blocked { 0.44 } else { 0.08 };
        total +=
            (blocked_bonus + forward.max(0.0) * 0.18 + receiver_air + cross_bonus).clamp(0.0, 1.0);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (total / count as f64).clamp(0.0, 1.0)
    }
}

fn aerial_pass_interception_risk_for_snapshot(
    snapshot: &WorldSnapshot,
    player: &PlayerSnapshot,
    player_position: Vec2,
    targets: &[usize],
) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }
    let mut total = 0.0;
    let mut count = 0usize;
    for target_id in targets {
        let Some(target) = snapshot.players.iter().find(|p| p.id == *target_id) else {
            continue;
        };
        let target_position = snapshot
            .player_position(target.id)
            .unwrap_or(target.position);
        let distance = player_position.distance(target_position);
        let nearest = snapshot
            .players
            .iter()
            .filter(|p| p.team != player.team)
            .map(|p| {
                snapshot
                    .player_position(p.id)
                    .unwrap_or(p.position)
                    .distance(target_position)
            })
            .fold(f64::INFINITY, f64::min);
        let nearby = snapshot
            .players
            .iter()
            .filter(|p| p.team != player.team)
            .filter(|p| {
                snapshot
                    .player_position(p.id)
                    .unwrap_or(p.position)
                    .distance(target_position)
                    <= 8.5
            })
            .count() as f64;
        let floor_like_risk =
            (0.05 + (1.0 - nearest / 16.0).clamp(0.0, 1.0) * 0.22 + nearby.min(4.0) * 0.045)
                .clamp(0.0, 0.45);
        let air_multiplier = if nearest <= 8.5 { 2.6 } else { 2.0 };
        let receiver_air = aerial_duel_skill_from_snapshot(target) * 0.62
            + ability01(target.skills.first_touch) * 0.20;
        let distance_risk = (distance / 48.0).clamp(0.0, 1.0) * 0.12;
        total += (floor_like_risk * air_multiplier + (1.0 - receiver_air) * 0.16 + distance_risk)
            .clamp(0.0, 1.0);
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        (total / count as f64).clamp(0.0, 1.0)
    }
}

fn pass_execution_skill(skills: &SkillProfile, flight: PassFlight, is_cross: bool) -> f64 {
    let base = if is_cross {
        ability01(skills.crossing_left.max(skills.crossing_right)) * 0.74
            + ability01(skills.passing_completion_rate) * 0.18
            + ability01(skills.flair_passing) * 0.08
    } else {
        ability01(skills.passing_completion_rate) * 0.72
            + ability01(skills.passing) * 0.20
            + ability01(skills.flair_passing) * 0.08
    };
    if flight.is_aerial() {
        (base * 0.86 + ability01(skills.strength) * 0.08 + ability01(skills.flair_passing) * 0.06)
            .clamp(0.05, 0.99)
    } else {
        base.clamp(0.05, 0.99)
    }
}

fn incoming_context_from_pass(
    pass: &PendingPass,
    receiver: usize,
    speed_yps: f64,
    received_tick: u64,
) -> IncomingBallContext {
    IncomingBallContext {
        from_player: Some(pass.from),
        target_player: Some(receiver),
        team: Some(pass.team),
        kind: match (pass.flight, pass.is_cross) {
            (PassFlight::Floor, false) => IncomingBallKind::GroundPass,
            (PassFlight::Aerial, false) => IncomingBallKind::AerialPass,
            (PassFlight::Floor, true) => IncomingBallKind::Cross,
            (PassFlight::Aerial, true) => IncomingBallKind::AerialCross,
        },
        origin: Some(pass.origin),
        intended_target: Some(pass.intended_target),
        speed_yps,
        distance_yards: pass.distance_yards,
        received_tick,
        is_cross: pass.is_cross,
        is_aerial: pass.flight.is_aerial(),
    }
}

fn aerial_interception_multiplier(pass: &PendingPass, ball_position: Vec2) -> f64 {
    if !pass.flight.is_aerial() {
        return 1.0;
    }
    let landing_distance = ball_position.distance(pass.intended_target);
    if landing_distance <= 8.5 {
        2.6
    } else {
        2.0
    }
}

fn pass_ball_altitude_yards(pass: &PendingPass, ball_position: Vec2) -> f64 {
    if !pass.flight.is_aerial() {
        return 0.0;
    }
    let path = pass.intended_target - pass.origin;
    let denom = path.x * path.x + path.y * path.y;
    if denom <= 1e-9 {
        return 0.0;
    }
    let progress = dot(ball_position - pass.origin, path) / denom;
    let progress = progress.clamp(0.0, 1.0);
    let apex = (3.2 + pass.distance_yards.max(0.0) * 0.055).clamp(4.0, 11.5);
    (std::f64::consts::PI * progress).sin().max(0.0) * apex
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

fn mph_to_yps(mph: f64) -> f64 {
    mph.max(0.0) * 1760.0 / 3600.0
}

fn shot_speed_yps_from_power(power: f64, skills: &SkillProfile) -> f64 {
    let shooting = ability01(skills.shooting);
    let foot_power = ability01(
        skills
            .right_foot_shot_power
            .max(skills.left_foot_shot_power),
    );
    let strength = ability01(skills.strength);
    let technique_power = (shooting * 0.42 + foot_power * 0.40 + strength * 0.18).clamp(0.0, 1.0);
    let mph = 30.0 + power.clamp(0.0, 1.0) * (18.0 + technique_power * 12.0);
    mph_to_yps(mph.clamp(18.0, 60.0))
}

fn pass_speed_yps_from_power(
    power: f64,
    flight: PassFlight,
    is_cross: bool,
    skills: &SkillProfile,
) -> f64 {
    let passing = ability01(skills.passing_completion_rate);
    let crossing = ability01(skills.crossing_left.max(skills.crossing_right));
    let strength = ability01(skills.strength);
    let skill_power = if is_cross {
        crossing * 0.58 + passing * 0.28 + strength * 0.14
    } else {
        passing * 0.68 + ability01(skills.passing) * 0.20 + strength * 0.12
    }
    .clamp(0.0, 1.0);
    let floor = if flight.is_aerial() || is_cross {
        8.0
    } else {
        3.0
    };
    let mph = floor + power.clamp(0.0, 1.0) * (26.0 - floor) + skill_power * 4.0;
    mph_to_yps(mph.clamp(3.0, 30.0))
}

fn shot_curl_probability_for_player(
    skills: &SkillProfile,
    pressure: f64,
    yards_to_goal: f64,
    goal_angle_degrees: f64,
) -> f64 {
    let shooting = ability01(skills.shooting);
    let foot_power = ability01(
        skills
            .right_foot_shot_power
            .max(skills.left_foot_shot_power),
    );
    let flair = ability01(skills.flair_passing);
    let technique = (shooting * 0.52 + foot_power * 0.30 + flair * 0.18).clamp(0.0, 1.0);
    let range_fit = (1.0 - (yards_to_goal - 20.0).abs() / 34.0).clamp(0.10, 1.0);
    let narrow_angle_need = (1.0 - goal_angle_degrees / 42.0).clamp(0.0, 1.0);
    (0.03 + technique * 0.50 + range_fit * 0.10 + narrow_angle_need * 0.08
        - pressure.clamp(0.0, 1.0) * 0.20)
        .clamp(0.02, 0.74)
}

fn pass_curl_probability_for_player(
    skills: &SkillProfile,
    flight: PassFlight,
    is_cross: bool,
    distance: f64,
    pressure: f64,
) -> f64 {
    let passing = ability01(skills.passing_completion_rate);
    let crossing = ability01(skills.crossing_left.max(skills.crossing_right));
    let flair = ability01(skills.flair_passing);
    let technique = if is_cross {
        crossing * 0.52 + passing * 0.26 + flair * 0.22
    } else {
        passing * 0.54 + ability01(skills.passing) * 0.22 + flair * 0.24
    }
    .clamp(0.0, 1.0);
    let distance_fit = (distance / 34.0).clamp(0.12, 1.0);
    let aerial_bonus = if flight.is_aerial() { 0.08 } else { 0.0 };
    (0.025 + technique * 0.46 + distance_fit * 0.10 + aerial_bonus
        - pressure.clamp(0.0, 1.0) * 0.18)
        .clamp(0.01, 0.72)
}

fn pass_curl_probability_for_snapshot(
    snapshot: &WorldSnapshot,
    player: &PlayerSnapshot,
    player_position: Vec2,
    floor_targets: &[usize],
    aerial_targets: &[usize],
    pressure: f64,
) -> f64 {
    let mut best: f64 = 0.0;
    for (targets, flight) in [
        (floor_targets, PassFlight::Floor),
        (aerial_targets, PassFlight::Aerial),
    ] {
        for target_id in targets.iter().take(3) {
            let Some(target) = snapshot.players.iter().find(|p| p.id == *target_id) else {
                continue;
            };
            let target_position = snapshot
                .player_position(target.id)
                .unwrap_or(target.position);
            let is_cross = pass_would_be_cross(
                player_position,
                target_position,
                player.team,
                snapshot.field_width,
                snapshot.field_length,
            );
            best = best.max(pass_curl_probability_for_player(
                &player.skills,
                flight,
                is_cross,
                player_position.distance(target_position),
                pressure,
            ));
        }
    }
    best.clamp(0.0, 1.0)
}

fn led_pass_target_for_receiver(
    from: Vec2,
    target_position: Vec2,
    target_velocity: Vec2,
    speed_yps: f64,
    skill: f64,
) -> Vec2 {
    let distance = from.distance(target_position);
    let travel_time = distance / speed_yps.max(1.0);
    let lead = (travel_time * (0.20 + skill.clamp(0.0, 1.0) * 0.55)).clamp(0.0, 1.35);
    target_position + target_velocity * lead
}

fn curl_acceleration_for_path(
    from: Vec2,
    initial_target: Vec2,
    final_target: Vec2,
    speed_yps: f64,
    bend_yards: f64,
) -> Vec2 {
    let path = initial_target - from;
    let distance = path.len();
    if distance <= 1e-6 || speed_yps <= 1e-6 || bend_yards.abs() <= 1e-6 {
        return Vec2::zero();
    }
    let dir = path.normalized();
    let lateral = Vec2::new(-dir.y, dir.x);
    let side = dot(final_target - initial_target, lateral).signum();
    if side == 0.0 {
        return Vec2::zero();
    }
    let travel_time = (distance / speed_yps).clamp(0.12, 3.5);
    let magnitude =
        (2.0 * bend_yards.abs() / (travel_time * travel_time)).clamp(0.0, MAX_BALL_CURL_YPS2);
    lateral * side * magnitude
}

fn apply_ball_curl(velocity: Vec2, curl_acceleration: Vec2, dt_seconds: f64) -> Vec2 {
    let speed = velocity.len();
    if speed <= 1e-6 || curl_acceleration.len() <= 1e-6 || dt_seconds <= 0.0 {
        return velocity;
    }
    let dir = velocity.normalized();
    let lateral_accel = curl_acceleration - dir * dot(curl_acceleration, dir);
    let curved = velocity + lateral_accel * dt_seconds;
    if curved.len() <= 1e-6 {
        velocity
    } else {
        curved.normalized() * speed
    }
}

fn decayed_ball_curl(curl_acceleration: Vec2, dt_seconds: f64) -> Vec2 {
    if curl_acceleration.len() <= 1e-6 || dt_seconds <= 0.0 {
        return curl_acceleration;
    }
    let factor = (-BALL_CURL_DECAY_PER_SECOND * dt_seconds).exp();
    let decayed = curl_acceleration * factor;
    if decayed.len() < 0.025 {
        Vec2::zero()
    } else {
        decayed
    }
}

fn player_reaction_time_seconds_from_traits(
    skills: &SkillProfile,
    fatigue: f64,
    role: PlayerRole,
) -> f64 {
    let quickness = ability01(skills.acceleration) * 0.30
        + ability01(skills.defensive_tracking) * 0.22
        + ability01(skills.defending) * 0.18
        + ability01(skills.aggression) * 0.10
        + ability01(skills.vision) * 0.20;
    let role_floor = if role == PlayerRole::Goalkeeper {
        0.18
    } else {
        0.21
    };
    (0.44 - quickness * 0.18 + fatigue.clamp(0.0, 1.0) * 0.13).clamp(role_floor, 0.62)
}

fn goalkeeper_reaction_time_seconds(skills: &SkillProfile, fatigue: f64) -> f64 {
    let keeper_quickness = ability01(skills.goalkeeping) * 0.42
        + ability01(skills.acceleration) * 0.22
        + ability01(skills.defensive_tracking) * 0.16
        + ability01(skills.vision) * 0.20;
    (0.36 - keeper_quickness * 0.16 + fatigue.clamp(0.0, 1.0) * 0.11).clamp(0.16, 0.52)
}

fn shot_block_probability_for_candidate(
    skills: &SkillProfile,
    role: PlayerRole,
    fatigue: f64,
    from: Vec2,
    to: Vec2,
    position: Vec2,
    shot_speed_yps: f64,
    quick_release: bool,
) -> Option<(f64, Vec2, f64, f64, f64)> {
    if role == PlayerRole::Goalkeeper {
        return None;
    }
    let path = to - from;
    let distance = path.len();
    if distance <= 0.35 {
        return None;
    }
    let t = segment_projection_factor(from, to, position);
    if !(0.01..=0.99).contains(&t) {
        return None;
    }
    let block_position = from + path * t;
    let lateral_distance = position.distance(block_position);
    if lateral_distance > SHOT_BLOCK_LANE_RADIUS_YARDS {
        return None;
    }
    let line_score = if lateral_distance <= PLAYER_BODY_RADIUS_YARDS {
        1.0
    } else {
        (1.0 - (lateral_distance - PLAYER_BODY_RADIUS_YARDS)
            / (SHOT_BLOCK_LANE_RADIUS_YARDS - PLAYER_BODY_RADIUS_YARDS))
            .clamp(0.0, 1.0)
    };
    let distance_to_ball = from.distance(position);
    let screen_score = if (SHOT_SCREEN_IDEAL_MIN_YARDS..=SHOT_SCREEN_IDEAL_MAX_YARDS)
        .contains(&distance_to_ball)
    {
        1.0
    } else if distance_to_ball < SHOT_SCREEN_IDEAL_MIN_YARDS {
        (0.62 + distance_to_ball / SHOT_SCREEN_IDEAL_MIN_YARDS * 0.38).clamp(0.62, 1.0)
    } else if distance_to_ball <= 18.0 {
        (1.0 - (distance_to_ball - SHOT_SCREEN_IDEAL_MAX_YARDS) / 15.0 * 0.38).clamp(0.62, 1.0)
    } else {
        (0.62 - (distance_to_ball - 18.0) / 46.0 * 0.30).clamp(0.26, 0.62)
    };
    let arrival_time = distance_to_ball / shot_speed_yps.max(1.0);
    let reaction_time = player_reaction_time_seconds_from_traits(skills, fatigue, role);
    let reaction_factor = if arrival_time >= reaction_time {
        1.0
    } else {
        (0.62 + arrival_time / reaction_time.max(1e-6) * 0.38).clamp(0.62, 1.0)
    };
    let readiness = (0.88
        + ability01(skills.defending) * 0.14
        + ability01(skills.defensive_tracking) * 0.09
        + ability01(skills.aggression) * 0.05
        + ability01(skills.strength) * 0.04
        - fatigue.clamp(0.0, 1.0) * 0.14)
        .clamp(0.68, 1.16);
    let quick_release_factor = if quick_release {
        SHOT_BLOCK_QUICK_RELEASE_MULTIPLIER
    } else {
        1.0
    };
    let probability = (SHOT_BLOCK_DIRECT_PROBABILITY
        * line_score
        * (0.72 + screen_score * 0.28)
        * reaction_factor
        * readiness
        * quick_release_factor)
        .clamp(0.0, 0.92);
    (probability >= 0.025).then_some((
        probability,
        block_position,
        distance_to_ball,
        lateral_distance,
        screen_score,
    ))
}

fn combine_shot_block_assessments(
    mut assessments: Vec<ShotBlockAssessment>,
) -> Option<ShotBlockAssessment> {
    if assessments.is_empty() {
        return None;
    }
    assessments.sort_by(|a, b| {
        b.probability
            .partial_cmp(&a.probability)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let combined = (1.0
        - assessments
            .iter()
            .map(|assessment| 1.0 - assessment.probability.clamp(0.0, 1.0))
            .product::<f64>())
    .clamp(0.0, 0.96);
    let mut best = assessments.remove(0);
    best.probability = combined;
    Some(best)
}

fn shot_block_assessment_for_agents(
    players: &[PlayerAgent],
    from: Vec2,
    to: Vec2,
    attacking_team: Team,
    shot_speed_yps: f64,
    quick_release: bool,
) -> Option<ShotBlockAssessment> {
    let assessments = players
        .iter()
        .filter(|player| player.team == attacking_team.other())
        .filter_map(|player| {
            let (probability, block_position, distance_to_ball, lateral_distance, screen_score) =
                shot_block_probability_for_candidate(
                    &player.skills,
                    player.role,
                    player.fatigue,
                    from,
                    to,
                    player.position,
                    shot_speed_yps,
                    quick_release,
                )?;
            Some(ShotBlockAssessment {
                blocker_id: player.id,
                defending_team: player.team,
                block_position,
                probability,
                distance_to_ball,
                lateral_distance,
                screen_score,
            })
        })
        .collect::<Vec<_>>();
    combine_shot_block_assessments(assessments)
}

fn shot_block_assessment_for_snapshot(
    snapshot: &WorldSnapshot,
    from: Vec2,
    attacking_team: Team,
    shot_speed_yps: f64,
    quick_release: bool,
) -> Option<ShotBlockAssessment> {
    let goal = Vec2::new(
        snapshot.field_width * 0.5,
        attacking_team.goal_y(snapshot.field_length),
    );
    let assessments = snapshot
        .players
        .iter()
        .filter(|player| player.team == attacking_team.other())
        .filter_map(|player| {
            let position = snapshot
                .player_position(player.id)
                .unwrap_or(player.position);
            let (probability, block_position, distance_to_ball, lateral_distance, screen_score) =
                shot_block_probability_for_candidate(
                    &player.skills,
                    player.role,
                    player.fatigue,
                    from,
                    goal,
                    position,
                    shot_speed_yps,
                    quick_release,
                )?;
            Some(ShotBlockAssessment {
                blocker_id: player.id,
                defending_team: player.team,
                block_position,
                probability,
                distance_to_ball,
                lateral_distance,
                screen_score,
            })
        })
        .collect::<Vec<_>>();
    combine_shot_block_assessments(assessments)
}

fn shot_miss_window_yards(
    goal_width: f64,
    shooting_skill: f64,
    pressure: f64,
    yards_to_goal: f64,
) -> f64 {
    let skill_error = 1.0 - shooting_skill.clamp(0.05, 0.99);
    let pressure = pressure.clamp(0.0, 1.0);
    goal_width * (0.28 + skill_error * 0.80 + pressure * 0.48)
        + yards_to_goal.max(0.0) * 0.018 * (0.45 + skill_error)
}

fn triangular_abs_within_probability(bound: f64, window: f64) -> f64 {
    if bound <= 0.0 || window <= 0.0 {
        return 0.0;
    }
    if bound >= window {
        return 1.0;
    }
    let ratio = (bound / window).clamp(0.0, 1.0);
    (2.0 * ratio - ratio * ratio).clamp(0.0, 1.0)
}

fn shot_on_frame_probability(
    goal_width: f64,
    shooting_skill: f64,
    pressure: f64,
    yards_to_goal: f64,
    goal_angle_degrees: f64,
    shot_block_probability: f64,
) -> f64 {
    let miss_window = shot_miss_window_yards(goal_width, shooting_skill, pressure, yards_to_goal);
    let geometry_probability = triangular_abs_within_probability(goal_width * 0.5, miss_window);
    let angle_factor = (0.45 + (goal_angle_degrees / 36.0).clamp(0.0, 1.0) * 0.55).clamp(0.45, 1.0);
    let distance_factor = (1.08 - (yards_to_goal.max(0.0) / 52.0).powi(2) * 0.25).clamp(0.70, 1.0);
    let block_factor = (1.0 - shot_block_probability.clamp(0.0, 1.0) * 0.74).clamp(0.20, 1.0);
    (geometry_probability * angle_factor * distance_factor * block_factor).clamp(0.0, 1.0)
}

fn noisy_shot_target_x(
    goal_center_x: f64,
    goal_width: f64,
    shooting_skill: f64,
    pressure: f64,
    yards_to_goal: f64,
    rng: &mut SeededRandom,
) -> f64 {
    let miss_window = shot_miss_window_yards(goal_width, shooting_skill, pressure, yards_to_goal);
    goal_center_x + triangular_sample(rng) * miss_window
}

fn goalkeeper_save_probability_from_traits(
    skills: &SkillProfile,
    keeper_position: Vec2,
    shot_crossing: Vec2,
    shot_speed: f64,
    goal_width: f64,
) -> f64 {
    let reaction = ability01(skills.goalkeeping) * 0.48
        + ability01(skills.defending) * 0.18
        + ability01(skills.first_touch) * 0.20
        + ability01(skills.acceleration) * 0.14;
    let distance_to_shot = keeper_position.distance(shot_crossing);
    let reach_penalty = (distance_to_shot / (goal_width * 0.72)).clamp(0.0, 1.5);
    let speed_penalty = (shot_speed / 48.0).clamp(0.0, 1.0) * 0.12;
    (0.38 + reaction * 0.85 - reach_penalty * 0.22 - speed_penalty).clamp(0.20, 0.94)
}

fn goalkeeper_save_probability(
    keeper: &PlayerAgent,
    shot_crossing: Vec2,
    shot_speed: f64,
    goal_width: f64,
) -> f64 {
    goalkeeper_save_probability_from_traits(
        &keeper.skills,
        keeper.position,
        shot_crossing,
        shot_speed,
        goal_width,
    )
}

fn shot_beat_goalkeeper_probability_for_snapshot(
    snapshot: &WorldSnapshot,
    player: &PlayerSnapshot,
    player_position: Vec2,
) -> f64 {
    let Some(keeper_id) = snapshot.goalkeeper_for(player.team.other()) else {
        return 0.35;
    };
    let Some(keeper) = snapshot.players.iter().find(|p| p.id == keeper_id) else {
        return 0.35;
    };
    let keeper_position = snapshot
        .player_position(keeper_id)
        .unwrap_or(keeper.position);
    let goal_y = player.team.goal_y(snapshot.field_length);
    let goal_center_x = snapshot.field_width * 0.5;
    let half_width = snapshot.goal_width * 0.5;
    let keeper_opposite_x = if keeper_position.x <= goal_center_x {
        goal_center_x + half_width * 0.72
    } else {
        goal_center_x - half_width * 0.72
    };
    let shooter_angle_x = if player_position.x <= goal_center_x {
        goal_center_x + half_width * 0.38
    } else {
        goal_center_x - half_width * 0.38
    };
    let crossing_x = (keeper_opposite_x * 0.72 + shooter_angle_x * 0.28)
        .clamp(goal_center_x - half_width, goal_center_x + half_width);
    let shot_crossing = Vec2::new(crossing_x, goal_y);
    let shooting_skill = ability01(player.skills.shooting);
    let foot_power = ability01(
        player
            .skills
            .right_foot_shot_power
            .max(player.skills.left_foot_shot_power),
    );
    let shot_power = 0.72 + 0.28 * (shooting_skill * 0.72 + foot_power * 0.28);
    let shot_speed = 28.0 + 18.0 * shot_power.clamp(0.0, 1.0);
    let save_probability = goalkeeper_save_probability_from_traits(
        &keeper.skills,
        keeper_position,
        shot_crossing,
        shot_speed,
        snapshot.goal_width,
    );
    (1.0 - save_probability).clamp(0.0, 1.0)
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
    pending_pass: Option<&PendingPass>,
    rng: &mut SeededRandom,
) -> Option<(usize, Team)> {
    let ball_speed = ball_velocity.len();
    let mut candidates = Vec::new();
    for p in players {
        let mut control_radius = PLAYER_CONTROL_RADIUS_YARDS
            + ability01(p.skills.first_touch) * 0.48
            + (1.0 - (ball_speed / 18.0).clamp(0.0, 1.0)) * 0.24;
        let mut aerial_score_bonus = 0.0;
        if let Some(pass) = pending_pass {
            if pass.flight.is_aerial() {
                let is_target = pass.target == Some(p.id);
                let landing_distance = ball_pos.distance(pass.intended_target);
                let interception_multiplier = aerial_interception_multiplier(pass, ball_pos);
                if !is_target && landing_distance > 7.5 {
                    control_radius *= if p.team == pass.team { 0.70 } else { 0.55 };
                }
                let aerial_duel = aerial_duel_skill_from_agent(p);
                control_radius += aerial_duel * 0.45;
                aerial_score_bonus += aerial_duel * 0.72;
                if p.team != pass.team {
                    control_radius +=
                        PLAYER_CONTROL_RADIUS_YARDS * (interception_multiplier - 1.0) * 0.26;
                    aerial_score_bonus += (interception_multiplier - 1.0) * 0.34;
                    aerial_score_bonus += if landing_distance <= 8.5 { 0.62 } else { 0.28 };
                }
            }
        }
        let dist = p.position.distance(ball_pos);
        if dist > control_radius {
            continue;
        }
        let to_ball = (ball_pos - p.position).normalized();
        let closing_speed = dot(p.velocity - ball_velocity, to_ball).clamp(-8.0, 8.0);
        let score = -dist * 1.45
            + ability01(p.skills.first_touch) * 0.72
            + ability01(p.skills.aggression) * 0.18
            + closing_speed * 0.055
            + aerial_score_bonus;
        candidates.push((p.id, p.team, score));
    }
    sample_control_candidate(&candidates, rng)
}

fn triangular_sample(rng: &mut SeededRandom) -> f64 {
    rng.next_float() + rng.next_float() - 1.0
}

fn boundary_crossing_fraction(start: f64, end: f64, lower: f64, upper: f64) -> Option<f64> {
    let boundary = if end < lower {
        lower
    } else if end > upper {
        upper
    } else {
        return None;
    };
    let delta = end - start;
    if delta.abs() <= 1e-9 {
        return Some(1.0);
    }
    Some(((boundary - start) / delta).clamp(0.0, 1.0))
}

pub fn segment_distance_to_point(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    let t = segment_projection_factor(a, b, p);
    let projection = a + (b - a) * t;
    p.distance(projection)
}

fn segment_projection_factor(a: Vec2, b: Vec2, p: Vec2) -> f64 {
    let ab = b - a;
    let denom = ab.x * ab.x + ab.y * ab.y;
    if denom <= 1e-12 {
        return 0.0;
    }
    let ap = p - a;
    ((ap.x * ab.x + ap.y * ab.y) / denom).clamp(0.0, 1.0)
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
        "shoot" => observation.has_ball && shot_decision_is_qualified(&observation),
        "pass" => observation.has_ball && snapshot.best_visible_pass_target(player_id).is_some(),
        "aerial-pass" => {
            observation.has_ball && snapshot.best_aerial_pass_target(player_id).is_some()
        }
        "first-time-shot" | "first-time-header" => {
            observation.has_ball
                && observation.first_touch_available
                && first_time_shot_decision_is_qualified(&observation)
        }
        "first-time-pass" => {
            observation.has_ball
                && observation.first_touch_available
                && snapshot.best_visible_pass_target(player_id).is_some()
        }
        "control-touch" => observation.has_ball && observation.first_touch_available,
        "dribble" => observation.has_ball,
        "defend" => snapshot.controlled_possession_team() == Some(player.team.other()),
        "recover" => snapshot.controlled_possession_team().is_none(),
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
        "hold" => false,
        "human-move" => false,
        _ => false,
    }
}

pub fn shot_lane_is_clear(snapshot: &WorldSnapshot, player_id: usize) -> bool {
    let Some(player) = snapshot.players.iter().find(|p| p.id == player_id) else {
        return false;
    };
    snapshot.shot_lane_clear(player.position, player.team, 3.0)
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

fn classify_movement_gait(team: Team, to_target: Vec2, sprint: bool) -> MovementGait {
    let distance = to_target.len();
    if distance < 0.18 {
        return MovementGait::Stand;
    }

    let dir = to_target.normalized();
    let forwardness = dir.dot(Vec2::new(0.0, team.attack_dir()));
    let lateral = dir.x.abs();
    let vertical = dir.y.abs();
    let retreating = forwardness < -0.34;
    let lateral_dominant = lateral > vertical * 1.18;

    if retreating {
        if distance <= 2.4 {
            MovementGait::BackWalk
        } else if !sprint && distance <= 7.5 {
            MovementGait::BackSkip
        } else {
            MovementGait::Run
        }
    } else if sprint && distance > 1.0 {
        MovementGait::Sprint
    } else if lateral_dominant && distance > 1.2 && distance <= 7.0 {
        MovementGait::SideStep
    } else if distance <= 1.8 {
        MovementGait::Walk
    } else if distance <= 4.6 {
        MovementGait::Skip
    } else if distance <= 10.0 {
        MovementGait::Jog
    } else {
        MovementGait::Run
    }
}

fn ball_velocity_after_resistance(
    velocity: Vec2,
    dt_seconds: f64,
    linear_drag_per_tick: f64,
    air_resistance: f64,
    grass_resistance_yps2: f64,
) -> Vec2 {
    let speed = velocity.len();
    if speed <= 0.0 || dt_seconds <= 0.0 {
        return velocity;
    }
    let linear_deceleration = speed * linear_drag_per_tick.clamp(0.0, 0.95) / dt_seconds;
    let air_deceleration = air_resistance.clamp(0.0, 0.10) * speed * speed;
    let low_speed_grass_bonus = (1.0 - (speed / 12.0).clamp(0.0, 1.0)) * 0.38;
    let grass_deceleration = grass_resistance_yps2.clamp(0.0, 5.0)
        * (1.0 + low_speed_grass_bonus)
        * (1.0 + (speed / 30.0).clamp(0.0, 1.0) * 0.10);
    let speed_loss = (linear_deceleration + air_deceleration + grass_deceleration) * dt_seconds;
    velocity.normalized() * (speed - speed_loss).max(0.0)
}

fn zone(v: f64, max: f64, buckets: usize) -> usize {
    ((v / max).clamp(0.0, 0.999_999) * buckets as f64).floor() as usize
}

fn pitch_grid_cell(
    position: Vec2,
    field_width: f64,
    field_length: f64,
    columns: usize,
    rows: usize,
    level: PitchGridLevel,
    parent_id: Option<usize>,
) -> PitchGridCell {
    let columns = columns.max(1);
    let rows = rows.max(1);
    let x = zone(position.x, field_width, columns);
    let y = zone(position.y, field_length, rows);
    PitchGridCell {
        level,
        columns,
        rows,
        x,
        y,
        id: y * columns + x,
        parent_id,
    }
}

fn pitch_grid_address(position: Vec2, field_width: f64, field_length: f64) -> PitchGridAddress {
    let whole_pitch = PitchGridCell::default();
    let macro_zone = pitch_grid_cell(
        position,
        field_width,
        field_length,
        PITCH_MACRO_GRID_COLUMNS,
        PITCH_MACRO_GRID_ROWS,
        PitchGridLevel::Macro,
        Some(whole_pitch.id),
    );
    let tactical = pitch_grid_cell(
        position,
        field_width,
        field_length,
        PITCH_TACTICAL_GRID_COLUMNS,
        PITCH_TACTICAL_GRID_ROWS,
        PitchGridLevel::Tactical,
        Some(macro_zone.id),
    );
    let fine = pitch_grid_cell(
        position,
        field_width,
        field_length,
        PITCH_FINE_GRID_COLUMNS,
        PITCH_FINE_GRID_ROWS,
        PitchGridLevel::Fine,
        Some(tactical.id),
    );
    PitchGridAddress {
        fine,
        tactical,
        macro_zone,
        whole_pitch,
    }
}

fn facing_bucket_from_vector(v: Vec2) -> FacingBucket {
    if v.len() <= 1e-6 {
        return FacingBucket::Unknown;
    }
    let mut degrees = v.y.atan2(v.x).to_degrees();
    if degrees < 0.0 {
        degrees += 360.0;
    }
    let idx = ((degrees + 22.5) / 45.0).floor() as usize % 8;
    match idx {
        0 => FacingBucket::East,
        1 => FacingBucket::SouthEast,
        2 => FacingBucket::South,
        3 => FacingBucket::SouthWest,
        4 => FacingBucket::West,
        5 => FacingBucket::NorthWest,
        6 => FacingBucket::North,
        _ => FacingBucket::NorthEast,
    }
}

fn facing_bucket_matches(a: FacingBucket, b: FacingBucket) -> bool {
    a == b || a == FacingBucket::Unknown || b == FacingBucket::Unknown
}

fn default_team_facing(team: Team) -> FacingBucket {
    facing_bucket_from_vector(Vec2::new(0.0, team.attack_dir()))
}

fn facing_bucket_for_player_motion(player: &PlayerAgent) -> FacingBucket {
    let facing = facing_bucket_from_vector(player.velocity);
    if facing == FacingBucket::Unknown {
        default_team_facing(player.team)
    } else {
        facing
    }
}

fn mark_player_receive_facing(players: &mut [PlayerAgent], holder_id: usize) {
    let Some(player) = players.iter_mut().find(|player| player.id == holder_id) else {
        return;
    };
    let facing = facing_bucket_for_player_motion(player);
    player.receive_facing = facing;
    player.action_facing = facing;
}

fn distance_bucket(value: f64, edges: &[f64]) -> u8 {
    edges
        .iter()
        .position(|edge| value <= *edge)
        .unwrap_or(edges.len()) as u8
}

fn confidence_bucket(value: f64) -> u8 {
    distance_bucket(value.clamp(0.0, 1.0), &[0.20, 0.40, 0.60, 0.80])
}

fn skill_bucket(value: f64) -> u8 {
    distance_bucket(ability_score(value), &[3.0, 5.0, 7.0, 9.0])
}

fn fatigue_advantage_bucket(value: f64) -> u8 {
    distance_bucket(value.clamp(-1.0, 1.0), &[-0.45, -0.15, 0.15, 0.45])
}

fn fatigue_speed_factor(stamina: f64, fatigue: f64) -> f64 {
    let cardio = ability01(stamina);
    let fatigue = fatigue.clamp(0.0, 1.0);
    (0.78 + 0.22 * cardio - fatigue * (0.18 + (1.0 - cardio) * 0.20)).clamp(0.45, 1.05)
}

fn fatigue_dribble_multiplier(observation: &SoccerPomdpObservation) -> f64 {
    let advantage = observation.perceived_fatigue_advantage.clamp(-1.0, 1.0);
    let confidence = observation
        .nearest_defender_fatigue_confidence
        .clamp(0.0, 1.0);
    let fresh_attacker = 1.0 - observation.fatigue.clamp(0.0, 1.0);
    let defender_tired = observation
        .perceived_nearest_defender_fatigue
        .clamp(0.0, 1.0);
    let cue_strength = confidence * (0.45 + fresh_attacker * 0.35 + defender_tired * 0.20);
    let bonus = (advantage - 0.10).max(0.0) * cue_strength * 1.15;
    let penalty = (-advantage - 0.10).max(0.0) * (0.45 + confidence * 0.25)
        + observation.fatigue.clamp(0.0, 1.0) * 0.24;
    (1.0 + bonus - penalty).clamp(0.45, 1.55)
}

fn shot_creation_carry_multiplier(observation: &SoccerPomdpObservation) -> f64 {
    if shot_decision_is_qualified(observation) {
        return 1.0;
    }
    let attacking_range =
        (1.0 - ((observation.yards_to_goal - 14.0).max(0.0) / 54.0)).clamp(0.0, 1.0);
    let almost_on_frame =
        (observation.shot_on_frame_probability / SHOT_ON_FRAME_MIN_PROBABILITY).clamp(0.0, 1.0);
    let almost_beats_keeper = (observation.shot_beat_goalkeeper_probability
        / SHOT_KEEPER_BEAT_MIN_PROBABILITY)
        .clamp(0.0, 1.0);
    let low_block = (1.0 - observation.shot_block_probability).clamp(0.0, 1.0);
    let shot_promise =
        (almost_on_frame * 0.54 + almost_beats_keeper * 0.30 + low_block * 0.16).clamp(0.0, 1.0);
    let open_grass = (observation.forward_dribble_space_yards / 16.0).clamp(0.0, 1.0);
    (1.0 + attacking_range * shot_promise * 0.72 + open_grass * 0.18).clamp(1.0, 1.82)
}

fn ability_score(value: f64) -> f64 {
    if !value.is_finite() {
        return 1.0;
    }
    if (0.0..1.0).contains(&value) {
        1.0 + value.clamp(0.0, 1.0) * 9.0
    } else {
        value.clamp(1.0, 10.0)
    }
}

fn ability01(value: f64) -> f64 {
    (ability_score(value) - 1.0) / 9.0
}

fn top_speed_yps_from_score(score: f64) -> f64 {
    5.5 + ability01(score) * 3.5
}

fn acceleration_yps2_from_score(score: f64) -> f64 {
    5.2 + ability01(score) * 4.1
}

fn height_inches_from_score(score: f64) -> f64 {
    65.0 + ability01(score) * 12.0
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
                jerk: Vec2::zero(),
                movement_gait: MovementGait::Stand,
                position_history: VecDeque::from([pos]),
                receive_facing: FacingBucket::Unknown,
                action_facing: default_team_facing(team),
                incoming_ball: None,
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

    fn test_decision_trace(
        snapshot: &WorldSnapshot,
        player_id: usize,
        action: &str,
    ) -> AgentDecisionTrace {
        let observation = snapshot.observation_for(player_id);
        AgentDecisionTrace {
            mdp_state: snapshot.mdp_state_for_player(player_id),
            belief: belief_from_observation(&observation),
            observation,
            operation_order: vec![action.to_string()],
            action_options: single_action_option(action),
            action_target: None,
            action: action.to_string(),
        }
    }

    #[test]
    fn default_match_has_22_players_and_3_officials() {
        let sim = SoccerMatch::default_11v11(MatchConfig::default());
        assert_eq!(sim.players.len(), 22);
        assert_eq!(sim.officials.len(), 3);
        assert_eq!(sim.config.dt_seconds, 0.1);
        assert_eq!(sim.config.duration_seconds, 600.0);
        assert_eq!(sim.config.total_ticks(), 6000);
    }

    #[test]
    fn player_skill_dimensions_are_one_to_ten_scores() {
        let sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 910,
            ..Default::default()
        });

        for player in &sim.players {
            let skills = &player.skills;
            for (name, score) in [
                ("top speed", skills.top_speed),
                ("acceleration", skills.acceleration),
                ("strength", skills.strength),
                ("height", skills.height),
                ("dribbling", skills.dribbling),
                ("aggression", skills.aggression),
                ("defensive ability", skills.defending),
                ("right shot", skills.right_foot_shot_power),
                ("left shot", skills.left_foot_shot_power),
                ("passing completion", skills.passing_completion_rate),
                ("flair passing", skills.flair_passing),
                ("left crossing", skills.crossing_left),
                ("right crossing", skills.crossing_right),
                ("goalkeeping", skills.goalkeeping),
                ("defensive tracking", skills.defensive_tracking),
            ] {
                assert!(
                    (1.0..=10.0).contains(&score),
                    "{} {} score out of range: {}",
                    player.name,
                    name,
                    score
                );
            }
        }

        assert!(top_speed_yps_from_score(10.0) > top_speed_yps_from_score(1.0));
        assert!(acceleration_yps2_from_score(10.0) > acceleration_yps2_from_score(1.0));
        assert!(height_inches_from_score(10.0) > height_inches_from_score(1.0));

        let player = &sim.players[9];
        let snapshot = WorldSnapshot::from_match(&sim);
        let observation = snapshot.observation_for(player.id);
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(player.id),
            &observation,
            player.team,
            player.role,
        );
        assert_eq!(
            state.skill_top_speed_bin,
            skill_bucket(player.skills.top_speed)
        );
        assert_eq!(
            state.skill_acceleration_bin,
            skill_bucket(player.skills.acceleration)
        );
        assert_eq!(
            state.skill_strength_bin,
            skill_bucket(player.skills.strength)
        );
        assert_eq!(state.skill_height_bin, skill_bucket(player.skills.height));
        assert_eq!(
            state.skill_dribbling_bin,
            skill_bucket(player.skills.dribbling)
        );
        assert_eq!(
            state.skill_aggression_bin,
            skill_bucket(player.skills.aggression)
        );
        assert_eq!(
            observation.skill_defending, player.skills.defending,
            "POMDP observation should carry defensive ability"
        );
        assert_eq!(
            state.skill_defending_bin,
            skill_bucket(player.skills.defending)
        );
        assert_eq!(
            state.skill_right_foot_shot_bin,
            skill_bucket(player.skills.right_foot_shot_power)
        );
        assert_eq!(
            state.skill_left_foot_shot_bin,
            skill_bucket(player.skills.left_foot_shot_power)
        );
        assert_eq!(
            state.skill_passing_completion_bin,
            skill_bucket(player.skills.passing_completion_rate)
        );
        assert_eq!(
            state.skill_flair_passing_bin,
            skill_bucket(player.skills.flair_passing)
        );
        assert_eq!(
            state.skill_crossing_bin,
            skill_bucket(
                player
                    .skills
                    .crossing_left
                    .max(player.skills.crossing_right)
            )
        );
        assert_eq!(
            state.skill_crossing_left_bin,
            skill_bucket(player.skills.crossing_left)
        );
        assert_eq!(
            state.skill_crossing_right_bin,
            skill_bucket(player.skills.crossing_right)
        );
        assert_eq!(
            state.skill_goalkeeping_bin,
            skill_bucket(player.skills.goalkeeping)
        );
        assert_eq!(
            state.skill_defensive_tracking_bin,
            skill_bucket(player.skills.defensive_tracking)
        );
    }

    #[test]
    fn time_window_probability_scales_with_timestep() {
        let per_second = time_window_probability(0.50, 1.0);
        let per_half_second = time_window_probability(0.50, 0.5);
        let per_tenth = time_window_probability(0.50, 0.1);
        let compounded_tenths = 1.0 - (1.0 - per_tenth).powf(10.0);

        assert!((per_second - 0.50).abs() < 1e-12);
        assert!(per_tenth > 0.0);
        assert!(per_tenth < per_half_second);
        assert!(per_half_second < per_second);
        assert!((compounded_tenths - per_second).abs() < 1e-12);
    }

    #[test]
    fn completed_pass_reward_prioritizes_forward_progression() {
        let field_length = 120.0;

        assert_eq!(
            completed_pass_reward(
                Team::Home,
                Vec2::new(40.0, 40.0),
                Vec2::new(40.0, 52.0),
                field_length
            ),
            5.0
        );
        assert_eq!(
            completed_pass_reward(
                Team::Home,
                Vec2::new(40.0, 76.0),
                Vec2::new(40.0, 88.0),
                field_length
            ),
            6.0
        );
        assert_eq!(
            completed_pass_reward(
                Team::Home,
                Vec2::new(40.0, 76.0),
                Vec2::new(52.0, 76.2),
                field_length
            ),
            3.0
        );
        assert_eq!(
            completed_pass_reward(
                Team::Home,
                Vec2::new(40.0, 40.0),
                Vec2::new(40.0, 30.0),
                field_length
            ),
            0.2
        );
        assert_eq!(
            completed_pass_reward(
                Team::Home,
                Vec2::new(40.0, 76.0),
                Vec2::new(40.0, 66.0),
                field_length
            ),
            1.4
        );
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
    fn agent_schedule_records_shuffled_field_agents_and_ball_loop() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.2,
            seed: 303,
            ..Default::default()
        });

        sim.run_time_step();

        let frame = sim.to_frame();
        assert_eq!(frame.agent_schedule.len(), 26);
        assert_eq!(
            frame
                .agent_schedule
                .last()
                .map(|entry| (&entry.kind, entry.id)),
            Some((&AgentScheduleKind::Ball, BALL_AGENT_ID))
        );
        assert_eq!(
            frame
                .agent_schedule
                .iter()
                .filter(|entry| entry.kind == AgentScheduleKind::Player)
                .count(),
            22
        );
        assert_eq!(
            frame
                .agent_schedule
                .iter()
                .filter(|entry| entry.kind == AgentScheduleKind::Official)
                .count(),
            3
        );
        let scheduled_player_ids = frame
            .agent_schedule
            .iter()
            .filter(|entry| entry.kind == AgentScheduleKind::Player)
            .map(|entry| entry.id)
            .collect::<std::collections::BTreeSet<_>>();
        let expected_player_ids = (0..22).collect::<std::collections::BTreeSet<_>>();
        assert_eq!(scheduled_player_ids, expected_player_ids);
        let scheduled_official_ids = frame
            .agent_schedule
            .iter()
            .filter(|entry| entry.kind == AgentScheduleKind::Official)
            .map(|entry| entry.id)
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(
            scheduled_official_ids,
            [22, 23, 24]
                .into_iter()
                .collect::<std::collections::BTreeSet<_>>()
        );
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
        assert!(sim.ball.velocity.len() < 10.0);
        assert!(sim.ball.acceleration.x <= 0.0);
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
        let mut rng = mulberry32(1001);
        sim.central_brain.run_time_step(&before, &mut rng);
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
    fn team_brain_samples_defensive_cover_targets_from_distribution() {
        let mut rng = mulberry32(404);
        let mut counts = [0usize; 5];
        for _ in 0..2_000 {
            counts[sample_defensive_cover_target(&mut rng)] += 1;
        }

        assert_eq!(counts.iter().sum::<usize>(), 2_000);
        assert!((160..=240).contains(&counts[0]));
        assert!((160..=240).contains(&counts[1]));
        assert!((520..=680).contains(&counts[2]));
        assert!((520..=680).contains(&counts[3]));
        assert!((320..=480).contains(&counts[4]));
    }

    #[test]
    fn team_brain_cover_rule_sets_goal_side_defensive_line() {
        let conservative = tactical_directive_for_team(
            Team::Home,
            TacticalPhase::AwayAttack,
            Some(Team::Away),
            Vec2::new(40.0, 37.0),
            0,
            DEFAULT_FIELD_WIDTH_YARDS,
            DEFAULT_FIELD_LENGTH_YARDS,
            DefensiveCoverProfile {
                target: 3,
                actual: 0,
                foremost_attacker_y: Some(33.0),
            },
        );
        let aggressive = tactical_directive_for_team(
            Team::Home,
            TacticalPhase::AwayAttack,
            Some(Team::Away),
            Vec2::new(40.0, 37.0),
            0,
            DEFAULT_FIELD_WIDTH_YARDS,
            DEFAULT_FIELD_LENGTH_YARDS,
            DefensiveCoverProfile {
                target: 0,
                actual: 4,
                foremost_attacker_y: Some(33.0),
            },
        );

        assert_eq!(conservative.defensive_cover_target, 3);
        assert_eq!(conservative.defensive_cover_actual, 0);
        assert_eq!(conservative.foremost_attacker_y, Some(33.0));
        assert!(conservative.defensive_line_y < 33.0);
        assert!(conservative.defensive_line_y < aggressive.defensive_line_y);
    }

    #[test]
    fn match_frame_exposes_central_brain_global_awareness() {
        let kickoff = SoccerMatch::default_11v11(MatchConfig::default()).to_frame();
        assert_eq!(kickoff.central_brain.possession_team, Some(Team::Home));
        assert_eq!(kickoff.central_brain.ball_holder, Some(5));

        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.players[9].position = Vec2::new(41.0, 94.0);
        sim.players[9].velocity = Vec2::new(0.5, 1.0);
        sim.ball.holder = Some(9);
        sim.ball.position = sim.players[9].position;
        sim.ball.velocity = Vec2::new(0.5, 1.0);
        sim.ball.last_touch_team = Some(Team::Home);
        let before = WorldSnapshot::from_match(&sim);
        let mut rng = mulberry32(1002);
        sim.central_brain.run_time_step(&before, &mut rng);

        let frame = sim.to_frame();

        assert_eq!(frame.central_brain.tracked_players.len(), 22);
        assert_eq!(frame.central_brain.tracked_officials, 3);
        assert_eq!(frame.central_brain.possession_team, Some(Team::Home));
        assert_eq!(frame.central_brain.ball_holder, Some(9));
        assert_eq!(frame.central_brain.ball_position, sim.ball.position);
        let striker = frame
            .central_brain
            .tracked_players
            .iter()
            .find(|p| p.id == 9)
            .expect("central brain tracks striker");
        assert_eq!(striker.position, sim.players[9].position);
        assert_eq!(striker.velocity, sim.players[9].velocity);
    }

    #[test]
    fn pomdp_observation_tracks_player_visibility() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let observer = 6;
        sim.players[observer].position = Vec2::new(40.0, 60.0);
        sim.players[observer].velocity = Vec2::new(3.0, 0.0);
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(70.0, 60.0);
        sim.ball.velocity = Vec2::zero();
        for away in 11..22 {
            sim.players[away].position = Vec2::new(12.0, 60.0);
        }

        let snapshot = WorldSnapshot::from_match(&sim);
        assert!(snapshot.player_can_see_point(observer, sim.ball.position));
        let visible = snapshot.observation_for(observer);
        assert!(visible.visible_ball);
        assert_eq!(visible.visible_opponents, 0);

        sim.ball.position = Vec2::new(10.0, 60.0);
        let snapshot = WorldSnapshot::from_match(&sim);
        let hidden = snapshot.observation_for(observer);
        assert!(!hidden.visible_ball);
        assert!(hidden.ball_distance > visible.ball_distance);
        assert!(belief_from_observation(&hidden).possession_confidence < 0.35);
    }

    #[test]
    fn visible_pass_targets_filter_hidden_teammates() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let passer = 6;
        let visible_teammate = 7;
        let hidden_teammate = 8;
        sim.players[passer].position = Vec2::new(40.0, 60.0);
        sim.players[passer].velocity = Vec2::new(4.0, 0.0);
        sim.players[visible_teammate].position = Vec2::new(56.0, 60.0);
        sim.players[hidden_teammate].position = Vec2::new(18.0, 60.0);
        for home in 0..11 {
            if ![passer, visible_teammate, hidden_teammate].contains(&home) {
                sim.players[home].position = Vec2::new(6.0, 18.0 + home as f64);
            }
        }
        for away in 11..22 {
            sim.players[away].position = Vec2::new(72.0, 95.0);
        }
        sim.ball.holder = Some(passer);
        sim.ball.position = sim.players[passer].position;
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        assert!(snapshot.player_can_see_player(passer, visible_teammate));
        assert!(!snapshot.player_can_see_player(passer, hidden_teammate));
        let visible_targets = snapshot.ranked_visible_pass_targets(passer, 3);
        assert!(visible_targets.contains(&visible_teammate));
        assert!(!visible_targets.contains(&hidden_teammate));
        assert_eq!(
            snapshot.observation_for(passer).visible_pass_options,
            visible_targets.len()
        );
    }

    #[test]
    fn player_operation_order_is_weighted_by_internal_preferences() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            seed: 203,
            ..Default::default()
        });
        let player_id = 5;
        sim.players[player_id].position = Vec2::new(40.0, 101.0);
        sim.players[player_id].velocity = Vec2::new(0.0, 5.0);
        sim.players[player_id].skills.decision_noise = 1.0;
        sim.players[6].position = Vec2::new(46.0, 102.0);
        sim.players[7].position = Vec2::new(34.0, 100.0);
        sim.players[8].position = Vec2::new(52.0, 99.0);
        for away in 11..22 {
            sim.players[away].position = Vec2::new(8.0, 110.0);
        }
        sim.ball.holder = Some(player_id);
        sim.ball.position = sim.players[player_id].position;
        sim.ball.velocity = Vec2::zero();
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        assert!(snapshot.observation_for(player_id).shot_lane_open);
        assert!(!snapshot
            .ranked_visible_pass_targets(player_id, 3)
            .is_empty());

        let sample_first_order = |preferences: AgentPreferences| {
            let mut first_counts: HashMap<String, usize> = HashMap::new();
            for seed in 0..180 {
                let mut player = sim.players[player_id].clone();
                player.preferences = preferences.clone();
                let mut rng = mulberry32(10_000 + seed);
                let _ = player.run_time_step(&snapshot, None, None, &mut rng);
                let first = player
                    .last_decision
                    .as_ref()
                    .and_then(|decision| decision.operation_order.first())
                    .cloned()
                    .unwrap_or_default();
                *first_counts.entry(first).or_insert(0) += 1;
            }
            first_counts
        };

        let shoot_first = sample_first_order(AgentPreferences {
            shoot_bias: 0.98,
            pass_bias: 0.04,
            dribble_bias: 0.04,
            open_space_bias: 0.70,
        });
        let pass_first = sample_first_order(AgentPreferences {
            shoot_bias: 0.04,
            pass_bias: 0.98,
            dribble_bias: 0.04,
            open_space_bias: 0.70,
        });

        let shoot_pref_shoot = *shoot_first.get("shoot").unwrap_or(&0);
        let pass_pref_pass = pass_first
            .iter()
            .filter(|(label, _)| label.starts_with("pass"))
            .map(|(_, count)| *count)
            .sum::<usize>();
        let pass_pref_shoot = *pass_first.get("shoot").unwrap_or(&0);
        assert!(
            shoot_pref_shoot > pass_pref_shoot,
            "shoot bias should make shoot appear first more often than pass bias: shoot={shoot_first:?} pass={pass_first:?}"
        );
        assert!(
            pass_pref_pass > pass_pref_shoot,
            "pass-biased player should inspect passes first more often: {pass_first:?}"
        );
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
        assert!(sim.players[0]
            .history_jerk_estimate(sim.config.dt_seconds)
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
        assert!(snapshot
            .player_jerk(0)
            .expect("snapshot jerk")
            .len()
            .is_finite());
        assert!(snapshot
            .shared_positions
            .latest_for(0)
            .expect("latest player sample")
            .jerk
            .len()
            .is_finite());
        let frame = sim.to_frame();
        assert_eq!(
            frame.players[0].position_history.len(),
            PLAYER_POSITION_HISTORY_LIMIT
        );
        assert_eq!(
            frame.players[0].skills.top_speed,
            sim.players[0].skills.top_speed
        );
        assert_eq!(
            frame.players[0].skills.stamina,
            sim.players[0].skills.stamina
        );
        assert_eq!(frame.players[0].fatigue, sim.players[0].fatigue);
        assert_eq!(frame.ball_history.len(), BALL_POSITION_HISTORY_LIMIT);
    }

    #[test]
    fn officials_track_rolling_position_history_and_jerk() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 6.0,
            seed: 205,
            ..Default::default()
        });

        for _ in 0..60 {
            sim.run_time_step();
        }

        for official in &sim.officials {
            assert_eq!(
                official.position_history.len(),
                PLAYER_POSITION_HISTORY_LIMIT
            );
            assert!(official
                .history_velocity_estimate(sim.config.dt_seconds)
                .len()
                .is_finite());
            assert!(official
                .history_acceleration_estimate(sim.config.dt_seconds)
                .len()
                .is_finite());
            assert!(official
                .history_jerk_estimate(sim.config.dt_seconds)
                .len()
                .is_finite());
        }

        let frame = sim.to_frame();
        assert!(frame.officials.iter().all(|official| {
            official.velocity.len().is_finite()
                && official.acceleration.len().is_finite()
                && official.jerk.len().is_finite()
        }));
        assert!(frame
            .officials
            .iter()
            .all(|official| { official.position_history.len() == PLAYER_POSITION_HISTORY_LIMIT }));
    }

    #[test]
    fn player_movement_uses_discrete_soccer_gaits() {
        assert_eq!(
            classify_movement_gait(Team::Home, Vec2::new(0.0, 1.0), false),
            MovementGait::Walk
        );
        assert_eq!(
            classify_movement_gait(Team::Home, Vec2::new(0.0, -1.0), false),
            MovementGait::BackWalk
        );
        assert_eq!(
            classify_movement_gait(Team::Home, Vec2::new(0.0, -5.0), false),
            MovementGait::BackSkip
        );
        assert_eq!(
            classify_movement_gait(Team::Away, Vec2::new(0.0, 5.0), false),
            MovementGait::BackSkip
        );
        assert_eq!(
            classify_movement_gait(Team::Home, Vec2::new(4.0, 0.2), false),
            MovementGait::SideStep
        );
        assert_eq!(
            classify_movement_gait(Team::Home, Vec2::new(0.0, 6.0), false),
            MovementGait::Jog
        );
        assert_eq!(
            classify_movement_gait(Team::Home, Vec2::new(0.0, 12.0), false),
            MovementGait::Run
        );
        assert_eq!(
            classify_movement_gait(Team::Home, Vec2::new(0.0, 6.0), true),
            MovementGait::Sprint
        );

        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.players[0].position = Vec2::new(40.0, 60.0);
        sim.players[0].velocity = Vec2::zero();
        sim.move_player_towards(0, Vec2::new(40.0, 55.0), false);
        assert_eq!(sim.players[0].movement_gait, MovementGait::BackSkip);
    }

    #[test]
    fn fatigue_accumulates_from_repeated_sprints_and_recovers_at_rest() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 1.0,
            ..Default::default()
        });
        let high_stamina = 0;
        let low_stamina = 1;
        for player_id in [high_stamina, low_stamina] {
            sim.players[player_id].position = Vec2::new(25.0 + player_id as f64 * 10.0, 20.0);
            sim.players[player_id].velocity = Vec2::zero();
            sim.players[player_id].skills.top_speed = 8.0;
            sim.players[player_id].skills.acceleration = 8.0;
            sim.players[player_id].fatigue = 0.0;
        }
        sim.players[high_stamina].skills.stamina = 10.0;
        sim.players[low_stamina].skills.stamina = 1.0;

        for _ in 0..12 {
            sim.move_player_towards(high_stamina, Vec2::new(25.0, 220.0), true);
            sim.move_player_towards(low_stamina, Vec2::new(35.0, 220.0), true);
        }

        let high_after_sprints = sim.players[high_stamina].fatigue;
        let low_after_sprints = sim.players[low_stamina].fatigue;
        assert!(low_after_sprints > high_after_sprints + 0.12);

        for _ in 0..8 {
            let rest_spot = sim.players[low_stamina].position;
            sim.move_player_towards(low_stamina, rest_spot, false);
        }

        assert!(sim.players[low_stamina].fatigue < low_after_sprints);
    }

    #[test]
    fn center_referee_keeps_clear_of_ball_space() {
        let sim = SoccerMatch::default_11v11(MatchConfig::default());
        let mut snapshot = WorldSnapshot::from_match(&sim);
        let center = Vec2::new(snapshot.field_width * 0.5, snapshot.field_length * 0.5);
        snapshot.ball.position = center;
        snapshot.ball.holder = None;

        let mut center_ref = OfficialAgent::new(99, OfficialKind::CenterReferee, center);
        let mut rng = SeededRandom::new(300);
        for _ in 0..24 {
            center_ref.run_time_step(&snapshot, &mut rng);
        }

        assert!(
            center_ref.position.distance(center) > 2.5,
            "center ref should not sit on top of the ball"
        );
        assert!(
            official_clearance_target(OfficialKind::CenterReferee, center, center, &snapshot)
                .distance(center)
                >= CENTER_REF_BALL_CLEARANCE_YARDS
        );
    }

    #[test]
    fn loose_ball_control_ignores_officials() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        for player in &mut sim.players {
            player.position = Vec2::new(5.0, 5.0);
            player.velocity = Vec2::zero();
        }
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(74.0, 114.0);
        sim.ball.velocity = Vec2::zero();
        sim.officials[0].position = sim.ball.position;

        sim.integrate_ball();

        assert_eq!(sim.ball.holder, None);
        assert_eq!(sim.ball.position, Vec2::new(74.0, 114.0));
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
            pass_flight: PassFlight::Floor,
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
            pass_flight: PassFlight::Floor,
            shoot: false,
            target_player: None,
        }));
        let latest = q.drain_latest_by_slot();
        assert_eq!(latest.get(&0).unwrap().seq, 2);
        assert!(latest.get(&0).unwrap().sprint);
    }

    #[test]
    fn human_input_queue_can_drain_one_controller_slot() {
        let q = SharedHumanInputs::new();
        assert!(q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(0),
            seq: 1,
            axis: Vec2::new(1.0, 0.0),
            sprint: false,
            pass: false,
            pass_flight: PassFlight::Floor,
            shoot: false,
            target_player: None,
        }));
        assert!(q.push(HumanInputFrame {
            controller_slot: 1,
            player_id: Some(1),
            seq: 4,
            axis: Vec2::new(0.0, 1.0),
            sprint: true,
            pass: false,
            pass_flight: PassFlight::Floor,
            shoot: false,
            target_player: None,
        }));
        assert!(q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(0),
            seq: 2,
            axis: Vec2::new(-1.0, 0.0),
            sprint: true,
            pass: false,
            pass_flight: PassFlight::Floor,
            shoot: false,
            target_player: None,
        }));

        let slot_zero = q.drain_latest_for_slot(0).expect("slot 0 input");
        assert_eq!(slot_zero.seq, 2);
        assert_eq!(slot_zero.axis, Vec2::new(-1.0, 0.0));
        assert_eq!(q.queued_len(), 1);

        let latest = q.drain_latest_by_slot();
        assert_eq!(latest.len(), 1);
        assert_eq!(latest.get(&1).expect("slot 1 input").seq, 4);
    }

    #[test]
    fn shared_human_inputs_notify_waiting_main_loop() {
        let q = SharedHumanInputs::new();
        let version = q.notification_version();
        let waiter_queue = q.clone();
        let waiter = std::thread::spawn(move || {
            waiter_queue.wait_for_change_since(version, Duration::from_millis(200))
        });
        std::thread::sleep(Duration::from_millis(2));
        assert!(q.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(0),
            seq: 1,
            axis: Vec2::new(1.0, 0.0),
            sprint: true,
            pass: false,
            pass_flight: PassFlight::Floor,
            shoot: false,
            target_player: None,
        }));

        let next_version = waiter.join().expect("waiter joins");
        assert!(next_version > version);
        assert!(q.wait_for_pending_input(Duration::from_millis(0)));
    }

    #[test]
    fn native_controller_threads_debounce_and_cap_at_four_slots() {
        let q = SharedHumanInputs::new();
        let controllers = spawn_human_controller_threads(q.clone(), 6, Duration::from_millis(1))
            .expect("spawn controller threads");
        assert_eq!(controllers.len(), 4);
        assert_eq!(
            controllers
                .iter()
                .map(|controller| controller.controller_slot())
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3]
        );

        controllers[2]
            .send_input(HumanInputFrame {
                controller_slot: 99,
                player_id: Some(2),
                seq: 1,
                axis: Vec2::new(0.0, 1.0),
                sprint: false,
                pass: false,
                pass_flight: PassFlight::Floor,
                shoot: false,
                target_player: None,
            })
            .expect("send first input");
        controllers[2]
            .send_input(HumanInputFrame {
                controller_slot: 99,
                player_id: Some(2),
                seq: 2,
                axis: Vec2::new(1.0, 0.0),
                sprint: true,
                pass: false,
                pass_flight: PassFlight::Floor,
                shoot: false,
                target_player: None,
            })
            .expect("send latest input");

        assert!(q.wait_for_pending_input(Duration::from_millis(200)));
        std::thread::sleep(Duration::from_millis(5));
        let input = q.drain_latest_for_slot(2).expect("slot 2 input");
        assert_eq!(input.controller_slot, 2);
        assert_eq!(input.seq, 2);
        assert!(input.sprint);

        for controller in controllers {
            controller.stop().expect("controller stops");
        }
    }

    #[test]
    fn controller_mailbox_overwrites_pending_input_without_queue_growth() {
        let q = SharedHumanInputs::new();
        let controller = HumanControllerThread::spawn(q.clone(), 0, Duration::from_millis(50))
            .expect("spawn controller thread");

        for seq in 1..=20 {
            assert!(controller
                .send_input(HumanInputFrame {
                    controller_slot: 99,
                    player_id: Some(0),
                    seq,
                    axis: Vec2::new(seq as f64, 0.0),
                    sprint: seq == 20,
                    pass: false,
                    pass_flight: PassFlight::Floor,
                    shoot: false,
                    target_player: None,
                })
                .expect("send input"));
        }
        assert!(!controller
            .send_input(HumanInputFrame {
                controller_slot: 99,
                player_id: Some(0),
                seq: 19,
                axis: Vec2::new(-1.0, 0.0),
                sprint: false,
                pass: false,
                pass_flight: PassFlight::Floor,
                shoot: false,
                target_player: None,
            })
            .expect("stale input is rejected without stopping controller"));

        let pending_stats = controller.stats();
        assert_eq!(pending_stats.accepted_frames, 20);
        assert_eq!(pending_stats.rejected_stale_frames, 1);
        assert!(pending_stats.overwritten_frames > 0);
        assert!(pending_stats.pending);

        assert!(q.wait_for_pending_input(Duration::from_millis(300)));
        let input = q.drain_latest_for_slot(0).expect("slot 0 latest input");
        assert_eq!(input.controller_slot, 0);
        assert_eq!(input.seq, 20);
        assert!(input.sprint);
        assert_eq!(q.queued_len(), 0);

        let pushed_stats = controller.stats();
        assert_eq!(pushed_stats.pushed_frames, 1);
        assert!(!pushed_stats.pending);

        controller.stop().expect("controller stops");
    }

    #[test]
    fn realtime_session_owns_default_controller_threads() {
        let session = SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 8,
            seed: 781,
            ..Default::default()
        });

        assert_eq!(session.owned_controller_thread_count(), 4);
        assert_eq!(session.match_ref().config.human_slots(), 4);
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
            pass_flight: PassFlight::Floor,
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
            pass_flight: PassFlight::Floor,
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
            pass_flight: PassFlight::Floor,
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
            pass_flight: PassFlight::Floor,
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
                        pass_flight: PassFlight::Floor,
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
        session
            .match_mut()
            .assign_controller_slot(0, Some(0))
            .expect("assign human controller");
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
                pass_flight: PassFlight::Floor,
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
    fn player_loop_polls_only_its_assigned_controller_slot() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 2,
            seed: 780,
            ..Default::default()
        });
        sim.clear_controller_assignments();
        sim.assign_controller_slot(0, Some(0))
            .expect("assign slot 0");
        let input_queue = sim.human_inputs.clone();
        assert!(input_queue.push(HumanInputFrame {
            controller_slot: 0,
            player_id: Some(0),
            seq: 1,
            axis: Vec2::new(1.0, 0.0),
            sprint: true,
            pass: false,
            pass_flight: PassFlight::Floor,
            shoot: false,
            target_player: None,
        }));
        assert!(input_queue.push(HumanInputFrame {
            controller_slot: 1,
            player_id: Some(1),
            seq: 1,
            axis: Vec2::new(0.0, 1.0),
            sprint: true,
            pass: false,
            pass_flight: PassFlight::Floor,
            shoot: false,
            target_player: None,
        }));

        sim.run_time_step();

        assert_eq!(
            sim.players[0]
                .last_decision
                .as_ref()
                .expect("controlled player decision")
                .operation_order
                .first()
                .map(String::as_str),
            Some("human-input")
        );
        let remaining = input_queue.drain_latest_by_slot();
        assert!(!remaining.contains_key(&0));
        assert_eq!(remaining.get(&1).expect("unassigned slot remains").seq, 1);
    }

    #[test]
    fn realtime_session_starts_unassigned_so_players_remain_autonomous_until_selected() {
        let mut session = SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 1,
            seed: 771,
            ..Default::default()
        });
        assert!(session.match_ref().controller_assignments().is_empty());

        let response = session.step(SoccerStepRequest {
            inputs: vec![HumanInputFrame {
                controller_slot: 0,
                player_id: Some(0),
                seq: 1,
                axis: Vec2::new(1.0, 0.0),
                sprint: true,
                pass: false,
                pass_flight: PassFlight::Floor,
                shoot: false,
                target_player: None,
            }],
            ticks: 1,
            record_every_ticks: Some(1),
        });

        assert_eq!(response.accepted_inputs, 1);
        assert_eq!(session.match_ref().players[0].controller_slot, None);
        assert_ne!(
            session.match_ref().players[0]
                .last_decision
                .as_ref()
                .expect("autonomous decision")
                .operation_order
                .first()
                .map(String::as_str),
            Some("human-input")
        );
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
        let passer_position = session.match_ref().players[5].position;
        let target_position = session.match_ref().players[8].position;

        let response = session.step(SoccerStepRequest {
            inputs: vec![HumanInputFrame {
                controller_slot: 0,
                player_id: Some(5),
                seq: 1,
                axis: Vec2::zero(),
                sprint: false,
                pass: true,
                pass_flight: PassFlight::Floor,
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
        let decision = session.match_ref().players[5]
            .last_decision
            .as_ref()
            .expect("human pass decision");
        assert_eq!(decision.action, "pass");
        let action_target = decision
            .action_target
            .as_ref()
            .expect("human pass target trace");
        assert_eq!(action_target.player_id, Some(8));
        assert_eq!(action_target.point, Some(target_position));
        assert_eq!(
            action_target.grid.expect("target grid").fine.id,
            pitch_grid_address(
                target_position,
                session.match_ref().config.field_width_yards,
                session.match_ref().config.field_length_yards
            )
            .fine
            .id
        );
        assert_eq!(
            action_target.facing,
            facing_bucket_from_vector(target_position - passer_position)
        );
    }

    #[test]
    fn human_input_can_choose_aerial_pass_flight() {
        let mut session = SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 1,
            seed: 79,
            ..Default::default()
        });
        {
            let sim = session.match_mut();
            sim.players[5].controller_slot = Some(0);
            sim.players[5].position = Vec2::new(14.0, 84.0);
            sim.players[8].position = Vec2::new(42.0, 102.0);
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
                pass_flight: PassFlight::Aerial,
                shoot: false,
                target_player: Some(8),
            }],
            ticks: 1,
            record_every_ticks: Some(1),
        });

        assert_eq!(response.accepted_inputs, 1);
        let pass = session
            .match_ref()
            .pending_pass
            .as_ref()
            .expect("pending aerial pass");
        assert_eq!(pass.from, 5);
        assert_eq!(pass.target, Some(8));
        assert_eq!(pass.flight, PassFlight::Aerial);
        let decision = session.match_ref().players[5]
            .last_decision
            .as_ref()
            .expect("human aerial decision");
        assert_eq!(decision.action, "aerial-pass");
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
                pass_flight: PassFlight::Floor,
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
    fn realtime_session_accepts_four_human_controller_slots() {
        let mut session = SoccerRealtimeSession::new_without_controller_threads(MatchConfig {
            duration_seconds: 1.0,
            seed: 83,
            ..Default::default()
        });

        for slot in 0..4 {
            session
                .assign_controller_slot(SoccerControllerAssignmentRequest {
                    controller_slot: slot,
                    player_id: Some(slot),
                })
                .expect("assign four human slots");
        }

        assert_eq!(session.match_ref().config.human_slots(), 4);
        assert_eq!(session.match_ref().controller_assignments().len(), 4);

        let response = session.step(SoccerStepRequest {
            inputs: (0..4)
                .map(|slot| HumanInputFrame {
                    controller_slot: slot,
                    player_id: Some(slot),
                    seq: 1,
                    axis: Vec2::new(1.0, 0.0),
                    sprint: slot % 2 == 0,
                    pass: false,
                    pass_flight: PassFlight::Floor,
                    shoot: false,
                    target_player: None,
                })
                .collect(),
            ticks: 1,
            record_every_ticks: Some(1),
        });

        assert_eq!(response.accepted_inputs, 4);
        for player_id in 0..4 {
            let decision = session.match_ref().players[player_id]
                .last_decision
                .as_ref()
                .expect("human-controlled player decision");
            assert_eq!(decision.operation_order, vec!["human-input".to_string()]);
            assert_eq!(decision.action, "human-move");
        }
    }

    #[test]
    fn realtime_session_exports_live_tracking_dataset() {
        let mut session = SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            seed: 86,
            ..Default::default()
        });

        assert_eq!(session.tracking_dataset().frames.len(), 1);
        let response = session.step(SoccerStepRequest {
            ticks: 2,
            record_every_ticks: Some(1),
            ..Default::default()
        });
        let tracking = session.tracking_dataset();

        assert_eq!(tracking.source, "live-session");
        assert_eq!(tracking.frames.len(), 3);
        assert_eq!(tracking.frames[0].tick, 0);
        assert_eq!(tracking.frames[2].tick, 2);
        assert_eq!(tracking.frames[2].players.len(), 22);
        assert_eq!(
            tracking.frames[2].ball_position,
            response.frame.ball.position
        );
        assert!(tracking.to_learning_dataset().is_ok());
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
        assert_eq!(value["controllerAssignments"].as_array().unwrap().len(), 0);
        assert_eq!(value["learningTransitions"].as_array().unwrap().len(), 66);
        assert_eq!(value["learning"]["totalTransitions"], 66);
        assert_eq!(value["learning"]["teamPoliciesEnabled"], true);
        assert_eq!(value["learning"]["adversarialLearningEnabled"], true);
        assert!(value["learning"]["homePolicyEntries"].as_u64().unwrap() > 0);
        assert!(value["learning"]["awayPolicyEntries"].as_u64().unwrap() > 0);
        assert!(
            value["learning"]["homePolicyTargetEntries"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            value["learning"]["awayPolicyTargetEntries"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(value["learning"]["homePolicyVisits"].as_u64().unwrap() > 0);
        assert!(value["learning"]["awayPolicyVisits"].as_u64().unwrap() > 0);
        let player_decision = &value["frame"]["players"][0]["lastDecision"];
        assert!(player_decision.get("mdpState").is_some());
        assert!(player_decision.get("observation").is_some());
        assert!(player_decision.get("belief").is_some());
        assert!(player_decision.get("action").is_some());
        let action_options = player_decision["actionOptions"]
            .as_array()
            .expect("decision action options");
        assert!(!action_options.is_empty());
        assert!(action_options[0].get("label").is_some());
        assert!(action_options[0].get("probability").is_some());
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
    fn player_mdp_state_includes_hierarchical_pitch_grid_and_facing() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 141,
            ..Default::default()
        });
        sim.players[5].position = Vec2::new(22.0, 66.0);
        sim.players[5].velocity = Vec2::new(2.0, 0.0);
        sim.players[5].receive_facing = FacingBucket::NorthWest;
        sim.players[5].action_facing = FacingBucket::West;

        let snapshot = WorldSnapshot::from_match(&sim);
        let state = snapshot.mdp_state_for_player(5);
        let observation = snapshot.observation_for(5);

        assert_eq!(state.player_grid.fine.x, 3);
        assert_eq!(state.player_grid.fine.y, 8);
        assert_eq!(state.player_grid.fine.id, 99);
        assert_eq!(state.player_grid.fine.parent_id, Some(25));
        assert_eq!(state.player_grid.tactical.id, 25);
        assert_eq!(state.player_grid.tactical.parent_id, Some(6));
        assert_eq!(state.player_grid.macro_zone.id, 6);
        assert_eq!(state.player_grid.macro_zone.parent_id, Some(0));
        assert_eq!(state.player_grid.whole_pitch.id, 0);
        assert_eq!(state.receive_facing, FacingBucket::NorthWest);
        assert_eq!(state.action_facing, FacingBucket::East);
        assert_eq!(observation.player_grid.fine.id, state.player_grid.fine.id);
        assert_eq!(observation.action_facing, FacingBucket::East);
    }

    #[test]
    fn pomdp_observation_includes_goal_keeper_pressure_and_forward_space_features() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1411,
            ..Default::default()
        });
        let player_id = 9;
        sim.players[player_id].position = Vec2::new(40.0, 90.0);
        sim.players[player_id].velocity = Vec2::new(0.0, 3.0);
        sim.players[11].position = Vec2::new(42.0, 116.0);
        for away in 12..22 {
            sim.players[away].position = Vec2::new(74.0, 30.0 + away as f64);
        }
        sim.players[13].position = Vec2::new(40.8, 97.0);
        sim.ball.holder = Some(player_id);
        sim.ball.position = sim.players[player_id].position;
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        let observation = snapshot.observation_for(player_id);
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(player_id),
            &observation,
            Team::Home,
            sim.players[player_id].role,
        );

        assert!(observation.yards_to_goal > 0.0);
        assert!(observation.yards_to_own_goal > observation.yards_to_goal);
        assert!(observation.opponent_goal_angle_degrees > 0.0);
        assert!(observation.opposing_goalkeeper_distance > 0.0);
        assert!(observation.opposing_goalkeeper_angle_degrees > 0.0);
        assert!(observation.forward_dribble_space_yards < 10.0);
        assert!(observation.real_pressure > 0.0);
        assert!(observation.perceived_pressure >= observation.real_pressure - 0.15);
        assert!(observation.real_time_on_ball_seconds < 2.8);
        assert!(observation.perceived_time_on_ball_seconds < 2.8);
        assert!(state.yards_to_own_goal_bin > 0);
        assert!(state.opponent_goal_angle_bin > 0);
        assert!(state.opposing_goalkeeper_distance_bin > 0);
        assert!(state.perceived_pressure_bin > 0);
    }

    #[test]
    fn shot_decision_gate_requires_quality_or_near_goal_pressure_bailout() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1414,
            ..Default::default()
        });
        let attacker = 9;
        sim.players[attacker].position = Vec2::new(40.0, 104.0);
        sim.ball.holder = Some(attacker);
        sim.ball.position = sim.players[attacker].position;
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        let mut observation = snapshot.observation_for(attacker);
        observation.shot_lane_open = true;
        observation.yards_to_goal = 9.0;
        observation.immediate_dispossession_risk = 0.20;
        observation.shot_on_frame_probability = SHOT_ON_FRAME_MIN_PROBABILITY - 0.01;
        observation.shot_beat_goalkeeper_probability = SHOT_KEEPER_BEAT_MIN_PROBABILITY + 0.20;
        assert!(!shot_decision_is_qualified(&observation));

        observation.shot_on_frame_probability = SHOT_ON_FRAME_MIN_PROBABILITY + 0.01;
        observation.shot_beat_goalkeeper_probability = SHOT_KEEPER_BEAT_MIN_PROBABILITY - 0.01;
        assert!(!shot_decision_is_qualified(&observation));

        observation.shot_beat_goalkeeper_probability = SHOT_KEEPER_BEAT_MIN_PROBABILITY + 0.01;
        assert!(shot_decision_is_qualified(&observation));

        observation.shot_on_frame_probability = SHOT_BAILOUT_ON_FRAME_PROBABILITY + 0.01;
        observation.shot_beat_goalkeeper_probability = 0.02;
        observation.immediate_dispossession_risk = SHOT_BAILOUT_DISPOSSESSION_RISK + 0.01;
        assert!(shot_decision_is_qualified(&observation));

        observation.yards_to_goal = SHOT_BAILOUT_NEAR_GOAL_YARDS + 0.5;
        assert!(!shot_decision_is_qualified(&observation));
    }

    #[test]
    fn pomdp_observation_exposes_calibrated_shot_probabilities() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1415,
            ..Default::default()
        });
        let attacker = 9;
        sim.players[attacker].position = Vec2::new(40.0, 103.0);
        sim.players[attacker].skills.shooting = 9.0;
        sim.players[attacker].skills.right_foot_shot_power = 9.0;
        sim.players[attacker].skills.left_foot_shot_power = 8.0;
        for away in 11..22 {
            sim.players[away].position = Vec2::new(74.0, 22.0 + away as f64);
        }
        sim.players[11].position = Vec2::new(38.0, 116.0);
        sim.players[11].skills.goalkeeping = 5.4;
        sim.ball.holder = Some(attacker);
        sim.ball.position = sim.players[attacker].position;
        sim.ball.last_touch_team = Some(Team::Home);

        let good_snapshot = WorldSnapshot::from_match(&sim);
        let good = good_snapshot.observation_for(attacker);
        assert!(good.shot_lane_open);
        assert!(
            good.shot_on_frame_probability >= SHOT_ON_FRAME_MIN_PROBABILITY,
            "good shot on-frame probability: {}",
            good.shot_on_frame_probability
        );
        assert!(
            good.shot_beat_goalkeeper_probability >= SHOT_KEEPER_BEAT_MIN_PROBABILITY,
            "good shot keeper-beat probability: {}",
            good.shot_beat_goalkeeper_probability
        );
        assert!(shot_decision_is_qualified(&good));

        sim.players[attacker].position = Vec2::new(8.0, 82.0);
        sim.ball.position = sim.players[attacker].position;
        let poor_snapshot = WorldSnapshot::from_match(&sim);
        let poor = poor_snapshot.observation_for(attacker);
        assert!(poor.shot_on_frame_probability < good.shot_on_frame_probability);
        assert!(!shot_decision_is_qualified(&poor));
    }

    #[test]
    fn shot_creation_space_improves_finishing_window_score() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1416,
            ..Default::default()
        });
        let attacker = 9;
        sim.players[attacker].position = Vec2::new(18.0, 88.0);
        sim.players[attacker].home_position = Vec2::new(58.0, 92.0);
        sim.players[attacker].skills.shooting = 8.6;
        for away in 11..22 {
            sim.players[away].position = Vec2::new(70.0, 20.0 + away as f64);
        }
        sim.players[11].position = Vec2::new(40.0, 116.0);
        sim.ball.holder = Some(attacker);
        sim.ball.position = sim.players[attacker].position;
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        let player = snapshot
            .players
            .iter()
            .find(|player| player.id == attacker)
            .expect("attacker snapshot");
        let start = snapshot
            .player_position(attacker)
            .expect("attacker position");
        let target =
            snapshot.shot_creation_space_for(attacker, sim.players[attacker].home_position);
        assert!(target.y > start.y);
        assert!(
            (target.x - snapshot.field_width * 0.5).abs()
                < (start.x - snapshot.field_width * 0.5).abs()
        );
        assert!(
            snapshot.shooting_window_score_at(player, target)
                > snapshot.shooting_window_score_at(player, start)
        );
    }

    #[test]
    fn pomdp_observation_tracks_per_player_position_confidence() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1412,
            ..Default::default()
        });
        let observer = 6;
        let front_teammate = 7;
        let behind_teammate = 8;
        sim.players[observer].position = Vec2::new(40.0, 60.0);
        sim.players[observer].velocity = Vec2::new(0.0, 4.0);
        sim.players[front_teammate].position = Vec2::new(40.0, 80.0);
        sim.players[behind_teammate].position = Vec2::new(40.0, 40.0);

        let snapshot = WorldSnapshot::from_match(&sim);
        let observation = snapshot.observation_for(observer);
        let front = observation
            .player_position_confidences
            .iter()
            .find(|entry| entry.player_id == front_teammate)
            .expect("front confidence entry");
        let behind = observation
            .player_position_confidences
            .iter()
            .find(|entry| entry.player_id == behind_teammate)
            .expect("behind confidence entry");

        assert_eq!(observation.player_position_confidences.len(), 21);
        assert!(front.in_front);
        assert!(!behind.in_front);
        assert!(front.confidence > behind.confidence * 1.75);
        assert!(observation.teammate_position_confidence > 0.0);
    }

    #[test]
    fn pomdp_observation_uses_tired_nearest_defender_as_dribble_cue() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1413,
            ..Default::default()
        });
        let attacker = 9;
        let defender = 13;
        sim.players[attacker].position = Vec2::new(40.0, 88.0);
        sim.players[attacker].velocity = Vec2::new(0.0, 4.0);
        sim.players[attacker].fatigue = 0.05;
        sim.players[attacker].skills.dribbling = 9.0;
        sim.players[attacker].skills.stamina = 9.0;
        sim.players[defender].position = Vec2::new(40.5, 92.0);
        sim.players[defender].fatigue = 0.90;
        for away in 11..22 {
            if away != defender {
                sim.players[away].position = Vec2::new(74.0, 22.0 + away as f64);
                sim.players[away].fatigue = 0.05;
            }
        }
        sim.ball.holder = Some(attacker);
        sim.ball.position = sim.players[attacker].position;
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        let observation = snapshot.observation_for(attacker);
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(attacker),
            &observation,
            Team::Home,
            sim.players[attacker].role,
        );

        assert!(observation.nearest_defender_fatigue > 0.85);
        assert!(observation.nearest_defender_fatigue_confidence > 0.70);
        assert!(observation.perceived_nearest_defender_fatigue > 0.70);
        assert!(observation.perceived_fatigue_advantage > 0.60);
        assert!(state.perceived_fatigue_advantage_bin >= 3);

        let directive = snapshot.tactical_directive(Team::Home);
        let favorable =
            sim.players[attacker].possession_action_options(&observation, &directive, 0, 0);
        let mut unfavorable_observation = observation.clone();
        unfavorable_observation.perceived_nearest_defender_fatigue = 0.05;
        unfavorable_observation.perceived_fatigue_advantage = unfavorable_observation
            .perceived_nearest_defender_fatigue
            - unfavorable_observation.fatigue;
        let unfavorable = sim.players[attacker].possession_action_options(
            &unfavorable_observation,
            &directive,
            0,
            0,
        );

        assert!(
            action_option_score(&favorable, "dribble")
                > action_option_score(&unfavorable, "dribble") * 1.12
        );
    }

    #[test]
    fn aerial_pass_targeting_can_bypass_blocked_floor_lane() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1771,
            ..Default::default()
        });
        let passer = 7;
        let target = 9;
        let blocker = 12;
        sim.players[passer].position = Vec2::new(10.0, 86.0);
        sim.players[target].position = Vec2::new(40.0, 104.0);
        sim.players[target].skills.height = 9.0;
        sim.players[target].skills.strength = 8.5;
        sim.players[blocker].position = Vec2::new(25.0, 95.0);
        for away in 11..22 {
            if away != blocker {
                sim.players[away].position = Vec2::new(72.0, 24.0 + away as f64);
            }
        }
        sim.players[11].position = Vec2::new(58.0, 112.0);
        sim.players[13].position = Vec2::new(72.0, 108.0);
        sim.players[14].position = Vec2::new(43.0, 100.0);
        sim.ball.holder = Some(passer);
        sim.ball.position = sim.players[passer].position;
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        let observation = snapshot.observation_for(passer);
        let state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(passer),
            &observation,
            Team::Home,
            sim.players[passer].role,
        );
        assert!(!snapshot.clear_line(
            sim.players[passer].position,
            sim.players[target].position,
            Team::Away,
            2.5
        ));
        assert!(!snapshot
            .ranked_visible_pass_targets(passer, 11)
            .contains(&target));
        assert!(snapshot
            .ranked_visible_aerial_pass_targets(passer, 11)
            .contains(&target));
        assert!(observation.visible_aerial_pass_options > 0);
        assert!(observation.aerial_pass_bypass_score > 0.25);
        assert!(observation.aerial_pass_interception_risk > 0.15);
        assert!(state.visible_aerial_pass_options_bin > 0);
        assert!(state.aerial_pass_bypass_score_bin > 0);
        assert!(state.aerial_pass_interception_risk_bin > 0);

        sim.apply_player_intent(PlayerIntent {
            player_id: passer,
            action: SoccerAction::Pass {
                target_player: Some(target),
                power: 0.85,
                flight: PassFlight::Aerial,
            },
            sprint: false,
        });

        let pending = sim.pending_pass.as_ref().expect("pending aerial pass");
        assert_eq!(pending.flight, PassFlight::Aerial);
        assert!(pending.is_cross);
        assert_eq!(pending.target, Some(target));
        sim.integrate_ball();
        assert!(sim.ball.to_state().altitude_yards > 0.05);
    }

    #[test]
    fn aerial_pass_interception_pressure_doubles_or_triples_near_landing() {
        let floor = PendingPass {
            team: Team::Home,
            from: 7,
            target: Some(9),
            flight: PassFlight::Floor,
            is_cross: false,
            origin: Vec2::new(10.0, 80.0),
            intended_target: Vec2::new(40.0, 104.0),
            distance_yards: 38.0,
            offside: None,
        };
        let aerial = PendingPass {
            flight: PassFlight::Aerial,
            is_cross: true,
            ..floor.clone()
        };

        assert_eq!(
            aerial_interception_multiplier(&floor, Vec2::new(35.0, 100.0)),
            1.0
        );
        assert!(aerial_interception_multiplier(&aerial, Vec2::new(25.0, 92.0)) >= 2.0);
        assert!(aerial_interception_multiplier(&aerial, Vec2::new(39.0, 103.0)) >= 2.5);
    }

    #[test]
    fn aerial_cross_reception_exposes_first_touch_header_and_control_choices() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1772,
            ..Default::default()
        });
        let passer = 7;
        let receiver = 9;
        sim.players[passer].position = Vec2::new(9.0, 94.0);
        sim.players[receiver].position = Vec2::new(40.0, 105.0);
        sim.players[receiver].skills.height = 9.5;
        sim.players[receiver].skills.strength = 8.8;
        sim.players[receiver].skills.first_touch = 8.4;
        for away in 11..22 {
            sim.players[away].position = Vec2::new(72.0, 20.0 + away as f64);
        }
        sim.ball.position = sim.players[receiver].position;
        sim.ball.velocity = Vec2::new(5.0, 2.0);
        sim.pending_pass = Some(PendingPass {
            team: Team::Home,
            from: passer,
            target: Some(receiver),
            flight: PassFlight::Aerial,
            is_cross: true,
            origin: sim.players[passer].position,
            intended_target: sim.players[receiver].position,
            distance_yards: sim.players[passer]
                .position
                .distance(sim.players[receiver].position),
            offside: None,
        });

        sim.apply_ball_outcome(BallStepOutcome::Controlled {
            holder: receiver,
            holder_team: Team::Home,
            possession_result: BallPossessionResult::PassCompleted(Team::Home),
        });

        let snapshot = WorldSnapshot::from_match(&sim);
        let observation = snapshot.observation_for(receiver);
        assert!(observation.first_touch_available);
        assert_eq!(
            observation.incoming_ball_kind,
            IncomingBallKind::AerialCross
        );
        assert!(observation.first_time_shot_score > 0.0);
        assert!(observation.control_touch_score > 0.0);
        let options = sim.players[receiver].first_touch_action_options(&observation, 1);
        assert!(options.iter().any(|option| option.label == "header"));
        assert!(options.iter().any(|option| option.label == "chest-control"));

        sim.apply_player_intent(PlayerIntent {
            player_id: receiver,
            action: SoccerAction::ControlTouch {
                target: sim.players[receiver].position,
            },
            sprint: false,
        });
        assert!(sim.players[receiver].incoming_ball.is_none());
        assert_eq!(sim.ball.holder, Some(receiver));
    }

    #[test]
    fn q_policy_keys_separate_same_action_by_player_grid_cell() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 142,
            ..Default::default()
        });
        sim.ball.holder = Some(5);
        sim.players[5].position = Vec2::new(18.0, 68.0);
        let snapshot_left = WorldSnapshot::from_match(&sim);

        let mut policy = SoccerQPolicy::default();
        assert!(policy.set_action_value_for_snapshot(&snapshot_left, 5, "dribble", 4.0));
        let left_state = SoccerQStateKey::from_parts(
            &snapshot_left.mdp_state_for_player(5),
            &snapshot_left.observation_for(5),
            Team::Home,
            sim.players[5].role,
        );
        assert_eq!(
            policy
                .best_action_for_snapshot(&snapshot_left, 5)
                .as_deref(),
            Some("dribble")
        );

        sim.players[5].position = Vec2::new(62.0, 68.0);
        let snapshot_right = WorldSnapshot::from_match(&sim);
        let right_state = SoccerQStateKey::from_parts(
            &snapshot_right.mdp_state_for_player(5),
            &snapshot_right.observation_for(5),
            Team::Home,
            sim.players[5].role,
        );

        assert_ne!(
            left_state.player_fine_cell_id,
            right_state.player_fine_cell_id
        );
        assert_ne!(
            left_state.player_tactical_cell_id,
            right_state.player_tactical_cell_id
        );
        assert!(policy.q_value(&right_state, "dribble").is_none());
    }

    #[test]
    fn q_policy_uses_parent_grid_backoff_for_spatial_correlation() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1421,
            ..Default::default()
        });
        sim.ball.holder = Some(5);
        sim.ball.position = Vec2::new(21.0, 68.0);
        sim.players[5].position = Vec2::new(18.0, 68.0);
        let snapshot = WorldSnapshot::from_match(&sim);
        let mut base_state = SoccerQStateKey::from_parts(
            &snapshot.mdp_state_for_player(5),
            &snapshot.observation_for(5),
            Team::Home,
            sim.players[5].role,
        );

        let mut policy = SoccerQPolicy::default();
        policy.set_action_value(base_state.clone(), "dribble", 4.0);
        assert_eq!(
            policy.best_action_hierarchical(&base_state).as_deref(),
            Some("dribble")
        );

        let mut tactical_sibling = base_state.clone();
        tactical_sibling.player_fine_cell_id += 1;
        assert!(policy.q_value(&tactical_sibling, "dribble").is_none());
        assert_eq!(
            policy
                .best_action_hierarchical(&tactical_sibling)
                .as_deref(),
            Some("dribble")
        );

        let mut macro_sibling = base_state.clone();
        macro_sibling.player_fine_cell_id += 7;
        macro_sibling.player_tactical_cell_id += 1;
        assert!(policy.q_value(&macro_sibling, "dribble").is_none());
        assert_eq!(
            policy.best_action_hierarchical(&macro_sibling).as_deref(),
            Some("dribble")
        );

        let mut whole_pitch_sibling = base_state.clone();
        whole_pitch_sibling.player_fine_cell_id += 80;
        whole_pitch_sibling.player_tactical_cell_id += 20;
        whole_pitch_sibling.player_macro_cell_id += 2;
        assert!(policy.q_value(&whole_pitch_sibling, "dribble").is_none());
        assert_eq!(
            policy
                .best_action_hierarchical(&whole_pitch_sibling)
                .as_deref(),
            Some("dribble")
        );

        base_state.action_facing = FacingBucket::North;
        assert_eq!(policy.best_action_hierarchical(&base_state), None);
    }

    #[test]
    fn player_receive_facing_is_recorded_when_possession_changes() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 143,
            ..Default::default()
        });
        sim.players[8].velocity = Vec2::new(-3.0, 0.0);
        sim.apply_ball_outcome(BallStepOutcome::Controlled {
            holder: 8,
            holder_team: Team::Home,
            possession_result: BallPossessionResult::LooseBallRecovery(Team::Home),
        });

        assert_eq!(sim.ball.holder, Some(8));
        assert_eq!(sim.players[8].receive_facing, FacingBucket::West);
        assert_eq!(sim.players[8].action_facing, FacingBucket::West);
    }

    #[test]
    fn learned_policy_biases_agent_decision_when_legal() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 15,
            ..Default::default()
        });
        let before = WorldSnapshot::from_match(&sim);
        let mut rng = mulberry32(1501);
        sim.central_brain.run_time_step(&before, &mut rng);
        let snapshot = WorldSnapshot::from_match(&sim);
        assert_eq!(snapshot.ball.holder, Some(5));
        assert!(snapshot.best_pass_target(5).is_some());

        let mut policy = SoccerQPolicy::default();
        assert!(policy.set_action_value_for_snapshot(&snapshot, 5, "pass", 5.0));
        sim.set_learned_policy(policy);
        sim.run_time_step();

        let decision = sim.players[5]
            .last_decision
            .as_ref()
            .expect("player decision");
        assert_eq!(decision.action, "pass");
        assert_eq!(decision.operation_order[0], "learned-policy");
        assert!(sim.stats.passes_attempted_home > 0);
        assert!(!sim
            .learned_policy()
            .expect("online policy")
            .q_values
            .is_empty());
    }

    #[test]
    fn learned_policy_uses_target_grid_preference_for_pass_target() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1505,
            ..Default::default()
        });
        let passer = 5;
        let candidate_a = 6;
        let candidate_b = 8;
        sim.players[passer].position = Vec2::new(40.0, 60.0);
        sim.players[passer].velocity = Vec2::new(0.0, 2.0);
        sim.players[candidate_a].position = Vec2::new(40.0, 78.0);
        sim.players[candidate_b].position = Vec2::new(62.0, 66.0);
        for id in 0..11 {
            if ![passer, candidate_a, candidate_b].contains(&id) {
                sim.players[id].position = Vec2::new(8.0 + id as f64, 48.0);
            }
        }
        for away in 11..22 {
            sim.players[away].position = Vec2::new(72.0, 108.0);
        }
        sim.players[11].position = Vec2::new(40.0, 118.0);
        sim.players[12].position = Vec2::new(45.0, 104.0);
        sim.ball.holder = Some(passer);
        sim.ball.position = sim.players[passer].position;
        sim.ball.velocity = Vec2::zero();
        sim.ball.last_touch_team = Some(Team::Home);
        sim.pending_pass = None;

        let before = WorldSnapshot::from_match(&sim);
        let mut rng = mulberry32(1505);
        sim.central_brain.run_time_step(&before, &mut rng);
        let snapshot = WorldSnapshot::from_match(&sim);
        let visible_targets = snapshot.ranked_visible_pass_targets(passer, 11);
        let heuristic_target = visible_targets
            .first()
            .copied()
            .expect("heuristic pass target");
        let learned_target = [candidate_a, candidate_b]
            .into_iter()
            .find(|target| *target != heuristic_target && visible_targets.contains(target))
            .expect("alternate visible pass target");

        let mut policy = SoccerQPolicy::default();
        assert!(policy.set_action_value_for_snapshot(&snapshot, passer, "pass", 5.0));
        assert!(policy.set_target_value_for_snapshot(
            &snapshot,
            passer,
            "pass",
            sim.players[learned_target].position,
            5.0,
        ));
        sim.set_learned_policy(policy);
        sim.run_time_step();

        let pass = sim.pending_pass.as_ref().expect("learned targeted pass");
        assert_eq!(pass.from, passer);
        assert_eq!(pass.target, Some(learned_target));
        let decision = sim.players[passer]
            .last_decision
            .as_ref()
            .expect("passer decision");
        assert_eq!(decision.action, "pass");
        assert_eq!(decision.operation_order[0], "learned-policy");
        assert_eq!(
            decision
                .action_target
                .as_ref()
                .and_then(|target| target.player_id),
            Some(learned_target)
        );
    }

    #[test]
    fn learning_runtime_can_train_without_retaining_transition_logs() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            learning_logging_enabled: false,
            seed: 1506,
            ..Default::default()
        })
        .with_team_policies(SoccerTeamQPolicies::new(SoccerQPolicyOptions::default()));

        sim.run_time_step();

        assert!(sim.learning_transitions.is_empty());
        let policies = sim.team_policies().expect("team policies");
        assert!(policies.total_entries() > 0);
        assert!(policies.home.target_values.len() + policies.away.target_values.len() > 0);
        let learning = sim.learning_snapshot();
        assert_eq!(learning.total_transitions, 0);
        assert!(learning.team_policies_enabled);
        assert!(!learning.learning_logging_enabled);
        assert!(learning.home_policy_entries + learning.away_policy_entries > 0);
    }

    #[test]
    fn learning_runtime_can_pause_learned_decisions_and_logging() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            learning_enabled: false,
            learning_logging_enabled: false,
            seed: 1507,
            ..Default::default()
        });
        let snapshot = WorldSnapshot::from_match(&sim);
        let mut policy = SoccerQPolicy::default();
        assert!(policy.set_action_value_for_snapshot(&snapshot, 5, "pass", 5.0));
        let seeded_visits = policy.visit_count();
        sim.set_learned_policy(policy);

        sim.run_time_step();

        let decision = sim.players[5]
            .last_decision
            .as_ref()
            .expect("holder decision");
        assert_ne!(
            decision.operation_order.first().map(String::as_str),
            Some("learned-policy")
        );
        assert!(sim.learning_transitions.is_empty());
        let policy = sim.learned_policy().expect("learned policy");
        assert_eq!(policy.visit_count(), seeded_visits);
        assert!(!sim.learning_snapshot().learning_enabled);
    }

    #[test]
    fn dense_reward_discourages_still_holder_and_rewards_forward_progression() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1508,
            ..Default::default()
        });
        let holder = 5;
        sim.ball.holder = Some(holder);
        sim.ball.position = sim.players[holder].position;
        sim.ball.velocity = Vec2::zero();
        let before = WorldSnapshot::from_match(&sim);

        let hold_decision = test_decision_trace(&before, holder, "hold");
        let still_reward = soccer_transition_reward(
            &sim.players[holder],
            &hold_decision,
            &before,
            &before,
            0,
            0,
            0,
            0,
            false,
        );

        let mut after = before.clone();
        after.ball.position.y += 6.0;
        if let Some(player) = after.players.iter_mut().find(|player| player.id == holder) {
            player.position.y += 2.0;
        }
        let dribble_decision = test_decision_trace(&before, holder, "dribble");
        let progress_reward = soccer_transition_reward(
            &sim.players[holder],
            &dribble_decision,
            &before,
            &after,
            0,
            0,
            0,
            0,
            false,
        );

        assert!(still_reward < -0.4, "still reward: {still_reward}");
        assert!(
            progress_reward > 0.6,
            "progression reward: {progress_reward}"
        );
    }

    #[test]
    fn loose_ball_fifty_fifty_duel_is_labeled_and_rewarded() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1509,
            ..Default::default()
        });
        let home = 5;
        let away = 11;
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(40.0, 60.0);
        sim.ball.velocity = Vec2::zero();
        for player in &mut sim.players {
            player.position = match player.team {
                Team::Home => Vec2::new(10.0, 20.0 + player.id as f64),
                Team::Away => Vec2::new(70.0, 80.0 + player.id as f64),
            };
            player.velocity = Vec2::zero();
        }
        sim.players[home].position = Vec2::new(38.0, 60.0);
        sim.players[home].velocity = Vec2::new(1.0, 0.0);
        sim.players[away].position = Vec2::new(42.0, 60.0);
        sim.players[away].velocity = Vec2::new(-1.0, 0.0);
        let before = WorldSnapshot::from_match(&sim);

        assert_eq!(loose_ball_fifty_fifty_duel(&before), Some((home, away)));

        let mut chaser = sim.players[home].clone();
        let mut rng = mulberry32(1509);
        let intent = chaser.run_time_step(&before, None, None, &mut rng);
        match intent.action {
            SoccerAction::MoveTo(target) => assert!(target.distance(sim.ball.position) < 0.01),
            other => panic!("expected 50:50 recovery move, got {other:?}"),
        }
        let decision = chaser.last_decision.as_ref().expect("50:50 decision");
        assert_eq!(decision.action, "recover");
        assert_eq!(decision.operation_order[0], "fifty-fifty-duel");
        assert_eq!(decision.action_options[0].label, "fifty-fifty-duel");

        sim.players[home].position = Vec2::new(39.5, 60.0);
        let after = WorldSnapshot::from_match(&sim);
        let reward = soccer_transition_reward(
            &sim.players[home],
            &test_decision_trace(&before, home, "recover"),
            &before,
            &after,
            0,
            0,
            0,
            0,
            false,
        );
        assert!(reward > 0.8, "50:50 contest reward: {reward}");
    }

    #[test]
    fn defensive_reward_prefers_goal_side_of_ball_and_attacker() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1510,
            ..Default::default()
        });
        let defender = 1;
        let attacker = 11;
        sim.ball.holder = Some(attacker);
        sim.players[attacker].position = Vec2::new(40.0, 48.0);
        sim.ball.position = sim.players[attacker].position;
        sim.ball.velocity = Vec2::zero();

        sim.players[defender].position = Vec2::new(40.0, 42.0);
        let goal_side = WorldSnapshot::from_match(&sim);
        let goal_side_reward = soccer_transition_reward(
            &sim.players[defender],
            &test_decision_trace(&goal_side, defender, "defend"),
            &goal_side,
            &goal_side,
            0,
            0,
            0,
            0,
            false,
        );

        sim.players[defender].position = Vec2::new(40.0, 54.0);
        let wrong_side = WorldSnapshot::from_match(&sim);
        let wrong_side_reward = soccer_transition_reward(
            &sim.players[defender],
            &test_decision_trace(&wrong_side, defender, "defend"),
            &wrong_side,
            &wrong_side,
            0,
            0,
            0,
            0,
            false,
        );

        assert!(
            goal_side_reward > wrong_side_reward + 0.45,
            "goal-side {goal_side_reward}, wrong-side {wrong_side_reward}"
        );
    }

    #[test]
    fn learned_policy_cannot_make_hold_the_runtime_optimum() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 1512,
            ..Default::default()
        });
        sim.ball.holder = Some(5);
        sim.ball.position = sim.players[5].position;
        let before = WorldSnapshot::from_match(&sim);
        let mut rng = mulberry32(1512);
        sim.central_brain.run_time_step(&before, &mut rng);
        let snapshot = WorldSnapshot::from_match(&sim);
        let mut policy = SoccerQPolicy::default();
        assert!(policy.set_action_value_for_snapshot(&snapshot, 5, "hold", 50.0));
        sim.set_learned_policy(policy);

        sim.run_time_step();

        let decision = sim.players[5].last_decision.as_ref().expect("decision");
        assert_ne!(
            decision.operation_order.first().map(String::as_str),
            Some("learned-policy")
        );
        assert_ne!(decision.action, "hold");
    }

    #[test]
    fn team_q_policies_train_each_side_separately() {
        let dataset = run_learning_episode(MatchConfig {
            duration_seconds: 0.2,
            seed: 151,
            ..Default::default()
        });
        let mut policies = SoccerTeamQPolicies::new(SoccerQPolicyOptions::default());
        policies.train(&dataset.transitions);

        assert!(!policies.home.q_values.is_empty());
        assert!(!policies.away.q_values.is_empty());
        assert_eq!(
            policies.total_entries(),
            policies.home.q_values.len() + policies.away.q_values.len()
        );
    }

    #[test]
    fn adversarial_team_q_policies_train_against_opponent_rewards() {
        let dataset = run_learning_episode(MatchConfig {
            duration_seconds: 0.1,
            seed: 1511,
            ..Default::default()
        });
        let mut home_transition = dataset
            .transitions
            .iter()
            .find(|t| t.tick == 0 && t.team == Team::Home)
            .expect("home transition")
            .clone();
        let mut away_transition = dataset
            .transitions
            .iter()
            .find(|t| t.tick == 0 && t.team == Team::Away)
            .expect("away transition")
            .clone();
        home_transition.reward = 1.0;
        home_transition.done = true;
        away_transition.reward = 0.25;
        away_transition.done = true;

        let options = SoccerQPolicyOptions {
            alpha: 0.5,
            gamma: 0.9,
        };
        let mut policies = SoccerTeamQPolicies::new(options);
        policies.train_adversarial(&[home_transition.clone(), away_transition.clone()]);

        let home_state = SoccerQStateKey::from_transition(&home_transition);
        let away_state = SoccerQStateKey::from_transition(&away_transition);
        let home_value = policies
            .home
            .q_value(&home_state, &home_transition.action)
            .expect("home q value");
        let away_value = policies
            .away
            .q_value(&away_state, &away_transition.action)
            .expect("away q value");

        assert!((home_value - 0.375).abs() < 1e-9);
        assert!((away_value + 0.375).abs() < 1e-9);
    }

    #[test]
    fn team_learned_policy_biases_matching_team_decision() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 152,
            ..Default::default()
        });
        let before = WorldSnapshot::from_match(&sim);
        let mut rng = mulberry32(1502);
        sim.central_brain.run_time_step(&before, &mut rng);
        let snapshot = WorldSnapshot::from_match(&sim);
        assert_eq!(snapshot.ball.holder, Some(5));
        assert!(snapshot.best_pass_target(5).is_some());

        let mut policies = SoccerTeamQPolicies::new(SoccerQPolicyOptions::default());
        assert!(policies
            .policy_mut(Team::Home)
            .set_action_value_for_snapshot(&snapshot, 5, "pass", 5.0));
        sim.set_team_policies(policies);
        sim.run_time_step();

        let decision = sim.players[5]
            .last_decision
            .as_ref()
            .expect("player decision");
        assert_eq!(decision.action, "pass");
        assert_eq!(decision.operation_order[0], "learned-policy");
        let team_policies = sim.team_policies().expect("online team policies");
        assert!(!team_policies.home.q_values.is_empty());
        assert!(team_policies.away.q_values.len() > 0);
    }

    #[test]
    fn self_play_training_returns_home_and_away_policy_entries() {
        let artifact = train_soccer_team_policies_from_self_play(
            MatchConfig {
                duration_seconds: 0.2,
                seed: 153,
                ..Default::default()
            },
            2,
            SoccerQPolicyOptions::default(),
        );

        assert_eq!(artifact.episodes.len(), 2);
        assert_eq!(artifact.episodes[0].transitions, 44);
        assert_eq!(artifact.episodes[1].transitions, 44);
        assert!(!artifact.home_entries.is_empty());
        assert!(!artifact.away_entries.is_empty());
        assert!(!artifact.home_target_entries.is_empty());
        assert!(!artifact.away_target_entries.is_empty());
        assert!(
            artifact.episodes[1].home_policy_entries >= artifact.episodes[0].home_policy_entries
        );
        assert!(
            artifact.episodes[1].away_policy_entries >= artifact.episodes[0].away_policy_entries
        );
        assert!(
            artifact.episodes[1].home_policy_target_entries
                >= artifact.episodes[0].home_policy_target_entries
        );
        assert!(
            artifact.episodes[1].away_policy_target_entries
                >= artifact.episodes[0].away_policy_target_entries
        );
    }

    #[test]
    fn configured_periods_reset_between_halves_and_alternate_kickoff() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 1.0,
            duration_seconds: 2.0,
            period_count: 2,
            period_break_recovery_seconds: 10.0,
            seed: 1532,
            ..Default::default()
        });
        sim.players[0].fatigue = 0.5;

        sim.run_time_step();

        let holder_id = sim.ball.holder.expect("second period kickoff holder");
        let holder = sim
            .players
            .iter()
            .find(|player| player.id == holder_id)
            .expect("holder player");
        assert_eq!(sim.tick, 1);
        assert_eq!(holder.team, Team::Away);
        assert!(sim.players[0].fatigue < 0.5);
        assert!(sim.events.iter().any(|event| event.kind == "period-break"));
        assert!(sim
            .events
            .iter()
            .any(|event| { event.kind == "period-start" && event.team == Some(Team::Away) }));

        sim.run_time_step();
        assert!(sim.is_done());
    }

    #[test]
    fn default_tactical_learning_rewards_flanks_and_defensive_contraction() {
        let weights = SoccerTacticalLearningWeights::default();

        assert!(weights.attack_width_delta_weight >= 0.50);
        assert!(weights.attack_flank_lane_weight >= 0.25);
        assert!(weights.defense_contract_delta_weight >= 0.40);
        assert!(weights.defense_compactness_score_weight >= 0.12);
    }

    #[test]
    fn self_play_training_artifact_persists_tactical_learning_weights() {
        let tactical_learning = SoccerTacticalLearningWeights {
            attack_flank_lane_weight: 0.31,
            defense_contract_delta_weight: 0.42,
            ..Default::default()
        };
        let artifact = train_soccer_team_policies_from_self_play(
            MatchConfig {
                duration_seconds: 0.2,
                seed: 1531,
                tactical_learning: tactical_learning.clone(),
                ..Default::default()
            },
            1,
            SoccerQPolicyOptions::default(),
        );
        let value = serde_json::to_value(&artifact).expect("self-play artifact json");

        assert_eq!(artifact.tactical_learning.attack_flank_lane_weight, 0.31);
        assert_eq!(
            value["tacticalLearning"]["attackFlankLaneWeight"],
            serde_json::json!(0.31)
        );
        assert_eq!(
            value["config"]["tacticalLearning"]["defenseContractDeltaWeight"],
            serde_json::json!(0.42)
        );
    }

    #[test]
    fn goal_reward_divides_pool_across_recent_attacking_chain() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 154,
            ..Default::default()
        });
        sim.tick = 33;
        sim.possession_chain.clear();
        sim.possession_chain.push_back(5);
        sim.possession_chain.push_back(7);
        sim.possession_chain.push_back(12);

        sim.record_goal_rewards(Team::Home, Some(9));

        let reward_for = |player_id| {
            sim.reward_events
                .iter()
                .filter(|event| event.player_id == player_id)
                .map(|event| event.amount)
                .sum::<f64>()
        };
        let attacking_total = [5, 7, 9]
            .into_iter()
            .map(|player_id| reward_for(player_id))
            .sum::<f64>();
        assert!((attacking_total - GOAL_REWARD_POINTS).abs() < 1e-9);
        assert!(reward_for(9) > reward_for(7));
        assert!(reward_for(7) > reward_for(5));
        assert!(reward_for(5) > 0.0);
        assert!(reward_for(12) <= 0.0);
        assert!(sim
            .reward_events
            .iter()
            .any(|event| { event.tick == 33 && event.player_id == 9 && event.amount > 70.0 }));
    }

    #[test]
    fn shot_on_target_reward_divides_fifty_point_pool() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            duration_seconds: 0.1,
            seed: 155,
            ..Default::default()
        });
        sim.tick = 37;
        sim.possession_chain.clear();
        sim.possession_chain.push_back(5);
        sim.possession_chain.push_back(7);

        sim.record_shot_on_target_rewards(Team::Home, 9);

        let reward_for = |player_id| {
            sim.reward_events
                .iter()
                .filter(|event| event.player_id == player_id)
                .map(|event| event.amount)
                .sum::<f64>()
        };
        let attacking_total = [5, 7, 9]
            .into_iter()
            .map(|player_id| reward_for(player_id))
            .sum::<f64>();
        assert!((attacking_total - SHOT_ON_TARGET_REWARD_POINTS).abs() < 1e-9);
        assert!(reward_for(9) > reward_for(7));
        assert!(reward_for(7) > reward_for(5));
        assert!(reward_for(5) > 0.0);
    }

    #[test]
    fn defensive_delay_reward_flows_into_transition_tick() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 0.2,
            duration_seconds: 1.0,
            seed: 155,
            ..Default::default()
        });
        let attacker = 9;
        let defender = 12;
        sim.tick = 40;
        sim.players[attacker].position = Vec2::new(40.0, 80.0);
        sim.players[defender].position = Vec2::new(40.8, 80.5);
        sim.ball.holder = Some(attacker);
        sim.ball.position = sim.players[attacker].position;
        sim.ball.last_touch_team = Some(Team::Home);
        let before = WorldSnapshot::from_match(&sim);
        sim.players[defender].last_decision =
            Some(test_decision_trace(&before, defender, "defend"));
        sim.defensive_delay_clocks.insert(defender, 1.9);

        sim.players[attacker].position.y += 0.2;
        sim.ball.position = sim.players[attacker].position;
        sim.tick = 41;
        let after = WorldSnapshot::from_match(&sim);
        let event_start = sim.reward_events.len();

        sim.update_defensive_reward_trackers(&before, &after);

        let event = sim.reward_events[event_start..]
            .iter()
            .find(|event| event.player_id == defender)
            .expect("delay reward event");
        assert_eq!(event.tick, before.tick);
        assert_eq!(event.amount, 2.0);
        let transitions = sim.learning_transitions_for(
            &before,
            &after,
            sim.score_home,
            sim.score_away,
            &sim.reward_events[event_start..],
        );
        let defender_transition = transitions
            .iter()
            .find(|transition| transition.player_id == defender)
            .expect("defender learning transition");
        assert!(
            defender_transition.reward > 2.0,
            "dense shaped delay reward: {}",
            defender_transition.reward
        );
    }

    #[test]
    fn beaten_defender_penalty_flows_into_transition_tick() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 0.1,
            duration_seconds: 1.0,
            seed: 156,
            ..Default::default()
        });
        let attacker = 9;
        let defender = 12;
        sim.tick = 50;
        sim.players[attacker].position = Vec2::new(40.0, 80.0);
        sim.players[defender].position = Vec2::new(40.4, 80.0);
        sim.ball.holder = Some(attacker);
        sim.ball.position = sim.players[attacker].position;
        sim.ball.last_touch_team = Some(Team::Home);
        let before = WorldSnapshot::from_match(&sim);
        sim.players[defender].last_decision =
            Some(test_decision_trace(&before, defender, "defend"));
        sim.defensive_beat_clocks.insert(defender, 0.20);

        sim.players[attacker].position.y += 1.2;
        sim.ball.position = sim.players[attacker].position;
        sim.tick = 51;
        let after = WorldSnapshot::from_match(&sim);
        let event_start = sim.reward_events.len();

        sim.update_defensive_reward_trackers(&before, &after);

        let event = sim.reward_events[event_start..]
            .iter()
            .find(|event| event.player_id == defender)
            .expect("beaten defender reward event");
        assert_eq!(event.tick, before.tick);
        assert_eq!(event.amount, -3.0);
        let transitions = sim.learning_transitions_for(
            &before,
            &after,
            sim.score_home,
            sim.score_away,
            &sim.reward_events[event_start..],
        );
        let defender_transition = transitions
            .iter()
            .find(|transition| transition.player_id == defender)
            .expect("defender learning transition");
        assert!(
            defender_transition.reward < -2.8,
            "dense shaped beaten reward: {}",
            defender_transition.reward
        );
    }

    #[test]
    fn possession_chase_signal_rewards_attack_and_adds_defender_fatigue() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 1.0,
            duration_seconds: 2.0,
            seed: 157,
            ..Default::default()
        });
        let holder = 5;
        let defenders = [11, 12, 13];
        sim.tick = 70;
        sim.ball.holder = Some(holder);
        sim.ball.last_touch_team = Some(Team::Home);
        sim.ball.position = Vec2::new(22.0, 72.0);
        sim.players[holder].position = sim.ball.position;
        sim.possession_chain.clear();
        sim.possession_chain.push_back(7);
        sim.possession_chain.push_back(holder);
        for player in &mut sim.players {
            player.velocity = Vec2::zero();
            player.acceleration = Vec2::zero();
            player.movement_gait = MovementGait::Stand;
            if player.team == Team::Away {
                player.position = Vec2::new(58.0, 70.0 + (player.id - 11) as f64 * 2.0);
            }
        }
        sim.players[holder].position = Vec2::new(22.0, 72.0);
        sim.players[7].position = Vec2::new(32.0, 76.0);
        let setup = WorldSnapshot::from_match(&sim);
        sim.players[holder].last_decision = Some(test_decision_trace(&setup, holder, "pass"));
        for defender in defenders {
            sim.players[defender].last_decision =
                Some(test_decision_trace(&setup, defender, "defend"));
            sim.players[defender].skills.stamina = 5.0;
            sim.players[defender].fatigue = 0.10;
        }
        let before = WorldSnapshot::from_match(&sim);
        let defender_fatigue_before = sim.players[11].fatigue;

        sim.ball.position = Vec2::new(38.0, 74.0);
        sim.players[holder].position = Vec2::new(22.6, 72.2);
        for (offset, defender) in defenders.into_iter().enumerate() {
            sim.players[defender].position.x -= 2.8 - offset as f64 * 0.25;
            sim.players[defender].velocity = Vec2::new(-2.8 + offset as f64 * 0.25, 0.0);
            sim.players[defender].acceleration = Vec2::new(-1.2, 0.0);
            sim.players[defender].movement_gait = MovementGait::Run;
        }
        let after = WorldSnapshot::from_match(&sim);
        let event_start = sim.reward_events.len();

        sim.update_possession_chase_trackers(&before, &after, true);

        assert!(sim.stats.defensive_chase_load_away > 0.0);
        assert!(sim.stats.possession_chase_advantage_home > 0.0);
        assert!(sim.players[11].fatigue > defender_fatigue_before);

        let holder_reward = sim.reward_events[event_start..]
            .iter()
            .filter(|event| event.player_id == holder)
            .map(|event| event.amount)
            .sum::<f64>();
        let defender_penalty = sim.reward_events[event_start..]
            .iter()
            .filter(|event| defenders.contains(&event.player_id))
            .map(|event| event.amount)
            .sum::<f64>();
        assert!(holder_reward > 0.0, "holder reward: {holder_reward}");
        assert!(
            defender_penalty < 0.0,
            "defender penalty: {defender_penalty}"
        );
    }

    #[test]
    fn compact_low_block_can_conserve_without_chase_penalty() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 1.0,
            duration_seconds: 2.0,
            seed: 158,
            ..Default::default()
        });
        let holder = 5;
        sim.tick = 80;
        sim.ball.holder = Some(holder);
        sim.ball.last_touch_team = Some(Team::Home);
        sim.ball.position = Vec2::new(24.0, 72.0);
        sim.players[holder].position = sim.ball.position;
        for player in &mut sim.players {
            player.velocity = Vec2::zero();
            player.acceleration = Vec2::zero();
            player.movement_gait = MovementGait::Stand;
            if player.team == Team::Away {
                player.position = Vec2::new(34.0 + (player.id - 11) as f64 * 0.8, 101.0);
                player.last_decision = None;
            }
        }
        let before = WorldSnapshot::from_match(&sim);

        sim.ball.position = Vec2::new(36.0, 72.5);
        let after = WorldSnapshot::from_match(&sim);
        let event_start = sim.reward_events.len();
        let fatigue_before = sim.players[11].fatigue;

        sim.update_possession_chase_trackers(&before, &after, true);

        assert_eq!(sim.reward_events.len(), event_start);
        assert_eq!(sim.stats.defensive_chase_load_away, 0.0);
        assert_eq!(sim.stats.possession_chase_advantage_home, 0.0);
        assert_eq!(sim.players[11].fatigue, fatigue_before);
    }

    #[test]
    fn relaxed_defenders_near_danger_yield_opening_scaled_by_energy_reserve() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 1.0,
            duration_seconds: 2.0,
            seed: 159,
            ..Default::default()
        });
        let holder = 5;
        let fresh_defender = 11;
        let tired_defender = 12;
        sim.tick = 90;
        sim.ball.holder = Some(holder);
        sim.ball.last_touch_team = Some(Team::Home);
        sim.ball.position = Vec2::new(38.0, 92.0);
        sim.players[holder].position = sim.ball.position;
        sim.possession_chain.clear();
        sim.possession_chain.push_back(holder);
        for player in &mut sim.players {
            player.velocity = Vec2::zero();
            player.acceleration = Vec2::zero();
            player.movement_gait = MovementGait::Stand;
            if player.team == Team::Away {
                player.position = Vec2::new(70.0, 104.0);
            }
        }
        sim.players[fresh_defender].position = Vec2::new(40.0, 98.0);
        sim.players[tired_defender].position = Vec2::new(44.0, 98.0);
        sim.players[fresh_defender].skills.stamina = 9.5;
        sim.players[fresh_defender].fatigue = 0.05;
        sim.players[tired_defender].skills.stamina = 2.0;
        sim.players[tired_defender].fatigue = 0.85;
        let setup = WorldSnapshot::from_match(&sim);
        sim.players[holder].last_decision = Some(test_decision_trace(&setup, holder, "pass"));
        sim.players[fresh_defender].last_decision =
            Some(test_decision_trace(&setup, fresh_defender, "hold"));
        sim.players[tired_defender].last_decision =
            Some(test_decision_trace(&setup, tired_defender, "hold"));
        let before = WorldSnapshot::from_match(&sim);

        sim.ball.position = Vec2::new(42.0, 96.0);
        let after = WorldSnapshot::from_match(&sim);
        let event_start = sim.reward_events.len();

        sim.update_possession_chase_trackers(&before, &after, true);

        let reward_for = |player_id| {
            sim.reward_events[event_start..]
                .iter()
                .filter(|event| event.player_id == player_id)
                .map(|event| event.amount)
                .sum::<f64>()
        };
        let holder_reward = reward_for(holder);
        let fresh_penalty = reward_for(fresh_defender);
        let tired_penalty = reward_for(tired_defender);
        assert!(holder_reward > 0.0, "holder reward: {holder_reward}");
        assert!(
            fresh_penalty < tired_penalty,
            "fresh reserve should be penalized more for relaxing: fresh={fresh_penalty}, tired={tired_penalty}"
        );
        assert!(sim.stats.possession_chase_advantage_home > 0.0);
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
        let action_target = passer
            .action_target
            .as_ref()
            .expect("tracking pass target trace");
        let target_position = tracking.frames[1]
            .players
            .iter()
            .find(|player| player.id == 1)
            .expect("tracking target player")
            .position;
        assert_eq!(action_target.player_id, Some(1));
        assert_eq!(action_target.point, Some(target_position));
        assert_eq!(
            action_target.grid.expect("tracking target grid").fine.id,
            pitch_grid_address(
                target_position,
                tracking.config.field_width_yards,
                tracking.config.field_length_yards
            )
            .fine
            .id
        );

        let policy =
            train_soccer_q_policy_from_tracking(&tracking, SoccerQPolicyOptions::default())
                .expect("tracking policy");
        let state = SoccerQStateKey::from_transition(passer);
        assert!(policy.q_value(&state, "pass").is_some());
        let target_entries = policy.target_entries();
        assert!(!target_entries.is_empty());
        let learned_target = policy
            .best_target_grid_for_state_action(&state, "pass")
            .expect("learned pass target grid");
        let target_grid = action_target.grid.expect("tracking target grid");
        assert_eq!(learned_target.target_fine_cell_id, target_grid.fine.id);
        assert_eq!(
            learned_target.target_tactical_cell_id,
            target_grid.tactical.id
        );

        let artifact =
            soccer_policy_artifact_from_learning_dataset(&dataset, SoccerQPolicyOptions::default());
        assert_eq!(artifact.transition_count, 3);
        assert!(!artifact.entries.is_empty());
        assert!(!artifact.target_entries.is_empty());
        let restored = SoccerQPolicy::from_entries_with_targets(
            artifact.options.clone(),
            &artifact.entries,
            &artifact.target_entries,
        )
        .expect("artifact restores target policy");
        assert_eq!(
            restored.target_entries().len(),
            artifact.target_entries.len()
        );
    }

    #[test]
    fn tracking_dataset_converts_aerial_pass_to_learning_transition_and_policy() {
        let mut tracking = sample_tracking_pass_dataset();
        tracking.source = "unit-aerial-pass".to_string();
        tracking.frames[1].pass_flight = Some(PassFlight::Aerial);
        tracking.frames[1].ball_altitude_yards = Some(4.2);

        let dataset = tracking.to_learning_dataset().expect("tracking conversion");
        let passer = dataset
            .transitions
            .iter()
            .find(|transition| transition.player_id == 0)
            .expect("passer transition");
        assert_eq!(passer.action, "aerial-pass");
        let action_target = passer
            .action_target
            .as_ref()
            .expect("tracking aerial pass target trace");
        assert_eq!(action_target.player_id, Some(1));

        let policy =
            train_soccer_q_policy_from_tracking(&tracking, SoccerQPolicyOptions::default())
                .expect("tracking policy");
        let state = SoccerQStateKey::from_transition(passer);
        assert!(policy.q_value(&state, "aerial-pass").is_some());
        assert!(policy
            .best_target_grid_for_state_action(&state, "aerial-pass")
            .is_some());
    }

    #[test]
    fn tracking_dataset_uses_imported_player_skills_in_learning_state() {
        let mut tracking = sample_tracking_pass_dataset();
        let mut skills = SkillProfile::default();
        skills.top_speed = 9.4;
        skills.acceleration = 8.9;
        skills.passing_completion_rate = 9.2;
        skills.crossing_left = 4.1;
        skills.crossing_right = 8.7;
        skills.defending = 3.3;
        skills.defensive_tracking = 5.2;
        skills.vision = 9.1;
        for frame in &mut tracking.frames {
            frame
                .players
                .iter_mut()
                .find(|player| player.id == 0)
                .expect("tracking passer")
                .skills = Some(skills.clone());
        }

        let json = serde_json::to_string(&tracking).expect("tracking json");
        let parsed = soccer_tracking_dataset_from_json(&json).expect("parse tracking");
        let dataset = parsed.to_learning_dataset().expect("tracking conversion");
        let passer = dataset
            .transitions
            .iter()
            .find(|transition| transition.player_id == 0)
            .expect("passer transition");
        assert_eq!(passer.observation.skill_top_speed, 9.4);
        assert_eq!(passer.observation.skill_acceleration, 8.9);
        assert_eq!(passer.observation.skill_passing_completion_rate, 9.2);
        assert_eq!(passer.observation.skill_crossing_left, 4.1);
        assert_eq!(passer.observation.skill_crossing_right, 8.7);
        assert_eq!(passer.observation.skill_defending, 3.3);
        assert_eq!(passer.observation.skill_defensive_tracking, 5.2);
        let state = SoccerQStateKey::from_transition(passer);
        assert_eq!(state.skill_top_speed_bin, skill_bucket(skills.top_speed));
        assert_eq!(
            state.skill_passing_completion_bin,
            skill_bucket(skills.passing_completion_rate)
        );
        assert_eq!(
            state.skill_crossing_left_bin,
            skill_bucket(skills.crossing_left)
        );
        assert_eq!(
            state.skill_crossing_right_bin,
            skill_bucket(skills.crossing_right)
        );
        assert_eq!(state.skill_defending_bin, skill_bucket(skills.defending));
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
    fn tracking_template_is_importable_training_data() {
        let template = soccer_tracking_template_dataset(&MatchConfig {
            duration_seconds: 0.2,
            seed: 302,
            ..Default::default()
        });

        assert_eq!(template.source, "tracking-template");
        assert_eq!(template.frames.len(), 2);
        assert_eq!(template.frames[0].players.len(), 3);
        template.validate().expect("template validates");
        let dataset = template.to_learning_dataset().expect("template learns");
        assert_eq!(dataset.transitions.len(), 3);
        assert!(dataset.transitions.iter().any(|t| t.action == "pass"));
        let policy =
            train_soccer_q_policy_from_tracking(&template, SoccerQPolicyOptions::default())
                .expect("template policy");
        assert!(!policy.entries().is_empty());
    }

    #[test]
    fn tracking_dataset_csv_imports_rows_and_trains_policy() {
        let config = MatchConfig {
            duration_seconds: 0.2,
            seed: 102,
            ..Default::default()
        };
        let raw = r#"tick,clock_seconds,player_id,name,team,role,shirt,x,y,vx,vy,home_x,home_y,ball_x,ball_y,ball_vx,ball_vy,ball_holder,last_touch_team,score_home,score_away
0,0.0,0,"Home, passer",Home,Midfielder,8,40.0,70.0,0.0,0.0,40.0,65.0,40.0,70.0,0.0,0.0,0,Home,0,0
0,0.0,1,Home runner,Home,Forward,9,44.0,82.0,0.0,0.0,44.0,80.0,40.0,70.0,0.0,0.0,0,Home,0,0
0,0.0,2,Away defender,Away,Defender,4,58.0,78.0,0.0,0.0,58.0,78.0,40.0,70.0,0.0,0.0,0,Home,0,0
1,0.1,0,"Home, passer",Home,Midfielder,8,40.2,70.4,2.0,4.0,40.0,65.0,44.0,82.0,8.0,16.0,1,Home,0,0
1,0.1,1,Home runner,Home,Forward,9,44.0,82.0,0.0,0.0,44.0,80.0,44.0,82.0,8.0,16.0,1,Home,0,0
1,0.1,2,Away defender,Away,Defender,4,56.5,78.5,-15.0,5.0,58.0,78.0,44.0,82.0,8.0,16.0,1,Home,0,0
"#;

        let tracking =
            soccer_tracking_dataset_from_csv(raw, config, "unit-csv").expect("csv tracking");
        assert_eq!(tracking.source, "unit-csv");
        assert_eq!(tracking.frames.len(), 2);
        assert_eq!(tracking.frames[0].players.len(), 3);
        assert_eq!(
            tracking.frames[0].players[0].name.as_deref(),
            Some("Home, passer")
        );
        assert_eq!(tracking.frames[1].ball_holder, Some(1));
        assert_eq!(
            tracking.frames[1].players[0].velocity,
            Some(Vec2::new(2.0, 4.0))
        );

        let dataset = tracking.to_learning_dataset().expect("learning dataset");
        assert_eq!(dataset.transitions.len(), 3);
        let passer = dataset
            .transitions
            .iter()
            .find(|transition| transition.player_id == 0)
            .expect("passer transition");
        assert_eq!(passer.action, "pass");
        let policy =
            train_soccer_q_policy_from_tracking(&tracking, SoccerQPolicyOptions::default())
                .expect("policy from csv");
        let state = SoccerQStateKey::from_transition(passer);
        assert!(policy.q_value(&state, "pass").is_some());
    }

    #[test]
    fn tracking_dataset_csv_imports_player_skill_columns() {
        let config = MatchConfig {
            duration_seconds: 0.2,
            seed: 103,
            ..Default::default()
        };
        let raw = r#"tick,clock_seconds,player_id,name,team,role,shirt,x,y,ball_x,ball_y,ball_holder,last_touch_team,top_speed,acceleration,dribbling,passing_completion_rate,crossing_left,crossing_right,defensive_ability,ability_in_goal,vision
0,0.0,0,Home passer,Home,Midfielder,8,40.0,70.0,40.0,70.0,0,Home,9.6,8.8,8.2,9.1,4.2,8.9,3.1,2.0,9.4
0,0.0,1,Home runner,Home,Forward,9,44.0,82.0,40.0,70.0,0,Home,8.0,8.0,8.0,7.0,7.0,7.0,4.0,2.0,7.0
0,0.0,2,Away defender,Away,Defender,4,58.0,78.0,40.0,70.0,0,Home,7.0,7.0,5.0,6.0,5.0,5.0,8.5,2.0,6.0
1,0.1,0,Home passer,Home,Midfielder,8,40.2,70.4,44.0,82.0,1,Home,9.6,8.8,8.2,9.1,4.2,8.9,3.1,2.0,9.4
1,0.1,1,Home runner,Home,Forward,9,44.0,82.0,44.0,82.0,1,Home,8.0,8.0,8.0,7.0,7.0,7.0,4.0,2.0,7.0
1,0.1,2,Away defender,Away,Defender,4,56.5,78.5,44.0,82.0,1,Home,7.0,7.0,5.0,6.0,5.0,5.0,8.5,2.0,6.0
"#;

        let tracking =
            soccer_tracking_dataset_from_csv(raw, config, "unit-csv-skills").expect("csv tracking");
        let imported = tracking.frames[0].players[0]
            .skills
            .as_ref()
            .expect("csv player skills");
        assert_eq!(imported.top_speed, 9.6);
        assert_eq!(imported.acceleration, 8.8);
        assert_eq!(imported.passing_completion_rate, 9.1);
        assert_eq!(imported.crossing_right, 8.9);
        assert_eq!(imported.defending, 3.1);
        assert_eq!(imported.goalkeeping, 2.0);
        assert_eq!(imported.vision, 9.4);

        let dataset = tracking.to_learning_dataset().expect("learning dataset");
        let passer = dataset
            .transitions
            .iter()
            .find(|transition| transition.player_id == 0)
            .expect("passer transition");
        assert_eq!(passer.observation.skill_top_speed, 9.6);
        assert_eq!(passer.observation.skill_crossing_left, 4.2);
        assert_eq!(passer.observation.skill_crossing_right, 8.9);
        assert_eq!(passer.observation.skill_defending, 3.1);
        let state = SoccerQStateKey::from_transition(passer);
        assert_eq!(state.skill_top_speed_bin, skill_bucket(9.6));
        assert_eq!(state.skill_crossing_left_bin, skill_bucket(4.2));
        assert_eq!(state.skill_crossing_right_bin, skill_bucket(8.9));
        assert_eq!(state.skill_defending_bin, skill_bucket(3.1));
    }

    #[test]
    fn tracking_dataset_csv_imports_aerial_pass_metadata() {
        let config = MatchConfig {
            duration_seconds: 0.2,
            seed: 104,
            ..Default::default()
        };
        let raw = r#"tick,clock_seconds,player_id,name,team,role,shirt,x,y,ball_x,ball_y,ball_vx,ball_vy,ball_altitude_yards,pass_flight,ball_holder,last_touch_team,score_home,score_away
0,0.0,0,Home passer,Home,Midfielder,8,40.0,70.0,40.0,70.0,0.0,0.0,0.0,,0,Home,0,0
0,0.0,1,Home runner,Home,Forward,9,44.0,82.0,40.0,70.0,0.0,0.0,0.0,,0,Home,0,0
0,0.0,2,Away defender,Away,Defender,4,58.0,78.0,40.0,70.0,0.0,0.0,0.0,,0,Home,0,0
1,0.1,0,Home passer,Home,Midfielder,8,40.2,70.4,44.0,82.0,8.0,16.0,3.2,aerial,1,Home,0,0
1,0.1,1,Home runner,Home,Forward,9,44.0,82.0,44.0,82.0,8.0,16.0,3.2,aerial,1,Home,0,0
1,0.1,2,Away defender,Away,Defender,4,56.5,78.5,44.0,82.0,8.0,16.0,3.2,aerial,1,Home,0,0
"#;

        let tracking =
            soccer_tracking_dataset_from_csv(raw, config, "unit-csv-aerial").expect("csv tracking");
        assert_eq!(tracking.frames[1].pass_flight, Some(PassFlight::Aerial));
        assert_eq!(tracking.frames[1].ball_altitude_yards, Some(3.2));

        let dataset = tracking.to_learning_dataset().expect("learning dataset");
        let passer = dataset
            .transitions
            .iter()
            .find(|transition| transition.player_id == 0)
            .expect("passer transition");
        assert_eq!(passer.action, "aerial-pass");
    }

    #[test]
    fn tracking_dataset_csv_imports_normalized_footage_coordinates() {
        let config = MatchConfig {
            duration_seconds: 0.2,
            seed: 105,
            ..Default::default()
        };
        let raw = r#"tick,clock_seconds,player_id,name,team,role,shirt,x_norm,y_norm,home_x_norm,home_y_norm,ball_x_norm,ball_y_norm,ball_holder,last_touch_team
0,0.0,0,Home passer,Home,Midfielder,8,0.500000,0.583333,0.500000,0.541667,0.500000,0.583333,0,Home
0,0.0,1,Home runner,Home,Forward,9,0.550000,0.683333,0.550000,0.666667,0.500000,0.583333,0,Home
0,0.0,2,Away defender,Away,Defender,4,0.725000,0.650000,0.725000,0.650000,0.500000,0.583333,0,Home
1,0.1,0,Home passer,Home,Midfielder,8,0.502500,0.586667,0.500000,0.541667,0.550000,0.683333,1,Home
1,0.1,1,Home runner,Home,Forward,9,0.550000,0.683333,0.550000,0.666667,0.550000,0.683333,1,Home
1,0.1,2,Away defender,Away,Defender,4,0.706250,0.654167,0.725000,0.650000,0.550000,0.683333,1,Home
"#;

        let tracking = soccer_tracking_dataset_from_csv(raw, config, "unit-csv-normalized")
            .expect("normalized csv tracking");
        let first = &tracking.frames[0];
        assert!((first.players[0].position.x - 40.0).abs() < 1e-6);
        assert!((first.players[0].position.y - 69.99996).abs() < 1e-4);
        assert!((first.ball_position.x - 40.0).abs() < 1e-6);
        assert!((first.ball_position.y - 69.99996).abs() < 1e-4);
        let home = first.players[0]
            .home_position
            .expect("normalized home position");
        assert!((home.x - 40.0).abs() < 1e-6);
        assert!((home.y - 65.00004).abs() < 1e-4);

        let dataset = tracking.to_learning_dataset().expect("learning dataset");
        let passer = dataset
            .transitions
            .iter()
            .find(|transition| transition.player_id == 0)
            .expect("passer transition");
        assert_eq!(passer.action, "pass");
    }

    #[test]
    fn tracking_dataset_csv_imports_pixel_footage_coordinates() {
        let config = MatchConfig {
            duration_seconds: 0.2,
            seed: 106,
            ..Default::default()
        };
        let raw = r#"tick,clock_seconds,player_id,name,team,role,shirt,pixel_x,pixel_y,home_pixel_x,home_pixel_y,ball_pixel_x,ball_pixel_y,image_width,image_height,ball_holder,last_touch_team
0,0.0,0,Home passer,Home,Midfielder,8,400,700,400,650,400,700,800,1200,0,Home
0,0.0,1,Home runner,Home,Forward,9,440,820,440,800,400,700,800,1200,0,Home
0,0.0,2,Away defender,Away,Defender,4,580,780,580,780,400,700,800,1200,0,Home
1,0.1,0,Home passer,Home,Midfielder,8,402,704,400,650,440,820,800,1200,1,Home
1,0.1,1,Home runner,Home,Forward,9,440,820,440,800,440,820,800,1200,1,Home
1,0.1,2,Away defender,Away,Defender,4,565,785,580,780,440,820,800,1200,1,Home
"#;

        let tracking = soccer_tracking_dataset_from_csv(raw, config, "unit-csv-pixels")
            .expect("pixel csv tracking");
        let first = &tracking.frames[0];
        assert_eq!(first.players[0].position, Vec2::new(40.0, 70.0));
        assert_eq!(first.ball_position, Vec2::new(40.0, 70.0));
        assert_eq!(first.players[0].home_position, Some(Vec2::new(40.0, 65.0)));

        let dataset = tracking.to_learning_dataset().expect("learning dataset");
        let passer = dataset
            .transitions
            .iter()
            .find(|transition| transition.player_id == 0)
            .expect("passer transition");
        assert_eq!(passer.action, "pass");
    }

    #[test]
    fn live_http_tracking_import_trains_team_policies() {
        let session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 2,
            seed: 114,
            ..Default::default()
        })));
        let input_queue = session.lock().unwrap().input_queue();
        let tracking = sample_tracking_pass_dataset();
        let tracking_json = serde_json::to_string(&tracking).expect("tracking json");
        let body = serde_json::json!({
            "source": "unit-tracking.json",
            "format": "json",
            "content": tracking_json
        })
        .to_string();

        let import = handle_live_soccer_request(
            &format!(
                "POST /api/tracking-policy HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
            &session,
            &input_queue,
        );

        assert_eq!(import.status, 200);
        let value: serde_json::Value =
            serde_json::from_str(&import.body).expect("tracking import json");
        assert_eq!(value["format"], "json");
        assert_eq!(value["frames"], 2);
        assert_eq!(value["importedTransitions"], 3);
        assert!(value["importedHomeEntries"].as_u64().unwrap() > 0);
        assert!(value["learning"]["homePolicyEntries"].as_u64().unwrap() > 0);
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
            sim.players[0].skills.first_touch = 1.0;
            sim.players[0].skills.dribbling = 1.0;
            sim.players[0].skills.aggression = 1.0;
            sim.players[11].position = Vec2::new(41.15, 60.0);
            sim.players[11].velocity = Vec2::new(-4.0, 0.0);
            sim.players[11].skills.first_touch = 9.8;
            sim.players[11].skills.dribbling = 9.8;
            sim.players[11].skills.aggression = 9.5;

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
    fn no_possession_nearest_players_chase_loose_ball() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(40.0, 60.0);
        sim.ball.velocity = Vec2::zero();
        sim.ball.last_touch_team = Some(Team::Away);
        for player in &mut sim.players {
            player.position = match player.team {
                Team::Home => Vec2::new(8.0, 20.0 + player.id as f64),
                Team::Away => Vec2::new(72.0, 80.0 + player.id as f64),
            };
        }
        sim.players[5].position = Vec2::new(38.0, 60.0);
        sim.players[6].position = Vec2::new(42.0, 60.0);
        sim.players[7].position = Vec2::new(64.0, 60.0);

        let snapshot = WorldSnapshot::from_match(&sim);
        let mut chaser = sim.players[5].clone();
        let mut rng = mulberry32(72);
        let intent = chaser.run_time_step(&snapshot, None, None, &mut rng);

        match intent.action {
            SoccerAction::MoveTo(target) => assert!(target.distance(sim.ball.position) < 0.01),
            other => panic!("expected loose-ball recovery move, got {other:?}"),
        }
        assert_eq!(
            chaser
                .last_decision
                .as_ref()
                .expect("decision trace")
                .action,
            "recover"
        );
    }

    #[test]
    fn held_ball_leads_in_carrier_movement_direction() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.ball.holder = Some(5);
        sim.players[5].position = Vec2::new(40.0, 60.0);
        sim.players[5].velocity = Vec2::new(6.0, 0.0);

        sim.integrate_ball();

        assert!(sim.ball.position.x > sim.players[5].position.x + 0.75);
        assert!((sim.ball.position.y - sim.players[5].position.y).abs() < 0.05);
        assert_eq!(sim.ball.last_touch_team, Some(Team::Home));
    }

    #[test]
    fn dribble_heavy_touch_probability_is_skill_and_pressure_weighted() {
        let sim = SoccerMatch::default_11v11(MatchConfig::default());
        let mut poor = sim.players[9].clone();
        poor.skills.dribbling = 1.0;
        poor.skills.first_touch = 1.2;
        poor.skills.stamina = 3.8;
        poor.fatigue = 0.80;
        let mut elite = sim.players[9].clone();
        elite.skills.dribbling = 9.8;
        elite.skills.first_touch = 9.6;
        elite.skills.stamina = 9.4;
        elite.fatigue = 0.0;

        let poor_under_pressure = dribble_heavy_touch_probability(&poor, 0.90);
        let elite_under_pressure = dribble_heavy_touch_probability(&elite, 0.90);
        let poor_unpressured = dribble_heavy_touch_probability(&poor, 0.05);

        assert!(poor_under_pressure > elite_under_pressure * 8.0);
        assert!(poor_under_pressure > poor_unpressured * 4.0);
        assert!(elite_under_pressure < 0.03);
    }

    #[test]
    fn poor_dribbler_can_take_heavy_touch_and_lose_ball() {
        let mut saw_heavy_touch = false;
        for seed in 0..320 {
            let mut sim = SoccerMatch::default_11v11(MatchConfig {
                seed,
                ..Default::default()
            });
            let dribbler = 9;
            sim.players[dribbler].position = Vec2::new(40.0, 60.0);
            sim.players[dribbler].skills.dribbling = 1.0;
            sim.players[dribbler].skills.first_touch = 1.0;
            sim.players[dribbler].skills.stamina = 3.0;
            sim.players[dribbler].fatigue = 0.90;
            sim.players[11].position = Vec2::new(41.0, 64.0);
            for away in 12..22 {
                sim.players[away].position = Vec2::new(72.0, 92.0);
            }
            sim.ball.holder = Some(dribbler);
            sim.ball.position = sim.players[dribbler].position;
            sim.ball.last_touch_team = Some(Team::Home);

            sim.apply_player_intent(PlayerIntent {
                player_id: dribbler,
                action: SoccerAction::Dribble(Vec2::new(40.0, 78.0)),
                sprint: true,
            });

            if sim.events.iter().any(|event| event.kind == "heavy-touch") {
                saw_heavy_touch = true;
                assert_eq!(sim.ball.holder, None);
                assert!(sim.ball.velocity.len() > sim.players[dribbler].velocity.len());
                assert!(
                    sim.ball.position.distance(sim.players[dribbler].position)
                        > PLAYER_CONTROL_RADIUS_YARDS
                );
                break;
            }
        }
        assert!(saw_heavy_touch);
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
    fn assistant_refs_track_effective_offside_line() {
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
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        let near_line =
            assistant_offside_line_snapshot(&snapshot, OfficialKind::AssistantRefereeNear)
                .expect("near assistant line");
        assert_eq!(near_line.flank, AssistantFlank::Near);
        assert_eq!(near_line.attacking_team, Team::Home);
        assert_eq!(near_line.defending_team, Team::Away);
        assert_eq!(near_line.second_last_defender_y, 96.0);
        assert_eq!(near_line.ball_y, 70.0);
        assert_eq!(near_line.halfway_y, 60.0);
        assert_eq!(near_line.effective_line_y, 96.0);
        assert!(near_line.players_beyond_line.contains(&9));

        let far_line =
            assistant_offside_line_snapshot(&snapshot, OfficialKind::AssistantRefereeFar)
                .expect("far assistant line");
        assert_eq!(far_line.flank, AssistantFlank::Far);
        assert!(!far_line.players_beyond_line.contains(&9));

        let y_before = sim
            .officials
            .iter()
            .find(|official| official.kind == OfficialKind::AssistantRefereeNear)
            .expect("near assistant")
            .position
            .y;
        sim.officials
            .iter_mut()
            .find(|official| official.kind == OfficialKind::AssistantRefereeNear)
            .expect("near assistant")
            .run_time_step(&snapshot, &mut mulberry32(607));
        let near_assistant = sim
            .officials
            .iter()
            .find(|official| official.kind == OfficialKind::AssistantRefereeNear)
            .expect("near assistant");
        assert!(near_assistant.position.y > y_before);

        let frame = sim.to_frame();
        let assistant_lines = frame
            .officials
            .iter()
            .filter_map(|official| official.offside_line.as_ref())
            .collect::<Vec<_>>();
        assert_eq!(assistant_lines.len(), 2);
        assert!(assistant_lines
            .iter()
            .any(|line| line.effective_line_y == 96.0));
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
                flight: PassFlight::Floor,
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
    fn attacking_support_seeks_forward_open_space_without_running_offside() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        sim.ball.holder = Some(5);
        sim.ball.position = Vec2::new(40.0, 56.0);
        sim.ball.last_touch_team = Some(Team::Home);
        sim.players[5].position = sim.ball.position;
        sim.players[9].position = Vec2::new(31.0, 60.0);
        for away in 11..22 {
            sim.players[away].position = Vec2::new(68.0, 92.0);
        }
        sim.players[11].position = Vec2::new(40.0, 118.0);
        sim.players[12].position = Vec2::new(44.0, 108.0);

        let before = WorldSnapshot::from_match(&sim);
        let mut rng = mulberry32(206);
        sim.central_brain.run_time_step(&before, &mut rng);
        let snapshot = WorldSnapshot::from_match(&sim);
        let target = snapshot.positional_open_space_for(9, sim.players[9].home_position, false);

        assert!(
            (target.y - sim.ball.position.y) * Team::Home.attack_dir() > 4.0,
            "target should be ahead of the ball: {target:?}"
        );
        assert!(
            target.y > sim.players[9].position.y,
            "target should invite the forward into higher space: {target:?}"
        );
        assert!(
            !snapshot.position_would_be_offside(Team::Home, target),
            "support target should avoid offside: {target:?}"
        );
        assert!(
            target.distance(sim.players[9].home_position) <= 20.0 + 1e-9,
            "non-roaming support should stay in positional radius"
        );
    }

    #[test]
    fn no_pressure_pass_ranking_filters_backward_outlets() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let passer = 6;
        let backward_outlet = 5;
        let forward_outlet = 7;
        sim.players[passer].position = Vec2::new(40.0, 50.0);
        sim.players[backward_outlet].position = Vec2::new(34.0, 41.0);
        sim.players[forward_outlet].position = Vec2::new(46.0, 60.0);
        for id in 11..22 {
            sim.players[id].position = Vec2::new(72.0, 102.0);
        }
        sim.ball.position = sim.players[passer].position;
        sim.ball.holder = Some(passer);
        sim.ball.last_touch_team = Some(Team::Home);

        let snapshot = WorldSnapshot::from_match(&sim);
        assert!(snapshot.no_pressure_at(Team::Home, sim.players[passer].position));
        let targets = snapshot.ranked_pass_targets(passer, 11);

        assert!(targets.contains(&forward_outlet));
        assert!(
            !targets.contains(&backward_outlet),
            "backward outlet should be ineligible without pressure: {targets:?}"
        );
    }

    #[test]
    fn own_half_possession_pushes_support_upfield() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let holder = 6;
        let runner = 7;
        sim.ball.holder = Some(holder);
        sim.ball.position = Vec2::new(40.0, 32.0);
        sim.ball.last_touch_team = Some(Team::Home);
        sim.players[holder].position = sim.ball.position;
        sim.players[runner].position = Vec2::new(47.0, 31.0);
        for id in 11..22 {
            sim.players[id].position = Vec2::new(70.0, 90.0);
        }

        let before = WorldSnapshot::from_match(&sim);
        let mut rng = mulberry32(706);
        sim.central_brain.run_time_step(&before, &mut rng);
        let snapshot = WorldSnapshot::from_match(&sim);
        let target =
            snapshot.positional_open_space_for(runner, sim.players[runner].home_position, false);

        assert!(
            target.y > sim.players[runner].position.y + 3.0,
            "own-half support should urgently move upfield: {target:?}"
        );
        assert!(
            snapshot.home_directive.carry_priority > 1.0,
            "own-half possession should increase carry urgency"
        );
    }

    #[test]
    fn marked_receiver_checks_to_ball_when_space_behind_opens() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let holder = 6;
        let receiver = 9;
        let marker = 12;
        sim.ball.holder = Some(holder);
        sim.ball.position = Vec2::new(40.0, 56.0);
        sim.ball.last_touch_team = Some(Team::Home);
        sim.players[holder].position = sim.ball.position;
        sim.players[receiver].position = Vec2::new(31.0, 70.0);
        sim.players[marker].position = Vec2::new(32.1, 70.0);
        for id in 11..22 {
            if id != marker {
                sim.players[id].position = Vec2::new(70.0, 98.0);
            }
        }
        sim.players[11].position = Vec2::new(40.0, 118.0);

        let snapshot = WorldSnapshot::from_match(&sim);
        let current = sim.players[receiver].position;
        let target = snapshot.positional_open_space_for(
            receiver,
            sim.players[receiver].home_position,
            false,
        );

        assert!(
            target.y < current.y,
            "marked receiver should check back toward the ball: {target:?}"
        );
        assert!(
            target.distance(sim.ball.position) < current.distance(sim.ball.position),
            "checking movement should get closer to the ball"
        );
    }

    #[test]
    fn off_target_pending_pass_receiver_sprints_to_ball() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let passer = 6;
        let receiver = 9;
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(34.0, 64.0);
        sim.ball.velocity = Vec2::new(8.0, 2.0);
        sim.ball.last_touch_team = Some(Team::Home);
        sim.players[passer].position = Vec2::new(40.0, 52.0);
        sim.players[receiver].position = Vec2::new(49.0, 72.0);
        sim.pending_pass = Some(PendingPass {
            team: Team::Home,
            from: passer,
            target: Some(receiver),
            flight: PassFlight::Floor,
            is_cross: false,
            origin: sim.players[passer].position,
            intended_target: sim.players[receiver].position,
            distance_yards: sim.players[passer]
                .position
                .distance(sim.players[receiver].position),
            offside: None,
        });
        sim.players[12].position = Vec2::new(36.0, 64.0);

        let snapshot = WorldSnapshot::from_match(&sim);
        let observation = snapshot.observation_for(receiver);
        assert!(observation.receiving_pending_pass);
        assert!(observation.pending_pass_off_target_yards > 1.25);
        let mut rng = mulberry32(808);
        let intent = sim.players[receiver].run_time_step(&snapshot, None, None, &mut rng);

        assert!(intent.sprint);
        assert_eq!(
            sim.players[receiver]
                .last_decision
                .as_ref()
                .map(|decision| decision.action.as_str()),
            Some("recover")
        );
        match intent.action {
            SoccerAction::MoveTo(target) => {
                assert!(
                    target.distance(sim.ball.position)
                        < sim.players[receiver].position.distance(sim.ball.position)
                );
            }
            _ => panic!("receiver should move to recover the pass"),
        }
    }

    #[test]
    fn neutral_staging_creates_occasional_in_behind_aerial_option() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let passer = 6;
        let runner = 9;
        sim.tick = 12;
        sim.clock_seconds = sim.tick as f64 * sim.config.dt_seconds;
        sim.ball.holder = Some(passer);
        sim.ball.position = Vec2::new(40.0, 58.0);
        sim.ball.last_touch_team = Some(Team::Home);
        sim.players[passer].position = sim.ball.position;
        sim.players[runner].position = Vec2::new(31.0, 76.0);
        sim.players[10].position = Vec2::new(52.0, 52.0);
        for id in 11..22 {
            sim.players[id].position = Vec2::new(68.0, 92.0);
        }
        sim.players[11].position = Vec2::new(40.0, 118.0);
        sim.players[12].position = Vec2::new(42.0, 96.0);

        let snapshot = WorldSnapshot::from_match(&sim);
        let run_target = snapshot
            .in_behind_run_target_for(runner)
            .expect("runner should have an in-behind window");
        let pass_point = snapshot
            .projected_in_behind_pass_point(passer, runner)
            .expect("pass should lead the runner behind the line");
        let targets = snapshot.ranked_visible_aerial_pass_targets(passer, 3);

        assert!(run_target.y > sim.players[runner].position.y);
        assert!(pass_point.y > snapshot.second_last_defender_line_for(Team::Home).unwrap());
        assert_eq!(targets.first().copied(), Some(runner));
    }

    #[test]
    fn spacing_scores_prefer_attack_spread_and_defensive_compactness() {
        assert!(
            spacing_score_from_distance(ATTACK_SPACING_IDEAL_YARDS, TeamSpacingMode::InPossession)
                > spacing_score_from_distance(3.0, TeamSpacingMode::InPossession)
        );
        assert!(
            spacing_score_from_distance(ATTACK_SPACING_IDEAL_YARDS, TeamSpacingMode::InPossession)
                > spacing_score_from_distance(22.0, TeamSpacingMode::InPossession)
        );
        assert!(
            spacing_score_from_distance(DEFENSE_SPACING_IDEAL_YARDS, TeamSpacingMode::Defending)
                > spacing_score_from_distance(10.0, TeamSpacingMode::Defending)
        );
    }

    #[test]
    fn tactical_learning_scores_prefer_flanks_and_defensive_contraction() {
        let field_width = DEFAULT_FIELD_WIDTH_YARDS;

        assert!(
            attack_width_score(field_width * 0.78, field_width)
                > attack_width_score(field_width * 0.30, field_width)
        );
        assert!(
            flank_lane_score(Vec2::new(field_width * 0.12, 40.0), field_width)
                > flank_lane_score(Vec2::new(field_width * 0.50, 40.0), field_width)
        );
        assert!(
            defense_contract_width_score(field_width * 0.38, field_width)
                > defense_contract_width_score(field_width * 0.82, field_width)
        );
    }

    #[test]
    fn defensive_support_blends_man_marking_with_zone_shape() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());
        let defender_id = 2;
        let threat_id = 20;
        sim.ball.holder = Some(threat_id);
        sim.ball.position = Vec2::new(37.0, 33.0);
        sim.ball.last_touch_team = Some(Team::Away);
        sim.players[threat_id].position = sim.ball.position;
        for id in 11..22 {
            if id != threat_id {
                sim.players[id].position = Vec2::new(70.0, 92.0);
            }
        }

        let before = WorldSnapshot::from_match(&sim);
        let mut rng = mulberry32(207);
        sim.central_brain.run_time_step(&before, &mut rng);
        let snapshot = WorldSnapshot::from_match(&sim);
        let zone =
            snapshot.defensive_shape_for(defender_id, sim.players[defender_id].home_position);
        let target = snapshot.defensive_assignment_for(
            defender_id,
            sim.players[defender_id].home_position,
            false,
        );
        let threat = sim.players[threat_id].position;

        assert!(
            target.distance(threat) < zone.distance(threat),
            "assignment should close down the dangerous attacker"
        );
        assert!(
            target.y < threat.y,
            "home defender should stay goal-side of the threat: {target:?} vs {threat:?}"
        );
        assert!(
            target.distance(sim.players[defender_id].home_position) <= 13.0 + 1e-9,
            "non-roaming defender should stay near role position"
        );
    }

    #[test]
    fn configurable_ball_drag_controls_roll_down_speed() {
        let mut short_grass = SoccerMatch::default_11v11(MatchConfig {
            ball_drag_per_tick: 0.005,
            ball_air_resistance: 0.0,
            ball_grass_resistance_yps2: 0.0,
            seed: 301,
            ..Default::default()
        });
        let mut long_grass = SoccerMatch::default_11v11(MatchConfig {
            ball_drag_per_tick: 0.20,
            ball_air_resistance: 0.0,
            ball_grass_resistance_yps2: 0.0,
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
    fn ball_air_and_grass_resistance_reduce_rolling_speed() {
        let base_velocity = Vec2::new(18.0, 0.0);
        let low_resistance =
            ball_velocity_after_resistance(base_velocity, 0.1, 0.0, 0.0, 0.0).len();
        let air_only = ball_velocity_after_resistance(base_velocity, 0.1, 0.0, 0.012, 0.0).len();
        let grass_only = ball_velocity_after_resistance(base_velocity, 0.1, 0.0, 0.0, 0.9).len();
        let combined = ball_velocity_after_resistance(base_velocity, 0.1, 0.0, 0.012, 0.9).len();

        assert_eq!(low_resistance, base_velocity.len());
        assert!(air_only < low_resistance);
        assert!(grass_only < low_resistance);
        assert!(combined < air_only);
        assert!(combined < grass_only);
    }

    #[test]
    fn runtime_surface_update_changes_next_ball_drag_tick() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            ball_drag_per_tick: 0.005,
            ball_air_resistance: 0.0,
            ball_grass_resistance_yps2: 0.0,
            seed: 302,
            ..Default::default()
        });
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(40.0, 60.0);
        sim.ball.velocity = Vec2::new(10.0, 0.0);
        sim.ball.last_touch_team = Some(Team::Home);

        sim.update_ball_surface(SoccerBallSurfaceRequest {
            ball_drag_per_tick: 0.20,
            ball_air_resistance: 0.0,
            ball_grass_resistance_yps2: 0.0,
            ball_stop_speed_yps: 0.7,
        })
        .expect("surface update");
        sim.integrate_ball();

        assert_eq!(sim.config.ball_drag_per_tick, 0.20);
        assert_eq!(sim.config.ball_stop_speed_yps, 0.7);
        assert!((sim.ball.velocity.len() - 8.0).abs() < 1e-9);
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
    fn goal_restart_sets_center_kickoff_for_conceding_team() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig::default());

        sim.score_goal(Team::Home);

        let center = Vec2::new(
            sim.config.field_width_yards * 0.5,
            sim.config.field_length_yards * 0.5,
        );
        let holder_id = sim.ball.holder.expect("kickoff holder");
        let holder = sim
            .players
            .iter()
            .find(|player| player.id == holder_id)
            .expect("holder player");
        assert_eq!(sim.score_home, 1);
        assert_eq!(sim.score_away, 0);
        assert_eq!(holder.team, Team::Away);
        assert_eq!(sim.ball.position, center);
        assert_eq!(holder.position, center);
        assert_eq!(sim.ball.velocity, Vec2::zero());
        assert_eq!(sim.ball.last_touch_team, Some(Team::Away));
        assert_eq!(
            sim.ball
                .last_decision
                .as_ref()
                .expect("ball decision")
                .action,
            "kickoff"
        );
        assert!(sim
            .events
            .iter()
            .any(|event| event.kind == "goal" && event.team == Some(Team::Home)));
        assert!(sim
            .events
            .iter()
            .any(|event| event.kind == "kickoff" && event.team == Some(Team::Away)));
    }

    #[test]
    fn fast_shot_scores_from_goal_line_crossing_even_when_tick_overshoots() {
        let mut sim = SoccerMatch::default_11v11(MatchConfig {
            dt_seconds: 2.0,
            duration_seconds: 2.0,
            seed: 164,
            ..Default::default()
        });
        if let Some(keeper_id) = sim.goalkeeper_for(Team::Away) {
            sim.players[keeper_id].role = PlayerRole::Defender;
        }
        sim.ball.holder = None;
        sim.ball.position = Vec2::new(60.0, 100.0);
        sim.ball.velocity = (Vec2::new(40.0, 120.0) - sim.ball.position).normalized() * 44.0;
        sim.ball.last_touch_team = Some(Team::Home);
        sim.pending_shot = Some(PendingShot {
            team: Team::Home,
            shooter: 9,
        });

        sim.integrate_ball();

        assert_eq!(sim.score_home, 1);
        assert_eq!(sim.stats.shots_on_target_home, 1);
        assert!(sim
            .events
            .iter()
            .any(|event| event.kind == "goal" && event.team == Some(Team::Home)));
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
            sim.players[0].skills.defending = 8.0;
            sim.players[0].skills.aggression = 5.5;
            sim.players[11].position = Vec2::new(41.2, 60.0);
            sim.players[11].skills.dribbling = 7.0;
            sim.players[11].skills.first_touch = 7.0;
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
            defending: 9.6,
            aggression: 1.8,
            ..neutral_tracking_skill_profile(PlayerRole::Defender)
        };
        let reckless_defender = SkillProfile {
            defending: 2.0,
            aggression: 9.6,
            ..neutral_tracking_skill_profile(PlayerRole::Defender)
        };
        let attacker = SkillProfile {
            dribbling: 8.8,
            first_touch: 8.6,
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
            sim.players[keeper_id].skills.defending = 9.8;
            sim.players[keeper_id].skills.first_touch = 9.8;
            sim.players[keeper_id].skills.acceleration = 9.5;
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
        let state_value: serde_json::Value = serde_json::from_str(&state.body).expect("state json");
        assert_eq!(state_value["learning"]["teamPoliciesEnabled"], true);
        assert_eq!(state_value["learning"]["totalTransitions"], 0);

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
        assert_eq!(value["learning"]["totalTransitions"], 44);
        assert!(value["learning"]["homePolicyEntries"].as_u64().unwrap() > 0);
        assert!(value["learning"]["awayPolicyEntries"].as_u64().unwrap() > 0);
        assert!(value["frame"]["players"][0]["lastDecision"]
            .get("mdpState")
            .is_some());
        assert!(value["frame"].get("homeDirective").is_some());
        assert!(value["frame"].get("awayDirective").is_some());
        let agent_schedule = value["frame"]["agentSchedule"]
            .as_array()
            .expect("agent schedule array");
        assert_eq!(agent_schedule.len(), 26);
        assert_eq!(agent_schedule.last().unwrap()["kind"], "ball");
        assert_eq!(agent_schedule.last().unwrap()["id"], BALL_AGENT_ID);
        let official_offside_lines = value["frame"]["officials"]
            .as_array()
            .expect("officials array")
            .iter()
            .filter(|official| official["offsideLine"].is_object())
            .count();
        assert_eq!(official_offside_lines, 2);
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

        let tracking = handle_live_soccer_request(
            "GET /api/tracking-dataset HTTP/1.1\r\nHost: local\r\n\r\n",
            &session,
            &input_queue,
        );
        assert_eq!(tracking.status, 200);
        let tracking_value: serde_json::Value =
            serde_json::from_str(&tracking.body).expect("tracking dataset json");
        assert_eq!(tracking_value["source"], "live-session");
        assert_eq!(tracking_value["frames"].as_array().unwrap().len(), 3);
        assert_eq!(
            tracking_value["frames"][2]["players"]
                .as_array()
                .unwrap()
                .len(),
            22
        );
        assert_eq!(tracking_value["frames"][2]["tick"], 2);

        let template = handle_live_soccer_request(
            "GET /api/tracking-template HTTP/1.1\r\nHost: local\r\n\r\n",
            &session,
            &input_queue,
        );
        assert_eq!(template.status, 200);
        let template_value: serde_json::Value =
            serde_json::from_str(&template.body).expect("tracking template json");
        assert_eq!(template_value["source"], "tracking-template");
        assert_eq!(template_value["frames"].as_array().unwrap().len(), 2);
        assert_eq!(
            template_value["frames"][0]["players"]
                .as_array()
                .unwrap()
                .len(),
            3
        );
        let template_dataset: SoccerTrackingDataset =
            serde_json::from_str(&template.body).expect("tracking template dataset");
        assert!(template_dataset.to_learning_dataset().is_ok());

        let policy = handle_live_soccer_request(
            "GET /api/team-policy HTTP/1.1\r\nHost: local\r\n\r\n",
            &session,
            &input_queue,
        );
        assert_eq!(policy.status, 200);
        let policy_value: serde_json::Value =
            serde_json::from_str(&policy.body).expect("team policy json");
        assert_eq!(policy_value["adversarial"], true);
        assert_eq!(policy_value["summary"]["ticks"], 2);
        assert_eq!(policy_value["learning"]["totalTransitions"], 44);
        assert!(policy_value["homeOptions"].get("alpha").is_some());
        assert!(policy_value["awayOptions"].get("gamma").is_some());
        assert!(!policy_value["homeEntries"].as_array().unwrap().is_empty());
        assert!(!policy_value["awayEntries"].as_array().unwrap().is_empty());
        assert!(!policy_value["homeTargetEntries"]
            .as_array()
            .unwrap()
            .is_empty());
        assert!(!policy_value["awayTargetEntries"]
            .as_array()
            .unwrap()
            .is_empty());

        let import_session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            max_human_players: 2,
            seed: 56,
            ..Default::default()
        })));
        let import_queue = import_session.lock().unwrap().input_queue();
        let import = handle_live_soccer_request(
            &format!(
                "POST /api/team-policy HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                policy.body.len(),
                policy.body
            ),
            &import_session,
            &import_queue,
        );
        assert_eq!(import.status, 200);
        let import_value: serde_json::Value =
            serde_json::from_str(&import.body).expect("policy import json");
        assert_eq!(import_value["learning"]["totalTransitions"], 0);
        assert_eq!(
            import_value["importedHomeEntries"],
            policy_value["homeEntries"].as_array().unwrap().len()
        );
        assert_eq!(
            import_value["importedAwayEntries"],
            policy_value["awayEntries"].as_array().unwrap().len()
        );
        assert_eq!(
            import_value["learning"]["homePolicyEntries"],
            import_value["importedHomeEntries"]
        );
        assert_eq!(
            import_value["learning"]["awayPolicyEntries"],
            import_value["importedAwayEntries"]
        );
        assert!(
            import_value["learning"]["homePolicyTargetEntries"]
                .as_u64()
                .unwrap()
                > 0
        );
        assert!(
            import_value["learning"]["awayPolicyTargetEntries"]
                .as_u64()
                .unwrap()
                > 0
        );
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
    fn live_http_surface_route_updates_ball_physics_config() {
        let session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            seed: 59,
            ..Default::default()
        })));
        let input_queue = session.lock().unwrap().input_queue();

        let body = r#"{"ballDragPerTick":0.045,"ballStopSpeedYps":0.65}"#;
        let surface = handle_live_soccer_request(
            &format!(
                "POST /api/surface HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
            &session,
            &input_queue,
        );
        assert_eq!(surface.status, 200);
        let value: serde_json::Value = serde_json::from_str(&surface.body).expect("surface json");
        assert_eq!(value["config"]["ballDragPerTick"], 0.045);
        assert_eq!(value["config"]["ballStopSpeedYps"], 0.65);
        assert_eq!(
            session
                .lock()
                .unwrap()
                .match_ref()
                .config
                .ball_drag_per_tick,
            0.045
        );

        let invalid = r#"{"ballDragPerTick":1.1,"ballStopSpeedYps":0.65}"#;
        let rejected = handle_live_soccer_request(
            &format!(
                "POST /api/surface HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                invalid.len(),
                invalid
            ),
            &session,
            &input_queue,
        );
        assert_eq!(rejected.status, 400);
    }

    #[test]
    fn live_http_self_play_training_route_runs_and_imports_policy() {
        let session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            seed: 590,
            ..Default::default()
        })));
        let input_queue = session.lock().unwrap().input_queue();
        let body = serde_json::json!({
            "episodes": 1,
            "minutes": 0.01,
            "periodCount": 2,
            "periodBreakRecoverySeconds": 15.0,
            "dtSeconds": 0.2,
            "learningIntervalTicks": 1,
            "seed": 990,
            "options": {
                "alpha": 0.2,
                "gamma": 0.94
            },
            "tacticalLearning": {
                "attackFlankLaneWeight": 0.22,
                "defenseContractDeltaWeight": 0.33
            },
            "importIntoSession": true
        })
        .to_string();

        let response = handle_live_soccer_request(
            &format!(
                "POST /api/train-self-play HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
            &session,
            &input_queue,
        );

        assert_eq!(response.status, 200);
        let value: serde_json::Value =
            serde_json::from_str(&response.body).expect("self-play response json");
        assert_eq!(value["artifact"]["episodes"].as_array().unwrap().len(), 1);
        assert_eq!(
            value["artifact"]["config"]["periodCount"],
            serde_json::json!(2)
        );
        assert_eq!(
            value["artifact"]["config"]["periodBreakRecoverySeconds"],
            serde_json::json!(15.0)
        );
        assert_eq!(
            value["artifact"]["tacticalLearning"]["attackFlankLaneWeight"],
            serde_json::json!(0.22)
        );
        assert_eq!(
            value["artifact"]["config"]["tacticalLearning"]["defenseContractDeltaWeight"],
            serde_json::json!(0.33)
        );
        assert!(value["importedHomeEntries"].as_u64().unwrap() > 0);
        assert!(value["importedAwayEntries"].as_u64().unwrap() > 0);
        assert_eq!(
            value["learning"]["homePolicyEntries"],
            value["importedHomeEntries"]
        );
        assert_eq!(
            session
                .lock()
                .unwrap()
                .match_ref()
                .config
                .tactical_learning
                .attack_flank_lane_weight,
            0.22
        );
    }

    #[test]
    fn live_http_learning_route_updates_runtime_switches() {
        let session = Arc::new(Mutex::new(SoccerRealtimeSession::new(MatchConfig {
            duration_seconds: 1.0,
            seed: 591,
            ..Default::default()
        })));
        let input_queue = session.lock().unwrap().input_queue();

        let body = r#"{"learningEnabled":false,"learningLoggingEnabled":false}"#;
        let response = handle_live_soccer_request(
            &format!(
                "POST /api/learning HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body.len(),
                body
            ),
            &session,
            &input_queue,
        );
        assert_eq!(response.status, 200);
        let value: serde_json::Value =
            serde_json::from_str(&response.body).expect("learning runtime json");
        assert_eq!(value["config"]["learningEnabled"], false);
        assert_eq!(value["config"]["learningLoggingEnabled"], false);
        assert_eq!(value["learning"]["learningEnabled"], false);
        assert_eq!(value["learning"]["learningLoggingEnabled"], false);
        assert!(!session.lock().unwrap().match_ref().config.learning_enabled);
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
        {
            session
                .lock()
                .unwrap()
                .match_mut()
                .assign_controller_slot(0, Some(0))
                .expect("assign human controller");
        }
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
        assert!(input_queue.wait_for_pending_input(Duration::from_millis(200)));
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
                    ball_altitude_yards: Some(0.0),
                    pass_flight: None,
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
                            skills: None,
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
                            skills: None,
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
                            skills: None,
                        },
                    ],
                },
                SoccerTrackingFrame {
                    tick: 1,
                    clock_seconds: 0.1,
                    ball_position: Vec2::new(44.0, 82.0),
                    ball_velocity: Some(Vec2::new(8.0, 16.0)),
                    ball_altitude_yards: Some(0.0),
                    pass_flight: Some(PassFlight::Floor),
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
                            skills: None,
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
                            skills: None,
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
                            skills: None,
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
