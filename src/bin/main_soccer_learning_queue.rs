//! Queue-style soccer self-play runner.
//!
//! Unlike the batch runner, this keeps a fixed number of simulation slots full:
//! when one game finishes, its deltas are merged and the next game starts from
//! the newest available policy.

use std::collections::{HashMap, VecDeque};
use std::error::Error;
use std::fs;
use std::io::{BufWriter, Error as IoError, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use des_engine::des::general::soccer::{
    MatchConfig, SoccerNeuralLearningBackend, SoccerNeuralLearningConfig,
    SoccerNeuralNetworkSnapshot, SoccerQPolicyOptions, SoccerSelfPlayLearnedParams,
    SoccerSelfPlayTrainingArtifact, SoccerTacticalLearningSummary, SoccerTacticalLearningWeights,
    SoccerTeamPolicyArtifact, SoccerTeamQPolicies,
};
use des_engine::des::soccer_learning::{
    evolve_soccer_tactical_learning_weights, evolve_soccer_team_policies,
    run_soccer_learning_queue_with_events, soccer_policy_version_insert_status_after_active_head,
    soccer_postgres_policy_refresh_decision, soccer_self_play_artifact_from_queue_report,
    soccer_should_flush_postgres_policy_versions_for_new_sim,
    soccer_should_refresh_postgres_for_new_sim, SoccerEvolutionOptions,
    SoccerLearningCompletedGame, SoccerLearningQueueEvent, SoccerLearningQueueRunnerConfig,
    SoccerPostgresPolicyRefreshCheck, SOCCER_POLICY_STATUS_ACTIVE,
};
use des_engine::des::soccer_learning_pg::{
    SoccerLearningPgCompletedRunInsert, SoccerLearningPgStore,
};
use serde::Serialize;
use uuid::Uuid;

const DEFAULT_SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES: usize = 10;
const DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES: usize = 10;
const DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE: usize = 16;
const DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES: usize = 16;
const DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS: usize = 2;
const DEFAULT_SOCCER_QUEUE_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE: bool = true;
const DEFAULT_SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT: bool = true;
const DEFAULT_SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM: bool = true;
const DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS: usize = 10;
const DEFAULT_SOCCER_QUEUE_EVOLUTION_ENABLED: bool = true;
const DEFAULT_SOCCER_QUEUE_EVOLUTION_ELITE_GAMES: usize = 4;

#[derive(Clone, Debug)]
struct TacticalEvolutionSample {
    summary: SoccerTacticalLearningSummary,
    fitness: f64,
}

#[derive(Clone, Debug)]
struct PolicyEvolutionSample {
    policies: SoccerTeamQPolicies,
    fitness: f64,
}

#[derive(Clone, Debug)]
struct PendingPostgresCompletedRun {
    completed_game: SoccerLearningCompletedGame,
    base_policy_version_id: Option<String>,
    output_policy_version_id: Option<String>,
    generation: i32,
}

#[derive(Debug)]
struct PendingPostgresPolicyVersion {
    id: String,
    parent_policy_version_id: Option<String>,
    generation: i32,
    version_label: String,
    source_kind: &'static str,
    status: &'static str,
    config: MatchConfig,
    home_options: SoccerQPolicyOptions,
    away_options: SoccerQPolicyOptions,
    policies: SoccerTeamQPolicies,
    fitness: f64,
    neural_network: Option<SoccerNeuralNetworkSnapshot>,
}

struct PostgresCompletedRunBatch {
    experiment_id: String,
    runner_id: String,
    pending_policy_versions: Vec<PendingPostgresPolicyVersion>,
    pending_runs: Vec<PendingPostgresCompletedRun>,
}

impl PostgresCompletedRunBatch {
    fn can_absorb(&self, other: &Self) -> bool {
        self.experiment_id == other.experiment_id && self.runner_id == other.runner_id
    }

    fn absorb(&mut self, mut other: Self) {
        self.pending_policy_versions
            .append(&mut other.pending_policy_versions);
        self.pending_runs.append(&mut other.pending_runs);
    }
}

struct PostgresCompletedRunWriteResult {
    queue_batches: usize,
    result: Result<usize, String>,
}

struct AsyncPostgresCompletedRunWriter {
    sender: Option<mpsc::SyncSender<PostgresCompletedRunBatch>>,
    receiver: mpsc::Receiver<PostgresCompletedRunWriteResult>,
    handle: Option<thread::JoinHandle<()>>,
    pending_batches: usize,
}

fn env_value(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_usize(name: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value.parse::<usize>().map_err(|_| {
        invalid_data(format!("{name} must be an unsigned integer, got {value:?}")).into()
    })
}

fn env_usize_alias(primary: &str, alias: &str, default: usize) -> Result<usize, Box<dyn Error>> {
    let Some(value) = env_value(primary).or_else(|| env_value(alias)) else {
        return Ok(default);
    };
    value.parse::<usize>().map_err(|_| {
        invalid_data(format!(
            "{primary}/{alias} must be an unsigned integer, got {value:?}"
        ))
        .into()
    })
}

fn env_u32(name: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value
        .parse::<u32>()
        .map_err(|_| invalid_data(format!("{name} must be a u32, got {value:?}")).into())
}

fn env_u32_alias(primary: &str, alias: &str, default: u32) -> Result<u32, Box<dyn Error>> {
    let Some(value) = env_value(primary).or_else(|| env_value(alias)) else {
        return Ok(default);
    };
    value
        .parse::<u32>()
        .map_err(|_| invalid_data(format!("{primary}/{alias} must be a u32, got {value:?}")).into())
}

fn env_f64(name: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    let parsed = value
        .parse::<f64>()
        .map_err(|_| invalid_data(format!("{name} must be a finite number, got {value:?}")))?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(invalid_data(format!("{name} must be finite, got {value:?}")).into())
    }
}

fn env_f64_alias(primary: &str, alias: &str, default: f64) -> Result<f64, Box<dyn Error>> {
    let Some(value) = env_value(primary).or_else(|| env_value(alias)) else {
        return Ok(default);
    };
    let parsed = value.parse::<f64>().map_err(|_| {
        invalid_data(format!(
            "{primary}/{alias} must be a finite number, got {value:?}"
        ))
    })?;
    if parsed.is_finite() {
        Ok(parsed)
    } else {
        Err(invalid_data(format!("{primary}/{alias} must be finite, got {value:?}")).into())
    }
}

fn env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => Err(invalid_data(format!("{name} must be a boolean, got {value:?}")).into()),
    }
}

fn env_bool_alias(primary: &str, alias: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    let Some(value) = env_value(primary).or_else(|| env_value(alias)) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => Err(invalid_data(format!(
            "{primary}/{alias} must be a boolean, got {value:?}"
        ))
        .into()),
    }
}

fn env_neural_learning_backend(
    default: SoccerNeuralLearningBackend,
) -> Result<SoccerNeuralLearningBackend, Box<dyn Error>> {
    let Some(value) =
        env_value("SOCCER_NEURAL_LEARNING_BACKEND").or_else(|| env_value("SOCCER_NEURAL_BACKEND"))
    else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "threaded" | "thread" | "worker" => Ok(SoccerNeuralLearningBackend::Threaded),
        "inline" | "sync" => Ok(SoccerNeuralLearningBackend::Inline),
        _ => Err(invalid_data(format!(
            "SOCCER_NEURAL_LEARNING_BACKEND/SOCCER_NEURAL_BACKEND must be inline or threaded, got {value:?}"
        ))
        .into()),
    }
}

fn env_neural_learning_config() -> Result<SoccerNeuralLearningConfig, Box<dyn Error>> {
    let default = SoccerNeuralLearningConfig {
        enabled: true,
        backend: SoccerNeuralLearningBackend::Threaded,
        max_pending_batches: 128,
        ..SoccerNeuralLearningConfig::default()
    };
    Ok(SoccerNeuralLearningConfig {
        enabled: env_bool_alias(
            "SOCCER_NEURAL_LEARNING_ENABLED",
            "SOCCER_NEURAL_LEARNING",
            default.enabled,
        )?,
        backend: env_neural_learning_backend(default.backend)?,
        learning_rate: env_f64_alias(
            "SOCCER_NEURAL_LEARNING_RATE",
            "SOCCER_NEURAL_RATE",
            default.learning_rate,
        )?,
        batch_size: env_usize_alias(
            "SOCCER_NEURAL_BATCH_SIZE",
            "SOCCER_NEURAL_LEARNING_BATCH_SIZE",
            default.batch_size,
        )?,
        train_every_ticks: env_usize_alias(
            "SOCCER_NEURAL_TRAIN_EVERY_TICKS",
            "SOCCER_NEURAL_LEARNING_TRAIN_EVERY_TICKS",
            default.train_every_ticks,
        )?,
        max_batches_per_tick: env_usize_alias(
            "SOCCER_NEURAL_MAX_BATCHES_PER_TICK",
            "SOCCER_NEURAL_LEARNING_MAX_BATCHES_PER_TICK",
            default.max_batches_per_tick,
        )?,
        hidden_units: env_usize_alias(
            "SOCCER_NEURAL_HIDDEN_UNITS",
            "SOCCER_NEURAL_LEARNING_HIDDEN_UNITS",
            default.hidden_units,
        )?,
        target_scale: env_f64_alias(
            "SOCCER_NEURAL_TARGET_SCALE",
            "SOCCER_NEURAL_LEARNING_TARGET_SCALE",
            default.target_scale,
        )?,
        max_pending_batches: env_usize_alias(
            "SOCCER_NEURAL_MAX_PENDING_BATCHES",
            "SOCCER_NEURAL_LEARNING_MAX_PENDING_BATCHES",
            default.max_pending_batches,
        )?,
        replay_capacity: env_usize_alias(
            "SOCCER_NEURAL_REPLAY_CAPACITY",
            "SOCCER_NEURAL_LEARNING_REPLAY_CAPACITY",
            default.replay_capacity,
        )?,
        replay_samples_per_tick: env_usize_alias(
            "SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK",
            "SOCCER_NEURAL_LEARNING_REPLAY_SAMPLES_PER_TICK",
            default.replay_samples_per_tick,
        )?,
        target_clip: env_f64_alias(
            "SOCCER_NEURAL_TARGET_CLIP",
            "SOCCER_NEURAL_LEARNING_TARGET_CLIP",
            default.target_clip,
        )?,
        snapshot_every_batches: env_usize_alias(
            "SOCCER_NEURAL_SNAPSHOT_EVERY_BATCHES",
            "SOCCER_NEURAL_LEARNING_SNAPSHOT_EVERY_BATCHES",
            default.snapshot_every_batches,
        )?,
    })
}

fn env_tactical_learning_weights() -> Result<SoccerTacticalLearningWeights, Box<dyn Error>> {
    let default = SoccerTacticalLearningWeights::default();
    Ok(SoccerTacticalLearningWeights {
        attack_spacing_delta_weight: env_f64(
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            default.attack_spacing_delta_weight,
        )?,
        attack_spacing_score_weight: env_f64(
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            default.attack_spacing_score_weight,
        )?,
        attack_width_delta_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            default.attack_width_delta_weight,
        )?,
        attack_width_score_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            default.attack_width_score_weight,
        )?,
        attack_flank_lane_weight: env_f64(
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            default.attack_flank_lane_weight,
        )?,
        defense_spacing_delta_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            default.defense_spacing_delta_weight,
        )?,
        defense_spacing_score_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            default.defense_spacing_score_weight,
        )?,
        defense_contract_delta_weight: env_f64(
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            default.defense_contract_delta_weight,
        )?,
        defense_compactness_score_weight: env_f64(
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            default.defense_compactness_score_weight,
        )?,
        defense_ball_depth_score_weight: env_f64(
            "SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT",
            default.defense_ball_depth_score_weight,
        )?,
        defense_endline_soft_penalty_weight: env_f64(
            "SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT",
            default.defense_endline_soft_penalty_weight,
        )?,
        defense_endline_hard_penalty_weight: env_f64(
            "SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT",
            default.defense_endline_hard_penalty_weight,
        )?,
        defender_midfielder_press_weight: env_f64(
            "SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT",
            default.defender_midfielder_press_weight,
        )?,
        midfielder_press_weight: env_f64(
            "SOCCER_MIDFIELDER_PRESS_WEIGHT",
            default.midfielder_press_weight,
        )?,
    })
}

fn validate_tactical_learning_weights(
    weights: &SoccerTacticalLearningWeights,
) -> Result<(), Box<dyn Error>> {
    for (name, value) in [
        (
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            weights.attack_spacing_delta_weight,
        ),
        (
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            weights.attack_spacing_score_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            weights.attack_width_delta_weight,
        ),
        (
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            weights.attack_width_score_weight,
        ),
        (
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            weights.attack_flank_lane_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            weights.defense_spacing_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            weights.defense_spacing_score_weight,
        ),
        (
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            weights.defense_contract_delta_weight,
        ),
        (
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            weights.defense_compactness_score_weight,
        ),
        (
            "SOCCER_DEFENSE_BALL_DEPTH_SCORE_WEIGHT",
            weights.defense_ball_depth_score_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_SOFT_PENALTY_WEIGHT",
            weights.defense_endline_soft_penalty_weight,
        ),
        (
            "SOCCER_DEFENSE_ENDLINE_HARD_PENALTY_WEIGHT",
            weights.defense_endline_hard_penalty_weight,
        ),
        (
            "SOCCER_DEFENDER_MIDFIELDER_PRESS_WEIGHT",
            weights.defender_midfielder_press_weight,
        ),
        (
            "SOCCER_MIDFIELDER_PRESS_WEIGHT",
            weights.midfielder_press_weight,
        ),
    ] {
        if !value.is_finite() {
            return Err(invalid_data(format!("{name} must be finite")).into());
        }
    }
    Ok(())
}

fn tactical_learning_weight_values(weights: &SoccerTacticalLearningWeights) -> [f64; 14] {
    [
        weights.attack_spacing_delta_weight,
        weights.attack_spacing_score_weight,
        weights.attack_width_delta_weight,
        weights.attack_width_score_weight,
        weights.attack_flank_lane_weight,
        weights.defense_spacing_delta_weight,
        weights.defense_spacing_score_weight,
        weights.defense_contract_delta_weight,
        weights.defense_compactness_score_weight,
        weights.defense_ball_depth_score_weight,
        weights.defense_endline_soft_penalty_weight,
        weights.defense_endline_hard_penalty_weight,
        weights.defender_midfielder_press_weight,
        weights.midfielder_press_weight,
    ]
}

fn tactical_learning_weights_match(
    left: &SoccerTacticalLearningWeights,
    right: &SoccerTacticalLearningWeights,
) -> bool {
    tactical_learning_weight_values(left)
        .into_iter()
        .zip(tactical_learning_weight_values(right))
        .all(|(left, right)| (left - right).abs() <= 1e-12)
}

fn maybe_apply_postgres_tactical_learning(
    event_label: &str,
    next_episode: usize,
    policy_version_id: &str,
    generation: i32,
    match_config: &mut MatchConfig,
    active_weights: &mut SoccerTacticalLearningWeights,
    postgres_weights: Option<SoccerTacticalLearningWeights>,
) -> Result<bool, String> {
    let Some(postgres_weights) = postgres_weights else {
        return Ok(false);
    };
    validate_tactical_learning_weights(&postgres_weights).map_err(|err| err.to_string())?;
    if tactical_learning_weights_match(active_weights, &postgres_weights)
        && tactical_learning_weights_match(&match_config.tactical_learning, &postgres_weights)
    {
        return Ok(false);
    }
    println!(
        "{} next_episode={} policy_version={} generation={} attack_flank_lane={:.3} defense_contract_delta={:.3}",
        event_label,
        next_episode,
        policy_version_id,
        generation,
        postgres_weights.attack_flank_lane_weight,
        postgres_weights.defense_contract_delta_weight
    );
    match_config.tactical_learning = postgres_weights.clone();
    *active_weights = postgres_weights;
    Ok(true)
}

fn invalid_data(message: impl Into<String>) -> IoError {
    IoError::new(ErrorKind::InvalidData, message.into())
}

fn default_run_id() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    format!("soccer-learning-queue-{seconds}")
}

fn default_postgres_policy_version_interval_games(parallel_games: usize) -> usize {
    parallel_games.max(DEFAULT_SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES)
}

fn default_postgres_completed_run_batch_games(parallel_games: usize) -> usize {
    parallel_games.max(DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES)
}

fn default_queue_evolution_interval_games(parallel_games: usize) -> usize {
    parallel_games.max(10)
}

fn take_episode_starting_policy_version(
    episode_starting_policy_versions: &mut HashMap<usize, (Option<String>, i32)>,
    episode: usize,
    current_policy_version_id: &Option<String>,
    current_generation: i32,
) -> (Option<String>, i32) {
    episode_starting_policy_versions
        .remove(&episode)
        .unwrap_or_else(|| (current_policy_version_id.clone(), current_generation))
}

fn flush_postgres_completed_runs(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    runner_id: &str,
    pending_policy_versions: &mut Vec<PendingPostgresPolicyVersion>,
    pending_runs: &mut Vec<PendingPostgresCompletedRun>,
) -> Result<usize, String> {
    if pending_policy_versions.is_empty() && pending_runs.is_empty() {
        return Ok(0);
    }
    let policy_versions_written = pending_policy_versions.len();
    for policy_version in pending_policy_versions.iter() {
        let latest_active_metadata = if policy_version.status == SOCCER_POLICY_STATUS_ACTIVE {
            store.load_latest_active_policy_metadata(experiment_id)?
        } else {
            None
        };
        let insert_status = soccer_policy_version_insert_status_after_active_head(
            policy_version.status,
            policy_version.parent_policy_version_id.as_deref(),
            policy_version.generation,
            latest_active_metadata
                .as_ref()
                .map(|metadata| metadata.id.as_str()),
            latest_active_metadata
                .as_ref()
                .map(|metadata| metadata.generation),
        );
        if insert_status != policy_version.status {
            println!(
                "postgres_policy_version_marked_stale policy_version={} parent_policy_version={} generation={} latest_active={} latest_generation={}",
                policy_version.id,
                policy_version
                    .parent_policy_version_id
                    .as_deref()
                    .unwrap_or("none"),
                policy_version.generation,
                latest_active_metadata
                    .as_ref()
                    .map(|metadata| metadata.id.as_str())
                    .unwrap_or("none"),
                latest_active_metadata
                    .as_ref()
                    .map(|metadata| metadata.generation.to_string())
                    .unwrap_or_else(|| "none".to_string())
            );
        }
        store.insert_policy_version_with_id_and_neural_network(
            &policy_version.id,
            experiment_id,
            policy_version.parent_policy_version_id.as_deref(),
            policy_version.generation,
            &policy_version.version_label,
            policy_version.source_kind,
            insert_status,
            &policy_version.config,
            policy_version.home_options.clone(),
            policy_version.away_options.clone(),
            &policy_version.policies,
            policy_version.fitness,
            policy_version.neural_network.as_ref(),
        )?;
    }
    if pending_runs.is_empty() {
        pending_policy_versions.clear();
        return Ok(0);
    }

    let inserts = pending_runs
        .iter()
        .map(|pending| SoccerLearningPgCompletedRunInsert {
            base_policy_version_id: pending.base_policy_version_id.as_deref(),
            output_policy_version_id: pending.output_policy_version_id.as_deref(),
            game: &pending.completed_game,
        })
        .collect::<Vec<_>>();
    let batch_size = inserts.len();
    let run_ids = store.insert_completed_runs(experiment_id, runner_id, &inserts)?;
    drop(inserts);
    if let (Some(first), Some(last), Some(first_run_id), Some(last_run_id)) = (
        pending_runs.first(),
        pending_runs.last(),
        run_ids.first(),
        run_ids.last(),
    ) {
        println!(
            "postgres_persisted_batch episodes={}..{} first_run_id={} last_run_id={} first_policy_version={} last_policy_version={} first_generation={} last_generation={} policy_versions_written={} batch_size={}",
            first.completed_game.episode + 1,
            last.completed_game.episode + 1,
            first_run_id,
            last_run_id,
            first
                .output_policy_version_id
                .as_deref()
                .unwrap_or("none"),
            last.output_policy_version_id.as_deref().unwrap_or("none"),
            first.generation,
            last.generation,
            policy_versions_written,
            batch_size,
        );
    }
    let flushed = pending_runs.len();
    pending_policy_versions.clear();
    pending_runs.clear();
    Ok(flushed)
}

fn flush_postgres_policy_versions_for_new_sims(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    runner_id: &str,
    pending_policy_versions: &mut Vec<PendingPostgresPolicyVersion>,
) -> Result<(), String> {
    if pending_policy_versions.is_empty() {
        return Ok(());
    }
    let pending_count = pending_policy_versions.len();
    let mut pending_runs = Vec::new();
    flush_postgres_completed_runs(
        store,
        experiment_id,
        runner_id,
        pending_policy_versions,
        &mut pending_runs,
    )?;
    println!(
        "postgres_policy_versions_flushed_for_new_sims policy_versions_written={pending_count}"
    );
    Ok(())
}

impl AsyncPostgresCompletedRunWriter {
    fn start(queue_batches: usize, coalesce_batches: usize, coalesce_wait: Duration) -> Self {
        let (sender, receiver) =
            mpsc::sync_channel::<PostgresCompletedRunBatch>(queue_batches.max(1));
        let (result_sender, result_receiver) = mpsc::channel::<PostgresCompletedRunWriteResult>();
        let coalesce_batches = coalesce_batches.max(1);
        let handle = thread::spawn(move || {
            let mut store = match SoccerLearningPgStore::connect_from_env() {
                Ok(Some(store)) => store,
                Ok(None) => {
                    while receiver.recv().is_ok() {
                        let _ = result_sender.send(PostgresCompletedRunWriteResult {
                            queue_batches: 1,
                            result: Err(
                                "postgres completed-run writer could not find a database URL"
                                    .to_string(),
                            ),
                        });
                    }
                    return;
                }
                Err(error) => {
                    while receiver.recv().is_ok() {
                        let _ = result_sender.send(PostgresCompletedRunWriteResult {
                            queue_batches: 1,
                            result: Err(format!(
                                "postgres completed-run writer connect failed: {error}"
                            )),
                        });
                    }
                    return;
                }
            };

            let mut deferred_batch = None::<PostgresCompletedRunBatch>;
            loop {
                let mut batch = match deferred_batch.take() {
                    Some(batch) => batch,
                    None => match receiver.recv() {
                        Ok(batch) => batch,
                        Err(_) => break,
                    },
                };
                let mut queue_batches = 1usize;
                let coalesce_started = Instant::now();
                while queue_batches < coalesce_batches {
                    match receiver.try_recv() {
                        Ok(next_batch) => {
                            if batch.can_absorb(&next_batch) {
                                batch.absorb(next_batch);
                                queue_batches = queue_batches.saturating_add(1);
                            } else {
                                deferred_batch = Some(next_batch);
                                break;
                            }
                        }
                        Err(mpsc::TryRecvError::Empty) => {
                            if coalesce_wait.is_zero() {
                                break;
                            }
                            let elapsed = coalesce_started.elapsed();
                            if elapsed >= coalesce_wait {
                                break;
                            }
                            match receiver.recv_timeout(coalesce_wait.saturating_sub(elapsed)) {
                                Ok(next_batch) => {
                                    if batch.can_absorb(&next_batch) {
                                        batch.absorb(next_batch);
                                        queue_batches = queue_batches.saturating_add(1);
                                    } else {
                                        deferred_batch = Some(next_batch);
                                        break;
                                    }
                                }
                                Err(mpsc::RecvTimeoutError::Timeout) => break,
                                Err(mpsc::RecvTimeoutError::Disconnected) => break,
                            }
                        }
                        Err(mpsc::TryRecvError::Disconnected) => break,
                    }
                }
                let result = flush_postgres_completed_runs(
                    &mut store,
                    &batch.experiment_id,
                    &batch.runner_id,
                    &mut batch.pending_policy_versions,
                    &mut batch.pending_runs,
                );
                let _ = result_sender.send(PostgresCompletedRunWriteResult {
                    queue_batches,
                    result,
                });
            }
        });

        AsyncPostgresCompletedRunWriter {
            sender: Some(sender),
            receiver: result_receiver,
            handle: Some(handle),
            pending_batches: 0,
        }
    }

    fn drain_finished(&mut self) -> Result<usize, String> {
        let mut persisted = 0usize;
        loop {
            match self.receiver.try_recv() {
                Ok(write_result) => {
                    self.pending_batches = self
                        .pending_batches
                        .saturating_sub(write_result.queue_batches);
                    persisted = persisted.saturating_add(write_result.result?);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if self.pending_batches > 0 {
                        return Err(
                            "postgres completed-run writer stopped before all batches finished"
                                .to_string(),
                        );
                    }
                    break;
                }
            }
        }
        Ok(persisted)
    }

    fn enqueue(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        pending_policy_versions: &mut Vec<PendingPostgresPolicyVersion>,
        pending_runs: &mut Vec<PendingPostgresCompletedRun>,
    ) -> Result<usize, String> {
        let persisted = self.drain_finished()?;
        if pending_policy_versions.is_empty() && pending_runs.is_empty() {
            return Ok(persisted);
        }
        let Some(sender) = &self.sender else {
            return Err("postgres completed-run writer is closed".to_string());
        };
        let batch = PostgresCompletedRunBatch {
            experiment_id: experiment_id.to_string(),
            runner_id: runner_id.to_string(),
            pending_policy_versions: std::mem::take(pending_policy_versions),
            pending_runs: std::mem::take(pending_runs),
        };
        sender
            .send(batch)
            .map_err(|err| format!("queue postgres completed-run batch: {err}"))?;
        self.pending_batches = self.pending_batches.saturating_add(1);
        Ok(persisted)
    }

    fn finish(mut self) -> Result<usize, String> {
        let mut persisted = self.drain_finished()?;
        let mut first_error = None::<String>;
        self.sender.take();

        while self.pending_batches > 0 {
            match self.receiver.recv() {
                Ok(write_result) => {
                    self.pending_batches = self
                        .pending_batches
                        .saturating_sub(write_result.queue_batches);
                    match write_result.result {
                        Ok(count) => persisted = persisted.saturating_add(count),
                        Err(error) => {
                            if first_error.is_none() {
                                first_error = Some(error);
                            }
                        }
                    }
                }
                Err(error) => {
                    if first_error.is_none() {
                        first_error = Some(format!(
                            "postgres completed-run writer channel closed early: {error}"
                        ));
                    }
                    break;
                }
            }
        }

        if let Some(handle) = self.handle.take() {
            if handle.join().is_err() && first_error.is_none() {
                first_error = Some("postgres completed-run writer thread panicked".to_string());
            }
        }

        if let Some(error) = first_error {
            Err(error)
        } else {
            Ok(persisted)
        }
    }
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(format!(".tmp-{}", std::process::id()));
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = fs::File::create(&tmp_path)?;
        let mut writer = BufWriter::new(file);
        serde_json::to_writer(&mut writer, value)?;
        writer.flush()?;
        let file = writer.into_inner()?;
        file.sync_all()?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path)?;
    if let Some(parent) = path.parent() {
        let _ = fs::File::open(parent).and_then(|dir| dir.sync_all());
    }
    Ok(())
}

fn validate_queue_settings(
    games: usize,
    halves: usize,
    half_minutes: f64,
    minutes: f64,
    period_break_recovery_seconds: f64,
    dt_seconds: f64,
    learning_interval_ticks: usize,
    parallel_games: usize,
    options: &SoccerQPolicyOptions,
) -> Result<(), Box<dyn Error>> {
    if games == 0 {
        return Err(invalid_data("SOCCER_QUEUE_GAMES/SOCCER_GAMES must be at least 1").into());
    }
    if !(1..=8).contains(&halves) {
        return Err(invalid_data("SOCCER_HALVES must be between 1 and 8").into());
    }
    if !half_minutes.is_finite() || half_minutes <= 0.0 || half_minutes > 120.0 {
        return Err(invalid_data("SOCCER_HALF_MINUTES must be finite and in (0, 120]").into());
    }
    if !minutes.is_finite() || minutes <= 0.0 || minutes > 24.0 * 60.0 {
        return Err(invalid_data("SOCCER_MINUTES must be finite and in (0, 1440]").into());
    }
    if env_value("SOCCER_MINUTES").is_some() {
        let expected = half_minutes * halves as f64;
        if (minutes - expected).abs() > 1e-6 {
            return Err(invalid_data(format!(
                "SOCCER_MINUTES ({minutes}) must equal SOCCER_HALF_MINUTES * SOCCER_HALVES ({expected})"
            ))
            .into());
        }
    }
    if !period_break_recovery_seconds.is_finite()
        || !(0.0..=60.0 * 60.0).contains(&period_break_recovery_seconds)
    {
        return Err(invalid_data(
            "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS must be finite and in [0, 3600]",
        )
        .into());
    }
    if !dt_seconds.is_finite() || !(0.01..=5.0).contains(&dt_seconds) {
        return Err(invalid_data("SOCCER_DT_SECONDS must be finite and in [0.01, 5.0]").into());
    }
    if learning_interval_ticks == 0 {
        return Err(invalid_data("SOCCER_LEARNING_INTERVAL_TICKS must be at least 1").into());
    }
    if !(1..=100).contains(&parallel_games) {
        return Err(invalid_data(
            "SOCCER_QUEUE_PARALLEL_GAMES/SOCCER_PARALLEL_GAMES must be between 1 and 100",
        )
        .into());
    }
    if !options.alpha.is_finite() || !(0.0..=1.0).contains(&options.alpha) {
        return Err(invalid_data("SOCCER_ALPHA must be finite and in [0, 1]").into());
    }
    if !options.gamma.is_finite() || !(0.0..=1.0).contains(&options.gamma) {
        return Err(invalid_data("SOCCER_GAMMA must be finite and in [0, 1]").into());
    }
    Ok(())
}

fn load_initial_policies(
    path: Option<&str>,
    options: SoccerQPolicyOptions,
) -> Result<SoccerTeamQPolicies, Box<dyn Error>> {
    let Some(path) = path else {
        return Ok(SoccerTeamQPolicies::new(options));
    };
    let raw = fs::read_to_string(path)?;
    if let Ok(artifact) = serde_json::from_str::<SoccerSelfPlayTrainingArtifact>(&raw) {
        return SoccerTeamQPolicies::from_self_play_artifact(&artifact)
            .map_err(|err| invalid_data(err).into());
    }
    if let Ok(artifact) = serde_json::from_str::<SoccerTeamPolicyArtifact>(&raw) {
        return SoccerTeamQPolicies::from_artifact(&artifact)
            .map_err(|err| invalid_data(err).into());
    }
    Err(invalid_data(format!(
        "resume artifact {path} is neither a self-play nor team-policy artifact"
    ))
    .into())
}

fn run() -> Result<(), Box<dyn Error>> {
    let default_games = env_usize("SOCCER_GAMES", 100)?;
    let games = env_usize("SOCCER_QUEUE_GAMES", default_games)?;
    let default_parallel_games = env_usize("SOCCER_PARALLEL_GAMES", 10)?;
    let parallel_games = env_usize("SOCCER_QUEUE_PARALLEL_GAMES", default_parallel_games)?;
    let halves = env_usize("SOCCER_HALVES", 2)?;
    let half_minutes = env_f64("SOCCER_HALF_MINUTES", 45.0)?;
    let minutes = env_f64("SOCCER_MINUTES", half_minutes * halves as f64)?;
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2)?;
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4)?;
    let seed = env_u32("SOCCER_SEED", 2026)?;
    let neural_learning = env_neural_learning_config()?;
    let neural_drain_timeout_ms = env_usize_alias(
        "SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS",
        "SOCCER_NEURAL_DRAIN_TIMEOUT_MS",
        DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS,
    )?;
    let neural_drain_timeout =
        Duration::from_millis(neural_drain_timeout_ms.min(u64::MAX as usize) as u64);
    let pg_policy_version_interval_games = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES",
        "SOCCER_POSTGRES_POLICY_VERSION_INTERVAL_GAMES",
        default_postgres_policy_version_interval_games(parallel_games),
    )?;
    let pg_completed_run_batch_games = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES",
        "SOCCER_POSTGRES_COMPLETED_RUN_BATCH_GAMES",
        default_postgres_completed_run_batch_games(parallel_games),
    )?;
    let pg_completed_run_async_queue_batches = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE",
        "SOCCER_POSTGRES_ASYNC_BATCH_QUEUE",
        DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE,
    )?;
    let pg_completed_run_async_coalesce_batches = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES",
        "SOCCER_POSTGRES_ASYNC_COALESCE_BATCHES",
        DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES,
    )?;
    let pg_completed_run_async_coalesce_wait_ms = env_usize_alias(
        "SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS",
        "SOCCER_POSTGRES_ASYNC_COALESCE_WAIT_MS",
        DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS,
    )?;
    let pg_tactical_learning_authoritative = env_bool_alias(
        "SOCCER_QUEUE_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE",
        "SOCCER_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE",
        DEFAULT_SOCCER_QUEUE_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE,
    )?;
    let pg_refresh_with_resume_artifact = env_bool_alias(
        "SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT",
        "SOCCER_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT",
        DEFAULT_SOCCER_QUEUE_POSTGRES_REFRESH_WITH_RESUME_ARTIFACT,
    )?;
    let pg_flush_policy_versions_before_new_sim = env_bool_alias(
        "SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM",
        "SOCCER_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM",
        DEFAULT_SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM,
    )?;
    let pg_policy_version_interval_games = pg_policy_version_interval_games.max(1);
    let pg_completed_run_batch_games = pg_completed_run_batch_games.max(1);
    let pg_completed_run_async_queue_batches = pg_completed_run_async_queue_batches.max(1);
    let pg_completed_run_async_coalesce_batches = pg_completed_run_async_coalesce_batches.max(1);
    let pg_completed_run_async_coalesce_wait = Duration::from_millis(
        pg_completed_run_async_coalesce_wait_ms.min(u64::MAX as usize) as u64,
    );
    let evolution_enabled = env_bool_alias(
        "SOCCER_QUEUE_EVOLUTION_ENABLED",
        "SOCCER_EVOLUTION_ENABLED",
        DEFAULT_SOCCER_QUEUE_EVOLUTION_ENABLED,
    )?;
    let evolution_interval_games = env_usize_alias(
        "SOCCER_QUEUE_EVOLUTION_INTERVAL_GAMES",
        "SOCCER_EVOLUTION_INTERVAL_GAMES",
        default_queue_evolution_interval_games(parallel_games),
    )?
    .max(1);
    let evolution_elite_games = env_usize_alias(
        "SOCCER_QUEUE_EVOLUTION_ELITE_GAMES",
        "SOCCER_EVOLUTION_ELITE_GAMES",
        DEFAULT_SOCCER_QUEUE_EVOLUTION_ELITE_GAMES,
    )?
    .max(1);
    let default_evolution_options = SoccerEvolutionOptions::default();
    let evolution_options = SoccerEvolutionOptions {
        mutation_rate: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_MUTATION_RATE",
            "SOCCER_EVOLUTION_MUTATION_RATE",
            default_evolution_options.mutation_rate,
        )?,
        mutation_scale: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_MUTATION_SCALE",
            "SOCCER_EVOLUTION_MUTATION_SCALE",
            default_evolution_options.mutation_scale,
        )?,
        crossover_rate: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_CROSSOVER_RATE",
            "SOCCER_EVOLUTION_CROSSOVER_RATE",
            default_evolution_options.crossover_rate,
        )?,
        exploration_rate: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_EXPLORATION_RATE",
            "SOCCER_EVOLUTION_EXPLORATION_RATE",
            default_evolution_options.exploration_rate,
        )?,
        exploration_scale: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_EXPLORATION_SCALE",
            "SOCCER_EVOLUTION_EXPLORATION_SCALE",
            default_evolution_options.exploration_scale,
        )?,
        elite_weight_floor: env_f64_alias(
            "SOCCER_QUEUE_EVOLUTION_ELITE_WEIGHT_FLOOR",
            "SOCCER_EVOLUTION_ELITE_WEIGHT_FLOOR",
            default_evolution_options.elite_weight_floor,
        )?,
        population_size: env_usize_alias(
            "SOCCER_QUEUE_EVOLUTION_POPULATION_SIZE",
            "SOCCER_EVOLUTION_POPULATION_SIZE",
            default_evolution_options.population_size,
        )?
        .max(1),
        seed: env_u32_alias(
            "SOCCER_QUEUE_EVOLUTION_SEED",
            "SOCCER_EVOLUTION_SEED",
            default_evolution_options.seed as u32,
        )? as u64,
    };
    let options = SoccerQPolicyOptions {
        alpha: env_f64("SOCCER_ALPHA", 0.20)?,
        gamma: env_f64("SOCCER_GAMMA", 0.96)?,
    };
    let period_break_recovery_seconds = env_f64(
        "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS",
        env_f64("SOCCER_HALFTIME_RECOVERY_SECONDS", 900.0)?,
    )?;
    validate_queue_settings(
        games,
        halves,
        half_minutes,
        minutes,
        period_break_recovery_seconds,
        dt_seconds,
        learning_interval_ticks,
        parallel_games,
        &options,
    )?;
    let mut tactical_learning = env_tactical_learning_weights()?;
    validate_tactical_learning_weights(&tactical_learning)?;
    let half_duration_seconds = half_minutes * 60.0;
    let halftime_fatigue_recovery = if half_duration_seconds > 0.0 {
        (period_break_recovery_seconds / half_duration_seconds).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut config = MatchConfig {
        dt_seconds,
        duration_seconds: minutes * 60.0,
        halves: halves as u8,
        half_duration_seconds,
        halftime_fatigue_recovery,
        period_count: halves,
        period_break_recovery_seconds,
        learning_enabled: true,
        learning_logging_enabled: env_bool("SOCCER_LEARNING_LOGGING", false)?,
        learning_interval_ticks,
        tactical_learning: tactical_learning.clone(),
        neural_learning: neural_learning.clone(),
        max_human_players: 0,
        seed,
        ..MatchConfig::default()
    };
    let resume_artifact =
        env_value("SOCCER_RESUME_ARTIFACT").or_else(|| env_value("SOCCER_RESUME_ARTIFACT_PATH"));
    let run_id = env_value("SOCCER_RUN_ID").unwrap_or_else(default_run_id);
    let run_dir = PathBuf::from(
        env_value("SOCCER_RUN_DIR")
            .unwrap_or_else(|| format!("out/soccer-learning-queue-runs/{run_id}")),
    );
    let artifact_path = env_value("SOCCER_ARTIFACT_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("final-policy.json"));
    let learned_params_path = env_value("SOCCER_LEARNED_PARAMS_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("learned-params.json"));
    let mut pg_store = SoccerLearningPgStore::connect_from_env().map_err(invalid_data)?;
    let postgres_required = env_bool("SOCCER_REQUIRE_POSTGRES", false)?;
    if postgres_required && pg_store.is_none() {
        return Err(
            invalid_data("SOCCER_REQUIRE_POSTGRES=true requires SOCCER_DATABASE_URL").into(),
        );
    }
    let mut pg_experiment_id = None::<String>;
    let mut pg_base_policy_version_id = None::<String>;
    let mut pg_last_policy_version_id = None::<String>;
    let mut pg_generation = 0i32;
    let mut pg_base_policy_version_updated_at_micros = 0i64;
    let mut pg_completed_games_seen = 0usize;
    let mut pg_policy_version_buffer = Vec::<PendingPostgresPolicyVersion>::new();
    let mut pg_completed_buffer = Vec::<PendingPostgresCompletedRun>::new();
    let mut pg_episode_starting_policy_versions = HashMap::<usize, (Option<String>, i32)>::new();
    let mut pg_persisted_games = 0usize;

    let mut initial_policies = load_initial_policies(resume_artifact.as_deref(), options.clone())?;
    let pg_refresh_for_new_sims = soccer_should_refresh_postgres_for_new_sim(
        resume_artifact.is_some(),
        pg_refresh_with_resume_artifact,
    );
    let mut initial_neural_network = None::<SoccerNeuralNetworkSnapshot>;
    if let Some(store) = pg_store.as_mut() {
        let experiment_slug =
            env_value("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| "soccer-self-play".to_string());
        let experiment_name =
            env_value("SOCCER_EXPERIMENT_NAME").unwrap_or_else(|| "Soccer self-play".to_string());
        let experiment_id = store
            .ensure_experiment(&experiment_slug, &experiment_name, &config)
            .map_err(invalid_data)?;
        if pg_refresh_for_new_sims {
            if let Some(version) = store
                .load_latest_active_policy(&experiment_id, options.clone(), options.clone())
                .map_err(invalid_data)?
            {
                println!(
                    "postgres_resume_policy experiment={} policy_version={} generation={} neural_network={}",
                    experiment_slug,
                    version.id,
                    version.generation,
                    version.neural_network.is_some()
                );
                initial_policies = version.policies;
                initial_neural_network = version.neural_network;
                maybe_apply_postgres_tactical_learning(
                    "postgres_resume_tactical_learning",
                    1,
                    &version.id,
                    version.generation,
                    &mut config,
                    &mut tactical_learning,
                    version.tactical_learning,
                )
                .map_err(invalid_data)?;
                pg_base_policy_version_id = Some(version.id.clone());
                pg_last_policy_version_id = Some(version.id);
                pg_generation = version.generation;
                pg_base_policy_version_updated_at_micros = version.updated_at_micros;
            }
        }
        pg_experiment_id = Some(experiment_id);
    }
    let mut pg_completed_writer = if pg_store.is_some() {
        Some(AsyncPostgresCompletedRunWriter::start(
            pg_completed_run_async_queue_batches,
            pg_completed_run_async_coalesce_batches,
            pg_completed_run_async_coalesce_wait,
        ))
    } else {
        None
    };

    println!(
        "soccer_learning_queue_start run_id={} games={} parallel_games={} minutes={:.1} dt={:.3}s ticks_per_game={} seed={} neural_enabled={} neural_backend={:?} neural_snapshot_every_batches={} neural_drain_timeout_ms={} postgres_required={} pg_policy_version_interval_games={} pg_completed_run_batch_games={} pg_completed_async={} pg_completed_async_queue_batches={} pg_completed_async_coalesce_batches={} pg_completed_async_coalesce_wait_ms={} pg_tactical_learning_authoritative={} pg_refresh_with_resume_artifact={} pg_flush_policy_versions_before_new_sim={}",
        run_id,
        games,
        parallel_games,
        minutes,
        dt_seconds,
        config.total_ticks(),
        seed,
        neural_learning.enabled,
        neural_learning.backend,
        neural_learning.snapshot_every_batches,
        neural_drain_timeout_ms,
        postgres_required,
        pg_policy_version_interval_games,
        pg_completed_run_batch_games,
        pg_completed_writer.is_some(),
        pg_completed_run_async_queue_batches,
        pg_completed_run_async_coalesce_batches,
        pg_completed_run_async_coalesce_wait_ms,
        pg_tactical_learning_authoritative,
        pg_refresh_with_resume_artifact,
        pg_flush_policy_versions_before_new_sim,
    );
    println!(
        "queue_evolution enabled={} interval_games={} elite_games={} mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} exploration_rate={:.4} exploration_scale={:.4} elite_weight_floor={:.4} population_size={} seed={}",
        evolution_enabled,
        evolution_interval_games,
        evolution_elite_games,
        evolution_options.mutation_rate,
        evolution_options.mutation_scale,
        evolution_options.crossover_rate,
        evolution_options.exploration_rate,
        evolution_options.exploration_scale,
        evolution_options.elite_weight_floor,
        evolution_options.population_size,
        evolution_options.seed
    );
    if let Some(path) = &resume_artifact {
        println!("resume_artifact={path}");
    }

    let mut active_config = config.clone();
    let mut queue_completed_games_seen = 0usize;
    let mut local_tactical_evolved_since_pg_refresh = false;
    let tactical_evolution_window_games = evolution_interval_games.max(evolution_elite_games);
    let mut tactical_evolution_samples =
        VecDeque::<TacticalEvolutionSample>::with_capacity(tactical_evolution_window_games);
    let mut policy_evolution_samples =
        VecDeque::<PolicyEvolutionSample>::with_capacity(tactical_evolution_window_games);
    let report = run_soccer_learning_queue_with_events(
        SoccerLearningQueueRunnerConfig {
            games,
            parallel_games,
            base_seed: seed,
            match_config: config.clone(),
            initial_neural_network: initial_neural_network.clone(),
            neural_drain_timeout,
            options: options.clone(),
            prune_action_entries_per_team: env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0)?,
            prune_target_entries_per_team: env_usize(
                "SOCCER_MAX_POLICY_TARGET_ENTRIES_PER_TEAM",
                env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0)?,
            )?,
            min_policy_visits: env_u32("SOCCER_MIN_POLICY_VISITS", 0)?,
        },
        initial_policies,
        |event| {
            match event {
                SoccerLearningQueueEvent::StartingBatch {
                    next_episode,
                    match_config,
                    policies: starting_policies,
                    neural_network,
                } => {
                    if pg_refresh_for_new_sims {
                        let pending_async_pg_batches =
                            if let Some(writer) = pg_completed_writer.as_mut() {
                                pg_persisted_games += writer.drain_finished()?;
                                writer.pending_batches
                            } else {
                                0
                            };
                        if let (Some(experiment_id), Some(store)) =
                            (pg_experiment_id.as_deref(), pg_store.as_mut())
                        {
                            if soccer_should_flush_postgres_policy_versions_for_new_sim(
                                pg_refresh_for_new_sims,
                                pg_flush_policy_versions_before_new_sim,
                                pg_policy_version_buffer.len(),
                            ) {
                                flush_postgres_policy_versions_for_new_sims(
                                    store,
                                    experiment_id,
                                    &run_id,
                                    &mut pg_policy_version_buffer,
                                )?;
                            }
                            if let Some(metadata) =
                                store.load_latest_active_policy_metadata(experiment_id)?
                            {
                                let refresh_decision =
                                    soccer_postgres_policy_refresh_decision(
                                        SoccerPostgresPolicyRefreshCheck {
                                            current_policy_version_id: pg_base_policy_version_id
                                                .as_deref(),
                                            current_generation: pg_generation,
                                            current_updated_at_micros:
                                                pg_base_policy_version_updated_at_micros,
                                            current_neural_network_present: neural_network.is_some(),
                                            latest_policy_version_id: &metadata.id,
                                            latest_generation: metadata.generation,
                                            latest_updated_at_micros: metadata.updated_at_micros,
                                            latest_neural_network_present: metadata
                                                .neural_network
                                                .is_some(),
                                            local_tactical_evolved_since_pg_refresh,
                                            postgres_tactical_learning_authoritative:
                                                pg_tactical_learning_authoritative,
                                        },
                                    );
                                if refresh_decision.apply_tactical_learning {
                                    if maybe_apply_postgres_tactical_learning(
                                        "postgres_refresh_tactical_learning_for_queue",
                                        next_episode + 1,
                                        &metadata.id,
                                        metadata.generation,
                                        match_config,
                                        &mut tactical_learning,
                                        metadata.tactical_learning.clone(),
                                    )? {
                                        local_tactical_evolved_since_pg_refresh = false;
                                    }
                                    active_config = match_config.clone();
                                }
                                if refresh_decision.refresh_policy {
                                    if let Some(version) = store.load_latest_active_policy(
                                        experiment_id,
                                        options.clone(),
                                        options.clone(),
                                    )? {
                                        println!(
                                            "postgres_refresh_policy_for_queue next_episode={} policy_version={} previous_policy_version={} generation={} neural_network={} pending_policy_versions={} pending_async_batches={}",
                                            next_episode + 1,
                                            version.id,
                                            pg_base_policy_version_id
                                                .as_deref()
                                                .unwrap_or("none"),
                                            version.generation,
                                            version.neural_network.is_some(),
                                            pg_policy_version_buffer.len(),
                                            pending_async_pg_batches
                                        );
                                        *starting_policies = version.policies;
                                        *neural_network = version.neural_network;
                                        pg_base_policy_version_id = Some(version.id.clone());
                                        pg_last_policy_version_id = Some(version.id);
                                        pg_generation = version.generation;
                                        pg_base_policy_version_updated_at_micros =
                                            version.updated_at_micros;
                                        local_tactical_evolved_since_pg_refresh = false;
                                    }
                                }
                            }
                        }
                    }
                    if !tactical_learning_weights_match(
                        &match_config.tactical_learning,
                        &tactical_learning,
                    ) {
                        match_config.tactical_learning = tactical_learning.clone();
                        active_config = match_config.clone();
                    }
                    if pg_experiment_id.is_some() {
                        pg_episode_starting_policy_versions.insert(
                            next_episode,
                            (pg_base_policy_version_id.clone(), pg_generation),
                        );
                    }
                    Ok(())
                }
                SoccerLearningQueueEvent::CompletedGame {
                    game,
                    merged_policies,
                } => {
                    queue_completed_games_seen = queue_completed_games_seen.saturating_add(1);
                    let game_fitness = game.score.match_fitness;
                    tactical_evolution_samples.push_back(TacticalEvolutionSample {
                        summary: game.tactical_summary.clone(),
                        fitness: game_fitness,
                    });
                    policy_evolution_samples.push_back(PolicyEvolutionSample {
                        policies: game.policies.clone(),
                        fitness: game_fitness,
                    });
                    while tactical_evolution_samples.len() > tactical_evolution_window_games {
                        tactical_evolution_samples.pop_front();
                    }
                    while policy_evolution_samples.len() > tactical_evolution_window_games {
                        policy_evolution_samples.pop_front();
                    }
                    let should_evolve_tactical = evolution_enabled
                        && (evolution_interval_games <= 1
                            || queue_completed_games_seen >= games
                            || queue_completed_games_seen % evolution_interval_games == 0);
                    let mut policy_evolved_fitness = None::<f64>;
                    if should_evolve_tactical && !policy_evolution_samples.is_empty() {
                        let mut ranked_policy_samples = policy_evolution_samples
                            .iter()
                            .enumerate()
                            .filter(|(_, sample)| sample.fitness.is_finite())
                            .map(|(sample_index, sample)| (sample_index, sample.fitness))
                            .collect::<Vec<_>>();
                        ranked_policy_samples.sort_by(|left, right| {
                            right
                                .1
                                .partial_cmp(&left.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if !ranked_policy_samples.is_empty() {
                            let elite_count = evolution_elite_games
                                .min(ranked_policy_samples.len())
                                .max(1);
                            let best_fitness = ranked_policy_samples
                                .first()
                                .map(|(_, fitness)| *fitness)
                                .unwrap_or(0.0);
                            let mut queue_policy_evolution_options = evolution_options;
                            queue_policy_evolution_options.seed = queue_policy_evolution_options
                                .seed
                                .wrapping_add(queue_completed_games_seen as u64)
                                .wrapping_add((game.episode as u64) << 32)
                                .wrapping_add(0x9e37_79b9_7f4a_7c15);
                            let evolved_policies = {
                                let mut parents = Vec::with_capacity(elite_count + 1);
                                parents.push((&*merged_policies, best_fitness));
                                for (sample_index, fitness) in
                                    ranked_policy_samples.iter().take(elite_count)
                                {
                                    parents
                                        .push((&policy_evolution_samples[*sample_index].policies, *fitness));
                                }
                                evolve_soccer_team_policies(&parents, queue_policy_evolution_options)
                                    .map_err(|err| err.to_string())?
                            };
                            *merged_policies = evolved_policies;
                            policy_evolved_fitness = Some(best_fitness);
                            println!(
                                "queue_policy_evolved completed_games={} elite_games={} best_fitness={:.4} mutation_rate={:.4} mutation_scale={:.4} crossover_rate={:.4} exploration_rate={:.4} exploration_scale={:.4} population_size={}",
                                queue_completed_games_seen,
                                elite_count,
                                best_fitness,
                                queue_policy_evolution_options.mutation_rate,
                                queue_policy_evolution_options.mutation_scale,
                                queue_policy_evolution_options.crossover_rate,
                                queue_policy_evolution_options.exploration_rate,
                                queue_policy_evolution_options.exploration_scale,
                                queue_policy_evolution_options.population_size
                            );
                        }
                    }
                    if should_evolve_tactical && !tactical_evolution_samples.is_empty() {
                        let mut ranked_samples = tactical_evolution_samples
                            .iter()
                            .enumerate()
                            .filter(|(_, sample)| sample.fitness.is_finite())
                            .map(|(sample_index, sample)| (sample_index, sample.fitness))
                            .collect::<Vec<_>>();
                        ranked_samples.sort_by(|left, right| {
                            right
                                .1
                                .partial_cmp(&left.1)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        });
                        if !ranked_samples.is_empty() {
                            let elite_count =
                                evolution_elite_games.min(ranked_samples.len()).max(1);
                            let best_fitness = ranked_samples
                                .first()
                                .map(|(_, fitness)| *fitness)
                                .unwrap_or(0.0);
                            let tactical_parents = ranked_samples
                                .iter()
                                .take(elite_count)
                                .map(|(sample_index, fitness)| {
                                    (&tactical_evolution_samples[*sample_index].summary, *fitness)
                                })
                                .collect::<Vec<_>>();
                            let mut queue_evolution_options = evolution_options;
                            queue_evolution_options.seed = queue_evolution_options
                                .seed
                                .wrapping_add(queue_completed_games_seen as u64)
                                .wrapping_add((game.episode as u64) << 32);
                            let previous_tactical_learning = tactical_learning.clone();
                            tactical_learning = evolve_soccer_tactical_learning_weights(
                                &tactical_learning,
                                &tactical_parents,
                                queue_evolution_options,
                            );
                            validate_tactical_learning_weights(&tactical_learning)
                                .map_err(|err| err.to_string())?;
                            active_config.tactical_learning = tactical_learning.clone();
                            local_tactical_evolved_since_pg_refresh = true;
                            println!(
                                "queue_tactical_weights_evolved completed_games={} elite_games={} best_fitness={:.4} population_size={} attack_width_delta={:.3}->{:.3} attack_flank_lane={:.3}->{:.3} defense_contract_delta={:.3}->{:.3}",
                                queue_completed_games_seen,
                                elite_count,
                                best_fitness,
                                queue_evolution_options.population_size,
                                previous_tactical_learning.attack_width_delta_weight,
                                tactical_learning.attack_width_delta_weight,
                                previous_tactical_learning.attack_flank_lane_weight,
                                tactical_learning.attack_flank_lane_weight,
                                previous_tactical_learning.defense_contract_delta_weight,
                                tactical_learning.defense_contract_delta_weight
                            );
                        }
                    }
                    let Some(experiment_id) = pg_experiment_id.as_deref() else {
                        return Ok(());
                    };
                    pg_completed_games_seen = pg_completed_games_seen.saturating_add(1);
                    let (pg_batch_base_policy_version_id, pg_batch_base_policy_generation) =
                        take_episode_starting_policy_version(
                            &mut pg_episode_starting_policy_versions,
                            game.episode,
                            &pg_base_policy_version_id,
                            pg_generation,
                        );
                    let should_write_policy_version = policy_evolved_fitness.is_some()
                        || pg_policy_version_interval_games <= 1
                        || pg_completed_games_seen >= games
                        || pg_completed_games_seen % pg_policy_version_interval_games == 0;
                    let output_policy_version_id = if should_write_policy_version {
                        let next_generation = pg_generation
                            .max(pg_batch_base_policy_generation)
                            .saturating_add(1);
                        let version_label = format!("{}-episode-{:06}", run_id, game.episode + 1);
                        let output_policy_version_id = Uuid::new_v4().to_string();
                        pg_policy_version_buffer.push(PendingPostgresPolicyVersion {
                            id: output_policy_version_id.clone(),
                            parent_policy_version_id: pg_batch_base_policy_version_id.clone(),
                            generation: next_generation,
                            version_label,
                            source_kind: if policy_evolved_fitness.is_some() {
                                "mutation"
                            } else {
                                "merge"
                            },
                            status: "active",
                            config: active_config.clone(),
                            home_options: options.clone(),
                            away_options: options.clone(),
                            policies: (*merged_policies).clone(),
                            fitness: policy_evolved_fitness.unwrap_or(game.score.match_fitness),
                            neural_network: game.neural_network.clone(),
                        });
                        pg_base_policy_version_id = Some(output_policy_version_id.clone());
                        pg_last_policy_version_id = Some(output_policy_version_id.clone());
                        pg_generation = next_generation;
                        pg_base_policy_version_updated_at_micros = 0;
                        Some(output_policy_version_id)
                    } else {
                        pg_last_policy_version_id.clone()
                    };
                    pg_completed_buffer.push(PendingPostgresCompletedRun {
                        completed_game: game.clone(),
                        base_policy_version_id: pg_batch_base_policy_version_id,
                        output_policy_version_id,
                        generation: pg_generation,
                    });
                    if pg_completed_buffer.len() >= pg_completed_run_batch_games {
                        if soccer_should_flush_postgres_policy_versions_for_new_sim(
                            pg_refresh_for_new_sims,
                            pg_flush_policy_versions_before_new_sim,
                            pg_policy_version_buffer.len(),
                        ) {
                            if let Some(store) = pg_store.as_mut() {
                                flush_postgres_policy_versions_for_new_sims(
                                    store,
                                    experiment_id,
                                    &run_id,
                                    &mut pg_policy_version_buffer,
                                )?;
                            }
                        }
                        pg_persisted_games += if let Some(writer) = pg_completed_writer.as_mut() {
                            writer.enqueue(
                                experiment_id,
                                &run_id,
                                &mut pg_policy_version_buffer,
                                &mut pg_completed_buffer,
                            )?
                        } else if let Some(store) = pg_store.as_mut() {
                            flush_postgres_completed_runs(
                                store,
                                experiment_id,
                                &run_id,
                                &mut pg_policy_version_buffer,
                                &mut pg_completed_buffer,
                            )?
                        } else {
                            0
                        };
                    }
                    Ok(())
                }
            }
        },
    )
    .map_err(invalid_data)?;

    if let Some(experiment_id) = pg_experiment_id.as_deref() {
        if soccer_should_flush_postgres_policy_versions_for_new_sim(
            pg_refresh_for_new_sims,
            pg_flush_policy_versions_before_new_sim,
            pg_policy_version_buffer.len(),
        ) {
            if let Some(store) = pg_store.as_mut() {
                flush_postgres_policy_versions_for_new_sims(
                    store,
                    experiment_id,
                    &run_id,
                    &mut pg_policy_version_buffer,
                )
                .map_err(invalid_data)?;
            }
        }
        pg_persisted_games += if let Some(writer) = pg_completed_writer.as_mut() {
            writer
                .enqueue(
                    experiment_id,
                    &run_id,
                    &mut pg_policy_version_buffer,
                    &mut pg_completed_buffer,
                )
                .map_err(invalid_data)?
        } else if let Some(store) = pg_store.as_mut() {
            flush_postgres_completed_runs(
                store,
                experiment_id,
                &run_id,
                &mut pg_policy_version_buffer,
                &mut pg_completed_buffer,
            )
            .map_err(invalid_data)?
        } else {
            0
        };
    }
    if let Some(writer) = pg_completed_writer {
        pg_persisted_games += writer.finish().map_err(invalid_data)?;
    }

    let artifact = soccer_self_play_artifact_from_queue_report(active_config, options, &report);
    write_json(&artifact_path, &artifact)?;
    let learned_params = SoccerSelfPlayLearnedParams::from_training_artifact_with_neural_network(
        &artifact,
        report.latest_neural_network.clone(),
    );
    write_json(&learned_params_path, &learned_params)?;

    println!(
        "soccer_learning_queue_done completed={} failed={} elapsed={:.2}s home_goals={} away_goals={} policy_entries={} target_entries={} postgres_persisted_games={}",
        report.completed_games,
        report.failed_games,
        report.elapsed_seconds,
        report.total_home_goals,
        report.total_away_goals,
        report.final_policy_entries,
        report.final_target_entries,
        pg_persisted_games,
    );
    println!("run_dir={}", run_dir.display());
    println!("artifact={}", artifact_path.display());
    println!("learned_params={}", learned_params_path.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("soccer learning queue failed: {error}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn empty_pg_batch(experiment_id: &str, runner_id: &str) -> PostgresCompletedRunBatch {
        PostgresCompletedRunBatch {
            experiment_id: experiment_id.to_string(),
            runner_id: runner_id.to_string(),
            pending_policy_versions: Vec::new(),
            pending_runs: Vec::new(),
        }
    }

    #[test]
    fn default_queue_postgres_policy_versions_are_batched() {
        assert_eq!(default_postgres_policy_version_interval_games(0), 10);
        assert_eq!(default_postgres_policy_version_interval_games(1), 10);
        assert_eq!(default_postgres_policy_version_interval_games(4), 10);
    }

    #[test]
    fn default_queue_postgres_batches_respect_parallelism() {
        assert_eq!(default_postgres_completed_run_batch_games(4), 10);
        assert_eq!(default_postgres_completed_run_batch_games(16), 16);
        assert_eq!(default_postgres_completed_run_batch_games(64), 64);
    }

    #[test]
    fn default_queue_neural_drain_timeout_keeps_worker_wait_bounded() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS, 10);
    }

    #[test]
    fn default_queue_evolution_interval_respects_parallelism() {
        assert_eq!(default_queue_evolution_interval_games(0), 10);
        assert_eq!(default_queue_evolution_interval_games(4), 10);
        assert_eq!(default_queue_evolution_interval_games(16), 16);
    }

    #[test]
    fn default_queue_postgres_async_writer_stays_bounded() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE, 16);
    }

    #[test]
    fn default_queue_postgres_async_writer_coalesces_io_batches() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_BATCHES, 16);
    }

    #[test]
    fn default_queue_postgres_async_writer_waits_briefly_to_coalesce_io() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_COALESCE_WAIT_MS, 2);
    }

    #[test]
    fn default_queue_postgres_tactical_learning_is_authoritative() {
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_TACTICAL_LEARNING_AUTHORITATIVE);
    }

    #[test]
    fn default_queue_postgres_policy_heads_flush_before_new_sims() {
        assert!(DEFAULT_SOCCER_QUEUE_POSTGRES_FLUSH_POLICY_VERSIONS_BEFORE_NEW_SIM);
    }

    #[test]
    fn queue_completed_game_uses_episode_starting_policy_snapshot() {
        let mut episode_versions = HashMap::new();
        episode_versions.insert(7, (Some("episode-v7".to_string()), 4));
        let current_policy_version_id = Some("current-v9".to_string());

        let recorded = take_episode_starting_policy_version(
            &mut episode_versions,
            7,
            &current_policy_version_id,
            9,
        );
        let fallback = take_episode_starting_policy_version(
            &mut episode_versions,
            8,
            &current_policy_version_id,
            9,
        );

        assert_eq!(recorded, (Some("episode-v7".to_string()), 4));
        assert_eq!(fallback, (Some("current-v9".to_string()), 9));
    }

    #[test]
    fn queue_postgres_batches_only_coalesce_for_same_run() {
        let mut batch = empty_pg_batch("experiment-a", "runner-a");

        assert!(batch.can_absorb(&empty_pg_batch("experiment-a", "runner-a")));
        assert!(!batch.can_absorb(&empty_pg_batch("experiment-b", "runner-a")));
        assert!(!batch.can_absorb(&empty_pg_batch("experiment-a", "runner-b")));

        batch.absorb(empty_pg_batch("experiment-a", "runner-a"));
        assert!(batch.pending_policy_versions.is_empty());
        assert!(batch.pending_runs.is_empty());
    }
}
