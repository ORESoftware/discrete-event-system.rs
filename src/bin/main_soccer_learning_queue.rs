//! Queue-style soccer self-play runner.
//!
//! Unlike the batch runner, this keeps a fixed number of simulation slots full:
//! when one game finishes, its deltas are merged and the next game starts from
//! the newest available policy.

use std::error::Error;
use std::fs;
use std::io::{BufWriter, Error as IoError, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use des_engine::des::general::soccer::{
    MatchConfig, SoccerNeuralLearningBackend, SoccerNeuralLearningConfig, SoccerQPolicyOptions,
    SoccerSelfPlayLearnedParams, SoccerSelfPlayTrainingArtifact, SoccerTacticalLearningWeights,
    SoccerTeamPolicyArtifact, SoccerTeamQPolicies,
};
use des_engine::des::soccer_learning::{
    run_soccer_learning_queue_with_observer, soccer_self_play_artifact_from_queue_report,
    SoccerLearningCompletedGame, SoccerLearningQueueRunnerConfig,
};
use des_engine::des::soccer_learning_pg::{
    SoccerLearningPgCompletedRunInsert, SoccerLearningPgStore,
};
use serde::Serialize;

const DEFAULT_SOCCER_QUEUE_POSTGRES_POLICY_VERSION_INTERVAL_GAMES: usize = 10;
const DEFAULT_SOCCER_QUEUE_POSTGRES_COMPLETED_RUN_BATCH_GAMES: usize = 10;
const DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE: usize = 4;
const DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS: usize = 0;

#[derive(Clone, Debug)]
struct PendingPostgresCompletedRun {
    completed_game: SoccerLearningCompletedGame,
    base_policy_version_id: Option<String>,
    output_policy_version_id: Option<String>,
    generation: i32,
    policy_version_written: bool,
}

struct PostgresCompletedRunBatch {
    experiment_id: String,
    runner_id: String,
    pending_runs: Vec<PendingPostgresCompletedRun>,
}

struct AsyncPostgresCompletedRunWriter {
    sender: Option<mpsc::SyncSender<PostgresCompletedRunBatch>>,
    receiver: mpsc::Receiver<Result<usize, String>>,
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
    })
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

fn flush_postgres_completed_runs(
    store: &mut SoccerLearningPgStore,
    experiment_id: &str,
    runner_id: &str,
    pending_runs: &mut Vec<PendingPostgresCompletedRun>,
) -> Result<usize, String> {
    if pending_runs.is_empty() {
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
    for (pending, run_id) in pending_runs.iter().zip(run_ids.iter()) {
        println!(
            "postgres_persisted_game episode={} run_id={} policy_version={} generation={} policy_version_written={} batch_size={}",
            pending.completed_game.episode + 1,
            run_id,
            pending
                .output_policy_version_id
                .as_deref()
                .unwrap_or("none"),
            pending.generation,
            pending.policy_version_written,
            batch_size,
        );
    }
    let flushed = pending_runs.len();
    pending_runs.clear();
    Ok(flushed)
}

impl AsyncPostgresCompletedRunWriter {
    fn start(queue_batches: usize) -> Self {
        let (sender, receiver) =
            mpsc::sync_channel::<PostgresCompletedRunBatch>(queue_batches.max(1));
        let (result_sender, result_receiver) = mpsc::channel::<Result<usize, String>>();
        let handle = thread::spawn(move || {
            let mut store = match SoccerLearningPgStore::connect_from_env() {
                Ok(Some(store)) => store,
                Ok(None) => {
                    while receiver.recv().is_ok() {
                        let _ = result_sender.send(Err(
                            "postgres completed-run writer could not find a database URL"
                                .to_string(),
                        ));
                    }
                    return;
                }
                Err(error) => {
                    while receiver.recv().is_ok() {
                        let _ = result_sender.send(Err(format!(
                            "postgres completed-run writer connect failed: {error}"
                        )));
                    }
                    return;
                }
            };

            while let Ok(mut batch) = receiver.recv() {
                let result = flush_postgres_completed_runs(
                    &mut store,
                    &batch.experiment_id,
                    &batch.runner_id,
                    &mut batch.pending_runs,
                );
                let _ = result_sender.send(result);
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
                Ok(result) => {
                    self.pending_batches = self.pending_batches.saturating_sub(1);
                    persisted = persisted.saturating_add(result?);
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
        pending_runs: &mut Vec<PendingPostgresCompletedRun>,
    ) -> Result<usize, String> {
        let persisted = self.drain_finished()?;
        if pending_runs.is_empty() {
            return Ok(persisted);
        }
        let Some(sender) = &self.sender else {
            return Err("postgres completed-run writer is closed".to_string());
        };
        let batch = PostgresCompletedRunBatch {
            experiment_id: experiment_id.to_string(),
            runner_id: runner_id.to_string(),
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
                Ok(result) => {
                    self.pending_batches = self.pending_batches.saturating_sub(1);
                    match result {
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
    let pg_policy_version_interval_games = pg_policy_version_interval_games.max(1);
    let pg_completed_run_batch_games = pg_completed_run_batch_games.max(1);
    let pg_completed_run_async_queue_batches = pg_completed_run_async_queue_batches.max(1);
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
    let tactical_learning = env_tactical_learning_weights()?;
    let half_duration_seconds = half_minutes * 60.0;
    let halftime_fatigue_recovery = if half_duration_seconds > 0.0 {
        (period_break_recovery_seconds / half_duration_seconds).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let config = MatchConfig {
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
        tactical_learning,
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
    let mut pg_experiment_id = None::<String>;
    let mut pg_base_policy_version_id = None::<String>;
    let mut pg_last_policy_version_id = None::<String>;
    let mut pg_generation = 0i32;
    let mut pg_completed_games_seen = 0usize;
    let mut pg_completed_buffer = Vec::<PendingPostgresCompletedRun>::new();
    let mut pg_persisted_games = 0usize;

    let mut initial_policies = load_initial_policies(resume_artifact.as_deref(), options.clone())?;
    if let Some(store) = pg_store.as_mut() {
        let experiment_slug =
            env_value("SOCCER_EXPERIMENT_SLUG").unwrap_or_else(|| "soccer-self-play".to_string());
        let experiment_name =
            env_value("SOCCER_EXPERIMENT_NAME").unwrap_or_else(|| "Soccer self-play".to_string());
        let experiment_id = store
            .ensure_experiment(&experiment_slug, &experiment_name, &config)
            .map_err(invalid_data)?;
        if resume_artifact.is_none() {
            if let Some(version) = store
                .load_latest_active_policy(&experiment_id, options.clone(), options.clone())
                .map_err(invalid_data)?
            {
                println!(
                    "postgres_resume_policy experiment={} policy_version={} generation={}",
                    experiment_slug, version.id, version.generation
                );
                initial_policies = version.policies;
                pg_base_policy_version_id = Some(version.id.clone());
                pg_last_policy_version_id = Some(version.id);
                pg_generation = version.generation;
            }
        }
        pg_experiment_id = Some(experiment_id);
    }
    let mut pg_completed_writer = if pg_store.is_some() {
        Some(AsyncPostgresCompletedRunWriter::start(
            pg_completed_run_async_queue_batches,
        ))
    } else {
        None
    };

    println!(
        "soccer_learning_queue_start run_id={} games={} parallel_games={} minutes={:.1} dt={:.3}s ticks_per_game={} seed={} neural_enabled={} neural_backend={:?} neural_drain_timeout_ms={} pg_policy_version_interval_games={} pg_completed_run_batch_games={} pg_completed_async={} pg_completed_async_queue_batches={}",
        run_id,
        games,
        parallel_games,
        minutes,
        dt_seconds,
        config.total_ticks(),
        seed,
        neural_learning.enabled,
        neural_learning.backend,
        neural_drain_timeout_ms,
        pg_policy_version_interval_games,
        pg_completed_run_batch_games,
        pg_completed_writer.is_some(),
        pg_completed_run_async_queue_batches,
    );
    if let Some(path) = &resume_artifact {
        println!("resume_artifact={path}");
    }

    let report = run_soccer_learning_queue_with_observer(
        SoccerLearningQueueRunnerConfig {
            games,
            parallel_games,
            base_seed: seed,
            match_config: config.clone(),
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
        |game, merged_policies| {
            let Some(store) = pg_store.as_mut() else {
                return Ok(());
            };
            let Some(experiment_id) = pg_experiment_id.as_deref() else {
                return Ok(());
            };
            pg_completed_games_seen = pg_completed_games_seen.saturating_add(1);
            let pg_batch_base_policy_version_id = pg_base_policy_version_id.clone();
            let should_write_policy_version = pg_policy_version_interval_games <= 1
                || pg_completed_games_seen >= games
                || pg_completed_games_seen % pg_policy_version_interval_games == 0;
            let output_policy_version_id = if should_write_policy_version {
                let next_generation = pg_generation.saturating_add(1);
                let version_label = format!("{}-episode-{:06}", run_id, game.episode + 1);
                let output_policy_version_id = store.insert_policy_version(
                    experiment_id,
                    pg_batch_base_policy_version_id.as_deref(),
                    next_generation,
                    &version_label,
                    "merge",
                    "active",
                    &config,
                    options.clone(),
                    options.clone(),
                    merged_policies,
                    game.score.match_fitness,
                )?;
                pg_base_policy_version_id = Some(output_policy_version_id.clone());
                pg_last_policy_version_id = Some(output_policy_version_id.clone());
                pg_generation = next_generation;
                Some(output_policy_version_id)
            } else {
                pg_last_policy_version_id.clone()
            };
            pg_completed_buffer.push(PendingPostgresCompletedRun {
                completed_game: game.clone(),
                base_policy_version_id: pg_batch_base_policy_version_id,
                output_policy_version_id,
                generation: pg_generation,
                policy_version_written: should_write_policy_version,
            });
            if pg_completed_buffer.len() >= pg_completed_run_batch_games {
                pg_persisted_games += if let Some(writer) = pg_completed_writer.as_mut() {
                    writer.enqueue(experiment_id, &run_id, &mut pg_completed_buffer)?
                } else {
                    flush_postgres_completed_runs(
                        store,
                        experiment_id,
                        &run_id,
                        &mut pg_completed_buffer,
                    )?
                };
            }
            Ok(())
        },
    )
    .map_err(invalid_data)?;

    if let Some(experiment_id) = pg_experiment_id.as_deref() {
        pg_persisted_games += if let Some(writer) = pg_completed_writer.as_mut() {
            writer
                .enqueue(experiment_id, &run_id, &mut pg_completed_buffer)
                .map_err(invalid_data)?
        } else if let Some(store) = pg_store.as_mut() {
            flush_postgres_completed_runs(store, experiment_id, &run_id, &mut pg_completed_buffer)
                .map_err(invalid_data)?
        } else {
            0
        };
    }
    if let Some(writer) = pg_completed_writer {
        pg_persisted_games += writer.finish().map_err(invalid_data)?;
    }

    let artifact = soccer_self_play_artifact_from_queue_report(config, options, &report);
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
        assert_eq!(DEFAULT_SOCCER_QUEUE_NEURAL_DRAIN_TIMEOUT_MS, 0);
    }

    #[test]
    fn default_queue_postgres_async_writer_stays_bounded() {
        assert_eq!(DEFAULT_SOCCER_QUEUE_POSTGRES_ASYNC_BATCH_QUEUE, 4);
    }
}
