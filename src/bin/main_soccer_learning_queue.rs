//! Queue-style soccer self-play runner.
//!
//! Unlike the batch runner, this keeps a fixed number of simulation slots full:
//! when one game finishes, its deltas are merged and the next game starts from
//! the newest available policy.

use std::error::Error;
use std::fs;
use std::io::{BufWriter, Error as IoError, ErrorKind};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use des_engine::des::general::soccer::{
    MatchConfig, SoccerQPolicyOptions, SoccerSelfPlayTrainingArtifact,
    SoccerTacticalLearningWeights, SoccerTeamPolicyArtifact, SoccerTeamQPolicies,
};
use des_engine::des::soccer_learning::{
    run_soccer_learning_queue_with_observer, soccer_self_play_artifact_from_queue_report,
    SoccerLearningQueueRunnerConfig,
};
use des_engine::des::soccer_learning_pg::SoccerLearningPgStore;
use serde::Serialize;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(default)
}

fn env_u32(name: &str, default: u32) -> u32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .unwrap_or(default)
}

fn env_f64(name: &str, default: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(default)
}

fn env_bool(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "y" | "on"
            )
        })
        .unwrap_or(default)
}

fn env_tactical_learning_weights() -> SoccerTacticalLearningWeights {
    let default = SoccerTacticalLearningWeights::default();
    SoccerTacticalLearningWeights {
        attack_spacing_delta_weight: env_f64(
            "SOCCER_ATTACK_SPACING_DELTA_WEIGHT",
            default.attack_spacing_delta_weight,
        ),
        attack_spacing_score_weight: env_f64(
            "SOCCER_ATTACK_SPACING_SCORE_WEIGHT",
            default.attack_spacing_score_weight,
        ),
        attack_width_delta_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_DELTA_WEIGHT",
            default.attack_width_delta_weight,
        ),
        attack_width_score_weight: env_f64(
            "SOCCER_ATTACK_WIDTH_SCORE_WEIGHT",
            default.attack_width_score_weight,
        ),
        attack_flank_lane_weight: env_f64(
            "SOCCER_ATTACK_FLANK_LANE_WEIGHT",
            default.attack_flank_lane_weight,
        ),
        defense_spacing_delta_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_DELTA_WEIGHT",
            default.defense_spacing_delta_weight,
        ),
        defense_spacing_score_weight: env_f64(
            "SOCCER_DEFENSE_SPACING_SCORE_WEIGHT",
            default.defense_spacing_score_weight,
        ),
        defense_contract_delta_weight: env_f64(
            "SOCCER_DEFENSE_CONTRACT_DELTA_WEIGHT",
            default.defense_contract_delta_weight,
        ),
        defense_compactness_score_weight: env_f64(
            "SOCCER_DEFENSE_COMPACTNESS_SCORE_WEIGHT",
            default.defense_compactness_score_weight,
        ),
    }
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

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut tmp_name = path.as_os_str().to_os_string();
    tmp_name.push(".tmp");
    let tmp_path = PathBuf::from(tmp_name);
    let result = (|| -> Result<(), Box<dyn Error>> {
        let file = fs::File::create(&tmp_path)?;
        serde_json::to_writer(BufWriter::new(file), value)?;
        Ok(())
    })();
    if let Err(err) = result {
        let _ = fs::remove_file(&tmp_path);
        return Err(err);
    }
    fs::rename(&tmp_path, path)?;
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
    let games = env_usize("SOCCER_QUEUE_GAMES", env_usize("SOCCER_GAMES", 100));
    let parallel_games = env_usize(
        "SOCCER_QUEUE_PARALLEL_GAMES",
        env_usize("SOCCER_PARALLEL_GAMES", 10),
    )
    .clamp(1, 100);
    let halves = env_usize("SOCCER_HALVES", 2).max(1);
    let half_minutes = env_f64("SOCCER_HALF_MINUTES", 45.0);
    let minutes = env_f64("SOCCER_MINUTES", half_minutes * halves as f64);
    let dt_seconds = env_f64("SOCCER_DT_SECONDS", 0.2);
    let learning_interval_ticks = env_usize("SOCCER_LEARNING_INTERVAL_TICKS", 4).max(1);
    let seed = env_u32("SOCCER_SEED", 2026);
    let options = SoccerQPolicyOptions {
        alpha: env_f64("SOCCER_ALPHA", 0.20),
        gamma: env_f64("SOCCER_GAMMA", 0.96),
    };
    let tactical_learning = env_tactical_learning_weights();
    let config = MatchConfig {
        dt_seconds,
        duration_seconds: minutes * 60.0,
        period_count: halves,
        period_break_recovery_seconds: env_f64(
            "SOCCER_PERIOD_BREAK_RECOVERY_SECONDS",
            env_f64("SOCCER_HALFTIME_RECOVERY_SECONDS", 900.0),
        ),
        learning_enabled: true,
        learning_logging_enabled: env_bool("SOCCER_LEARNING_LOGGING", false),
        learning_interval_ticks,
        tactical_learning,
        max_human_players: 0,
        seed,
        ..MatchConfig::default()
    };
    let resume_artifact = std::env::var("SOCCER_RESUME_ARTIFACT")
        .or_else(|_| std::env::var("SOCCER_RESUME_ARTIFACT_PATH"))
        .ok();
    let run_id = std::env::var("SOCCER_RUN_ID").unwrap_or_else(|_| default_run_id());
    let artifact_path = PathBuf::from(
        std::env::var("SOCCER_ARTIFACT_PATH").unwrap_or_else(|_| format!("out/{run_id}.json")),
    );
    let mut pg_store = SoccerLearningPgStore::connect_from_env().map_err(invalid_data)?;
    let mut pg_experiment_id = None::<String>;
    let mut pg_base_policy_version_id = None::<String>;
    let mut pg_generation = 0i32;

    let mut initial_policies = load_initial_policies(resume_artifact.as_deref(), options.clone())?;
    if let Some(store) = pg_store.as_mut() {
        let experiment_slug = std::env::var("SOCCER_EXPERIMENT_SLUG")
            .unwrap_or_else(|_| "soccer-self-play".to_string());
        let experiment_name = std::env::var("SOCCER_EXPERIMENT_NAME")
            .unwrap_or_else(|_| "Soccer self-play".to_string());
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
                pg_base_policy_version_id = Some(version.id);
                pg_generation = version.generation;
            }
        }
        pg_experiment_id = Some(experiment_id);
    }

    println!(
        "soccer_learning_queue_start run_id={} games={} parallel_games={} minutes={:.1} dt={:.3}s ticks_per_game={} seed={}",
        run_id,
        games,
        parallel_games,
        minutes,
        dt_seconds,
        config.total_ticks(),
        seed
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
            options: options.clone(),
            prune_action_entries_per_team: env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0),
            prune_target_entries_per_team: env_usize(
                "SOCCER_MAX_POLICY_TARGET_ENTRIES_PER_TEAM",
                env_usize("SOCCER_MAX_POLICY_ENTRIES_PER_TEAM", 0),
            ),
            min_policy_visits: env_u32("SOCCER_MIN_POLICY_VISITS", 0),
        },
        initial_policies,
        |game, merged_policies| {
            let Some(store) = pg_store.as_mut() else {
                return Ok(());
            };
            let Some(experiment_id) = pg_experiment_id.as_deref() else {
                return Ok(());
            };
            let next_generation = pg_generation.saturating_add(1);
            let version_label = format!("{}-episode-{:06}", run_id, game.episode + 1);
            let output_policy_version_id = store.insert_policy_version(
                experiment_id,
                pg_base_policy_version_id.as_deref(),
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
            let run_row_id = store.insert_completed_run(
                experiment_id,
                &run_id,
                pg_base_policy_version_id.as_deref(),
                Some(&output_policy_version_id),
                game,
            )?;
            println!(
                "postgres_persisted_game episode={} run_id={} policy_version={} generation={}",
                game.episode + 1,
                run_row_id,
                output_policy_version_id,
                next_generation
            );
            pg_base_policy_version_id = Some(output_policy_version_id);
            pg_generation = next_generation;
            Ok(())
        },
    )
    .map_err(invalid_data)?;

    let artifact = soccer_self_play_artifact_from_queue_report(config, options, &report);
    write_json(&artifact_path, &artifact)?;

    println!(
        "soccer_learning_queue_done completed={} failed={} elapsed={:.2}s home_goals={} away_goals={} policy_entries={} target_entries={}",
        report.completed_games,
        report.failed_games,
        report.elapsed_seconds,
        report.total_home_goals,
        report.total_away_goals,
        report.final_policy_entries,
        report.final_target_entries
    );
    println!("artifact={}", artifact_path.display());
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("soccer learning queue failed: {error}");
        std::process::exit(1);
    }
}
