use std::env;
use std::error::Error;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use des_engine::des::general::soccer::{
    train_soccer_set_play_restarts, train_soccer_set_play_restarts_with_initial_policies,
    MatchConfig, SoccerNeuralLearningBackend, SoccerNeuralLearningConfig, SoccerQPolicyOptions,
    SoccerSetPlayRestartKind, SoccerSetPlayTrainingRequest, Team, Vec2,
};
use des_engine::des::soccer_learning_pg::SoccerLearningPgStore;
use serde::Serialize;

fn env_value(name: &str) -> Option<String> {
    env::var(name)
        .ok()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
}

fn env_bool(name: &str, default: bool) -> Result<bool, Box<dyn Error>> {
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    match value.to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "no" | "n" | "off" => Ok(false),
        _ => Err(format!("{name} must be boolean, got {value:?}").into()),
    }
}

fn env_parse<T>(name: &str, default: T) -> Result<T, Box<dyn Error>>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let Some(value) = env_value(name) else {
        return Ok(default);
    };
    value
        .parse::<T>()
        .map_err(|err| format!("{name}={value:?} is invalid: {err}").into())
}

fn env_neural_backend() -> Result<SoccerNeuralLearningBackend, Box<dyn Error>> {
    let Some(value) = env_value("SOCCER_NEURAL_LEARNING_BACKEND") else {
        return Ok(SoccerNeuralLearningBackend::Threaded);
    };
    match value.to_ascii_lowercase().as_str() {
        "threaded" | "thread" | "worker" => Ok(SoccerNeuralLearningBackend::Threaded),
        "inline" | "sync" => Ok(SoccerNeuralLearningBackend::Inline),
        _ => Err(format!("SOCCER_NEURAL_LEARNING_BACKEND={value:?} is invalid").into()),
    }
}

fn env_team() -> Result<Team, Box<dyn Error>> {
    let Some(value) = env_value("SOCCER_SET_PLAY_TEAM") else {
        return Ok(Team::Home);
    };
    match value.to_ascii_lowercase().as_str() {
        "home" => Ok(Team::Home),
        "away" => Ok(Team::Away),
        _ => Err(format!("SOCCER_SET_PLAY_TEAM={value:?} is invalid").into()),
    }
}

fn parse_restart_token(value: &str) -> Result<SoccerSetPlayRestartKind, Box<dyn Error>> {
    match value.to_ascii_lowercase().replace('_', "-").as_str() {
        "direct-free-kick" | "direct-freekick" | "dfk" | "free-kick" | "freekick" | "fk" => {
            Ok(SoccerSetPlayRestartKind::DirectFreeKick)
        }
        "indirect-free-kick" | "indirect-freekick" | "ifk" => {
            Ok(SoccerSetPlayRestartKind::IndirectFreeKick)
        }
        _ => Err(format!("restart {value:?} is invalid").into()),
    }
}

fn env_restarts() -> Result<Vec<SoccerSetPlayRestartKind>, Box<dyn Error>> {
    let raw = env_value("SOCCER_SET_PLAY_RESTARTS")
        .or_else(|| env_value("SOCCER_SET_PLAY_RESTART"))
        .unwrap_or_else(|| "indirect-free-kick,direct-free-kick".to_string());
    let normalized = raw.to_ascii_lowercase().replace('_', "-");
    if matches!(
        normalized.as_str(),
        "both" | "all" | "free-kicks" | "free-kick"
    ) {
        return Ok(vec![
            SoccerSetPlayRestartKind::IndirectFreeKick,
            SoccerSetPlayRestartKind::DirectFreeKick,
        ]);
    }
    let mut restarts = Vec::new();
    for token in raw
        .split(',')
        .map(str::trim)
        .filter(|token| !token.is_empty())
    {
        let restart = parse_restart_token(token)?;
        if !restarts.contains(&restart) {
            restarts.push(restart);
        }
    }
    if restarts.is_empty() {
        return Err("SOCCER_SET_PLAY_RESTARTS must name at least one restart".into());
    }
    Ok(restarts)
}

fn run_id_default() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    format!("free-kick-{now}")
}

fn version_label_from_run_id(run_id: &str) -> String {
    let mut label = run_id
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | ':' | '/' | '-') {
                ch
            } else {
                '-'
            }
        })
        .collect::<String>();
    if label.is_empty() {
        label = run_id_default();
    }
    label.truncate(120);
    format!("{label}-set-play")
}

fn free_kick_spot(config: &MatchConfig, team: Team, distance_yards: f64) -> Vec2 {
    let width = config.field_width_yards.max(1.0);
    let length = config.field_length_yards.max(1.0);
    let distance_yards = if distance_yards.is_finite() && distance_yards > 0.0 {
        distance_yards
    } else {
        25.0
    };
    Vec2::new(
        width * 0.5,
        team.goal_y(length) - team.attack_dir() * distance_yards,
    )
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), Box<dyn Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent)?;
    }
    let temp_path = path.with_extension(format!(
        "{}.tmp",
        path.extension()
            .and_then(|ext| ext.to_str())
            .unwrap_or("json")
    ));
    {
        let mut file = File::create(&temp_path)?;
        serde_json::to_writer_pretty(&mut file, value)?;
        file.write_all(b"\n")?;
        file.flush()?;
        file.sync_all()?;
    }
    fs::rename(&temp_path, path)?;
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        let _ = File::open(parent).and_then(|dir| dir.sync_all());
    }
    Ok(())
}

fn main() {
    if let Err(err) = run() {
        eprintln!("main_soccer_set_play_learning_run: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let started = Instant::now();
    let run_id = env_value("SOCCER_SET_PLAY_RUN_ID")
        .or_else(|| env_value("SOCCER_RUN_ID"))
        .unwrap_or_else(run_id_default);
    let run_dir = env_value("SOCCER_SET_PLAY_RUN_DIR")
        .or_else(|| env_value("SOCCER_RUN_DIR"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("out/soccer-set-play-learning").join(&run_id));
    let artifact_path = env_value("SOCCER_SET_PLAY_ARTIFACT_PATH")
        .or_else(|| env_value("SOCCER_ARTIFACT_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|| run_dir.join("artifact.json"));

    let episodes = env_parse("SOCCER_SET_PLAY_EPISODES", 100usize)?;
    if episodes == 0 {
        return Err("SOCCER_SET_PLAY_EPISODES must be at least 1".into());
    }
    let duration_seconds = env_parse("SOCCER_SET_PLAY_DURATION_SECONDS", 10.0_f64)?;
    if !duration_seconds.is_finite() || duration_seconds <= 0.0 {
        return Err("SOCCER_SET_PLAY_DURATION_SECONDS must be positive and finite".into());
    }
    let distance_yards = env_parse("SOCCER_FREE_KICK_DISTANCE_YARDS", 25.0_f64)?;
    let team = env_team()?;
    let restarts = env_restarts()?;
    let restart = restarts[0];
    let mut config = MatchConfig::default();
    config.duration_seconds = duration_seconds;
    config.dt_seconds = env_parse("SOCCER_DT_SECONDS", config.dt_seconds)?;
    if !config.dt_seconds.is_finite() || config.dt_seconds <= 0.0 {
        return Err("SOCCER_DT_SECONDS must be positive and finite".into());
    }
    config.seed = env_parse("SOCCER_SEED", config.seed)?;
    config.learning_interval_ticks = env_parse(
        "SOCCER_LEARNING_INTERVAL_TICKS",
        config.learning_interval_ticks,
    )?
    .max(1);
    config.learning_enabled = true;
    config.learning_logging_enabled = false;
    config.max_human_players = 0;
    config.neural_learning = SoccerNeuralLearningConfig {
        enabled: env_bool("SOCCER_NEURAL_LEARNING_ENABLED", true)?,
        backend: env_neural_backend()?,
        learning_rate: env_parse(
            "SOCCER_NEURAL_LEARNING_RATE",
            SoccerNeuralLearningConfig::default().learning_rate,
        )?,
        batch_size: env_parse("SOCCER_NEURAL_BATCH_SIZE", 32usize)?,
        train_every_ticks: env_parse("SOCCER_NEURAL_TRAIN_EVERY_TICKS", 1usize)?,
        max_batches_per_tick: env_parse("SOCCER_NEURAL_MAX_BATCHES_PER_TICK", 1usize)?,
        hidden_units: env_parse("SOCCER_NEURAL_HIDDEN_UNITS", 24usize)?,
        target_scale: env_parse(
            "SOCCER_NEURAL_TARGET_SCALE",
            SoccerNeuralLearningConfig::default().target_scale,
        )?,
        max_pending_batches: env_parse("SOCCER_NEURAL_MAX_PENDING_BATCHES", 32usize)?,
        replay_capacity: env_parse("SOCCER_NEURAL_REPLAY_CAPACITY", 1024usize)?,
        replay_samples_per_tick: env_parse("SOCCER_NEURAL_REPLAY_SAMPLES_PER_TICK", 16usize)?,
        target_clip: env_parse(
            "SOCCER_NEURAL_TARGET_CLIP",
            SoccerNeuralLearningConfig::default().target_clip,
        )?,
    };
    let options = SoccerQPolicyOptions {
        alpha: env_parse("SOCCER_Q_ALPHA", SoccerQPolicyOptions::default().alpha)?,
        gamma: env_parse("SOCCER_Q_GAMMA", SoccerQPolicyOptions::default().gamma)?,
    };

    let mut pg_store = SoccerLearningPgStore::connect_from_env()?;
    let mut pg_experiment_id = None::<String>;
    let mut pg_base_policy_version_id = None::<String>;
    let mut pg_base_generation = None::<i32>;
    let initial_policies = if let Some(store) = pg_store.as_mut() {
        let slug = env_value("SOCCER_EXPERIMENT_SLUG")
            .unwrap_or_else(|| "soccer-free-kick-restarts".to_string());
        let display_name = env_value("SOCCER_EXPERIMENT_NAME")
            .unwrap_or_else(|| "Soccer free-kick restart learning".to_string());
        let experiment_id = store.ensure_experiment(&slug, &display_name, &config)?;
        let resume = env_bool("SOCCER_RESUME_POSTGRES_POLICY", true)?;
        let latest = if resume {
            store.load_latest_active_policy(&experiment_id, options.clone(), options.clone())?
        } else {
            None
        };
        if let Some(version) = latest {
            println!(
                "postgres_resume_policy experiment={} policy_version={} generation={}",
                experiment_id, version.id, version.generation
            );
            pg_base_policy_version_id = Some(version.id);
            pg_base_generation = Some(version.generation);
            pg_experiment_id = Some(experiment_id);
            Some(version.policies)
        } else {
            pg_experiment_id = Some(experiment_id);
            None
        }
    } else {
        None
    };

    let spot = free_kick_spot(&config, team, distance_yards);
    println!(
        "soccer_set_play_learning_start run_id={} episodes={} restarts={:?} team={:?} duration={:.3}s distance={:.2}yd dt={:.3}s neural_enabled={} neural_backend={:?}",
        run_id,
        episodes,
        restarts,
        team,
        duration_seconds,
        distance_yards,
        config.dt_seconds,
        config.neural_learning.enabled,
        config.neural_learning.backend,
    );

    let request = SoccerSetPlayTrainingRequest {
        config,
        episodes,
        restart,
        restarts,
        team,
        spot: Some(spot),
        duration_seconds,
        options: Some(options),
        vector_hint: None,
    };
    let artifact = if let Some(policies) = initial_policies {
        train_soccer_set_play_restarts_with_initial_policies(request, policies)?
    } else {
        train_soccer_set_play_restarts(request)?
    };
    let elapsed = started.elapsed().as_secs_f64();
    write_json(&artifact_path, &artifact)?;

    let mut pg_policy_version_id = None::<String>;
    let mut pg_run_id = None::<String>;
    if let (Some(store), Some(experiment_id)) = (pg_store.as_mut(), pg_experiment_id.as_deref()) {
        let generation = pg_base_generation.unwrap_or(-1).saturating_add(1);
        let version_label = env_value("SOCCER_POLICY_VERSION_LABEL")
            .unwrap_or_else(|| version_label_from_run_id(&run_id));
        let (policy_version_id, run_pg_id) = store.insert_set_play_training_artifact(
            experiment_id,
            &run_id,
            pg_base_policy_version_id.as_deref(),
            generation,
            &version_label,
            "active",
            &artifact,
            elapsed,
        )?;
        println!(
            "postgres_persisted_set_play run_id={} policy_version={} pg_run={}",
            run_id, policy_version_id, run_pg_id
        );
        pg_policy_version_id = Some(policy_version_id);
        pg_run_id = Some(run_pg_id);
    }

    println!(
        "soccer_set_play_learning_done episodes={} goals={} goal_rate={:.4} first_window={:.4} last_window={:.4} delta={:.4} elapsed={:.2}s neural_steps={} neural_samples={}",
        artifact.episodes.len(),
        artifact.goals,
        artifact.goal_rate,
        artifact.first_window_goal_rate,
        artifact.last_window_goal_rate,
        artifact.goal_rate_delta,
        elapsed,
        artifact.learning.neural_learning_training_steps,
        artifact.learning.neural_learning_samples,
    );
    println!("artifact={}", artifact_path.display());
    if let Some(policy_version_id) = pg_policy_version_id {
        println!("postgres_policy_version={policy_version_id}");
    }
    if let Some(run_id) = pg_run_id {
        println!("postgres_run={run_id}");
    }
    Ok(())
}
