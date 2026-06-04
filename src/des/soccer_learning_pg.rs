//! Postgres persistence for soccer self-play learning.
//!
//! The canonical table contract lives in `remote/libs/pg-defs/schema/schema.sql`.
//! This module is a small Rust adapter over that contract for queue runners.

use postgres::{Client, NoTls};
use serde_json::{json, Value};

use crate::des::general::soccer::{
    MatchConfig, SoccerQEntry, SoccerQPolicy, SoccerQPolicyOptions, SoccerQStateKey,
    SoccerQTargetEntry, SoccerSetPlayTrainingArtifact, SoccerTeamQPolicies, Team,
};
use crate::des::soccer_learning::{
    soccer_learning_from_micros, soccer_learning_to_micros, soccer_team_label,
    SoccerLearningCompletedGame, SoccerLearningPolicyEntryKind,
};

#[derive(Clone, Debug)]
pub struct SoccerLearningPgPolicyVersion {
    pub id: String,
    pub generation: i32,
    pub policies: SoccerTeamQPolicies,
}

pub struct SoccerLearningPgStore {
    client: Client,
}

impl SoccerLearningPgStore {
    pub fn connect(database_url: &str) -> Result<Self, String> {
        let client = Client::connect(database_url, NoTls)
            .map_err(|err| format!("connect soccer learning postgres: {err}"))?;
        Ok(Self { client })
    }

    pub fn connect_from_env() -> Result<Option<Self>, String> {
        let Some(database_url) = soccer_learning_database_url() else {
            return Ok(None);
        };
        Self::connect(&database_url).map(Some)
    }

    pub fn ensure_experiment(
        &mut self,
        slug: &str,
        display_name: &str,
        config: &MatchConfig,
    ) -> Result<String, String> {
        if let Some(row) = self
            .client
            .query_opt(
                r#"
                select id::text
                from des_soccer_learning_experiments
                where slug = $1 and is_soft_deleted = false
                limit 1
                "#,
                &[&slug],
            )
            .map_err(|err| format!("select soccer learning experiment: {err}"))?
        {
            return Ok(row.get(0));
        }

        let config_json =
            serde_json::to_value(config).map_err(|err| format!("serialize match config: {err}"))?;
        let row = self
            .client
            .query_one(
                r#"
                insert into des_soccer_learning_experiments
                  (slug, display_name, config, meta_data)
                values
                  ($1, $2, $3, '{}'::jsonb)
                returning id::text
                "#,
                &[&slug, &display_name, &config_json],
            )
            .map_err(|err| format!("insert soccer learning experiment: {err}"))?;
        Ok(row.get(0))
    }

    pub fn load_latest_active_policy(
        &mut self,
        experiment_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
    ) -> Result<Option<SoccerLearningPgPolicyVersion>, String> {
        let Some(row) = self
            .client
            .query_opt(
                r#"
                select id::text, generation
                from des_soccer_learning_policy_versions
                where experiment_id = $1::uuid and status = 'active'
                order by generation desc, updated_at desc
                limit 1
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("select latest soccer policy version: {err}"))?
        else {
            return Ok(None);
        };
        let id: String = row.get(0);
        let generation: i32 = row.get(1);
        let policies = self.load_policy_entries(&id, home_options, away_options)?;
        Ok(Some(SoccerLearningPgPolicyVersion {
            id,
            generation,
            policies,
        }))
    }

    pub fn insert_policy_version(
        &mut self,
        experiment_id: &str,
        parent_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        source_kind: &str,
        status: &str,
        config: &MatchConfig,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
        policies: &SoccerTeamQPolicies,
        fitness: f64,
    ) -> Result<String, String> {
        let config_json =
            serde_json::to_value(config).map_err(|err| format!("serialize match config: {err}"))?;
        let options_json = json!({
            "home": home_options,
            "away": away_options,
        });
        let lineage = parent_policy_version_id
            .map(|id| json!([id]))
            .unwrap_or_else(|| json!([]));
        let metrics = json!({ "fitness": fitness });
        let entry_count =
            checked_i32(policies.home.entries().len() + policies.away.entries().len());
        let target_entry_count = checked_i32(
            policies.home.target_entries().len() + policies.away.target_entries().len(),
        );
        let visit_count = checked_i64(policies.home.visit_count() + policies.away.visit_count());
        let fitness_micros = soccer_learning_to_micros(fitness);
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer policy version transaction: {err}"))?;

        if status == "active" {
            tx.execute(
                r#"
                update des_soccer_learning_policy_versions
                set status = 'archived', updated_at = now()
                where experiment_id = $1::uuid and status = 'active'
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("archive old soccer policy versions: {err}"))?;
        }

        let row = tx
            .query_one(
                r#"
                insert into des_soccer_learning_policy_versions
                  (
                    experiment_id,
                    parent_policy_version_id,
                    generation,
                    version_label,
                    source_kind,
                    status,
                    options,
                    config,
                    lineage,
                    metrics,
                    entry_count,
                    target_entry_count,
                    visit_count,
                    fitness_micros
                  )
                values
                  (
                    $1::uuid,
                    $2::uuid,
                    $3,
                    $4,
                    $5,
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14
                  )
                returning id::text
                "#,
                &[
                    &experiment_id,
                    &parent_policy_version_id,
                    &generation,
                    &version_label,
                    &source_kind,
                    &status,
                    &options_json,
                    &config_json,
                    &lineage,
                    &metrics,
                    &entry_count,
                    &target_entry_count,
                    &visit_count,
                    &fitness_micros,
                ],
            )
            .map_err(|err| format!("insert soccer policy version: {err}"))?;
        let policy_version_id: String = row.get(0);

        insert_policy_entries_for_team(
            &mut tx,
            &policy_version_id,
            Team::Home,
            &policies.home,
            None,
        )?;
        insert_policy_entries_for_team(
            &mut tx,
            &policy_version_id,
            Team::Away,
            &policies.away,
            None,
        )?;

        tx.commit()
            .map_err(|err| format!("commit soccer policy version: {err}"))?;
        Ok(policy_version_id)
    }

    pub fn insert_completed_run(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        base_policy_version_id: Option<&str>,
        output_policy_version_id: Option<&str>,
        game: &SoccerLearningCompletedGame,
    ) -> Result<String, String> {
        let summary_json = serde_json::to_value(&game.summary)
            .map_err(|err| format!("serialize summary: {err}"))?;
        let stats_json = serde_json::to_value(&game.summary.stats)
            .map_err(|err| format!("serialize stats: {err}"))?;
        let seed = checked_i64(game.seed);
        let episode_index = checked_i32(game.episode);
        let score_home = checked_i32(game.summary.score_home);
        let score_away = checked_i32(game.summary.score_away);
        let home_goal_diff = game.score.home.goal_diff;
        let away_goal_diff = game.score.away.goal_diff;
        let home_outcome = game.score.home.outcome.as_str();
        let away_outcome = game.score.away.outcome.as_str();
        let home_merge_weight_micros = game.score.home.merge_weight_micros;
        let away_merge_weight_micros = game.score.away.merge_weight_micros;
        let duration_ticks = checked_i64(game.summary.ticks);
        let simulated_seconds_micros = soccer_learning_to_micros(game.summary.simulated_seconds);
        let elapsed_millis = (game.elapsed_seconds * 1000.0).round().max(0.0) as i64;
        let transitions = checked_i32(game.episode_summary.transitions);
        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer run transaction: {err}"))?;
        let row = tx
            .query_one(
                r#"
                insert into des_soccer_learning_runs
                  (
                    experiment_id,
                    base_policy_version_id,
                    output_policy_version_id,
                    runner_id,
                    seed,
                    episode_index,
                    status,
                    score_home,
                    score_away,
                    home_goal_diff,
                    away_goal_diff,
                    home_outcome,
                    away_outcome,
                    home_merge_weight_micros,
                    away_merge_weight_micros,
                    fitness_micros,
                    duration_ticks,
                    simulated_seconds_micros,
                    elapsed_millis,
                    transitions,
                    summary,
                    stats
                  )
                values
                  (
                    $1::uuid,
                    $2::uuid,
                    $3::uuid,
                    $4,
                    $5,
                    $6,
                    'completed',
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14,
                    $15,
                    $16,
                    $17,
                    $18,
                    $19,
                    $20,
                    $21
                  )
                returning id::text
                "#,
                &[
                    &experiment_id,
                    &base_policy_version_id,
                    &output_policy_version_id,
                    &runner_id,
                    &seed,
                    &episode_index,
                    &score_home,
                    &score_away,
                    &home_goal_diff,
                    &away_goal_diff,
                    &home_outcome,
                    &away_outcome,
                    &home_merge_weight_micros,
                    &away_merge_weight_micros,
                    &game.score.match_fitness_micros,
                    &duration_ticks,
                    &simulated_seconds_micros,
                    &elapsed_millis,
                    &transitions,
                    &summary_json,
                    &stats_json,
                ],
            )
            .map_err(|err| format!("insert soccer learning run: {err}"))?;
        let run_id: String = row.get(0);

        for delta in &game.delta.entries {
            let team = soccer_team_label(delta.team);
            let entry_kind = delta.entry_kind.as_str();
            let visit_delta = checked_i32(delta.visit_delta);
            tx.execute(
                r#"
                insert into des_soccer_learning_run_deltas
                  (
                    run_id,
                    team,
                    entry_kind,
                    state_hash,
                    state_key,
                    action,
                    target_fine_cell_id,
                    target_tactical_cell_id,
                    target_macro_cell_id,
                    target_root_cell_id,
                    before_value_micros,
                    after_value_micros,
                    value_delta_micros,
                    visit_delta,
                    merge_weight_micros,
                    effective_visit_micros
                  )
                values
                  (
                    $1::uuid,
                    $2,
                    $3,
                    $4,
                    $5,
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14,
                    $15,
                    $16
                  )
                "#,
                &[
                    &run_id,
                    &team,
                    &entry_kind,
                    &delta.state_hash,
                    &delta.state_json,
                    &delta.action,
                    &delta.target_fine_cell_id,
                    &delta.target_tactical_cell_id,
                    &delta.target_macro_cell_id,
                    &delta.target_root_cell_id,
                    &delta.before_value_micros,
                    &delta.after_value_micros,
                    &delta.value_delta_micros,
                    &visit_delta,
                    &delta.merge_weight_micros,
                    &delta.effective_visit_micros,
                ],
            )
            .map_err(|err| format!("insert soccer learning run delta: {err}"))?;
        }

        tx.commit()
            .map_err(|err| format!("commit soccer learning run: {err}"))?;
        Ok(run_id)
    }

    pub fn insert_set_play_training_artifact(
        &mut self,
        experiment_id: &str,
        runner_id: &str,
        base_policy_version_id: Option<&str>,
        generation: i32,
        version_label: &str,
        status: &str,
        artifact: &SoccerSetPlayTrainingArtifact,
        elapsed_seconds: f64,
    ) -> Result<(String, String), String> {
        let policies = SoccerTeamQPolicies {
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
        };
        let config_json = serde_json::to_value(&artifact.config)
            .map_err(|err| format!("serialize set-play config: {err}"))?;
        let options_json = json!({
            "home": &artifact.options,
            "away": &artifact.options,
        });
        let lineage = base_policy_version_id
            .map(|id| json!([id]))
            .unwrap_or_else(|| json!([]));
        let neural = json!({
            "enabled": artifact.learning.neural_learning_enabled,
            "backend": artifact.learning.neural_learning_backend,
            "trainingSteps": artifact.learning.neural_learning_training_steps,
            "samples": artifact.learning.neural_learning_samples,
            "pendingBatches": artifact.learning.neural_learning_pending_batches,
            "droppedBatches": artifact.learning.neural_learning_dropped_batches,
            "replaySamples": artifact.learning.neural_learning_replay_samples,
            "replayCapacity": artifact.learning.neural_learning_replay_capacity,
            "parameterCount": artifact.learning.neural_learning_parameter_count,
            "targetClip": artifact.learning.neural_learning_target_clip,
            "lastLoss": artifact.learning.neural_learning_last_loss,
            "averageLoss": artifact.learning.neural_learning_average_loss,
        });
        let metrics = json!({
            "fitness": artifact.goal_rate,
            "kind": "set-play-restart-training",
            "restart": &artifact.restart,
            "restarts": &artifact.restarts,
            "team": &artifact.team,
            "spot": &artifact.spot,
            "durationSeconds": artifact.duration_seconds,
            "episodes": artifact.episodes.len(),
            "goals": artifact.goals,
            "goalRate": artifact.goal_rate,
            "firstWindowGoalRate": artifact.first_window_goal_rate,
            "lastWindowGoalRate": artifact.last_window_goal_rate,
            "goalRateDelta": artifact.goal_rate_delta,
            "neural": neural,
        });
        let summary_json = json!({
            "kind": "set-play-restart-training",
            "restart": &artifact.restart,
            "restarts": &artifact.restarts,
            "team": &artifact.team,
            "spot": &artifact.spot,
            "durationSeconds": artifact.duration_seconds,
            "episodes": artifact.episodes.len(),
            "goals": artifact.goals,
            "goalRate": artifact.goal_rate,
            "firstWindowGoalRate": artifact.first_window_goal_rate,
            "lastWindowGoalRate": artifact.last_window_goal_rate,
            "goalRateDelta": artifact.goal_rate_delta,
        });
        let stats_json = json!({
            "learning": &artifact.learning,
            "neural": metrics["neural"].clone(),
            "episodes": &artifact.episodes,
        });
        let entry_count =
            checked_i32(policies.home.entries().len() + policies.away.entries().len());
        let target_entry_count = checked_i32(
            policies.home.target_entries().len() + policies.away.target_entries().len(),
        );
        let visit_count = checked_i64(policies.home.visit_count() + policies.away.visit_count());
        let fitness_micros = soccer_learning_to_micros(artifact.goal_rate);
        let score_home = checked_i32(if artifact.team == Team::Home {
            artifact.goals
        } else {
            0
        });
        let score_away = checked_i32(if artifact.team == Team::Away {
            artifact.goals
        } else {
            0
        });
        let home_goal_diff = if artifact.team == Team::Home {
            score_home
        } else {
            -score_away
        };
        let away_goal_diff = -home_goal_diff;
        let trained_team_scored = artifact.goals > 0;
        let (home_outcome, away_outcome) = match (artifact.team, trained_team_scored) {
            (Team::Home, true) => ("win", "loss"),
            (Team::Away, true) => ("loss", "win"),
            _ => ("draw", "draw"),
        };
        let trained_merge_weight = soccer_learning_to_micros(1.0 + artifact.goal_rate);
        let defending_merge_weight = soccer_learning_to_micros((1.0 - artifact.goal_rate) * 0.5);
        let (home_merge_weight_micros, away_merge_weight_micros) = match artifact.team {
            Team::Home => (trained_merge_weight, defending_merge_weight),
            Team::Away => (defending_merge_weight, trained_merge_weight),
        };
        let duration_ticks = checked_i64(
            artifact
                .episodes
                .iter()
                .map(|episode| episode.ticks)
                .sum::<u64>(),
        );
        let simulated_seconds = artifact
            .episodes
            .iter()
            .map(|episode| episode.simulated_seconds)
            .sum::<f64>();
        let simulated_seconds_micros = soccer_learning_to_micros(simulated_seconds);
        let elapsed_millis = (elapsed_seconds.max(0.0) * 1000.0).round() as i64;
        let transitions = checked_i32(
            artifact
                .episodes
                .iter()
                .map(|episode| episode.policy_updates)
                .sum::<u64>(),
        );

        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin soccer set-play training transaction: {err}"))?;
        ensure_soccer_learning_set_play_tables(&mut tx)?;

        if status == "active" {
            tx.execute(
                r#"
                update des_soccer_learning_policy_versions
                set status = 'archived', updated_at = now()
                where experiment_id = $1::uuid and status = 'active'
                "#,
                &[&experiment_id],
            )
            .map_err(|err| format!("archive old soccer policy versions: {err}"))?;
        }

        let policy_row = tx
            .query_one(
                r#"
                insert into des_soccer_learning_policy_versions
                  (
                    experiment_id,
                    parent_policy_version_id,
                    generation,
                    version_label,
                    source_kind,
                    status,
                    options,
                    config,
                    lineage,
                    metrics,
                    entry_count,
                    target_entry_count,
                    visit_count,
                    fitness_micros
                  )
                values
                  (
                    $1::uuid,
                    $2::uuid,
                    $3,
                    $4,
                    'replay',
                    $5,
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13
                  )
                returning id::text
                "#,
                &[
                    &experiment_id,
                    &base_policy_version_id,
                    &generation,
                    &version_label,
                    &status,
                    &options_json,
                    &config_json,
                    &lineage,
                    &metrics,
                    &entry_count,
                    &target_entry_count,
                    &visit_count,
                    &fitness_micros,
                ],
            )
            .map_err(|err| format!("insert soccer set-play policy version: {err}"))?;
        let policy_version_id: String = policy_row.get(0);

        let run_row = tx
            .query_one(
                r#"
                insert into des_soccer_learning_runs
                  (
                    experiment_id,
                    base_policy_version_id,
                    output_policy_version_id,
                    runner_id,
                    seed,
                    episode_index,
                    status,
                    score_home,
                    score_away,
                    home_goal_diff,
                    away_goal_diff,
                    home_outcome,
                    away_outcome,
                    home_merge_weight_micros,
                    away_merge_weight_micros,
                    fitness_micros,
                    duration_ticks,
                    simulated_seconds_micros,
                    elapsed_millis,
                    transitions,
                    summary,
                    stats
                  )
                values
                  (
                    $1::uuid,
                    $2::uuid,
                    $3::uuid,
                    $4,
                    $5,
                    0,
                    'completed',
                    $6,
                    $7,
                    $8,
                    $9,
                    $10,
                    $11,
                    $12,
                    $13,
                    $14,
                    $15,
                    $16,
                    $17,
                    $18,
                    $19,
                    $20
                  )
                returning id::text
                "#,
                &[
                    &experiment_id,
                    &base_policy_version_id,
                    &policy_version_id,
                    &runner_id,
                    &(artifact.config.seed as i64),
                    &score_home,
                    &score_away,
                    &home_goal_diff,
                    &away_goal_diff,
                    &home_outcome,
                    &away_outcome,
                    &home_merge_weight_micros,
                    &away_merge_weight_micros,
                    &fitness_micros,
                    &duration_ticks,
                    &simulated_seconds_micros,
                    &elapsed_millis,
                    &transitions,
                    &summary_json,
                    &stats_json,
                ],
            )
            .map_err(|err| format!("insert soccer set-play learning run: {err}"))?;
        let run_id: String = run_row.get(0);

        insert_policy_entries_for_team(
            &mut tx,
            &policy_version_id,
            Team::Home,
            &policies.home,
            Some(&run_id),
        )?;
        insert_policy_entries_for_team(
            &mut tx,
            &policy_version_id,
            Team::Away,
            &policies.away,
            Some(&run_id),
        )?;
        insert_normalized_set_play_training_records(
            &mut tx,
            &run_id,
            &policy_version_id,
            artifact,
        )?;

        tx.commit()
            .map_err(|err| format!("commit soccer set-play training transaction: {err}"))?;
        Ok((policy_version_id, run_id))
    }

    fn load_policy_entries(
        &mut self,
        policy_version_id: &str,
        home_options: SoccerQPolicyOptions,
        away_options: SoccerQPolicyOptions,
    ) -> Result<SoccerTeamQPolicies, String> {
        let rows = self
            .client
            .query(
                r#"
                select
                  team,
                  entry_kind,
                  state_key,
                  action,
                  target_fine_cell_id,
                  target_tactical_cell_id,
                  target_macro_cell_id,
                  target_root_cell_id,
                  value_micros,
                  visits
                from des_soccer_learning_policy_entries
                where policy_version_id = $1::uuid
                order by team, entry_kind, state_hash, action
                "#,
                &[&policy_version_id],
            )
            .map_err(|err| format!("select soccer policy entries: {err}"))?;
        let mut home_entries = Vec::new();
        let mut away_entries = Vec::new();
        let mut home_targets = Vec::new();
        let mut away_targets = Vec::new();

        for row in rows {
            let team: String = row.get(0);
            let entry_kind: String = row.get(1);
            let state_key_json: Value = row.get(2);
            let state: SoccerQStateKey = serde_json::from_value(state_key_json)
                .map_err(|err| format!("decode soccer policy state key: {err}"))?;
            let action: String = row.get(3);
            let target_fine_cell_id: i32 = row.get(4);
            let target_tactical_cell_id: i32 = row.get(5);
            let target_macro_cell_id: i32 = row.get(6);
            let target_root_cell_id: i32 = row.get(7);
            let value_micros: i64 = row.get(8);
            let visits_i32: i32 = row.get(9);
            let visits = visits_i32.max(0) as u32;
            let value = soccer_learning_from_micros(value_micros);
            match (team.as_str(), entry_kind.as_str()) {
                ("home", "action") => home_entries.push(SoccerQEntry {
                    state,
                    action,
                    value,
                    visits,
                }),
                ("away", "action") => away_entries.push(SoccerQEntry {
                    state,
                    action,
                    value,
                    visits,
                }),
                ("home", "target") => home_targets.push(SoccerQTargetEntry {
                    state,
                    action,
                    target_fine_cell_id: target_fine_cell_id.max(0) as usize,
                    target_tactical_cell_id: target_tactical_cell_id.max(0) as usize,
                    target_macro_cell_id: target_macro_cell_id.max(0) as usize,
                    target_root_cell_id: target_root_cell_id.max(0) as usize,
                    value,
                    visits,
                }),
                ("away", "target") => away_targets.push(SoccerQTargetEntry {
                    state,
                    action,
                    target_fine_cell_id: target_fine_cell_id.max(0) as usize,
                    target_tactical_cell_id: target_tactical_cell_id.max(0) as usize,
                    target_macro_cell_id: target_macro_cell_id.max(0) as usize,
                    target_root_cell_id: target_root_cell_id.max(0) as usize,
                    value,
                    visits,
                }),
                _ => {}
            }
        }

        Ok(SoccerTeamQPolicies {
            home: SoccerQPolicy::from_entries_with_targets(
                home_options,
                &home_entries,
                &home_targets,
            )?,
            away: SoccerQPolicy::from_entries_with_targets(
                away_options,
                &away_entries,
                &away_targets,
            )?,
        })
    }
}

fn insert_policy_entries_for_team(
    tx: &mut postgres::Transaction<'_>,
    policy_version_id: &str,
    team: Team,
    policy: &SoccerQPolicy,
    source_run_id: Option<&str>,
) -> Result<(), String> {
    let team_label = soccer_team_label(team);
    for entry in policy.entries() {
        let state_json = serde_json::to_value(&entry.state)
            .map_err(|err| format!("serialize soccer action state key: {err}"))?;
        let state_hash = state_hash(&state_json);
        let entry_kind = SoccerLearningPolicyEntryKind::Action.as_str();
        let value_micros = soccer_learning_to_micros(entry.value);
        let visits = checked_i32(entry.visits);
        tx.execute(
            r#"
            insert into des_soccer_learning_policy_entries
              (
                policy_version_id,
                team,
                entry_kind,
                state_hash,
                state_key,
                action,
                value_micros,
                visits,
                source_run_id
              )
            values
              ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9::uuid)
            "#,
            &[
                &policy_version_id,
                &team_label,
                &entry_kind,
                &state_hash,
                &state_json,
                &entry.action,
                &value_micros,
                &visits,
                &source_run_id,
            ],
        )
        .map_err(|err| format!("insert soccer policy action entry: {err}"))?;
    }

    for entry in policy.target_entries() {
        let state_json = serde_json::to_value(&entry.state)
            .map_err(|err| format!("serialize soccer target state key: {err}"))?;
        let state_hash = state_hash(&state_json);
        let entry_kind = SoccerLearningPolicyEntryKind::Target.as_str();
        let value_micros = soccer_learning_to_micros(entry.value);
        let visits = checked_i32(entry.visits);
        let target_fine_cell_id = checked_i32(entry.target_fine_cell_id);
        let target_tactical_cell_id = checked_i32(entry.target_tactical_cell_id);
        let target_macro_cell_id = checked_i32(entry.target_macro_cell_id);
        let target_root_cell_id = checked_i32(entry.target_root_cell_id);
        tx.execute(
            r#"
            insert into des_soccer_learning_policy_entries
              (
                policy_version_id,
                team,
                entry_kind,
                state_hash,
                state_key,
                action,
                target_fine_cell_id,
                target_tactical_cell_id,
                target_macro_cell_id,
                target_root_cell_id,
                value_micros,
                visits,
                source_run_id
              )
            values
              ($1::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13::uuid)
            "#,
            &[
                &policy_version_id,
                &team_label,
                &entry_kind,
                &state_hash,
                &state_json,
                &entry.action,
                &target_fine_cell_id,
                &target_tactical_cell_id,
                &target_macro_cell_id,
                &target_root_cell_id,
                &value_micros,
                &visits,
                &source_run_id,
            ],
        )
        .map_err(|err| format!("insert soccer policy target entry: {err}"))?;
    }
    Ok(())
}

fn ensure_soccer_learning_set_play_tables(
    tx: &mut postgres::Transaction<'_>,
) -> Result<(), String> {
    tx.batch_execute(
        r#"
        create table if not exists des_soccer_learning_set_play_runs (
          run_id uuid primary key references des_soccer_learning_runs(id) on delete cascade,
          policy_version_id uuid not null references des_soccer_learning_policy_versions(id) on delete cascade,
          primary_restart varchar(40) not null,
          team varchar(8) not null,
          spot_x_micros bigint not null,
          spot_y_micros bigint not null,
          duration_seconds_micros bigint not null,
          episode_count integer not null,
          goals integer not null,
          goal_rate_micros bigint not null,
          first_window_goal_rate_micros bigint not null,
          last_window_goal_rate_micros bigint not null,
          goal_rate_delta_micros bigint not null,
          created_at timestamptz default now() not null,
          constraint des_soccer_learning_set_play_runs_restart_chk
            check (primary_restart in ('direct-free-kick', 'indirect-free-kick')),
          constraint des_soccer_learning_set_play_runs_team_chk
            check (team in ('home', 'away')),
          constraint des_soccer_learning_set_play_runs_duration_chk
            check (duration_seconds_micros >= 0),
          constraint des_soccer_learning_set_play_runs_episode_chk
            check (episode_count >= 0),
          constraint des_soccer_learning_set_play_runs_goals_chk
            check (goals >= 0),
          constraint des_soccer_learning_set_play_runs_goal_rate_chk
            check (goal_rate_micros between 0 and 1000000)
        );

        create table if not exists des_soccer_learning_set_play_restart_mix (
          run_id uuid not null references des_soccer_learning_set_play_runs(run_id) on delete cascade,
          ordinal integer not null,
          restart varchar(40) not null,
          primary key (run_id, ordinal),
          constraint des_soccer_learning_set_play_restart_mix_ordinal_chk
            check (ordinal >= 0),
          constraint des_soccer_learning_set_play_restart_mix_restart_chk
            check (restart in ('direct-free-kick', 'indirect-free-kick'))
        );

        create table if not exists des_soccer_learning_set_play_episode_metrics (
          run_id uuid not null references des_soccer_learning_set_play_runs(run_id) on delete cascade,
          episode_index integer not null,
          seed bigint not null,
          restart varchar(40) not null,
          routine varchar(80),
          scored boolean not null,
          score_delta_for_team integer not null,
          ticks bigint not null,
          simulated_seconds_micros bigint not null,
          policy_updates bigint not null,
          home_policy_entries integer not null,
          home_policy_target_entries integer not null,
          away_policy_entries integer not null,
          away_policy_target_entries integer not null,
          neural_training_steps integer not null,
          neural_samples bigint not null,
          neural_replay_samples integer not null,
          neural_last_loss_micros bigint,
          cumulative_goals integer not null,
          goal_rate_so_far_micros bigint not null,
          primary key (run_id, episode_index),
          constraint des_soccer_learning_set_play_episode_idx_chk
            check (episode_index >= 0),
          constraint des_soccer_learning_set_play_episode_seed_chk
            check (seed >= 0),
          constraint des_soccer_learning_set_play_episode_restart_chk
            check (restart in ('direct-free-kick', 'indirect-free-kick')),
          constraint des_soccer_learning_set_play_episode_ticks_chk
            check (ticks >= 0),
          constraint des_soccer_learning_set_play_episode_seconds_chk
            check (simulated_seconds_micros >= 0),
          constraint des_soccer_learning_set_play_episode_policy_updates_chk
            check (policy_updates >= 0),
          constraint des_soccer_learning_set_play_episode_entries_chk
            check (
              home_policy_entries >= 0
              and home_policy_target_entries >= 0
              and away_policy_entries >= 0
              and away_policy_target_entries >= 0
            ),
          constraint des_soccer_learning_set_play_episode_neural_chk
            check (
              neural_training_steps >= 0
              and neural_samples >= 0
              and neural_replay_samples >= 0
            ),
          constraint des_soccer_learning_set_play_episode_goals_chk
            check (cumulative_goals >= 0),
          constraint des_soccer_learning_set_play_episode_goal_rate_chk
            check (goal_rate_so_far_micros between 0 and 1000000)
        );

        create table if not exists des_soccer_learning_neural_run_metrics (
          run_id uuid primary key references des_soccer_learning_runs(id) on delete cascade,
          policy_version_id uuid not null references des_soccer_learning_policy_versions(id) on delete cascade,
          enabled boolean not null,
          backend varchar(32) not null,
          training_steps integer not null,
          samples bigint not null,
          pending_batches integer not null,
          dropped_batches integer not null,
          replay_samples integer not null,
          replay_capacity integer not null,
          parameter_count integer not null,
          target_clip_micros bigint not null,
          last_loss_micros bigint,
          average_loss_micros bigint,
          created_at timestamptz default now() not null,
          constraint des_soccer_learning_neural_run_backend_chk
            check (backend in ('inline', 'threaded')),
          constraint des_soccer_learning_neural_run_counts_chk
            check (
              training_steps >= 0
              and samples >= 0
              and pending_batches >= 0
              and dropped_batches >= 0
              and replay_samples >= 0
              and replay_capacity >= 0
              and parameter_count >= 0
            )
        );

        create index if not exists des_soccer_learning_set_play_episode_restart_idx
          on des_soccer_learning_set_play_episode_metrics (restart, scored, episode_index);

        create index if not exists des_soccer_learning_neural_run_steps_idx
          on des_soccer_learning_neural_run_metrics (training_steps desc, samples desc);
        "#,
    )
    .map_err(|err| format!("ensure soccer set-play learning tables: {err}"))?;
    Ok(())
}

fn insert_normalized_set_play_training_records(
    tx: &mut postgres::Transaction<'_>,
    run_id: &str,
    policy_version_id: &str,
    artifact: &SoccerSetPlayTrainingArtifact,
) -> Result<(), String> {
    let team = soccer_team_label(artifact.team);
    let primary_restart = artifact.restart.as_label();
    let spot_x_micros = soccer_learning_to_micros(artifact.spot.x);
    let spot_y_micros = soccer_learning_to_micros(artifact.spot.y);
    let duration_seconds_micros = soccer_learning_to_micros(artifact.duration_seconds);
    let episode_count = checked_i32(artifact.episodes.len());
    let goals = checked_i32(artifact.goals);
    let goal_rate_micros = soccer_learning_to_micros(artifact.goal_rate);
    let first_window_goal_rate_micros = soccer_learning_to_micros(artifact.first_window_goal_rate);
    let last_window_goal_rate_micros = soccer_learning_to_micros(artifact.last_window_goal_rate);
    let goal_rate_delta_micros = soccer_learning_to_micros(artifact.goal_rate_delta);

    tx.execute(
        r#"
        insert into des_soccer_learning_set_play_runs
          (
            run_id,
            policy_version_id,
            primary_restart,
            team,
            spot_x_micros,
            spot_y_micros,
            duration_seconds_micros,
            episode_count,
            goals,
            goal_rate_micros,
            first_window_goal_rate_micros,
            last_window_goal_rate_micros,
            goal_rate_delta_micros
          )
        values
          ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
        "#,
        &[
            &run_id,
            &policy_version_id,
            &primary_restart,
            &team,
            &spot_x_micros,
            &spot_y_micros,
            &duration_seconds_micros,
            &episode_count,
            &goals,
            &goal_rate_micros,
            &first_window_goal_rate_micros,
            &last_window_goal_rate_micros,
            &goal_rate_delta_micros,
        ],
    )
    .map_err(|err| format!("insert soccer set-play run metrics: {err}"))?;

    for (ordinal, restart) in artifact.restarts.iter().enumerate() {
        let ordinal = checked_i32(ordinal);
        let restart_label = restart.as_label();
        tx.execute(
            r#"
            insert into des_soccer_learning_set_play_restart_mix
              (run_id, ordinal, restart)
            values
              ($1::uuid, $2, $3)
            "#,
            &[&run_id, &ordinal, &restart_label],
        )
        .map_err(|err| format!("insert soccer set-play restart mix: {err}"))?;
    }

    for episode in &artifact.episodes {
        let episode_index = checked_i32(episode.episode);
        let seed = checked_i64(episode.seed);
        let restart = episode.restart.as_label();
        let routine = episode
            .routine
            .map(|routine| routine.as_label().to_string());
        let ticks = checked_i64(episode.ticks);
        let simulated_seconds_micros = soccer_learning_to_micros(episode.simulated_seconds);
        let policy_updates = checked_i64(episode.policy_updates);
        let home_policy_entries = checked_i32(episode.home_policy_entries);
        let home_policy_target_entries = checked_i32(episode.home_policy_target_entries);
        let away_policy_entries = checked_i32(episode.away_policy_entries);
        let away_policy_target_entries = checked_i32(episode.away_policy_target_entries);
        let neural_training_steps = checked_i32(episode.neural_training_steps);
        let neural_samples = checked_i64(episode.neural_samples as u64);
        let neural_replay_samples = checked_i32(episode.neural_replay_samples);
        let neural_last_loss_micros = episode.neural_last_loss.map(soccer_learning_to_micros);
        let cumulative_goals = checked_i32(episode.cumulative_goals);
        let goal_rate_so_far_micros = soccer_learning_to_micros(episode.goal_rate_so_far);
        tx.execute(
            r#"
            insert into des_soccer_learning_set_play_episode_metrics
              (
                run_id,
                episode_index,
                seed,
                restart,
                routine,
                scored,
                score_delta_for_team,
                ticks,
                simulated_seconds_micros,
                policy_updates,
                home_policy_entries,
                home_policy_target_entries,
                away_policy_entries,
                away_policy_target_entries,
                neural_training_steps,
                neural_samples,
                neural_replay_samples,
                neural_last_loss_micros,
                cumulative_goals,
                goal_rate_so_far_micros
              )
            values
              (
                $1::uuid,
                $2,
                $3,
                $4,
                $5,
                $6,
                $7,
                $8,
                $9,
                $10,
                $11,
                $12,
                $13,
                $14,
                $15,
                $16,
                $17,
                $18,
                $19,
                $20
              )
            "#,
            &[
                &run_id,
                &episode_index,
                &seed,
                &restart,
                &routine,
                &episode.scored,
                &episode.score_delta_for_team,
                &ticks,
                &simulated_seconds_micros,
                &policy_updates,
                &home_policy_entries,
                &home_policy_target_entries,
                &away_policy_entries,
                &away_policy_target_entries,
                &neural_training_steps,
                &neural_samples,
                &neural_replay_samples,
                &neural_last_loss_micros,
                &cumulative_goals,
                &goal_rate_so_far_micros,
            ],
        )
        .map_err(|err| format!("insert soccer set-play episode metrics: {err}"))?;
    }

    let enabled = artifact.learning.neural_learning_enabled;
    let backend = artifact.learning.neural_learning_backend.as_str();
    let training_steps = checked_i32(artifact.learning.neural_learning_training_steps);
    let samples = checked_i64(artifact.learning.neural_learning_samples as u64);
    let pending_batches = checked_i32(artifact.learning.neural_learning_pending_batches);
    let dropped_batches = checked_i32(artifact.learning.neural_learning_dropped_batches);
    let replay_samples = checked_i32(artifact.learning.neural_learning_replay_samples);
    let replay_capacity = checked_i32(artifact.learning.neural_learning_replay_capacity);
    let parameter_count = checked_i32(artifact.learning.neural_learning_parameter_count);
    let target_clip_micros =
        soccer_learning_to_micros(artifact.learning.neural_learning_target_clip);
    let last_loss_micros = artifact
        .learning
        .neural_learning_last_loss
        .map(soccer_learning_to_micros);
    let average_loss_micros = artifact
        .learning
        .neural_learning_average_loss
        .map(soccer_learning_to_micros);
    tx.execute(
        r#"
        insert into des_soccer_learning_neural_run_metrics
          (
            run_id,
            policy_version_id,
            enabled,
            backend,
            training_steps,
            samples,
            pending_batches,
            dropped_batches,
            replay_samples,
            replay_capacity,
            parameter_count,
            target_clip_micros,
            last_loss_micros,
            average_loss_micros
          )
        values
          ($1::uuid, $2::uuid, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
        "#,
        &[
            &run_id,
            &policy_version_id,
            &enabled,
            &backend,
            &training_steps,
            &samples,
            &pending_batches,
            &dropped_batches,
            &replay_samples,
            &replay_capacity,
            &parameter_count,
            &target_clip_micros,
            &last_loss_micros,
            &average_loss_micros,
        ],
    )
    .map_err(|err| format!("insert soccer neural run metrics: {err}"))?;

    Ok(())
}

fn soccer_learning_database_url() -> Option<String> {
    [
        "SOCCER_DATABASE_URL",
        "AGENT_TASKS_RDS_DATABASE_URL",
        "RDS_DATABASE_URL",
        "DATABASE_URL",
        "PG_DATABASE_URL",
    ]
    .into_iter()
    .find_map(|name| {
        std::env::var(name)
            .ok()
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty())
    })
}

fn state_hash(state_json: &Value) -> String {
    let raw = serde_json::to_string(state_json).unwrap_or_default();
    let mut hash = 0xcbf29ce484222325u64;
    for byte in raw.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn checked_i32(value: impl TryInto<i64>) -> i32 {
    let value = value.try_into().unwrap_or(i64::MAX);
    value.clamp(i32::MIN as i64, i32::MAX as i64) as i32
}

fn checked_i64(value: impl TryInto<u64>) -> i64 {
    let value = value.try_into().unwrap_or(u64::MAX);
    value.min(i64::MAX as u64) as i64
}
