//! Postgres persistence for soccer self-play learning.
//!
//! The canonical table contract lives in `remote/libs/pg-defs/schema/schema.sql`.
//! This module is a small Rust adapter over that contract for queue runners.

use native_tls::TlsConnector;
use postgres::Client;
use postgres_native_tls::MakeTlsConnector;
use serde_json::{json, Value};

use crate::des::general::soccer::{
    MatchConfig, SoccerQEntry, SoccerQPolicy, SoccerQPolicyOptions, SoccerQStateKey,
    SoccerQTargetEntry, SoccerTeamQPolicies, Team,
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
        let mut tls_builder = TlsConnector::builder();
        if !soccer_learning_pg_should_verify_certificates(database_url) {
            // Match libpq sslmode=require/prefer semantics: encrypt the wire,
            // but do not require an RDS CA bundle in minimal runner images.
            tls_builder.danger_accept_invalid_certs(true);
            tls_builder.danger_accept_invalid_hostnames(true);
        }
        let tls = tls_builder
            .build()
            .map_err(|err| format!("build soccer learning postgres tls connector: {err}"))?;
        let client = Client::connect(database_url, MakeTlsConnector::new(tls))
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

fn soccer_learning_pg_should_verify_certificates(database_url: &str) -> bool {
    soccer_learning_pg_sslmode(database_url).is_some_and(|sslmode| {
        sslmode.eq_ignore_ascii_case("verify-ca") || sslmode.eq_ignore_ascii_case("verify-full")
    })
}

fn soccer_learning_pg_sslmode(database_url: &str) -> Option<&str> {
    let query = database_url.split_once('?')?.1;
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        key.eq_ignore_ascii_case("sslmode").then_some(value)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn soccer_learning_pg_sslmode_parses_query_param() {
        assert_eq!(
            soccer_learning_pg_sslmode("postgres://u:p@host/db?sslmode=require"),
            Some("require")
        );
        assert_eq!(
            soccer_learning_pg_sslmode(
                "postgres://u:p@host/db?connect_timeout=5&sslmode=verify-full"
            ),
            Some("verify-full")
        );
        assert_eq!(soccer_learning_pg_sslmode("postgres://u:p@host/db"), None);
    }

    #[test]
    fn soccer_learning_pg_only_verifies_explicit_verify_modes() {
        assert!(!soccer_learning_pg_should_verify_certificates(
            "postgres://u:p@host/db"
        ));
        assert!(!soccer_learning_pg_should_verify_certificates(
            "postgres://u:p@host/db?sslmode=require"
        ));
        assert!(soccer_learning_pg_should_verify_certificates(
            "postgres://u:p@host/db?sslmode=verify-ca"
        ));
        assert!(soccer_learning_pg_should_verify_certificates(
            "postgres://u:p@host/db?sslmode=VERIFY-FULL"
        ));
    }
}
