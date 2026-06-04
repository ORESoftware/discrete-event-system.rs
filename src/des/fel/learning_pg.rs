//! Postgres persistence for FEL elevator learning artifacts.
//!
//! The canonical table contract lives in
//! `remote/libs/pg-defs/schema/schema.sql`. This adapter only writes to those
//! tables; it never creates or migrates schema.

use native_tls::TlsConnector;
use postgres::Client;
use postgres_native_tls::MakeTlsConnector;
use serde_json::{json, Value};

use super::elevator::ElevatorConfig;

const FEL_FIXED_SCALE: f64 = 1_000_000.0;

pub struct FelElevatorLearningPgStore {
    client: Client,
}

#[derive(Clone, Debug)]
struct PolicyStateRow {
    policy_kind: String,
    feature_dim: i32,
    output_dim: i32,
    parameter_count: i32,
    online_learning_updates: i64,
    loss_history: Value,
    state: Value,
}

impl FelElevatorLearningPgStore {
    pub fn connect(database_url: &str) -> Result<Self, String> {
        let mut tls_builder = TlsConnector::builder();
        if !fel_elevator_pg_should_verify_certificates(database_url) {
            // Match the existing soccer learning adapter: sslmode=require/prefer
            // encrypts transport without requiring an RDS CA bundle locally.
            tls_builder.danger_accept_invalid_certs(true);
            tls_builder.danger_accept_invalid_hostnames(true);
        }
        let tls = tls_builder
            .build()
            .map_err(|err| format!("build FEL elevator postgres tls connector: {err}"))?;
        let client = Client::connect(database_url, MakeTlsConnector::new(tls))
            .map_err(|err| format!("connect FEL elevator postgres: {err}"))?;
        Ok(Self { client })
    }

    pub fn connect_from_env() -> Result<Option<Self>, String> {
        let Some(database_url) = fel_elevator_learning_database_url() else {
            return Ok(None);
        };
        Self::connect(&database_url).map(Some)
    }

    pub fn insert_learning_run(
        &mut self,
        scenario_slug: &str,
        run_label: &str,
        config: &ElevatorConfig,
        artifact: &Value,
    ) -> Result<String, String> {
        let meta = artifact.get("meta").unwrap_or(&Value::Null);
        let config_json = elevator_config_json(config);
        let metrics_json = elevator_artifact_metrics_json(artifact);
        let dispatch_policy = meta_string(meta, "dispatchPolicy")
            .unwrap_or_else(|| config.dispatch_policy.label().to_string());
        let status = "completed";
        let seed = checked_i64(config.seed);
        let floors = meta_i32(meta, "floors").unwrap_or_else(|| checked_i32(config.floors.max(2)));
        let shafts = meta_i32(meta, "shafts").unwrap_or_else(|| checked_i32(config.shafts.max(1)));
        let capacity =
            meta_i32(meta, "capacity").unwrap_or_else(|| checked_i32(config.capacity.max(1)));
        let travel_seconds_micros = meta_micros(meta, "travel").unwrap_or_default().max(0);
        let dwell_seconds_micros = meta_micros(meta, "dwell").unwrap_or_default().max(0);
        let arrival_rate_micros = meta_micros(meta, "arrivalRate").unwrap_or_default().max(0);
        let horizon_seconds_micros = meta_micros(meta, "horizon").unwrap_or_default().max(0);
        let events = meta_i64(meta, "events").unwrap_or_default();
        let arrivals = meta_i64(meta, "arrivals").unwrap_or_default();
        let boarded = meta_i64(meta, "boarded").unwrap_or_default();
        let served = meta_i64(meta, "served").unwrap_or_default();
        let mean_wait_micros = meta_micros(meta, "meanWait").unwrap_or_default().max(0);
        let dispatch_decisions = meta_i32(meta, "dispatchDecisions")
            .unwrap_or_else(|| checked_i32(value_array_len(artifact, "decisions")));
        let pomdp_belief_updates = meta_i32(meta, "pomdpBeliefUpdates")
            .unwrap_or_else(|| checked_i32(value_array_len(artifact, "pomdpBeliefs")));
        let online_learning_updates = meta_i64(meta, "onlineLearningUpdates").unwrap_or_default();
        let online_learning_loss_last_micros =
            meta_micros(meta, "onlineLearningLossLast").map(|value| value.max(0));

        let mut tx = self
            .client
            .transaction()
            .map_err(|err| format!("begin FEL elevator learning transaction: {err}"))?;

        let row = tx
            .query_one(
                r#"
                insert into des_fel_elevator_learning_runs
                  (
                    run_label,
                    scenario_slug,
                    status,
                    dispatch_policy,
                    seed,
                    floors,
                    shafts,
                    capacity,
                    travel_seconds_micros,
                    dwell_seconds_micros,
                    arrival_rate_micros,
                    horizon_seconds_micros,
                    events,
                    arrivals,
                    boarded,
                    served,
                    mean_wait_micros,
                    dispatch_decisions,
                    pomdp_belief_updates,
                    online_learning_updates,
                    online_learning_loss_last_micros,
                    config,
                    metrics,
                    artifact
                  )
                values
                  (
                    $1,
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
                    $20,
                    $21,
                    $22,
                    $23,
                    $24
                  )
                returning id::text
                "#,
                &[
                    &run_label,
                    &scenario_slug,
                    &status,
                    &dispatch_policy,
                    &seed,
                    &floors,
                    &shafts,
                    &capacity,
                    &travel_seconds_micros,
                    &dwell_seconds_micros,
                    &arrival_rate_micros,
                    &horizon_seconds_micros,
                    &events,
                    &arrivals,
                    &boarded,
                    &served,
                    &mean_wait_micros,
                    &dispatch_decisions,
                    &pomdp_belief_updates,
                    &online_learning_updates,
                    &online_learning_loss_last_micros,
                    &config_json,
                    &metrics_json,
                    artifact,
                ],
            )
            .map_err(|err| format!("insert FEL elevator learning run: {err}"))?;
        let run_id: String = row.get(0);

        insert_policy_state(&mut tx, &run_id, artifact)?;
        insert_dispatch_decisions(&mut tx, &run_id, artifact)?;
        insert_pomdp_beliefs(&mut tx, &run_id, artifact)?;

        tx.commit()
            .map_err(|err| format!("commit FEL elevator learning transaction: {err}"))?;
        Ok(run_id)
    }
}

pub fn fel_elevator_learning_database_url() -> Option<String> {
    [
        "FEL_ELEVATOR_DATABASE_URL",
        "DES_FEL_ELEVATOR_DATABASE_URL",
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

pub fn elevator_config_json(config: &ElevatorConfig) -> Value {
    json!({
        "floors": config.floors,
        "shafts": config.shafts,
        "capacity": config.capacity,
        "travel": config.travel,
        "dwell": config.dwell,
        "arrivalRate": config.arrival_rate,
        "horizon": config.horizon,
        "seed": config.seed,
        "dispatchPolicy": config.dispatch_policy.label(),
    })
}

fn insert_policy_state(
    tx: &mut postgres::Transaction<'_>,
    run_id: &str,
    artifact: &Value,
) -> Result<(), String> {
    let Some(row) = policy_state_row_from_artifact(artifact) else {
        return Ok(());
    };
    let source_kind = "run-final";
    tx.execute(
        r#"
        insert into des_fel_elevator_policy_states
          (
            run_id,
            policy_kind,
            source_kind,
            feature_dim,
            output_dim,
            parameter_count,
            online_learning_updates,
            loss_history,
            state
          )
        values
          ($1::text::uuid, $2, $3, $4, $5, $6, $7, $8, $9)
        on conflict (run_id, source_kind, policy_kind) do update
        set
          feature_dim = excluded.feature_dim,
          output_dim = excluded.output_dim,
          parameter_count = excluded.parameter_count,
          online_learning_updates = excluded.online_learning_updates,
          loss_history = excluded.loss_history,
          state = excluded.state
        "#,
        &[
            &run_id,
            &row.policy_kind,
            &source_kind,
            &row.feature_dim,
            &row.output_dim,
            &row.parameter_count,
            &row.online_learning_updates,
            &row.loss_history,
            &row.state,
        ],
    )
    .map_err(|err| format!("insert FEL elevator policy state: {err}"))?;
    Ok(())
}

fn insert_dispatch_decisions(
    tx: &mut postgres::Transaction<'_>,
    run_id: &str,
    artifact: &Value,
) -> Result<(), String> {
    let Some(decisions) = artifact.get("decisions").and_then(Value::as_array) else {
        return Ok(());
    };
    for (index, decision) in decisions.iter().enumerate() {
        let decision_index = checked_i32(index);
        let sim_time_micros = value_micros(decision.get("t")).max(0);
        let call_floor = value_i32(decision.get("floor")).unwrap_or_default();
        let car_index = value_i32(decision.get("car")).unwrap_or_default();
        let policy_kind = value_string(decision.get("policy")).unwrap_or_else(|| "look".into());
        let meta_data = decision
            .as_object()
            .map(|_| decision.clone())
            .unwrap_or_else(|| json!({}));
        tx.execute(
            r#"
            insert into des_fel_elevator_dispatch_decisions
              (
                run_id,
                decision_index,
                sim_time_micros,
                call_floor,
                car_index,
                policy_kind,
                meta_data
              )
            values
              ($1::text::uuid, $2, $3, $4, $5, $6, $7)
            on conflict (run_id, decision_index) do update
            set
              sim_time_micros = excluded.sim_time_micros,
              call_floor = excluded.call_floor,
              car_index = excluded.car_index,
              policy_kind = excluded.policy_kind,
              meta_data = excluded.meta_data
            "#,
            &[
                &run_id,
                &decision_index,
                &sim_time_micros,
                &call_floor,
                &car_index,
                &policy_kind,
                &meta_data,
            ],
        )
        .map_err(|err| format!("insert FEL elevator dispatch decision: {err}"))?;
    }
    Ok(())
}

fn insert_pomdp_beliefs(
    tx: &mut postgres::Transaction<'_>,
    run_id: &str,
    artifact: &Value,
) -> Result<(), String> {
    let Some(beliefs) = artifact.get("pomdpBeliefs").and_then(Value::as_array) else {
        return Ok(());
    };
    for (index, trace) in beliefs.iter().enumerate() {
        let belief = trace
            .get("belief")
            .filter(|value| value.is_object())
            .cloned()
            .unwrap_or_else(|| json!({}));
        let belief_index = checked_i32(index);
        let sim_time_micros = value_micros(trace.get("t")).max(0);
        let floor = value_i32(trace.get("floor")).unwrap_or_default();
        let action = value_string(trace.get("action")).unwrap_or_else(|| "hold".into());
        let observation = value_string(trace.get("observation")).unwrap_or_else(|| "quiet".into());
        let empty_prob_micros = probability_micros(belief.get("empty"));
        let waiting_prob_micros = probability_micros(belief.get("waiting"));
        let crowded_prob_micros = probability_micros(belief.get("crowded"));
        tx.execute(
            r#"
            insert into des_fel_elevator_pomdp_beliefs
              (
                run_id,
                belief_index,
                sim_time_micros,
                floor,
                action,
                observation,
                empty_prob_micros,
                waiting_prob_micros,
                crowded_prob_micros,
                belief
              )
            values
              ($1::text::uuid, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            on conflict (run_id, belief_index) do update
            set
              sim_time_micros = excluded.sim_time_micros,
              floor = excluded.floor,
              action = excluded.action,
              observation = excluded.observation,
              empty_prob_micros = excluded.empty_prob_micros,
              waiting_prob_micros = excluded.waiting_prob_micros,
              crowded_prob_micros = excluded.crowded_prob_micros,
              belief = excluded.belief
            "#,
            &[
                &run_id,
                &belief_index,
                &sim_time_micros,
                &floor,
                &action,
                &observation,
                &empty_prob_micros,
                &waiting_prob_micros,
                &crowded_prob_micros,
                &belief,
            ],
        )
        .map_err(|err| format!("insert FEL elevator POMDP belief: {err}"))?;
    }
    Ok(())
}

fn policy_state_row_from_artifact(artifact: &Value) -> Option<PolicyStateRow> {
    let state = artifact.get("policyState")?.as_object()?;
    let policy_kind = state
        .get("kind")
        .and_then(Value::as_str)
        .or_else(|| {
            artifact
                .get("meta")
                .and_then(|meta| meta.get("dispatchPolicy"))
                .and_then(Value::as_str)
        })?
        .to_string();
    let network = state.get("network");
    let feature_dim = value_i32(network.and_then(|n| n.get("inputDim"))).unwrap_or_default();
    let output_dim = value_i32(network.and_then(|n| n.get("outputDim"))).unwrap_or_default();
    let table_entries = state
        .get("table")
        .and_then(Value::as_array)
        .map(|values| values.len())
        .unwrap_or_default();
    let parameter_count = value_i32(network.and_then(|n| n.get("parameterCount")))
        .unwrap_or_else(|| checked_i32(table_entries));
    let online_learning_updates = value_i64(state.get("updates"))
        .or_else(|| {
            artifact
                .get("meta")
                .and_then(|meta| value_i64(meta.get("onlineLearningUpdates")))
        })
        .unwrap_or_default();
    let loss_history = state
        .get("lossHistory")
        .filter(|value| value.is_array())
        .cloned()
        .unwrap_or_else(|| json!([]));
    Some(PolicyStateRow {
        policy_kind,
        feature_dim,
        output_dim,
        parameter_count,
        online_learning_updates,
        loss_history,
        state: Value::Object(state.clone()),
    })
}

fn elevator_artifact_metrics_json(artifact: &Value) -> Value {
    json!({
        "meta": artifact.get("meta").cloned().unwrap_or_else(|| json!({})),
        "frameCount": value_array_len(artifact, "frames"),
        "decisionCount": value_array_len(artifact, "decisions"),
        "pomdpBeliefCount": value_array_len(artifact, "pomdpBeliefs"),
        "hasPolicyState": artifact.get("policyState").is_some_and(Value::is_object),
    })
}

fn fel_elevator_pg_should_verify_certificates(database_url: &str) -> bool {
    fel_elevator_pg_sslmode(database_url).is_some_and(|sslmode| {
        sslmode.eq_ignore_ascii_case("verify-ca") || sslmode.eq_ignore_ascii_case("verify-full")
    })
}

fn fel_elevator_pg_sslmode(database_url: &str) -> Option<&str> {
    let query = database_url.split_once('?')?.1;
    query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=').unwrap_or((part, ""));
        key.eq_ignore_ascii_case("sslmode").then_some(value)
    })
}

fn meta_string(meta: &Value, key: &str) -> Option<String> {
    value_string(meta.get(key))
}

fn meta_i32(meta: &Value, key: &str) -> Option<i32> {
    value_i32(meta.get(key))
}

fn meta_i64(meta: &Value, key: &str) -> Option<i64> {
    value_i64(meta.get(key))
}

fn meta_micros(meta: &Value, key: &str) -> Option<i64> {
    meta.get(key).and_then(Value::as_f64).map(to_micros)
}

fn value_string(value: Option<&Value>) -> Option<String> {
    value.and_then(Value::as_str).map(str::to_string)
}

fn value_i32(value: Option<&Value>) -> Option<i32> {
    value.and_then(Value::as_u64).map(checked_i32)
}

fn value_i64(value: Option<&Value>) -> Option<i64> {
    value.and_then(Value::as_u64).map(checked_i64)
}

fn value_micros(value: Option<&Value>) -> i64 {
    value
        .and_then(Value::as_f64)
        .map(to_micros)
        .unwrap_or_default()
}

fn probability_micros(value: Option<&Value>) -> i32 {
    value
        .and_then(Value::as_f64)
        .map(|value| to_micros(value.clamp(0.0, 1.0)).clamp(0, 1_000_000) as i32)
        .unwrap_or_default()
}

fn value_array_len(value: &Value, key: &str) -> usize {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or_default()
}

fn to_micros(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    let scaled = (value * FEL_FIXED_SCALE).round();
    if scaled > i64::MAX as f64 {
        i64::MAX
    } else if scaled < i64::MIN as f64 {
        i64::MIN
    } else {
        scaled as i64
    }
}

fn checked_i32(value: impl TryInto<u64>) -> i32 {
    let value = value.try_into().unwrap_or(u64::MAX);
    value.min(i32::MAX as u64) as i32
}

fn checked_i64(value: impl TryInto<u64>) -> i64 {
    let value = value.try_into().unwrap_or(u64::MAX);
    value.min(i64::MAX as u64) as i64
}

#[cfg(test)]
mod tests {
    use super::super::elevator::{
        elevator_neural_td_dispatch_policy, run_fel_elevator_with_policy, ElevatorConfig,
        ElevatorNeuralTdDispatchOptions,
    };
    use super::*;

    #[test]
    fn fel_elevator_pg_sslmode_parses_verify_modes() {
        assert_eq!(
            fel_elevator_pg_sslmode("postgres://u:p@host/db?sslmode=require"),
            Some("require")
        );
        assert!(fel_elevator_pg_should_verify_certificates(
            "postgres://u:p@host/db?sslmode=verify-full"
        ));
        assert!(!fel_elevator_pg_should_verify_certificates(
            "postgres://u:p@host/db?sslmode=require"
        ));
    }

    #[test]
    fn policy_state_row_extracts_neural_td_weights() {
        let cfg = ElevatorConfig {
            floors: 4,
            shafts: 2,
            horizon: 30.0,
            seed: 57,
            ..Default::default()
        };
        let artifact = run_fel_elevator_with_policy(
            &cfg,
            elevator_neural_td_dispatch_policy(&ElevatorNeuralTdDispatchOptions {
                learning_rate: 0.03,
                gamma: 0.82,
                hidden_layers: vec![5],
                seed: 41,
            }),
        );
        let row = policy_state_row_from_artifact(&artifact).expect("policy state");
        assert_eq!(row.policy_kind, "neural-td");
        assert_eq!(row.feature_dim, 10);
        assert_eq!(row.output_dim, 1);
        assert!(row.parameter_count > 0);
        assert!(row.online_learning_updates > 0);
        assert!(row.loss_history.as_array().unwrap().len() > 0);
        assert!(row.state["network"]["layers"].as_array().unwrap().len() > 0);
    }

    #[test]
    fn fixed_point_helpers_clamp_probabilities() {
        assert_eq!(to_micros(1.25), 1_250_000);
        assert_eq!(probability_micros(Some(&json!(1.2))), 1_000_000);
        assert_eq!(probability_micros(Some(&json!(-0.2))), 0);
    }
}
