//! Port of `src/des/general/adapters/stochastic-flow-mdp-adapter.ts`
//! (module `des::general::adapters::stochastic_flow_mdp_adapter`).
//!
//! JSON adapter registering the stochastic-flow MDP (max-flow under stochastic
//! edge availability).
//!
//! ## Conversion notes
//!
//!   * `builtin: 'small-stochastic-network'` literal -> [`StochasticFlowBuiltin`].
//!   * `runtime.seed ?? params.seed ?? 7` -> `Option::or` chain coerced to `u32`.
//!   * `row.state.capacities` is JSON-encoded into the CSV (the TS `jsonCsvRow`);
//!     here the array is formatted to a JSON-style string at the call site.
//!   * NOTE (as in the TS source): near-duplicate of the stochastic-flow-mdp
//!     block in `network-flow-adapters.ts`; both register id `stochastic-flow-mdp`.
//!
//! PORT NOTE: `registerModel` / the model registry is not ported yet; the
//! adapter is exposed via [`adapter()`] for the integrator to wire in.

#![allow(dead_code)]

use crate::des::general::adapters::adapter_utils::{json_csv_row, write_csv_lines};
use crate::des::general::des_spec::{
    DESModelRegistration, DESModelSpec, DESRuntimeConfig, ParamSchema, RegistrationExample,
    DES_MODEL_SPEC_SCHEMA,
};
use crate::des::general::stochastic_flow_mdp::{
    build_default_stochastic_flow_mdp_problem, solve_stochastic_flow_mdp,
    SolveStochasticFlowMDPOptions, StochasticFlowMDPProblem, StochasticFlowMDPResult,
};

/// `builtin?: 'small-stochastic-network'`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StochasticFlowBuiltin {
    SmallStochasticNetwork,
}

/// `interface StochasticFlowMDPParams`.
#[derive(Clone, Debug, Default)]
pub struct StochasticFlowMDPParams {
    pub builtin: Option<StochasticFlowBuiltin>,
    pub problem: Option<StochasticFlowMDPProblem>,
    pub seed: Option<u32>,
    pub max_policy_rows: Option<usize>,
}

fn num(
    min: Option<f64>,
    max: Option<f64>,
    integer: Option<bool>,
    default: Option<f64>,
) -> ParamSchema {
    ParamSchema::Number {
        min,
        max,
        integer,
        default,
        description: None,
    }
}

fn stochastic_flow_edge_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            ("from".to_string(), num(Some(0.0), None, Some(true), None)),
            ("to".to_string(), num(Some(0.0), None, Some(true), None)),
            (
                "capacity".to_string(),
                num(Some(0.0), None, Some(true), None),
            ),
            (
                "successProb".to_string(),
                num(Some(0.0), Some(1.0), None, None),
            ),
            ("cost".to_string(), num(Some(0.0), None, None, None)),
            (
                "name".to_string(),
                ParamSchema::String {
                    allowed: None,
                    default: None,
                    description: None,
                },
            ),
        ],
        required: Some(vec![
            "from".to_string(),
            "to".to_string(),
            "capacity".to_string(),
            "successProb".to_string(),
        ]),
        description: None,
    }
}

fn stochastic_flow_problem_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            (
                "numNodes".to_string(),
                num(Some(2.0), None, Some(true), None),
            ),
            ("source".to_string(), num(Some(0.0), None, Some(true), None)),
            ("sink".to_string(), num(Some(0.0), None, Some(true), None)),
            (
                "edges".to_string(),
                ParamSchema::Array {
                    items: Box::new(stochastic_flow_edge_schema()),
                    min_length: Some(1),
                    max_length: None,
                    description: None,
                },
            ),
            (
                "horizon".to_string(),
                num(Some(1.0), None, Some(true), None),
            ),
            (
                "deliveredReward".to_string(),
                num(Some(1e-9), None, None, None),
            ),
            ("waitPenalty".to_string(), num(Some(0.0), None, None, None)),
            (
                "failurePenalty".to_string(),
                num(Some(0.0), None, None, None),
            ),
            (
                "discount".to_string(),
                num(Some(0.0), Some(1.0), None, None),
            ),
            (
                "maxStates".to_string(),
                num(Some(1.0), None, Some(true), None),
            ),
        ],
        required: Some(vec![
            "numNodes".to_string(),
            "source".to_string(),
            "sink".to_string(),
            "edges".to_string(),
            "horizon".to_string(),
        ]),
        description: Some(
            "Finite-horizon stochastic flow-control MDP on a directed network.".to_string(),
        ),
    }
}

/// `const stochasticFlowMDPSchema`.
pub fn stochastic_flow_mdp_schema() -> ParamSchema {
    ParamSchema::Object {
        fields: vec![
            (
                "builtin".to_string(),
                ParamSchema::String {
                    allowed: Some(vec!["small-stochastic-network".to_string()]),
                    default: Some("small-stochastic-network".to_string()),
                    description: None,
                },
            ),
            ("problem".to_string(), stochastic_flow_problem_schema()),
            ("seed".to_string(), num(None, None, Some(true), Some(7.0))),
            (
                "maxPolicyRows".to_string(),
                num(Some(1.0), None, Some(true), Some(24.0)),
            ),
        ],
        required: Some(vec![]),
        description: Some(
            "MDP interpretation of max-flow when edge availability/capacity is stochastic."
                .to_string(),
        ),
    }
}

/// `const stochasticFlowMDPAdapter`.
pub struct StochasticFlowMDPAdapter;

/// Construct the adapter (see the module's PORT NOTE about registration).
pub fn adapter() -> StochasticFlowMDPAdapter {
    StochasticFlowMDPAdapter
}

impl DESModelRegistration<StochasticFlowMDPParams, StochasticFlowMDPResult>
    for StochasticFlowMDPAdapter
{
    fn id(&self) -> &str {
        "stochastic-flow-mdp"
    }

    fn description(&self) -> &str {
        "MDP interpretation of max-flow: stochastic capacities/availability with sequential routing control."
    }

    fn schema(&self) -> ParamSchema {
        stochastic_flow_mdp_schema()
    }

    fn run(
        &self,
        params: StochasticFlowMDPParams,
        runtime: &DESRuntimeConfig,
    ) -> StochasticFlowMDPResult {
        let seed = runtime.seed.map(|s| s as u32).or(params.seed).unwrap_or(7);
        let problem = params
            .problem
            .unwrap_or_else(build_default_stochastic_flow_mdp_problem);
        solve_stochastic_flow_mdp(
            problem,
            SolveStochasticFlowMDPOptions {
                seed: Some(seed),
                max_policy_rows: Some(params.max_policy_rows.unwrap_or(24)),
            },
        )
    }

    fn summarize(
        &self,
        result: &StochasticFlowMDPResult,
        _params: &StochasticFlowMDPParams,
    ) -> String {
        let first = result
            .initial_policy
            .iter()
            .take(5)
            .map(|row| format!("t{}:{}", row.stage, row.action.label))
            .collect::<Vec<_>>()
            .join(" -> ");
        [
            "STOCHASTIC FLOW-CONTROL MDP".to_string(),
            "---------------------------".to_string(),
            format!("  Horizon:         {}", result.horizon),
            format!("  States:          {}", result.num_states),
            format!("  E[reward]*:      {:.6}", result.expected_reward),
            format!(
                "  Static max-flow: {:.6}  (deterministic upper bound)",
                result.deterministic_max_flow
            ),
            format!("  First policy:    {first}"),
            format!("  Sim delivered:   {}", result.simulation.delivered),
            format!("  Sim reward:      {:.6}", result.simulation.total_reward),
        ]
        .join("\n")
    }

    fn write_csv(&self, result: &StochasticFlowMDPResult, csv_path: &str) {
        let mut lines = vec!["stage,state_index,node,capacities,action,value".to_string()];
        for row in &result.policy {
            let caps = format!(
                "[{}]",
                row.state
                    .capacities
                    .iter()
                    .map(|c| c.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            );
            lines.push(json_csv_row([
                row.stage.to_string(),
                row.state_index.to_string(),
                row.state.node.to_string(),
                caps,
                row.action.label.clone(),
                row.value.to_string(),
            ]));
        }
        write_csv_lines(csv_path, &lines);
    }

    fn examples(&self) -> Vec<RegistrationExample<StochasticFlowMDPParams>> {
        vec![RegistrationExample {
            name: "small-stochastic-network".to_string(),
            spec: DESModelSpec {
                schema: DES_MODEL_SPEC_SCHEMA.to_string(),
                model: "stochastic-flow-mdp".to_string(),
                description: Some(
                    "MDP interpretation of max-flow with stochastic edge availability.".to_string(),
                ),
                parameters: StochasticFlowMDPParams {
                    builtin: Some(StochasticFlowBuiltin::SmallStochasticNetwork),
                    problem: None,
                    seed: Some(7),
                    max_policy_rows: None,
                },
                runtime: Some(DESRuntimeConfig {
                    seed: Some(7.0),
                    ..Default::default()
                }),
                metadata: None,
            },
        }]
    }
}
