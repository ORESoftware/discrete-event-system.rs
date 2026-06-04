//! Decision-analysis kernels: expected value, expected utility, and value of
//! information calculations for small decision tables.

const EPS: f64 = 1e-9;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Utility {
    /// `u(x) = x`.
    RiskNeutral,
    /// Constant absolute risk aversion:
    /// `u(x) = 1 - exp(-x / risk_tolerance)`.
    Exponential { risk_tolerance: f64 },
    /// Log utility over shifted wealth: `u(x)=ln(x + shift)`.
    Log { shift: f64 },
}

impl Utility {
    pub fn utility(self, x: f64) -> Result<f64, String> {
        if !x.is_finite() {
            return Err(format!("utility input must be finite; got {x}"));
        }
        match self {
            Utility::RiskNeutral => Ok(x),
            Utility::Exponential { risk_tolerance } => {
                if risk_tolerance <= 0.0 || !risk_tolerance.is_finite() {
                    Err("risk_tolerance must be positive and finite".to_string())
                } else {
                    Ok(1.0 - (-x / risk_tolerance).exp())
                }
            }
            Utility::Log { shift } => {
                if !shift.is_finite() {
                    return Err("log utility shift must be finite".to_string());
                }
                if x + shift <= 0.0 {
                    Err(format!(
                        "log utility requires x + shift > 0; got x={x}, shift={shift}"
                    ))
                } else {
                    Ok((x + shift).ln())
                }
            }
        }
    }

    pub fn inverse(self, u: f64) -> Result<f64, String> {
        if !u.is_finite() {
            return Err(format!("utility value must be finite; got {u}"));
        }
        match self {
            Utility::RiskNeutral => Ok(u),
            Utility::Exponential { risk_tolerance } => {
                if risk_tolerance <= 0.0 || !risk_tolerance.is_finite() {
                    Err("risk_tolerance must be positive and finite".to_string())
                } else if u >= 1.0 {
                    Err("exponential utility inverse requires u < 1".to_string())
                } else {
                    Ok(-risk_tolerance * (1.0 - u).ln())
                }
            }
            Utility::Log { shift } => {
                if !shift.is_finite() {
                    return Err("log utility shift must be finite".to_string());
                }
                let x = u.exp() - shift;
                if x.is_finite() {
                    Ok(x)
                } else {
                    Err("log utility inverse overflowed".to_string())
                }
            }
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProbabilityOutcome {
    pub probability: f64,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionAlternative {
    pub name: String,
    pub outcomes: Vec<ProbabilityOutcome>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlternativeEvaluation {
    pub name: String,
    pub expected_value: f64,
    pub expected_utility: f64,
    pub certainty_equivalent: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DecisionTable {
    /// Prior probability per state of nature.
    pub state_probabilities: Vec<f64>,
    /// `payoffs[action][state]`.
    pub payoffs: Vec<Vec<f64>>,
    pub action_names: Vec<String>,
    pub state_names: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValueOfInformation {
    pub current_best_action: usize,
    pub current_best_value: f64,
    pub value_with_information: f64,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SignalModel {
    pub table: DecisionTable,
    /// `likelihood[state][signal] = P(signal | state)`.
    pub likelihood: Vec<Vec<f64>>,
    pub signal_names: Vec<String>,
}

pub fn expected_value(outcomes: &[ProbabilityOutcome]) -> Result<f64, String> {
    validate_outcomes(outcomes)?;
    Ok(outcomes.iter().map(|o| o.probability * o.value).sum())
}

pub fn expected_utility(outcomes: &[ProbabilityOutcome], utility: Utility) -> Result<f64, String> {
    validate_outcomes(outcomes)?;
    outcomes.iter().try_fold(0.0, |acc, o| {
        Ok(acc + o.probability * utility.utility(o.value)?)
    })
}

pub fn certainty_equivalent(
    outcomes: &[ProbabilityOutcome],
    utility: Utility,
) -> Result<f64, String> {
    utility.inverse(expected_utility(outcomes, utility)?)
}

pub fn evaluate_alternatives(
    alternatives: &[DecisionAlternative],
    utility: Utility,
) -> Result<Vec<AlternativeEvaluation>, String> {
    if alternatives.is_empty() {
        return Err("alternatives must be non-empty".to_string());
    }
    let mut rows = Vec::with_capacity(alternatives.len());
    for alt in alternatives {
        rows.push(AlternativeEvaluation {
            name: alt.name.clone(),
            expected_value: expected_value(&alt.outcomes)?,
            expected_utility: expected_utility(&alt.outcomes, utility)?,
            certainty_equivalent: certainty_equivalent(&alt.outcomes, utility)?,
        });
    }
    rows.sort_by(|a, b| {
        b.expected_utility
            .partial_cmp(&a.expected_utility)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(rows)
}

pub fn expected_values_by_action(table: &DecisionTable) -> Result<Vec<f64>, String> {
    validate_table(table)?;
    Ok(table
        .payoffs
        .iter()
        .map(|row| {
            row.iter()
                .zip(&table.state_probabilities)
                .map(|(x, p)| x * p)
                .sum()
        })
        .collect())
}

pub fn expected_value_of_perfect_information(
    table: &DecisionTable,
) -> Result<ValueOfInformation, String> {
    validate_table(table)?;
    let current = expected_values_by_action(table)?;
    let (best_action, current_best_value) = argmax(&current)
        .ok_or_else(|| "decision table must contain at least one action".to_string())?;
    let mut with_info = 0.0;
    for s in 0..table.state_probabilities.len() {
        let best_state_payoff = table
            .payoffs
            .iter()
            .map(|row| row[s])
            .fold(f64::NEG_INFINITY, f64::max);
        with_info += table.state_probabilities[s] * best_state_payoff;
    }
    Ok(ValueOfInformation {
        current_best_action: best_action,
        current_best_value,
        value_with_information: with_info,
        value: with_info - current_best_value,
    })
}

pub fn expected_value_of_sample_information(
    model: &SignalModel,
) -> Result<ValueOfInformation, String> {
    validate_signal_model(model)?;
    let table = &model.table;
    let current = expected_values_by_action(table)?;
    let (best_action, current_best_value) = argmax(&current)
        .ok_or_else(|| "decision table must contain at least one action".to_string())?;

    let num_signals = model.signal_names.len();
    let mut with_sample = 0.0;
    for y in 0..num_signals {
        let p_signal: f64 = table
            .state_probabilities
            .iter()
            .enumerate()
            .map(|(s, p)| p * model.likelihood[s][y])
            .sum();
        if p_signal <= EPS {
            continue;
        }
        let mut best_conditional = f64::NEG_INFINITY;
        for action in &table.payoffs {
            let ev_given_signal: f64 = action
                .iter()
                .enumerate()
                .map(|(s, payoff)| {
                    let posterior =
                        table.state_probabilities[s] * model.likelihood[s][y] / p_signal;
                    posterior * payoff
                })
                .sum();
            best_conditional = best_conditional.max(ev_given_signal);
        }
        with_sample += p_signal * best_conditional;
    }

    Ok(ValueOfInformation {
        current_best_action: best_action,
        current_best_value,
        value_with_information: with_sample,
        value: with_sample - current_best_value,
    })
}

fn validate_outcomes(outcomes: &[ProbabilityOutcome]) -> Result<(), String> {
    if outcomes.is_empty() {
        return Err("outcomes must be non-empty".to_string());
    }
    let mut sum = 0.0;
    for (i, o) in outcomes.iter().enumerate() {
        if !o.probability.is_finite() || o.probability < -EPS {
            return Err(format!(
                "outcomes[{i}].probability must be non-negative and finite"
            ));
        }
        if !o.value.is_finite() {
            return Err(format!("outcomes[{i}].value must be finite"));
        }
        sum += o.probability;
    }
    if (sum - 1.0).abs() > 1e-7 {
        return Err(format!("outcome probabilities sum to {sum}, expected 1"));
    }
    Ok(())
}

fn validate_table(table: &DecisionTable) -> Result<(), String> {
    if table.state_probabilities.is_empty() {
        return Err("state_probabilities must be non-empty".to_string());
    }
    if table.payoffs.is_empty() {
        return Err("payoffs must contain at least one action".to_string());
    }
    if !table.action_names.is_empty() && table.action_names.len() != table.payoffs.len() {
        return Err(format!(
            "action_names has length {}, expected {}",
            table.action_names.len(),
            table.payoffs.len()
        ));
    }
    if !table.state_names.is_empty() && table.state_names.len() != table.state_probabilities.len() {
        return Err(format!(
            "state_names has length {}, expected {}",
            table.state_names.len(),
            table.state_probabilities.len()
        ));
    }
    let sum: f64 = table.state_probabilities.iter().sum();
    if (sum - 1.0).abs() > 1e-7 {
        return Err(format!("state probabilities sum to {sum}, expected 1"));
    }
    for (s, p) in table.state_probabilities.iter().enumerate() {
        if *p < -EPS || !p.is_finite() {
            return Err(format!(
                "state_probabilities[{s}] must be non-negative and finite"
            ));
        }
    }
    for (a, row) in table.payoffs.iter().enumerate() {
        if row.len() != table.state_probabilities.len() {
            return Err(format!(
                "payoffs[{a}] has {} states, expected {}",
                row.len(),
                table.state_probabilities.len()
            ));
        }
        if row.iter().any(|x| !x.is_finite()) {
            return Err(format!("payoffs[{a}] contains a non-finite payoff"));
        }
    }
    Ok(())
}

fn validate_signal_model(model: &SignalModel) -> Result<(), String> {
    validate_table(&model.table)?;
    let states = model.table.state_probabilities.len();
    if model.signal_names.is_empty() {
        return Err("signal_names must be non-empty".to_string());
    }
    if model.likelihood.len() != states {
        return Err(format!(
            "likelihood has {} state rows, expected {states}",
            model.likelihood.len()
        ));
    }
    for (s, row) in model.likelihood.iter().enumerate() {
        if row.len() != model.signal_names.len() {
            return Err(format!(
                "likelihood[{s}] has {} signals, expected {}",
                row.len(),
                model.signal_names.len()
            ));
        }
        let sum: f64 = row.iter().sum();
        for (y, p) in row.iter().enumerate() {
            if *p < -EPS || !p.is_finite() {
                return Err(format!(
                    "likelihood[{s}][{y}] must be non-negative and finite"
                ));
            }
        }
        if (sum - 1.0).abs() > 1e-7 {
            return Err(format!("likelihood[{s}] sums to {sum}, expected 1"));
        }
    }
    Ok(())
}

fn argmax(values: &[f64]) -> Option<(usize, f64)> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, v)| (i, *v))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn risk_averse_ce_is_below_expected_value() {
        let outcomes = vec![
            ProbabilityOutcome {
                probability: 0.5,
                value: 0.0,
            },
            ProbabilityOutcome {
                probability: 0.5,
                value: 100.0,
            },
        ];
        let ev = expected_value(&outcomes).unwrap();
        let ce = certainty_equivalent(
            &outcomes,
            Utility::Exponential {
                risk_tolerance: 40.0,
            },
        )
        .unwrap();
        assert!(ce < ev);
    }

    #[test]
    fn computes_evpi_and_evsi() {
        let table = DecisionTable {
            state_probabilities: vec![0.6, 0.4],
            payoffs: vec![vec![100.0, -20.0], vec![20.0, 20.0]],
            action_names: vec!["drill".to_string(), "sell".to_string()],
            state_names: vec!["wet".to_string(), "dry".to_string()],
        };
        let evpi = expected_value_of_perfect_information(&table).unwrap();
        assert_eq!(evpi.current_best_action, 0);
        assert!((evpi.value - 16.0).abs() < 1e-10);

        let evsi = expected_value_of_sample_information(&SignalModel {
            table,
            likelihood: vec![vec![0.8, 0.2], vec![0.25, 0.75]],
            signal_names: vec!["good".to_string(), "bad".to_string()],
        })
        .unwrap();
        assert!(evsi.value > 0.0);
        assert!(evsi.value <= evpi.value + 1e-10);
    }

    #[test]
    fn sample_information_rejects_bad_likelihood_entries() {
        let table = DecisionTable {
            state_probabilities: vec![0.5, 0.5],
            payoffs: vec![vec![1.0, 0.0]],
            action_names: vec!["act".to_string()],
            state_names: vec!["a".to_string(), "b".to_string()],
        };
        let err = expected_value_of_sample_information(&SignalModel {
            table,
            likelihood: vec![vec![1.2, -0.2], vec![0.5, 0.5]],
            signal_names: vec!["y".to_string(), "n".to_string()],
        })
        .unwrap_err();
        assert!(err.contains("likelihood[0][1]"));
    }

    #[test]
    fn decision_table_rejects_label_shape_mismatch() {
        let err = expected_values_by_action(&DecisionTable {
            state_probabilities: vec![1.0],
            payoffs: vec![vec![1.0]],
            action_names: vec!["a".to_string(), "extra".to_string()],
            state_names: vec!["s".to_string()],
        })
        .unwrap_err();
        assert!(err.contains("action_names"));
    }
}
