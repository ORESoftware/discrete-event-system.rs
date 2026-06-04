//! Decision-engineering helpers for framing alternatives, objectives, tradeoffs,
//! and sensitivity checks.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreferenceDirection {
    Maximize,
    Minimize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Objective {
    pub name: String,
    pub weight: f64,
    pub direction: PreferenceDirection,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Alternative {
    pub name: String,
    pub criteria: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AlternativeScore {
    pub index: usize,
    pub name: String,
    pub score: f64,
    pub normalized_criteria: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TornadoEntry {
    pub objective: String,
    pub low_weight_best: usize,
    pub high_weight_best: usize,
    pub score_span: f64,
}

pub fn normalize_weights(objectives: &[Objective]) -> Result<Vec<f64>, String> {
    if objectives.is_empty() {
        return Err("objectives must be non-empty".to_string());
    }
    for (i, o) in objectives.iter().enumerate() {
        if o.weight < 0.0 || !o.weight.is_finite() {
            return Err(format!(
                "objectives[{i}].weight must be non-negative and finite"
            ));
        }
    }
    let total: f64 = objectives.iter().map(|o| o.weight).sum();
    if total <= 0.0 || !total.is_finite() {
        return Err("objective weights must sum to a positive finite value".to_string());
    }
    Ok(objectives.iter().map(|o| o.weight / total).collect())
}

/// Min-max normalize criteria by objective direction, then compute weighted
/// additive scores. Higher scores are better.
pub fn weighted_additive_scores(
    alternatives: &[Alternative],
    objectives: &[Objective],
) -> Result<Vec<AlternativeScore>, String> {
    validate_matrix(alternatives, objectives)?;
    let weights = normalize_weights(objectives)?;
    let m = objectives.len();
    let mut minv = vec![f64::INFINITY; m];
    let mut maxv = vec![f64::NEG_INFINITY; m];
    for alt in alternatives {
        for j in 0..m {
            minv[j] = minv[j].min(alt.criteria[j]);
            maxv[j] = maxv[j].max(alt.criteria[j]);
        }
    }

    let mut rows = Vec::with_capacity(alternatives.len());
    for (i, alt) in alternatives.iter().enumerate() {
        let mut normalized = vec![0.0; m];
        for j in 0..m {
            let spread = maxv[j] - minv[j];
            normalized[j] = if spread.abs() <= 1e-12 {
                1.0
            } else {
                let raw = (alt.criteria[j] - minv[j]) / spread;
                match objectives[j].direction {
                    PreferenceDirection::Maximize => raw,
                    PreferenceDirection::Minimize => 1.0 - raw,
                }
            };
        }
        let score = normalized.iter().zip(&weights).map(|(x, w)| x * w).sum();
        rows.push(AlternativeScore {
            index: i,
            name: alt.name.clone(),
            score,
            normalized_criteria: normalized,
        });
    }
    rows.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.index.cmp(&b.index))
    });
    Ok(rows)
}

/// Return indices of alternatives that are not dominated across all objectives.
pub fn pareto_frontier(
    alternatives: &[Alternative],
    directions: &[PreferenceDirection],
) -> Result<Vec<usize>, String> {
    if alternatives.is_empty() {
        return Err("alternatives must be non-empty".to_string());
    }
    if directions.is_empty() {
        return Err("directions must be non-empty".to_string());
    }
    for (i, alt) in alternatives.iter().enumerate() {
        if alt.criteria.len() != directions.len() {
            return Err(format!(
                "alternatives[{i}] has {} criteria, expected {}",
                alt.criteria.len(),
                directions.len()
            ));
        }
        if alt.criteria.iter().any(|x| !x.is_finite()) {
            return Err(format!("alternatives[{i}] contains a non-finite criterion"));
        }
    }

    let mut frontier = Vec::new();
    'candidate: for i in 0..alternatives.len() {
        for j in 0..alternatives.len() {
            if i == j {
                continue;
            }
            if dominates(
                &alternatives[j].criteria,
                &alternatives[i].criteria,
                directions,
            ) {
                continue 'candidate;
            }
        }
        frontier.push(i);
    }
    Ok(frontier)
}

/// One-way weight sensitivity: vary each objective to `low` and `high`, scaling
/// other weights proportionally, and report whether the best alternative flips.
pub fn one_way_weight_tornado(
    alternatives: &[Alternative],
    objectives: &[Objective],
    low: f64,
    high: f64,
) -> Result<Vec<TornadoEntry>, String> {
    if !(0.0..=1.0).contains(&low) || !(0.0..=1.0).contains(&high) || low > high {
        return Err("low/high must satisfy 0 <= low <= high <= 1".to_string());
    }
    validate_matrix(alternatives, objectives)?;
    let base_weights = normalize_weights(objectives)?;
    let mut entries = Vec::with_capacity(objectives.len());
    for j in 0..objectives.len() {
        let low_weights = replace_weight(&base_weights, j, low);
        let high_weights = replace_weight(&base_weights, j, high);
        let low_scores = weighted_scores_with_weights(alternatives, objectives, &low_weights)?;
        let high_scores = weighted_scores_with_weights(alternatives, objectives, &high_weights)?;
        let low_best = low_scores[0].index;
        let high_best = high_scores[0].index;
        let score_span = alternatives
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let low = low_scores
                    .iter()
                    .find(|row| row.index == i)
                    .expect("every alternative has a low-weight score");
                let high = high_scores
                    .iter()
                    .find(|row| row.index == i)
                    .expect("every alternative has a high-weight score");
                (high.score - low.score).abs()
            })
            .fold(0.0_f64, f64::max);
        entries.push(TornadoEntry {
            objective: objectives[j].name.clone(),
            low_weight_best: low_best,
            high_weight_best: high_best,
            score_span,
        });
    }
    entries.sort_by(|a, b| {
        b.score_span
            .partial_cmp(&a.score_span)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.objective.cmp(&b.objective))
    });
    Ok(entries)
}

fn validate_matrix(alternatives: &[Alternative], objectives: &[Objective]) -> Result<(), String> {
    if alternatives.is_empty() {
        return Err("alternatives must be non-empty".to_string());
    }
    if objectives.is_empty() {
        return Err("objectives must be non-empty".to_string());
    }
    for (i, alt) in alternatives.iter().enumerate() {
        if alt.criteria.len() != objectives.len() {
            return Err(format!(
                "alternatives[{i}] has {} criteria, expected {}",
                alt.criteria.len(),
                objectives.len()
            ));
        }
        if alt.criteria.iter().any(|x| !x.is_finite()) {
            return Err(format!("alternatives[{i}] contains a non-finite criterion"));
        }
    }
    normalize_weights(objectives)?;
    Ok(())
}

fn dominates(a: &[f64], b: &[f64], directions: &[PreferenceDirection]) -> bool {
    let mut strictly_better = false;
    for k in 0..directions.len() {
        match directions[k] {
            PreferenceDirection::Maximize => {
                if a[k] < b[k] {
                    return false;
                }
                strictly_better |= a[k] > b[k];
            }
            PreferenceDirection::Minimize => {
                if a[k] > b[k] {
                    return false;
                }
                strictly_better |= a[k] < b[k];
            }
        }
    }
    strictly_better
}

fn replace_weight(weights: &[f64], index: usize, value: f64) -> Vec<f64> {
    if weights.len() == 1 {
        return vec![1.0];
    }
    let rest_total: f64 = weights
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != index)
        .map(|(_, w)| *w)
        .sum();
    let mut out = vec![0.0; weights.len()];
    out[index] = value;
    let remaining = 1.0 - value;
    for i in 0..weights.len() {
        if i != index {
            out[i] = if rest_total <= 1e-12 {
                remaining / (weights.len() - 1) as f64
            } else {
                remaining * weights[i] / rest_total
            };
        }
    }
    out
}

fn weighted_scores_with_weights(
    alternatives: &[Alternative],
    objectives: &[Objective],
    weights: &[f64],
) -> Result<Vec<AlternativeScore>, String> {
    let adjusted: Vec<Objective> = objectives
        .iter()
        .zip(weights)
        .map(|(o, w)| Objective {
            name: o.name.clone(),
            weight: *w,
            direction: o.direction,
        })
        .collect();
    weighted_additive_scores(alternatives, &adjusted)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn objectives() -> Vec<Objective> {
        vec![
            Objective {
                name: "value".to_string(),
                weight: 0.6,
                direction: PreferenceDirection::Maximize,
            },
            Objective {
                name: "cost".to_string(),
                weight: 0.4,
                direction: PreferenceDirection::Minimize,
            },
        ]
    }

    #[test]
    fn weighted_scores_sort_best_first() {
        let alternatives = vec![
            Alternative {
                name: "A".to_string(),
                criteria: vec![90.0, 50.0],
            },
            Alternative {
                name: "B".to_string(),
                criteria: vec![70.0, 20.0],
            },
        ];
        let scores = weighted_additive_scores(&alternatives, &objectives()).unwrap();
        assert_eq!(scores[0].name, "A");
    }

    #[test]
    fn pareto_frontier_drops_dominated_alternatives() {
        let alternatives = vec![
            Alternative {
                name: "dominating".to_string(),
                criteria: vec![10.0, 5.0],
            },
            Alternative {
                name: "dominated".to_string(),
                criteria: vec![8.0, 7.0],
            },
            Alternative {
                name: "tradeoff".to_string(),
                criteria: vec![6.0, 2.0],
            },
        ];
        let frontier = pareto_frontier(
            &alternatives,
            &[PreferenceDirection::Maximize, PreferenceDirection::Minimize],
        )
        .unwrap();
        assert_eq!(frontier, vec![0, 2]);
    }

    #[test]
    fn pareto_frontier_rejects_non_finite_criteria() {
        let err = pareto_frontier(
            &[Alternative {
                name: "bad".to_string(),
                criteria: vec![f64::INFINITY],
            }],
            &[PreferenceDirection::Maximize],
        )
        .unwrap_err();
        assert!(err.contains("non-finite"));
    }

    #[test]
    fn weighted_scores_break_ties_by_input_order() {
        let alternatives = vec![
            Alternative {
                name: "A".to_string(),
                criteria: vec![1.0],
            },
            Alternative {
                name: "B".to_string(),
                criteria: vec![1.0],
            },
        ];
        let objectives = vec![Objective {
            name: "same".to_string(),
            weight: 1.0,
            direction: PreferenceDirection::Maximize,
        }];
        let scores = weighted_additive_scores(&alternatives, &objectives).unwrap();
        assert_eq!(
            scores.iter().map(|s| s.index).collect::<Vec<_>>(),
            vec![0, 1]
        );
    }
}
