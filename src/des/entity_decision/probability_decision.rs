//! Rust port of `src/des/entity-decision/probability-decision.ts`.

use crate::core::{DesDecimal, DesError, DesResult, EntityConnection, RandomSource};
use crate::des::entity_decision::decision::Decision;
use crate::migration::MigrationFile;
use crate::numeric::decimal_from_f64;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/entity-decision/probability-decision.ts",
    "src/des/entity_decision/probability_decision.rs",
    &[
        "Probabilities are owned DesDecimal vectors validated before routing.",
        "Random sample is injected through RandomSource.",
    ],
    &["ProbabilisticDecision", "ProbabilityDecisionOpts"],
);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilityDecisionOpts {
    pub probabilities: Vec<DesDecimal>,
}

impl ProbabilityDecisionOpts {
    pub fn validate(&self) -> DesResult<()> {
        let sum = self
            .probabilities
            .iter()
            .copied()
            .fold(DesDecimal::ZERO, |total, probability| total + probability);
        if self
            .probabilities
            .iter()
            .any(|p| *p < DesDecimal::ZERO || *p > DesDecimal::ONE)
        {
            return Err(DesError::InvalidState {
                context: "ProbabilityDecisionOpts::validate",
                message: "probabilities must be in [0, 1]".to_owned(),
            });
        }
        if sum != DesDecimal::ONE {
            return Err(DesError::InvalidState {
                context: "ProbabilityDecisionOpts::validate",
                message: format!("probabilities must sum to 1, got {sum}"),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProbabilisticDecision<T> {
    pub decision: Decision<T>,
    pub opts: ProbabilityDecisionOpts,
}

impl<T> ProbabilisticDecision<T> {
    pub fn new(id: impl Into<String>, opts: ProbabilityDecisionOpts) -> DesResult<Self> {
        opts.validate()?;
        Ok(Self {
            decision: Decision::new(id),
            opts,
        })
    }

    pub fn choose_connection<R>(&self, rng: &mut R) -> Option<&EntityConnection>
    where
        R: RandomSource,
    {
        let sample = decimal_from_f64(rng.next_f64(), "ProbabilisticDecision::choose_connection")
            .unwrap_or(DesDecimal::ZERO);
        let mut cumulative = DesDecimal::ZERO;
        for (index, probability) in self.opts.probabilities.iter().enumerate() {
            cumulative += *probability;
            if sample <= cumulative {
                return self.decision.connections_out.get(index);
            }
        }
        self.decision.connections_out.last()
    }
}

impl<T> crate::core::Entity for ProbabilisticDecision<T>
where
    T: Debug,
{
    fn state(&self) -> &crate::core::EntityState {
        &self.decision.state
    }

    fn state_mut(&mut self) -> &mut crate::core::EntityState {
        &mut self.decision.state
    }

    fn validate_before_run(&self) -> DesResult<()> {
        self.opts.validate()?;
        if self.decision.connections_out.len() != self.opts.probabilities.len() {
            return Err(DesError::Validation {
                entity_id: self.decision.state.id.clone(),
                message: "out-connections length must match probabilities length".to_owned(),
            });
        }
        Ok(())
    }
}
