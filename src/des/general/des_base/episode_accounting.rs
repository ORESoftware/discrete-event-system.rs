//! Port of `src/des/general/des-base/episode-accounting.ts`.
//!
//! Reward/length bookkeeping for RL episodes (scalar + vector). Pure mutable
//! state; methods take `&mut self`. Dimension mismatch is a programmer error →
//! `panic!`.

#[derive(Clone, Copy, Debug)]
pub struct EpisodeSummary {
    pub reward: f64,
    pub length: f64,
}

#[derive(Clone, Debug)]
pub struct VectorEpisodeSummary {
    pub rewards: Vec<f64>,
    pub length: f64,
}

#[derive(Clone, Debug, Default)]
pub struct EpisodeAccounting {
    pub reward_history: Vec<f64>,
    pub length_history: Vec<f64>,
    pub current_reward: f64,
    pub current_length: f64,
    pub total_steps: u64,
}

impl EpisodeAccounting {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_step(&mut self, reward: f64) {
        self.total_steps += 1;
        self.current_reward += reward;
        self.current_length += 1.0;
    }

    pub fn finish_episode(&mut self) -> EpisodeSummary {
        let summary = EpisodeSummary {
            reward: self.current_reward,
            length: self.current_length,
        };
        self.reward_history.push(summary.reward);
        self.length_history.push(summary.length);
        self.reset_current();
        summary
    }

    pub fn reset_current(&mut self) {
        self.current_reward = 0.0;
        self.current_length = 0.0;
    }
}

#[derive(Clone, Debug)]
pub struct VectorEpisodeAccounting {
    pub dimension: usize,
    pub reward_history: Vec<Vec<f64>>,
    pub length_history: Vec<f64>,
    pub current_rewards: Vec<f64>,
    pub total_steps: u64,
}

impl VectorEpisodeAccounting {
    pub fn new(dimension: usize) -> Self {
        VectorEpisodeAccounting {
            dimension,
            reward_history: Vec::new(),
            length_history: Vec::new(),
            current_rewards: vec![0.0; dimension],
            total_steps: 0,
        }
    }

    pub fn record_step(&mut self, rewards: &[f64]) {
        if rewards.len() != self.dimension {
            panic!("expected {} rewards, got {}", self.dimension, rewards.len());
        }
        self.total_steps += 1;
        for i in 0..self.dimension {
            self.current_rewards[i] += rewards[i];
        }
    }

    pub fn finish_episode(&mut self, length: f64) -> VectorEpisodeSummary {
        let rewards = self.current_rewards.clone();
        self.reward_history.push(rewards.clone());
        self.length_history.push(length);
        self.reset_current();
        VectorEpisodeSummary { rewards, length }
    }

    pub fn reset_current(&mut self) {
        self.current_rewards.iter_mut().for_each(|x| *x = 0.0);
    }

    pub fn set_current_rewards(&mut self, rewards: &[f64]) {
        if rewards.len() != self.dimension {
            panic!("expected {} rewards, got {}", self.dimension, rewards.len());
        }
        self.current_rewards.copy_from_slice(rewards);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scalar_episode() {
        let mut a = EpisodeAccounting::new();
        a.record_step(1.0);
        a.record_step(2.0);
        let s = a.finish_episode();
        assert_eq!(s.reward, 3.0);
        assert_eq!(s.length, 2.0);
        assert_eq!(a.current_reward, 0.0);
        assert_eq!(a.reward_history, vec![3.0]);
    }

    #[test]
    fn vector_episode() {
        let mut a = VectorEpisodeAccounting::new(2);
        a.record_step(&[1.0, 2.0]);
        a.record_step(&[3.0, 4.0]);
        let s = a.finish_episode(2.0);
        assert_eq!(s.rewards, vec![4.0, 6.0]);
        assert_eq!(a.current_rewards, vec![0.0, 0.0]);
    }
}
