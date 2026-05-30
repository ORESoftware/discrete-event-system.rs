//! Port of `src/des/general/des-base/rl-tokens.ts` — common token payloads
//! exchanged by RL stations.
//!
//! TS `class X implements Token` → plain data structs. The Rust framework has
//! **no `Token` marker trait**: tokens travel as `Rc<dyn Any>` (any `'static`
//! type qualifies), so the `import {Token}` marker is dropped. The generic
//! defaults `S = number` / `A = number` map to `S = f64` / `A = usize` (action
//! index), per the migration rules.

/// Sent by the environment to the agent at the start of each episode.
#[derive(Clone, Debug, PartialEq)]
pub struct StateToken<S = f64> {
    pub state: S,
    pub episode_id: f64,
}

impl<S> StateToken<S> {
    pub fn new(state: S, episode_id: f64) -> Self {
        StateToken { state, episode_id }
    }
}

/// Sent by the agent to the environment to apply an action.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionToken<S = f64, A = usize> {
    pub state: S,
    pub action: A,
    pub episode_id: f64,
}

impl<S, A> ActionToken<S, A> {
    pub fn new(state: S, action: A, episode_id: f64) -> Self {
        ActionToken { state, action, episode_id }
    }
}

/// Sent by the environment after each step.
#[derive(Clone, Debug, PartialEq)]
pub struct TransitionToken<S = f64, A = usize> {
    pub state: S,
    pub action: A,
    pub reward: f64,
    pub next_state: S,
    pub done: bool,
    pub episode_id: f64,
}

impl<S, A> TransitionToken<S, A> {
    pub fn new(state: S, action: A, reward: f64, next_state: S, done: bool, episode_id: f64) -> Self {
        TransitionToken { state, action, reward, next_state, done, episode_id }
    }
}

/// Sent by an agent to a "policy update" station when its rollout is full.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct TrainTriggerToken;

/// Sent back to an agent that paused awaiting fresh parameters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResumeToken;

#[cfg(test)]
mod tests {
    use super::*;
    use std::any::Any;
    use std::rc::Rc;

    #[test]
    fn builds_transition_with_default_params() {
        let t = TransitionToken::new(1.0_f64, 2_usize, -0.5, 1.5, false, 7.0);
        assert_eq!(t.state, 1.0);
        assert_eq!(t.action, 2);
        assert_eq!(t.reward, -0.5);
        assert_eq!(t.next_state, 1.5);
        assert!(!t.done);
        assert_eq!(t.episode_id, 7.0);
    }

    #[test]
    fn travels_as_any_token_and_downcasts() {
        let tok: Rc<dyn Any> = Rc::new(StateToken::new(0.25_f64, 3.0));
        let recovered = tok.downcast::<StateToken<f64>>().expect("downcast");
        assert_eq!(recovered.state, 0.25);
        assert_eq!(recovered.episode_id, 3.0);
    }

    #[test]
    fn unit_marker_tokens_are_distinct_types() {
        let train: Rc<dyn Any> = Rc::new(TrainTriggerToken);
        let resume: Rc<dyn Any> = Rc::new(ResumeToken);
        assert!(train.downcast_ref::<TrainTriggerToken>().is_some());
        assert!(train.downcast_ref::<ResumeToken>().is_none());
        assert!(resume.downcast_ref::<ResumeToken>().is_some());
    }
}
