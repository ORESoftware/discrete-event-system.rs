//! Port of `src/des/general/cartesian-state-space.ts` — a reversible
//! index ↔ coordinate bridge for multi-dimensional discrete MDP/POMDP state
//! spaces.
//!
//! The algorithms in this repo mostly consume compact integer state IDs, while
//! models are easier to read as coordinates like `(x, y, inventory, backlog)`.
//! [`CartesianStateSpace`] is the reversible bridge between those two views.
//!
//! Mapping notes (from the TS "RUST MIGRATION" header):
//!   * `interface CartesianDimension` / `CoordinateTransition` → `#[derive(Clone)]`
//!     structs.
//!   * `interface CoordinateMDPSpec` → struct holding boxed callbacks
//!     (`Box<dyn Fn(&[usize], usize) -> _>`; optionals → `Option<...>`).
//!   * `class CartesianStateSpace` → struct `{ dimensions, strides, num_states }`
//!     + impl.
//!   * `fn coordinateMDPToSpec` → [`coordinate_mdp_to_spec`].
//!   * The dup-name guard uses `HashSet<String>`; integer indices are `usize`
//!     and `Math.floor(index / stride) % size` is plain integer division.
//!   * Constructor `throw`s are invariant violations → `panic!`.

use std::collections::HashSet;
use std::rc::Rc;

use crate::des::general::value_iteration::{MDPSpec, Outcome};

/// One axis of the Cartesian state space.
#[derive(Clone, Debug)]
pub struct CartesianDimension {
    pub name: String,
    pub size: usize,
    /// Optional human-readable labels, one per value in `0..size`.
    pub labels: Option<Vec<String>>,
}

/// A single coordinate-space transition: probability, reward, and the
/// destination coordinate (which gets encoded to an integer state id).
#[derive(Clone, Debug)]
pub struct CoordinateTransition {
    pub prob: f64,
    pub reward: f64,
    pub next: Vec<usize>,
}

/// MDP description expressed over coordinates rather than flat state ids.
///
/// `num_actions`/`outcomes`/`is_terminal`/`terminal_reward`/`action_label` are
/// the TS callbacks expressed as boxed closures. The first argument is always
/// the decoded coordinate vector; `state_index` is the flat id.
pub struct CoordinateMDPSpec {
    pub space: CartesianStateSpace,
    pub num_actions: Box<dyn Fn(&[usize], usize) -> usize>,
    pub outcomes: Box<dyn Fn(&[usize], usize, usize) -> Vec<CoordinateTransition>>,
    pub is_terminal: Option<Box<dyn Fn(&[usize], usize) -> bool>>,
    pub terminal_reward: Option<Box<dyn Fn(&[usize], usize) -> f64>>,
    pub action_label: Option<Box<dyn Fn(usize) -> String>>,
}

/// Shared indexing for multi-dimensional discrete MDP/POMDP state spaces.
#[derive(Clone, Debug)]
pub struct CartesianStateSpace {
    pub dimensions: Vec<CartesianDimension>,
    pub strides: Vec<usize>,
    pub num_states: usize,
}

impl CartesianStateSpace {
    /// Build the space from its dimensions, precomputing strides and the total
    /// state count. Panics on invariant violations (empty dimension list, empty
    /// or duplicate names, non-positive sizes, mismatched label lengths) —
    /// these map to the `throw`s in the TS constructor.
    pub fn new(dimensions: Vec<CartesianDimension>) -> Self {
        if dimensions.is_empty() {
            panic!("CartesianStateSpace: at least one dimension is required");
        }
        let mut names: HashSet<String> = HashSet::new();
        let mut strides: Vec<usize> = Vec::with_capacity(dimensions.len());
        let mut n: usize = 1;
        for dim in &dimensions {
            if dim.name.is_empty() {
                panic!("CartesianStateSpace: dimension names must be non-empty");
            }
            if names.contains(&dim.name) {
                panic!("CartesianStateSpace: duplicate dimension \"{}\"", dim.name);
            }
            names.insert(dim.name.clone());
            // `size` is a `usize`, so the "positive integer" check reduces to
            // a nonzero check.
            if dim.size == 0 {
                panic!(
                    "CartesianStateSpace: dimension \"{}\" size must be a positive integer",
                    dim.name
                );
            }
            if let Some(labels) = &dim.labels {
                if labels.len() != dim.size {
                    panic!(
                        "CartesianStateSpace: dimension \"{}\" labels length must equal size",
                        dim.name
                    );
                }
            }
            strides.push(n);
            n *= dim.size;
        }
        CartesianStateSpace {
            dimensions,
            strides,
            num_states: n,
        }
    }

    /// Number of dimensions (the coordinate rank).
    pub fn rank(&self) -> usize {
        self.dimensions.len()
    }

    /// Encode a coordinate vector to its flat integer state id. Panics if the
    /// rank is wrong or any coordinate falls outside its dimension's range.
    pub fn encode(&self, coords: &[usize]) -> usize {
        if coords.len() != self.dimensions.len() {
            panic!(
                "CartesianStateSpace.encode: coordinate rank {} != {}",
                coords.len(),
                self.dimensions.len()
            );
        }
        let mut index = 0;
        for i in 0..coords.len() {
            let c = coords[i];
            let dim = &self.dimensions[i];
            if c >= dim.size {
                panic!(
                    "CartesianStateSpace.encode: {}={} outside [0, {}]",
                    dim.name,
                    c,
                    dim.size - 1
                );
            }
            index += c * self.strides[i];
        }
        index
    }

    /// Decode a flat integer state id back to its coordinate vector. Panics if
    /// the index is out of range.
    pub fn decode(&self, index: usize) -> Vec<usize> {
        if index >= self.num_states {
            panic!(
                "CartesianStateSpace.decode: index {} outside [0, {}]",
                index,
                self.num_states - 1
            );
        }
        let mut coords = vec![0_usize; self.dimensions.len()];
        for i in (0..self.dimensions.len()).rev() {
            coords[i] = (index / self.strides[i]) % self.dimensions[i].size;
        }
        coords
    }

    /// Human-readable label for a flat state id.
    pub fn label(&self, index: usize) -> String {
        self.coord_label(&self.decode(index))
    }

    /// Human-readable label for a coordinate vector, e.g. `x=2,y=hot`.
    pub fn coord_label(&self, coords: &[usize]) -> String {
        if coords.len() != self.dimensions.len() {
            panic!(
                "CartesianStateSpace.coordLabel: coordinate rank {} != {}",
                coords.len(),
                self.dimensions.len()
            );
        }
        coords
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let dim = &self.dimensions[i];
                let value = match &dim.labels {
                    Some(labels) => labels[c].clone(),
                    None => c.to_string(),
                };
                format!("{}={}", dim.name, value)
            })
            .collect::<Vec<_>>()
            .join(",")
    }

    /// Every coordinate vector in flat-id order.
    pub fn all_coords(&self) -> Vec<Vec<usize>> {
        let mut out = Vec::with_capacity(self.num_states);
        for i in 0..self.num_states {
            out.push(self.decode(i));
        }
        out
    }
}

/// Lower a coordinate-space MDP description into a flat [`MDPSpec`] that the
/// generic value-iteration solver consumes.
///
/// The shared [`CartesianStateSpace`] is wrapped in an [`Rc`] so each generated
/// callback can hold its own handle (the TS closures all capture the same
/// `space` object by reference).
pub fn coordinate_mdp_to_spec(spec: CoordinateMDPSpec) -> MDPSpec {
    let CoordinateMDPSpec {
        space,
        num_actions,
        outcomes,
        is_terminal,
        terminal_reward,
        action_label,
    } = spec;
    let space = Rc::new(space);
    let num_states = space.num_states;

    let space_na = space.clone();
    let num_actions_fn: Box<dyn Fn(usize) -> usize> =
        Box::new(move |s| num_actions(&space_na.decode(s), s));

    let space_oc = space.clone();
    let outcomes_fn: Box<dyn Fn(usize, usize) -> Vec<Outcome>> = Box::new(move |s, a| {
        outcomes(&space_oc.decode(s), a, s)
            .into_iter()
            .map(|o| Outcome {
                prob: o.prob,
                reward: o.reward,
                next_state: space_oc.encode(&o.next),
            })
            .collect()
    });

    let is_terminal_fn: Option<Box<dyn Fn(usize) -> bool>> = is_terminal.map(|f| {
        let space_it = space.clone();
        Box::new(move |s| f(&space_it.decode(s), s)) as Box<dyn Fn(usize) -> bool>
    });

    let terminal_reward_fn: Option<Box<dyn Fn(usize) -> f64>> = terminal_reward.map(|f| {
        let space_tr = space.clone();
        Box::new(move |s| f(&space_tr.decode(s), s)) as Box<dyn Fn(usize) -> f64>
    });

    let space_sl = space.clone();
    let state_label_fn: Option<Box<dyn Fn(usize) -> String>> =
        Some(Box::new(move |s| space_sl.label(s)));

    MDPSpec {
        num_states,
        num_actions: num_actions_fn,
        outcomes: outcomes_fn,
        is_terminal: is_terminal_fn,
        terminal_reward: terminal_reward_fn,
        state_label: state_label_fn,
        action_label,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dim(name: &str, size: usize) -> CartesianDimension {
        CartesianDimension {
            name: name.to_string(),
            size,
            labels: None,
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        let space = CartesianStateSpace::new(vec![dim("x", 3), dim("y", 4)]);
        assert_eq!(space.num_states, 12);
        assert_eq!(space.strides, vec![1, 3]);
        for i in 0..space.num_states {
            let coords = space.decode(i);
            assert_eq!(space.encode(&coords), i);
        }
        // index 7 → x = 7 % 3 = 1, y = (7 / 3) % 4 = 2.
        assert_eq!(space.decode(7), vec![1, 2]);
        assert_eq!(space.encode(&[1, 2]), 7);
    }

    #[test]
    fn labels_and_coord_labels() {
        let space = CartesianStateSpace::new(vec![
            CartesianDimension {
                name: "temp".to_string(),
                size: 2,
                labels: Some(vec!["cold".to_string(), "hot".to_string()]),
            },
            dim("n", 2),
        ]);
        // state 3 → temp = 1 (hot), n = 1.
        assert_eq!(space.label(3), "temp=hot,n=1");
        assert_eq!(space.coord_label(&[0, 0]), "temp=cold,n=0");
    }

    #[test]
    #[should_panic(expected = "duplicate dimension")]
    fn duplicate_dimension_panics() {
        let _ = CartesianStateSpace::new(vec![dim("x", 2), dim("x", 3)]);
    }

    #[test]
    fn coordinate_mdp_to_spec_encodes_outcomes() {
        let space = CartesianStateSpace::new(vec![dim("x", 3)]);
        let coord_spec = CoordinateMDPSpec {
            space,
            num_actions: Box::new(|_coords, _s| 1),
            // Self-loop with reward equal to the coordinate value.
            outcomes: Box::new(|coords, _a, _s| {
                vec![CoordinateTransition {
                    prob: 1.0,
                    reward: coords[0] as f64,
                    next: vec![coords[0]],
                }]
            }),
            is_terminal: None,
            terminal_reward: None,
            action_label: None,
        };
        let mdp = coordinate_mdp_to_spec(coord_spec);
        assert_eq!(mdp.num_states, 3);
        assert_eq!((mdp.num_actions)(2), 1);
        let outs = (mdp.outcomes)(2, 0);
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].next_state, 2);
        assert_eq!(outs[0].reward, 2.0);
        assert_eq!((mdp.state_label.as_ref().unwrap())(1), "x=1");
    }
}
