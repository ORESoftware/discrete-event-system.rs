//! The first-class-citizen contract and registry.
//!
//! A **first-class model** is any modeling paradigm that implements
//! [`ModelCitizen`]: it advertises a self-describing [`ModelDescriptor`] (kind,
//! schema, solve methods, an example spec for an LLM/UI to target) and can
//! validate-and-run a JSON spec into a uniform [`RunArtifact`]. MDP, POMDP, the
//! hybrid block-diagram engine, DES networks, and optimization solvers are all
//! peers under this one seam — none is privileged.
//!
//! This is the contract the platform's "English → spec → run → render" loop
//! targets: the server's LLM emits JSON matching a citizen's `spec_schema`
//! (using `example_spec` as a template and `descriptor` for discovery), the
//! library validates it (returning [`CitizenError::InvalidSpec`] with a message
//! the LLM can self-correct against) and runs it, and the UI renders the
//! returned artifact.

use serde::Serialize;
use serde_json::Value;

use super::artifact::RunArtifact;

/// Self-describing metadata for a first-class model kind. JSON-first so a server
/// can advertise it for discovery (alongside [`crate::des::service`]).
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDescriptor {
    /// Stable kind id, e.g. `"mdp"`, `"pomdp"`, `"hybrid"`.
    pub kind: String,
    pub title: String,
    pub description: String,
    /// The `$schema` value a spec for this kind carries, e.g. `"des/mdp/v1"`.
    pub spec_schema: String,
    /// Solve/run methods this kind supports, e.g. `["value-iteration"]`.
    pub methods: Vec<String>,
    /// A minimal valid spec — a template for an LLM or UI to start from.
    pub example_spec: Value,
}

/// Why running a citizen failed (recoverable; never panics out).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CitizenError {
    /// The spec did not validate. The message is phrased for an LLM/user to fix.
    InvalidSpec(String),
    /// The model ran but failed (e.g. solver did not converge / panicked).
    Run(String),
    /// No citizen is registered for the requested kind.
    UnknownKind(String),
}

impl std::fmt::Display for CitizenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CitizenError::InvalidSpec(m) => write!(f, "invalid spec: {m}"),
            CitizenError::Run(m) => write!(f, "run failed: {m}"),
            CitizenError::UnknownKind(k) => write!(f, "unknown model kind `{k}`"),
        }
    }
}

impl std::error::Error for CitizenError {}

/// A first-class modeling paradigm. Object-safe so a registry can hold a
/// heterogeneous set behind `dyn`.
pub trait ModelCitizen {
    /// Self-describing metadata for discovery + LLM targeting.
    fn descriptor(&self) -> ModelDescriptor;

    /// Validate and run a JSON spec into a uniform artifact.
    fn run_json(&self, spec: &Value) -> Result<RunArtifact, CitizenError>;
}

/// A registry of first-class model citizens, keyed by [`ModelDescriptor::kind`].
#[derive(Default)]
pub struct CitizenRegistry {
    citizens: Vec<Box<dyn ModelCitizen>>,
}

impl CitizenRegistry {
    pub fn new() -> Self {
        CitizenRegistry::default()
    }

    /// Register a citizen. A later registration of the same kind shadows an
    /// earlier one (last wins) — convenient for overriding builtins in tests.
    pub fn register(&mut self, citizen: Box<dyn ModelCitizen>) {
        self.citizens.push(citizen);
    }

    /// All registered kinds, in registration order.
    pub fn kinds(&self) -> Vec<String> {
        self.citizens.iter().map(|c| c.descriptor().kind).collect()
    }

    /// Discovery: descriptors for every registered citizen.
    pub fn descriptors(&self) -> Vec<ModelDescriptor> {
        self.citizens.iter().map(|c| c.descriptor()).collect()
    }

    /// Look up the citizen for a kind (last registration wins).
    pub fn get(&self, kind: &str) -> Option<&dyn ModelCitizen> {
        self.citizens
            .iter()
            .rev()
            .find(|c| c.descriptor().kind == kind)
            .map(|b| b.as_ref())
    }

    /// Run a spec by kind. The spec's `$schema`/`kind` is not required to be
    /// passed separately — the caller selects the citizen by `kind`.
    pub fn run(&self, kind: &str, spec: &Value) -> Result<RunArtifact, CitizenError> {
        match self.get(kind) {
            Some(citizen) => citizen.run_json(spec),
            None => Err(CitizenError::UnknownKind(kind.to_string())),
        }
    }
}
