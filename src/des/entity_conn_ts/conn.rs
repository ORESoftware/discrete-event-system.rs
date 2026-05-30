//! Canonical use path: `crate::des::entity_conn_ts::conn::ConnectionOpts`
//!
//! Port of `src/des/entity-conn.ts/conn.ts` — connection configuration options
//! for graph edges.
//!
//! The TS parent directory was literally named `entity-conn.ts` (a dot in a
//! directory name), which is not a valid Rust module segment, so the Rust home
//! is `entity_conn_ts` (module `des::entity_conn_ts::conn`).
//!
//! PORT NOTE: the TS `ConnectionOpts.isBidirectional` was typed as the literal
//! `false`; here it is a plain `bool` (defaulting to `false`). `travelTime`
//! (`number`) becomes `f64`. This struct is distinct from the framework's
//! connection `ConnectionOpts` in `r#abstract::r#abstract` (that one is empty);
//! this is the richer edge-config the TS file declared.

#![allow(dead_code)]

/// `interface ConnectionOpts` — per-edge configuration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConnectionOpts {
    pub travel_time: f64,
    pub is_bidirectional: bool,
}

impl Default for ConnectionOpts {
    fn default() -> Self {
        ConnectionOpts {
            travel_time: 0.0,
            is_bidirectional: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_are_zero_and_unidirectional() {
        let c = ConnectionOpts::default();
        assert_eq!(c.travel_time, 0.0);
        assert!(!c.is_bidirectional);
    }

    #[test]
    fn can_configure_travel_time() {
        let c = ConnectionOpts {
            travel_time: 2.5,
            is_bidirectional: false,
        };
        assert_eq!(c.travel_time, 2.5);
    }
}
