//! Port of `src/des/general/des-base/model-topology.ts`.
//!
//! Minimal station-graph topology metadata for the visual/animation layer.
//! Trivial DTO; no `DESStation`, no logic.

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StationGraphTopology {
    pub stations: Vec<String>,
    pub movables: Vec<String>,
}

pub fn station_graph_topology(stations: &[String], movables: &[String]) -> StationGraphTopology {
    StationGraphTopology {
        stations: stations.to_vec(),
        movables: movables.to_vec(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn copies_inputs() {
        let s = vec!["a".to_string(), "b".to_string()];
        let m = vec!["x".to_string()];
        let t = station_graph_topology(&s, &m);
        assert_eq!(t.stations, s);
        assert_eq!(t.movables, m);
    }
}
