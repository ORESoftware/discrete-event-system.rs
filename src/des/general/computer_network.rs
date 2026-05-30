//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/computer-network.ts`
//! Rust target: `src/des/general/computer_network.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/computer-network.ts",
    "src/des/general/computer_network.rs",
    &["RUST MIGRATION: target module src/des/general/computer_network.rs.", "RUST MIGRATION: Network specs, snapshots, metrics, and result interfaces become serde structs; node/link/protocol/routing string unions become enums.", "RUST MIGRATION: NetworkPacket, NetworkStation, node/delay/link stations, and ComputerNetworkStation become structs implementing Token, Entity, and Station traits rather than inheritance.", "RUST MIGRATION: Map/Set adjacency, flow, queue, and routing tables map to HashMap/HashSet/VecDeque; preserve deterministic ordering explicitly where reports depend on it.", "RUST MIGRATION: runComputerNetworkSimulation is graph-visible and should be a PureTransform entry struct; validation and normalization return Result."],
    &["ComputerNetworkProblem", "ComputerNetworkResult", "ComputerNetworkStation", "NetworkBottleneckReport", "NetworkDelayStation", "NetworkFlowSpec", "NetworkFlowStats", "NetworkHostStation", "NetworkLinkSpec", "NetworkLinkStation", "NetworkLinkStats", "NetworkNodeKind", "NetworkNodeSpec", "NetworkNodeStation", "NetworkNodeStats", "NetworkPacket", "NetworkPacketSnapshot", "NetworkProtocol", "NetworkRouterStation", "NetworkRoutingMetric", "NetworkStation", "NetworkSwitchStation", "NetworkTimeSample", "PacketDropReason", "buildBottleneckComputerNetworkProblem", "buildDefaultComputerNetworkProblem", "runComputerNetworkSimulation", "validateComputerNetworkProblem"],
);
