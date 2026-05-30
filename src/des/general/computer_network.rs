//! Port of `src/des/general/computer-network.ts` — module
//! `des::general::computer_network`.
//!
//! Packet-switched computer networking (hosts / routers / switches / links /
//! packets) modelled as a discrete-event simulation. Nodes hold input queues,
//! routing state and counters; links model bandwidth serialization,
//! propagation latency, queue limits and cost; packets carry per-flow timing,
//! route, hop, cost and drop/delivery state through the topology.
//!
//! Conversion notes from the TS source:
//!   * Deep station inheritance (Host/Router/Switch <- NetworkNodeStation <-
//!     NetworkStation <- DESStation) flattens to one [`NetworkNodeStation`]
//!     struct carrying a `kind`; the whole simulation is driven by a single
//!     [`ComputerNetworkStation`] (a `DESStation`) whose `run_time_step` steps
//!     every node and link each tick.
//!   * `NetworkPacket` is an OWNED value moved between queues (a packet is only
//!     ever in one place at a time), so no `Rc`/`RefCell` is needed and the
//!     `node.step(route, deliver, drop)` callbacks are replaced by inline
//!     handling over destructured fields (avoids `&mut self` aliasing).
//!   * Routing (`shortestNextLink` Dijkstra) only depends on the STATIC
//!     topology (node ids + link specs + metric), so it is precomputed into a
//!     `RouteEdge` adjacency and a memoising `route_cache`, independent of the
//!     live node/link state.
//!   * `Map`/`Set` -> `HashMap`/`HashSet`; insertion order is preserved with
//!     explicit `node_order` / `link_order` vectors.
//!   * `Token` is not a Rust trait (tokens are `Rc<dyn Any>` in the framework);
//!     packets are plain owned values here, so the TS `implements Token` has no
//!     analogue. `NetworkPacket` still embeds a [`BasicMovingEntity`] (the TS
//!     `extends BasicMovingEntity`) and calls `do_finish` on delivery/drop.
//!   * `Preconditions` throw -> `Result`; `validate_*` propagates and the
//!     constructor panics on an invalid problem (matching the TS throw).

#![allow(dead_code)]

use std::any::Any;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::rc::Rc;

use crate::des::entity_moving::moving::{BasicMovingEntity, MovingEntity};
use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::general::des_base::runner::{
    assert_no_validation_failures, run_iterative_des, IterativeRunOptions,
};
use crate::des::general::des_base::station::{DESStation, StationCore, StationRef};
use crate::des::general::des_base::validation::intrinsic_check;

const MODEL: &str = "computer-network";

// =============================================================================
// Enums (TS string unions).
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkNodeKind {
    Host,
    Router,
    Switch,
}

impl NetworkNodeKind {
    pub fn as_str(self) -> &'static str {
        match self {
            NetworkNodeKind::Host => "host",
            NetworkNodeKind::Router => "router",
            NetworkNodeKind::Switch => "switch",
        }
    }
    pub fn parse(s: &str) -> Option<NetworkNodeKind> {
        match s {
            "host" => Some(NetworkNodeKind::Host),
            "router" => Some(NetworkNodeKind::Router),
            "switch" => Some(NetworkNodeKind::Switch),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkRoutingMetric {
    Latency,
    Cost,
    Hop,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NetworkProtocol {
    Raw,
    Tcp,
    Udp,
    Http,
}

impl NetworkProtocol {
    pub fn as_str(self) -> &'static str {
        match self {
            NetworkProtocol::Raw => "raw",
            NetworkProtocol::Tcp => "tcp",
            NetworkProtocol::Udp => "udp",
            NetworkProtocol::Http => "http",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketDropReason {
    NodeQueueOverflow,
    LinkQueueOverflow,
    NoRoute,
    TtlExpired,
    MaxPacketsInSystem,
}

impl PacketDropReason {
    pub fn as_str(self) -> &'static str {
        match self {
            PacketDropReason::NodeQueueOverflow => "node-queue-overflow",
            PacketDropReason::LinkQueueOverflow => "link-queue-overflow",
            PacketDropReason::NoRoute => "no-route",
            PacketDropReason::TtlExpired => "ttl-expired",
            PacketDropReason::MaxPacketsInSystem => "max-packets-in-system",
        }
    }
}

// =============================================================================
// Problem specs.
// =============================================================================

#[derive(Clone, Debug)]
pub struct NetworkNodeSpec {
    pub id: String,
    pub kind: NetworkNodeKind,
    pub forwarding_rate_pps: Option<f64>,
    pub queue_limit_packets: Option<usize>,
}

#[derive(Clone, Debug)]
pub struct NetworkLinkSpec {
    pub id: String,
    pub from: String,
    pub to: String,
    pub bandwidth_mbps: f64,
    pub latency_ms: f64,
    pub cost_per_mb: Option<f64>,
    pub queue_limit_packets: Option<usize>,
    pub bidirectional: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct NetworkFlowSpec {
    pub id: String,
    pub source: String,
    pub destination: String,
    pub protocol: Option<NetworkProtocol>,
    pub rate_pps: f64,
    pub packet_size_bytes: f64,
    pub start_ms: Option<f64>,
    pub end_ms: Option<f64>,
    pub max_packets: Option<u64>,
    pub ttl_hops: Option<i64>,
}

#[derive(Clone, Debug)]
pub struct ComputerNetworkProblem {
    pub nodes: Vec<NetworkNodeSpec>,
    pub links: Vec<NetworkLinkSpec>,
    pub flows: Vec<NetworkFlowSpec>,
    pub duration_ms: f64,
    pub dt_ms: f64,
    pub routing_metric: Option<NetworkRoutingMetric>,
    pub drain_after_sources_ms: Option<f64>,
    pub max_packets_in_system: Option<u64>,
    pub sample_every_ms: Option<f64>,
}

// =============================================================================
// Output structs.
// =============================================================================

#[derive(Clone, Debug)]
pub struct NetworkPacketSnapshot {
    pub packet_id: u64,
    pub flow_id: String,
    pub protocol: NetworkProtocol,
    pub source: String,
    pub destination: String,
    pub payload_bytes: f64,
    pub size_bytes: f64,
    pub created_at_ms: f64,
    pub delivered_at_ms: Option<f64>,
    pub dropped_at_ms: Option<f64>,
    pub drop_reason: Option<PacketDropReason>,
    pub current_node_id: Option<String>,
    pub current_link_id: Option<String>,
    pub hops: Vec<String>,
    pub cost: f64,
}

#[derive(Clone, Debug)]
pub struct NetworkFlowStats {
    pub id: String,
    pub protocol: NetworkProtocol,
    pub source: String,
    pub destination: String,
    pub generated_packets: f64,
    pub delivered_packets: f64,
    pub dropped_packets: f64,
    pub delivery_ratio: f64,
    pub generated_bytes: f64,
    pub delivered_bytes: f64,
    pub offered_load_mbps: f64,
    pub throughput_mbps: f64,
    pub goodput_mbps: f64,
    pub mean_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub mean_time_in_system_ms: f64,
    pub p95_time_in_system_ms: f64,
    pub total_cost: f64,
    pub mean_cost_per_delivered_packet: f64,
}

#[derive(Clone, Debug)]
pub struct NetworkNodeStats {
    pub id: String,
    pub kind: NetworkNodeKind,
    pub forwarding_rate_pps: f64,
    pub queue_limit_packets: f64,
    pub received_packets: f64,
    pub forwarded_packets: f64,
    pub delivered_packets: f64,
    pub dropped_packets: f64,
    pub final_queue: f64,
    pub max_queue: f64,
    pub avg_queue: f64,
    pub mean_queue_delay_ms: f64,
    pub max_queue_delay_ms: f64,
}

#[derive(Clone, Debug)]
pub struct NetworkLinkStats {
    pub id: String,
    pub from: String,
    pub to: String,
    pub bandwidth_mbps: f64,
    pub latency_ms: f64,
    pub cost_per_mb: f64,
    pub queue_limit_packets: f64,
    pub enqueued_packets: f64,
    pub delivered_packets: f64,
    pub dropped_packets: f64,
    pub transmitted_bytes: f64,
    pub throughput_mbps: f64,
    pub utilization: f64,
    pub final_in_flight: f64,
    pub max_in_flight: f64,
    pub avg_in_flight: f64,
    pub mean_queue_delay_ms: f64,
    pub max_queue_delay_ms: f64,
    pub mean_time_on_link_ms: f64,
    pub max_time_on_link_ms: f64,
    pub total_cost: f64,
}

#[derive(Clone, Debug)]
pub struct NetworkTimeSample {
    pub t_ms: f64,
    pub generated_packets: f64,
    pub delivered_packets: f64,
    pub dropped_packets: f64,
    pub active_packets: f64,
    pub node_queues: HashMap<String, f64>,
    pub link_in_flight: HashMap<String, f64>,
    pub link_utilization: HashMap<String, f64>,
}

#[derive(Clone, Debug)]
pub struct NetworkBottleneckReport {
    pub id: String,
    pub kind: String,
    pub score: f64,
    pub reason: String,
    pub utilization: Option<f64>,
    pub avg_queue: f64,
    pub max_queue: f64,
    pub dropped_packets: f64,
    pub mean_queue_delay_ms: f64,
}

#[derive(Clone, Debug)]
pub struct ComputerNetworkResult {
    pub generated_packets: f64,
    pub delivered_packets: f64,
    pub dropped_packets: f64,
    pub active_packets: f64,
    pub max_active_packets: f64,
    pub delivery_ratio: f64,
    pub offered_load_mbps: f64,
    pub throughput_mbps: f64,
    pub goodput_mbps: f64,
    pub mean_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub total_cost: f64,
    pub total_simulated_ms: f64,
    pub routing_metric: NetworkRoutingMetric,
    pub flow_stats: Vec<NetworkFlowStats>,
    pub node_stats: Vec<NetworkNodeStats>,
    pub link_stats: Vec<NetworkLinkStats>,
    pub bottlenecks: Vec<NetworkBottleneckReport>,
    pub time_series: Vec<NetworkTimeSample>,
    pub delivered_packets_trace: Vec<NetworkPacketSnapshot>,
    pub dropped_packets_trace: Vec<NetworkPacketSnapshot>,
    pub invariant_violations: Vec<String>,
}

// =============================================================================
// Packet (movable) + per-node / per-link queued wrappers.
// =============================================================================

/// A network packet — the movable entity (TS `class NetworkPacket extends
/// BasicMovingEntity implements Token`).
pub struct NetworkPacket {
    pub packet_id: u64,
    pub flow_id: String,
    pub protocol: NetworkProtocol,
    pub source: String,
    pub destination: String,
    pub payload_bytes: f64,
    pub size_bytes: f64,
    pub created_at_ms: f64,
    pub ttl_hops: i64,
    pub delivered_at_ms: Option<f64>,
    pub dropped_at_ms: Option<f64>,
    pub drop_reason: Option<PacketDropReason>,
    pub current_node_id: Option<String>,
    pub current_link_id: Option<String>,
    pub hops: Vec<String>,
    pub cost: f64,
    moving: BasicMovingEntity,
}

impl NetworkPacket {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        packet_id: u64,
        flow_id: String,
        protocol: NetworkProtocol,
        source: String,
        destination: String,
        payload_bytes: f64,
        size_bytes: f64,
        created_at_ms: f64,
        ttl_hops: i64,
    ) -> Self {
        let hops = vec![source.clone()];
        let current_node_id = Some(source.clone());
        NetworkPacket {
            packet_id,
            flow_id,
            protocol,
            source,
            destination,
            payload_bytes,
            size_bytes,
            created_at_ms,
            ttl_hops,
            delivered_at_ms: None,
            dropped_at_ms: None,
            drop_reason: None,
            current_node_id,
            current_link_id: None,
            hops,
            cost: 0.0,
            moving: BasicMovingEntity::new(),
        }
    }

    pub fn snapshot(&self) -> NetworkPacketSnapshot {
        NetworkPacketSnapshot {
            packet_id: self.packet_id,
            flow_id: self.flow_id.clone(),
            protocol: self.protocol,
            source: self.source.clone(),
            destination: self.destination.clone(),
            payload_bytes: self.payload_bytes,
            size_bytes: self.size_bytes,
            created_at_ms: self.created_at_ms,
            delivered_at_ms: self.delivered_at_ms,
            dropped_at_ms: self.dropped_at_ms,
            drop_reason: self.drop_reason,
            current_node_id: self.current_node_id.clone(),
            current_link_id: self.current_link_id.clone(),
            hops: self.hops.clone(),
            cost: self.cost,
        }
    }

    fn finish(&mut self) {
        self.moving.do_finish();
    }
}

struct QueuedNodePacket {
    packet: NetworkPacket,
    enqueued_at_ms: f64,
}

struct ScheduledPacket {
    packet: NetworkPacket,
    deliver_at_ms: f64,
}

// =============================================================================
// Node station.
// =============================================================================

/// A network node (host / router / switch). Flattens the TS
/// `NetworkNodeStation` hierarchy into one struct carrying `kind`.
pub struct NetworkNodeStation {
    spec: NetworkNodeSpec,
    queue_limit_packets: usize,
    forwarding_rate_pps: f64,
    input_queue: VecDeque<QueuedNodePacket>,
    forwarding_credit: f64,
    queue_area: f64,
    max_queue: usize,
    total_queue_delay_ms: f64,
    max_queue_delay_ms: f64,
    processed_from_queue: u64,
    received_packets: u64,
    forwarded_packets: u64,
    delivered_packets: u64,
    dropped_packets: u64,
}

impl NetworkNodeStation {
    fn new(spec: NetworkNodeSpec) -> Self {
        let queue_limit_packets = spec
            .queue_limit_packets
            .unwrap_or_else(|| default_node_queue_limit(spec.kind));
        let forwarding_rate_pps = spec
            .forwarding_rate_pps
            .unwrap_or_else(|| default_forwarding_rate(spec.kind));
        NetworkNodeStation {
            spec,
            queue_limit_packets,
            forwarding_rate_pps,
            input_queue: VecDeque::new(),
            forwarding_credit: 0.0,
            queue_area: 0.0,
            max_queue: 0,
            total_queue_delay_ms: 0.0,
            max_queue_delay_ms: 0.0,
            processed_from_queue: 0,
            received_packets: 0,
            forwarded_packets: 0,
            delivered_packets: 0,
            dropped_packets: 0,
        }
    }

    fn node_id(&self) -> &str {
        &self.spec.id
    }

    fn can_accept_packet(&self, reserved_incoming: usize) -> bool {
        self.input_queue.len() + reserved_incoming < self.queue_limit_packets
    }

    /// Enqueue an arriving packet. Caller must have checked `can_accept_packet`.
    fn receive_packet(&mut self, mut packet: NetworkPacket, time_ms: f64) {
        packet.current_node_id = Some(self.spec.id.clone());
        packet.current_link_id = None;
        self.input_queue.push_back(QueuedNodePacket {
            packet,
            enqueued_at_ms: time_ms,
        });
        self.received_packets += 1;
        self.max_queue = self.max_queue.max(self.input_queue.len());
    }

    fn queued_packets(&self) -> usize {
        self.input_queue.len()
    }

    fn record_queue(&mut self, dt_ms: f64) {
        self.queue_area += self.input_queue.len() as f64 * dt_ms;
        self.max_queue = self.max_queue.max(self.input_queue.len());
    }

    fn stats(&self, total_ms: f64) -> NetworkNodeStats {
        NetworkNodeStats {
            id: self.spec.id.clone(),
            kind: self.spec.kind,
            forwarding_rate_pps: self.forwarding_rate_pps,
            queue_limit_packets: self.queue_limit_packets as f64,
            received_packets: self.received_packets as f64,
            forwarded_packets: self.forwarded_packets as f64,
            delivered_packets: self.delivered_packets as f64,
            dropped_packets: self.dropped_packets as f64,
            final_queue: self.input_queue.len() as f64,
            max_queue: self.max_queue as f64,
            avg_queue: self.queue_area / total_ms.max(1.0),
            mean_queue_delay_ms: self.total_queue_delay_ms
                / (self.processed_from_queue.max(1) as f64),
            max_queue_delay_ms: self.max_queue_delay_ms,
        }
    }
}

// =============================================================================
// Link station.
// =============================================================================

pub struct NetworkLinkStation {
    spec: NetworkLinkSpec,
    queue_limit_packets: usize,
    cost_per_mb: f64,
    scheduled: Vec<ScheduledPacket>,
    available_at_ms: f64,
    occupancy_area: f64,
    max_in_flight: usize,
    enqueued_packets: u64,
    delivered_packets: u64,
    dropped_packets: u64,
    transmitted_bytes: f64,
    total_serialization_ms: f64,
    total_queue_delay_ms: f64,
    max_queue_delay_ms: f64,
    total_time_on_link_ms: f64,
    max_time_on_link_ms: f64,
    total_cost: f64,
}

impl NetworkLinkStation {
    fn new(spec: NetworkLinkSpec) -> Self {
        let queue_limit_packets = spec.queue_limit_packets.unwrap_or(64);
        let cost_per_mb = spec.cost_per_mb.unwrap_or(0.0);
        NetworkLinkStation {
            spec,
            queue_limit_packets,
            cost_per_mb,
            scheduled: Vec::new(),
            available_at_ms: 0.0,
            occupancy_area: 0.0,
            max_in_flight: 0,
            enqueued_packets: 0,
            delivered_packets: 0,
            dropped_packets: 0,
            transmitted_bytes: 0.0,
            total_serialization_ms: 0.0,
            total_queue_delay_ms: 0.0,
            max_queue_delay_ms: 0.0,
            total_time_on_link_ms: 0.0,
            max_time_on_link_ms: 0.0,
            total_cost: 0.0,
        }
    }

    fn link_id(&self) -> &str {
        &self.spec.id
    }

    fn can_accept_packet(&self) -> bool {
        self.scheduled.len() < self.queue_limit_packets
    }

    fn serialization_ms(&self, packet: &NetworkPacket) -> f64 {
        packet.size_bytes * 8.0 / (self.spec.bandwidth_mbps * 1e6) * 1000.0
    }

    fn enqueue_packet(&mut self, mut packet: NetworkPacket, time_ms: f64) {
        let serialization_ms = self.serialization_ms(&packet);
        let start_at_ms = time_ms.max(self.available_at_ms);
        let queue_delay_ms = (start_at_ms - time_ms).max(0.0);
        let deliver_at_ms = start_at_ms + serialization_ms + self.spec.latency_ms;
        let time_on_link_ms = queue_delay_ms + serialization_ms + self.spec.latency_ms;
        self.available_at_ms = start_at_ms + serialization_ms;
        let packet_cost = mb(packet.size_bytes) * self.cost_per_mb;
        packet.cost += packet_cost;
        packet.current_node_id = None;
        packet.current_link_id = Some(self.spec.id.clone());
        packet.hops.push(self.spec.to.clone());
        self.enqueued_packets += 1;
        self.transmitted_bytes += packet.size_bytes;
        self.total_serialization_ms += serialization_ms;
        self.total_queue_delay_ms += queue_delay_ms;
        self.max_queue_delay_ms = self.max_queue_delay_ms.max(queue_delay_ms);
        self.total_time_on_link_ms += time_on_link_ms;
        self.max_time_on_link_ms = self.max_time_on_link_ms.max(time_on_link_ms);
        self.total_cost += packet_cost;
        self.scheduled.push(ScheduledPacket {
            packet,
            deliver_at_ms,
        });
        self.max_in_flight = self.max_in_flight.max(self.scheduled.len());
    }

    fn release_arrivals(&mut self, time_ms: f64) -> Vec<NetworkPacket> {
        let mut ready: Vec<NetworkPacket> = Vec::new();
        let mut keep: Vec<ScheduledPacket> = Vec::new();
        for item in std::mem::take(&mut self.scheduled) {
            if item.deliver_at_ms <= time_ms + 1e-9 {
                ready.push(item.packet);
                self.delivered_packets += 1;
            } else {
                keep.push(item);
            }
        }
        self.scheduled = keep;
        ready
    }

    fn step_occupancy(&mut self, dt_ms: f64) {
        self.occupancy_area += self.scheduled.len() as f64 * dt_ms;
        self.max_in_flight = self.max_in_flight.max(self.scheduled.len());
    }

    fn scheduled_count(&self) -> usize {
        self.scheduled.len()
    }

    fn stats(&self, total_ms: f64) -> NetworkLinkStats {
        let simulated_sec = (total_ms / 1000.0).max(1e-9);
        NetworkLinkStats {
            id: self.spec.id.clone(),
            from: self.spec.from.clone(),
            to: self.spec.to.clone(),
            bandwidth_mbps: self.spec.bandwidth_mbps,
            latency_ms: self.spec.latency_ms,
            cost_per_mb: self.cost_per_mb,
            queue_limit_packets: self.queue_limit_packets as f64,
            enqueued_packets: self.enqueued_packets as f64,
            delivered_packets: self.delivered_packets as f64,
            dropped_packets: self.dropped_packets as f64,
            transmitted_bytes: self.transmitted_bytes,
            throughput_mbps: self.transmitted_bytes * 8.0 / simulated_sec / 1e6,
            utilization: 1.0_f64.min(self.total_serialization_ms / total_ms.max(1e-9)),
            final_in_flight: self.scheduled.len() as f64,
            max_in_flight: self.max_in_flight as f64,
            avg_in_flight: self.occupancy_area / total_ms.max(1.0),
            mean_queue_delay_ms: self.total_queue_delay_ms / (self.enqueued_packets.max(1) as f64),
            max_queue_delay_ms: self.max_queue_delay_ms,
            mean_time_on_link_ms: self.total_time_on_link_ms
                / (self.enqueued_packets.max(1) as f64),
            max_time_on_link_ms: self.max_time_on_link_ms,
            total_cost: self.total_cost,
        }
    }
}

// =============================================================================
// Routing topology (static).
// =============================================================================

#[derive(Clone, Debug)]
struct RouteEdge {
    link_id: String,
    to: String,
    weight: f64,
}

#[derive(Clone, Debug)]
struct FlowRuntimeState {
    spec: NetworkFlowSpec,
    pending: f64,
    generated: u64,
    dropped_at_source: u64,
}

// =============================================================================
// The network simulation station.
// =============================================================================

pub struct ComputerNetworkStation {
    core: StationCore,
    p: ComputerNetworkProblem,
    nodes: HashMap<String, NetworkNodeStation>,
    node_order: Vec<String>,
    links: HashMap<String, NetworkLinkStation>,
    link_order: Vec<String>,
    node_id_set: HashSet<String>,
    topo_outgoing: HashMap<String, Vec<RouteEdge>>,
    route_cache: HashMap<String, Option<String>>,
    flows: Vec<FlowRuntimeState>,
    delivered: Vec<NetworkPacket>,
    dropped: Vec<NetworkPacket>,
    time_series: Vec<NetworkTimeSample>,
    invariant_violations: Vec<String>,
    next_packet_id: u64,
    time_ms: f64,
    max_active_packets: usize,
    next_sample_at_ms: f64,
}

impl ComputerNetworkStation {
    pub fn new(p: ComputerNetworkProblem) -> Self {
        validate_computer_network_problem(&p).unwrap_or_else(|e| panic!("{e}"));
        let p = normalize_computer_network_problem(&p);
        let metric = p.routing_metric.unwrap_or(NetworkRoutingMetric::Latency);

        let mut nodes: HashMap<String, NetworkNodeStation> = HashMap::new();
        let mut node_order: Vec<String> = Vec::new();
        let mut node_id_set: HashSet<String> = HashSet::new();
        for n in &p.nodes {
            nodes.insert(n.id.clone(), NetworkNodeStation::new(n.clone()));
            node_order.push(n.id.clone());
            node_id_set.insert(n.id.clone());
        }

        let mut links: HashMap<String, NetworkLinkStation> = HashMap::new();
        let mut link_order: Vec<String> = Vec::new();
        let mut topo_outgoing: HashMap<String, Vec<RouteEdge>> = HashMap::new();
        for l in &p.links {
            links.insert(l.id.clone(), NetworkLinkStation::new(l.clone()));
            link_order.push(l.id.clone());
            topo_outgoing
                .entry(l.from.clone())
                .or_default()
                .push(RouteEdge {
                    link_id: l.id.clone(),
                    to: l.to.clone(),
                    weight: link_weight(l, metric),
                });
        }

        let flows: Vec<FlowRuntimeState> = p
            .flows
            .iter()
            .map(|spec| FlowRuntimeState {
                spec: spec.clone(),
                pending: 0.0,
                generated: 0,
                dropped_at_source: 0,
            })
            .collect();

        let mut station = ComputerNetworkStation {
            core: StationCore::new("computer-network"),
            p,
            nodes,
            node_order,
            links,
            link_order,
            node_id_set,
            topo_outgoing,
            route_cache: HashMap::new(),
            flows,
            delivered: Vec::new(),
            dropped: Vec::new(),
            time_series: Vec::new(),
            invariant_violations: Vec::new(),
            next_packet_id: 1,
            time_ms: 0.0,
            max_active_packets: 0,
            next_sample_at_ms: 0.0,
        };

        let conservation = intrinsic_check::<dyn DESStation>(
            "computer-network.conservation",
            |st: &dyn DESStation| {
                let s = downcast(st);
                s.generated_packets() as usize
                    == s.delivered.len() + s.dropped.len() + s.active_packets()
            },
            Some("generated = delivered + dropped + active".to_string()),
            Some(Box::new(|st: &dyn DESStation| {
                let s = downcast(st);
                format!(
                    "generated={}, delivered={}, dropped={}, active={}",
                    s.generated_packets(),
                    s.delivered.len(),
                    s.dropped.len(),
                    s.active_packets()
                )
            })),
            Some("computer-network-intrinsic".to_string()),
            None,
        )
        .boxed();
        station.add_validator(conservation);

        let capacity = intrinsic_check::<dyn DESStation>(
            "computer-network.queues-within-capacity",
            |st: &dyn DESStation| downcast(st).all_queues_within_capacity(),
            Some("node and link queues within configured packet limits".to_string()),
            Some(Box::new(|st: &dyn DESStation| {
                format!("violations={}", downcast(st).invariant_violations.len())
            })),
            Some("computer-network-intrinsic".to_string()),
            None,
        )
        .boxed();
        station.add_validator(capacity);

        station
    }

    pub fn build_result(&self) -> ComputerNetworkResult {
        let mut latencies: Vec<f64> = self
            .delivered
            .iter()
            .map(|p| p.delivered_at_ms.unwrap_or(self.time_ms) - p.created_at_ms)
            .collect();
        latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let total_delivered_bytes: f64 = self.delivered.iter().map(|p| p.size_bytes).sum();
        let total_delivered_payload_bytes: f64 =
            self.delivered.iter().map(|p| p.payload_bytes).sum();
        let total_generated_bytes: f64 = self
            .flows
            .iter()
            .map(|f| f.generated as f64 * effective_packet_size_bytes(&f.spec))
            .sum();
        let total_cost: f64 = self.delivered.iter().map(|p| p.cost).sum::<f64>()
            + self.dropped.iter().map(|p| p.cost).sum::<f64>();
        let simulated_sec = (self.p.duration_ms / 1000.0).max(1e-9);
        let node_stats: Vec<NetworkNodeStats> = self
            .node_order
            .iter()
            .map(|id| self.nodes[id].stats(self.time_ms))
            .collect();
        let link_stats: Vec<NetworkLinkStats> = self
            .link_order
            .iter()
            .map(|id| self.links[id].stats(self.time_ms))
            .collect();
        let flow_stats = self.build_flow_stats();
        let bottlenecks = identify_bottlenecks(&node_stats, &link_stats);
        ComputerNetworkResult {
            generated_packets: self.generated_packets() as f64,
            delivered_packets: self.delivered.len() as f64,
            dropped_packets: self.dropped.len() as f64,
            active_packets: self.active_packets() as f64,
            max_active_packets: self.max_active_packets as f64,
            delivery_ratio: self.delivered.len() as f64 / (self.generated_packets().max(1) as f64),
            offered_load_mbps: total_generated_bytes * 8.0 / simulated_sec / 1e6,
            throughput_mbps: total_delivered_bytes * 8.0 / simulated_sec / 1e6,
            goodput_mbps: total_delivered_payload_bytes * 8.0 / simulated_sec / 1e6,
            mean_latency_ms: mean(&latencies),
            p95_latency_ms: percentile(&latencies, 0.95),
            total_cost,
            total_simulated_ms: self.time_ms,
            routing_metric: self
                .p
                .routing_metric
                .unwrap_or(NetworkRoutingMetric::Latency),
            flow_stats,
            node_stats,
            link_stats,
            bottlenecks,
            time_series: self.time_series.clone(),
            delivered_packets_trace: self
                .delivered
                .iter()
                .take(200)
                .map(|p| p.snapshot())
                .collect(),
            dropped_packets_trace: self
                .dropped
                .iter()
                .take(200)
                .map(|p| p.snapshot())
                .collect(),
            invariant_violations: self.invariant_violations.clone(),
        }
    }

    // ── per-tick phases ──────────────────────────────────────────────────────

    fn release_link_arrivals(&mut self) {
        let time_ms = self.time_ms;
        // Process each link's arrivals immediately (matching the TS per-link
        // ordering); `release_arrivals` returns owned packets so the link
        // borrow is released before we touch the destination node.
        for li in 0..self.link_order.len() {
            let lid = self.link_order[li].clone();
            let to = self.links[&lid].spec.to.clone();
            let ready = self.links.get_mut(&lid).unwrap().release_arrivals(time_ms);
            for packet in ready {
                let outcome: Option<(NetworkPacket, PacketDropReason, String)> =
                    match self.nodes.get_mut(&to) {
                        Some(node) => {
                            if node.can_accept_packet(0) {
                                node.receive_packet(packet, time_ms);
                                None
                            } else {
                                node.dropped_packets += 1;
                                Some((
                                    packet,
                                    PacketDropReason::NodeQueueOverflow,
                                    node.spec.id.clone(),
                                ))
                            }
                        }
                        None => Some((packet, PacketDropReason::NoRoute, to.clone())),
                    };
                if let Some((packet, reason, sid)) = outcome {
                    self.drop(packet, reason, &sid);
                }
            }
        }
    }

    fn generate_flow_packets(&mut self) {
        let time_ms = self.time_ms;
        let dt_ms = self.p.dt_ms;
        let duration_ms = self.p.duration_ms;
        let max_in_system = self.p.max_packets_in_system;
        let default_ttl = (self.p.nodes.len() as i64 * 4).max(8);

        // Iterate flows by index so we can re-borrow `self` fields freely.
        let num_flows = self.flows.len();
        for fi in 0..num_flows {
            let spec = self.flows[fi].spec.clone();
            let profile = protocol_profile(spec.protocol);
            let flow_start_ms = spec.start_ms.unwrap_or(0.0) + profile.startup_delay_ms;
            let flow_end_ms = spec.end_ms.unwrap_or(duration_ms);
            if time_ms < flow_start_ms || time_ms > flow_end_ms {
                continue;
            }
            self.flows[fi].pending += spec.rate_pps * dt_ms / 1000.0;
            while self.flows[fi].pending >= 1.0 - 1e-12 {
                if let Some(mp) = spec.max_packets {
                    if self.flows[fi].generated >= mp {
                        self.flows[fi].pending = 0.0;
                        break;
                    }
                }
                let packet = NetworkPacket::new(
                    self.next_packet_id,
                    spec.id.clone(),
                    profile.protocol,
                    spec.source.clone(),
                    spec.destination.clone(),
                    spec.packet_size_bytes,
                    effective_packet_size_bytes(&spec),
                    time_ms,
                    spec.ttl_hops.unwrap_or(default_ttl),
                );
                self.next_packet_id += 1;
                self.flows[fi].generated += 1;
                self.flows[fi].pending -= 1.0;

                let active = self.active_packets() as u64;
                let over_cap = match max_in_system {
                    Some(cap) => active >= cap,
                    None => false,
                };
                if over_cap {
                    self.flows[fi].dropped_at_source += 1;
                    if let Some(source) = self.nodes.get_mut(&spec.source) {
                        source.dropped_packets += 1;
                    }
                    self.drop(packet, PacketDropReason::MaxPacketsInSystem, &spec.source);
                    continue;
                }
                let can = self
                    .nodes
                    .get(&spec.source)
                    .map(|n| n.can_accept_packet(0))
                    .unwrap_or(false);
                if can {
                    self.nodes
                        .get_mut(&spec.source)
                        .unwrap()
                        .receive_packet(packet, time_ms);
                } else {
                    self.flows[fi].dropped_at_source += 1;
                    if let Some(source) = self.nodes.get_mut(&spec.source) {
                        source.dropped_packets += 1;
                    }
                    self.drop(packet, PacketDropReason::NodeQueueOverflow, &spec.source);
                }
            }
        }
    }

    fn step_all_nodes(&mut self) {
        let time_ms = self.time_ms;
        let dt_ms = self.p.dt_ms;
        let node_order = self.node_order.clone();
        let ComputerNetworkStation {
            nodes,
            links,
            node_id_set,
            topo_outgoing,
            route_cache,
            delivered,
            dropped,
            ..
        } = self;
        for nid in &node_order {
            let node = match nodes.get_mut(nid) {
                Some(n) => n,
                None => continue,
            };
            node.forwarding_credit += node.forwarding_rate_pps * dt_ms / 1000.0;
            let mut budget = (node.forwarding_credit + 1e-12).floor() as i64;
            while budget > 0 && !node.input_queue.is_empty() {
                let queued = node.input_queue.pop_front().unwrap();
                let packet = queued.packet;
                let queue_delay_ms = (time_ms - queued.enqueued_at_ms).max(0.0);
                node.total_queue_delay_ms += queue_delay_ms;
                node.max_queue_delay_ms = node.max_queue_delay_ms.max(queue_delay_ms);
                node.processed_from_queue += 1;
                node.forwarding_credit -= 1.0;
                budget -= 1;

                let node_id = node.spec.id.clone();
                if packet.destination == node_id {
                    node.delivered_packets += 1;
                    deliver_packet(delivered, packet, &node_id, time_ms);
                    continue;
                }
                if packet.hops.len() as i64 > packet.ttl_hops {
                    node.dropped_packets += 1;
                    drop_into(
                        dropped,
                        packet,
                        PacketDropReason::TtlExpired,
                        &node_id,
                        time_ms,
                    );
                    continue;
                }
                let link_id = next_link(
                    node_id_set,
                    topo_outgoing,
                    route_cache,
                    &node_id,
                    &packet.destination,
                );
                match link_id {
                    None => {
                        node.dropped_packets += 1;
                        drop_into(
                            dropped,
                            packet,
                            PacketDropReason::NoRoute,
                            &node_id,
                            time_ms,
                        );
                    }
                    Some(lid) => {
                        let link = links.get_mut(&lid).unwrap();
                        if !link.can_accept_packet() {
                            node.dropped_packets += 1;
                            link.dropped_packets += 1;
                            let link_id = link.spec.id.clone();
                            drop_into(
                                dropped,
                                packet,
                                PacketDropReason::LinkQueueOverflow,
                                &link_id,
                                time_ms,
                            );
                        } else {
                            node.forwarded_packets += 1;
                            link.enqueue_packet(packet, time_ms);
                        }
                    }
                }
            }
            node.record_queue(dt_ms);
        }
    }

    fn deliver(&mut self, packet: NetworkPacket, at_node_id: &str) {
        deliver_packet(&mut self.delivered, packet, at_node_id, self.time_ms);
    }

    fn drop(&mut self, packet: NetworkPacket, reason: PacketDropReason, at_station_id: &str) {
        drop_into(
            &mut self.dropped,
            packet,
            reason,
            at_station_id,
            self.time_ms,
        );
    }

    fn build_flow_stats(&self) -> Vec<NetworkFlowStats> {
        self.flows
            .iter()
            .map(|flow| {
                let spec = &flow.spec;
                let protocol = protocol_profile(spec.protocol).protocol;
                let delivered: Vec<&NetworkPacket> = self
                    .delivered
                    .iter()
                    .filter(|p| p.flow_id == spec.id)
                    .collect();
                let dropped: Vec<&NetworkPacket> = self
                    .dropped
                    .iter()
                    .filter(|p| p.flow_id == spec.id)
                    .collect();
                let mut latencies: Vec<f64> = delivered
                    .iter()
                    .map(|p| p.delivered_at_ms.unwrap_or(self.time_ms) - p.created_at_ms)
                    .collect();
                latencies.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let delivered_bytes: f64 = delivered.iter().map(|p| p.size_bytes).sum();
                let delivered_payload_bytes: f64 = delivered.iter().map(|p| p.payload_bytes).sum();
                let total_cost: f64 = delivered.iter().map(|p| p.cost).sum::<f64>()
                    + dropped.iter().map(|p| p.cost).sum::<f64>();
                let simulated_sec = (self.p.duration_ms / 1000.0).max(1e-9);
                NetworkFlowStats {
                    id: spec.id.clone(),
                    protocol,
                    source: spec.source.clone(),
                    destination: spec.destination.clone(),
                    generated_packets: flow.generated as f64,
                    delivered_packets: delivered.len() as f64,
                    dropped_packets: dropped.len() as f64,
                    delivery_ratio: delivered.len() as f64 / (flow.generated.max(1) as f64),
                    generated_bytes: flow.generated as f64 * effective_packet_size_bytes(spec),
                    delivered_bytes,
                    offered_load_mbps: flow.generated as f64
                        * effective_packet_size_bytes(spec)
                        * 8.0
                        / simulated_sec
                        / 1e6,
                    throughput_mbps: delivered_bytes * 8.0 / simulated_sec / 1e6,
                    goodput_mbps: delivered_payload_bytes * 8.0 / simulated_sec / 1e6,
                    mean_latency_ms: mean(&latencies),
                    p95_latency_ms: percentile(&latencies, 0.95),
                    mean_time_in_system_ms: mean(&latencies),
                    p95_time_in_system_ms: percentile(&latencies, 0.95),
                    total_cost,
                    mean_cost_per_delivered_packet: total_cost / (delivered.len().max(1) as f64),
                }
            })
            .collect()
    }

    fn active_packets(&self) -> usize {
        let mut n = 0;
        for node in self.nodes.values() {
            n += node.queued_packets();
        }
        for link in self.links.values() {
            n += link.scheduled_count();
        }
        n
    }

    fn generated_packets(&self) -> u64 {
        self.flows.iter().map(|f| f.generated).sum()
    }

    fn all_queues_within_capacity(&self) -> bool {
        for node in self.nodes.values() {
            if node.queued_packets() > node.queue_limit_packets {
                return false;
            }
        }
        for link in self.links.values() {
            if link.scheduled_count() > link.queue_limit_packets {
                return false;
            }
        }
        true
    }

    fn record_stats(&mut self) {
        let active = self.active_packets();
        self.max_active_packets = self.max_active_packets.max(active);
        let sample_every_ms = self
            .p
            .sample_every_ms
            .unwrap_or_else(|| self.p.dt_ms.max(100.0));
        if self.time_ms + 1e-9 < self.next_sample_at_ms {
            return;
        }
        let mut node_queues: HashMap<String, f64> = HashMap::new();
        for id in &self.node_order {
            node_queues.insert(id.clone(), self.nodes[id].queued_packets() as f64);
        }
        let mut link_in_flight: HashMap<String, f64> = HashMap::new();
        let mut link_utilization: HashMap<String, f64> = HashMap::new();
        let elapsed_ms = (self.time_ms + self.p.dt_ms).max(1.0);
        for id in &self.link_order {
            link_in_flight.insert(id.clone(), self.links[id].scheduled_count() as f64);
            link_utilization.insert(id.clone(), self.links[id].stats(elapsed_ms).utilization);
        }
        self.time_series.push(NetworkTimeSample {
            t_ms: self.time_ms,
            generated_packets: self.generated_packets() as f64,
            delivered_packets: self.delivered.len() as f64,
            dropped_packets: self.dropped.len() as f64,
            active_packets: active as f64,
            node_queues,
            link_in_flight,
            link_utilization,
        });
        self.next_sample_at_ms += sample_every_ms;
    }

    fn record_invariants(&mut self) {
        for id in &self.node_order {
            let node = &self.nodes[id];
            if node.queued_packets() > node.queue_limit_packets {
                self.invariant_violations.push(format!(
                    "{}: node queue {} > {}",
                    node.spec.id,
                    node.queued_packets(),
                    node.queue_limit_packets
                ));
            }
        }
        for id in &self.link_order {
            let link = &self.links[id];
            if link.scheduled_count() > link.queue_limit_packets {
                self.invariant_violations.push(format!(
                    "{}: link queue {} > {}",
                    link.spec.id,
                    link.scheduled_count(),
                    link.queue_limit_packets
                ));
            }
        }
    }
}

impl DESStation for ComputerNetworkStation {
    fn core(&self) -> &StationCore {
        &self.core
    }
    fn core_mut(&mut self) -> &mut StationCore {
        &mut self.core
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn has_work(&self) -> bool {
        let drain_until = self.p.duration_ms + self.p.drain_after_sources_ms.unwrap_or(1000.0);
        self.time_ms < self.p.duration_ms
            || (self.active_packets() > 0 && self.time_ms < drain_until)
    }
    fn run_time_step(&mut self) {
        self.release_link_arrivals();
        if self.time_ms < self.p.duration_ms {
            self.generate_flow_packets();
        }
        self.step_all_nodes();
        let dt = self.p.dt_ms;
        for id in &self.link_order.clone() {
            self.links.get_mut(id).unwrap().step_occupancy(dt);
        }
        self.record_invariants();
        self.record_stats();
        self.time_ms += self.p.dt_ms;
    }
}

/// Downcast a `&dyn DESStation` to this concrete station (validators).
fn downcast(st: &dyn DESStation) -> &ComputerNetworkStation {
    st.as_any()
        .downcast_ref::<ComputerNetworkStation>()
        .expect("validator received a non-ComputerNetworkStation")
}

/// Mark `packet` delivered and record it (TS `deliver`).
fn deliver_packet(
    delivered: &mut Vec<NetworkPacket>,
    mut packet: NetworkPacket,
    node_id: &str,
    time_ms: f64,
) {
    packet.delivered_at_ms = Some(time_ms);
    packet.current_node_id = Some(node_id.to_string());
    packet.current_link_id = None;
    packet.finish();
    delivered.push(packet);
}

/// Mark `packet` dropped and record it (TS `drop`).
fn drop_into(
    dropped: &mut Vec<NetworkPacket>,
    mut packet: NetworkPacket,
    reason: PacketDropReason,
    station_id: &str,
    time_ms: f64,
) {
    packet.dropped_at_ms = Some(time_ms);
    packet.drop_reason = Some(reason);
    packet.current_node_id = Some(station_id.to_string());
    packet.current_link_id = None;
    packet.finish();
    dropped.push(packet);
}

// =============================================================================
// Routing (Dijkstra over the static topology).
// =============================================================================

fn next_link(
    node_id_set: &HashSet<String>,
    outgoing: &HashMap<String, Vec<RouteEdge>>,
    cache: &mut HashMap<String, Option<String>>,
    source: &str,
    destination: &str,
) -> Option<String> {
    let key = format!("{source}->{destination}");
    if let Some(v) = cache.get(&key) {
        return v.clone();
    }
    let link = shortest_next_link(node_id_set, outgoing, source, destination);
    cache.insert(key, link.clone());
    link
}

fn shortest_next_link(
    node_id_set: &HashSet<String>,
    outgoing: &HashMap<String, Vec<RouteEdge>>,
    source: &str,
    destination: &str,
) -> Option<String> {
    if source == destination {
        return None;
    }
    let mut dist: HashMap<String, f64> = node_id_set
        .iter()
        .map(|id| (id.clone(), f64::INFINITY))
        .collect();
    let mut prev_node: HashMap<String, String> = HashMap::new();
    let mut prev_link: HashMap<String, String> = HashMap::new();
    let mut unsettled: HashSet<String> = node_id_set.clone();
    dist.insert(source.to_string(), 0.0);
    while !unsettled.is_empty() {
        let mut u: Option<String> = None;
        let mut best = f64::INFINITY;
        for id in &unsettled {
            let d = *dist.get(id).unwrap_or(&f64::INFINITY);
            if d < best {
                best = d;
                u = Some(id.clone());
            }
        }
        let u = match u {
            Some(x) if best.is_finite() => x,
            _ => break,
        };
        unsettled.remove(&u);
        if u == destination {
            break;
        }
        if let Some(edges) = outgoing.get(&u) {
            for e in edges {
                if !unsettled.contains(&e.to) {
                    continue;
                }
                let nd = best + e.weight;
                if nd < *dist.get(&e.to).unwrap_or(&f64::INFINITY) {
                    dist.insert(e.to.clone(), nd);
                    prev_node.insert(e.to.clone(), u.clone());
                    prev_link.insert(e.to.clone(), e.link_id.clone());
                }
            }
        }
    }
    prev_link.get(destination)?;
    let mut cur = destination.to_string();
    let mut first = prev_link.get(&cur).unwrap().clone();
    while prev_node.get(&cur).map(|s| s.as_str()).unwrap_or(source) != source {
        cur = prev_node.get(&cur).unwrap().clone();
        first = prev_link.get(&cur).unwrap().clone();
    }
    Some(first)
}

// =============================================================================
// Validation / build helpers / construction.
// =============================================================================

/// Validate a problem (TS `validateComputerNetworkProblem`). Returns the first
/// failing precondition; the constructor turns this into a panic.
pub fn validate_computer_network_problem(p: &ComputerNetworkProblem) -> Check {
    Preconditions::non_empty(MODEL, "nodes", &p.nodes)?;
    Preconditions::non_empty(MODEL, "links", &p.links)?;
    Preconditions::non_empty(MODEL, "flows", &p.flows)?;
    Preconditions::positive(MODEL, "durationMs", p.duration_ms)?;
    Preconditions::positive(MODEL, "dtMs", p.dt_ms)?;
    if let Some(d) = p.drain_after_sources_ms {
        Preconditions::non_negative(MODEL, "drainAfterSourcesMs", d)?;
    }
    if let Some(m) = p.max_packets_in_system {
        Preconditions::integer_in_range(MODEL, "maxPacketsInSystem", m as f64, 1.0, 1e7)?;
    }
    if let Some(s) = p.sample_every_ms {
        Preconditions::positive(MODEL, "sampleEveryMs", s)?;
    }

    let mut node_ids: HashSet<String> = HashSet::new();
    let mut node_by_id: HashMap<String, NetworkNodeSpec> = HashMap::new();
    for n in &p.nodes {
        Preconditions::check(
            MODEL,
            &format!("node {}", n.id),
            "have a non-empty id",
            !n.id.is_empty(),
            Some(n.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("node {}", n.id),
            "be unique",
            !node_ids.contains(&n.id),
            Some(n.id.clone()),
        )?;
        node_ids.insert(n.id.clone());
        node_by_id.insert(n.id.clone(), n.clone());
        if let Some(r) = n.forwarding_rate_pps {
            Preconditions::positive(MODEL, &format!("{}.forwardingRatePps", n.id), r)?;
        }
        if let Some(q) = n.queue_limit_packets {
            Preconditions::integer_in_range(
                MODEL,
                &format!("{}.queueLimitPackets", n.id),
                q as f64,
                1.0,
                1e7,
            )?;
        }
    }

    let mut link_ids: HashSet<String> = HashSet::new();
    for l in &p.links {
        Preconditions::check(
            MODEL,
            &format!("link {}", l.id),
            "have a non-empty id",
            !l.id.is_empty(),
            Some(l.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("link {}", l.id),
            "be unique",
            !link_ids.contains(&l.id),
            Some(l.id.clone()),
        )?;
        link_ids.insert(l.id.clone());
        Preconditions::check(
            MODEL,
            &format!("{}.from", l.id),
            "reference a node",
            node_ids.contains(&l.from),
            Some(l.from.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.to", l.id),
            "reference a node",
            node_ids.contains(&l.to),
            Some(l.to.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.from != to", l.id),
            "hold",
            l.from != l.to,
            None,
        )?;
        Preconditions::positive(MODEL, &format!("{}.bandwidthMbps", l.id), l.bandwidth_mbps)?;
        Preconditions::non_negative(MODEL, &format!("{}.latencyMs", l.id), l.latency_ms)?;
        if let Some(c) = l.cost_per_mb {
            Preconditions::non_negative(MODEL, &format!("{}.costPerMb", l.id), c)?;
        }
        if let Some(q) = l.queue_limit_packets {
            Preconditions::integer_in_range(
                MODEL,
                &format!("{}.queueLimitPackets", l.id),
                q as f64,
                1.0,
                1e7,
            )?;
        }
    }

    let normalized = normalize_computer_network_problem(p);
    let mut outgoing: HashMap<String, Vec<String>> = HashMap::new();
    for l in &normalized.links {
        outgoing
            .entry(l.from.clone())
            .or_default()
            .push(l.to.clone());
    }
    let mut flow_ids: HashSet<String> = HashSet::new();
    for f in &p.flows {
        Preconditions::check(
            MODEL,
            &format!("flow {}", f.id),
            "have a non-empty id",
            !f.id.is_empty(),
            Some(f.id.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("flow {}", f.id),
            "be unique",
            !flow_ids.contains(&f.id),
            Some(f.id.clone()),
        )?;
        flow_ids.insert(f.id.clone());
        Preconditions::check(
            MODEL,
            &format!("{}.source", f.id),
            "reference a node",
            node_ids.contains(&f.source),
            Some(f.source.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.destination", f.id),
            "reference a node",
            node_ids.contains(&f.destination),
            Some(f.destination.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.source", f.id),
            "reference a host source entity",
            node_by_id.get(&f.source).map(|n| n.kind) == Some(NetworkNodeKind::Host),
            Some(f.source.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.destination", f.id),
            "reference a host sink entity",
            node_by_id.get(&f.destination).map(|n| n.kind) == Some(NetworkNodeKind::Host),
            Some(f.destination.clone()),
        )?;
        Preconditions::check(
            MODEL,
            &format!("{}.source != destination", f.id),
            "hold",
            f.source != f.destination,
            None,
        )?;
        Preconditions::non_negative(MODEL, &format!("{}.ratePps", f.id), f.rate_pps)?;
        Preconditions::integer_in_range(
            MODEL,
            &format!("{}.packetSizeBytes", f.id),
            f.packet_size_bytes,
            1.0,
            1e9,
        )?;
        if let Some(s) = f.start_ms {
            Preconditions::non_negative(MODEL, &format!("{}.startMs", f.id), s)?;
            Preconditions::check(
                MODEL,
                &format!("{}.startMs", f.id),
                "fall within durationMs",
                s <= p.duration_ms,
                None,
            )?;
        }
        if let Some(e) = f.end_ms {
            Preconditions::non_negative(MODEL, &format!("{}.endMs", f.id), e)?;
        }
        if let (Some(s), Some(e)) = (f.start_ms, f.end_ms) {
            Preconditions::check(
                MODEL,
                &format!("{}.startMs <= endMs", f.id),
                "hold",
                s <= e,
                None,
            )?;
        }
        if let Some(mp) = f.max_packets {
            Preconditions::integer_in_range(
                MODEL,
                &format!("{}.maxPackets", f.id),
                mp as f64,
                0.0,
                1e9,
            )?;
        }
        if let Some(t) = f.ttl_hops {
            Preconditions::integer_in_range(
                MODEL,
                &format!("{}.ttlHops", f.id),
                t as f64,
                1.0,
                1e6,
            )?;
        }
        Preconditions::check(
            MODEL,
            &format!("{}.route", f.id),
            "exist in directed link graph",
            has_directed_path(&f.source, &f.destination, &outgoing),
            None,
        )?;
    }
    Ok(())
}

/// Run a full simulation (TS `runComputerNetworkSimulation`).
pub fn run_computer_network_simulation(p: &ComputerNetworkProblem) -> ComputerNetworkResult {
    let problem = normalize_computer_network_problem(p);
    let station = Rc::new(RefCell::new(ComputerNetworkStation::new(p.clone())));
    let drain = problem.drain_after_sources_ms.unwrap_or(1000.0);
    let max_ticks = ((problem.duration_ms + drain) / problem.dt_ms).ceil() as usize + 5;
    let summary = run_iterative_des(
        vec![station.clone() as StationRef],
        IterativeRunOptions {
            shuffle: false,
            max_ticks: Some(max_ticks),
            ..Default::default()
        },
    );
    assert_no_validation_failures(&summary, MODEL).unwrap_or_else(|e| panic!("{e}"));
    let out = station.borrow().build_result();
    out
}

fn normalize_computer_network_problem(p: &ComputerNetworkProblem) -> ComputerNetworkProblem {
    let mut links: Vec<NetworkLinkSpec> = Vec::new();
    let mut ids: HashSet<String> = HashSet::new();
    for link in &p.links {
        let mut fwd = link.clone();
        fwd.bidirectional = Some(false);
        links.push(fwd);
        ids.insert(link.id.clone());
        if link.bidirectional == Some(true) {
            let mut reverse_id = format!("{}:rev", link.id);
            let mut i = 2;
            while ids.contains(&reverse_id) {
                reverse_id = format!("{}:rev{}", link.id, i);
                i += 1;
            }
            ids.insert(reverse_id.clone());
            let mut rev = link.clone();
            rev.id = reverse_id;
            rev.from = link.to.clone();
            rev.to = link.from.clone();
            rev.bidirectional = Some(false);
            links.push(rev);
        }
    }
    ComputerNetworkProblem {
        nodes: p.nodes.clone(),
        links,
        flows: p.flows.clone(),
        duration_ms: p.duration_ms,
        dt_ms: p.dt_ms,
        routing_metric: Some(p.routing_metric.unwrap_or(NetworkRoutingMetric::Latency)),
        drain_after_sources_ms: p.drain_after_sources_ms,
        max_packets_in_system: p.max_packets_in_system,
        sample_every_ms: p.sample_every_ms,
    }
}

struct ProtocolProfile {
    protocol: NetworkProtocol,
    overhead_bytes: f64,
    startup_delay_ms: f64,
}

fn protocol_profile(protocol: Option<NetworkProtocol>) -> ProtocolProfile {
    match protocol.unwrap_or(NetworkProtocol::Raw) {
        NetworkProtocol::Http => ProtocolProfile {
            protocol: NetworkProtocol::Http,
            overhead_bytes: 640.0,
            startup_delay_ms: 40.0,
        },
        NetworkProtocol::Tcp => ProtocolProfile {
            protocol: NetworkProtocol::Tcp,
            overhead_bytes: 40.0,
            startup_delay_ms: 20.0,
        },
        NetworkProtocol::Udp => ProtocolProfile {
            protocol: NetworkProtocol::Udp,
            overhead_bytes: 28.0,
            startup_delay_ms: 0.0,
        },
        NetworkProtocol::Raw => ProtocolProfile {
            protocol: NetworkProtocol::Raw,
            overhead_bytes: 0.0,
            startup_delay_ms: 0.0,
        },
    }
}

fn effective_packet_size_bytes(spec: &NetworkFlowSpec) -> f64 {
    spec.packet_size_bytes + protocol_profile(spec.protocol).overhead_bytes
}

fn identify_bottlenecks(
    node_stats: &[NetworkNodeStats],
    link_stats: &[NetworkLinkStats],
) -> Vec<NetworkBottleneckReport> {
    let mut reports: Vec<NetworkBottleneckReport> = Vec::new();
    for l in link_stats {
        let queue_pressure = l.avg_in_flight / l.queue_limit_packets.max(1.0);
        let delay_pressure = 1.0_f64.min(l.mean_queue_delay_ms / 1000.0);
        let drop_pressure =
            1.0_f64.min(l.dropped_packets / (l.enqueued_packets + l.dropped_packets).max(1.0));
        let score = l.utilization + queue_pressure + delay_pressure + drop_pressure;
        reports.push(NetworkBottleneckReport {
            id: l.id.clone(),
            kind: "link".to_string(),
            score,
            reason: bottleneck_reason(
                Some(l.utilization),
                l.avg_in_flight,
                l.max_in_flight,
                l.dropped_packets,
                l.mean_queue_delay_ms,
            ),
            utilization: Some(l.utilization),
            avg_queue: l.avg_in_flight,
            max_queue: l.max_in_flight,
            dropped_packets: l.dropped_packets,
            mean_queue_delay_ms: l.mean_queue_delay_ms,
        });
    }
    for n in node_stats {
        let queue_pressure = n.avg_queue / n.queue_limit_packets.max(1.0);
        let delay_pressure = 1.0_f64.min(n.mean_queue_delay_ms / 1000.0);
        let drop_pressure =
            1.0_f64.min(n.dropped_packets / (n.received_packets + n.dropped_packets).max(1.0));
        let service_pressure = if n.forwarded_packets > 0.0 && n.avg_queue > 0.0 {
            0.25
        } else {
            0.0
        };
        let score = queue_pressure + delay_pressure + drop_pressure + service_pressure;
        reports.push(NetworkBottleneckReport {
            id: n.id.clone(),
            kind: "node".to_string(),
            score,
            reason: bottleneck_reason(
                None,
                n.avg_queue,
                n.max_queue,
                n.dropped_packets,
                n.mean_queue_delay_ms,
            ),
            utilization: None,
            avg_queue: n.avg_queue,
            max_queue: n.max_queue,
            dropped_packets: n.dropped_packets,
            mean_queue_delay_ms: n.mean_queue_delay_ms,
        });
    }
    reports.retain(|r| r.score > 0.0 || r.dropped_packets > 0.0 || r.max_queue > 0.0);
    reports.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
    reports.truncate(8);
    reports
}

fn bottleneck_reason(
    utilization: Option<f64>,
    avg_queue: f64,
    max_queue: f64,
    dropped_packets: f64,
    mean_queue_delay_ms: f64,
) -> String {
    if dropped_packets > 0.0 {
        return "drops observed".to_string();
    }
    if let Some(u) = utilization {
        if u >= 0.95 {
            return "saturated link".to_string();
        }
    }
    if mean_queue_delay_ms >= 10.0 {
        return "queueing delay".to_string();
    }
    if avg_queue >= 1.0 || max_queue >= 10.0 {
        return "queue buildup".to_string();
    }
    if let Some(u) = utilization {
        if u >= 0.75 {
            return "high utilization".to_string();
        }
    }
    "capacity pressure".to_string()
}

fn default_forwarding_rate(kind: NetworkNodeKind) -> f64 {
    match kind {
        NetworkNodeKind::Host => 1000.0,
        NetworkNodeKind::Switch => 10000.0,
        NetworkNodeKind::Router => 5000.0,
    }
}

fn default_node_queue_limit(kind: NetworkNodeKind) -> usize {
    match kind {
        NetworkNodeKind::Host => 128,
        NetworkNodeKind::Switch => 512,
        NetworkNodeKind::Router => 256,
    }
}

fn link_weight(link: &NetworkLinkSpec, metric: NetworkRoutingMetric) -> f64 {
    match metric {
        NetworkRoutingMetric::Hop => 1.0,
        NetworkRoutingMetric::Cost => link.cost_per_mb.unwrap_or(0.0),
        NetworkRoutingMetric::Latency => {
            link.latency_ms + (1500.0 * 8.0 / (link.bandwidth_mbps * 1e6) * 1000.0)
        }
    }
}

fn has_directed_path(source: &str, sink: &str, outgoing: &HashMap<String, Vec<String>>) -> bool {
    let mut seen: HashSet<String> = HashSet::new();
    seen.insert(source.to_string());
    let mut q: Vec<String> = vec![source.to_string()];
    let mut qi = 0;
    while qi < q.len() {
        let u = q[qi].clone();
        qi += 1;
        if u == sink {
            return true;
        }
        if let Some(neighbours) = outgoing.get(&u) {
            for v in neighbours {
                if seen.contains(v) {
                    continue;
                }
                seen.insert(v.clone());
                q.push(v.clone());
            }
        }
    }
    false
}

fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    xs.iter().sum::<f64>() / xs.len() as f64
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((p * (sorted.len() as f64 - 1.0)).floor() as usize).min(sorted.len() - 1);
    sorted[idx]
}

fn mb(bytes: f64) -> f64 {
    bytes / 1_000_000.0
}

// =============================================================================
// Example problems.
// =============================================================================

fn node(id: &str, kind: NetworkNodeKind, fwd: f64, qlim: usize) -> NetworkNodeSpec {
    NetworkNodeSpec {
        id: id.to_string(),
        kind,
        forwarding_rate_pps: Some(fwd),
        queue_limit_packets: Some(qlim),
    }
}

#[allow(clippy::too_many_arguments)]
fn link(
    id: &str,
    from: &str,
    to: &str,
    bw: f64,
    lat: f64,
    cost: f64,
    qlim: usize,
) -> NetworkLinkSpec {
    NetworkLinkSpec {
        id: id.to_string(),
        from: from.to_string(),
        to: to.to_string(),
        bandwidth_mbps: bw,
        latency_ms: lat,
        cost_per_mb: Some(cost),
        queue_limit_packets: Some(qlim),
        bidirectional: Some(true),
    }
}

pub fn build_default_computer_network_problem() -> ComputerNetworkProblem {
    ComputerNetworkProblem {
        nodes: vec![
            node("client-a", NetworkNodeKind::Host, 2000.0, 256),
            node("client-b", NetworkNodeKind::Host, 2000.0, 256),
            node("edge-1", NetworkNodeKind::Router, 6000.0, 512),
            node("core-1", NetworkNodeKind::Router, 8000.0, 512),
            node("server", NetworkNodeKind::Host, 4000.0, 512),
        ],
        links: vec![
            link(
                "client-a-edge",
                "client-a",
                "edge-1",
                100.0,
                1.0,
                0.001,
                128,
            ),
            link("client-b-edge", "client-b", "edge-1", 50.0, 2.0, 0.001, 128),
            link("edge-core", "edge-1", "core-1", 25.0, 8.0, 0.004, 96),
            link("core-server", "core-1", "server", 100.0, 3.0, 0.002, 128),
        ],
        flows: vec![
            NetworkFlowSpec {
                id: "a-to-server".to_string(),
                source: "client-a".to_string(),
                destination: "server".to_string(),
                protocol: Some(NetworkProtocol::Http),
                rate_pps: 650.0,
                packet_size_bytes: 1200.0,
                start_ms: None,
                end_ms: None,
                max_packets: Some(650),
                ttl_hops: None,
            },
            NetworkFlowSpec {
                id: "b-to-server".to_string(),
                source: "client-b".to_string(),
                destination: "server".to_string(),
                protocol: Some(NetworkProtocol::Tcp),
                rate_pps: 300.0,
                packet_size_bytes: 1000.0,
                start_ms: None,
                end_ms: None,
                max_packets: Some(300),
                ttl_hops: None,
            },
        ],
        duration_ms: 1000.0,
        dt_ms: 1.0,
        routing_metric: Some(NetworkRoutingMetric::Latency),
        drain_after_sources_ms: Some(1500.0),
        max_packets_in_system: Some(5000),
        sample_every_ms: Some(100.0),
    }
}

pub fn build_bottleneck_computer_network_problem() -> ComputerNetworkProblem {
    ComputerNetworkProblem {
        nodes: vec![
            node("web-client", NetworkNodeKind::Host, 6000.0, 512),
            node("telemetry-client", NetworkNodeKind::Host, 6000.0, 512),
            node("edge", NetworkNodeKind::Switch, 12000.0, 1024),
            node("wan-router", NetworkNodeKind::Router, 9000.0, 1024),
            node("api-server", NetworkNodeKind::Host, 9000.0, 1024),
        ],
        links: vec![
            link("web-edge", "web-client", "edge", 100.0, 1.0, 0.001, 256),
            link(
                "telemetry-edge",
                "telemetry-client",
                "edge",
                100.0,
                1.0,
                0.001,
                256,
            ),
            link("edge-wan", "edge", "wan-router", 5.0, 25.0, 0.010, 96),
            link(
                "wan-api",
                "wan-router",
                "api-server",
                100.0,
                4.0,
                0.002,
                256,
            ),
        ],
        flows: vec![
            NetworkFlowSpec {
                id: "http-api".to_string(),
                source: "web-client".to_string(),
                destination: "api-server".to_string(),
                protocol: Some(NetworkProtocol::Http),
                rate_pps: 900.0,
                packet_size_bytes: 1100.0,
                start_ms: None,
                end_ms: None,
                max_packets: Some(1800),
                ttl_hops: None,
            },
            NetworkFlowSpec {
                id: "udp-telemetry".to_string(),
                source: "telemetry-client".to_string(),
                destination: "api-server".to_string(),
                protocol: Some(NetworkProtocol::Udp),
                rate_pps: 700.0,
                packet_size_bytes: 900.0,
                start_ms: None,
                end_ms: None,
                max_packets: Some(1400),
                ttl_hops: None,
            },
            NetworkFlowSpec {
                id: "tcp-bulk".to_string(),
                source: "web-client".to_string(),
                destination: "api-server".to_string(),
                protocol: Some(NetworkProtocol::Tcp),
                rate_pps: 350.0,
                packet_size_bytes: 1400.0,
                start_ms: None,
                end_ms: None,
                max_packets: Some(700),
                ttl_hops: None,
            },
        ],
        duration_ms: 2000.0,
        dt_ms: 1.0,
        routing_metric: Some(NetworkRoutingMetric::Latency),
        drain_after_sources_ms: Some(4000.0),
        max_packets_in_system: Some(10000),
        sample_every_ms: Some(100.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_problem_validates_and_runs() {
        let p = build_default_computer_network_problem();
        assert!(validate_computer_network_problem(&p).is_ok());
        let result = run_computer_network_simulation(&p);
        // Conservation: generated = delivered + dropped + active.
        let conserved = result.delivered_packets + result.dropped_packets + result.active_packets;
        assert!((result.generated_packets - conserved).abs() < 0.5);
        assert!(result.generated_packets > 0.0);
        assert!(result.delivered_packets > 0.0);
        assert_eq!(result.node_stats.len(), 5);
    }

    #[test]
    fn rejects_empty_topology() {
        let mut p = build_default_computer_network_problem();
        p.nodes.clear();
        assert!(validate_computer_network_problem(&p).is_err());
    }
}
