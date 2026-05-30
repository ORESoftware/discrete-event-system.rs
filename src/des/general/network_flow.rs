//! Rust port of the max-flow slice from `src/des/general/network-flow.ts`.

use crate::core::DesDecimal;
use crate::des::general::des_base::{
    preconditions::Preconditions,
    runner::{run_iterative_des, IterativeRunOptions},
    station::{DESRunLoopEntity, DESStation, HasRunTimeStep, StationCore},
    validation::{intrinsic_check, IntrinsicCheckOptions, ValidationCheck},
};
use crate::migration::MigrationFile;
use serde::{Deserialize, Serialize};
use serde_json::json;

pub const MIGRATION: MigrationFile = MigrationFile::ported_core(
    "src/des/general/network-flow.ts",
    "src/des/general/network_flow.rs",
    &[
        "Flow params/results are serde structs.",
        "MaxFlowOptimizationStation implements the shared DESStation traits.",
        "Capacity, residual, bottleneck, and accumulated flow math uses DesDecimal, not f64.",
        "Precondition failures and runner failures return Result.",
        "Traffic-flow declarations remain pending their own mapped port.",
    ],
    &[
        "AugmentingPathToken",
        "CarToken",
        "FlowEdge",
        "FlowEdgeResult",
        "MaxFlowMinCut",
        "MaxFlowOptimizationStation",
        "MaxFlowParams",
        "MaxFlowResult",
        "MaxFlowTraceRow",
        "OptimizationLogger",
        "TrafficCarSnapshot",
        "TrafficCellStation",
        "TrafficCellStats",
        "TrafficGridStation",
        "TrafficLane",
        "TrafficNetwork",
        "TrafficNode",
        "TrafficNodeKind",
        "TrafficParams",
        "TrafficResult",
        "TrafficScheduledTrip",
        "TrafficSignal",
        "TrafficSignalPhase",
        "TrafficSink",
        "TrafficSource",
        "TrafficTraceRow",
        "buildFiveIntersectionTrafficNetwork",
        "runMaxFlow",
        "runTrafficFlow",
    ],
);

pub type NetworkFlowResult<T> = Result<T, String>;

fn residual_epsilon() -> DesDecimal {
    DesDecimal::new(1, 9)
}

fn validation_epsilon() -> DesDecimal {
    DesDecimal::new(1, 7)
}

fn decimal_abs(value: DesDecimal) -> DesDecimal {
    if value < DesDecimal::ZERO {
        -value
    } else {
        value
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FlowEdge {
    pub from: usize,
    pub to: usize,
    pub capacity: DesDecimal,
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaxFlowParams {
    pub num_nodes: usize,
    pub source: usize,
    pub sink: usize,
    pub edges: Vec<FlowEdge>,
    pub max_augmentations: Option<usize>,
    pub node_coordinates: Option<Vec<(f64, f64)>>,
    pub node_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FlowEdgeResult {
    pub from: usize,
    pub to: usize,
    pub capacity: DesDecimal,
    pub name: Option<String>,
    pub flow: DesDecimal,
    pub residual: DesDecimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaxFlowTraceRow {
    pub iter: usize,
    pub path_nodes: Vec<usize>,
    pub path_edges: Vec<usize>,
    pub bottleneck: DesDecimal,
    pub value: DesDecimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaxFlowMinCut {
    pub source_side: Vec<usize>,
    pub sink_side: Vec<usize>,
    pub cut_edges: Vec<usize>,
    pub capacity: DesDecimal,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaxFlowResult {
    pub params: MaxFlowParams,
    pub max_flow: DesDecimal,
    pub edge_flows: Vec<FlowEdgeResult>,
    pub min_cut: MaxFlowMinCut,
    pub trace: Vec<MaxFlowTraceRow>,
    pub validation: Vec<ValidationCheck>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct ResidualStep {
    edge: usize,
    dir: i8,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AugmentingPathToken {
    pub row: MaxFlowTraceRow,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MaxFlowLogEvent {
    pub kind: String,
    pub level: String,
    pub row: MaxFlowTraceRow,
}

pub trait OptimizationLogger {
    fn log(&mut self, event: MaxFlowLogEvent);
}

pub struct MaxFlowOptimizationStation {
    core: StationCore<Self>,
    pub params: MaxFlowParams,
    flow: Vec<DesDecimal>,
    pub trace: Vec<MaxFlowTraceRow>,
    done: bool,
    value: DesDecimal,
    logger: Option<Box<dyn OptimizationLogger>>,
}

impl MaxFlowOptimizationStation {
    pub fn new(params: MaxFlowParams, logger: Option<Box<dyn OptimizationLogger>>) -> Self {
        let flow = vec![DesDecimal::ZERO; params.edges.len()];
        let mut station = Self {
            core: StationCore::new("max-flow"),
            params,
            flow,
            trace: Vec::new(),
            done: false,
            value: DesDecimal::ZERO,
            logger,
        };
        station.attach_validators();
        station
    }

    fn attach_validators(&mut self) {
        let mut capacity =
            IntrinsicCheckOptions::<Self>::new("max-flow-capacity-feasible", |station| {
                station.edge_flows().iter().all(|edge| {
                    edge.flow >= -validation_epsilon()
                        && edge.flow <= edge.capacity + validation_epsilon()
                })
            });
        capacity.expected = Some("0 <= flow <= capacity on every edge".to_owned());
        capacity.group = Some("max-flow".to_owned());
        self.add_validator(intrinsic_check(capacity));

        let mut conservation =
            IntrinsicCheckOptions::<Self>::new("max-flow-conservation", |station| {
                station.flow_conservation_ok()
            });
        conservation.expected =
            Some("inflow equals outflow at every transshipment node".to_owned());
        conservation.group = Some("max-flow".to_owned());
        self.add_validator(intrinsic_check(conservation));

        let mut min_cut = IntrinsicCheckOptions::<Self>::new("max-flow-min-cut-tight", |station| {
            decimal_abs(station.current_value() - station.min_cut().capacity)
                <= validation_epsilon()
        });
        min_cut.observed_fn = Some(Box::new(|station| {
            format!(
                "flow={} cut={}",
                station.current_value(),
                station.min_cut().capacity
            )
        }));
        min_cut.expected = Some("max-flow value equals residual min-cut capacity".to_owned());
        min_cut.group = Some("max-flow".to_owned());
        self.add_validator(intrinsic_check(min_cut));
    }

    pub fn current_value(&self) -> DesDecimal {
        self.value
    }

    pub fn edge_flows(&self) -> Vec<FlowEdgeResult> {
        self.params
            .edges
            .iter()
            .enumerate()
            .map(|(index, edge)| FlowEdgeResult {
                from: edge.from,
                to: edge.to,
                capacity: edge.capacity,
                name: edge.name.clone(),
                flow: self.flow[index],
                residual: edge.capacity - self.flow[index],
            })
            .collect()
    }

    pub fn result(&self, validation: Vec<ValidationCheck>) -> MaxFlowResult {
        MaxFlowResult {
            params: self.params.clone(),
            max_flow: self.value,
            edge_flows: self.edge_flows(),
            min_cut: self.min_cut(),
            trace: self.trace.clone(),
            validation,
        }
    }

    pub fn min_cut(&self) -> MaxFlowMinCut {
        let seen = self.residual_reachable();
        let mut source_side = Vec::new();
        let mut sink_side = Vec::new();
        for node in 0..self.params.num_nodes {
            if seen[node] {
                source_side.push(node);
            } else {
                sink_side.push(node);
            }
        }

        let mut cut_edges = Vec::new();
        let mut capacity = DesDecimal::ZERO;
        for (index, edge) in self.params.edges.iter().enumerate() {
            if seen[edge.from] && !seen[edge.to] {
                cut_edges.push(index);
                capacity += edge.capacity;
            }
        }
        MaxFlowMinCut {
            source_side,
            sink_side,
            cut_edges,
            capacity,
        }
    }

    fn validate_params(&self) -> NetworkFlowResult<()> {
        let params = &self.params;
        Preconditions::integer_in_range(
            "MaxFlowOptimizationStation",
            "numNodes",
            params.num_nodes as f64,
            2,
            10000,
        )
        .map_err(|err| err.to_string())?;
        Preconditions::integer_in_range(
            "MaxFlowOptimizationStation",
            "source",
            params.source as f64,
            0,
            params.num_nodes as i64 - 1,
        )
        .map_err(|err| err.to_string())?;
        Preconditions::integer_in_range(
            "MaxFlowOptimizationStation",
            "sink",
            params.sink as f64,
            0,
            params.num_nodes as i64 - 1,
        )
        .map_err(|err| err.to_string())?;
        Preconditions::check(
            "MaxFlowOptimizationStation",
            "sink",
            "differ from source",
            params.sink != params.source,
            Some(json!(params.sink)),
        )
        .map_err(|err| err.to_string())?;
        Preconditions::non_empty("MaxFlowOptimizationStation", "edges", &params.edges)
            .map_err(|err| err.to_string())?;
        for (index, edge) in params.edges.iter().enumerate() {
            Preconditions::integer_in_range(
                "MaxFlowOptimizationStation",
                &format!("edges[{index}].from"),
                edge.from as f64,
                0,
                params.num_nodes as i64 - 1,
            )
            .map_err(|err| err.to_string())?;
            Preconditions::integer_in_range(
                "MaxFlowOptimizationStation",
                &format!("edges[{index}].to"),
                edge.to as f64,
                0,
                params.num_nodes as i64 - 1,
            )
            .map_err(|err| err.to_string())?;
            Preconditions::check(
                "MaxFlowOptimizationStation",
                &format!("edges[{index}]"),
                "not be a self-loop",
                edge.from != edge.to,
                Some(json!(edge)),
            )
            .map_err(|err| err.to_string())?;
            Preconditions::check(
                "MaxFlowOptimizationStation",
                &format!("edges[{index}].capacity"),
                "be non-negative",
                edge.capacity >= DesDecimal::ZERO,
                Some(json!(edge.capacity)),
            )
            .map_err(|err| err.to_string())?;
        }
        if let Some(max_augmentations) = params.max_augmentations {
            Preconditions::integer_in_range(
                "MaxFlowOptimizationStation",
                "maxAugmentations",
                max_augmentations as f64,
                1,
                i64::MAX,
            )
            .map_err(|err| err.to_string())?;
        }
        if let Some(node_coordinates) = &params.node_coordinates {
            Preconditions::length_eq(
                "MaxFlowOptimizationStation",
                "nodeCoordinates",
                node_coordinates,
                params.num_nodes,
            )
            .map_err(|err| err.to_string())?;
        }
        if let Some(node_names) = &params.node_names {
            Preconditions::length_eq(
                "MaxFlowOptimizationStation",
                "nodeNames",
                node_names,
                params.num_nodes,
            )
            .map_err(|err| err.to_string())?;
        }
        Ok(())
    }

    fn flow_conservation_ok(&self) -> bool {
        let mut balance = vec![DesDecimal::ZERO; self.params.num_nodes];
        for (index, edge) in self.params.edges.iter().enumerate() {
            balance[edge.from] -= self.flow[index];
            balance[edge.to] += self.flow[index];
        }
        for (node, value) in balance.iter().enumerate() {
            if node == self.params.source || node == self.params.sink {
                continue;
            }
            if decimal_abs(*value) > validation_epsilon() {
                return false;
            }
        }
        decimal_abs(balance[self.params.sink] - self.value) <= validation_epsilon()
            && decimal_abs(balance[self.params.source] + self.value) <= validation_epsilon()
    }

    fn residual_capacity(&self, step: &ResidualStep) -> DesDecimal {
        let edge = &self.params.edges[step.edge];
        if step.dir == 1 {
            edge.capacity - self.flow[step.edge]
        } else {
            self.flow[step.edge]
        }
    }

    fn neighbors(&self, node: usize) -> Vec<(usize, ResidualStep)> {
        let mut out = Vec::new();
        for (index, edge) in self.params.edges.iter().enumerate() {
            if edge.from == node && edge.capacity - self.flow[index] > residual_epsilon() {
                out.push((
                    edge.to,
                    ResidualStep {
                        edge: index,
                        dir: 1,
                    },
                ));
            }
            if edge.to == node && self.flow[index] > residual_epsilon() {
                out.push((
                    edge.from,
                    ResidualStep {
                        edge: index,
                        dir: -1,
                    },
                ));
            }
        }
        out
    }

    fn find_augmenting_path(&self) -> Option<(Vec<usize>, Vec<ResidualStep>)> {
        let mut parent: Vec<Option<(usize, ResidualStep)>> = vec![None; self.params.num_nodes];
        let mut queue = vec![self.params.source];
        parent[self.params.source] = Some((
            usize::MAX,
            ResidualStep {
                edge: usize::MAX,
                dir: 1,
            },
        ));

        let mut cursor = 0usize;
        while cursor < queue.len() {
            let node = queue[cursor];
            if node == self.params.sink {
                break;
            }
            for (next, step) in self.neighbors(node) {
                if parent[next].is_some() {
                    continue;
                }
                parent[next] = Some((node, step));
                queue.push(next);
            }
            cursor += 1;
        }

        parent[self.params.sink].as_ref()?;
        let mut nodes = Vec::new();
        let mut steps = Vec::new();
        let mut current = self.params.sink;
        while current != self.params.source {
            let (previous, step) = parent[current].clone()?;
            nodes.push(current);
            steps.push(step);
            current = previous;
        }
        nodes.push(self.params.source);
        nodes.reverse();
        steps.reverse();
        Some((nodes, steps))
    }

    fn residual_reachable(&self) -> Vec<bool> {
        let mut seen = vec![false; self.params.num_nodes];
        let mut queue = vec![self.params.source];
        seen[self.params.source] = true;
        let mut cursor = 0usize;
        while cursor < queue.len() {
            for (next, _) in self.neighbors(queue[cursor]) {
                if seen[next] {
                    continue;
                }
                seen[next] = true;
                queue.push(next);
            }
            cursor += 1;
        }
        seen
    }
}

impl HasRunTimeStep for MaxFlowOptimizationStation {
    fn run_time_step(&mut self) {
        if self.done {
            return;
        }
        if self
            .params
            .max_augmentations
            .map(|max| self.trace.len() >= max)
            .unwrap_or(false)
        {
            self.done = true;
            return;
        }
        let Some((path_nodes, steps)) = self.find_augmenting_path() else {
            self.done = true;
            return;
        };

        let bottleneck = steps
            .iter()
            .map(|step| self.residual_capacity(step))
            .reduce(|left, right| if left <= right { left } else { right })
            .unwrap_or(DesDecimal::ZERO);
        for step in &steps {
            if step.dir == 1 {
                self.flow[step.edge] += bottleneck;
            } else {
                self.flow[step.edge] -= bottleneck;
            }
        }
        self.value += bottleneck;
        let row = MaxFlowTraceRow {
            iter: self.trace.len() + 1,
            path_nodes,
            path_edges: steps.iter().map(|step| step.edge).collect(),
            bottleneck,
            value: self.value,
        };
        self.trace.push(row.clone());
        if let Some(logger) = self.logger.as_mut() {
            logger.log(MaxFlowLogEvent {
                kind: "max-flow-augment".to_owned(),
                level: "info".to_owned(),
                row,
            });
        }
    }
}

impl DESRunLoopEntity for MaxFlowOptimizationStation {
    fn id(&self) -> Option<&str> {
        Some(self.core.id())
    }

    fn assert_preconditions(&self) -> Result<(), String> {
        self.validate_params()
    }

    fn has_work(&self) -> bool {
        !self.done
    }

    fn num_validators(&self) -> usize {
        self.core.num_validators()
    }

    fn run_validation(&self) -> Vec<ValidationCheck> {
        self.core.run_validation(self)
    }
}

impl DESStation for MaxFlowOptimizationStation {
    fn core(&self) -> &StationCore<Self> {
        &self.core
    }

    fn core_mut(&mut self) -> &mut StationCore<Self> {
        &mut self.core
    }
}

pub fn run_max_flow(params: MaxFlowParams) -> NetworkFlowResult<MaxFlowResult> {
    let max_ticks = params
        .max_augmentations
        .unwrap_or(params.edges.len() * params.num_nodes + 1)
        + 2;
    let mut station = MaxFlowOptimizationStation::new(params, None);
    let summary = {
        let mut participants: [&mut dyn DESRunLoopEntity; 1] = [&mut station];
        run_iterative_des(
            &mut participants,
            IterativeRunOptions {
                max_ticks: Some(max_ticks),
                shuffle: false,
                ..Default::default()
            },
        )?
    };
    Ok(station.result(summary.validation.unwrap_or_default()))
}
