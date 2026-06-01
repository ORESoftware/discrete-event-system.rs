//! Model-analysis data for the Studio UI: component summaries, connection
//! tables, N2 matrix cells, validation status, and executive selection.

use serde::{Deserialize, Serialize};

use crate::des::exec::{requirements_for_studio, select};

use super::spec::{
    compile_model_spec, studio_block_io, StudioBlockKind, StudioConstraintSpec,
    StudioDesignVariableSpec, StudioModelSpec, StudioObjectiveSpec,
};

/// One component row/column in the UI analysis views.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioComponentAnalysis {
    pub id: String,
    pub label: String,
    pub kind: StudioBlockKind,
    pub role: String,
    pub inputs: usize,
    pub outputs: usize,
    pub stateful: bool,
    pub elements: Vec<String>,
    pub x: f64,
    pub y: f64,
}

/// One explicit connection between block ports.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioConnectionAnalysis {
    pub from: String,
    pub from_port: usize,
    pub to: String,
    pub to_port: usize,
}

/// Sparse cell in an OpenMDAO-inspired N2 matrix. Rows are consumers, columns
/// are providers.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioN2Cell {
    pub row: usize,
    pub col: usize,
    pub connections: Vec<StudioConnectionAnalysis>,
}

/// Validation and run-routing status for a model spec.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioValidationAnalysis {
    pub ok: bool,
    pub message: Option<String>,
    pub execution_order: Vec<String>,
    pub executive: Option<String>,
}

/// Full analysis payload consumed by the generated workbench and APIs.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StudioAnalysis {
    pub name: String,
    pub components: Vec<StudioComponentAnalysis>,
    pub connections: Vec<StudioConnectionAnalysis>,
    pub n2: Vec<StudioN2Cell>,
    pub design_variables: Vec<StudioDesignVariableSpec>,
    pub objectives: Vec<StudioObjectiveSpec>,
    pub constraints: Vec<StudioConstraintSpec>,
    pub validation: StudioValidationAnalysis,
    pub warnings: Vec<String>,
}

/// Build a structural analysis payload without mutating or running the model.
pub fn analyze_model_spec(spec: &StudioModelSpec) -> StudioAnalysis {
    let mut warnings = Vec::new();
    let mut components = Vec::with_capacity(spec.blocks.len());

    for block in &spec.blocks {
        match studio_block_io(block) {
            Ok(io) => components.push(StudioComponentAnalysis {
                id: block.id.clone(),
                label: block.label.clone().unwrap_or_else(|| block.id.clone()),
                kind: block.kind,
                role: block.kind.role().as_str().to_string(),
                inputs: io.inputs,
                outputs: io.outputs,
                stateful: io.stateful,
                elements: io.elements,
                x: block.x,
                y: block.y,
            }),
            Err(e) => {
                warnings.push(e.to_string());
                components.push(StudioComponentAnalysis {
                    id: block.id.clone(),
                    label: block.label.clone().unwrap_or_else(|| block.id.clone()),
                    kind: block.kind,
                    role: block.kind.role().as_str().to_string(),
                    inputs: 0,
                    outputs: 0,
                    stateful: false,
                    elements: Vec::new(),
                    x: block.x,
                    y: block.y,
                });
            }
        }
    }

    let connections: Vec<StudioConnectionAnalysis> = spec
        .wires
        .iter()
        .map(|w| StudioConnectionAnalysis {
            from: w.from.clone(),
            from_port: w.from_port,
            to: w.to.clone(),
            to_port: w.to_port,
        })
        .collect();

    let mut n2 = Vec::new();
    for (row, dst) in spec.blocks.iter().enumerate() {
        for (col, src) in spec.blocks.iter().enumerate() {
            let cell_connections: Vec<StudioConnectionAnalysis> = connections
                .iter()
                .filter(|c| c.from == src.id && c.to == dst.id)
                .cloned()
                .collect();
            if !cell_connections.is_empty() {
                n2.push(StudioN2Cell {
                    row,
                    col,
                    connections: cell_connections,
                });
            }
        }
    }

    let validation = match compile_model_spec(spec) {
        Ok(compiled) => {
            let req = requirements_for_studio(&compiled);
            let executive = select(req).map(|p| p.kind.to_string());
            let execution_order = compiled
                .order()
                .iter()
                .filter_map(|&idx| compiled.nodes().get(idx))
                .map(|node| node.id.clone())
                .collect();
            StudioValidationAnalysis {
                ok: true,
                message: None,
                execution_order,
                executive,
            }
        }
        Err(e) => StudioValidationAnalysis {
            ok: false,
            message: Some(e.to_string()),
            execution_order: Vec::new(),
            executive: None,
        },
    };

    StudioAnalysis {
        name: spec.name.clone(),
        components,
        connections,
        n2,
        design_variables: spec.design_variables.clone(),
        objectives: spec.objectives.clone(),
        constraints: spec.constraints.clone(),
        validation,
        warnings,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::studio::starter_model_spec;

    #[test]
    fn analysis_builds_n2_connection_and_exec_metadata() {
        let spec = starter_model_spec();
        let analysis = analyze_model_spec(&spec);
        assert!(analysis.validation.ok);
        assert_eq!(analysis.validation.executive.as_deref(), Some("studio"));
        assert_eq!(analysis.components.len(), 3);
        assert_eq!(analysis.n2.len(), 2);
        assert!(analysis
            .n2
            .iter()
            .any(|cell| cell.connections[0].from == "gain" && cell.connections[0].to == "out"));
        assert_eq!(analysis.design_variables[0].name, "gain.k");
    }
}
