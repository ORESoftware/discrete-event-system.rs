//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/math-blocks.ts`
//! Rust target: `src/des/general/math_blocks.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/math-blocks.ts",
    "src/des/general/math_blocks.rs",
    &["RUST MIGRATION: target module src/des/general/math_blocks.rs.", "RUST MIGRATION: BlockModelLogger should become a logging trait, MathSignal/MathSample/options/results/node/edge/trace structs become serde structs, and operator unions become enums.", "RUST MIGRATION: MathBlock and each concrete block become structs implementing VisualBlock/MathBlock traits; inheritance chains such as SubtractBlock extends SumBlock become shared helper traits/composition.", "RUST MIGRATION: runMathBlockDiagram, runODEBlockSystem, and runHeat1DBlockGrid are graph-visible transforms and should be PureTransform entry structs returning Result.", "RUST MIGRATION: Record<string, number> signal maps become HashMap<String, f64>, expression callbacks need a parser/trait port, and validation returns Result/ValidationCheck vectors."],
    &["BlockGraphEdge", "BlockGraphNode", "BlockModelLogger", "ComparatorBlock", "ComparatorOp", "ConstantSourceBlock", "DerivativeBlock", "ExpressionBlock", "ExpressionSourceBlock", "FirstOrderFilterBlock", "FunctionSourceBlock", "GainBlock", "Heat1DBlockParams", "Heat1DBlockResult", "Heat1DTraceRow", "IntegratorBlock", "IntegratorMethod", "Laplacian1DBlock", "LogicBlock", "LogicOp", "MATH_IN", "MATH_OUT", "MathBlock", "MathBlockOptions", "MathBlockRunResult", "MathSample", "MathSignal", "ODEBlockSystemParams", "ODEBlockSystemResult", "ODEStateSpec", "ODETraceRow", "ProductBlock", "SaturationBlock", "SinkBlock", "SubtractBlock", "SumBlock", "runHeat1DBlockGrid", "runMathBlockDiagram", "runODEBlockSystem"],
);
