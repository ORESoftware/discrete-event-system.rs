//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/math-equation-input.ts`
//! Rust target: `src/des/general/math_equation_input.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/math-equation-input.ts",
    "src/des/general/math_equation_input.rs",
    &["RUST MIGRATION: target module src/des/general/math_equation_input.rs.", "RUST MIGRATION: EquationInputFormat, EquationProblemKind, and normalizeMathEquationProblem's union return become enums; params/network/result structs become serde structs.", "RUST MIGRATION: runMathEquationProblem is a graph-visible input transform and should be a PureTransform entry struct; latexToExpression can remain a free parser helper.", "RUST MIGRATION: Record<string, unknown/number> maps become serde_json::Map/HashMap<String, f64>, and Attrs/ExprToken become private structs/enums.", "RUST MIGRATION: Manual JSON/XML/LaTeX parsing should return Result with typed parse errors; consider quick-xml and a small expression tokenizer module in Rust."],
    &["EquationInputFormat", "EquationProblemKind", "MathEquationInputParams", "MathEquationNetwork", "MathEquationResult", "latexToExpression", "normalizeMathEquationProblem", "runMathEquationProblem"],
);
