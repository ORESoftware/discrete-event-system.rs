//! File-for-file migration scaffold generated from the TypeScript source.
//! TypeScript source: `src/des/general/expr.ts`
//! Rust target: `src/des/general/expr.rs`

#![allow(dead_code)]

use crate::migration::MigrationFile;

pub const MIGRATION: MigrationFile = MigrationFile::scaffolded(
    "src/des/general/expr.ts",
    "src/des/general/expr.rs",
    &[
        "RUST MIGRATION:",
        "- Target: src/des/general/expr.rs",
        "- Expr is a direct Rust enum with variants Num, Var, Bin, Neg, and Func.",
        "- Parser and lexer helpers can remain private module functions. Parse/evaluate",
        "- FUNC_IMPL becomes a match on FuncName. FuncName itself should be a Rust enum",
        "- toFunction is JavaScript-specific closure packaging; in Rust prefer an",
        "- numericalDerivative/numericalGradient may stay free functions. If used in a",
        "- Parse human strings: \"x^2 * sin(x) + exp(-x)\"",
        "- Construct AST programmatically: mul(num(2), v('x'))",
        "- Numerically evaluate over an environment: eval(ast, {x: 2})",
        "- Symbolically differentiate w.r.t. any variable: diff(ast, 'x')",
        "- Algebraic simplification (constant folding, x*0=0, x*1=x, …)",
        "- Pretty-print back to a string: stringify(ast)",
        "- Convert to JS function: toFunction(ast, ['x']) → (x) => number",
    ],
    &[
        "BinNode",
        "Env",
        "Expr",
        "FuncName",
        "FuncNode",
        "NumNode",
        "ONE",
        "UnaryNeg",
        "VarNode",
        "ZERO",
        "add",
        "diff",
        "div",
        "evaluate",
        "fn",
        "mul",
        "neg",
        "num",
        "numericalDerivative",
        "numericalGradient",
        "parse",
        "pow",
        "richardsonDerivative",
        "simplify",
        "stringify",
        "sub",
        "toFunction",
        "v",
    ],
);
