//! Genetic programming over [`Expr`] trees.
//!
//! Subtree crossover and point mutation; fitness is supplied by the caller
//! (typically weighted MSE + parsimony on curve data).

use crate::des::general::evolution::curve_fitting::{
    gp_fitness, CurveConstraints, CurveDataset, FitMetric,
};
use crate::des::general::evolution::ga_core::{
    run_ga, FitnessEvaluator, GaOptions, GaResult, GeneticOperators, PopulationInitializer,
};
use crate::des::general::expr::{BinOp, Expr, FuncName};
use crate::des::shared::capabilities::RandomSource;

/// GP population-update style.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpFlavor {
    Standard,
    /// Extra complexity penalty beyond user constraints.
    ParsimonyPressure,
}

#[derive(Clone, Debug)]
pub struct GpTreeConfig {
    pub var_names: Vec<String>,
    pub functions: Vec<FuncName>,
    pub max_depth: usize,
    pub max_nodes: usize,
}

impl Default for GpTreeConfig {
    fn default() -> Self {
        GpTreeConfig {
            var_names: vec!["x".to_string()],
            functions: vec![
                FuncName::Sin,
                FuncName::Cos,
                FuncName::Exp,
                FuncName::Log,
                FuncName::Sqrt,
            ],
            max_depth: 5,
            max_nodes: 40,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GpOptions {
    pub ga: GaOptions,
    pub tree: GpTreeConfig,
    pub flavor: Option<GpFlavor>,
    pub parsimony_coef: Option<f64>,
}

#[derive(Clone, Debug)]
pub struct GpResult {
    pub expression: Expr,
    pub fitness: f64,
    pub ga: GaResult<Expr>,
}

pub fn tree_size(e: &Expr) -> usize {
    1 + match e {
        Expr::Neg(a) => tree_size(a),
        Expr::Func { arg, .. } => tree_size(arg),
        Expr::Bin { left, right, .. } => tree_size(left) + tree_size(right),
        _ => 0,
    }
}

fn clone_expr(e: &Expr) -> Expr {
    match e {
        Expr::Num(v) => Expr::Num(*v),
        Expr::Var(s) => Expr::Var(s.clone()),
        Expr::Neg(a) => Expr::Neg(Box::new(clone_expr(a))),
        Expr::Func { name, arg } => Expr::Func {
            name: *name,
            arg: Box::new(clone_expr(arg)),
        },
        Expr::Bin { op, left, right } => Expr::Bin {
            op: *op,
            left: Box::new(clone_expr(left)),
            right: Box::new(clone_expr(right)),
        },
    }
}

fn random_terminal(cfg: &GpTreeConfig, rng: &mut dyn RandomSource) -> Expr {
    if rng.next_float() < 0.4 {
        let v = &cfg.var_names[(rng.next_float() * cfg.var_names.len() as f64).floor() as usize
            % cfg.var_names.len()];
        Expr::Var(v.clone())
    } else {
        Expr::Num((rng.next_float() * 4.0 - 2.0).round() * 0.25)
    }
}

pub fn random_tree(cfg: &GpTreeConfig, rng: &mut dyn RandomSource, depth: usize) -> Expr {
    if depth >= cfg.max_depth || rng.next_float() < 0.25 {
        return random_terminal(cfg, rng);
    }
    if rng.next_float() < 0.5 {
        let fname = cfg.functions[(rng.next_float() * cfg.functions.len() as f64).floor() as usize
            % cfg.functions.len()];
        return Expr::Func {
            name: fname,
            arg: Box::new(random_tree(cfg, rng, depth + 1)),
        };
    }
    let op = if rng.next_float() < 0.5 {
        BinOp::Add
    } else {
        BinOp::Mul
    };
    Expr::Bin {
        op,
        left: Box::new(random_tree(cfg, rng, depth + 1)),
        right: Box::new(random_tree(cfg, rng, depth + 1)),
    }
}

fn node_list(e: &Expr) -> Vec<Expr> {
    let mut out = vec![clone_expr(e)];
    match e {
        Expr::Neg(a) => out.extend(node_list(a)),
        Expr::Func { arg, .. } => out.extend(node_list(arg)),
        Expr::Bin { left, right, .. } => {
            out.extend(node_list(left));
            out.extend(node_list(right));
        }
        _ => {}
    }
    out
}

fn pick_subtree_index(e: &Expr, rng: &mut dyn RandomSource) -> usize {
    let nodes = node_list(e);
    (rng.next_float() * nodes.len() as f64).floor() as usize % nodes.len()
}

pub fn replace_subtree(root: &Expr, index: usize, donor: &Expr) -> Expr {
    let mut cur = 0;
    fn rec(e: &Expr, idx: &mut usize, target: usize, donor: &Expr) -> Expr {
        if *idx == target {
            return clone_expr(donor);
        }
        *idx += 1;
        match e {
            Expr::Num(v) => Expr::Num(*v),
            Expr::Var(s) => Expr::Var(s.clone()),
            Expr::Neg(a) => Expr::Neg(Box::new(rec(a, idx, target, donor))),
            Expr::Func { name, arg } => Expr::Func {
                name: *name,
                arg: Box::new(rec(arg, idx, target, donor)),
            },
            Expr::Bin { op, left, right } => Expr::Bin {
                op: *op,
                left: Box::new(rec(left, idx, target, donor)),
                right: Box::new(rec(right, idx, target, donor)),
            },
        }
    }
    rec(root, &mut cur, index, donor)
}

pub fn subtree_crossover(a: &Expr, b: &Expr, rng: &mut dyn RandomSource) -> Expr {
    let ia = pick_subtree_index(a, rng);
    let ib = pick_subtree_index(b, rng);
    let donor = node_list(b)[ib].clone();
    replace_subtree(a, ia, &donor)
}

pub fn point_mutate(e: &Expr, cfg: &GpTreeConfig, rng: &mut dyn RandomSource) -> Expr {
    if rng.next_float() < 0.2 {
        return random_tree(cfg, rng, 0);
    }
    let i = pick_subtree_index(e, rng);
    let fresh = random_tree(cfg, rng, 1);
    replace_subtree(e, i, &fresh)
}

/// Curve-fitting GP problem.
pub struct GpCurveProblem {
    pub data: CurveDataset,
    pub constraints: CurveConstraints,
    pub metric: FitMetric,
    pub tree: GpTreeConfig,
    pub flavor: GpFlavor,
    pub parsimony_coef: f64,
}

impl PopulationInitializer<Expr> for GpCurveProblem {
    fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<Expr> {
        (0..size).map(|_| random_tree(&self.tree, rng, 0)).collect()
    }
}

impl FitnessEvaluator<Expr> for GpCurveProblem {
    fn evaluate(&self, individual: &Expr) -> f64 {
        let mut f = gp_fitness(individual, &self.data, &self.constraints, self.metric);
        if self.flavor == GpFlavor::ParsimonyPressure {
            f += self.parsimony_coef * tree_size(individual) as f64;
        }
        f
    }
}

impl GeneticOperators<Expr> for GpCurveProblem {
    fn crossover(&self, a: &Expr, b: &Expr, rng: &mut dyn RandomSource) -> Expr {
        subtree_crossover(a, b, rng)
    }
    fn mutate(&self, child: Expr, rng: &mut dyn RandomSource) -> Expr {
        point_mutate(&child, &self.tree, rng)
    }
    fn accept_child(&self, child: &Expr) -> bool {
        tree_size(child) <= self.tree.max_nodes
    }
}

pub fn run_gp(problem: GpCurveProblem, opts: GpOptions) -> GpResult {
    let ga_result = run_ga(problem, opts.ga, None);
    let expr = ga_result.best.clone();
    GpResult {
        fitness: ga_result.best_fitness,
        expression: expr,
        ga: ga_result,
    }
}
