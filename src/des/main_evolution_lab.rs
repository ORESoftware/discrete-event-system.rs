//! CLI demo: GA flavors, GP curve fitting, piecewise models, and toy bio-design.

use crate::des::general::evolution::curve_fitting::predict_holdout;
use crate::des::general::evolution::curve_fitting::ParametricCurveProblem;
use crate::des::general::evolution::genetic_programming::tree_size;
use crate::des::general::evolution::{
    run_curve_fit_ga, run_curve_fit_gp, run_ga_as_des, run_hp_protein_ga, run_ligand_design_ga,
    run_piecewise_ga, synthetic_noisy_sine, synthetic_piecewise_step, CurveConstraints, FitMetric,
    GaFlavor, GaOptions, GpFlavor, GpOptions, GpTreeConfig, ParametricFamily,
};
use crate::des::general::expr::ExprPrinter;

fn ga_opts(pop: usize, gens: usize, flavor: GaFlavor) -> GaOptions {
    GaOptions {
        population_size: pop,
        num_generations: gens,
        tournament_size: Some(3),
        crossover_prob: Some(0.85),
        mutation_prob: Some(0.3),
        elitism: Some(2),
        seed: Some(7),
        flavor: Some(flavor),
        num_islands: Some(4),
        migration_interval: Some(4),
        lambda_offspring: None,
        child_retry_limit: None,
    }
}

pub fn run() {
    println!("# Evolution lab — GA / GP / curve fitting / bio-design\n");

    let full = synthetic_noisy_sine(80, 0.08, 1);
    let (train, holdout) = full.train_holdout_split(0.2);

    println!("== Parametric GA + ridge hybrid (Fourier) ==");
    let fam = ParametricFamily::Fourier { harmonics: 3 };
    let constraints = CurveConstraints {
        ridge: Some(0.01),
        ..CurveConstraints::default()
    };
    let param = run_curve_fit_ga(
        train.clone(),
        fam.clone(),
        constraints.clone(),
        ga_opts(60, 40, GaFlavor::Generational),
    );
    let shape = param.chromosome.decode();
    let hold_mse = predict_holdout(&fam, &param.coefficients, &shape, &holdout);
    println!(
        "  train MSE {:.6}  holdout MSE {:.6}  genes={}",
        param.train_mse,
        hold_mse,
        param.chromosome.genes.len()
    );
    let des_param = run_ga_as_des(
        ParametricCurveProblem {
            data: train.clone(),
            family: fam.clone(),
            constraints: constraints.clone(),
            metric: FitMetric::Mse,
            use_hybrid: true,
        },
        ga_opts(40, 15, GaFlavor::Generational),
    );
    println!(
        "  DES station ticks={}  snapshots={}  best {:.6}",
        des_param.run.ticks,
        des_param.generation_events.len(),
        des_param.ga.best_fitness
    );

    println!("\n== Genetic programming (symbolic) ==");
    let gp = run_curve_fit_gp(
        train.clone(),
        CurveConstraints {
            max_terms: Some(25),
            ..Default::default()
        },
        GpOptions {
            ga: ga_opts(80, 50, GaFlavor::SteadyState),
            tree: GpTreeConfig::default(),
            flavor: Some(GpFlavor::ParsimonyPressure),
            parsimony_coef: Some(0.003),
        },
    );
    let printer = ExprPrinter;
    println!(
        "  fitness {:.6}  nodes={}  expr={}",
        gp.fitness,
        tree_size(&gp.expression),
        printer.print(&gp.expression, 0)
    );

    println!("\n== Piecewise polynomial GA ==");
    let pw = run_piecewise_ga(
        synthetic_piecewise_step(60, 2),
        3,
        2,
        ga_opts(50, 35, GaFlavor::MuPlusLambda),
    );
    println!(
        "  train MSE {:.6}  knots={:?}",
        pw.train_mse,
        pw.model.knots()
    );

    println!("\n== HP lattice protein (toy folding) ==");
    let hp = run_hp_protein_ga(14, ga_opts(40, 60, GaFlavor::Island));
    println!(
        "  energy {:.3}  H count={}",
        hp.energy,
        hp.genome
            .sequence
            .iter()
            .filter(|&&m| m == crate::des::general::evolution::bio_design::HpMonomer::H)
            .count()
    );

    println!("\n== Ligand scaffold design (toy pharmacophore) ==");
    let lig = run_ligand_design_ga(8, ga_opts(50, 45, GaFlavor::Generational));
    let filled = lig.genome.sites.iter().filter(|s| s.is_some()).count();
    println!(
        "  score {:.4}  occupied sites {}/{}",
        lig.score,
        filled,
        lig.genome.sites.len()
    );

    println!("\nDone. See src/des/general/evolution/README.md for API details.");
}
