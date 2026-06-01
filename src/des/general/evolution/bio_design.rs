//! Toy bio-design problems for evolutionary search.
//!
//! These are **not** full physics models; they are differentiable-free fitness
//! landscapes inspired by protein lattice folding and ligand scaffold design,
//! suitable for demonstrating GA flavors inside the DES engine.

use crate::des::general::evolution::ga_core::{
    run_ga, FitnessEvaluator, GaOptions, GaResult, GeneticOperators, PopulationInitializer,
};
use crate::des::shared::capabilities::RandomSource;

// =============================================================================
// HP lattice protein (2-D)
// =============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HpMonomer {
    H,
    P,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HpDirection {
    N,
    E,
    S,
    W,
}

impl HpDirection {
    fn delta(self) -> (i32, i32) {
        match self {
            HpDirection::N => (0, 1),
            HpDirection::E => (1, 0),
            HpDirection::S => (0, -1),
            HpDirection::W => (-1, 0),
        }
    }

    fn all() -> [HpDirection; 4] {
        [
            HpDirection::N,
            HpDirection::E,
            HpDirection::S,
            HpDirection::W,
        ]
    }
}

/// Genome: monomer sequence + self-avoiding walk on the square lattice.
#[derive(Clone, Debug)]
pub struct HpGenome {
    pub sequence: Vec<HpMonomer>,
    pub moves: Vec<HpDirection>,
}

pub fn hp_embed(genome: &HpGenome) -> Vec<(i32, i32)> {
    let mut pos = vec![(0, 0)];
    let mut x = 0i32;
    let mut y = 0i32;
    for &mv in &genome.moves {
        let (dx, dy) = mv.delta();
        x += dx;
        y += dy;
        pos.push((x, y));
    }
    pos
}

/// HP model energy: reward H–H contacts, penalize clashes (non-self-avoiding).
pub fn hp_energy(genome: &HpGenome) -> f64 {
    let pos = hp_embed(genome);
    if pos.len() != genome.sequence.len() {
        return 1e6;
    }
    let mut seen = std::collections::HashSet::new();
    for p in &pos {
        if !seen.insert(*p) {
            return 1e6;
        }
    }
    let n = genome.sequence.len();
    let mut hh = 0i32;
    for i in 0..n {
        for j in (i + 2)..n {
            let dx = (pos[i].0 - pos[j].0).abs();
            let dy = (pos[i].1 - pos[j].1).abs();
            if dx + dy == 1
                && genome.sequence[i] == HpMonomer::H
                && genome.sequence[j] == HpMonomer::H
            {
                hh += 1;
            }
        }
    }
    -(hh as f64)
}

pub struct HpProteinProblem {
    pub length: usize,
}

impl PopulationInitializer<HpGenome> for HpProteinProblem {
    fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<HpGenome> {
        (0..size)
            .map(|_| {
                let sequence = (0..self.length)
                    .map(|_| {
                        if rng.next_float() < 0.5 {
                            HpMonomer::H
                        } else {
                            HpMonomer::P
                        }
                    })
                    .collect();
                let moves = (0..self.length.saturating_sub(1))
                    .map(|_| HpDirection::all()[(rng.next_float() * 4.0).floor() as usize % 4])
                    .collect();
                HpGenome { sequence, moves }
            })
            .collect()
    }
}

impl FitnessEvaluator<HpGenome> for HpProteinProblem {
    fn evaluate(&self, individual: &HpGenome) -> f64 {
        hp_energy(individual)
    }
}

impl GeneticOperators<HpGenome> for HpProteinProblem {
    fn crossover(&self, a: &HpGenome, b: &HpGenome, rng: &mut dyn RandomSource) -> HpGenome {
        let cut = (rng.next_float() * a.sequence.len() as f64).floor() as usize % a.sequence.len();
        let mut sequence = a.sequence[..cut].to_vec();
        sequence.extend_from_slice(&b.sequence[cut..]);
        let mut moves = a.moves.clone();
        if cut > 0 && cut - 1 < moves.len() && cut < b.moves.len() {
            moves[cut - 1] = b.moves[cut - 1];
        }
        HpGenome { sequence, moves }
    }

    fn mutate(&self, mut child: HpGenome, rng: &mut dyn RandomSource) -> HpGenome {
        if rng.next_float() < 0.5 {
            let i = (rng.next_float() * child.sequence.len() as f64).floor() as usize
                % child.sequence.len();
            child.sequence[i] = if child.sequence[i] == HpMonomer::H {
                HpMonomer::P
            } else {
                HpMonomer::H
            };
        } else if !child.moves.is_empty() {
            let i =
                (rng.next_float() * child.moves.len() as f64).floor() as usize % child.moves.len();
            child.moves[i] = HpDirection::all()[(rng.next_float() * 4.0).floor() as usize % 4];
        }
        child
    }

    fn accept_child(&self, child: &HpGenome) -> bool {
        hp_energy(child) < 1e5
    }
}

#[derive(Clone, Debug)]
pub struct HpGaResult {
    pub genome: HpGenome,
    pub energy: f64,
    pub ga: GaResult<HpGenome>,
}

pub fn run_hp_protein_ga(length: usize, ga_opts: GaOptions) -> HpGaResult {
    let ga = run_ga(HpProteinProblem { length }, ga_opts, None);
    HpGaResult {
        energy: hp_energy(&ga.best),
        genome: ga.best.clone(),
        ga,
    }
}

// =============================================================================
// Ligand scaffold (toy pharmacophore)
// =============================================================================

/// Functional groups available on a virtual scaffold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LigandGroup {
    Methyl,
    Hydroxyl,
    Amine,
    Carboxyl,
    Phenyl,
    Halogen,
}

pub struct LigandPalette;

impl LigandPalette {
    pub const GROUPS: [LigandGroup; 6] = [
        LigandGroup::Methyl,
        LigandGroup::Hydroxyl,
        LigandGroup::Amine,
        LigandGroup::Carboxyl,
        LigandGroup::Phenyl,
        LigandGroup::Halogen,
    ];

    pub fn mw(g: LigandGroup) -> f64 {
        match g {
            LigandGroup::Methyl => 15.0,
            LigandGroup::Hydroxyl => 17.0,
            LigandGroup::Amine => 16.0,
            LigandGroup::Carboxyl => 45.0,
            LigandGroup::Phenyl => 77.0,
            LigandGroup::Halogen => 35.0,
        }
    }

    pub fn logp(g: LigandGroup) -> f64 {
        match g {
            LigandGroup::Methyl => 0.5,
            LigandGroup::Hydroxyl => -0.7,
            LigandGroup::Amine => -0.4,
            LigandGroup::Carboxyl => -0.3,
            LigandGroup::Phenyl => 2.0,
            LigandGroup::Halogen => 1.2,
        }
    }
}

/// Bitmask over attachment sites on a fixed scaffold.
#[derive(Clone, Debug)]
pub struct LigandGenome {
    pub sites: Vec<Option<LigandGroup>>,
}

/// Precomputed receptor pocket: grid of attraction weights.
#[derive(Clone, Debug)]
pub struct ReceptorPocket {
    pub targets: Vec<(f64, f64, f64)>,
}

impl ReceptorPocket {
    pub fn demo() -> Self {
        ReceptorPocket {
            targets: vec![(0.0, 0.0, 1.0), (1.2, 0.3, 0.8), (-0.5, 1.0, 0.6)],
        }
    }
}

fn group_pharmacophore(g: LigandGroup) -> (f64, f64, f64) {
    match g {
        LigandGroup::Methyl => (0.2, 0.1, 0.0),
        LigandGroup::Hydroxyl => (-0.5, 0.8, 0.9),
        LigandGroup::Amine => (-0.6, 0.7, 0.3),
        LigandGroup::Carboxyl => (-0.9, 0.9, 0.2),
        LigandGroup::Phenyl => (0.8, 0.2, 0.1),
        LigandGroup::Halogen => (0.6, -0.3, 0.4),
    }
}

pub fn ligand_score(genome: &LigandGenome, pocket: &ReceptorPocket) -> f64 {
    let mut mw = 120.0;
    let mut logp = 1.0;
    let mut vec = (0.0, 0.0, 0.0);
    let mut filled = 0usize;
    for (i, slot) in genome.sites.iter().enumerate() {
        if let Some(g) = slot {
            mw += LigandPalette::mw(*g);
            logp += LigandPalette::logp(*g);
            let p = group_pharmacophore(*g);
            vec.0 += p.0;
            vec.1 += p.1;
            vec.2 += p.2;
            filled += 1;
        }
        let _ = i;
    }
    if filled == 0 {
        return 1e4;
    }
    vec.0 /= filled as f64;
    vec.1 /= filled as f64;
    vec.2 /= filled as f64;
    let mut dock = 0.0;
    for t in &pocket.targets {
        let d = ((vec.0 - t.0).powi(2) + (vec.1 - t.1).powi(2) + (vec.2 - t.2).powi(2)).sqrt();
        dock -= (-d).exp();
    }
    let mut penalty = 0.0;
    if mw > 500.0 {
        penalty += (mw - 500.0) * 0.05;
    }
    if logp > 5.0 || logp < -1.0 {
        penalty += 5.0;
    }
    if filled < 2 {
        penalty += 3.0;
    }
    -dock + penalty
}

pub struct LigandDesignProblem {
    pub num_sites: usize,
    pub pocket: ReceptorPocket,
}

impl PopulationInitializer<LigandGenome> for LigandDesignProblem {
    fn initial_population(&self, size: usize, rng: &mut dyn RandomSource) -> Vec<LigandGenome> {
        (0..size)
            .map(|_| LigandGenome {
                sites: (0..self.num_sites)
                    .map(|_| {
                        if rng.next_float() < 0.55 {
                            Some(
                                LigandPalette::GROUPS
                                    [(rng.next_float() * 6.0).floor() as usize % 6],
                            )
                        } else {
                            None
                        }
                    })
                    .collect(),
            })
            .collect()
    }
}

impl FitnessEvaluator<LigandGenome> for LigandDesignProblem {
    fn evaluate(&self, individual: &LigandGenome) -> f64 {
        ligand_score(individual, &self.pocket)
    }
}

impl GeneticOperators<LigandGenome> for LigandDesignProblem {
    fn crossover(
        &self,
        a: &LigandGenome,
        b: &LigandGenome,
        rng: &mut dyn RandomSource,
    ) -> LigandGenome {
        LigandGenome {
            sites: a
                .sites
                .iter()
                .zip(&b.sites)
                .map(|(&sa, &sb)| if rng.next_float() < 0.5 { sa } else { sb })
                .collect(),
        }
    }

    fn mutate(&self, mut child: LigandGenome, rng: &mut dyn RandomSource) -> LigandGenome {
        let i = (rng.next_float() * child.sites.len() as f64).floor() as usize % child.sites.len();
        if rng.next_float() < 0.3 {
            child.sites[i] = None;
        } else {
            child.sites[i] =
                Some(LigandPalette::GROUPS[(rng.next_float() * 6.0).floor() as usize % 6]);
        }
        child
    }
}

#[derive(Clone, Debug)]
pub struct LigandGaResult {
    pub genome: LigandGenome,
    pub score: f64,
    pub ga: GaResult<LigandGenome>,
}

pub fn run_ligand_design_ga(num_sites: usize, ga_opts: GaOptions) -> LigandGaResult {
    let ga = run_ga(
        LigandDesignProblem {
            num_sites,
            pocket: ReceptorPocket::demo(),
        },
        ga_opts,
        None,
    );
    LigandGaResult {
        score: ligand_score(&ga.best, &ReceptorPocket::demo()),
        genome: ga.best.clone(),
        ga,
    }
}
