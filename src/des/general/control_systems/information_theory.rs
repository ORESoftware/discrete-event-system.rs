//! Shannon-style information measures for finite distributions and channels.
//!
//! The control-system modules answer "can I drive/see/distinguish this system?"
//! with ranks, reachability, and partition refinement. This module adds the
//! complementary information-theoretic quantities: how many bits of uncertainty
//! remain, how noisy a transition or sensor channel is, and how much a channel
//! says about a hidden state.
#![allow(dead_code)]

use crate::des::general::des_base::preconditions::{Check, Preconditions};
use crate::des::shared::linalg::Matrix;

/// Exact SI value, in joules per kelvin.
pub const BOLTZMANN_CONSTANT_J_PER_K: f64 = 1.380_649e-23;

fn require(check: Check) {
    if let Err(e) = check {
        panic!("{e}");
    }
}

fn validate_same_len(model: &str, a_name: &str, a: &[f64], b_name: &str, b: &[f64]) {
    require(Preconditions::length_eq(model, a_name, a, b.len()));
    require(Preconditions::length_eq(model, b_name, b, a.len()));
}

fn validate_joint(joint: &Matrix) {
    let cls = "InformationTheory";
    require(Preconditions::rectangular_matrix(cls, "joint", joint));
    let mut sum = 0.0;
    for (i, row) in joint.iter().enumerate() {
        for (j, &p) in row.iter().enumerate() {
            require(Preconditions::in_range(
                cls,
                &format!("joint[{i}][{j}]"),
                p,
                0.0,
                1.0,
            ));
            sum += p;
        }
    }
    require(Preconditions::check(
        cls,
        "joint",
        "sum to 1 (within 1e-9)",
        (sum - 1.0).abs() <= 1e-9,
        Some(sum.to_string()),
    ));
}

fn clean_bits(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        0.0
    } else {
        x
    }
}

fn clean_near_zero(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        0.0
    } else {
        x
    }
}

fn validate_energy_levels(model: &str, energies: &[f64]) {
    require(Preconditions::non_empty(model, "energy_levels", energies));
    require(Preconditions::all_finite(model, "energy_levels", energies));
}

fn validate_temperature_and_k(model: &str, temperature: f64, boltzmann_constant: f64) {
    require(Preconditions::positive(model, "temperature", temperature));
    require(Preconditions::positive(
        model,
        "boltzmann_constant",
        boltzmann_constant,
    ));
}

fn validate_channel(channel: &Matrix) {
    let cls = "ChannelCapacity";
    require(Preconditions::rectangular_matrix(cls, "channel", channel));
    for (i, row) in channel.iter().enumerate() {
        require(Preconditions::probability_vector(
            cls,
            &format!("channel[{i}]"),
            row,
            1e-9,
        ));
    }
}

fn entropy_nats(pmf: &[f64]) -> f64 {
    shannon_entropy_bits(pmf) * std::f64::consts::LN_2
}

fn log_sum_exp(values: &[f64]) -> f64 {
    require(Preconditions::non_empty(
        "InformationTheory",
        "values",
        values,
    ));
    for (i, &v) in values.iter().enumerate() {
        require(Preconditions::check(
            "InformationTheory",
            &format!("values[{i}]"),
            "not be NaN",
            !v.is_nan(),
            Some(v.to_string()),
        ));
    }
    let m = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    require(Preconditions::check(
        "InformationTheory",
        "values",
        "contain at least one finite value",
        m.is_finite(),
        None,
    ));
    let sum: f64 = values.iter().map(|&v| (v - m).exp()).sum();
    m + sum.ln()
}

fn relative_entropy_nats(p: &[f64], q: &[f64]) -> f64 {
    validate_same_len("InformationTheory", "p", p, "q", q);
    require(Preconditions::probability_vector(
        "InformationTheory",
        "p",
        p,
        1e-9,
    ));
    require(Preconditions::probability_vector(
        "InformationTheory",
        "q",
        q,
        1e-9,
    ));
    let mut d = 0.0;
    for i in 0..p.len() {
        if p[i] == 0.0 {
            continue;
        }
        if q[i] == 0.0 {
            return f64::INFINITY;
        }
        d += p[i] * (p[i] / q[i]).ln();
    }
    clean_near_zero(d.max(0.0))
}

/// Summary of a finite source distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct EntropySummary {
    /// Shannon entropy H(X), in bits.
    pub entropy_bits: f64,
    /// Maximum entropy for this alphabet, log2(|X|), in bits.
    pub max_entropy_bits: f64,
    /// H(X) / log2(|X|), or 0 for a one-symbol alphabet.
    pub normalized_entropy: f64,
    /// Perplexity / effective alphabet size, 2^H.
    pub effective_symbols: f64,
}

impl EntropySummary {
    pub fn new(pmf: &[f64]) -> Self {
        let entropy_bits = shannon_entropy_bits(pmf);
        let max_entropy_bits = if pmf.len() <= 1 {
            0.0
        } else {
            (pmf.len() as f64).log2()
        };
        let normalized_entropy = if max_entropy_bits > 0.0 {
            entropy_bits / max_entropy_bits
        } else {
            0.0
        };
        EntropySummary {
            entropy_bits,
            max_entropy_bits,
            normalized_entropy,
            effective_symbols: 2.0_f64.powf(entropy_bits),
        }
    }
}

/// Summary for a finite channel Y ~ P(.|X) under an input prior P(X).
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelInformationSummary {
    /// H(X), the prior uncertainty in the hidden/input variable.
    pub input_entropy_bits: f64,
    /// H(Y), the marginal uncertainty in the emitted/output variable.
    pub output_entropy_bits: f64,
    /// H(X,Y), the joint entropy.
    pub joint_entropy_bits: f64,
    /// H(Y|X), channel noise from each input to emitted output.
    pub noise_entropy_bits: f64,
    /// H(X|Y), residual uncertainty about input after observing output.
    pub equivocation_bits: f64,
    /// I(X;Y), the information the output carries about the input.
    pub mutual_information_bits: f64,
    /// I(X;Y) / H(X), or 0 when H(X)=0.
    pub normalized_mutual_information: f64,
}

/// A named bridge from a pre-Shannon idea to the executable model here.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InformationPhysicsDescriptor {
    pub predecessor: &'static str,
    pub historical_model: &'static str,
    pub abstraction: &'static str,
    pub modern_model: &'static str,
}

/// Discovery metadata for the information/thermodynamics model family.
pub fn information_physics_catalog() -> Vec<InformationPhysicsDescriptor> {
    vec![
        InformationPhysicsDescriptor {
            predecessor: "Ludwig Boltzmann",
            historical_model: "S = k_B ln Omega",
            abstraction: "entropy as logarithmic state count",
            modern_model: "microcanonical entropy, Hartley codebook size, density-of-states models",
        },
        InformationPhysicsDescriptor {
            predecessor: "Josiah Willard Gibbs",
            historical_model: "canonical ensemble p_i proportional to exp(-beta E_i)",
            abstraction: "probability distribution over physical states",
            modern_model: "maximum-entropy equilibrium and nonequilibrium free-energy models",
        },
        InformationPhysicsDescriptor {
            predecessor: "Ralph Hartley",
            historical_model: "H_0 = log_b N for N equiprobable symbols",
            abstraction: "information as distinguishable alternatives",
            modern_model: "fixed-length source coding and zero-order Renyi entropy",
        },
        InformationPhysicsDescriptor {
            predecessor: "Claude Shannon",
            historical_model: "H(X), I(X;Y), and noisy channel capacity",
            abstraction: "information independent of meaning or message content",
            modern_model: "coding theorem calculations and Blahut-Arimoto channel optimization",
        },
        InformationPhysicsDescriptor {
            predecessor: "James Clerk Maxwell / Leo Szilard",
            historical_model: "measurement can be traded for thermodynamic work",
            abstraction: "information as a physical resource",
            modern_model: "Landauer erasure, feedback control, and stochastic thermodynamics",
        },
    ]
}

/// Boltzmann's microcanonical entropy: `S = k_B ln(Omega)`.
#[derive(Clone, Debug, PartialEq)]
pub struct BoltzmannEntropySummary {
    /// Number of accessible microstates, `Omega`.
    pub microstates: f64,
    /// Dimensionless entropy, `ln(Omega)`, in nats.
    pub entropy_nats: f64,
    /// Physical entropy, `k_B ln(Omega)`, in J/K.
    pub entropy_j_per_k: f64,
    /// The same count expressed as Hartley/Shannon bits, `log2(Omega)`.
    pub state_count_bits: f64,
}

/// Count accessible microstates and report Boltzmann entropy.
pub fn boltzmann_entropy(microstates: f64, boltzmann_constant: f64) -> BoltzmannEntropySummary {
    let cls = "BoltzmannEntropy";
    require(Preconditions::positive(cls, "microstates", microstates));
    require(Preconditions::positive(
        cls,
        "boltzmann_constant",
        boltzmann_constant,
    ));
    let entropy_nats = microstates.ln();
    BoltzmannEntropySummary {
        microstates,
        entropy_nats,
        entropy_j_per_k: boltzmann_constant * entropy_nats,
        state_count_bits: entropy_nats / std::f64::consts::LN_2,
    }
}

/// Hartley's equiprobable-alternative information measure.
#[derive(Clone, Debug, PartialEq)]
pub struct HartleyInformationSummary {
    pub symbols: usize,
    pub information_bits: f64,
    pub information_nats: f64,
    pub uniform_distribution: Vec<f64>,
}

/// `log2(N)` bits for an alphabet/codebook of `N` equiprobable symbols.
pub fn hartley_information(symbols: usize) -> HartleyInformationSummary {
    require(Preconditions::integer_in_range(
        "HartleyInformation",
        "symbols",
        symbols as f64,
        1.0,
        f64::MAX,
    ));
    let information_bits = (symbols as f64).log2();
    HartleyInformationSummary {
        symbols,
        information_bits,
        information_nats: information_bits * std::f64::consts::LN_2,
        uniform_distribution: vec![1.0 / symbols as f64; symbols],
    }
}

/// Gibbs canonical ensemble summary for finite energy levels.
#[derive(Clone, Debug, PartialEq)]
pub struct GibbsCanonicalSummary {
    pub energy_levels: Vec<f64>,
    pub degeneracies: Vec<f64>,
    pub temperature: f64,
    pub boltzmann_constant: f64,
    pub beta: f64,
    pub probabilities: Vec<f64>,
    pub log_partition_function: f64,
    pub partition_function: f64,
    pub mean_energy: f64,
    pub entropy_nats: f64,
    pub entropy_j_per_k: f64,
    pub helmholtz_free_energy: f64,
}

/// Gibbs canonical distribution with unit degeneracy for every energy level.
pub fn gibbs_canonical_ensemble(
    energy_levels: &[f64],
    temperature: f64,
    boltzmann_constant: f64,
) -> GibbsCanonicalSummary {
    gibbs_canonical_ensemble_with_degeneracy(
        energy_levels,
        &vec![1.0; energy_levels.len()],
        temperature,
        boltzmann_constant,
    )
}

/// Gibbs canonical distribution with optional degeneracy counts per energy.
pub fn gibbs_canonical_ensemble_with_degeneracy(
    energy_levels: &[f64],
    degeneracies: &[f64],
    temperature: f64,
    boltzmann_constant: f64,
) -> GibbsCanonicalSummary {
    let cls = "GibbsCanonicalEnsemble";
    validate_energy_levels(cls, energy_levels);
    require(Preconditions::length_eq(
        cls,
        "degeneracies",
        degeneracies,
        energy_levels.len(),
    ));
    require(Preconditions::arr_non_negative(
        cls,
        "degeneracies",
        degeneracies,
    ));
    require(Preconditions::check(
        cls,
        "degeneracies",
        "contain at least one positive entry",
        degeneracies.iter().any(|&g| g > 0.0),
        None,
    ));
    validate_temperature_and_k(cls, temperature, boltzmann_constant);

    let beta = 1.0 / (boltzmann_constant * temperature);
    let log_weights: Vec<f64> = energy_levels
        .iter()
        .zip(degeneracies.iter())
        .map(|(&e, &g)| {
            if g == 0.0 {
                f64::NEG_INFINITY
            } else {
                g.ln() - beta * e
            }
        })
        .collect();
    let log_partition_function = log_sum_exp(&log_weights);
    let probabilities: Vec<f64> = log_weights
        .iter()
        .map(|&lw| {
            if lw.is_finite() {
                (lw - log_partition_function).exp()
            } else {
                0.0
            }
        })
        .collect();
    let mean_energy: f64 = probabilities
        .iter()
        .zip(energy_levels.iter())
        .map(|(&p, &e)| p * e)
        .sum();
    let entropy_nats = clean_near_zero(log_partition_function + beta * mean_energy);
    let entropy_j_per_k = boltzmann_constant * entropy_nats;
    let helmholtz_free_energy = -log_partition_function / beta;

    GibbsCanonicalSummary {
        energy_levels: energy_levels.to_vec(),
        degeneracies: degeneracies.to_vec(),
        temperature,
        boltzmann_constant,
        beta,
        probabilities,
        log_partition_function,
        partition_function: log_partition_function.exp(),
        mean_energy,
        entropy_nats,
        entropy_j_per_k,
        helmholtz_free_energy,
    }
}

/// Nonequilibrium free-energy decomposition for a finite distribution.
#[derive(Clone, Debug, PartialEq)]
pub struct NonequilibriumFreeEnergySummary {
    pub probabilities: Vec<f64>,
    pub equilibrium_probabilities: Vec<f64>,
    pub mean_energy: f64,
    pub entropy_nats: f64,
    pub entropy_j_per_k: f64,
    pub free_energy: f64,
    pub equilibrium_free_energy: f64,
    pub relative_entropy_to_equilibrium_nats: f64,
    pub excess_free_energy: f64,
}

/// Modern Gibbs view: `F[p] = <E> - T S[p] = F_eq + k_B T D(p || p_eq)`.
pub fn nonequilibrium_free_energy(
    probabilities: &[f64],
    energy_levels: &[f64],
    temperature: f64,
    boltzmann_constant: f64,
) -> NonequilibriumFreeEnergySummary {
    let cls = "NonequilibriumFreeEnergy";
    require(Preconditions::probability_vector(
        cls,
        "probabilities",
        probabilities,
        1e-9,
    ));
    validate_energy_levels(cls, energy_levels);
    require(Preconditions::length_eq(
        cls,
        "energy_levels",
        energy_levels,
        probabilities.len(),
    ));
    validate_temperature_and_k(cls, temperature, boltzmann_constant);

    let equilibrium = gibbs_canonical_ensemble(energy_levels, temperature, boltzmann_constant);
    let mean_energy: f64 = probabilities
        .iter()
        .zip(energy_levels.iter())
        .map(|(&p, &e)| p * e)
        .sum();
    let entropy_nats = entropy_nats(probabilities);
    let entropy_j_per_k = boltzmann_constant * entropy_nats;
    let free_energy = mean_energy - temperature * entropy_j_per_k;
    let relative_entropy_to_equilibrium_nats =
        relative_entropy_nats(probabilities, &equilibrium.probabilities);
    let excess_free_energy =
        boltzmann_constant * temperature * relative_entropy_to_equilibrium_nats;

    NonequilibriumFreeEnergySummary {
        probabilities: probabilities.to_vec(),
        equilibrium_probabilities: equilibrium.probabilities.clone(),
        mean_energy,
        entropy_nats,
        entropy_j_per_k,
        free_energy: clean_near_zero(free_energy),
        equilibrium_free_energy: clean_near_zero(equilibrium.helmholtz_free_energy),
        relative_entropy_to_equilibrium_nats,
        excess_free_energy: clean_near_zero(excess_free_energy),
    }
}

/// Optimized Shannon channel capacity for a finite row-stochastic channel.
#[derive(Clone, Debug, PartialEq)]
pub struct ChannelCapacitySummary {
    pub capacity_bits: f64,
    pub input_distribution: Vec<f64>,
    pub output_distribution: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Blahut-Arimoto channel-capacity optimization.
pub fn channel_capacity_blahut_arimoto_bits(
    channel: &Matrix,
    tol: f64,
    max_iter: usize,
) -> ChannelCapacitySummary {
    let cls = "ChannelCapacity";
    validate_channel(channel);
    require(Preconditions::positive(cls, "tol", tol));
    require(Preconditions::integer_in_range(
        cls,
        "max_iter",
        max_iter as f64,
        1.0,
        f64::MAX,
    ));

    let inputs = channel.len();
    let outputs = channel[0].len();
    let mut prior = vec![1.0 / inputs as f64; inputs];
    let mut previous_capacity = f64::NEG_INFINITY;
    let mut iterations = 0;
    let mut converged = false;

    for iter in 0..max_iter {
        iterations = iter + 1;
        let mut output = vec![0.0; outputs];
        for x in 0..inputs {
            for y in 0..outputs {
                output[y] += prior[x] * channel[x][y];
            }
        }

        let mut d = vec![0.0; inputs];
        for x in 0..inputs {
            for y in 0..outputs {
                let w = channel[x][y];
                if w > 0.0 {
                    d[x] += w * (w / output[y]).ln();
                }
            }
        }

        let z: f64 = prior
            .iter()
            .zip(d.iter())
            .map(|(&q, &dx)| q * dx.exp())
            .sum();
        let mut next = vec![0.0; inputs];
        for x in 0..inputs {
            next[x] = prior[x] * d[x].exp() / z;
        }

        let capacity_bits = channel_information(&next, channel).mutual_information_bits;
        if (capacity_bits - previous_capacity).abs() <= tol {
            prior = next;
            previous_capacity = capacity_bits;
            converged = true;
            break;
        }
        prior = next;
        previous_capacity = capacity_bits;
    }

    let joint = channel_joint_distribution(&prior, channel);
    ChannelCapacitySummary {
        capacity_bits: clean_bits(previous_capacity),
        input_distribution: prior,
        output_distribution: marginal_y(&joint),
        iterations,
        converged,
    }
}

/// Maxwell/Szilard/Landauer work budget for information used as feedback.
#[derive(Clone, Debug, PartialEq)]
pub struct InformationThermodynamicsBudget {
    pub temperature: f64,
    pub information_bits: f64,
    pub information_nats: f64,
    pub max_extractable_work_j: f64,
    pub landauer_erasure_work_j: f64,
    pub reversible_cycle_net_work_bound_j: f64,
    pub entropy_reduction_j_per_k: f64,
}

/// Work and erasure bounds for a Szilard-style engine with `information_bits`.
pub fn szilard_landauer_budget(
    information_bits: f64,
    temperature: f64,
    boltzmann_constant: f64,
) -> InformationThermodynamicsBudget {
    let cls = "SzilardLandauerBudget";
    require(Preconditions::non_negative(
        cls,
        "information_bits",
        information_bits,
    ));
    validate_temperature_and_k(cls, temperature, boltzmann_constant);
    let information_nats = information_bits * std::f64::consts::LN_2;
    let work = boltzmann_constant * temperature * information_nats;
    InformationThermodynamicsBudget {
        temperature,
        information_bits,
        information_nats,
        max_extractable_work_j: work,
        landauer_erasure_work_j: work,
        reversible_cycle_net_work_bound_j: 0.0,
        entropy_reduction_j_per_k: boltzmann_constant * information_nats,
    }
}

/// Feedback-work budget from a measurement joint distribution `P(system, memory)`.
pub fn maxwell_demon_budget_from_joint(
    joint: &Matrix,
    temperature: f64,
    boltzmann_constant: f64,
) -> InformationThermodynamicsBudget {
    validate_temperature_and_k("MaxwellDemonBudget", temperature, boltzmann_constant);
    let information_bits = mutual_information_bits(joint);
    szilard_landauer_budget(information_bits, temperature, boltzmann_constant)
}

/// Jarzynski/stochastic-thermodynamics summary for nonequilibrium work samples.
#[derive(Clone, Debug, PartialEq)]
pub struct StochasticThermodynamicsSummary {
    pub samples: usize,
    pub mean_work_j: f64,
    pub supplied_delta_free_energy_j: f64,
    pub jarzynski_delta_free_energy_j: f64,
    pub dissipated_work_j: f64,
    pub jarzynski_gap_j: f64,
    pub second_law_satisfied: bool,
}

/// Jarzynski estimator: `Delta F = -k_B T ln <exp(-W / k_B T)>`.
pub fn jarzynski_free_energy_estimate(
    work_samples_j: &[f64],
    temperature: f64,
    boltzmann_constant: f64,
) -> f64 {
    let cls = "JarzynskiEstimator";
    require(Preconditions::non_empty(
        cls,
        "work_samples_j",
        work_samples_j,
    ));
    require(Preconditions::all_finite(
        cls,
        "work_samples_j",
        work_samples_j,
    ));
    validate_temperature_and_k(cls, temperature, boltzmann_constant);
    let thermal_energy = boltzmann_constant * temperature;
    let scaled: Vec<f64> = work_samples_j
        .iter()
        .map(|&w| -w / thermal_energy)
        .collect();
    let log_mean_exp = log_sum_exp(&scaled) - (work_samples_j.len() as f64).ln();
    clean_near_zero(-thermal_energy * log_mean_exp)
}

/// Compare observed work to a supplied free-energy difference and Jarzynski estimate.
pub fn stochastic_thermodynamics_summary(
    work_samples_j: &[f64],
    delta_free_energy_j: f64,
    temperature: f64,
    boltzmann_constant: f64,
) -> StochasticThermodynamicsSummary {
    let cls = "StochasticThermodynamics";
    require(Preconditions::finite(
        cls,
        "delta_free_energy_j",
        delta_free_energy_j,
    ));
    let jarzynski_delta_free_energy_j =
        jarzynski_free_energy_estimate(work_samples_j, temperature, boltzmann_constant);
    let mean_work_j = work_samples_j.iter().sum::<f64>() / work_samples_j.len() as f64;
    let dissipated_work_j = mean_work_j - delta_free_energy_j;
    StochasticThermodynamicsSummary {
        samples: work_samples_j.len(),
        mean_work_j,
        supplied_delta_free_energy_j: delta_free_energy_j,
        jarzynski_delta_free_energy_j,
        dissipated_work_j: clean_near_zero(dissipated_work_j),
        jarzynski_gap_j: clean_near_zero(mean_work_j - jarzynski_delta_free_energy_j),
        second_law_satisfied: dissipated_work_j >= -1e-12,
    }
}

impl ChannelInformationSummary {
    pub fn from_joint(joint: &Matrix) -> Self {
        validate_joint(joint);
        let px = marginal_x(joint);
        let py = marginal_y(joint);
        let input_entropy_bits = shannon_entropy_bits(&px);
        let output_entropy_bits = shannon_entropy_bits(&py);
        let joint_entropy_bits = joint_entropy_bits(joint);
        let noise_entropy_bits = (joint_entropy_bits - input_entropy_bits).max(0.0);
        let equivocation_bits = (joint_entropy_bits - output_entropy_bits).max(0.0);
        let mutual_information_bits =
            (input_entropy_bits + output_entropy_bits - joint_entropy_bits).max(0.0);
        let normalized_mutual_information = if input_entropy_bits > 0.0 {
            mutual_information_bits / input_entropy_bits
        } else {
            0.0
        };
        ChannelInformationSummary {
            input_entropy_bits,
            output_entropy_bits,
            joint_entropy_bits,
            noise_entropy_bits,
            equivocation_bits,
            mutual_information_bits,
            normalized_mutual_information,
        }
    }
}

/// Self-information of an event with probability p: -log2(p).
///
/// Impossible events (p=0) have infinite self-information.
pub fn self_information_bits(probability: f64) -> f64 {
    require(Preconditions::in_range(
        "InformationTheory",
        "probability",
        probability,
        0.0,
        1.0,
    ));
    if probability == 0.0 {
        f64::INFINITY
    } else {
        -probability.log2()
    }
}

/// Shannon entropy H(X) = -sum_i p_i log2(p_i), in bits.
pub fn shannon_entropy_bits(pmf: &[f64]) -> f64 {
    require(Preconditions::probability_vector(
        "InformationTheory",
        "pmf",
        pmf,
        1e-9,
    ));
    clean_bits(
        pmf.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum(),
    )
}

/// One-shot entropy summary for a source distribution.
pub fn entropy_summary(pmf: &[f64]) -> EntropySummary {
    EntropySummary::new(pmf)
}

/// Cross entropy H(P,Q) = -sum_i P(i) log2(Q(i)), in bits.
pub fn cross_entropy_bits(p: &[f64], q: &[f64]) -> f64 {
    let cls = "InformationTheory";
    validate_same_len(cls, "p", p, "q", q);
    require(Preconditions::probability_vector(cls, "p", p, 1e-9));
    require(Preconditions::probability_vector(cls, "q", q, 1e-9));
    let mut h = 0.0;
    for i in 0..p.len() {
        if p[i] == 0.0 {
            continue;
        }
        if q[i] == 0.0 {
            return f64::INFINITY;
        }
        h -= p[i] * q[i].log2();
    }
    h
}

/// Kullback-Leibler divergence D_KL(P || Q), in bits.
pub fn kl_divergence_bits(p: &[f64], q: &[f64]) -> f64 {
    let ce = cross_entropy_bits(p, q);
    if ce.is_infinite() {
        ce
    } else {
        clean_bits((ce - shannon_entropy_bits(p)).max(0.0))
    }
}

/// Jensen-Shannon divergence, a symmetric finite divergence in bits.
pub fn jensen_shannon_divergence_bits(p: &[f64], q: &[f64]) -> f64 {
    validate_same_len("InformationTheory", "p", p, "q", q);
    let mix: Vec<f64> = p
        .iter()
        .zip(q.iter())
        .map(|(&a, &b)| 0.5 * (a + b))
        .collect();
    0.5 * kl_divergence_bits(p, &mix) + 0.5 * kl_divergence_bits(q, &mix)
}

/// Joint entropy H(X,Y), in bits, for a joint pmf matrix P(x,y).
pub fn joint_entropy_bits(joint: &Matrix) -> f64 {
    validate_joint(joint);
    clean_bits(
        joint
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.log2())
            .sum(),
    )
}

/// Marginal distribution P(X) from a joint pmf P(X,Y).
pub fn marginal_x(joint: &Matrix) -> Vec<f64> {
    validate_joint(joint);
    joint.iter().map(|row| row.iter().sum()).collect()
}

/// Marginal distribution P(Y) from a joint pmf P(X,Y).
pub fn marginal_y(joint: &Matrix) -> Vec<f64> {
    validate_joint(joint);
    let cols = joint[0].len();
    let mut out = vec![0.0; cols];
    for row in joint {
        for j in 0..cols {
            out[j] += row[j];
        }
    }
    out
}

/// Conditional entropy H(Y|X), in bits, for a joint pmf P(X,Y).
pub fn conditional_entropy_y_given_x_bits(joint: &Matrix) -> f64 {
    let hxy = joint_entropy_bits(joint);
    let hx = shannon_entropy_bits(&marginal_x(joint));
    clean_bits((hxy - hx).max(0.0))
}

/// Conditional entropy H(X|Y), in bits, for a joint pmf P(X,Y).
pub fn conditional_entropy_x_given_y_bits(joint: &Matrix) -> f64 {
    let hxy = joint_entropy_bits(joint);
    let hy = shannon_entropy_bits(&marginal_y(joint));
    clean_bits((hxy - hy).max(0.0))
}

/// Mutual information I(X;Y), in bits, for a joint pmf P(X,Y).
pub fn mutual_information_bits(joint: &Matrix) -> f64 {
    let hx = shannon_entropy_bits(&marginal_x(joint));
    let hy = shannon_entropy_bits(&marginal_y(joint));
    let hxy = joint_entropy_bits(joint);
    clean_bits((hx + hy - hxy).max(0.0))
}

/// Build the joint pmf P(X,Y) = P(X) P(Y|X) for a discrete channel.
pub fn channel_joint_distribution(prior: &[f64], channel: &Matrix) -> Matrix {
    let cls = "InformationTheory";
    require(Preconditions::probability_vector(cls, "prior", prior, 1e-9));
    require(Preconditions::rectangular_matrix(cls, "channel", channel));
    require(Preconditions::length_eq(
        cls,
        "channel",
        channel,
        prior.len(),
    ));
    for (i, row) in channel.iter().enumerate() {
        require(Preconditions::probability_vector(
            cls,
            &format!("channel[{i}]"),
            row,
            1e-9,
        ));
    }

    channel
        .iter()
        .enumerate()
        .map(|(i, row)| row.iter().map(|&p| prior[i] * p).collect())
        .collect()
}

/// Full channel information summary for P(Y|X) under prior P(X).
pub fn channel_information(prior: &[f64], channel: &Matrix) -> ChannelInformationSummary {
    let joint = channel_joint_distribution(prior, channel);
    ChannelInformationSummary::from_joint(&joint)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn entropy_uses_bits_and_effective_symbols() {
        let h = entropy_summary(&[0.25, 0.25, 0.25, 0.25]);
        assert!(close(h.entropy_bits, 2.0));
        assert!(close(h.max_entropy_bits, 2.0));
        assert!(close(h.normalized_entropy, 1.0));
        assert!(close(h.effective_symbols, 4.0));
        assert!(close(shannon_entropy_bits(&[1.0, 0.0]), 0.0));
        assert!(self_information_bits(0.0).is_infinite());
        assert!(close(self_information_bits(0.5), 1.0));
    }

    #[test]
    fn divergence_and_channel_information_are_consistent() {
        assert!(close(kl_divergence_bits(&[0.5, 0.5], &[0.5, 0.5]), 0.0));
        assert!(jensen_shannon_divergence_bits(&[1.0, 0.0], &[0.0, 1.0]) > 0.99);

        let perfect = channel_information(&[0.5, 0.5], &vec![vec![1.0, 0.0], vec![0.0, 1.0]]);
        assert!(close(perfect.input_entropy_bits, 1.0));
        assert!(close(perfect.mutual_information_bits, 1.0));
        assert!(close(perfect.equivocation_bits, 0.0));
        assert!(close(perfect.normalized_mutual_information, 1.0));

        let aliased = channel_information(&[0.5, 0.5], &vec![vec![0.5, 0.5], vec![0.5, 0.5]]);
        assert!(close(aliased.input_entropy_bits, 1.0));
        assert!(close(aliased.mutual_information_bits, 0.0));
        assert!(close(aliased.equivocation_bits, 1.0));
    }

    #[test]
    fn catalog_covers_pre_shannon_and_modern_models() {
        let catalog = information_physics_catalog();
        assert_eq!(catalog.len(), 5);
        assert!(catalog.iter().any(|d| d.predecessor.contains("Boltzmann")));
        assert!(catalog.iter().any(|d| d.predecessor.contains("Gibbs")));
        assert!(catalog.iter().any(|d| d.predecessor.contains("Hartley")));
        assert!(catalog.iter().any(|d| d.predecessor.contains("Shannon")));
        assert!(catalog.iter().any(|d| d.predecessor.contains("Szilard")));
    }

    #[test]
    fn boltzmann_hartley_and_uniform_shannon_entropy_align() {
        let boltzmann = boltzmann_entropy(4.0, 1.0);
        assert!(close(boltzmann.entropy_nats, 4.0_f64.ln()));
        assert!(close(boltzmann.state_count_bits, 2.0));

        let hartley = hartley_information(4);
        assert!(close(hartley.information_bits, 2.0));
        assert!(close(
            shannon_entropy_bits(&hartley.uniform_distribution),
            hartley.information_bits
        ));
    }

    #[test]
    fn gibbs_and_nonequilibrium_free_energy_match_at_equilibrium() {
        let energies = vec![0.0, 1.0];
        let gibbs = gibbs_canonical_ensemble(&energies, 1.0, 1.0);
        let z = 1.0 + (-1.0_f64).exp();
        assert!(close(gibbs.partition_function, z));
        assert!(close(gibbs.probabilities[0], 1.0 / z));
        assert!(close(gibbs.probabilities[1], (-1.0_f64).exp() / z));

        let eq = nonequilibrium_free_energy(&gibbs.probabilities, &energies, 1.0, 1.0);
        assert!(close(eq.free_energy, gibbs.helmholtz_free_energy));
        assert!(close(eq.excess_free_energy, 0.0));

        let pinned = nonequilibrium_free_energy(&[1.0, 0.0], &energies, 1.0, 1.0);
        assert!(pinned.excess_free_energy > 0.0);
        assert!(close(
            pinned.free_energy - pinned.equilibrium_free_energy,
            pinned.excess_free_energy
        ));
    }

    #[test]
    fn channel_capacity_finds_binary_symmetric_channel_capacity() {
        let p = 0.1;
        let channel = vec![vec![1.0 - p, p], vec![p, 1.0 - p]];
        let capacity = channel_capacity_blahut_arimoto_bits(&channel, 1e-12, 200);
        let expected = 1.0 - shannon_entropy_bits(&[1.0 - p, p]);
        assert!(capacity.converged);
        assert!(close(capacity.capacity_bits, expected));
        assert!(close(capacity.input_distribution[0], 0.5));
        assert!(close(capacity.input_distribution[1], 0.5));
    }

    #[test]
    fn szilard_landauer_and_maxwell_budget_use_information_as_work() {
        let budget = szilard_landauer_budget(1.0, 2.0, 1.0);
        assert!(close(
            budget.max_extractable_work_j,
            2.0 * std::f64::consts::LN_2
        ));
        assert!(close(
            budget.max_extractable_work_j,
            budget.landauer_erasure_work_j
        ));

        let perfect_measurement = vec![vec![0.5, 0.0], vec![0.0, 0.5]];
        let demon = maxwell_demon_budget_from_joint(&perfect_measurement, 2.0, 1.0);
        assert!(close(demon.information_bits, 1.0));
        assert!(close(
            demon.max_extractable_work_j,
            budget.max_extractable_work_j
        ));
    }

    #[test]
    fn jarzynski_estimator_reports_reversible_and_dissipative_work() {
        let reversible = jarzynski_free_energy_estimate(&[2.0, 2.0, 2.0], 1.0, 1.0);
        assert!(close(reversible, 2.0));

        let summary = stochastic_thermodynamics_summary(&[1.0, 2.0, 3.0], 1.5, 1.0, 1.0);
        assert_eq!(summary.samples, 3);
        assert!(close(summary.mean_work_j, 2.0));
        assert!(close(summary.dissipated_work_j, 0.5));
        assert!(summary.jarzynski_delta_free_energy_j <= summary.mean_work_j);
        assert!(summary.second_law_satisfied);
    }
}
