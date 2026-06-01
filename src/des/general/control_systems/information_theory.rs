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
    pmf.iter()
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
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
        (ce - shannon_entropy_bits(p)).max(0.0)
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
    joint
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&p| p > 0.0)
        .map(|&p| -p * p.log2())
        .sum()
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
    (hxy - hx).max(0.0)
}

/// Conditional entropy H(X|Y), in bits, for a joint pmf P(X,Y).
pub fn conditional_entropy_x_given_y_bits(joint: &Matrix) -> f64 {
    let hxy = joint_entropy_bits(joint);
    let hy = shannon_entropy_bits(&marginal_y(joint));
    (hxy - hy).max(0.0)
}

/// Mutual information I(X;Y), in bits, for a joint pmf P(X,Y).
pub fn mutual_information_bits(joint: &Matrix) -> f64 {
    let hx = shannon_entropy_bits(&marginal_x(joint));
    let hy = shannon_entropy_bits(&marginal_y(joint));
    let hxy = joint_entropy_bits(joint);
    (hx + hy - hxy).max(0.0)
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
}
