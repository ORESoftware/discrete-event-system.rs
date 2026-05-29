//! Port of `src/des/shared/transform.ts`.
//!
//! The "function as a type" abstraction. In Rust the TypeScript classes become
//! the `Transform` trait plus a blanket newtype for closures:
//!
//!   interface Transform<I, O>          ->  trait Transform<I, O>
//!   abstract class PureTransform<I,O>  ->  (just implement Transform)
//!   transform(input: I): O             ->  fn transform(&self, input: I) -> O
//!
//! CONVENTIONS (same as the TS source):
//!   * One input, one output. Bundle multiple arguments into one input struct
//!     (named fields).
//!   * Configuration/parameters live as struct fields, read by `transform`.
//!   * A pure transform must be deterministic & side-effect free; inject RNG /
//!     clock via `super::capabilities` rather than reaching for globals.
//!   * For fallible behaviour return `Result<O, E>` from a `FallibleTransform`.
//!
//! Note: Rust has no abstract base classes, so `PureTransform` /
//! `StatefulTransform` from the TS file collapse into a single `Transform`
//! trait. Purity vs. statefulness is expressed by whether the implementing type
//! holds mutable state and whether methods take `&self` or `&mut self` (use the
//! `StatefulTransform` trait below when in-place mutation is needed).

/// The fundamental trait: turn an `I` into an `O`.
pub trait Transform<I, O> {
    fn transform(&self, input: I) -> O;
}

/// A transform that mutates internal state across invocations
/// (running accumulators, in-place iterative solvers, RNG-backed samplers).
/// Equivalent to a method taking `&mut self`.
pub trait StatefulTransform<I, O> {
    fn transform(&mut self, input: I) -> O;
}

/// A transform whose failure is an expected value, not a panic (Rust `Result`).
pub trait FallibleTransform<I, O, E = String> {
    fn transform(&self, input: I) -> Result<O, E>;
}

/// Adapter wrapping a closure as a `Transform`, mirroring `FnTransform` in the
/// TS source. Prefer a named struct for anything reused.
pub struct FnTransform<F> {
    f: F,
}

impl<F> FnTransform<F> {
    pub fn new<I, O>(f: F) -> Self
    where
        F: Fn(I) -> O,
    {
        FnTransform { f }
    }
}

impl<I, O, F> Transform<I, O> for FnTransform<F>
where
    F: Fn(I) -> O,
{
    fn transform(&self, input: I) -> O {
        (self.f)(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Doubler;
    impl Transform<i32, i32> for Doubler {
        fn transform(&self, input: i32) -> i32 {
            input * 2
        }
    }

    #[test]
    fn named_transform() {
        assert_eq!(Doubler.transform(21), 42);
    }

    #[test]
    fn fn_transform() {
        let t = FnTransform::new(|x: i32| x + 1);
        assert_eq!(t.transform(9), 10);
    }
}
