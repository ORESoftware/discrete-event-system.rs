//! Port of src/des/test/ts-test.ts
//!
//! A tiny TypeScript inheritance / override demo (no assertions). The
//! `abstract class HasMethod -> SuperClass -> SubClass` chain has no Rust
//! equivalent via `extends`; per the migration note it becomes a
//! `trait HasMethod { fn implement_me(&self) -> T }` with structs implementing
//! it, and `SubClass` simply provides its own `implement_me`. A trivial test
//! pins the overriding behaviour.

#![allow(dead_code)]

/// `abstract class HasMethod<T> { abstract implementMe(): T }`.
trait HasMethod {
    type Output;
    fn implement_me(&self) -> Self::Output;
}

/// `class SuperClass extends HasMethod`.
struct SuperClass;

impl HasMethod for SuperClass {
    type Output = (&'static str, &'static str);
    fn implement_me(&self) -> Self::Output {
        ("foo", "bar")
    }
}

/// `class SubClass extends SuperClass` — overrides `implementMe`.
struct SubClass;

impl HasMethod for SubClass {
    type Output = (&'static str, &'static str);
    fn implement_me(&self) -> Self::Output {
        ("north", "star")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn override_returns_own_value() {
        assert_eq!(SuperClass.implement_me(), ("foo", "bar"));
        assert_eq!(SubClass.implement_me(), ("north", "star"));
    }
}
