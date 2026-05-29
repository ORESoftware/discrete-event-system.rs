//! Port of `src/des/shared/result.ts`.
//!
//! The TypeScript file hand-rolls `Result<T, E>` and `Option<T>` as tagged
//! unions specifically so they map onto Rust's std enums. In Rust those types
//! already exist (`std::result::Result`, `std::option::Option`), so this module
//! is thin: it provides a default-error alias and 1:1 helper functions matching
//! the TS API. Idiomatic ported code should prefer `Ok(..)` / `Err(..)` /
//! `Some(..)` / `None` and the inherent methods (`map`, `map_err`,
//! `unwrap_or`, `is_ok`, …) directly.

/// `Result<T, E = string>` in TypeScript ⇒ default the error to `String`.
pub type Res<T, E = String> = std::result::Result<T, E>;

#[inline]
pub fn ok<T, E>(value: T) -> Result<T, E> {
    Ok(value)
}

#[inline]
pub fn err<T, E>(error: E) -> Result<T, E> {
    Err(error)
}

#[inline]
pub fn is_ok<T, E>(r: &Result<T, E>) -> bool {
    r.is_ok()
}

#[inline]
pub fn is_err<T, E>(r: &Result<T, E>) -> bool {
    r.is_err()
}

#[inline]
pub fn some<T>(value: T) -> Option<T> {
    Some(value)
}

#[inline]
pub fn none<T>() -> Option<T> {
    None
}

#[inline]
pub fn is_some<T>(o: &Option<T>) -> bool {
    o.is_some()
}

#[inline]
pub fn is_none<T>(o: &Option<T>) -> bool {
    o.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ok_err_helpers() {
        let a: Res<i32> = ok(5);
        assert!(is_ok(&a));
        assert_eq!(a.unwrap_or(0), 5);

        let b: Res<i32> = err("boom".to_string());
        assert!(is_err(&b));
        assert_eq!(b.unwrap_or(-1), -1);
    }

    #[test]
    fn option_helpers() {
        assert!(is_some(&some(1)));
        assert!(is_none(&none::<i32>()));
    }
}
