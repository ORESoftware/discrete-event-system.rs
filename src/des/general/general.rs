//! Canonical use path: `crate::des::general::general::*`
//!
//! Port of `src/des/general/general.ts` — grab-bag framework utilities: the
//! `DESSet`/`DESMap` collection wrappers, the `bgn` decimal constructor,
//! reasonable-`u` random draws, an in-place Fisher–Yates shuffle, short-UUID
//! generation, BigNumber histograms, the `HasComputedProperties` trait, and an
//! error builder.
//!
//! Randomness is injected (`&mut dyn RandomSource`) rather than ambient — there
//! is no `DEFAULT_RANDOM`, matching the engine's capability port.
//!
//! PORT NOTES:
//!   * `sendRaw` (ws WebSocket) and `deJSON` (JSON.parse callback) are OMITTED —
//!     the foundation assumes no `ws`/`serde` dependency. Reintroduce behind a
//!     feature when transport/serialization is ported.
//!   * The `(math.bignumber(69) as any).__proto__.toJSON = ...` prototype
//!     monkey-patch and the per-value `toJSON` patch inside `bgn` are dropped;
//!     decimal rounding is done explicitly where serialization happens.

#![allow(dead_code)]

use std::collections::{BTreeMap, HashMap};
use std::hash::Hash;

use crate::des::r#abstract::interfaces::{EntityGraphData, HasId};
use crate::des::shared::capabilities::RandomSource;
use crate::des::shared::precision::{bgn as precision_bgn, to_f64, Decimal};

/// Re-export of the engine-wide exact-decimal constructor (`math.bignumber(String(x))`).
/// `general.ts` defined its own `bgn`; here it is the shared `precision::bgn` so
/// the whole engine shares one decimal policy.
#[inline]
pub fn bgn(v: f64) -> Decimal {
    precision_bgn(v)
}

/// `interface HasComputedProperties` — entities that can expose a derived view.
/// The TS generic return type is erased to [`EntityGraphData`].
pub trait HasComputedProperties {
    fn get_with_computed_properties(&self) -> EntityGraphData;
}

/// `getShortUUID()` — last 10 chars of a v4 UUID's (hyphenated) string form.
pub fn get_short_uuid() -> String {
    let s = uuid::Uuid::new_v4().to_string();
    // UUID strings are ASCII, so byte-slicing the last 10 chars is safe and
    // matches JS `uuid.v4().slice(-10)`.
    s[s.len() - 10..].to_string()
}

/// In-place Fisher–Yates shuffle (`fisherYatesShuffle`), drawing from an injected
/// `RandomSource`.
///
/// PORT NOTE: the TS version was a `function*` that *yielded* each freshly-swapped
/// element. Here we shuffle in place and return nothing; callers that needed the
/// yielded order can read the slice afterwards.
pub fn fisher_yates_shuffle<T>(deck: &mut [T], rng: &mut dyn RandomSource) {
    if deck.is_empty() {
        return;
    }
    let mut i = deck.len() - 1;
    loop {
        let swap_index = (rng.next_float() * (i as f64 + 1.0)).floor() as usize;
        deck.swap(i, swap_index);
        if i == 0 {
            break;
        }
        i -= 1;
    }
}

/// `getReasonableU` — a `u` clamped to `[0.001, 0.999]`, returned as an exact decimal.
pub fn get_reasonable_u(rng: &mut dyn RandomSource) -> Decimal {
    let u = 0.001_f64.max(rng.next_float());
    bgn(0.999_f64.min(u))
}

/// `getReasonableUNative` — the same clamp, as a plain `f64`.
pub fn get_reasonable_u_native(rng: &mut dyn RandomSource) -> f64 {
    let u = 0.001_f64.max(rng.next_float());
    0.999_f64.min(u)
}

/// `class DESSet<V extends HasId> extends Set<V>` — wrap, don't inherit.
/// Set semantics (dedup) are by `HasId::id`.
pub struct DESSet<V: HasId> {
    items: Vec<V>,
}

impl<V: HasId> Default for DESSet<V> {
    fn default() -> Self {
        DESSet { items: Vec::new() }
    }
}

impl<V: HasId> DESSet<V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// `add(v)` — insert if no existing element shares its id.
    pub fn add(&mut self, v: V) {
        let id = v.id();
        if !self.items.iter().any(|e| e.id() == id) {
            self.items.push(v);
        }
    }

    /// `size`.
    pub fn size(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn has(&self, id: &str) -> bool {
        self.items.iter().any(|e| e.id() == id)
    }

    /// `values()` — iterate the contained elements.
    pub fn values(&self) -> impl Iterator<Item = &V> {
        self.items.iter()
    }

    pub fn iter(&self) -> impl Iterator<Item = &V> {
        self.items.iter()
    }

    /// `toJSON()` analog: `{ size, values: [ids...] }` as a compact JSON string.
    pub fn to_json(&self) -> String {
        let ids: Vec<String> = self.items.iter().map(|v| format!("{:?}", v.id())).collect();
        format!("{{\"size\":{},\"values\":[{}]}}", self.size(), ids.join(","))
    }
}

impl<V: HasId> IntoIterator for DESSet<V> {
    type Item = V;
    type IntoIter = std::vec::IntoIter<V>;
    fn into_iter(self) -> Self::IntoIter {
        self.items.into_iter()
    }
}

/// `class DESMap<K extends Number, V extends BigNumber> extends Map<K, V>`.
/// Wrap a `HashMap`; the TS `toJSON` (coerce values to `Number`) becomes
/// [`DESMap::to_json`].
pub struct DESMap<K: Eq + Hash, V> {
    items: HashMap<K, V>,
}

impl<K: Eq + Hash, V> Default for DESMap<K, V> {
    fn default() -> Self {
        DESMap {
            items: HashMap::new(),
        }
    }
}

impl<K: Eq + Hash, V> DESMap<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set(&mut self, k: K, v: V) {
        self.items.insert(k, v);
    }

    pub fn get(&self, k: &K) -> Option<&V> {
        self.items.get(k)
    }

    pub fn get_mut(&mut self, k: &K) -> Option<&mut V> {
        self.items.get_mut(k)
    }

    pub fn size(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn entries(&self) -> impl Iterator<Item = (&K, &V)> {
        self.items.iter()
    }
}

impl<K: Eq + Hash + std::fmt::Display> DESMap<K, Decimal> {
    /// `toJSON()`: map each value to its `f64`, keyed by the stringified key.
    pub fn to_json(&self) -> String {
        let parts: Vec<String> = self
            .items
            .iter()
            .map(|(k, v)| format!("\"{}\":{}", k, to_f64(*v)))
            .collect();
        format!("{{{}}}", parts.join(","))
    }
}

/// `getSortedTimeHistogram` — return the histogram sorted by (numeric) key.
pub fn get_sorted_time_histogram(h: &HashMap<i64, Decimal>) -> BTreeMap<i64, Decimal> {
    h.iter().map(|(k, v)| (*k, *v)).collect()
}

/// `getSortedHistogram` — normalize each bucket to a percentage of the total,
/// round (3 dp if > 10%, else 4 dp), and return sorted by key.
pub fn get_sorted_histogram(h: &HashMap<i64, Decimal>) -> BTreeMap<i64, Decimal> {
    let mut total = bgn(0.0);
    for v in h.values() {
        total += *v;
    }
    let mut out: BTreeMap<i64, Decimal> = BTreeMap::new();
    for (k, v) in h.iter() {
        if total == Decimal::ZERO {
            out.insert(*k, bgn(0.0));
            continue;
        }
        let product = Decimal::from(100) * (*v / total);
        let round_to = if product > bgn(10.0) { 3 } else { 4 };
        out.insert(*k, product.round_dp(round_to));
    }
    out
}

/// `getSortedHistogram_NEW` — array-shaped histogram (index = bucket key); the
/// rounding precision branch is `4 dp if > 10%, else 3 dp` (the inverse of
/// [`get_sorted_histogram`], preserved verbatim from the TS).
///
/// PORT NOTE: TS wrote into a sparse `Array` indexed by key; here gaps are filled
/// with `0` and negative keys are skipped (histogram keys are non-negative counts).
pub fn get_sorted_histogram_new(h: &HashMap<i64, Decimal>) -> Vec<Decimal> {
    let mut total = bgn(0.0);
    for v in h.values() {
        total += *v;
    }
    let max_key = h.keys().copied().filter(|k| *k >= 0).max().unwrap_or(-1);
    let mut out: Vec<Decimal> = vec![bgn(0.0); (max_key + 1).max(0) as usize];
    for (k, v) in h.iter() {
        if *k < 0 || total == Decimal::ZERO {
            continue;
        }
        let product = Decimal::from(100) * (*v / total);
        let round_to = if product > bgn(10.0) { 4 } else { 3 };
        out[*k as usize] = product.round_dp(round_to);
    }
    out
}

/// `makeError(...msg)` — build an error message string.
///
/// PORT NOTE: the TS used `cli-color` bolding and `util.inspect` for rich object
/// dumps and returned an `Error`. Invariant-violation call sites `throw` it,
/// which maps to `panic!`; here `make_error` simply returns the message and
/// callers `panic!("{}", make_error(...))` (or panic directly).
pub fn make_error(msg: &str) -> String {
    msg.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::des::shared::capabilities::SeededRandom;

    struct Item {
        id: String,
    }
    impl HasId for Item {
        fn id(&self) -> String {
            self.id.clone()
        }
    }

    #[test]
    fn des_set_dedups_by_id() {
        let mut s = DESSet::new();
        s.add(Item { id: "a".into() });
        s.add(Item { id: "a".into() });
        s.add(Item { id: "b".into() });
        assert_eq!(s.size(), 2);
        assert!(s.has("a"));
        assert!(!s.has("z"));
    }

    #[test]
    fn des_map_to_json() {
        let mut m: DESMap<i64, Decimal> = DESMap::new();
        m.set(1, bgn(0.5));
        assert_eq!(m.size(), 1);
        assert!(m.to_json().contains("0.5"));
    }

    #[test]
    fn reasonable_u_is_clamped() {
        let mut rng = SeededRandom::new(7);
        for _ in 0..1000 {
            let u = get_reasonable_u_native(&mut rng);
            assert!((0.001..=0.999).contains(&u));
        }
    }

    #[test]
    fn shuffle_is_a_permutation() {
        let mut rng = SeededRandom::new(3);
        let mut deck: Vec<i32> = (0..20).collect();
        fisher_yates_shuffle(&mut deck, &mut rng);
        let mut sorted = deck.clone();
        sorted.sort();
        assert_eq!(sorted, (0..20).collect::<Vec<_>>());
    }

    #[test]
    fn short_uuid_is_ten_chars() {
        assert_eq!(get_short_uuid().len(), 10);
    }

    #[test]
    fn histogram_normalizes_to_percent() {
        let mut h: HashMap<i64, Decimal> = HashMap::new();
        h.insert(0, bgn(1.0));
        h.insert(1, bgn(3.0));
        let sorted = get_sorted_histogram(&h);
        // 1 of 4 -> 25%, 3 of 4 -> 75%
        assert_eq!(sorted.get(&0), Some(&bgn(25.0)));
        assert_eq!(sorted.get(&1), Some(&bgn(75.0)));
    }
}
