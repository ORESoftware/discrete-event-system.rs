//! Canonical use path: `crate::des::shared::linked_queue::{LinkedQueue, is_void}`
//!
//! Port shim for the npm package `@oresoftware/linked-queue` (no crate exists).
//!
//! The TypeScript `LinkedQueue<V, K=...>` is used two different ways across the
//! engine:
//!
//!   * UNKEYED FIFO of items — `enqueue(v)`, `dequeue()`, `size`, `peek`/`tail`
//!     (e.g. `LinkedQueue<MovingEntity>` as a station's internal queue).
//!   * KEYED map-queue — `enqueue(key, value)`, `get(key)`, `remove(key)`
//!     (e.g. `processingTimeByStation` keyed by station id).
//!
//! This shim satisfies both with a `VecDeque<(K, V)>` backing store plus an
//! auto-incrementing id used by the unkeyed FIFO specialization. Fidelity to the
//! npm internals (an actual doubly-linked list) is intentionally NOT preserved —
//! the public surface and FIFO/keyed semantics are what downstream code needs.
//!
//! PORT NOTE: the npm `LinkedQueue` had O(1) keyed lookup via an internal index;
//! here `get`/`remove` are O(n) linear scans over the deque. Correct, just not
//! asymptotically identical. Swap in a `HashMap<K, usize>` index later if a hot
//! path needs it.

#![allow(dead_code)]

use std::collections::VecDeque;

/// FIFO queue that can be used unkeyed (auto-id) or keyed.
pub struct LinkedQueue<K, V> {
    items: VecDeque<(K, V)>,
    /// Monotonic id handed out by the unkeyed [`LinkedQueue::enqueue`].
    next_auto_id: u64,
}

impl<K, V> Default for LinkedQueue<K, V> {
    fn default() -> Self {
        LinkedQueue {
            items: VecDeque::new(),
            next_auto_id: 0,
        }
    }
}

impl<K: Clone + PartialEq, V> LinkedQueue<K, V> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of items currently queued (TS `size`).
    pub fn size(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Keyed enqueue (TS `enqueue(key, value)`): append `(key, value)` at the tail.
    pub fn enqueue_keyed(&mut self, key: K, value: V) {
        self.items.push_back((key, value));
    }

    /// Remove and return the head `(key, value)` (TS `dequeue()`).
    pub fn dequeue(&mut self) -> Option<(K, V)> {
        self.items.pop_front()
    }

    /// Look up by key (TS `get(key) -> [key, value]`).
    pub fn get(&self, key: &K) -> Option<(&K, &V)> {
        self.items
            .iter()
            .find(|(k, _)| k == key)
            .map(|(k, v)| (k, v))
    }

    /// Mutable value lookup by key (for the in-place field updates the engine does).
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        self.items
            .iter_mut()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    /// Remove the first entry matching `key`, returning its value (TS `remove(key)`).
    pub fn remove(&mut self, key: &K) -> Option<V> {
        if let Some(pos) = self.items.iter().position(|(k, _)| k == key) {
            self.items.remove(pos).map(|(_, v)| v)
        } else {
            None
        }
    }

    /// Head of the queue without removing it (TS `peek`/`head`).
    pub fn peek(&self) -> Option<&(K, V)> {
        self.items.front()
    }

    /// Tail of the queue without removing it (TS `tail`).
    pub fn tail(&self) -> Option<&(K, V)> {
        self.items.back()
    }
}

impl<V> LinkedQueue<u64, V> {
    /// Unkeyed FIFO enqueue (TS `enqueue(v)`): assigns a monotonic auto-id key.
    pub fn enqueue(&mut self, value: V) {
        let key = self.next_auto_id;
        self.next_auto_id += 1;
        self.items.push_back((key, value));
    }
}

/// `IsVoid.check(x)` analog: `true` when the option is `None`.
pub fn is_void<T>(opt: &Option<T>) -> bool {
    opt.is_none()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keyed_enqueue_get_remove() {
        let mut q: LinkedQueue<String, i32> = LinkedQueue::new();
        q.enqueue_keyed("a".to_string(), 1);
        q.enqueue_keyed("b".to_string(), 2);
        assert_eq!(q.size(), 2);
        assert_eq!(q.get(&"a".to_string()).map(|(_, v)| *v), Some(1));
        *q.get_mut(&"b".to_string()).unwrap() = 20;
        assert_eq!(q.get(&"b".to_string()).map(|(_, v)| *v), Some(20));
        assert_eq!(q.remove(&"a".to_string()), Some(1));
        assert_eq!(q.size(), 1);
    }

    #[test]
    fn unkeyed_fifo() {
        let mut q: LinkedQueue<u64, &str> = LinkedQueue::new();
        q.enqueue("first");
        q.enqueue("second");
        assert_eq!(q.peek().map(|(_, v)| *v), Some("first"));
        assert_eq!(q.tail().map(|(_, v)| *v), Some("second"));
        let (k, v) = q.dequeue().unwrap();
        assert_eq!((k, v), (0, "first"));
    }

    #[test]
    fn is_void_detects_none() {
        let some: Option<i32> = Some(3);
        let none: Option<i32> = None;
        assert!(!is_void(&some));
        assert!(is_void(&none));
    }
}
