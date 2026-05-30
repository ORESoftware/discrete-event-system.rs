//! Port of src/des/test/queue-bias-test.ts
//!
//! Bias / correctness tests for the FIFO queue backing the processors. The
//! TypeScript original exercised the npm `@oresoftware/linked-queue`, which is a
//! true doubly-linked list exposing `head`/`tail`/`lookup` internals plus
//! `getRandomKey()`, `addToFront()`, `iterator()` and `reverseIterator()`.
//!
//! The Rust port [`crate::des::shared::linked_queue::LinkedQueue`] is a
//! `VecDeque<(K, V)>` shim: it preserves FIFO + keyed `enqueue`/`dequeue`/
//! `remove`/`size` semantics but intentionally does NOT replicate the linked
//! list internals. The portable behavioural tests below cover:
//!
//!   T1  Pure FIFO:   N enqueues then N dequeues yield items in order.
//!   T2  Mixed FIFO:  random interleaved enqueue/dequeue still yields items in
//!                    original enqueue order.
//!   T3  Remove:      `remove(k)` returns the right value and does not perturb
//!                    the relative order of the survivors.
//!   T5  No leaks:    after equal enqueue+dequeue, `size() == 0`.
//!
//! PORT NOTE: T4 (getRandomKey uniformity / chi-square), T6 (head/tail/lookup
//! structural invariants under load), T7 (iterator / reverseIterator order) and
//! T8 (addToFront LIFO) target methods that the VecDeque shim does not expose,
//! so those cases are deferred until the shim grows the linked-list surface.

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::general::prng::mulberry32;
    use crate::des::shared::capabilities::RandomSource;
    use crate::des::shared::linked_queue::LinkedQueue;

    /// T1: pure FIFO with N enqueue then N dequeue.
    #[test]
    fn t1_pure_fifo() {
        let n = 10_000;
        let mut q: LinkedQueue<String, i32> = LinkedQueue::new();
        for i in 0..n {
            q.enqueue_keyed(format!("k{i}"), i);
        }
        assert_eq!(q.size(), n as usize);

        for i in 0..n {
            let (_, val) = q.dequeue().expect("dequeue should yield an item");
            assert_eq!(val, i, "FIFO order broke at i={i}");
        }
        assert_eq!(q.size(), 0);
    }

    /// T2: random interleaved enqueue/dequeue still yields items in enqueue order.
    #[test]
    fn t2_mixed_fifo() {
        let steps = 50_000;
        let mut rng = mulberry32(0xC0FFEE);
        let mut q: LinkedQueue<String, i32> = LinkedQueue::new();
        let mut next_id = 0i32;
        let mut dequeued: Vec<i32> = Vec::new();

        for _ in 0..steps {
            if rng.next_float() < 0.6 || q.size() == 0 {
                q.enqueue_keyed(format!("k{next_id}"), next_id);
                next_id += 1;
            } else {
                let (_, val) = q.dequeue().unwrap();
                dequeued.push(val);
            }
        }
        while q.size() > 0 {
            let (_, val) = q.dequeue().unwrap();
            dequeued.push(val);
        }

        for (i, &v) in dequeued.iter().enumerate() {
            assert_eq!(v, i as i32, "enqueue order broke at i={i}");
        }
        assert_eq!(q.size(), 0);
    }

    /// T3: `remove(k)` returns the right value and preserves survivor order.
    #[test]
    fn t3_remove_preserves_order() {
        let n = 2_000i32;
        let remove_frac = 0.30;
        let mut rng = mulberry32(0xDEADBEEF);
        let mut q: LinkedQueue<String, i32> = LinkedQueue::new();
        for i in 0..n {
            q.enqueue_keyed(format!("k{i}"), i);
        }

        let target = (n as f64 * remove_frac).floor() as i32;
        let mut removed: std::collections::HashSet<i32> = std::collections::HashSet::new();
        while (removed.len() as i32) < target {
            let i = (rng.next_float() * n as f64).floor() as i32;
            if !removed.contains(&i) {
                let val = q.remove(&format!("k{i}")).expect("remove should find the key");
                assert_eq!(val, i, "remove returned the wrong value at i={i}");
                removed.insert(i);
            }
        }
        assert_eq!(q.size(), (n - removed.len() as i32) as usize);

        let mut survivors: Vec<i32> = Vec::new();
        while q.size() > 0 {
            let (_, val) = q.dequeue().unwrap();
            survivors.push(val);
        }
        for i in 1..survivors.len() {
            assert!(
                survivors[i] > survivors[i - 1],
                "survivor order inversion at i={i}"
            );
        }
    }

    /// T5: no leaks after equal enqueue + dequeue.
    #[test]
    fn t5_no_leaks() {
        let n = 5_000;
        let mut q: LinkedQueue<String, i32> = LinkedQueue::new();
        for i in 0..n {
            q.enqueue_keyed(format!("k{i}"), i);
        }
        for _ in 0..n {
            q.dequeue();
        }
        assert_eq!(q.size(), 0);
        assert!(q.is_empty());
    }
}
