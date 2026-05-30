//! Port of src/des/test/iterator-test.ts
//!
//! A `LinkedQueue` iterator smoke/demo. The TypeScript file only printed the
//! yielded items (no assertions); the Rust port promotes it to a `#[test]`
//! that asserts the FIFO order, draining the queue (the npm `@oresoftware/
//! linked-queue` maps onto `crate::des::shared::linked_queue::LinkedQueue`,
//! which exposes `enqueue`/`dequeue`/`peek` rather than an `iterator()`).

#![allow(dead_code)]

#[cfg(test)]
mod tests {
    use crate::des::shared::linked_queue::LinkedQueue;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Item {
        foo: i32,
    }

    #[test]
    fn full_iteration_yields_fifo_order() {
        let mut q: LinkedQueue<u64, Item> = LinkedQueue::new();
        q.enqueue(Item { foo: 1 });
        q.enqueue(Item { foo: 2 });
        q.enqueue(Item { foo: 3 });
        q.enqueue(Item { foo: 4 });

        let mut collected: Vec<i32> = Vec::new();
        while let Some((_, v)) = q.dequeue() {
            collected.push(v.foo);
        }
        assert_eq!(collected, vec![1, 2, 3, 4]);
    }

    #[test]
    fn partial_iteration_breaks_after_first() {
        // The TS demo's second loop `break`s after the first item; here the
        // head is observed without draining the rest.
        let mut q: LinkedQueue<u64, Item> = LinkedQueue::new();
        q.enqueue(Item { foo: 1 });
        q.enqueue(Item { foo: 2 });
        q.enqueue(Item { foo: 3 });
        q.enqueue(Item { foo: 4 });

        let first = q.peek().map(|(_, v)| v.foo);
        assert_eq!(first, Some(1));
        assert_eq!(q.size(), 4);
    }
}
