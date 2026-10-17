//! Handles that identify stored items without hashing them.
//!
//! Every structure returns an [`ItemHandle`] from `insert` and accepts it in
//! `update` and `remove`. The handle indexes a [`Slots`] table recording where
//! the item is stored, so neither operation hashes the item or scans a node
//! for it, and the item type needs no `Eq` or `Hash` implementation.

use crate::rect::Rectangle;
use std::ptr;

/// Identifies an inserted item for `update` and `remove`.
///
/// Slots are reused after removal; the generation distinguishes a handle whose
/// item was removed from the handle of the item that reused its slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ItemHandle {
    index: u32,
    generation: u32,
}

impl ItemHandle {
    /// Slot index of the handle, usable as an index into per-item tables.
    #[inline]
    pub(crate) fn index(self) -> usize {
        self.index as usize
    }
}

/// One stored item. `bounds` comes first, so intersection tests read the
/// front of the entry; `handle` is needed only when entries move.
#[repr(C)]
pub(crate) struct Entry<T> {
    pub(crate) bounds: Rectangle,
    pub(crate) item: T,
    pub(crate) handle: ItemHandle,
}

/// Location of an entry in a structure whose levels are grids of nodes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct GridLocation {
    pub(crate) level: u32,
    pub(crate) x: u32,
    pub(crate) y: u32,
    /// Index of the entry within its node.
    pub(crate) index: u32,
}

// These methods are not generic but run inside generic tree methods, which
// other crates instantiate; `#[inline]` keeps them inlinable there.
impl GridLocation {
    #[inline]
    pub(crate) fn new((level, x, y): (usize, usize, usize), index: usize) -> Self {
        Self {
            level: level as u32,
            x: x as u32,
            y: y as u32,
            index: index as u32,
        }
    }

    /// `(level, x, y)` of the node.
    #[inline]
    pub(crate) fn coord(&self) -> (usize, usize, usize) {
        (self.level as usize, self.x as usize, self.y as usize)
    }
}

/// Location of an entry in a tree of pointer-linked nodes.
pub(crate) struct NodeLocation<N> {
    pub(crate) node: *mut N,
    /// Index of the entry within its node.
    pub(crate) index: u32,
}

impl<N> NodeLocation<N> {
    /// Placeholder for an entry that has not reached its node yet.
    pub(crate) fn detached() -> Self {
        Self {
            node: ptr::null_mut(),
            index: 0,
        }
    }
}

// Derived impls would needlessly require `N: Clone` and `N: Copy`.
impl<N> Clone for NodeLocation<N> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<N> Copy for NodeLocation<N> {}

/// Table of the location of every live handle's item.
pub(crate) struct Slots<L> {
    slots: Vec<Slot<L>>,
    /// Slots released by `free`, reused by subsequent `allocate` calls.
    free: Vec<u32>,
}

/// A freed slot already holds the generation that its next `allocate` hands
/// out, so every handle issued before the release stops matching.
struct Slot<L> {
    generation: u32,
    location: L,
}

impl<L: Copy> Slots<L> {
    pub(crate) fn new() -> Self {
        Self {
            slots: Vec::new(),
            free: Vec::new(),
        }
    }

    /// Number of live handles.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }

    /// Number of slots, live and freed.
    #[cfg(test)]
    pub(crate) fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Return a handle whose slot records `location`, reusing a freed slot if
    /// one exists.
    #[inline]
    pub(crate) fn allocate(&mut self, location: L) -> ItemHandle {
        if let Some(index) = self.free.pop() {
            let slot = &mut self.slots[index as usize];
            slot.location = location;
            ItemHandle {
                index,
                generation: slot.generation,
            }
        } else {
            let index = u32::try_from(self.slots.len()).expect("Too many items for u32 handles.");
            self.slots.push(Slot {
                generation: 0,
                location,
            });
            ItemHandle {
                index,
                generation: 0,
            }
        }
    }

    /// Location of a live handle.
    ///
    /// Panics if the handle has been removed.
    #[inline]
    pub(crate) fn get(&self, handle: ItemHandle) -> L {
        let slot = &self.slots[handle.index as usize];
        assert!(
            slot.generation == handle.generation,
            "The handle has been removed."
        );
        slot.location
    }

    /// Location of a handle known to be live, e.g. that of an entry just read
    /// from a node. Liveness is checked only in debug builds.
    #[inline]
    pub(crate) fn get_mut(&mut self, handle: ItemHandle) -> &mut L {
        let slot = &mut self.slots[handle.index as usize];
        debug_assert!(
            slot.generation == handle.generation,
            "The handle has been removed."
        );
        &mut slot.location
    }

    /// Release the slot of a live handle. The handle and all its copies are
    /// rejected from then on.
    #[inline]
    pub(crate) fn free(&mut self, handle: ItemHandle) {
        let slot = &mut self.slots[handle.index as usize];
        debug_assert!(
            slot.generation == handle.generation,
            "The handle has been removed."
        );
        slot.generation = handle.generation.wrapping_add(1);
        self.free.push(handle.index);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handles_find_their_locations() {
        let mut slots = Slots::new();
        let a = slots.allocate(1u32);
        let b = slots.allocate(2);
        assert_ne!(a, b);
        assert_eq!(slots.get(a), 1);
        assert_eq!(slots.get(b), 2);

        *slots.get_mut(a) = 3;
        assert_eq!(slots.get(a), 3);
        assert_eq!(slots.get(b), 2);
        assert_eq!(slots.len(), 2);
    }

    #[test]
    fn freed_slots_are_reused_with_a_new_generation() {
        let mut slots = Slots::new();
        let kept = slots.allocate(1u32);
        let stale = slots.allocate(2);
        slots.free(stale);
        assert_eq!(slots.len(), 1);

        let reused = slots.allocate(3);
        assert_eq!(reused.index, stale.index);
        assert_ne!(reused.generation, stale.generation);
        assert_ne!(reused, stale);
        assert_eq!(slots.get(reused), 3);
        assert_eq!(slots.get(kept), 1);
        assert_eq!(slots.len(), 2);
        assert_eq!(slots.slot_count(), 2);
    }

    #[test]
    #[should_panic(expected = "has been removed")]
    fn stale_handle_is_rejected_after_reuse() {
        let mut slots = Slots::new();
        let stale = slots.allocate(1u32);
        slots.free(stale);
        slots.allocate(2);
        slots.get(stale);
    }
}
