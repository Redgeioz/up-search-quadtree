//! Counted multiset of f64 values with an exact maximum: a pooled, compressed
//! radix-64 bitmap over order-preserving 64-bit keys. The top bit of an
//! entry's payload marks branches, which keeps entries at 16 bytes on 64-bit
//! targets. Each branch holds up to four children inline; larger child buffers
//! and freed branches are retained for reuse. A path has at most 11 branches.
//! Operations are O(1) for the fixed key width, with amortized pool growth;
//! storage follows the high-water mark until drop.

const NONE: usize = usize::MAX;
const BRANCH: usize = 1 << (usize::BITS - 1);
const SIGN: u64 = 1 << 63;
/// Unshifted key of -infinity; subtracting it maps -infinity to key zero.
const KEY_OFFSET: u64 = !f64::NEG_INFINITY.to_bits();

/// Order-preserving key of `value`. A nonnegative value maps to its bits with
/// the sign bit set, a negative one to the complement of its bits, which
/// orders all values except NaN. Subtracting `KEY_OFFSET` maps -infinity to
/// zero, so the maximum key of an empty set reads as -infinity. Negative zero
/// sorts immediately below zero.
#[inline]
pub(crate) fn key_of(value: f64) -> u64 {
    let bits = value.to_bits();
    let flip = ((bits as i64 >> 63) as u64) | SIGN;
    (bits ^ flip).wrapping_sub(KEY_OFFSET)
}

/// Inverse of [`key_of`].
#[inline]
fn value_of(key: u64) -> f64 {
    let ordered = key.wrapping_add(KEY_OFFSET);
    let flip = !((ordered as i64 >> 63) as u64) | SIGN;
    f64::from_bits(ordered ^ flip)
}

#[derive(Default)]
pub(crate) struct MaxMultiset {
    root: Option<BitmapEntry>,
    /// Key of the largest value; zero, the key of -infinity, while empty.
    maximum: u64,
    branches: Vec<BitmapBranch>,
    free: Option<usize>,
}

#[derive(Clone, Copy, Default)]
struct BitmapEntry {
    /// Leaf: the key. Branch: the prefix common to its keys.
    key: u64,
    /// Leaf: the duplicate count. Branch: the stable pool index, top bit set.
    tagged_payload: usize,
}

impl BitmapEntry {
    #[inline]
    fn leaf(key: u64, count: usize) -> Self {
        debug_assert!(count > 0 && count & BRANCH == 0);
        Self {
            key,
            tagged_payload: count,
        }
    }
    #[inline]
    fn branch(prefix: u64, index: usize) -> Self {
        debug_assert_eq!(index & BRANCH, 0);
        Self {
            key: prefix,
            tagged_payload: index | BRANCH,
        }
    }
    #[inline]
    fn key(self) -> u64 {
        self.key
    }
    #[inline]
    fn is_branch(self) -> bool {
        self.tagged_payload & BRANCH != 0
    }
    #[inline]
    fn count(self) -> usize {
        self.tagged_payload
    }
    #[inline]
    fn index(self) -> usize {
        self.tagged_payload & !BRANCH
    }
}

enum Children {
    Inline {
        len: usize,
        entries: [BitmapEntry; 4],
    },
    Heap(Vec<BitmapEntry>),
}

impl Default for Children {
    fn default() -> Self {
        Self::Inline {
            len: 0,
            entries: [BitmapEntry::default(); 4],
        }
    }
}

impl std::ops::Deref for Children {
    type Target = [BitmapEntry];
    #[inline]
    fn deref(&self) -> &[BitmapEntry] {
        match self {
            Self::Inline { len, entries } => &entries[..*len],
            Self::Heap(entries) => entries,
        }
    }
}

impl std::ops::DerefMut for Children {
    #[inline]
    fn deref_mut(&mut self) -> &mut [BitmapEntry] {
        match self {
            Self::Inline { len, entries } => &mut entries[..*len],
            Self::Heap(entries) => entries,
        }
    }
}

impl Children {
    #[inline]
    fn insert(&mut self, index: usize, entry: BitmapEntry) {
        match self {
            Self::Inline { len, entries } if *len < 4 => {
                assert!(index <= *len);
                entries.copy_within(index..*len, index + 1);
                entries[index] = entry;
                *len += 1;
            }
            Self::Inline { len, entries } => {
                let mut heap = Vec::with_capacity(8);
                heap.extend_from_slice(&entries[..*len]);
                heap.insert(index, entry);
                *self = Self::Heap(heap);
            }
            Self::Heap(entries) => entries.insert(index, entry),
        }
    }
    #[inline]
    fn remove(&mut self, index: usize) {
        match self {
            Self::Inline { len, entries } => {
                assert!(index < *len);
                entries.copy_within(index + 1..*len, index);
                *len -= 1;
            }
            Self::Heap(entries) => {
                entries.remove(index);
            }
        }
    }
    #[inline]
    fn clear(&mut self) {
        match self {
            Self::Inline { len, .. } => *len = 0,
            Self::Heap(entries) => entries.clear(),
        }
    }
    fn extend(&mut self, entries: [BitmapEntry; 2]) {
        for entry in entries {
            self.insert(self.len(), entry);
        }
    }
    #[cfg(test)]
    fn capacity(&self) -> usize {
        match self {
            Self::Inline { .. } => 4,
            Self::Heap(entries) => entries.capacity(),
        }
    }
}

struct BitmapBranch {
    shift: u32,
    bitmap: u64,
    children: Children,
    next_free: usize,
}

impl BitmapBranch {
    #[inline]
    fn prefix(key: u64, shift: u32) -> u64 {
        if shift == 60 {
            0
        } else {
            key & (u64::MAX << (shift + 6))
        }
    }
    #[inline]
    fn digit(&self, key: u64) -> u32 {
        ((key >> self.shift) & 63) as u32
    }
    #[inline]
    fn index(&self, digit: u32) -> usize {
        (self.bitmap & ((1u64 << digit) - 1)).count_ones() as usize
    }
}

#[derive(Clone, Copy)]
enum Link {
    Root,
    Child { branch: usize, child: usize },
}

impl MaxMultiset {
    #[inline]
    fn write(&mut self, link: Link, entry: BitmapEntry) {
        match link {
            Link::Root => self.root = Some(entry),
            Link::Child { branch, child } => self.branches[branch].children[child] = entry,
        }
    }

    fn split(&mut self, old: BitmapEntry, key: u64, shift: u32) -> BitmapEntry {
        let index = if let Some(index) = self.free {
            let next = self.branches[index].next_free;
            self.free = (next != NONE).then_some(next);
            index
        } else {
            let index = self.branches.len();
            self.branches.push(BitmapBranch {
                shift: 0,
                bitmap: 0,
                children: Children::default(),
                next_free: NONE,
            });
            index
        };
        let old_digit = ((old.key() >> shift) & 63) as u32;
        let new_digit = ((key >> shift) & 63) as u32;
        let node = &mut self.branches[index];
        debug_assert!(node.children.is_empty());
        node.shift = shift;
        node.bitmap = (1u64 << old_digit) | (1u64 << new_digit);
        node.next_free = NONE;
        let new = BitmapEntry::leaf(key, 1);
        node.children.extend(if old_digit < new_digit {
            [old, new]
        } else {
            [new, old]
        });
        BitmapEntry::branch(BitmapBranch::prefix(key, shift), index)
    }

    #[inline]
    fn recycle(&mut self, index: usize) {
        let branch = &mut self.branches[index];
        branch.children.clear();
        branch.bitmap = 0;
        branch.next_free = self.free.unwrap_or(NONE);
        self.free = Some(index);
    }

    fn insert_key(&mut self, key: u64) {
        let Some(mut entry) = self.root else {
            self.root = Some(BitmapEntry::leaf(key, 1));
            return;
        };
        let mut link = Link::Root;
        loop {
            if !entry.is_branch() && entry.key() == key {
                self.write(link, BitmapEntry::leaf(key, entry.count() + 1));
                return;
            }
            let difference = entry.key() ^ key;
            let split_shift = if difference == 0 {
                0
            } else {
                ((63 - difference.leading_zeros()) / 6) * 6
            };
            if entry.is_branch()
                && (difference == 0 || split_shift <= self.branches[entry.index()].shift)
            {
                let index = entry.index();
                let branch = &mut self.branches[index];
                let digit = branch.digit(key);
                let bit = 1u64 << digit;
                let child = branch.index(digit);
                if branch.bitmap & bit == 0 {
                    branch.children.insert(child, BitmapEntry::leaf(key, 1));
                    branch.bitmap |= bit;
                    return;
                }
                entry = branch.children[child];
                link = Link::Child {
                    branch: index,
                    child,
                };
            } else {
                let split = self.split(entry, key, split_shift);
                self.write(link, split);
                return;
            }
        }
    }

    fn remove_key(&mut self, key: u64) {
        let mut entry = self.root.expect("missing overflow distance");
        let mut link = Link::Root;
        let mut parent: Option<(usize, u32, usize, Link)> = None;
        loop {
            if !entry.is_branch() {
                assert_eq!(entry.key(), key, "missing overflow distance");
                if entry.count() > 1 {
                    self.write(link, BitmapEntry::leaf(key, entry.count() - 1));
                } else if let Some((index, digit, child, parent_link)) = parent {
                    let branch = &mut self.branches[index];
                    branch.children.remove(child);
                    branch.bitmap &= !(1u64 << digit);
                    if branch.children.len() == 1 {
                        let remaining = branch.children[0];
                        self.write(parent_link, remaining);
                        self.recycle(index);
                    }
                } else {
                    self.root = None;
                }
                return;
            }
            let index = entry.index();
            let branch = &self.branches[index];
            assert_eq!(
                BitmapBranch::prefix(key, branch.shift),
                entry.key(),
                "missing overflow distance"
            );
            let digit = branch.digit(key);
            assert_ne!(
                branch.bitmap & (1u64 << digit),
                0,
                "missing overflow distance"
            );
            let child = branch.index(digit);
            parent = Some((index, digit, child, link));
            link = Link::Child {
                branch: index,
                child,
            };
            entry = branch.children[child];
        }
    }

    fn maximum_key(&self) -> u64 {
        let Some(mut entry) = self.root else {
            return 0;
        };
        while entry.is_branch() {
            let branch = &self.branches[entry.index()];
            let digit = 63 - branch.bitmap.leading_zeros();
            entry = branch.children[branch.index(digit)];
        }
        entry.key()
    }

    #[inline]
    fn add(&mut self, key: u64) {
        self.insert_key(key);
        self.maximum = self.maximum.max(key);
    }
    #[inline]
    fn subtract(&mut self, key: u64) {
        self.remove_key(key);
        if self.maximum == key {
            self.maximum = self.maximum_key();
        }
    }

    #[inline]
    pub(crate) fn insert(&mut self, value: f64) {
        assert!(!value.is_nan(), "overflow distance must not be NaN");
        self.add(key_of(value));
    }
    #[inline]
    pub(crate) fn remove(&mut self, value: f64) {
        self.subtract(key_of(value));
    }
    #[inline]
    pub(crate) fn replace(&mut self, old: f64, new: f64) {
        assert!(!new.is_nan(), "overflow distance must not be NaN");
        // Compare keys, not values: keys distinguish zero from negative zero,
        // and the removal must match the value inserted.
        let (old, new) = (key_of(old), key_of(new));
        if old == new {
            return;
        }
        // Insert first, so that a growing maximum needs no intermediate walk.
        self.add(new);
        self.subtract(old);
    }
    /// Largest value, or -infinity while empty.
    #[inline]
    pub(crate) fn maximum(&self) -> f64 {
        value_of(self.maximum)
    }
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.root.is_none()
    }

    #[cfg(test)]
    fn depth(&self, entry: BitmapEntry) -> usize {
        if !entry.is_branch() {
            0
        } else {
            1 + self.branches[entry.index()]
                .children
                .iter()
                .map(|&child| self.depth(child))
                .max()
                .unwrap()
        }
    }
    /// Count of every key; asserts the structural invariants on the way.
    #[cfg(test)]
    pub(crate) fn counts(&self) -> std::collections::BTreeMap<u64, usize> {
        fn visit(
            set: &MaxMultiset,
            entry: BitmapEntry,
            parent_shift: u32,
            counts: &mut std::collections::BTreeMap<u64, usize>,
            visited: &mut [bool],
        ) {
            if !entry.is_branch() {
                assert!(entry.count() > 0);
                assert!(counts.insert(entry.key(), entry.count()).is_none());
                return;
            }
            let index = entry.index();
            assert!(!visited[index]);
            visited[index] = true;
            let branch = &set.branches[index];
            assert!(branch.shift <= 60 && branch.shift % 6 == 0 && branch.shift < parent_shift);
            assert_eq!(branch.children.len(), branch.bitmap.count_ones() as usize);
            assert!((2..=64).contains(&branch.children.len()));
            let mut bitmap = branch.bitmap;
            for &child in branch.children.iter() {
                let digit = bitmap.trailing_zeros();
                assert_eq!(branch.digit(child.key()), digit);
                assert_eq!(BitmapBranch::prefix(child.key(), branch.shift), entry.key());
                bitmap &= bitmap - 1;
                visit(set, child, branch.shift, counts, visited);
            }
            assert_eq!(bitmap, 0);
        }
        let mut counts = std::collections::BTreeMap::new();
        let mut visited = vec![false; self.branches.len()];
        if let Some(root) = self.root {
            visit(self, root, 66, &mut counts, &mut visited);
        }
        let mut free = self.free;
        while let Some(index) = free {
            assert!(!visited[index], "branch pool cycle or live/free overlap");
            visited[index] = true;
            let branch = &self.branches[index];
            assert!(branch.children.is_empty() && branch.bitmap == 0);
            free = (branch.next_free != NONE).then_some(branch.next_free);
        }
        assert!(
            visited.iter().all(|&visited| visited),
            "unreachable branch pool slot"
        );
        assert_eq!(
            self.maximum,
            counts.last_key_value().map_or(0, |(&key, _)| key)
        );
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::collections::BTreeMap;

    /// Key of infinity. Keys from zero up to it are the values from -infinity
    /// to infinity; larger keys are NaN.
    fn largest_key() -> u64 {
        key_of(f64::INFINITY)
    }

    fn insert(set: &mut MaxMultiset, expected: &mut BTreeMap<u64, usize>, key: u64) {
        set.insert(value_of(key));
        *expected.entry(key).or_default() += 1;
    }

    fn remove(set: &mut MaxMultiset, expected: &mut BTreeMap<u64, usize>, key: u64) {
        set.remove(value_of(key));
        let count = expected.get_mut(&key).unwrap();
        *count -= 1;
        if *count == 0 {
            expected.remove(&key);
        }
    }

    fn assert_maximum(set: &MaxMultiset, expected: &BTreeMap<u64, usize>) {
        assert_eq!(
            key_of(set.maximum()),
            expected.last_key_value().map_or(0, |(&key, _)| key)
        );
    }

    /// Values in increasing order from -infinity through both zeros to
    /// infinity, including the neighbors of the subnormal and unit boundaries.
    fn ordered_values() -> Vec<f64> {
        let positive = [
            f64::from_bits(1), // Smallest subnormal.
            f64::MIN_POSITIVE.next_down(),
            f64::MIN_POSITIVE,
            1.0f64.next_down(),
            1.0,
            1.0f64.next_up(),
            f64::MAX,
            f64::INFINITY,
        ];
        positive
            .iter()
            .rev()
            .map(|value| -value)
            .chain([-0.0, 0.0])
            .chain(positive)
            .collect()
    }

    #[test]
    fn keys_follow_value_order_and_round_trip() {
        let values = ordered_values();
        assert_eq!(key_of(f64::NEG_INFINITY), 0);
        for pair in values.windows(2) {
            assert!(key_of(pair[0]) < key_of(pair[1]), "{} before {}", pair[0], pair[1]);
        }
        for value in values {
            assert_eq!(value_of(key_of(value)).to_bits(), value.to_bits());
        }
        assert!(value_of(largest_key() + 1).is_nan());
        assert!(value_of(u64::MAX).is_nan());
    }

    #[test]
    fn exact_float_extremes_duplicates_and_signed_zero() {
        let mut set = MaxMultiset::default();
        assert_eq!(set.maximum(), f64::NEG_INFINITY);
        let mut expected = BTreeMap::new();
        let keys: Vec<u64> = ordered_values().into_iter().map(key_of).collect();
        for &key in &keys {
            insert(&mut set, &mut expected, key);
            insert(&mut set, &mut expected, key);
            assert_eq!(set.counts(), expected);
            assert_eq!(set.maximum().to_bits(), value_of(key).to_bits());
        }
        for &key in keys.iter().rev() {
            remove(&mut set, &mut expected, key);
            assert_eq!(key_of(set.maximum()), key); // The duplicate remains.
            remove(&mut set, &mut expected, key);
            assert_maximum(&set, &expected);
            assert_eq!(set.counts(), expected);
        }
        assert!(set.is_empty());
        assert_eq!(set.maximum(), f64::NEG_INFINITY);
        // Zero and negative zero have distinct keys; each removes only its own.
        set.insert(-0.0);
        set.insert(-3.0);
        set.replace(-0.0, 0.0);
        assert_eq!(set.maximum().to_bits(), 0.0f64.to_bits());
        set.remove(0.0);
        assert_eq!(set.maximum(), -3.0);
        set.remove(-3.0);
        assert!(set.is_empty());
        // An emptied set must retain no bitmap bits or counts on reuse.
        insert(&mut set, &mut expected, key_of(1.0));
        assert_eq!(set.counts(), expected);
    }

    #[test]
    #[should_panic(expected = "must not be NaN")]
    fn nan_is_rejected() {
        MaxMultiset::default().insert(f64::NAN);
    }

    #[test]
    fn every_bitmap_digit_at_every_depth_and_branch_collapse() {
        for shift in (0..=60).step_by(6) {
            let mut set = MaxMultiset::default();
            let mut expected = BTreeMap::new();
            // The top digit spans only bits 60 to 63, so 16 values; every
            // lower digit has all 64.
            let digits = if shift == 60 { 16 } else { 64 };
            let keys: Vec<u64> = (0..digits).map(|digit| 1 | (digit << shift)).collect();
            // At shift zero the OR aliases digits 0/1, 2/3, ...; a prefix
            // above the digit keeps all 64 children distinct.
            let keys: Vec<u64> = if shift == 0 {
                (0..64).map(|digit| 64 | digit).collect()
            } else {
                keys
            };
            assert!(keys.iter().all(|&key| key <= largest_key()));
            for &key in keys.iter().rev() {
                insert(&mut set, &mut expected, key);
            }
            assert_eq!(set.counts(), expected);
            // Remove every other digit from the lowest up, then the rest from
            // the highest down, so both word edges are covered.
            for &key in keys.iter().step_by(2) {
                remove(&mut set, &mut expected, key);
                assert_maximum(&set, &expected);
                assert_eq!(set.counts(), expected);
            }
            for &key in keys.iter().skip(1).step_by(2).rev() {
                remove(&mut set, &mut expected, key);
                assert_maximum(&set, &expected);
                assert_eq!(set.counts(), expected);
            }
            assert!(set.is_empty());
        }
    }

    #[test]
    fn compressed_paths_split_above_and_below_existing_branches() {
        let mut set = MaxMultiset::default();
        let mut expected = BTreeMap::new();
        let keys = [
            1,
            2,
            1 << 18,
            1 << 12,
            1 << 54,
            (1 << 54) | 1,
            1 << 36,
            1 << 60,
            1 << 63,
        ];
        for key in keys {
            insert(&mut set, &mut expected, key);
            assert_eq!(set.counts(), expected);
        }
        for key in [
            2,
            1 << 54,
            1 << 63,
            1 << 60,
            1 << 18,
            1,
            1 << 12,
            1 << 36,
            (1 << 54) | 1,
        ] {
            remove(&mut set, &mut expected, key);
            assert_maximum(&set, &expected);
            assert_eq!(set.counts(), expected);
        }
        assert!(set.is_empty());
    }

    #[test]
    fn full_key_width_has_at_most_eleven_branches() {
        let mut set = MaxMultiset::default();
        let mut expected = BTreeMap::new();
        let keys: Vec<u64> = [1, 2]
            .into_iter()
            .chain((6..=60).step_by(6).map(|shift| 1u64 << shift))
            .collect();
        for &key in &keys {
            insert(&mut set, &mut expected, key);
            assert!(set.depth(set.root.unwrap()) <= 11);
        }
        assert_eq!(set.depth(set.root.unwrap()), 11);
        assert_eq!(set.counts(), expected);
        // Removing each maximum in turn collapses its split; the cached
        // maximum must stay exact down to the last leaf.
        for &key in keys.iter().rev() {
            remove(&mut set, &mut expected, key);
            assert_eq!(set.counts(), expected);
            assert_maximum(&set, &expected);
            if let Some(root) = set.root {
                assert!(set.depth(root) <= 11);
            }
        }
        assert!(set.is_empty());
    }

    #[test]
    fn random_mutations_match_ordered_multiset() {
        let mut rng = StdRng::seed_from_u64(826);
        let mut set = MaxMultiset::default();
        let mut expected = BTreeMap::new();
        let pool: Vec<u64> = (0..2048)
            .map(|_| rng.gen_range(0..=largest_key()))
            .collect();
        let mut live = Vec::new();
        for step in 0..30_000 {
            if live.is_empty() || rng.gen_bool(0.55) {
                let key = pool[rng.gen_range(0..pool.len())];
                insert(&mut set, &mut expected, key);
                live.push(key);
            } else {
                let index = rng.gen_range(0..live.len());
                let key = live.swap_remove(index);
                remove(&mut set, &mut expected, key);
            }
            assert_maximum(&set, &expected);
            if step % 128 == 0 {
                assert_eq!(set.counts(), expected);
            }
        }
        live.shuffle(&mut rng);
        for key in live {
            remove(&mut set, &mut expected, key);
            assert_maximum(&set, &expected);
        }
        assert_eq!(set.counts(), expected);
        assert!(set.is_empty());
    }

    #[test]
    #[should_panic(expected = "missing overflow distance")]
    fn absent_key_sharing_a_compressed_path_is_rejected() {
        let mut set = MaxMultiset::default();
        set.insert(value_of(1));
        set.insert(value_of(2));
        set.remove(value_of(1 << 18));
    }

    #[test]
    fn replace_handles_reordering_duplicates_and_maximum_changes() {
        let mut set = MaxMultiset::default();
        let mut expected = BTreeMap::new();
        for key in [64, 70, 90, 120, 120] {
            insert(&mut set, &mut expected, key);
        }
        for (old, new) in [
            (70, 110),
            (110, 65),
            (120, 90),
            (120, 0),
            (64, 1 << 60),
            (1 << 60, 2),
            (0, 72),
            (90, 90),
            (90, 65),
            (2, 1 << 63),
        ] {
            set.replace(value_of(old), value_of(new));
            let count = expected.get_mut(&old).unwrap();
            *count -= 1;
            if *count == 0 {
                expected.remove(&old);
            }
            *expected.entry(new).or_default() += 1;
            assert_eq!(set.counts(), expected);
            assert_maximum(&set, &expected);
        }
    }

    #[test]
    fn branch_pool_and_child_buffers_are_reused_after_emptying() {
        let mut set = MaxMultiset::default();
        let mut expected = BTreeMap::new();
        let keys: Vec<u64> = (0..64).map(|digit| 64 | digit).collect();
        for &key in &keys {
            insert(&mut set, &mut expected, key);
        }
        for &key in &keys {
            remove(&mut set, &mut expected, key);
        }
        let branch_slots = set.branches.len();
        let pool_capacity = set.branches.capacity();
        let capacities: Vec<_> = set
            .branches
            .iter()
            .map(|node| node.children.capacity())
            .collect();
        for _ in 0..100 {
            for &key in &keys {
                insert(&mut set, &mut expected, key);
            }
            for &key in keys.iter().rev() {
                remove(&mut set, &mut expected, key);
            }
            assert_eq!(set.counts(), expected);
            assert!(set.is_empty());
            assert_eq!(set.branches.len(), branch_slots);
            assert_eq!(set.branches.capacity(), pool_capacity);
            assert_eq!(
                set.branches
                    .iter()
                    .map(|node| node.children.capacity())
                    .collect::<Vec<_>>(),
                capacities
            );
        }
    }

    #[test]
    fn random_replacements_match_ordered_multiset() {
        let mut rng = StdRng::seed_from_u64(12487);
        let mut set = MaxMultiset::default();
        let mut expected = BTreeMap::new();
        let mut live: Vec<u64> = (0..512).map(|_| rng.gen_range(0..512)).collect();
        for &key in &live {
            insert(&mut set, &mut expected, key);
        }
        for step in 0..40_000 {
            let index = rng.gen_range(0..live.len());
            let old = live[index];
            let new = if rng.gen_bool(0.5) {
                rng.gen_range(0..512) // Duplicates and dense digit collisions.
            } else {
                rng.gen_range(0..=largest_key())
            };
            set.replace(value_of(old), value_of(new));
            let count = expected.get_mut(&old).unwrap();
            *count -= 1;
            if *count == 0 {
                expected.remove(&old);
            }
            *expected.entry(new).or_default() += 1;
            live[index] = new;
            assert_maximum(&set, &expected);
            if step % 128 == 0 {
                assert_eq!(set.counts(), expected);
            }
        }
        for key in live {
            remove(&mut set, &mut expected, key);
        }
        assert!(set.is_empty());
        assert_eq!(set.counts(), expected);
    }
}
