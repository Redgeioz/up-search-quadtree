//! Sort and sweep on the x axis: the items sorted by min x, every pair found
//! by scanning forward from each item over the items that start before it
//! ends. This is the forward scan of Brinkhoff, Kriegel and Seeger's spatial
//! join (1993) and Terdiman's complete box pruning, kept here to compare the
//! pair queries of the grids with an approach that keeps no cells at all.
//! It is not a spatial index: a search scans every item that starts before
//! the query ends.
//!
//! Bounds and items are stored by position, as one array per coordinate, so
//! that a scan reads each coordinate of consecutive items contiguously.
//! Inserts, updates and removals write in place and leave the order stale;
//! [`refresh_order`](SortAndSweep::refresh_order) restores it by insertion
//! sort, which is cheap while items move little between refreshes, and falls
//! back to a full sort when the shifts exceed a budget. While the order is
//! stale, the pair queries sort a copy.

use crate::handle::{ItemHandle, Slots};
use crate::rect::Rectangle;

/// Items that the chunked sweep tests together.
const LANES: usize = 8;

/// Shifts per item that the insertion sort of a refresh makes at most before
/// it falls back to a full sort.
const SHIFT_BUDGET: usize = 8;

/// Bounds and items by position, one array per coordinate.
struct Columns<T> {
    min_x: Vec<f64>,
    max_x: Vec<f64>,
    min_y: Vec<f64>,
    max_y: Vec<f64>,
    items: Vec<T>,
}

impl<T: Copy> Columns<T> {
    /// The items at the positions of `order`, in that order.
    fn gather(&self, order: &[u32]) -> Self {
        let pick = |column: &[f64]| order.iter().map(|&p| column[p as usize]).collect();
        Self {
            min_x: pick(&self.min_x),
            max_x: pick(&self.max_x),
            min_y: pick(&self.min_y),
            max_y: pick(&self.max_y),
            items: order.iter().map(|&p| self.items[p as usize]).collect(),
        }
    }

    /// Every pair, item by item; the columns must be sorted by min x.
    fn sweep_scalar(&self, callback: &mut impl FnMut(T, T)) {
        let len = self.min_x.len();
        for i in 0..len {
            let (a_min_x, a_max_x) = (self.min_x[i], self.max_x[i]);
            let (a_min_y, a_max_y) = (self.min_y[i], self.max_y[i]);
            for j in i + 1..len {
                if !(self.min_x[j] < a_max_x) {
                    break;
                }
                // Equal min x leaves the order of the two open, so the
                // later one may end where the earlier one starts.
                if a_min_x < self.max_x[j] && self.min_y[j] < a_max_y && a_min_y < self.max_y[j] {
                    callback(self.items[i], self.items[j]);
                }
            }
        }
    }

    /// Every pair, testing [`LANES`] items at a time without branches, which
    /// compiles to vector compares; the columns must be sorted by min x.
    fn sweep_chunked(&self, callback: &mut impl FnMut(T, T)) {
        let len = self.min_x.len();
        for i in 0..len {
            let (a_min_x, a_max_x) = (self.min_x[i], self.max_x[i]);
            let (a_min_y, a_max_y) = (self.min_y[i], self.max_y[i]);
            let mut j = i + 1;
            // A chunk whose first item starts at or after the end of `a` is
            // past the end of the scan, as is everything after it.
            while j + LANES <= len && self.min_x[j] < a_max_x {
                let chunk = |column: &Vec<f64>| -> [f64; LANES] {
                    column[j..j + LANES].try_into().unwrap()
                };
                let (min_x, max_x) = (chunk(&self.min_x), chunk(&self.max_x));
                let (min_y, max_y) = (chunk(&self.min_y), chunk(&self.max_y));
                let mut hits = 0u32;
                for k in 0..LANES {
                    let hit = (min_x[k] < a_max_x)
                        & (a_min_x < max_x[k])
                        & (min_y[k] < a_max_y)
                        & (a_min_y < max_y[k]);
                    hits |= (hit as u32) << k;
                }
                while hits != 0 {
                    let k = hits.trailing_zeros() as usize;
                    callback(self.items[i], self.items[j + k]);
                    hits &= hits - 1;
                }
                j += LANES;
            }
            while j < len && self.min_x[j] < a_max_x {
                if (a_min_x < self.max_x[j]) & (self.min_y[j] < a_max_y) & (a_min_y < self.max_y[j])
                {
                    callback(self.items[i], self.items[j]);
                }
                j += 1;
            }
        }
    }
}

/// Positions of `min_x` in ascending order. NaN of either sign sorts to an
/// end, where it neither meets nor cuts short the scans of other items.
fn order_by_min_x(min_x: &[f64]) -> Vec<u32> {
    let mut keyed: Vec<(f64, u32)> = min_x.iter().zip(0..).map(|(&x, p)| (x, p)).collect();
    keyed.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    keyed.into_iter().map(|(_, p)| p).collect()
}

pub struct SortAndSweep<T: Copy> {
    columns: Columns<T>,
    /// Handle of the item at each position.
    handles: Vec<ItemHandle>,
    /// Position of each live handle.
    slots: Slots<u32>,
    /// Whether the positions are in order of min x.
    sorted: bool,
}

impl<T: Copy> Default for SortAndSweep<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy> SortAndSweep<T> {
    pub fn new() -> Self {
        Self {
            columns: Columns {
                min_x: Vec::new(),
                max_x: Vec::new(),
                min_y: Vec::new(),
                max_y: Vec::new(),
                items: Vec::new(),
            },
            handles: Vec::new(),
            slots: Slots::new(),
            sorted: true,
        }
    }

    /// Insert an item at the end of the order and return the handle that
    /// identifies it for [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let position =
            u32::try_from(self.handles.len()).expect("Too many items for u32 positions.");
        let handle = self.slots.allocate(position);
        let columns = &mut self.columns;
        columns.min_x.push(bounds.min_x);
        columns.max_x.push(bounds.max_x);
        columns.min_y.push(bounds.min_y);
        columns.max_y.push(bounds.max_y);
        columns.items.push(item);
        self.handles.push(handle);
        self.sorted = false;
        handle
    }

    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let position = self.slots.get(handle) as usize;
        let columns = &mut self.columns;
        columns.min_x.swap_remove(position);
        columns.max_x.swap_remove(position);
        columns.min_y.swap_remove(position);
        columns.max_y.swap_remove(position);
        columns.items.swap_remove(position);
        self.handles.swap_remove(position);
        if let Some(&moved) = self.handles.get(position) {
            *self.slots.get_mut(moved) = position as u32;
            self.sorted = false;
        }
        self.slots.free(handle);
    }

    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let position = self.slots.get(handle) as usize;
        let columns = &mut self.columns;
        columns.min_x[position] = bounds.min_x;
        columns.max_x[position] = bounds.max_x;
        columns.min_y[position] = bounds.min_y;
        columns.max_y[position] = bounds.max_y;
        self.sorted = false;
    }

    /// Sort the items by min x again after inserts, updates and removals.
    pub fn refresh_order(&mut self) {
        if !self.sorted {
            if !self.insertion_sort() {
                self.full_sort();
            }
            self.sorted = true;
        }
    }

    /// Sort by insertion, moving each item past the ones before it that start
    /// later. Gives up once the shifts exceed [`SHIFT_BUDGET`] per item,
    /// leaving the items in some order, and returns whether it finished.
    fn insertion_sort(&mut self) -> bool {
        let Self {
            columns: c,
            handles,
            slots,
            ..
        } = self;
        let len = handles.len();
        let mut budget = SHIFT_BUDGET * len;
        for i in 1..len {
            let min_x = c.min_x[i];
            if c.min_x[i - 1].total_cmp(&min_x).is_le() {
                continue;
            }
            let (max_x, min_y, max_y, item, handle) =
                (c.max_x[i], c.min_y[i], c.max_y[i], c.items[i], handles[i]);
            let mut j = i;
            while j > 0 && c.min_x[j - 1].total_cmp(&min_x).is_gt() {
                c.min_x[j] = c.min_x[j - 1];
                c.max_x[j] = c.max_x[j - 1];
                c.min_y[j] = c.min_y[j - 1];
                c.max_y[j] = c.max_y[j - 1];
                c.items[j] = c.items[j - 1];
                handles[j] = handles[j - 1];
                *slots.get_mut(handles[j]) = j as u32;
                j -= 1;
            }
            c.min_x[j] = min_x;
            c.max_x[j] = max_x;
            c.min_y[j] = min_y;
            c.max_y[j] = max_y;
            c.items[j] = item;
            handles[j] = handle;
            *slots.get_mut(handle) = j as u32;
            let shifts = i - j;
            if shifts > budget {
                return false;
            }
            budget -= shifts;
        }
        true
    }

    fn full_sort(&mut self) {
        let order = order_by_min_x(&self.columns.min_x);
        self.columns = self.columns.gather(&order);
        self.handles = order.iter().map(|&p| self.handles[p as usize]).collect();
        for (position, &handle) in self.handles.iter().enumerate() {
            *self.slots.get_mut(handle) = position as u32;
        }
    }

    /// Run `sweep` on the columns in order of min x: the stored ones if the
    /// order is up to date, else a sorted copy.
    fn with_sorted(&self, sweep: impl FnOnce(&Columns<T>)) {
        if self.sorted {
            sweep(&self.columns);
        } else {
            sweep(&self.columns.gather(&order_by_min_x(&self.columns.min_x)));
        }
    }

    /// Call `callback` once for every pair of distinct stored items whose
    /// bounds intersect, with the two items in either order, testing
    /// [`LANES`] items of a scan at a time.
    pub fn for_each_pair(&self, mut callback: impl FnMut(T, T)) {
        self.with_sorted(|columns| columns.sweep_chunked(&mut callback));
    }

    /// [`for_each_pair`](Self::for_each_pair), testing one item at a time and
    /// ending each scan at the first item that starts too late.
    pub fn for_each_pair_scalar(&self, mut callback: impl FnMut(T, T)) {
        self.with_sorted(|columns| columns.sweep_scalar(&mut callback));
    }

    /// Call `callback` once for every item whose bounds intersect `bounds`.
    /// Tests every item that starts before the query ends, or every item
    /// while the order is stale.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        let c = &self.columns;
        for p in 0..c.min_x.len() {
            // Not `!(min_x < max_x)`: NaN min x sorts to either end and must
            // not end the scan at the front.
            if self.sorted && c.min_x[p] >= bounds.max_x {
                break;
            }
            if c.min_x[p] < bounds.max_x
                && bounds.min_x < c.max_x[p]
                && c.min_y[p] < bounds.max_y
                && bounds.min_y < c.max_y[p]
            {
                callback(c.items[p]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    fn brute_force(items: &[Rectangle]) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        for (a, a_bounds) in items.iter().enumerate() {
            for (b, b_bounds) in items.iter().enumerate().skip(a + 1) {
                if a_bounds.intersects(b_bounds) {
                    pairs.push((a, b));
                }
            }
        }
        pairs
    }

    /// Both sweeps must report exactly the pairs of brute force, with the
    /// order stale after inserting, sorted after a refresh, and stale again
    /// after an update.
    fn assert_pairs(items: &[Rectangle], context: &str) {
        let mut sweep = SortAndSweep::new();
        let handles: Vec<_> = items
            .iter()
            .enumerate()
            .map(|(id, bounds)| sweep.insert(bounds.clone(), id))
            .collect();
        let expected = brute_force(items);
        let check = |sweep: &SortAndSweep<usize>, state: &str| {
            for scalar in [false, true] {
                let mut reported = Vec::new();
                let mut callback = |a: usize, b: usize| reported.push((a.min(b), a.max(b)));
                if scalar {
                    sweep.for_each_pair_scalar(&mut callback);
                } else {
                    sweep.for_each_pair(&mut callback);
                }
                reported.sort_unstable();
                assert_eq!(reported, expected, "{context}, {state}, scalar {scalar}");
            }
            for query in items {
                let mut found = Vec::new();
                sweep.search(query, |id| found.push(id));
                found.sort_unstable();
                let expected: Vec<_> = (0..items.len())
                    .filter(|&id| items[id].intersects(query))
                    .collect();
                assert_eq!(found, expected, "{context}, {state}, search {query:?}");
            }
        };
        check(&sweep, "inserted");
        sweep.refresh_order();
        check(&sweep, "refreshed");
        sweep.update(items[0].clone(), handles[0]);
        check(&sweep, "updated");
    }

    /// Items starting at the same x meet only if both have width: the scan
    /// from the first of them must still test the second one's max x.
    #[test]
    fn ties_on_min_x_need_the_full_x_test() {
        let items = [
            Rectangle::new(0.0, 0.0, 10.0, 10.0),
            Rectangle::new(0.0, 0.0, 0.0, 10.0),
            Rectangle::new(0.0, 2.0, 5.0, 5.0),
        ];
        assert_pairs(&items, "box first");
        let reversed: Vec<_> = items.iter().rev().cloned().collect();
        assert_pairs(&reversed, "point first");
    }

    /// Items with a NaN corner meet nothing, and NaN min x of either sign
    /// sorts to an end of the order without cutting the other scans short,
    /// wherever it was inserted.
    #[test]
    fn nan_items_pair_with_nothing() {
        let nan_items = [
            Rectangle::new(-f64::NAN, 0.0, 10.0, 10.0),
            Rectangle::new(f64::NAN, 0.0, 10.0, 10.0),
            Rectangle::new(0.0, 0.0, f64::NAN, 10.0),
            Rectangle::new(0.0, f64::NAN, 10.0, 10.0),
            Rectangle::new(0.0, 0.0, 10.0, f64::NAN),
        ];
        let mut items = Vec::new();
        for x in 0..20 {
            let x = x as f64;
            items.push(Rectangle::new(x, 0.0, x + 3.0, 10.0));
        }
        for (i, nan_item) in nan_items.iter().enumerate() {
            items.insert(2 + 4 * i, nan_item.clone());
        }
        assert_pairs(&items, "NaN corners");
    }

    /// Small moves leave the order nearly sorted, so a refresh sorts by
    /// insertion; the handles must still reach their items afterwards, and
    /// removals alone must leave the order stale.
    #[test]
    fn handles_survive_refreshes_by_insertion() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut sweep = SortAndSweep::new();
        let mut items: Vec<Option<(ItemHandle, Rectangle)>> = (0..200)
            .map(|id| {
                let (x, y) = (rng.gen_range(0.0..1000.0), rng.gen_range(0.0..100.0));
                let bounds = Rectangle::new(
                    x,
                    y,
                    x + rng.gen_range(1.0..20.0),
                    y + rng.gen_range(1.0..20.0),
                );
                Some((sweep.insert(bounds.clone(), id), bounds))
            })
            .collect();
        let check = |sweep: &SortAndSweep<usize>,
                     items: &[Option<(ItemHandle, Rectangle)>],
                     context: &str| {
            let mut expected = Vec::new();
            for (a, a_item) in items.iter().enumerate() {
                for (b, b_item) in items.iter().enumerate().skip(a + 1) {
                    if let (Some((_, a_bounds)), Some((_, b_bounds))) = (a_item, b_item) {
                        if a_bounds.intersects(b_bounds) {
                            expected.push((a, b));
                        }
                    }
                }
            }
            let mut reported = Vec::new();
            sweep.for_each_pair(|a, b| reported.push((a.min(b), a.max(b))));
            reported.sort_unstable();
            assert_eq!(reported, expected, "{context}");
        };
        sweep.refresh_order();
        for round in 0..4 {
            for (handle, bounds) in items.iter_mut().flatten() {
                let (dx, dy) = (rng.gen_range(-5.0..5.0), rng.gen_range(-5.0..5.0));
                *bounds = Rectangle::new(
                    bounds.min_x + dx,
                    bounds.min_y + dy,
                    bounds.max_x + dx,
                    bounds.max_y + dy,
                );
                sweep.update(bounds.clone(), *handle);
            }
            check(&sweep, &items, &format!("round {round}, moved"));
            assert!(
                sweep.insertion_sort(),
                "round {round}: the small moves should sort by insertion"
            );
            sweep.sorted = true;
            check(&sweep, &items, &format!("round {round}, refreshed"));
            for id in (round..items.len()).step_by(37) {
                if let Some((handle, _)) = items[id].take() {
                    sweep.remove(handle);
                }
            }
            check(&sweep, &items, &format!("round {round}, removed"));
            sweep.refresh_order();
        }
    }

    /// Scans longer than a chunk, ending inside one and at the last item,
    /// with every other item in the scan missed on y.
    #[test]
    fn long_scans_cross_whole_chunks() {
        for count in [7, 8, 9, 16, 17, 40] {
            let items: Vec<_> = (0..count)
                .map(|i| {
                    let x = i as f64;
                    let y = if i % 2 == 0 { 0.0 } else { 20.0 };
                    Rectangle::new(x, y, x + 13.5, y + 10.0)
                })
                .collect();
            assert_pairs(&items, &format!("{count} items"));
        }
    }
}
