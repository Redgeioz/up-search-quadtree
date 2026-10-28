//! Size-class grids after the `multi` neighbor style of LAMMPS (Shire,
//! Hanley and Stratford, Comput. Part. Mech. 2021; Monti et al., Phys. Rev. E
//! 2022): items are split by size into a few collections, and every
//! collection has a uniform grid of its own whose cell size is the largest
//! item size of the collection. The papers found two or three collections
//! with log-spaced sizes optimal for continuous size distributions.
//!
//! Adapted to region queries over boxes: an item goes to the first collection
//! whose cell size is at least its larger side, and is registered once, in
//! the cell containing its center, so it reaches at most half a cell beyond
//! that cell. A search scans, in every collection, the cells within half a
//! cell plus a rounding margin of the query, and reports the items of cells
//! well inside the query untested. Items larger than every cell, or centered
//! outside the world, go to an overflow list that every search tests.
//!
//! [`for_each_pair`](MultiGrid::for_each_pair) reports every intersecting
//! pair once by the pair rules of LAMMPS: a half stencil within a collection
//! and a one-way search from smaller to larger collections. Region queries
//! report both directions and use neither.

use crate::cell_order::for_each_cell_in_morton_order;
use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::overflow_bound::rounding_slack;
use crate::rect::Rectangle;

/// (collection, x, y); the overflow list is the collection past the last.
type Coord = (usize, usize, usize);

struct Collection<T: Copy> {
    cell_size: f64,
    inverse_cell_size: f64,
    /// How far an item reaches beyond its cell: half the cell size plus the
    /// rounding margin.
    reach: f64,
    cols: usize,
    rows: usize,
    /// Cells in row-major order.
    cells: Vec<Vec<Entry<T>>>,
    /// Items in the collection; a search skips an empty one.
    count: usize,
}

impl<T: Copy> Collection<T> {
    /// Cell column or row of `relative`, a coordinate relative to the world
    /// origin. The cast saturates, so coordinates below the origin land in
    /// cell zero; `min` clamps the far side. The result never decreases as
    /// the coordinate grows.
    #[inline]
    fn index(&self, relative: f64, count: usize) -> usize {
        ((relative * self.inverse_cell_size) as usize).min(count - 1)
    }
}

pub struct MultiGrid<T: Copy> {
    /// By ascending cell size.
    collections: Vec<Collection<T>>,
    /// Items larger than every cell or centered outside the world.
    overflow: Vec<Entry<T>>,
    world_bounds: Rectangle,
    /// Where the entry of each live handle is stored.
    slots: Slots<GridLocation>,
}

impl<T: Copy> MultiGrid<T> {
    /// Grids over `world_bounds`, one per entry of `cell_sizes`, which must
    /// be positive, finite and ascending; each collection holds the items
    /// whose larger side exceeds the previous cell size but not its own.
    pub fn new(world_bounds: Rectangle, cell_sizes: &[f64]) -> Self {
        assert!(!cell_sizes.is_empty(), "At least one collection is needed.");
        assert!(
            cell_sizes.iter().all(|size| size.is_finite() && *size > 0.0),
            "Cell sizes must be positive."
        );
        assert!(
            cell_sizes.windows(2).all(|pair| pair[0] < pair[1]),
            "Cell sizes must ascend."
        );
        let slack = rounding_slack(&world_bounds);
        let collections = cell_sizes
            .iter()
            .map(|&cell_size| {
                let cols = ((world_bounds.get_width() / cell_size).ceil() as usize).max(1);
                let rows = ((world_bounds.get_height() / cell_size).ceil() as usize).max(1);
                Collection {
                    cell_size,
                    inverse_cell_size: 1.0 / cell_size,
                    reach: cell_size * 0.5 + slack,
                    cols,
                    rows,
                    cells: (0..cols * rows).map(|_| Vec::new()).collect(),
                    count: 0,
                }
            })
            .collect();
        Self {
            collections,
            overflow: Vec::new(),
            world_bounds,
            slots: Slots::new(),
        }
    }

    pub fn get_bounds(&self) -> &Rectangle {
        &self.world_bounds
    }

    /// Collection and cell that store an item with `bounds`.
    fn position(&self, bounds: &Rectangle) -> Coord {
        let center = bounds.get_center();
        let overflow = (self.collections.len(), 0, 0);
        if !self.world_bounds.contains_point(center) {
            return overflow;
        }
        let size = bounds.get_width().max(bounds.get_height());
        let Some(k) = self.collections.iter().position(|c| size <= c.cell_size) else {
            return overflow;
        };
        let collection = &self.collections[k];
        let (origin_x, origin_y) = self.world_bounds.get_min();
        let x = collection.index(center.0 - origin_x, collection.cols);
        let y = collection.index(center.1 - origin_y, collection.rows);
        (k, x, y)
    }

    fn node_mut(&mut self, (k, x, y): Coord) -> &mut Vec<Entry<T>> {
        match self.collections.get_mut(k) {
            Some(collection) => &mut collection.cells[y * collection.cols + x],
            None => &mut self.overflow,
        }
    }

    /// Count one more (`delta` 1) or one fewer (`delta` -1) item in the
    /// collection of `coord`; the overflow list keeps no count.
    fn count(&mut self, (k, _, _): Coord, delta: isize) {
        if let Some(collection) = self.collections.get_mut(k) {
            collection.count = collection.count.wrapping_add_signed(delta);
        }
    }

    /// Insert an item and return the handle that identifies it for
    /// [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord = self.position(&bounds);
        let node = self.node_mut(coord);
        let index = node.len();
        // The handle is needed in the entry, so allocate it first.
        let handle = self.slots.allocate(GridLocation::new(coord, index));
        self.node_mut(coord).push(Entry {
            bounds,
            item,
            handle,
        });
        self.count(coord, 1);
        handle
    }

    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let coord = location.coord();
        let entry = self.take_entry(coord, location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        self.count(coord, -1);
        self.slots.free(handle);
    }

    /// Update the bounds of an item and, if necessary, move it to another
    /// cell or collection.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let old = location.coord();
        let index = location.index as usize;
        let new = self.position(&bounds);
        if new == old {
            self.node_mut(old)[index].bounds = bounds;
            return;
        }
        let entry = self.take_entry(old, index);
        self.count(old, -1);
        let node = self.node_mut(new);
        let new_index = node.len();
        node.push(Entry { bounds, ..entry });
        *self.slots.get_mut(handle) = GridLocation::new(new, new_index);
        self.count(new, 1);
    }

    /// Take the entry at `index` out of a node by swap removal, and update
    /// the slot of the entry moved into its place.
    fn take_entry(&mut self, coord: Coord, index: usize) -> Entry<T> {
        let node = self.node_mut(coord);
        let entry = node.swap_remove(index);
        if let Some(moved) = node.get(index) {
            let moved = moved.handle;
            self.slots.get_mut(moved).index = index as u32;
        }
        entry
    }

    /// Search with the bounds of every stored item, as [`search`](Self::search)
    /// would item by item, and call `callback` with the item searched for and
    /// each item found, the searched one included. The items are taken
    /// collection by collection, cell by cell in Morton order, then the
    /// overflow list.
    pub fn for_each_self_query(&self, mut callback: impl FnMut(T, T)) {
        for collection in &self.collections {
            let cols = collection.cols;
            for_each_cell_in_morton_order(cols, collection.rows, |x, y| {
                for entry in &collection.cells[y * cols + x] {
                    self.search(&entry.bounds, |found| callback(entry.item, found));
                }
            });
        }
        for entry in &self.overflow {
            self.search(&entry.bounds, |found| callback(entry.item, found));
        }
    }

    /// Call `callback` once for every pair of distinct stored items whose
    /// bounds intersect, with the two items in either order, by the rules of
    /// LAMMPS `multi`. Within a collection an item reads only the items after
    /// it in its own cell and the cells after its own in row-major order (the
    /// half stencil); across collections only the smaller item searches, in
    /// every larger collection (the one-way search). The overflow list counts
    /// as the largest collection. The cells are visited in Morton order.
    pub fn for_each_pair(&self, mut callback: impl FnMut(T, T)) {
        for (k, collection) in self.collections.iter().enumerate() {
            let cols = collection.cols;
            for_each_cell_in_morton_order(cols, collection.rows, |x, y| {
                let cell = &collection.cells[y * cols + x];
                for (index, entry) in cell.iter().enumerate() {
                    let bounds = &entry.bounds;
                    let mut report = |found| callback(entry.item, found);
                    cell[index + 1..]
                        .iter()
                        .filter(|other| other.bounds.intersects(bounds))
                        .for_each(|other| report(other.item));
                    self.pair_scan(collection, bounds, Some((x, y)), &mut report);
                    for larger in &self.collections[k + 1..] {
                        if larger.count != 0 {
                            self.pair_scan(larger, bounds, None, &mut report);
                        }
                    }
                    self.overflow
                        .iter()
                        .filter(|other| other.bounds.intersects(bounds))
                        .for_each(|other| report(other.item));
                }
            });
        }
        for (index, entry) in self.overflow.iter().enumerate() {
            self.overflow[index + 1..]
                .iter()
                .filter(|other| other.bounds.intersects(&entry.bounds))
                .for_each(|other| callback(entry.item, other.item));
        }
    }

    /// Call `callback` once for every item whose bounds intersect `bounds`.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        // Every comparison with NaN is false, so a query with a NaN corner
        // meets nothing, while the scans would report the items of the cells
        // they take as inside it untested.
        if bounds.has_nan() {
            return;
        }
        let (origin_x, origin_y) = self.world_bounds.get_min();
        for collection in &self.collections {
            if collection.count == 0 {
                continue;
            }
            let reach = collection.reach;
            // An item centered in cell x reaches at most `reach` beyond it, so
            // with `index` never decreasing, these ranges hold every cell
            // whose items may meet the query.
            let min_x = collection.index(bounds.min_x - reach - origin_x, collection.cols);
            let min_y = collection.index(bounds.min_y - reach - origin_y, collection.rows);
            let max_x = collection.index(bounds.max_x + reach - origin_x, collection.cols);
            let max_y = collection.index(bounds.max_y + reach - origin_y, collection.rows);
            for y in min_y..=max_y {
                let row = &collection.cells[y * collection.cols..(y + 1) * collection.cols];
                for (x, cell) in row.iter().enumerate().take(max_x + 1).skip(min_x) {
                    if cell.is_empty() {
                        continue;
                    }
                    // A cell two or more inside the range starts over half a
                    // cell past the query's minimum and ends over half a cell
                    // before its maximum, so every item centered in it meets
                    // the query.
                    if y > min_y + 1 && y < max_y - 1 && x > min_x + 1 && x < max_x - 1 {
                        cell.iter().for_each(|entry| callback(entry.item));
                    } else {
                        cell.iter()
                            .filter(|entry| entry.bounds.intersects(bounds))
                            .for_each(|entry| callback(entry.item));
                    }
                }
            }
        }
        self.overflow
            .iter()
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }

    /// Call `callback` for every item of `collection` whose bounds intersect
    /// `bounds`, as `search` does; with `after` set to a cell, read only the
    /// cells after it in row-major order.
    fn pair_scan(
        &self,
        collection: &Collection<T>,
        bounds: &Rectangle,
        after: Option<(usize, usize)>,
        callback: &mut impl FnMut(T),
    ) {
        let (origin_x, origin_y) = self.world_bounds.get_min();
        let reach = collection.reach;
        // An item centered in cell x reaches at most `reach` beyond it, so
        // with `index` never decreasing, these ranges hold every cell whose
        // items may meet the query.
        let min_x = collection.index(bounds.min_x - reach - origin_x, collection.cols);
        let min_y = collection.index(bounds.min_y - reach - origin_y, collection.rows);
        let max_x = collection.index(bounds.max_x + reach - origin_x, collection.cols);
        let max_y = collection.index(bounds.max_y + reach - origin_y, collection.rows);
        let first_row = after.map_or(min_y, |(_, y)| min_y.max(y));
        for y in first_row..=max_y {
            let first_column = match after {
                Some((after_x, after_y)) if y == after_y => min_x.max(after_x + 1),
                _ => min_x,
            };
            let row = &collection.cells[y * collection.cols..(y + 1) * collection.cols];
            for (x, cell) in row.iter().enumerate().take(max_x + 1).skip(first_column) {
                if cell.is_empty() {
                    continue;
                }
                // A cell two or more inside the range starts over half a cell
                // past the query's minimum and ends over half a cell before its
                // maximum, so every item centered in it meets the query.
                if y > min_y + 1 && y < max_y - 1 && x > min_x + 1 && x < max_x - 1 {
                    cell.iter().for_each(|entry| callback(entry.item));
                } else {
                    cell.iter()
                        .filter(|entry| entry.bounds.intersects(bounds))
                        .for_each(|entry| callback(entry.item));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    type Tree = MultiGrid<usize>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    /// Bounds from a hundredth of the world to beyond it, centered anywhere
    /// in the world grown by a tenth, so some items overflow by size or by
    /// center; half of them start on a cell edge of some collection, give or
    /// take a float.
    fn random_bounds(rng: &mut StdRng, world: &Rectangle, cells: &[f64]) -> Rectangle {
        let side = world.get_width().min(world.get_height());
        let size = |rng: &mut StdRng| side * 2f64.powf(rng.gen_range(-9.0..0.5));
        let (x, y) = world.scale(1.1).get_random_point(rng);
        let mut bounds = Rectangle::center_rect(x, y, size(rng), size(rng));
        if rng.gen_bool(0.5) {
            let cell = cells[rng.gen_range(0..cells.len())];
            let edge = world.min_x + ((bounds.min_x - world.min_x) / cell).round() * cell;
            let edge = [edge, edge.next_up(), edge.next_down()][rng.gen_range(0..3)];
            if edge < bounds.max_x {
                bounds.min_x = edge;
            }
        }
        bounds
    }

    /// Every slot must point at its entry, every entry sit where `position`
    /// puts its bounds, and every collection count its items.
    fn assert_layout(tree: &Tree, live: &[(ItemHandle, Rectangle, usize)]) {
        for (handle, bounds, item) in live {
            let location = tree.slots.get(*handle);
            assert_eq!(location.coord(), tree.position(bounds), "item {item}");
            let (k, x, y) = location.coord();
            let entry = match tree.collections.get(k) {
                Some(c) => &c.cells[y * c.cols + x][location.index as usize],
                None => &tree.overflow[location.index as usize],
            };
            assert_eq!((&entry.bounds, entry.item, entry.handle), (bounds, *item, *handle));
        }
        let stored: usize = tree.collections.iter().map(|c| c.count).sum::<usize>() + tree.overflow.len();
        assert_eq!(stored, live.len());
        for c in &tree.collections {
            assert_eq!(c.count, c.cells.iter().map(Vec::len).sum::<usize>());
        }
    }

    fn assert_searches(tree: &Tree, live: &[(ItemHandle, Rectangle, usize)], rng: &mut StdRng, cells: &[f64]) {
        let world = tree.get_bounds().clone();
        for round in 0..200 {
            let query = if round % 2 == 0 || live.is_empty() {
                random_bounds(rng, &world, cells)
            } else {
                let (_, b, _) = &live[rng.gen_range(0..live.len())];
                let (w, h) = (b.get_width() * 2.0, b.get_height() * 2.0);
                match rng.gen_range(0..4) {
                    0 => Rectangle::new(b.max_x.next_down(), b.min_y, b.max_x + w, b.max_y),
                    1 => Rectangle::new(b.min_x - w, b.min_y, b.min_x.next_up(), b.max_y),
                    2 => Rectangle::new(b.min_x, b.max_y.next_down(), b.max_x, b.max_y + h),
                    _ => Rectangle::new(b.min_x, b.min_y - h, b.max_x, b.min_y.next_up()),
                }
            };
            let mut expected: Vec<usize> = live
                .iter()
                .filter(|(_, bounds, _)| bounds.intersects(&query))
                .map(|&(_, _, item)| item)
                .collect();
            expected.sort_unstable();
            let mut found = Vec::new();
            tree.search(&query, |item| found.push(item));
            found.sort_unstable();
            assert_eq!(found, expected, "query {query:?}");
        }
    }

    /// Random insertions, updates within and across cells and collections,
    /// and removals keep every entry where `position` puts it and every
    /// search exact, each item reported once.
    #[test]
    fn random_operations_match_brute_force() {
        let mut rng = StdRng::seed_from_u64(0);
        for world in WORLDS {
            let side = world.get_width().min(world.get_height());
            for cells in [vec![side / 64.0, side / 16.0, side / 4.0], vec![side / 32.0, side / 4.0]] {
                let mut tree = Tree::new(world.clone(), &cells);
                let mut live: Vec<(ItemHandle, Rectangle, usize)> = Vec::new();
                for step in 0..3000 {
                    match rng.gen_range(0..5) {
                        0 | 1 => {
                            let bounds = random_bounds(&mut rng, &world, &cells);
                            live.push((tree.insert(bounds.clone(), step), bounds, step));
                        }
                        2 | 3 if !live.is_empty() => {
                            let index = rng.gen_range(0..live.len());
                            let old = &live[index].1;
                            let bounds = if rng.gen_bool(0.5) {
                                let (x, y) = old.get_center();
                                let step = old.get_width() * rng.gen_range(-0.3..0.3);
                                Rectangle::center_rect(x + step, y - step, old.get_width(), old.get_height())
                            } else {
                                random_bounds(&mut rng, &world, &cells)
                            };
                            tree.update(bounds.clone(), live[index].0);
                            live[index].1 = bounds;
                        }
                        _ if !live.is_empty() => {
                            let (handle, _, _) = live.swap_remove(rng.gen_range(0..live.len()));
                            tree.remove(handle);
                        }
                        _ => {}
                    }
                    if step % 250 == 0 {
                        assert_layout(&tree, &live);
                        assert_searches(&tree, &live, &mut rng, &cells);
                    }
                }
                assert_layout(&tree, &live);
                assert_searches(&tree, &live, &mut rng, &cells);
            }
        }
    }

    /// Items go to the first collection whose cell fits their larger side,
    /// and to the overflow list when none does or their center lies outside
    /// the world.
    #[test]
    fn items_go_to_the_smallest_fitting_collection() {
        let tree = Tree::new(Rectangle::new(0.0, 0.0, 1000.0, 1000.0), &[10.0, 100.0]);
        assert_eq!(tree.position(&Rectangle::center_rect(15.0, 25.0, 10.0, 4.0)), (0, 1, 2));
        assert_eq!(tree.position(&Rectangle::center_rect(15.0, 25.0, 4.0, 10.5)), (1, 0, 0));
        assert_eq!(tree.position(&Rectangle::center_rect(15.0, 25.0, 100.5, 4.0)).0, 2);
        assert_eq!(tree.position(&Rectangle::center_rect(-1.0, 25.0, 4.0, 4.0)).0, 2);
    }
}
