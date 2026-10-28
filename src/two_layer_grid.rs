//! Two-layer partitioning (Tsitsigkos, Bouros, Mamoulis et al., ICDE 2021,
//! extended in IEEE TKDE): a single-level grid that registers every object in
//! each cell its bounds overlap, like [`UniformGrid`], and splits the entries
//! of every cell into four classes by whether the object starts in the cell
//! or before it, along x and along y:
//!
//! - A: starts in the cell along both axes;
//! - B: starts in the cell along x, before it along y;
//! - C: starts before the cell along x, in it along y;
//! - D: starts before the cell along both axes.
//!
//! A search reads only the classes of objects that cannot have been met in an
//! earlier cell of the query range: all four in the query's first cell, A and
//! B in the rest of its first row, A and C in the rest of its first column,
//! and A elsewhere. It thus reports every object exactly once without a
//! deduplication test. It also tests only the sides that a cell's position in
//! the query range leaves open (Lemmas 3 and 4 of the paper): the query's
//! minimum along an axis only in its first column or row, its maximum only in
//! its last, and the latter only for objects that start in the cell. Entries
//! of cells strictly inside the query range are reported untested.
//!
//! Entries hold the bounds, as in the paper, so tests need no lookup; an
//! update that keeps the object's cell range rewrites the bounds in every cell
//! of the range, and one that changes it moves the entries.
//!
//! [`UniformGrid`]: crate::UniformGrid

use crate::cell_order::for_each_cell_in_morton_order;
use crate::handle::{Entry, ItemHandle, Slots};
use crate::rect::Rectangle;

/// Inclusive range of cells covered by an object or a query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct CellRange {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

/// Index into `TwoLayerCell::classes`: bit 1 is set for objects that start
/// before the cell along x, bit 0 for those that start before it along y.
const A: usize = 0b00;
const B: usize = 0b01;
const C: usize = 0b10;
const D: usize = 0b11;

/// Class of the entries in cell `(x, y)` of an object covering `range`.
#[inline]
fn class_of(range: CellRange, x: u32, y: u32) -> usize {
    ((x > range.min_x) as usize) << 1 | (y > range.min_y) as usize
}

struct TwoLayerCell<T: Copy> {
    /// Entries by class, indexed by [`A`], [`B`], [`C`] and [`D`].
    classes: [Vec<Entry<T>>; 4],
}

impl<T: Copy> TwoLayerCell<T> {
    fn new() -> Self {
        Self {
            classes: Default::default(),
        }
    }
}

pub struct TwoLayerGrid<T: Copy> {
    origin_x: f64,
    origin_y: f64,
    cell_size: f64,
    /// Multiplying by the reciprocal is cheaper than dividing and, being
    /// monotone, keeps object and query ranges mutually consistent, which the
    /// skipped tests rely on.
    inverse_cell_size: f64,
    cols: usize,
    rows: usize,
    /// Cells in row-major order.
    cells: Vec<TwoLayerCell<T>>,
    /// Cell range and item of each live handle. The item is kept here as
    /// well, so that bounds covering no cell keep it.
    slots: Slots<(CellRange, T)>,
}

impl<T: Copy> TwoLayerGrid<T> {
    /// Grid of square cells covering `world_bounds`. Coordinates outside the
    /// world clamp to the border cells.
    pub fn new(world_bounds: Rectangle, cell_size: f64) -> Self {
        assert!(
            cell_size.is_finite() && cell_size > 0.0,
            "Cell size must be positive."
        );
        let cols = ((world_bounds.get_width() / cell_size).ceil() as usize).max(1);
        let rows = ((world_bounds.get_height() / cell_size).ceil() as usize).max(1);
        assert!(
            cols <= u32::MAX as usize && rows <= u32::MAX as usize,
            "Too many cells per axis for the u32 cell coordinates."
        );
        Self {
            origin_x: world_bounds.min_x,
            origin_y: world_bounds.min_y,
            cell_size,
            inverse_cell_size: 1.0 / cell_size,
            cols,
            rows,
            cells: (0..cols * rows).map(|_| TwoLayerCell::new()).collect(),
            slots: Slots::new(),
        }
    }

    pub fn cell_size(&self) -> f64 {
        self.cell_size
    }

    /// Area covered by the cells, from the world's min corner over whole cells.
    pub fn get_bounds(&self) -> Rectangle {
        Rectangle::new(
            self.origin_x,
            self.origin_y,
            self.origin_x + self.cols as f64 * self.cell_size,
            self.origin_y + self.rows as f64 * self.cell_size,
        )
    }

    /// Cell containing a point. The cast saturates, so coordinates below the
    /// origin land in cell zero; `min` clamps the far side. The result never
    /// decreases as a coordinate grows.
    #[inline]
    fn cell_of(&self, x: f64, y: f64) -> (u32, u32) {
        let cx = (((x - self.origin_x) * self.inverse_cell_size) as usize).min(self.cols - 1);
        let cy = (((y - self.origin_y) * self.inverse_cell_size) as usize).min(self.rows - 1);
        (cx as u32, cy as u32)
    }

    #[inline]
    fn range_of(&self, bounds: &Rectangle) -> CellRange {
        let (min_x, min_y) = self.cell_of(bounds.min_x, bounds.min_y);
        let (max_x, max_y) = self.cell_of(bounds.max_x, bounds.max_y);
        CellRange {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    #[inline]
    fn cell_mut(&mut self, x: u32, y: u32) -> &mut TwoLayerCell<T> {
        &mut self.cells[y as usize * self.cols + x as usize]
    }

    /// Insert an item and return the handle that identifies it for
    /// [`update`](Self::update) and [`remove`](Self::remove).
    ///
    /// Bounds inverted along an axis by a cell or more cover no cell and are
    /// stored nowhere; they intersect no query either.
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let range = self.range_of(&bounds);
        let handle = self.slots.allocate((range, item));
        self.attach(range, &bounds, item, handle);
        handle
    }

    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let (range, _) = self.slots.get(handle);
        self.detach(range, handle);
        self.slots.free(handle);
    }

    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let (old_range, item) = self.slots.get(handle);
        let new_range = self.range_of(&bounds);
        if new_range == old_range {
            for y in old_range.min_y..=old_range.max_y {
                for x in old_range.min_x..=old_range.max_x {
                    let class = class_of(old_range, x, y);
                    let entries = &mut self.cell_mut(x, y).classes[class];
                    let entry = entries
                        .iter_mut()
                        .find(|entry| entry.handle == handle)
                        .expect("Object missing from its cell.");
                    entry.bounds = bounds.clone();
                }
            }
            return;
        }
        self.detach(old_range, handle);
        self.slots.get_mut(handle).0 = new_range;
        self.attach(new_range, &bounds, item, handle);
    }

    fn attach(&mut self, range: CellRange, bounds: &Rectangle, item: T, handle: ItemHandle) {
        for y in range.min_y..=range.max_y {
            for x in range.min_x..=range.max_x {
                let class = class_of(range, x, y);
                self.cell_mut(x, y).classes[class].push(Entry {
                    bounds: bounds.clone(),
                    item,
                    handle,
                });
            }
        }
    }

    /// Remove the entries of `handle` from every cell of `range`.
    fn detach(&mut self, range: CellRange, handle: ItemHandle) {
        for y in range.min_y..=range.max_y {
            for x in range.min_x..=range.max_x {
                let class = class_of(range, x, y);
                let entries = &mut self.cell_mut(x, y).classes[class];
                let position = entries
                    .iter()
                    .position(|entry| entry.handle == handle)
                    .expect("Object missing from its cell.");
                entries.swap_remove(position);
            }
        }
    }

    /// Search with the bounds of every stored item, as [`search`](Self::search)
    /// would item by item, and call `callback` with the item searched for and
    /// each item found, the searched one included. The objects are taken cell
    /// by cell in Morton order, each from class A of the first cell of its
    /// range, so that consecutive searches read nearby cells. It visits every
    /// cell, so where cells far outnumber objects, searching item by item is
    /// faster.
    pub fn for_each_self_query(&self, mut callback: impl FnMut(T, T)) {
        let cols = self.cols;
        for_each_cell_in_morton_order(cols, self.rows, |x, y| {
            for entry in &self.cells[y * cols + x].classes[A] {
                self.search(&entry.bounds, |found| callback(entry.item, found));
            }
        });
    }

    /// Call `callback` once for every pair of distinct stored items whose
    /// bounds intersect, with the two items in either order. A pair meets
    /// first in the cell where neither both start before it along x nor both
    /// along y, so of the class pairs within a cell only A-A, A-B, A-C, A-D
    /// and B-C are joined, as in the paper's spatial joins, and every pair is
    /// tested once without a duplicate check. The cells are visited in Morton
    /// order.
    pub fn for_each_pair(&self, mut callback: impl FnMut(T, T)) {
        let cols = self.cols;
        for_each_cell_in_morton_order(cols, self.rows, |x, y| {
            let [a, b, c, d] = &self.cells[y * cols + x].classes;
            for (index, entry) in a.iter().enumerate() {
                for other in &a[index + 1..] {
                    if entry.bounds.intersects(&other.bounds) {
                        callback(entry.item, other.item);
                    }
                }
            }
            let mut join = |left: &[Entry<T>], right: &[Entry<T>]| {
                for entry in left {
                    for other in right {
                        if entry.bounds.intersects(&other.bounds) {
                            callback(entry.item, other.item);
                        }
                    }
                }
            };
            join(a, b);
            join(a, c);
            join(a, d);
            join(b, c);
        });
    }

    /// Call `callback` once for every item whose bounds intersect `bounds`.
    ///
    /// An item with a NaN coordinate intersects no query, but if its range
    /// covers cells strictly inside the query range, it may be reported from
    /// there untested.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        // A query with a NaN corner meets nothing; with a NaN minimum its
        // range would start at the first cell and report the cells inside it
        // untested.
        if bounds.has_nan() {
            return;
        }
        let range = self.range_of(bounds);
        for y in range.min_y..=range.max_y {
            let first_row = y == range.min_y;
            let last_row = y == range.max_y;
            let row = y as usize * self.cols;
            for x in range.min_x..=range.max_x {
                let first_column = x == range.min_x;
                let last_column = x == range.max_x;
                let classes = &self.cells[row + x as usize].classes;
                // Objects starting before the cell along an axis were met in
                // an earlier cell unless this is the query's first along it.
                // Whether the query's minimum needs a test depends only on the
                // cell; its maximum needs none for objects starting before the
                // cell, which start before the query's last cell.
                let sides = |before_x: bool, before_y: bool| Sides {
                    min_x: first_column,
                    max_x: last_column && !before_x,
                    min_y: first_row,
                    max_y: last_row && !before_y,
                };
                report(&classes[A], bounds, sides(false, false), &mut callback);
                if first_row {
                    report(&classes[B], bounds, sides(false, true), &mut callback);
                }
                if first_column {
                    report(&classes[C], bounds, sides(true, false), &mut callback);
                    if first_row {
                        report(&classes[D], bounds, sides(true, true), &mut callback);
                    }
                }
            }
        }
    }
}

/// Which sides of the query an entry must be tested against: `min_x` tests
/// `query.min_x < entry.max_x`, `max_x` tests `entry.min_x < query.max_x`,
/// and likewise in y.
///
/// A side needs no test where the cell ranges already decide it: `cell_of`
/// never decreases, so an object whose range ends in a later cell than the
/// query's minimum lies past that minimum, and one whose range starts in an
/// earlier cell than the query's maximum starts before that maximum, both
/// strictly, as [`Rectangle::intersects`] requires.
#[derive(Clone, Copy)]
struct Sides {
    min_x: bool,
    max_x: bool,
    min_y: bool,
    max_y: bool,
}

/// Call `callback` for every entry of `entries` that passes the tests of
/// `sides` against `query`. A side without a test compares with an infinite
/// limit, which every entry with that side finite passes, so one loop serves
/// every combination except none at all, which skips the comparisons.
#[inline]
fn report<T: Copy>(
    entries: &[Entry<T>],
    query: &Rectangle,
    sides: Sides,
    callback: &mut impl FnMut(T),
) {
    if !(sides.min_x | sides.max_x | sides.min_y | sides.max_y) {
        entries.iter().for_each(|entry| callback(entry.item));
        return;
    }
    let pick = |test: bool, limit: f64, open: f64| if test { limit } else { open };
    let min_x = pick(sides.min_x, query.min_x, f64::NEG_INFINITY);
    let max_x = pick(sides.max_x, query.max_x, f64::INFINITY);
    let min_y = pick(sides.min_y, query.min_y, f64::NEG_INFINITY);
    let max_y = pick(sides.max_y, query.max_y, f64::INFINITY);
    entries
        .iter()
        .filter(|entry| {
            let b = &entry.bounds;
            (min_x < b.max_x) & (b.min_x < max_x) & (min_y < b.max_y) & (b.min_y < max_y)
        })
        .for_each(|entry| callback(entry.item));
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    type Tree = TwoLayerGrid<usize>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    /// Bounds from a fraction of a cell to several cells per side, centered
    /// anywhere in the world grown by a cell, so some extend past it; half of
    /// them start or end on a cell edge, give or take a float.
    fn random_bounds(rng: &mut StdRng, world: &Rectangle, cell: f64) -> Rectangle {
        let side = |rng: &mut StdRng| cell * 2f64.powf(rng.gen_range(-3.0..2.5));
        let (width, height) = (side(rng), side(rng));
        let (x, y) = world.scale(1.0 + 2.0 * cell / world.get_width().min(world.get_height()))
            .get_random_point(rng);
        let bounds = Rectangle::center_rect(x, y, width, height);
        if rng.gen_bool(0.5) {
            let mut snapped = bounds.clone();
            let snap = |v: f64, origin: f64, rng: &mut StdRng| {
                let edge = origin + ((v - origin) / cell).round() * cell;
                match rng.gen_range(0..3) {
                    0 => edge,
                    1 => edge.next_up(),
                    _ => edge.next_down(),
                }
            };
            snapped.min_x = snap(bounds.min_x, world.min_x, rng);
            snapped.max_y = snap(bounds.max_y, world.min_y, rng);
            // Snapping a narrow box can invert it, which no grid supports.
            if snapped.min_x < snapped.max_x && snapped.min_y < snapped.max_y {
                return snapped;
            }
        }
        bounds
    }

    /// Every entry must sit in a cell of its object's range, in the class its
    /// position there gives, once per cell, with the object's current bounds.
    fn assert_layout(tree: &Tree, live: &[(ItemHandle, Rectangle, usize)]) {
        let mut entries = 0;
        for (handle, bounds, item) in live {
            let (range, stored_item) = tree.slots.get(*handle);
            assert_eq!((range, stored_item), (tree.range_of(bounds), *item));
            for y in range.min_y..=range.max_y {
                for x in range.min_x..=range.max_x {
                    let cell = &tree.cells[y as usize * tree.cols + x as usize];
                    for (class, list) in cell.classes.iter().enumerate() {
                        let count = list.iter().filter(|e| e.handle == *handle).count();
                        let expected = (class == class_of(range, x, y)) as usize;
                        assert_eq!(count, expected, "item {item}, cell ({x}, {y}), class {class}");
                    }
                    entries += 1;
                }
            }
            // The first cell holds the entry in class A.
            let cell = &tree.cells[range.min_y as usize * tree.cols + range.min_x as usize];
            let entry = cell.classes[A].iter().find(|e| e.handle == *handle).unwrap();
            assert_eq!((&entry.bounds, entry.item), (bounds, *item));
        }
        let stored: usize = tree
            .cells
            .iter()
            .map(|cell| cell.classes.iter().map(Vec::len).sum::<usize>())
            .sum();
        assert_eq!(stored, entries);
    }

    /// Queries of every size, half of them touching a live item by a float.
    fn assert_searches(tree: &Tree, live: &[(ItemHandle, Rectangle, usize)], rng: &mut StdRng) {
        let world = tree.get_bounds();
        for round in 0..200 {
            let query = if round % 2 == 0 || live.is_empty() {
                random_bounds(rng, &world, tree.cell_size() * 4.0)
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

    /// Random insertions, updates within and across cell ranges and removals
    /// keep every entry in its class and every search exact, each item
    /// reported once.
    #[test]
    fn random_operations_match_brute_force() {
        let mut rng = StdRng::seed_from_u64(0);
        for world in WORLDS {
            for cell in [world.get_width() / 16.0, world.get_width() / 61.0] {
                let mut tree = Tree::new(world.clone(), cell);
                let mut live: Vec<(ItemHandle, Rectangle, usize)> = Vec::new();
                for step in 0..3000 {
                    match rng.gen_range(0..5) {
                        0 | 1 => {
                            let bounds = random_bounds(&mut rng, &world, cell);
                            live.push((tree.insert(bounds.clone(), step), bounds, step));
                        }
                        2 | 3 if !live.is_empty() => {
                            let index = rng.gen_range(0..live.len());
                            let old = &live[index].1;
                            let bounds = if rng.gen_bool(0.5) {
                                let (x, y) = old.get_center();
                                let step = cell * rng.gen_range(-0.3..0.3);
                                Rectangle::center_rect(
                                    x + step,
                                    y - step,
                                    old.get_width(),
                                    old.get_height(),
                                )
                            } else {
                                random_bounds(&mut rng, &world, cell)
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
                        assert_searches(&tree, &live, &mut rng);
                    }
                }
                assert_layout(&tree, &live);
                assert_searches(&tree, &live, &mut rng);
            }
        }
    }

    /// Bounds inverted by a cell or more cover no cell, are found by no query
    /// and keep their item through updates and removal.
    #[test]
    fn inverted_bounds_are_stored_nowhere() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 100.0, 100.0), 10.0);
        let handle = tree.insert(Rectangle::new(55.0, 15.0, 35.0, 25.0), 7);
        assert!(tree.cells.iter().all(|cell| cell.classes.iter().all(Vec::is_empty)));
        let mut found = Vec::new();
        tree.search(&Rectangle::new(0.0, 0.0, 100.0, 100.0), |item| found.push(item));
        assert!(found.is_empty());
        tree.update(Rectangle::new(35.0, 15.0, 55.0, 25.0), handle);
        tree.search(&Rectangle::new(0.0, 0.0, 100.0, 100.0), |item| found.push(item));
        assert_eq!(found, [7]);
        tree.remove(handle);
        assert!(tree.cells.iter().all(|cell| cell.classes.iter().all(Vec::is_empty)));
    }

    /// A query spanning several cells per axis finds an object covering all
    /// of them once, and one strictly inside its interior cells untested.
    #[test]
    fn large_object_is_reported_once() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 100.0, 100.0), 10.0);
        tree.insert(Rectangle::new(5.0, 5.0, 95.0, 95.0), 0);
        tree.insert(Rectangle::new(41.0, 41.0, 49.0, 49.0), 1);
        let mut found = Vec::new();
        tree.search(&Rectangle::new(25.0, 25.0, 55.0, 55.0), |item| found.push(item));
        found.sort_unstable();
        assert_eq!(found, [0, 1]);
    }
}
