//! Constants and the common interface of the structures, shared by the
//! benchmarks.

use lib::rect::*;
use lib::*;

pub const MAX_LEVEL: u8 = 7;
pub const MAX_ITEMS: usize = 8;

pub const BOUND_WIDTH: f64 = 20000.0;
pub const BOUND_HEIGHT: f64 = 20000.0;

pub const AMOUNT: usize = 10000;
pub const SEED: u64 = 0;

// Diameters of 50 to 100, the size of levels 8 to 9; with `MAX_LEVEL` 7,
// balls that do not double sit on the finest level.
pub const MIN_RADIUS: f64 = 25.0;
pub const MAX_RADIUS: f64 = 50.0;

pub const BOUNDS: Rectangle = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);

/// Items of the large scenes: sixteen times [`AMOUNT`] at the same density,
/// in a world four times as wide and high.
#[allow(dead_code, reason = "used by some benchmarks only")]
pub const LARGE_AMOUNT: usize = 16 * AMOUNT;
#[allow(dead_code, reason = "used by some benchmarks only")]
pub const LARGE_BOUNDS: Rectangle = Rectangle::new(0.0, 0.0, 4.0 * BOUND_WIDTH, 4.0 * BOUND_HEIGHT);
/// Levels of the hierarchical structures in the large scenes: two more, so
/// that their finest cell stays [`GRID_CELL_SIZE`].
#[allow(dead_code, reason = "used by some benchmarks only")]
pub const LARGE_MAX_LEVEL: u8 = MAX_LEVEL + 2;

/// Cell size of the uniform grids: the finest cell size of the hierarchical
/// structures, 312.5.
pub const GRID_CELL_SIZE: f64 = BOUND_WIDTH / (1u32 << (MAX_LEVEL - 1)) as f64;

/// Cell sizes of the `MultiGrid` collections: three log-spaced size classes
/// covering the ball diameters of the benches, 50 to 3200, as two or three
/// log-spaced collections were optimal in LAMMPS.
pub const MULTI_CELL_SIZES: [f64; 3] = [200.0, 800.0, 3200.0];
/// Two log-spaced collections over the same diameters.
#[allow(dead_code, reason = "used by some benchmarks only")]
pub const MULTI_CELL_SIZES_2: [f64; 2] = [400.0, 3200.0];

/// Common interface to the item management of every structure.
pub trait QTree<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle;
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle);
    #[allow(dead_code, reason = "unused by the search benchmarks")]
    fn remove_item(&mut self, handle: ItemHandle);
    /// Called after a scene is built and after each round of updates, like
    /// the end of a frame, so that lazy structures bring their bounds up to
    /// date.
    fn end_frame(&mut self) {}
    /// Pass every item whose bounds meet `bounds` to `callback`.
    #[allow(dead_code, reason = "unused by the update benchmarks")]
    fn search_items(&self, bounds: &Rectangle, callback: impl FnMut(T));
}

/// Implements [`QTree`] for `$tree`, whose parameters after the item type are
/// `$param`; given `$end_frame`, `end_frame` calls that method.
macro_rules! qtree {
    ($tree:ident<T $(, const $param:ident: $kind:ty)*> $(, $end_frame:ident)?) => {
        impl<T: Copy $(, const $param: $kind)*> QTree<T> for $tree<T $(, $param)*> {
            fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
                self.insert(bounds, item)
            }
            fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
                self.update(bounds, handle);
            }
            fn remove_item(&mut self, handle: ItemHandle) {
                self.remove(handle);
            }
            $(
                fn end_frame(&mut self) {
                    self.$end_frame();
                }
            )?
            // Copied into each caller, like a closure of the benchmark's own,
            // so that the timed loops can inline it.
            #[inline]
            fn search_items(&self, bounds: &Rectangle, callback: impl FnMut(T)) {
                self.search(bounds, callback);
            }
        }
    };
}

qtree!(LooseQuadTree<T, const MAX_LEVEL: u8, const MAX_ITEMS: usize>);
qtree!(HierarchicalLooseGrid<T, const MAX_LEVEL: u8>);
qtree!(ExactOverflowBoundGrid<T, const MAX_LEVEL: u8>);
qtree!(BoxedLooseGrid<T, const MAX_LEVEL: u8>, refresh_bounds);
qtree!(LazyOverflowBoundGrid<T, const MAX_LEVEL: u8>, refresh_bounds);
qtree!(SpanOverflowBoundGrid<T, const MAX_LEVEL: u8>, refresh_bounds);
qtree!(HaloOverflowBoundGrid<T, const MAX_LEVEL: u8>, refresh_bounds);
qtree!(UniformGrid<T>);
qtree!(TwoLayerGrid<T>);
qtree!(MultiGrid<T>);
qtree!(HierarchicalGrid<T, const MAX_LEVEL: u8>);
qtree!(SpanGrid<T>, refresh_bounds);
qtree!(HaloGrid<T>, refresh_bounds);
qtree!(SortAndSweep<T>, refresh_order);

/// [`SortAndSweep`] that never refreshes its order, so that every pair query
/// sorts a copy, as a sweep that keeps no order between frames would.
pub struct UnsortedSweep(pub SortAndSweep<usize>);

impl UnsortedSweep {
    #[allow(dead_code, reason = "unused by the update benchmarks")]
    pub fn for_each_pair(&self, callback: impl FnMut(usize, usize)) {
        self.0.for_each_pair(callback);
    }
}

impl QTree<usize> for UnsortedSweep {
    fn insert_item(&mut self, bounds: Rectangle, item: usize) -> ItemHandle {
        self.0.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.0.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.0.remove(handle);
    }
    #[inline]
    fn search_items(&self, bounds: &Rectangle, callback: impl FnMut(usize)) {
        self.0.search(bounds, callback);
    }
}

/// A new, empty `$tree` of `usize` items over [`BOUNDS`]: a hierarchical
/// structure with [`MAX_LEVEL`] levels, or, given a cell size, a single-layer
/// grid.
macro_rules! tree {
    (LooseQuadTree) => {
        LooseQuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone())
    };
    ($tree:ident) => {
        $tree::<usize, MAX_LEVEL>::new(BOUNDS.clone())
    };
    ($tree:ident, $cell:expr) => {
        $tree::<usize>::new(BOUNDS.clone(), $cell)
    };
}
pub(crate) use tree;
