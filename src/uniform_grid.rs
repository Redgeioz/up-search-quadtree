//! Single-level grid: every object is registered in each cell its bounds overlap.
//!
//! Objects are stored once, in a table indexed by the slot index of their
//! handle; cells hold 8-byte references into that table. An update that keeps
//! the object's cell range is therefore a single write; one that changes it
//! removes the reference from every cell of the old range, scanning each, and
//! appends one to every cell of the new range.

use crate::cell_order::for_each_cell_in_morton_order;
use crate::handle::{ItemHandle, Slots};
use crate::rect::Rectangle;

/// Inclusive range of cells covered by an object or a query.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
struct CellRange {
    min_x: u32,
    min_y: u32,
    max_x: u32,
    max_y: u32,
}

struct Object<T> {
    bounds: Rectangle,
    item: T,
}

/// Reference to an object from one of the cells it overlaps.
#[derive(Clone, Copy)]
struct Membership {
    /// Index into `UniformGrid::objects`.
    object: u32,
    /// First cell of the object's range. A search reports the object only
    /// from the first cell it shares with the query, which deduplicates
    /// without reading the object.
    first_x: u16,
    first_y: u16,
}

pub struct UniformGrid<T: Copy> {
    origin_x: f64,
    origin_y: f64,
    cell_size: f64,
    /// Multiplying by the reciprocal is cheaper than dividing and, being
    /// monotone, keeps object and query ranges mutually consistent.
    inverse_cell_size: f64,
    cols: usize,
    rows: usize,
    /// Cells in row-major order.
    cells: Vec<Vec<Membership>>,
    /// Indexed by the slot index of the handle. The object of a removed
    /// handle stays behind, unreferenced, until its slot is reused.
    objects: Vec<Object<T>>,
    /// Cell range of each live handle.
    slots: Slots<CellRange>,
}

impl<T: Copy> UniformGrid<T> {
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
            cols <= u16::MAX as usize + 1 && rows <= u16::MAX as usize + 1,
            "Too many cells per axis for the u16 cell coordinates of `Membership`."
        );
        Self {
            origin_x: world_bounds.min_x,
            origin_y: world_bounds.min_y,
            cell_size,
            inverse_cell_size: 1.0 / cell_size,
            cols,
            rows,
            cells: (0..cols * rows).map(|_| Vec::new()).collect(),
            objects: Vec::new(),
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
    /// origin land in cell zero; `min` clamps the far side.
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

    /// Cells of row `y` within the columns of `range`.
    #[inline]
    fn row_mut(&mut self, y: u32, range: CellRange) -> &mut [Vec<Membership>] {
        let row = y as usize * self.cols;
        &mut self.cells[row + range.min_x as usize..=row + range.max_x as usize]
    }

    /// Insert an item and return the handle that identifies it for
    /// [`update`](Self::update) and [`remove`](Self::remove).
    ///
    /// Panics if the max x of `bounds` falls two or more columns before its
    /// min x while its rows are in order, as for bounds inverted by more
    /// than a cell or with a NaN max x, which lands in the first column.
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let range = self.range_of(&bounds);
        let handle = self.slots.allocate(range);
        let index = handle.index();
        let object = Object { bounds, item };
        if index == self.objects.len() {
            self.objects.push(object);
        } else {
            self.objects[index] = object;
        }
        self.attach(index, range);
        handle
    }

    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let range = self.slots.get(handle);
        self.detach(handle.index(), range);
        self.slots.free(handle);
    }

    /// Panics if the handle has already been removed.
    ///
    /// Panics if the max x of `bounds` falls two or more columns before its
    /// min x while its rows are in order, as for bounds inverted by more
    /// than a cell or with a NaN max x, which lands in the first column.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let old_range = self.slots.get(handle);
        let new_range = self.range_of(&bounds);
        let index = handle.index();
        self.objects[index].bounds = bounds;
        if new_range != old_range {
            self.detach(index, old_range);
            *self.slots.get_mut(handle) = new_range;
            self.attach(index, new_range);
        }
    }

    fn attach(&mut self, index: usize, range: CellRange) {
        let membership = Membership {
            object: index as u32,
            first_x: range.min_x as u16,
            first_y: range.min_y as u16,
        };
        for y in range.min_y..=range.max_y {
            for cell in self.row_mut(y, range) {
                cell.push(membership);
            }
        }
    }

    fn detach(&mut self, index: usize, range: CellRange) {
        let object = index as u32;
        for y in range.min_y..=range.max_y {
            for cell in self.row_mut(y, range) {
                let position = cell
                    .iter()
                    .position(|membership| membership.object == object)
                    .expect("Object missing from its cell.");
                cell.swap_remove(position);
            }
        }
    }

    /// Search with the bounds of every stored item, as [`search`](Self::search)
    /// would item by item, and call `callback` with the item searched for and
    /// each item found, the searched one included. The objects are taken cell
    /// by cell in Morton order, so that consecutive searches read nearby
    /// cells, each from the first cell of its range only. It visits every
    /// cell, so where cells far outnumber objects, searching item by item is
    /// faster. Each search first reads its object's bounds from the object
    /// table, which is in slot order, so the batch keeps only part of the gain
    /// of the cell order. A one-off probe outside the repository, on balls
    /// that all fit a cell, measured it 7% faster than searching item by
    /// item in random order with 10,000 objects, where searching in Morton
    /// order gains 14%, and 25% faster with 640,000 at the same density.
    pub fn for_each_self_query(&self, mut callback: impl FnMut(T, T)) {
        let cols = self.cols;
        for_each_cell_in_morton_order(cols, self.rows, |x, y| {
            for membership in &self.cells[y * cols + x] {
                if membership.first_x as usize == x && membership.first_y as usize == y {
                    let object = &self.objects[membership.object as usize];
                    self.search(&object.bounds, |found| callback(object.item, found));
                }
            }
        });
    }

    /// Call `callback` once for every pair of distinct stored items whose
    /// bounds intersect, with the two items in either order. Two items share
    /// every cell where their ranges overlap; the pair is tested only in the
    /// first of those, as in reference-point duplicate avoidance, so the
    /// first-cell check on the memberships precedes reading the objects. The
    /// cells are visited in Morton order.
    pub fn for_each_pair(&self, mut callback: impl FnMut(T, T)) {
        let cols = self.cols;
        for_each_cell_in_morton_order(cols, self.rows, |x, y| {
            let cell = &self.cells[y * cols + x];
            for (index, a) in cell.iter().enumerate() {
                let object = &self.objects[a.object as usize];
                for b in &cell[index + 1..] {
                    let first_common = a.first_x.max(b.first_x) as usize == x
                        && a.first_y.max(b.first_y) as usize == y;
                    if first_common {
                        let other = &self.objects[b.object as usize];
                        if object.bounds.intersects(&other.bounds) {
                            callback(object.item, other.item);
                        }
                    }
                }
            }
        });
    }

    /// Call `callback` once for every item whose bounds intersect `bounds`.
    ///
    /// Panics if the max x of `bounds` falls two or more columns before its
    /// min x while its rows are in order, as for bounds inverted by more
    /// than a cell.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        // A query with a NaN corner meets nothing; a NaN max x would land
        // in the first column and could leave the range of a row inverted.
        if bounds.has_nan() {
            return;
        }
        let range = self.range_of(bounds);
        for y in range.min_y..=range.max_y {
            let row = y as usize * self.cols;
            let cells = &self.cells[row + range.min_x as usize..=row + range.max_x as usize];
            for (x, cell) in (range.min_x..).zip(cells) {
                for membership in cell {
                    // An object in several scanned cells is reported only from
                    // the first cell it shares with the query.
                    let is_first_common_cell = (membership.first_x as u32).max(range.min_x) == x
                        && (membership.first_y as u32).max(range.min_y) == y;
                    if is_first_common_cell {
                        let object = &self.objects[membership.object as usize];
                        if object.bounds.intersects(bounds) {
                            callback(object.item);
                        }
                    }
                }
            }
        }
    }
}
