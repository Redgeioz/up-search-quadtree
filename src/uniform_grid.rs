//! Single-level grid: every object is registered in each cell its bounds overlap.
//!
//! Objects are stored once, in a table indexed by the slot index of their
//! handle; cells hold 8-byte references to that table. Moving an object within
//! the cells it already covers is therefore a single write, and moving it to
//! other cells scans only the cells it leaves.

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
    /// from the first cell that it and the query have in common, which
    /// deduplicates without touching the object.
    first_x: u16,
    first_y: u16,
}

pub struct UniformGrid<T: Copy> {
    origin_x: f64,
    origin_y: f64,
    cell_size: f64,
    /// Multiplying by the reciprocal is cheaper than dividing and, being
    /// monotone, keeps object and query ranges consistent with each other.
    inverse_cell_size: f64,
    cols: usize,
    rows: usize,
    /// Row-major cells.
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

    /// Cell containing a point. Casting saturates, so coordinates before the
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

    /// Call `callback` once for every item whose bounds intersect `bounds`.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        let range = self.range_of(bounds);
        for y in range.min_y..=range.max_y {
            let row = y as usize * self.cols;
            let cells = &self.cells[row + range.min_x as usize..=row + range.max_x as usize];
            for (x, cell) in (range.min_x..).zip(cells) {
                for membership in cell {
                    // An object shared by several scanned cells is reported from
                    // the first cell that it and the query have in common.
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
