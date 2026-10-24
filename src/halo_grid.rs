//! [`SpanGrid`] with the neighbor pruning of its 3 × 3 walk moved from the
//! column and row spans to the cells, as [`HaloOverflowBoundGrid`] does for
//! [`SpanOverflowBoundGrid`]: a single-level loose grid of chosen cell
//! size, for scenes whose items all fit a cell.
//!
//! Every item is stored in the cell that contains its center, so an item no
//! larger than a cell extends at most half a cell beyond it. Each cell keeps
//! a box covering its items and copies of the sides of its eight neighbors'
//! boxes that face it, that is, how far each neighbor's items reach towards
//! the cell. A query no larger than a cell reads the cell of its center and,
//! judged from that cell's copies alone, those of the eight neighbors whose
//! items may reach the query, instead of judging them by the span of their
//! whole column or row, which on a dense grid nearly always meets the query.
//! A neighbor that is read is still skipped when its box misses the query. A
//! larger query scans the cells within half a cell of it, dropping edge
//! columns and rows by their spans and cells by their boxes, as
//! [`SpanGrid`] does.
//!
//! Insertion and update only widen the box of the target cell and the spans
//! of its column and row, and mark the grid stale and the row touched; the
//! copies in the neighbors are not updated. Until the refresh, the walk
//! judges the neighbors of a query cell in or next to a touched row by the
//! spans, as [`SpanGrid`] does. Removal and moving out leave boxes, spans
//! and copies unchanged, except that an emptied cell resets its box, so all
//! three stay valid as items move, shrink or leave, and tighten only in
//! [`refresh_bounds`], which also brings the copies of the rows near touched
//! ones up to date. Items larger than a cell, degenerate (zero or NaN size)
//! or centered outside the world go to one overflow node that every search
//! reads.
//!
//! [`HaloGrid`]: crate::HaloGrid
//! [`HaloOverflowBoundGrid`]: crate::HaloOverflowBoundGrid
//! [`SpanOverflowBoundGrid`]: crate::SpanOverflowBoundGrid
//! [`SpanGrid`]: crate::SpanGrid
//! [`refresh_bounds`]: HaloGrid::refresh_bounds

use crate::cell_order::for_each_cell_in_morton_order;
use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::layer_geometry::{LayerGeometry, first_reaching_node, last_reaching_node};
use crate::rect::{EMPTY, NOWHERE, Rectangle};
use grid::*;

/// `level` of a [`GridLocation`] in the overflow node; cells are level 1.
const OVERFLOW: usize = 0;

pub struct HaloGrid<T: Copy> {
    cells: Grid<Node<T>>,
    geometry: LayerGeometry,
    /// Reciprocals of the cell width and height. Position and search multiply
    /// by them instead of dividing by the cell size, and `geometry` builds its
    /// cell edges with the same arithmetic.
    inverse_node_size: (f64, f64),
    lines: LineSpan,
    /// Copies of every cell's neighbors' facing box sides, in row-major order.
    /// Kept apart from the cells so that these stay small for the scan and
    /// for updates, which never touch the copies.
    halo: Box<[Halo]>,
    /// Items larger than a cell, degenerate or centered outside the world.
    overflow: Node<T>,
    overflow_touched: bool,
    /// Square of whole cells anchored at the world's min corner, from which
    /// the geometry is derived, as in the hierarchical grids.
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    slots: Slots<GridLocation>,
}

impl<T: Copy> HaloGrid<T> {
    /// Create a grid of cells of about `cell_size` covering `world_bounds`.
    pub fn new(world_bounds: Rectangle, cell_size: f64) -> Self {
        assert!(cell_size > 0.0 && cell_size.is_finite(), "invalid cell size");
        let cols = (world_bounds.get_width() / cell_size).ceil().max(1.0) as usize;
        let rows = (world_bounds.get_height() / cell_size).ceil().max(1.0) as usize;
        let side = cols.max(rows) as f64 * cell_size;
        let (min_x, min_y) = world_bounds.get_min();
        let root_bounds = Rectangle::new(min_x, min_y, min_x + side, min_y + side);
        let mut nodes = Vec::with_capacity(rows * cols);
        for _ in 0..rows * cols {
            nodes.push(Node::new());
        }
        let (node_width, node_height) = LayerGeometry::node_size(&root_bounds, cols, rows);
        let inverse_node_size = (1.0 / node_width, 1.0 / node_height);
        Self {
            cells: Grid::from_vec(nodes, cols),
            // Level 2, as every level but the root gets the update shortcut.
            geometry: LayerGeometry::with_reciprocal(
                &root_bounds,
                2,
                cols,
                rows,
                inverse_node_size.0,
                inverse_node_size.1,
            ),
            inverse_node_size,
            lines: LineSpan::new(cols, rows),
            halo: vec![NO_HALO; rows * cols].into_boxed_slice(),
            overflow: Node::new(),
            overflow_touched: false,
            root_bounds,
            world_bounds,
            slots: Slots::new(),
        }
    }

    /// Square of whole cells anchored at the world's min corner.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Size of a cell along each axis.
    pub fn cell_size(&self) -> (f64, f64) {
        (self.geometry.node_width, self.geometry.node_height)
    }

    /// Whether an item of this size belongs in a cell, not the overflow node.
    #[inline]
    fn fits_cell(&self, bounds: &Rectangle) -> bool {
        self.geometry
            .matches_layer(bounds.get_width(), bounds.get_height(), true)
    }

    /// Cell whose area contains `point`, clamped into the grid.
    #[inline]
    fn cell_containing(&self, point: (f64, f64)) -> (usize, usize) {
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let x = (((point.0 - origin_x) * self.inverse_node_size.0) as usize).min(self.cells.cols() - 1);
        let y = (((point.1 - origin_y) * self.inverse_node_size.1) as usize).min(self.cells.rows() - 1);
        (x, y)
    }

    /// `(level, x, y)` of the node an item with these bounds belongs in.
    pub fn position(&self, bounds: &Rectangle) -> (usize, usize, usize) {
        let center = bounds.get_center();
        if !self.fits_cell(bounds) || !self.world_bounds.contains_point(center) {
            return (OVERFLOW, 0, 0);
        }
        let (x, y) = self.cell_containing(center);
        (1, x, y)
    }

    /// Cheap check that `position` would still return `(level, x, y)` for
    /// `bounds`.
    fn stays_in_node(&self, bounds: &Rectangle, (level, x, y): (usize, usize, usize)) -> bool {
        if !self.geometry.has_update_shortcut() {
            return false;
        }
        let center = bounds.get_center();
        let in_a_cell = self.fits_cell(bounds) && self.world_bounds.contains_point(center);
        if level == OVERFLOW {
            return !in_a_cell;
        }
        let (origin_x, origin_y) = self.root_bounds.get_min();
        in_a_cell
            && self
                .geometry
                .matches_cell((center.0 - origin_x, center.1 - origin_y), (x, y))
    }

    #[inline]
    fn node_mut(&mut self, (level, x, y): (usize, usize, usize)) -> &mut Node<T> {
        if level == OVERFLOW {
            &mut self.overflow
        } else {
            &mut self.cells[y][x]
        }
    }

    /// Store `entry` in the node at `coord`, widening its box and, for a
    /// cell, the spans of its column and row.
    fn store(&mut self, coord: (usize, usize, usize), entry: Entry<T>) {
        let (level, x, y) = coord;
        if level == OVERFLOW {
            self.overflow_touched = true;
        } else {
            self.lines.grow((x, y), &entry.bounds);
        }
        let handle = entry.handle;
        let node = self.node_mut(coord);
        node.grow(&entry.bounds);
        let index = node.items.len();
        node.items.push(entry);
        *self.slots.get_mut(handle) = GridLocation::new(coord, index);
    }

    /// Insert an item and return the handle that identifies it for
    /// [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord = self.position(&bounds);
        let handle = self.slots.allocate(GridLocation::default());
        self.store(
            coord,
            Entry {
                bounds,
                item,
                handle,
            },
        );
        handle
    }

    /// Remove an item.
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let entry = self.take_entry(location.coord(), location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        self.slots.free(handle);
    }

    /// Update the bounds of an item and, if necessary, its node.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let prev = location.coord();
        let index = location.index as usize;
        if self.stays_in_node(&bounds, prev) {
            self.replace_bounds(prev, index, bounds);
            return;
        }
        let next = self.position(&bounds);
        if next == prev {
            self.replace_bounds(prev, index, bounds);
            return;
        }
        let entry = self.take_entry(prev, index);
        self.store(next, Entry { bounds, ..entry });
    }

    /// Overwrite the bounds of an entry that stays in its node, widening its
    /// box and, for a cell, the spans of its column and row.
    fn replace_bounds(&mut self, coord: (usize, usize, usize), index: usize, bounds: Rectangle) {
        let (level, x, y) = coord;
        if level == OVERFLOW {
            self.overflow_touched = true;
        } else {
            self.lines.grow((x, y), &bounds);
        }
        let node = self.node_mut(coord);
        node.grow(&bounds);
        node.items[index].bounds = bounds;
    }

    /// Take the entry at `index` out of the node at `coord`. The entry that
    /// fills the gap has its slot updated; an emptied node resets its box.
    fn take_entry(&mut self, coord: (usize, usize, usize), index: usize) -> Entry<T> {
        let (level, x, y) = coord;
        if level == OVERFLOW {
            self.overflow_touched = true;
        } else {
            self.lines.mark_taken(y);
        }
        let node = if level == OVERFLOW {
            &mut self.overflow
        } else {
            &mut self.cells[y][x]
        };
        let entry = node.items.swap_remove(index);
        if let Some(moved) = node.items.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        if node.items.is_empty() {
            node.bounds = EMPTY;
        }
        entry
    }

    /// Shrink the boxes of the touched rows to their items, rebuild the spans
    /// of those rows and of every column, pull the copies of every row in or
    /// next to a touched one, and shrink the overflow node's box if it was
    /// touched.
    ///
    /// A refreshed box is the union of its items, so the spans and copies
    /// follow from the boxes. Each cell pulls the sides it needs from its
    /// neighbors, one row behind the shrinking, rather than each cell pushing
    /// its sides to its neighbors, which would first require resetting every
    /// copy.
    pub fn refresh_bounds(&mut self) {
        if std::mem::take(&mut self.overflow_touched) {
            self.overflow.refresh();
        }
        if !std::mem::take(&mut self.lines.touched) {
            return;
        }
        let lines = &mut self.lines;
        let cells = &mut self.cells;
        let halo = &mut self.halo;
        let rows = cells.rows();
        // A column span runs across every row, so it is rebuilt from the
        // boxes of all rows; a row's own span and boxes are rebuilt only if
        // the row is touched.
        lines.columns.fill(NOWHERE);
        let shrink_row = |cells: &mut Grid<Node<T>>, lines: &mut LineSpan, y: usize| {
            if lines.row_touched[y] {
                lines.rows[y] = NOWHERE;
                for (x, node) in cells[y].iter_mut().enumerate() {
                    node.refresh();
                    if !node.items.is_empty() {
                        lines.widen((x, y), &node.bounds);
                    }
                }
            } else {
                for (x, node) in cells[y].iter().enumerate() {
                    if !node.items.is_empty() {
                        lines.widen_column(x, &node.bounds);
                    }
                }
            }
        };
        // The copies of a row read the boxes of that row and the rows on
        // both sides, so they are pulled one row behind the shrinking, and
        // only if one of these three rows is touched.
        shrink_row(cells, lines, 0);
        for y in 0..rows {
            if y + 1 < rows {
                shrink_row(cells, lines, y + 1);
            }
            if lines.near_touched(y) {
                pull_halo_row(cells, halo, y);
            }
        }
        lines.row_touched.fill(false);
        lines.stale = false;
    }

    /// Search with the bounds of every stored item, as [`search`](Self::search)
    /// would item by item, and call `callback` with the item searched for and
    /// each item found, the searched one included. The items are taken cell
    /// by cell in Morton order, so that consecutive searches read nearby
    /// cells, and those of the overflow node last. It visits every cell, so
    /// where cells far outnumber items, searching item by item is faster. A
    /// one-off probe outside the repository, on balls that all fit a cell,
    /// measured it 19% faster than searching item by item in random order
    /// with 10,000 items and 37% with 640,000 at the same density, where the
    /// random order misses the cache; the `self_query` bench group, whose
    /// distribution puts its largest balls in the overflow node, measured 8%.
    pub fn for_each_self_query(&self, mut callback: impl FnMut(T, T)) {
        let cells = &self.cells;
        for_each_cell_in_morton_order(cells.cols(), cells.rows(), |x, y| {
            for entry in &cells[y][x].items {
                self.search(&entry.bounds, |found| callback(entry.item, found));
            }
        });
        for entry in &self.overflow.items {
            self.search(&entry.bounds, |found| callback(entry.item, found));
        }
    }

    /// Search the grid and execute the callback for each item whose bounds
    /// intersect `bounds`.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        // Every comparison with NaN is false, so a query with a NaN corner
        // meets nothing, while for a NaN minimum the scans would report the
        // items of the nodes they take as inside it untested.
        if bounds.has_nan() {
            return;
        }
        if self.fits_cell(bounds) {
            self.walk_3x3(bounds, |node| node.search_items(bounds, &mut callback));
        } else {
            self.scan(bounds, &mut callback);
        }
        self.overflow.search_items(bounds, &mut callback);
    }

    /// Visit the cell of the query's center and each neighbor whose items may
    /// reach the query, judged by the cell's copies, or, while those may be
    /// behind the boxes, by the spans. For queries no larger than a cell.
    #[inline]
    fn walk_3x3<'a>(&'a self, bounds: &Rectangle, visit: impl FnMut(&'a Node<T>)) {
        let (x, y) = self.cell_containing(bounds.get_center());
        // Copies can be behind only after a store, and then only near a
        // touched row.
        if self.lines.stale && self.lines.near_touched(y) {
            self.walk_3x3_by_spans((x, y), bounds, visit);
        } else {
            self.walk_3x3_by_copies((x, y), bounds, visit);
        }
    }

    /// Visit cell `(x, y)` and those of its eight neighbors whose copies in
    /// the cell reach the query.
    #[inline]
    fn walk_3x3_by_copies<'a>(
        &'a self,
        (x, y): (usize, usize),
        bounds: &Rectangle,
        mut visit: impl FnMut(&'a Node<T>),
    ) {
        let grid = &self.cells;
        // SAFETY: `(x, y)` is a cell, and `halo` holds one entry per cell
        // in the same order. An edge cell keeps the initial copies for its
        // missing neighbors, which fail every comparison, so the indices below
        // stay inside the grid.
        unsafe {
            let halo = self.halo.get_unchecked(y * grid.cols() + x);
            // All eight are computed up front; short-circuit branches here
            // measured search 50% slower in the multi-level grid on a scene
            // where the outcomes vary.
            let before = bounds.min_x < halo.left;
            let after = bounds.max_x > halo.right;
            let up = bounds.min_y < halo.top;
            let down = bounds.max_y > halo.bottom;
            let up_before =
                (bounds.min_x < halo.top_left[0]) & (bounds.min_y < halo.top_left[1]);
            let up_after =
                (bounds.max_x > halo.top_right[0]) & (bounds.min_y < halo.top_right[1]);
            let down_before = (bounds.min_x < halo.bottom_left[0])
                & (bounds.max_y > halo.bottom_left[1]);
            let down_after = (bounds.max_x > halo.bottom_right[0])
                & (bounds.max_y > halo.bottom_right[1]);
            debug_assert!(!(before | up_before | down_before) || x != 0);
            debug_assert!(!(after | up_after | down_after) || x + 1 != grid.cols());
            debug_assert!(!(up | up_before | up_after) || y != 0);
            debug_assert!(!(down | down_before | down_after) || y + 1 != grid.rows());

            visit(grid.get_unchecked(y, x));
            if up_before {
                visit(grid.get_unchecked(y - 1, x - 1));
            }
            if up {
                visit(grid.get_unchecked(y - 1, x));
            }
            if up_after {
                visit(grid.get_unchecked(y - 1, x + 1));
            }
            if before {
                visit(grid.get_unchecked(y, x - 1));
            }
            if after {
                visit(grid.get_unchecked(y, x + 1));
            }
            if down_before {
                visit(grid.get_unchecked(y + 1, x - 1));
            }
            if down {
                visit(grid.get_unchecked(y + 1, x));
            }
            if down_after {
                visit(grid.get_unchecked(y + 1, x + 1));
            }
        }
    }

    /// Visit cell `(x, y)` and each neighbor whose column and row, where they
    /// differ from the cell's, have spans that meet the query, as
    /// [`SpanGrid`](crate::SpanGrid) does.
    #[inline]
    fn walk_3x3_by_spans<'a>(
        &'a self,
        (x, y): (usize, usize),
        bounds: &Rectangle,
        mut visit: impl FnMut(&'a Node<T>),
    ) {
        let grid = &self.cells;
        let lines = &self.lines;
        let (cols, rows) = (grid.cols(), grid.rows());
        // SAFETY: `(x, y)` is a cell, the clamped line indices are in bounds,
        // and each neighbor is read only after the test that it exists.
        unsafe {
            let above = lines.rows.get_unchecked(y.saturating_sub(1))[1];
            let below = lines.rows.get_unchecked((y + 1).min(rows - 1))[0];
            let left_of = lines.columns.get_unchecked(x.saturating_sub(1))[1];
            let right_of = lines.columns.get_unchecked((x + 1).min(cols - 1))[0];
            let up = (y != 0) & (bounds.min_y < above);
            let down = (y + 1 != rows) & (bounds.max_y > below);
            let before = (x != 0) & (bounds.min_x < left_of);
            let after = (x + 1 != cols) & (bounds.max_x > right_of);

            if up {
                if before {
                    visit(grid.get_unchecked(y - 1, x - 1));
                }
                visit(grid.get_unchecked(y - 1, x));
                if after {
                    visit(grid.get_unchecked(y - 1, x + 1));
                }
            }
            if before {
                visit(grid.get_unchecked(y, x - 1));
            }
            visit(grid.get_unchecked(y, x));
            if after {
                visit(grid.get_unchecked(y, x + 1));
            }
            if down {
                if before {
                    visit(grid.get_unchecked(y + 1, x - 1));
                }
                visit(grid.get_unchecked(y + 1, x));
                if after {
                    visit(grid.get_unchecked(y + 1, x + 1));
                }
            }
        }
    }

    /// Scan every cell within half a cell of a query larger than a cell,
    /// dropping each edge column and row whose span misses the query.
    fn scan(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        let grid = &self.cells;
        let lines = &self.lines;
        let (cols, rows) = (grid.cols(), grid.rows());
        let (offset_x, offset_y) = self.root_bounds.get_min();
        let (inverse_width, inverse_height) = self.inverse_node_size;
        let min_coord_x = first_reaching_node((bounds.min_x - offset_x) * inverse_width);
        let min_coord_y = first_reaching_node((bounds.min_y - offset_y) * inverse_height);
        let max_coord_x =
            last_reaching_node((bounds.max_x - offset_x) * inverse_width).min(cols - 1);
        let max_coord_y =
            last_reaching_node((bounds.max_y - offset_y) * inverse_height).min(rows - 1);

        let column_reaches = |x: usize| {
            let [low, high] = unsafe { *lines.columns.get_unchecked(x) };
            high > bounds.min_x && low < bounds.max_x
        };
        let row_reaches = |y: usize| {
            let [low, high] = unsafe { *lines.rows.get_unchecked(y) };
            high > bounds.min_y && low < bounds.max_y
        };
        let (mut first_x, mut end_x) = (min_coord_x.min(cols), max_coord_x + 1);
        if first_x < end_x && !column_reaches(first_x) {
            first_x += 1;
        }
        if end_x > first_x && !column_reaches(end_x - 1) {
            end_x -= 1;
        }
        let (mut first_y, mut end_y) = (min_coord_y.min(rows), max_coord_y + 1);
        if first_y < end_y && !row_reaches(first_y) {
            first_y += 1;
        }
        if end_y > first_y && !row_reaches(end_y - 1) {
            end_y -= 1;
        }

        for y in first_y..end_y {
            for x in first_x..end_x {
                let node = unsafe { grid.get_unchecked(y, x) };
                if node.items.is_empty() {
                    continue;
                }
                if y > min_coord_y + 1
                    && y < max_coord_y - 1
                    && x > min_coord_x + 1
                    && x < max_coord_x - 1
                {
                    node.iter_items(callback);
                } else {
                    node.search_items(bounds, callback);
                }
            }
        }
    }
}

/// Span of the items of every column along x and of every row along y since
/// the last refresh, as `[min, max]`, with the touched marks for the refresh.
struct LineSpan {
    columns: Box<[[f64; 2]]>,
    rows: Box<[[f64; 2]]>,
    touched: bool,
    /// An item was stored since the last refresh, so a box may have grown
    /// past the copies of its sides held by the neighbors: until the
    /// refresh, the walk judges the neighbors of a query cell in or next to a
    /// touched row by the spans instead.
    stale: bool,
    row_touched: Box<[bool]>,
}

impl LineSpan {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            columns: vec![NOWHERE; cols].into_boxed_slice(),
            rows: vec![NOWHERE; rows].into_boxed_slice(),
            touched: false,
            stale: false,
            row_touched: vec![false; rows].into_boxed_slice(),
        }
    }

    #[inline]
    fn grow(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.touched = true;
        self.stale = true;
        self.row_touched[y] = true;
        self.widen((x, y), bounds);
    }

    /// Whether row `y` or a row next to it is touched, the condition under
    /// which the copies of row `y` may be behind the boxes they stand for.
    #[inline]
    fn near_touched(&self, y: usize) -> bool {
        let rows = &self.row_touched;
        let last = rows.len() - 1;
        rows[y] | rows[y.saturating_sub(1)] | rows[(y + 1).min(last)]
    }

    #[inline]
    fn mark_taken(&mut self, y: usize) {
        self.touched = true;
        self.row_touched[y] = true;
    }

    #[inline]
    fn widen(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.widen_column(x, bounds);
        let row = &mut self.rows[y];
        row[0] = row[0].min(bounds.min_y);
        row[1] = row[1].max(bounds.max_y);
    }

    #[inline]
    fn widen_column(&mut self, x: usize, bounds: &Rectangle) {
        let column = &mut self.columns[x];
        column[0] = column[0].min(bounds.min_x);
        column[1] = column[1].max(bounds.max_x);
    }
}

/// Sides of the boxes of the eight neighbors of a cell that face the cell,
/// in world coordinates: how far each neighbor's items reach towards it. The
/// initial values fail every comparison of the walk, so the copies never
/// select a neighbor that held no items at the last refresh, and the copies
/// of an edge cell for its missing neighbors never change.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Halo {
    /// Largest `max_x` of the items of the left neighbor `(x - 1, y)`.
    left: f64,
    /// Smallest `min_x` of the items of the right neighbor `(x + 1, y)`.
    right: f64,
    /// Largest `max_y` of the items of the neighbor above, `(x, y - 1)`.
    top: f64,
    /// Smallest `min_y` of the items of the neighbor below, `(x, y + 1)`.
    bottom: f64,
    /// Largest `max_x` and `max_y` of the items of `(x - 1, y - 1)`.
    top_left: [f64; 2],
    /// Smallest `min_x` and largest `max_y` of the items of `(x + 1, y - 1)`.
    top_right: [f64; 2],
    /// Largest `max_x` and smallest `min_y` of the items of `(x - 1, y + 1)`.
    bottom_left: [f64; 2],
    /// Smallest `min_x` and `min_y` of the items of `(x + 1, y + 1)`.
    bottom_right: [f64; 2],
}

/// Copies for eight neighbors without items, that is, with [`EMPTY`] boxes.
const NO_HALO: Halo = Halo::of_boxes([&EMPTY; 8]);

impl Halo {
    /// Copies of the facing sides of the eight neighbors' boxes, given in
    /// the order left, right, top, bottom, top left, top right, bottom left,
    /// bottom right.
    const fn of_boxes(
        [left, right, top, bottom, top_left, top_right, bottom_left, bottom_right]: [&Rectangle; 8],
    ) -> Self {
        Self {
            left: left.max_x,
            right: right.min_x,
            top: top.max_y,
            bottom: bottom.min_y,
            top_left: [top_left.max_x, top_left.max_y],
            top_right: [top_right.min_x, top_right.max_y],
            bottom_left: [bottom_left.max_x, bottom_left.min_y],
            bottom_right: [bottom_right.min_x, bottom_right.min_y],
        }
    }

    /// Copies of the facing sides of the current boxes of the eight
    /// neighbors of `(x, y)` in `grid`, with [`EMPTY`] standing in for the
    /// missing neighbors of an edge cell, as given by the flags.
    ///
    /// # Safety
    ///
    /// `(x, y)` must be a cell of `grid`, and each flag must be false if the
    /// grid has no cell on that side of it.
    #[inline(always)]
    unsafe fn of_neighbors<T: Copy>(
        grid: &Grid<Node<T>>,
        (x, y): (usize, usize),
        [has_left, has_right]: [bool; 2],
        [has_top, has_bottom]: [bool; 2],
    ) -> Self {
        // Wrapped indices are never read, since their flag is false.
        let (left, right) = (x.wrapping_sub(1), x + 1);
        let (top, bottom) = (y.wrapping_sub(1), y + 1);
        unsafe {
            Self::of_boxes([
                box_at(grid, has_left, left, y),
                box_at(grid, has_right, right, y),
                box_at(grid, has_top, x, top),
                box_at(grid, has_bottom, x, bottom),
                box_at(grid, has_left & has_top, left, top),
                box_at(grid, has_right & has_top, right, top),
                box_at(grid, has_left & has_bottom, left, bottom),
                box_at(grid, has_right & has_bottom, right, bottom),
            ])
        }
    }
}

/// Box of cell `(x, y)` of `grid` if `present`, else [`EMPTY`].
///
/// # Safety
///
/// `(x, y)` must be a cell of `grid` when `present`.
#[inline(always)]
unsafe fn box_at<T: Copy>(grid: &Grid<Node<T>>, present: bool, x: usize, y: usize) -> &Rectangle {
    if present {
        unsafe { &grid.get_unchecked(y, x).bounds }
    } else {
        &EMPTY
    }
}

/// Set the copies in `halo` of every cell of row `y` of `grid` to the
/// facing sides of its neighbors' boxes, which must already be shrunk in
/// rows `y - 1` to `y + 1`. Interior cells of the row have both horizontal
/// neighbors, so only the two end cells test for the left and right edges.
fn pull_halo_row<T: Copy>(grid: &Grid<Node<T>>, halo: &mut [Halo], y: usize) {
    let (cols, rows) = (grid.cols(), grid.rows());
    let vertical = [y != 0, y + 1 != rows];
    let row = &mut halo[y * cols..(y + 1) * cols];
    let mut pull = |x: usize, horizontal: [bool; 2]| {
        // SAFETY: `y` is a row and `x` a column of `grid`, and the flags
        // state which neighbors the cell lacks.
        unsafe {
            *row.get_unchecked_mut(x) = Halo::of_neighbors(grid, (x, y), horizontal, vertical);
        }
    };
    if cols == 1 {
        pull(0, [false, false]);
        return;
    }
    pull(0, [false, true]);
    for x in 1..cols - 1 {
        pull(x, [true, true]);
    }
    pull(cols - 1, [true, false]);
}

struct Node<T: Copy> {
    /// Covers the bounds of every item in `items`; `EMPTY` while there are none.
    bounds: Rectangle,
    items: Vec<Entry<T>>,
}

impl<T: Copy> Node<T> {
    fn new() -> Self {
        Self {
            bounds: EMPTY,
            items: Vec::new(),
        }
    }

    #[inline]
    fn grow(&mut self, bounds: &Rectangle) {
        self.bounds.min_x = self.bounds.min_x.min(bounds.min_x);
        self.bounds.min_y = self.bounds.min_y.min(bounds.min_y);
        self.bounds.max_x = self.bounds.max_x.max(bounds.max_x);
        self.bounds.max_y = self.bounds.max_y.max(bounds.max_y);
    }

    fn refresh(&mut self) {
        self.bounds = EMPTY;
        let items = std::mem::take(&mut self.items);
        for entry in &items {
            self.grow(&entry.bounds);
        }
        self.items = items;
    }

    fn search_items(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        if !self.bounds.intersects(bounds) {
            return;
        }
        self.items
            .iter()
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }

    fn iter_items(&self, callback: &mut impl FnMut(T)) {
        self.items.iter().for_each(|entry| callback(entry.item));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    type Tree = HaloGrid<usize>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    crate::grid_tests::single_level_tests!();

    /// Copies the walk may rely on: for a cell whose row is not near a
    /// touched one, the facing sides of its neighbors' current boxes, with
    /// `EMPTY` boxes for empty or missing neighbors. Derived here from the
    /// boxes directly.
    fn expected_copies(tree: &Tree, x: usize, y: usize) -> [f64; 12] {
        let cells = &tree.cells;
        let (cols, rows) = (cells.cols() as isize, cells.rows() as isize);
        let at = |dx: isize, dy: isize| {
            let (nx, ny) = (x as isize + dx, y as isize + dy);
            if nx < 0 || ny < 0 || nx >= cols || ny >= rows {
                EMPTY
            } else {
                cells[ny as usize][nx as usize].bounds.clone()
            }
        };
        [
            at(-1, 0).max_x,
            at(1, 0).min_x,
            at(0, -1).max_y,
            at(0, 1).min_y,
            at(-1, -1).max_x,
            at(-1, -1).max_y,
            at(1, -1).min_x,
            at(1, -1).max_y,
            at(-1, 1).max_x,
            at(-1, 1).min_y,
            at(1, 1).min_x,
            at(1, 1).min_y,
        ]
    }

    fn copies(tree: &Tree, x: usize, y: usize) -> [f64; 12] {
        let c = tree.halo[y * tree.cells.cols() + x];
        [
            c.left,
            c.right,
            c.top,
            c.bottom,
            c.top_left[0],
            c.top_left[1],
            c.top_right[0],
            c.top_right[1],
            c.bottom_left[0],
            c.bottom_left[1],
            c.bottom_right[0],
            c.bottom_right[1],
        ]
    }

    /// After random operations, the copies of every cell outside the rows
    /// near a touched one equal its neighbors' facing sides; after a refresh,
    /// those of every cell do.
    #[test]
    fn copies_match_the_neighbor_boxes_outside_touched_rows() {
        let mut rng = StdRng::seed_from_u64(5);
        for world in WORLDS {
            let cell = 64.0;
            let mut tree = Tree::new(world.clone(), cell);
            let mut items: Vec<ItemHandle> = (0..600)
                .map(|id| tree.insert(random_bounds(&mut rng, &world, cell), id))
                .collect();
            tree.refresh_bounds();
            for round in 0..4 {
                // A few changes, so that most rows stay untouched.
                for _ in 0..5 {
                    let handle = items[rng.gen_range(0..items.len())];
                    tree.update(random_bounds(&mut rng, &world, cell), handle);
                }
                let index = rng.gen_range(0..items.len());
                tree.remove(items.swap_remove(index));
                items.push(tree.insert(random_bounds(&mut rng, &world, cell), 1000 + round));

                for refreshed in [false, true] {
                    if refreshed {
                        tree.refresh_bounds();
                    }
                    for y in 0..tree.cells.rows() {
                        if !refreshed && tree.lines.near_touched(y) {
                            continue;
                        }
                        for x in 0..tree.cells.cols() {
                            assert_eq!(
                                copies(&tree, x, y),
                                expected_copies(&tree, x, y),
                                "round {round}, refreshed {refreshed}, cell ({x}, {y})"
                            );
                        }
                    }
                }
            }
        }
    }

    fn walked(tree: &Tree, query: &Rectangle) -> Vec<(usize, usize)> {
        let cells = &tree.cells;
        let mut read = Vec::new();
        tree.walk_3x3(query, |node| {
            let index = (0..cells.rows() * cells.cols())
                .find(|&i| std::ptr::eq(&cells[i / cells.cols()][i % cells.cols()], node))
                .unwrap();
            read.push((index % cells.cols(), index / cells.cols()));
        });
        read.sort();
        read
    }

    /// Cells of 100 units; the query lies in cell (5, 5). The left neighbor
    /// (4, 5) holds an item ending at x = 460, short of the query, while
    /// another item of column 4, in row 8, reaches x = 525 and so carries
    /// the column's span into the query. After a refresh the walk judges the
    /// left neighbor by its own box and leaves it out; before, it falls back
    /// to the span and reads it.
    #[test]
    fn walk_reads_only_neighbors_whose_copies_reach_the_query() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1000.0, 1000.0), 100.0);
        tree.insert(Rectangle::new(440.0, 540.0, 460.0, 560.0), 0);
        tree.insert(Rectangle::new(465.0, 820.0, 525.0, 880.0), 1);
        tree.insert(Rectangle::new(540.0, 540.0, 560.0, 560.0), 2);
        let query = Rectangle::new(520.0, 530.0, 580.0, 570.0);

        assert_eq!(walked(&tree, &query), vec![(4, 5), (5, 5)]);
        tree.refresh_bounds();
        assert_eq!(walked(&tree, &query), vec![(5, 5)]);
    }

    /// An item stored in a neighbor after the refresh reaches past the
    /// neighbor's copied side; the walk must still find it before the next
    /// refresh.
    #[test]
    fn item_stored_after_the_refresh_is_found_before_the_next() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1000.0, 1000.0), 100.0);
        tree.insert(Rectangle::new(440.0, 540.0, 460.0, 560.0), 0);
        tree.refresh_bounds();
        tree.insert(Rectangle::new(470.0, 540.0, 510.0, 560.0), 1);

        let query = Rectangle::new(505.0, 530.0, 580.0, 570.0);
        let mut found = Vec::new();
        tree.search(&query, |id| found.push(id));
        assert_eq!(found, vec![1]);
    }
}
