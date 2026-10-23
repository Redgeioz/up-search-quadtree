//! A single-level loose grid with column and row spans:
//! [`SpanOverflowBoundGrid`] reduced
//! to its finest level, for scenes whose items all fit a cell. It serves as a
//! comparison with the uniform grid, which instead registers an item in every
//! cell it overlaps.
//!
//! Every item is stored in the cell that contains its center, so an item no
//! larger than a cell extends at most half a cell beyond it (the fixed
//! reach). A query no larger than a cell reads the cell of its center and the
//! eight around it; a larger query scans the cells within half a cell of it.
//! Each cell keeps a box covering its items, and each column and row the span
//! of its items along its own axis; neighbor reads and the edge columns and
//! rows of a scan are skipped where no span meets the query. Boxes and spans
//! only widen, except that an emptied node resets its box, until
//! [`refresh_bounds`] shrinks them on the touched rows. Items larger than a
//! cell, degenerate (zero or NaN size) or centered outside the world go to
//! one overflow node that every search reads.
//!
//! [`SpanOverflowBoundGrid`]: crate::SpanOverflowBoundGrid
//! [`refresh_bounds`]: SpanGrid::refresh_bounds

use crate::cell_order::for_each_cell_in_morton_order;
use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::layer_geometry::{LayerGeometry, first_reaching_node, last_reaching_node};
use crate::rect::Rectangle;
use grid::*;

/// Box of a node without items, which meets no query.
const EMPTY: Rectangle = Rectangle::new(
    f64::INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
);

/// Span of a line without items, which meets no query.
const NOWHERE: [f64; 2] = [f64::INFINITY, f64::NEG_INFINITY];

/// `level` of a [`GridLocation`] in the overflow node; cells are level 1.
const OVERFLOW: usize = 0;

pub struct SpanGrid<T: Copy> {
    cells: Grid<Node<T>>,
    geometry: LayerGeometry,
    /// Reciprocals of the cell width and height. Position and search multiply
    /// by them instead of dividing by the cell size, and `geometry` builds its
    /// cell edges with the same arithmetic.
    inverse_node_size: (f64, f64),
    lines: LineSpan,
    /// Items larger than a cell, degenerate or centered outside the world.
    overflow: Node<T>,
    overflow_touched: bool,
    /// Square of whole cells anchored at the world's min corner, from which
    /// the geometry is derived, as in the hierarchical grids.
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    slots: Slots<GridLocation>,
}

impl<T: Copy> SpanGrid<T> {
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
    /// of those rows and of every column, and shrink the overflow node's box
    /// if it was touched.
    pub fn refresh_bounds(&mut self) {
        if std::mem::take(&mut self.overflow_touched) {
            self.overflow.refresh();
        }
        if !std::mem::take(&mut self.lines.touched) {
            return;
        }
        let lines = &mut self.lines;
        let cells = &mut self.cells;
        lines.columns.fill(NOWHERE);
        for y in 0..cells.rows() {
            if std::mem::take(&mut lines.row_touched[y]) {
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
        }
    }

    /// Search with the bounds of every stored item, as [`search`](Self::search)
    /// would item by item, and call `callback` with the item searched for and
    /// each item found, the searched one included. The items are taken cell
    /// by cell in Morton order, so that consecutive searches read nearby
    /// cells, and those of the overflow node last. It visits every cell, so
    /// where cells far outnumber items, searching item by item is faster. A
    /// one-off probe outside the repository, on balls that all fit a cell,
    /// measured it 8% faster than searching item by item in random order
    /// with 10,000 items and 23% with 640,000 at the same density, where the
    /// random order misses the cache; the `self_query` bench group, whose
    /// distribution puts its largest balls in the overflow node, measured 5%.
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
            self.search_3x3(bounds, &mut callback);
        } else {
            self.scan(bounds, &mut callback);
        }
        self.overflow.search_items(bounds, &mut callback);
    }

    /// Search the cell of the query's center and each neighbor whose column
    /// and row, where they differ from the center's, have spans that meet the
    /// query. For queries no larger than a cell.
    fn search_3x3(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        let (x, y) = self.cell_containing(bounds.get_center());
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
                    grid.get_unchecked(y - 1, x - 1).search_items(bounds, callback);
                }
                grid.get_unchecked(y - 1, x).search_items(bounds, callback);
                if after {
                    grid.get_unchecked(y - 1, x + 1).search_items(bounds, callback);
                }
            }
            if before {
                grid.get_unchecked(y, x - 1).search_items(bounds, callback);
            }
            grid.get_unchecked(y, x).search_items(bounds, callback);
            if after {
                grid.get_unchecked(y, x + 1).search_items(bounds, callback);
            }
            if down {
                if before {
                    grid.get_unchecked(y + 1, x - 1).search_items(bounds, callback);
                }
                grid.get_unchecked(y + 1, x).search_items(bounds, callback);
                if after {
                    grid.get_unchecked(y + 1, x + 1).search_items(bounds, callback);
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
    row_touched: Box<[bool]>,
}

impl LineSpan {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            columns: vec![NOWHERE; cols].into_boxed_slice(),
            rows: vec![NOWHERE; rows].into_boxed_slice(),
            touched: false,
            row_touched: vec![false; rows].into_boxed_slice(),
        }
    }

    #[inline]
    fn grow(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.touched = true;
        self.row_touched[y] = true;
        self.widen((x, y), bounds);
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

    type Tree = SpanGrid<usize>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    /// The update shortcut must place a center in the same cell as
    /// `position`, also within a few floats of every cell edge, where
    /// multiplying by the reciprocal of the cell size and dividing by it can
    /// round to different cells.
    #[test]
    fn update_shortcut_agrees_with_position_at_cell_edges() {
        for world in WORLDS {
            for cell in [64.0, 312.5, 77.7] {
                let tree = Tree::new(world.clone(), cell);
                let origin = tree.root_bounds.get_min();
                let (cell_width, cell_height) = tree.cell_size();
                let inside = (origin.0 + cell_width * 0.5, origin.1 + cell_height * 0.5);
                let near = |edge: f64| {
                    let mut points = vec![edge];
                    let (mut up, mut down) = (edge, edge);
                    for _ in 0..3 {
                        up = up.next_up();
                        down = down.next_down();
                        points.extend([up, down]);
                    }
                    points
                };
                let mut centers = Vec::new();
                for i in 1..tree.cells.cols() {
                    for x in near(origin.0 + cell_width * i as f64) {
                        centers.push((x, inside.1));
                    }
                }
                for i in 1..tree.cells.rows() {
                    for y in near(origin.1 + cell_height * i as f64) {
                        centers.push((inside.0, y));
                    }
                }
                for center in centers {
                    let cell = tree.cell_containing(center);
                    let relative = (center.0 - origin.0, center.1 - origin.1);
                    assert!(
                        tree.geometry.matches_cell(relative, cell),
                        "world {world:?}, cell size {cell_width}, center {center:?}, cell {cell:?}"
                    );
                }
            }
        }
    }

    /// Bounds mostly smaller than a cell; some are larger and some centered
    /// outside the world.
    fn random_bounds(rng: &mut StdRng, world: &Rectangle, cell: f64) -> Rectangle {
        let (x, y) = world.get_random_point(rng);
        let (x, y) = (x + rng.gen_range(-cell..cell), y + rng.gen_range(-cell..cell));
        let scale = if rng.gen_bool(0.1) {
            cell * rng.gen_range(1.0..4.0)
        } else {
            cell * rng.gen_range(0.05..1.0)
        };
        Rectangle::center_rect(x, y, scale * rng.gen_range(0.5..1.0), scale * rng.gen_range(0.5..1.0))
    }

    /// Boxes must cover their items, and spans the items of their column or
    /// row; both must be exact right after a refresh. An empty node must have
    /// the empty box at any time.
    fn assert_covering(tree: &Tree, exact: bool) {
        let cells = &tree.cells;
        let (cols, rows) = (cells.cols(), cells.rows());
        let mut columns = vec![NOWHERE; cols];
        let mut row_spans = vec![NOWHERE; rows];
        for y in 0..rows {
            for x in 0..cols {
                let node = &cells[y][x];
                let mut union = EMPTY;
                for entry in &node.items {
                    union.min_x = union.min_x.min(entry.bounds.min_x);
                    union.min_y = union.min_y.min(entry.bounds.min_y);
                    union.max_x = union.max_x.max(entry.bounds.max_x);
                    union.max_y = union.max_y.max(entry.bounds.max_y);
                }
                if exact || node.items.is_empty() {
                    assert_eq!(node.bounds, union, "cell ({x}, {y})");
                } else {
                    assert!(node.bounds.contains(&union), "cell ({x}, {y})");
                }
                if !node.items.is_empty() {
                    columns[x] = [columns[x][0].min(union.min_x), columns[x][1].max(union.max_x)];
                    row_spans[y] = [row_spans[y][0].min(union.min_y), row_spans[y][1].max(union.max_y)];
                }
            }
        }
        for x in 0..cols {
            let span = tree.lines.columns[x];
            if exact {
                assert_eq!(span, columns[x], "column {x}");
            } else {
                assert!(span[0] <= columns[x][0] && span[1] >= columns[x][1], "column {x}");
            }
        }
        for y in 0..rows {
            let span = tree.lines.rows[y];
            if exact {
                assert_eq!(span, row_spans[y], "row {y}");
            } else {
                assert!(span[0] <= row_spans[y][0] && span[1] >= row_spans[y][1], "row {y}");
            }
        }
    }

    #[test]
    fn random_operations_match_brute_force() {
        let mut rng = StdRng::seed_from_u64(21);
        for world in WORLDS {
            for cell in [64.0, 312.5] {
                let mut tree = Tree::new(world.clone(), cell);
                let mut items: Vec<(ItemHandle, Rectangle)> = Vec::new();
                for step in 0..3000 {
                    match rng.gen_range(0..5) {
                        0 | 1 => {
                            let bounds = random_bounds(&mut rng, &world, cell);
                            items.push((tree.insert(bounds.clone(), step), bounds));
                        }
                        2 | 3 if !items.is_empty() => {
                            let index = rng.gen_range(0..items.len());
                            let (handle, old) = &items[index];
                            let bounds = if rng.gen_bool(0.7) {
                                let (x, y) = old.get_center();
                                Rectangle::center_rect(
                                    x + rng.gen_range(-0.2..0.2) * cell,
                                    y + rng.gen_range(-0.2..0.2) * cell,
                                    old.get_width() * rng.gen_range(0.8..1.2),
                                    old.get_height() * rng.gen_range(0.8..1.2),
                                )
                            } else {
                                random_bounds(&mut rng, &world, cell)
                            };
                            tree.update(bounds.clone(), *handle);
                            items[index].1 = bounds;
                        }
                        _ if !items.is_empty() => {
                            let (handle, _) = items.swap_remove(rng.gen_range(0..items.len()));
                            tree.remove(handle);
                        }
                        _ => {}
                    }
                    let refreshed = step % 500 == 499;
                    if refreshed {
                        tree.refresh_bounds();
                    }
                    if refreshed || step % 100 == 0 {
                        assert_covering(&tree, refreshed);
                        for _ in 0..10 {
                            let query = random_bounds(&mut rng, &world, cell);
                            let mut expected: Vec<usize> = items
                                .iter()
                                .filter(|(_, b)| b.intersects(&query))
                                .map(|(h, _)| {
                                    let location = tree.slots.get(*h);
                                    let (level, x, y) = location.coord();
                                    let node = if level == OVERFLOW {
                                        &tree.overflow
                                    } else {
                                        &tree.cells[y][x]
                                    };
                                    node.items[location.index as usize].item
                                })
                                .collect();
                            expected.sort_unstable();
                            let mut found = Vec::new();
                            tree.search(&query, |id| found.push(id));
                            found.sort_unstable();
                            assert_eq!(found, expected, "cell {cell}, step {step}, query {query:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn items_that_fit_a_cell_never_overflow() {
        let world = Rectangle::new(0.0, 0.0, 1000.0, 1000.0);
        let mut tree = Tree::new(world, 100.0);
        let (cell_w, cell_h) = tree.cell_size();
        assert_eq!((cell_w, cell_h), (100.0, 100.0));
        tree.insert(Rectangle::center_rect(50.0, 50.0, 100.0, 100.0), 0);
        tree.insert(Rectangle::center_rect(999.0, 999.0, 100.0, 100.0), 1);
        assert!(tree.overflow.items.is_empty());
        tree.insert(Rectangle::center_rect(500.0, 500.0, 100.1, 10.0), 2);
        tree.insert(Rectangle::center_rect(-1.0, 500.0, 10.0, 10.0), 3);
        assert_eq!(tree.overflow.items.len(), 2);
    }
}
