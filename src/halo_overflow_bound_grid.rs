use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::cell_order::for_each_cell_hierarchically;
use crate::layer_geometry::{LayerGeometry, half_node_corners, scan_range, size_level};
use crate::overflow_bound_grid::DirectionalOverflow;
use crate::overflow_bound::{cell_overflow, higher, lower};
use crate::rect::Rectangle;
use grid::*;

/// `(level, x, y)`
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<HaloOverflowBoundGridNode<T>>>;

/// Box of a node without items. Every comparison of [`Rectangle::intersects`]
/// fails against it, so search rejects the node without a separate check.
const EMPTY: Rectangle = Rectangle::new(
    f64::INFINITY,
    f64::INFINITY,
    f64::NEG_INFINITY,
    f64::NEG_INFINITY,
);

/// [`SpanOverflowBoundGrid`] with the neighbor pruning of the ancestor
/// walk moved from the column and row spans of a level to the cells
/// themselves. It uses the storage, positioning and handles of
/// [`HierarchicalLooseGrid`]; every
/// node keeps a box covering its items and copies of the sides of its eight
/// neighbors' boxes that face it, that is, how far each neighbor's items
/// reach towards the node. The walk reads the query cell's node on every
/// level anyway, and from these copies alone decides which of the eight
/// neighbors within the fixed reach (half a node) to read, instead of judging
/// them by the span of their whole column or row, which on a dense level
/// nearly always meets the query. A neighbor that is read is still skipped
/// when its box misses the query.
///
/// The column and row spans of [`SpanOverflowBoundGrid`] remain, but only
/// trim the edges of the scan over finer levels.
///
/// Insertion and update only widen the box of the target node and the spans
/// of its column and row, and mark the level stale and the row touched; the
/// copies in the neighbors are not updated. Until the refresh, the walk
/// judges the neighbors of a query cell in or next to a touched row of a
/// stale level by the spans, as [`SpanOverflowBoundGrid`] does and at its
/// cost.
/// Pushing every grown side to the three neighbors it faces on each update
/// instead measured the update benchmark 50-90% slower than
/// [`SpanOverflowBoundGrid`], for a search gain realized only once a frame's updates are done.
/// Removal and moving out leave boxes, spans and copies unchanged, except
/// that a node left without items resets its box. Boxes and spans therefore
/// stay valid as items move, shrink or leave, but tighten only in
/// [`refresh_bounds`], which also brings the copies up to date and clears
/// the marks; the caller chooses when, e.g. once per frame after the updates.
///
/// The walk also stops at the coarsest level below the root that holds items,
/// which a bitmask of the occupied levels keeps exact at all times.
///
/// Boxes, spans and copies consist of item coordinates, so their checks
/// compare the same values as the intersection tests and need no rounding
/// margin. None of them needs cell edges, which also keeps them cheap to
/// maintain.
///
/// [`SpanOverflowBoundGrid`]: crate::SpanOverflowBoundGrid
/// [`HierarchicalLooseGrid`]: crate::HierarchicalLooseGrid
/// [`refresh_bounds`]: Self::refresh_bounds
pub struct HaloOverflowBoundGrid<T: Copy, const MAX_LEVEL: u8> {
    layers: Layers<T>,
    /// Indexed by level, parallel to `layers`.
    geometry: Vec<LayerGeometry>,
    /// Reciprocals of the node width and height, indexed by level. Position
    /// and search multiply by them instead of dividing by the node size, and
    /// `geometry` builds its cell edges with the same arithmetic.
    inverse_node_size: Vec<(f64, f64)>,
    /// Indexed by level, parallel to `layers`.
    lines: Vec<LineSpan>,
    /// Indexed by level, parallel to `layers`.
    marks: Box<[Marks]>,
    /// Copies of every node, indexed by level and then by node in row-major
    /// order. Kept apart from the nodes so that these stay 56 bytes for the
    /// scan and for updates, which never touch the copies; the walk reads one
    /// entry here beside its node on every level.
    halo: Vec<Box<[Halo]>>,
    /// Bit `L` is set while level `L` holds items: storing an item sets the
    /// bit of its level, and taking the last item off a level clears it. The
    /// coarsest level below the root that holds items, where the ancestor walk
    /// of search stops, is therefore exact at all times, not only after a
    /// refresh. With all items on the finest level, stopping the walk there
    /// measured 10% faster than walking every level.
    occupied_levels: u64,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    /// Location of the entry of each live handle.
    slots: Slots<GridLocation>,
}

impl<T: Copy, const MAX_LEVEL: u8> HaloOverflowBoundGrid<T, MAX_LEVEL> {
    /// Create a new grid with the given bounds.
    ///
    /// Use square nodes, which usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> HaloOverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new grid with the given bounds.
    ///
    /// Fit the nodes to the given bounds instead of making them square.
    pub fn new_fit(world_bounds: Rectangle) -> HaloOverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(
        world_bounds: Rectangle,
    ) -> HaloOverflowBoundGrid<T, MAX_LEVEL> {
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let world_width = world_bounds.get_width();
        let world_height = world_bounds.get_height();

        // Initialize the levels.
        let mut layers = Vec::with_capacity(MAX_LEVEL as usize + 1);
        let mut size = 0;
        for n in 0..=MAX_LEVEL {
            let (mut rows, mut cols) = (size, size);
            if !FIT {
                if world_width > world_height {
                    rows = (size as f64 * world_height / world_width).ceil() as usize;
                } else if world_height > world_width {
                    cols = (size as f64 / world_height * world_width).ceil() as usize;
                }
            }
            let mut vec = Vec::with_capacity(rows * cols);
            for _ in 0..rows {
                for _ in 0..cols {
                    vec.push(HaloOverflowBoundGridNode::new());
                }
            }
            layers.push(Grid::from_vec(vec, cols));
            size = 1 << n;
        }

        // Determine the root bounds.
        let root_bounds = if !FIT {
            let len = world_width.max(world_height);
            let root_width = len;
            let root_height = len;

            let (min_x, min_y) = world_bounds.get_min();
            let center_x = min_x + len * 0.5;
            let center_y = min_y + len * 0.5;
            Rectangle::center_rect(center_x, center_y, root_width, root_height)
        } else {
            world_bounds.clone()
        };

        let mut geometry = Vec::with_capacity(layers.len());
        let mut inverse_node_size = Vec::with_capacity(layers.len());
        // Level zero is unused, like `layers[0]`.
        geometry.push(LayerGeometry::default());
        inverse_node_size.push((0.0, 0.0));
        for (level, grid) in layers.iter().enumerate().skip(1) {
            let (node_width, node_height) =
                LayerGeometry::node_size(&root_bounds, grid.cols(), grid.rows());
            let inverse = (1.0 / node_width, 1.0 / node_height);
            geometry.push(LayerGeometry::with_reciprocal(
                &root_bounds,
                level,
                grid.cols(),
                grid.rows(),
                inverse.0,
                inverse.1,
            ));
            inverse_node_size.push(inverse);
        }

        // The root's spans grow like those of every level but are never read:
        // search reads the root whole.
        let lines = layers
            .iter()
            .map(|grid| LineSpan::new(grid.cols(), grid.rows()))
            .collect();

        let marks = vec![Marks::default(); layers.len()].into_boxed_slice();
        let halo = layers
            .iter()
            .map(|grid| vec![NO_HALO; grid.rows() * grid.cols()].into_boxed_slice())
            .collect();

        HaloOverflowBoundGrid {
            layers,
            geometry,
            inverse_node_size,
            lines,
            marks,
            halo,
            occupied_levels: 0,
            root_bounds,
            world_bounds,
            slots: Slots::new(),
        }
    }

    /// Return the real bounds of the root.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Largest overflow beyond the cells on `level` since the last
    /// [`refresh_bounds`](Self::refresh_bounds), in world units, as implied
    /// by the spans of its columns and rows: negative when every item keeps
    /// that far off its cell edge, -infinity on a level without items. Search
    /// uses the spans directly.
    ///
    /// Panics unless `level` is in `2..=MAX_LEVEL`; the root has no neighbors
    /// to bound.
    pub fn overflow_bound(&self, level: usize) -> DirectionalOverflow {
        assert!((2..=MAX_LEVEL as usize).contains(&level), "invalid level");
        let geometry = &self.geometry[level];
        let node_size = (geometry.node_width, geometry.node_height);
        let origin = self.root_bounds.get_min();
        let lines = &self.lines[level];
        // Use the cell edges of `cell_overflow`, so the bound matches it item
        // by item.
        let mut largest = [f64::NEG_INFINITY; 4];
        for (x, &[min_x, max_x]) in lines.columns.iter().enumerate() {
            let span = Rectangle::new(min_x, origin.1, max_x, origin.1);
            let [left, right, ..] = cell_overflow(&span, origin, node_size, (x, 0));
            largest[0] = largest[0].max(left);
            largest[1] = largest[1].max(right);
        }
        for (y, &[min_y, max_y]) in lines.rows.iter().enumerate() {
            let span = Rectangle::new(origin.0, min_y, origin.0, max_y);
            let [.., top, bottom] = cell_overflow(&span, origin, node_size, (0, y));
            largest[2] = largest[2].max(top);
            largest[3] = largest[3].max(bottom);
        }
        let [left, right, top, bottom] = largest;
        DirectionalOverflow {
            left,
            right,
            top,
            bottom,
        }
    }

    /// On every level changed since the last refresh, shrink the boxes and
    /// row spans of the touched rows to the items held now, rebuild every
    /// column span from the boxes, set the copies in and next to the touched
    /// rows to the sides of the shrunk boxes, and clear the marks of the
    /// level. This undoes the growth from items that have since moved, shrunk
    /// or left. Visits every node of such a level once, and the items of its
    /// touched rows; a level nothing was stored in or taken from is already
    /// exact and is skipped.
    ///
    /// A refreshed box is the union of its items, so the spans and copies
    /// follow from the boxes, one per node rather than one per item. Every
    /// node pulls the sides it needs from its neighbors, one row behind the
    /// shrinking, rather than every node pushing its sides to its neighbors,
    /// which would first require resetting every copy.
    pub fn refresh_bounds(&mut self) {
        for level in 1..=MAX_LEVEL as usize {
            if !std::mem::take(&mut self.marks[level]).touched {
                continue;
            }
            let lines = &mut self.lines[level];
            let grid = &mut self.layers[level];
            let halo = &mut self.halo[level];
            if self.geometry[level].is_empty() {
                // Every node reset its box as it emptied; only the spans and
                // copies remain to reset.
                lines.clear();
                lines.row_touched.fill(false);
                halo.fill(NO_HALO);
                continue;
            }
            let rows = grid.rows();
            // A column span runs across every row, so it is rebuilt from the
            // boxes of all rows; a row's own span and boxes are rebuilt only
            // if the row is touched.
            lines.columns.fill(NOWHERE);
            let shrink_row = |grid: &mut Grid<HaloOverflowBoundGridNode<T>>,
                              lines: &mut LineSpan,
                              y: usize| {
                if lines.row_touched[y] {
                    lines.rows[y] = NOWHERE;
                    for (x, node) in grid[y].iter_mut().enumerate() {
                        node.refresh();
                        if !node.items.is_empty() {
                            lines.widen((x, y), &node.bounds);
                        }
                    }
                } else {
                    for (x, node) in grid[y].iter().enumerate() {
                        if !node.items.is_empty() {
                            lines.grow_column(x, &node.bounds);
                        }
                    }
                }
            };
            // The copies of a row read the boxes of that row and the rows on
            // both sides, so they are pulled one row behind the shrinking,
            // and only if one of these three rows is touched.
            shrink_row(grid, lines, 0);
            for y in 0..rows {
                if y + 1 < rows {
                    shrink_row(grid, lines, y + 1);
                }
                if lines.near_touched(y) {
                    pull_halo_row(grid, halo, y);
                }
            }
            lines.row_touched.fill(false);
        }
    }

    /// Record that an item was stored on `level`, so a box there may have
    /// grown past the copies its neighbors hold. `grow_lines`, which every
    /// store goes through, marks the row.
    #[inline]
    fn mark_stored(&mut self, level: usize) {
        self.marks[level] = Marks::STORED;
    }

    /// Record that an item was taken from row `y` of `level`, so a box or
    /// span there may be looser than its items.
    #[inline]
    fn mark_taken(&mut self, level: usize, y: usize) {
        self.marks[level].touched = true;
        self.lines[level].row_touched[y] = true;
    }

    /// Widen the spans of the column and row of `coord` to cover `bounds`
    /// stored there, and mark the row touched. Every store runs it. Once
    /// `cover` and the spans grew with `lower` and `higher`, the inliner left
    /// its call in `replace_bounds` out of line; that cost nothing measurable
    /// here, unlike in [`SpanOverflowBoundGrid`](crate::SpanOverflowBoundGrid),
    /// but it is forced inline as there.
    #[inline(always)]
    fn grow_lines(&mut self, bounds: &Rectangle, (level, x, y): Coord) {
        self.lines[level].grow((x, y), bounds);
    }

    /// Return the level and coordinates of the node that stores an item with
    /// `bounds`.
    pub fn position(&self, bounds: &Rectangle) -> Coord {
        let center = bounds.get_center();
        if !self.world_bounds.contains_point(center) {
            return (1, 0, 0);
        }

        let level = self.level_for_size(bounds.get_width(), bounds.get_height());
        let (x, y) = self.cell_containing(level, center);
        (level, x, y)
    }

    /// Level whose nodes fit an object of the given size.
    ///
    /// Level `L` holds objects that fit across the root at least `2^(L - 1)`
    /// times but fewer than `2^L` times, clamped to `1..=MAX_LEVEL`.
    /// [`LayerGeometry`] caches the matching size limits of each level.
    fn level_for_size(&self, width: f64, height: f64) -> usize {
        // The bit length of `(root / size) as usize` on each axis, from the
        // exponents rather than by dividing.
        let level_x = size_level(self.root_bounds.get_width(), width);
        let level_y = size_level(self.root_bounds.get_height(), height);
        level_x.min(level_y).clamp(1, MAX_LEVEL as i64) as usize
    }

    /// Cell of `level` whose area contains `point`, clamped into the grid.
    fn cell_containing(&self, level: usize, point: (f64, f64)) -> (usize, usize) {
        let grid = &self.layers[level];
        let (inverse_width, inverse_height) = self.inverse_node_size[level];
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let x = (((point.0 - origin_x) * inverse_width) as usize).min(grid.cols() - 1);
        let y = (((point.1 - origin_y) * inverse_height) as usize).min(grid.rows() - 1);
        (x, y)
    }

    fn get_root(&self) -> &HaloOverflowBoundGridNode<T> {
        unsafe { self.layers.get_unchecked(1).get_unchecked(0, 0) }
    }

    /// Insert an item into the grid and return the handle that identifies
    /// it for [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord @ (level, x, y) = self.position(&bounds);
        self.grow_lines(&bounds, coord);
        let grid = &mut self.layers[level];
        let node = &mut grid[y][x];
        node.grow(&bounds);
        let handle = self
            .slots
            .allocate(GridLocation::new(coord, node.items.len()));
        node.items.push(Entry {
            bounds,
            item,
            handle,
        });
        self.mark_stored(level);
        self.occupy(level);
        handle
    }

    /// Remove an item from the grid.
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let (level, x, y) = location.coord();
        let entry = self.take_entry(level, x, y, location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        self.slots.free(handle);
        self.mark_taken(level, y);
        self.vacate(level);
    }

    /// Update the bounds of an item and, if necessary, its position in the grid.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let prev_coord @ (level, x, y) = location.coord();
        let i = location.index as usize;

        if self.stays_in_node(&bounds, prev_coord) {
            self.replace_bounds(prev_coord, i, bounds);
            return;
        }

        let curt_coord = self.position(&bounds);
        if curt_coord == prev_coord {
            self.replace_bounds(prev_coord, i, bounds);
            return;
        }

        let (nl, nx, ny) = curt_coord;
        let entry = self.take_entry(level, x, y, i);
        self.grow_lines(&bounds, curt_coord);
        let grid = &mut self.layers[nl];
        let new_node = &mut grid[ny][nx];
        new_node.grow(&bounds);
        *self.slots.get_mut(handle) = GridLocation::new(curt_coord, new_node.items.len());
        new_node.items.push(Entry { bounds, ..entry });
        self.mark_stored(nl);
        self.mark_taken(level, y);
        self.occupy(nl);
        self.vacate(level);
    }

    /// Overwrite the bounds of an entry that stays in its node, and grow the
    /// node's box and the spans of its column and row to cover them.
    fn replace_bounds(&mut self, coord: Coord, index: usize, bounds: Rectangle) {
        self.grow_lines(&bounds, coord);
        let (level, x, y) = coord;
        let node = &mut self.layers[level][y][x];
        node.grow(&bounds);
        node.items[index].bounds = bounds;
        self.mark_stored(level);
    }

    /// Count one more item on `level`.
    fn occupy(&mut self, level: usize) {
        self.geometry[level].object_count += 1;
        self.occupied_levels |= 1 << level;
    }

    /// Count one item fewer on `level`, and clear its bit, without a branch,
    /// if it is now empty.
    fn vacate(&mut self, level: usize) {
        self.geometry[level].object_count -= 1;
        let emptied = self.geometry[level].is_empty() as u64;
        self.occupied_levels &= !(emptied << level);
    }

    /// The coarsest level below the root that holds items, or `MAX_LEVEL + 1`
    /// if none does.
    fn coarsest_level(&self) -> usize {
        // Level zero is unused and the root is searched separately.
        ((self.occupied_levels & !0b11).trailing_zeros() as usize).min(MAX_LEVEL as usize + 1)
    }

    /// Remove the entry at `index` from a node and return it. The entry that
    /// fills the gap has its slot updated, so handles stay valid without a
    /// lookup.
    fn take_entry(&mut self, level: usize, x: usize, y: usize, index: usize) -> Entry<T> {
        let node = &mut self.layers[level][y][x];
        let entry = node.items.swap_remove(index);
        if let Some(moved) = node.items.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        if node.items.is_empty() {
            // Nothing is left to cover; searches skip the node immediately.
            node.bounds = EMPTY;
        }
        entry
    }

    /// Check cheaply whether `position` would still return `coord` for
    /// `bounds`.
    ///
    /// See [`HierarchicalLooseGrid`](crate::HierarchicalLooseGrid) for why the
    /// cached limits agree with `position` exactly.
    fn stays_in_node(&self, bounds: &Rectangle, coord: Coord) -> bool {
        let (level, x, y) = coord;
        let geometry = &self.geometry[level];
        if !geometry.has_update_shortcut() {
            return false;
        }

        let center = bounds.get_center();
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let relative_center = (center.0 - origin_x, center.1 - origin_y);
        let is_finest_layer = level == MAX_LEVEL as usize;
        geometry.matches_layer(bounds.get_width(), bounds.get_height(), is_finest_layer)
            && geometry.matches_cell(relative_center, (x, y))
            && self.world_bounds.contains_point(center)
    }

    /// Search with the bounds of every stored item, as [`search`](Self::search)
    /// would item by item, and call `callback` with the item searched for and
    /// each item found, the searched one included. The items are taken cell
    /// by cell, depth first through the levels and in Morton order within
    /// each, so that consecutive searches read nearby cells and the same
    /// ancestors. The walk stops at the finest level that holds items but
    /// visits every cell down to it, so where cells far outnumber items,
    /// searching item by item is faster. On the search bench's distribution,
    /// the `self_query` bench group measured it 15% faster than searching
    /// item by item in random order with 10,000 items, and a one-off probe
    /// outside the repository 43% with 640,000 items at the same
    /// density, where the random order misses the cache.
    pub fn for_each_self_query(&self, mut callback: impl FnMut(T, T)) {
        let layers = &self.layers;
        // Levels finer than the finest one holding items need no visit.
        let Some(finest) =
            (1..=MAX_LEVEL as usize).rev().find(|&level| !self.geometry[level].is_empty())
        else {
            return;
        };
        for_each_cell_hierarchically(
            finest,
            |level| (layers[level].cols(), layers[level].rows()),
            |level, x, y| {
                for entry in &layers[level][y][x].items {
                    self.search(&entry.bounds, |found| callback(entry.item, found));
                }
            },
        );
    }

    /// Search from the bottom up. Call `callback` for each item found.
    ///
    /// Follows [`HierarchicalLooseGrid::search`], except that the ancestor
    /// walk reads a neighbor of the query cell only if the copies of its
    /// sides in the query cell's node reach the query (judged by the spans
    /// instead in and next to the touched rows of a stale level), the scan
    /// drops edge columns and rows whose spans miss the query, and a node
    /// whose box misses the query is skipped without reading its items.
    ///
    /// [`HierarchicalLooseGrid::search`]: crate::HierarchicalLooseGrid::search
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        // Every comparison with NaN is false, so a query with a NaN corner
        // meets nothing, while for a NaN minimum the scans would report the
        // items of the nodes they take as inside it untested.
        if bounds.has_nan() {
            return;
        }
        let max_level = MAX_LEVEL as usize;
        let level = self.level_for_size(bounds.get_width(), bounds.get_height());
        let (offset_x, offset_y) = self.root_bounds.get_min();

        let layers = &self.layers;
        if level != max_level {
            // The query's corners relative to the root origin, in half nodes
            // of the finest level; every level's scan range follows by shifts.
            let half = half_node_corners(
                [
                    bounds.min_x - offset_x,
                    bounds.min_y - offset_y,
                    bounds.max_x - offset_x,
                    bounds.max_y - offset_y,
                ],
                self.inverse_node_size[max_level],
            );

            layers[level + 1..=max_level]
                .iter()
                .zip(&self.geometry[level + 1..=max_level])
                .zip(&self.lines[level + 1..=max_level])
                .rev()
                .enumerate()
                .for_each(|(shift, ((grid, geometry), lines))| {
                    if geometry.is_empty() {
                        return;
                    }
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();

                    let [min_coord_x, min_coord_y, max_coord_x, max_coord_y] =
                        scan_range(half, shift as u32, (grid_width - 1, grid_height - 1));

                    // The fixed reach overshoots by about one column and row
                    // per side; drop an edge column or row whose span misses
                    // the query. Loops dropping more measured slower on large
                    // queries.
                    let column_reaches = |x: usize| {
                        let [low, high] = unsafe { *lines.columns.get_unchecked(x) };
                        high > bounds.min_x && low < bounds.max_x
                    };
                    let row_reaches = |y: usize| {
                        let [low, high] = unsafe { *lines.rows.get_unchecked(y) };
                        high > bounds.min_y && low < bounds.max_y
                    };
                    let (mut first_x, mut end_x) = (min_coord_x, max_coord_x + 1);
                    if first_x < end_x && !column_reaches(first_x) {
                        first_x += 1;
                    }
                    if end_x > first_x && !column_reaches(end_x - 1) {
                        end_x -= 1;
                    }
                    let (mut first_y, mut end_y) = (min_coord_y, max_coord_y + 1);
                    if first_y < end_y && !row_reaches(first_y) {
                        first_y += 1;
                    }
                    if end_y > first_y && !row_reaches(end_y - 1) {
                        end_y -= 1;
                    }

                    // Scan the remaining range.
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
                                // The node lies within the query, loose bounds
                                // included, so its items need no intersection test.
                                node.iter_items(&mut callback);
                            } else {
                                node.search_items(bounds, &mut callback);
                            }
                        }
                    }
                });
        }

        if level == 1 {
            self.get_root().search_items(bounds, &mut callback);
            return;
        }

        self.search_up_3x3(level, bounds, &mut callback);
    }

    fn search_up_3x3(&self, level: usize, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        let (mut x, mut y) = self.cell_containing(level, bounds.get_center());

        // The root is searched below; levels coarser than this hold nothing.
        let coarsest = self.coarsest_level().min(level + 1);
        self.layers[coarsest..=level]
            .iter()
            .zip(&self.geometry[coarsest..=level])
            .zip(&self.lines[coarsest..=level])
            .zip(&self.marks[coarsest..=level])
            .zip(&self.halo[coarsest..=level])
            .rev()
            .for_each(|((((grid, geometry), lines), marks), halo)| {
                if !geometry.is_empty() {
                    // The unchecked reads of both searches rely on the walk
                    // staying in the grid.
                    debug_assert!(
                        x < grid.cols() && y < grid.rows(),
                        "walk cell ({x}, {y}) outside {}x{}",
                        grid.cols(),
                        grid.rows()
                    );
                    // Copies can be stale only on a stale level, so the row
                    // marks are read only there.
                    if marks.stale && lines.near_touched(y) {
                        search_3x3_by_spans(grid, lines, (x, y), bounds, callback);
                    } else {
                        search_3x3_by_copies(grid, halo, (x, y), bounds, callback);
                    }
                }
                // Advance along the ancestor path even on an empty level.
                x >>= 1;
                y >>= 1;
            });

        self.get_root().search_items(bounds, callback);
    }
}

/// Search node `(x, y)` of `grid` and those of its eight neighbors whose
/// copies in `halo` reach the query, where the copies of row `y` are not
/// stale.
#[inline]
fn search_3x3_by_copies<T: Copy>(
    grid: &Grid<HaloOverflowBoundGridNode<T>>,
    halo: &[Halo],
    (x, y): (usize, usize),
    bounds: &Rectangle,
    callback: &mut impl FnMut(T),
) {
    // SAFETY: `(x, y)` is a node of `grid`, and `halo` holds one entry
    // per node in the same order. An edge node keeps the initial copies for
    // its missing neighbors, which fail every comparison, so the indices
    // below stay inside the grid.
    unsafe {
        let node = grid.get_unchecked(y, x);
        let halo = halo.get_unchecked(y * grid.cols() + x);

        // Whether each neighbor's items may reach the query, per the copies
        // in this node. All eight are computed up front; short-circuit
        // branches here measured search 50% slower on a scene where the
        // outcomes vary.
        let before = bounds.min_x < halo.left;
        let after = bounds.max_x > halo.right;
        let up = bounds.min_y < halo.top;
        let down = bounds.max_y > halo.bottom;
        let up_before =
            (bounds.min_x < halo.top_left[0]) & (bounds.min_y < halo.top_left[1]);
        let up_after =
            (bounds.max_x > halo.top_right[0]) & (bounds.min_y < halo.top_right[1]);
        let down_before =
            (bounds.min_x < halo.bottom_left[0]) & (bounds.max_y > halo.bottom_left[1]);
        let down_after =
            (bounds.max_x > halo.bottom_right[0]) & (bounds.max_y > halo.bottom_right[1]);
        debug_assert!(!(before | up_before | down_before) || x != 0);
        debug_assert!(!(after | up_after | down_after) || x + 1 != grid.cols());
        debug_assert!(!(up | up_before | up_after) || y != 0);
        debug_assert!(!(down | down_before | down_after) || y + 1 != grid.rows());

        node.search_items(bounds, callback);
        if up_before {
            let node = grid.get_unchecked(y - 1, x - 1);
            node.search_items(bounds, callback);
        }
        if up {
            let node = grid.get_unchecked(y - 1, x);
            node.search_items(bounds, callback);
        }
        if up_after {
            let node = grid.get_unchecked(y - 1, x + 1);
            node.search_items(bounds, callback);
        }
        if before {
            let node = grid.get_unchecked(y, x - 1);
            node.search_items(bounds, callback);
        }
        if after {
            let node = grid.get_unchecked(y, x + 1);
            node.search_items(bounds, callback);
        }
        if down_before {
            let node = grid.get_unchecked(y + 1, x - 1);
            node.search_items(bounds, callback);
        }
        if down {
            let node = grid.get_unchecked(y + 1, x);
            node.search_items(bounds, callback);
        }
        if down_after {
            let node = grid.get_unchecked(y + 1, x + 1);
            node.search_items(bounds, callback);
        }
    }
}

/// Search node `(x, y)` of `grid` and those of its eight neighbors whose
/// column and row spans reach the query, where the copies of row `y` may be
/// behind the boxes; [`SpanOverflowBoundGrid`] does this on every level.
///
/// [`SpanOverflowBoundGrid`]: crate::SpanOverflowBoundGrid
#[inline]
fn search_3x3_by_spans<T: Copy>(
    grid: &Grid<HaloOverflowBoundGridNode<T>>,
    lines: &LineSpan,
    (x, y): (usize, usize),
    bounds: &Rectangle,
    callback: &mut impl FnMut(T),
) {
    let grid_width = grid.cols();
    let grid_height = grid.rows();

    // SAFETY: `(x, y)` is a node of `grid`, and each neighbor is read only
    // after testing that it exists.
    unsafe {
        // Whether the span of each neighboring column and row reaches the
        // query. Clamped indices read an in-bounds line, whose result the
        // edge tests discard. Short-circuit branches here measured search
        // 50% slower on a scene where these outcomes vary.
        let above = lines.rows.get_unchecked(y.saturating_sub(1))[1];
        let below = lines.rows.get_unchecked((y + 1).min(grid_height - 1))[0];
        let left_of = lines.columns.get_unchecked(x.saturating_sub(1))[1];
        let right_of = lines.columns.get_unchecked((x + 1).min(grid_width - 1))[0];
        let up = (y != 0) & (bounds.min_y < above);
        let down = (y + 1 != grid_height) & (bounds.max_y > below);
        let before = (x != 0) & (bounds.min_x < left_of);
        let after = (x + 1 != grid_width) & (bounds.max_x > right_of);

        if up {
            if before {
                let node = grid.get_unchecked(y - 1, x - 1);
                node.search_items(bounds, callback);
            }
            {
                let node = grid.get_unchecked(y - 1, x);
                node.search_items(bounds, callback);
            }
            if after {
                let node = grid.get_unchecked(y - 1, x + 1);
                node.search_items(bounds, callback);
            }
        }
        {
            if before {
                let node = grid.get_unchecked(y, x - 1);
                node.search_items(bounds, callback);
            }
            {
                let node = grid.get_unchecked(y, x);
                node.search_items(bounds, callback);
            }
            if after {
                let node = grid.get_unchecked(y, x + 1);
                node.search_items(bounds, callback);
            }
        }
        if down {
            if before {
                let node = grid.get_unchecked(y + 1, x - 1);
                node.search_items(bounds, callback);
            }
            {
                let node = grid.get_unchecked(y + 1, x);
                node.search_items(bounds, callback);
            }
            if after {
                let node = grid.get_unchecked(y + 1, x + 1);
                node.search_items(bounds, callback);
            }
        }
    }
}

/// Changes to a level since its last refresh.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Marks {
    /// An item was stored on, replaced on or taken from the level, so a box
    /// or span may be looser than its items and the refresh must visit it.
    touched: bool,
    /// An item was stored on or replaced on the level, so a box may have
    /// grown past the copies of its sides held by the neighbors: until the
    /// refresh, search judges the neighbors of a query cell in or next to a
    /// touched row by the spans instead.
    stale: bool,
}

impl Marks {
    /// Marks of a level an item was just stored on.
    const STORED: Self = Self {
        touched: true,
        stale: true,
    };
}

/// Extent of the items of every column along x and every row along y of one
/// level since the last refresh, as `[min, max]`; `[+infinity, -infinity]`
/// where no item has been, which meets no query.
struct LineSpan {
    columns: Box<[[f64; 2]]>,
    rows: Box<[[f64; 2]]>,
    /// Whether an item was stored in or taken from each row since the last
    /// refresh. The refresh shrinks the boxes and rebuilds the spans of these
    /// rows and pulls the copies of them and their neighbor rows again; until
    /// then, on a stale level, search avoids the copies of a touched row and
    /// of the rows next to it.
    row_touched: Box<[bool]>,
}

/// Span of a line without items, which meets no query.
const NOWHERE: [f64; 2] = [f64::INFINITY, f64::NEG_INFINITY];

impl LineSpan {
    fn new(cols: usize, rows: usize) -> Self {
        Self {
            columns: vec![NOWHERE; cols].into_boxed_slice(),
            rows: vec![NOWHERE; rows].into_boxed_slice(),
            row_touched: vec![false; rows].into_boxed_slice(),
        }
    }

    /// Whether row `y` or a row next to it is touched, the condition under
    /// which the copies of row `y` may be behind the boxes they stand for.
    #[inline]
    fn near_touched(&self, y: usize) -> bool {
        let rows = &self.row_touched;
        let last = rows.len() - 1;
        debug_assert!(y <= last);
        // SAFETY: `y` is a row, and the clamped neighbors are rows as well.
        unsafe {
            *rows.get_unchecked(y)
                | *rows.get_unchecked(y.saturating_sub(1))
                | *rows.get_unchecked((y + 1).min(last))
        }
    }

    /// Widen column `x` and row `y` to cover `bounds` stored there, and mark
    /// the row touched.
    #[inline]
    fn grow(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.row_touched[y] = true;
        self.widen((x, y), bounds);
    }

    /// Widen column `x` and row `y` to cover `bounds` without marking the
    /// row; the refresh does this for every box of a touched row.
    #[inline]
    fn widen(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.grow_column(x, bounds);
        let row = &mut self.rows[y];
        row[0] = lower(row[0], bounds.min_y);
        row[1] = higher(row[1], bounds.max_y);
    }

    /// Widen column `x` alone to cover `bounds`.
    #[inline]
    fn grow_column(&mut self, x: usize, bounds: &Rectangle) {
        let column = &mut self.columns[x];
        column[0] = lower(column[0], bounds.min_x);
        column[1] = higher(column[1], bounds.max_x);
    }

    fn clear(&mut self) {
        self.columns.fill(NOWHERE);
        self.rows.fill(NOWHERE);
    }
}

/// Sides of the boxes of the eight neighbors of a node that face the node,
/// in world coordinates: how far each neighbor's items reach towards it. The
/// initial values fail every comparison of the walk, so the copies never
/// select a neighbor that held no items at the last refresh, and the slots
/// of an edge node for its missing neighbors never change.
///
/// Absolute coordinates rather than distances past the cell edges, so that
/// the refresh copies box sides as they are and the walk compares them with
/// the query directly, with no cell edges.
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
    /// missing neighbors of an edge node, as given by the flags.
    ///
    /// # Safety
    ///
    /// `(x, y)` must be a node of `grid`, and each flag must be false if the
    /// grid has no node on that side of it.
    #[inline(always)]
    unsafe fn of_neighbors<T: Copy>(
        grid: &Grid<HaloOverflowBoundGridNode<T>>,
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

/// Box of node `(x, y)` of `grid` if `present`, else [`EMPTY`].
///
/// # Safety
///
/// `(x, y)` must be a node of `grid` when `present`.
#[inline(always)]
unsafe fn box_at<T: Copy>(
    grid: &Grid<HaloOverflowBoundGridNode<T>>,
    present: bool,
    x: usize,
    y: usize,
) -> &Rectangle {
    if present {
        unsafe { &grid.get_unchecked(y, x).bounds }
    } else {
        &EMPTY
    }
}

/// Set the copies in `halo` of every node of row `y` of `grid` to the
/// facing sides of its neighbors' boxes, which must already be shrunk in
/// rows `y - 1` to `y + 1`. Interior nodes of the row have both horizontal
/// neighbors, so only the two end nodes test for the left and right edges.
fn pull_halo_row<T: Copy>(
    grid: &Grid<HaloOverflowBoundGridNode<T>>,
    halo: &mut [Halo],
    y: usize,
) {
    let (cols, rows) = (grid.cols(), grid.rows());
    let vertical = [y != 0, y + 1 != rows];
    let row = &mut halo[y * cols..(y + 1) * cols];
    let mut pull = |x: usize, horizontal: [bool; 2]| {
        // SAFETY: `y` is a row and `x` a column of `grid`, and the flags
        // state which neighbors the node lacks.
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

/// The box comes first, as every test of the node reads it; the node's
/// copies live in the grid's `halo` array.
#[repr(C)]
struct HaloOverflowBoundGridNode<T: Copy> {
    /// Covers the bounds of every item in `items`; [`EMPTY`] while there are
    /// none.
    bounds: Rectangle,
    items: Vec<Entry<T>>,
}

impl<T: Copy> HaloOverflowBoundGridNode<T> {
    fn new() -> Self {
        Self {
            bounds: EMPTY,
            items: Vec::new(),
        }
    }

    /// Grow the box to cover `bounds`.
    #[inline]
    fn grow(&mut self, bounds: &Rectangle) {
        cover(&mut self.bounds, bounds);
    }

    /// Shrink the box to the bounds of the items stored now.
    fn refresh(&mut self) {
        let Self { bounds, items, .. } = self;
        *bounds = EMPTY;
        for entry in items.iter() {
            cover(bounds, &entry.bounds);
        }
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

/// Grow `target` to cover `bounds`. A NaN coordinate leaves its side
/// unchanged, which is harmless since an item with one intersects no query.
#[inline]
fn cover(target: &mut Rectangle, bounds: &Rectangle) {
    target.min_x = lower(target.min_x, bounds.min_x);
    target.min_y = lower(target.min_y, bounds.min_y);
    target.max_x = higher(target.max_x, bounds.max_x);
    target.max_y = higher(target.max_y, bounds.max_y);
}

unsafe impl<T: Copy, const MAX_LEVEL: u8> Send for HaloOverflowBoundGrid<T, MAX_LEVEL> {}
unsafe impl<T: Copy, const MAX_LEVEL: u8> Sync for HaloOverflowBoundGrid<T, MAX_LEVEL> {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    const MAX_LEVEL: u8 = 7;
    type Tree = HaloOverflowBoundGrid<usize, MAX_LEVEL>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];
    /// The update shortcut must place a center in the same cell as
    /// `position`, also within a few floats of every cell edge, where
    /// multiplying by the reciprocal of the node size and dividing by it can
    /// round to different cells.
    #[test]
    fn update_shortcut_agrees_with_position_at_cell_edges() {
        for world in WORLDS {
            for tree in [Tree::new(world.clone()), Tree::new_fit(world.clone())] {
                let origin = tree.root_bounds.get_min();
                for level in 2..=MAX_LEVEL as usize {
                    let geometry = &tree.geometry[level];
                    let grid = &tree.layers[level];
                    let (node_width, node_height) = (geometry.node_width, geometry.node_height);
                    let inside = (origin.0 + node_width * 0.5, origin.1 + node_height * 0.5);
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
                    for i in 1..grid.cols() {
                        for x in near(origin.0 + node_width * i as f64) {
                            centers.push((x, inside.1));
                        }
                    }
                    for i in 1..grid.rows() {
                        for y in near(origin.1 + node_height * i as f64) {
                            centers.push((inside.0, y));
                        }
                    }
                    for center in centers {
                        let cell = tree.cell_containing(level, center);
                        let relative = (center.0 - origin.0, center.1 - origin.1);
                        assert!(
                            geometry.matches_cell(relative, cell),
                            "world {world:?}, level {level}, center {center:?}, cell {cell:?}"
                        );
                    }
                }
            }
        }
    }


    /// Bounds of any size from the root down to below the finest node,
    /// centered anywhere in the root and therefore sometimes outside the
    /// world.
    fn random_bounds(rng: &mut StdRng, root: &Rectangle) -> Rectangle {
        let scale = 2f64.powf(-rng.gen_range(0.0..MAX_LEVEL as f64 + 2.0));
        let width = root.get_width() * scale * rng.gen_range(0.5..1.0);
        let height = root.get_height() * scale * rng.gen_range(0.5..1.0);
        let (x, y) = root.get_random_point(rng);
        Rectangle::center_rect(x, y, width, height)
    }

    /// A small move or resize that usually keeps the item in its node.
    fn nudge(rng: &mut StdRng, bounds: &Rectangle) -> Rectangle {
        let (x, y) = bounds.get_center();
        let (width, height) = (bounds.get_width(), bounds.get_height());
        Rectangle::center_rect(
            x + width * rng.gen_range(-0.2..0.2),
            y + height * rng.gen_range(-0.2..0.2),
            width * rng.gen_range(0.8..1.2),
            height * rng.gen_range(0.8..1.2),
        )
    }

    /// Union of the bounds of the items of `node`; `EMPTY` if it has none.
    fn union<T: Copy>(node: &HaloOverflowBoundGridNode<T>) -> Rectangle {
        let side = |pick: fn(&Rectangle) -> f64, start: f64, fold: fn(f64, f64) -> f64| {
            node.items
                .iter()
                .map(|entry| pick(&entry.bounds))
                .fold(start, fold)
        };
        Rectangle::new(
            side(|b| b.min_x, f64::INFINITY, f64::min),
            side(|b| b.min_y, f64::INFINITY, f64::min),
            side(|b| b.max_x, f64::NEG_INFINITY, f64::max),
            side(|b| b.max_y, f64::NEG_INFINITY, f64::max),
        )
    }

    /// Every box must cover the items of its node and, if `exact`, equal
    /// their union. A node without items must have the empty box either way.
    fn assert_boxes(tree: &Tree, exact: bool) {
        for (level, grid) in tree.layers.iter().enumerate().skip(1) {
            for (index, node) in grid.iter().enumerate() {
                let union = union(node);
                let context = format!("node {index} of level {level}");
                if exact || node.items.is_empty() {
                    assert_eq!(node.bounds, union, "{context}");
                } else {
                    let b = &node.bounds;
                    assert!(
                        b.min_x <= union.min_x
                            && b.min_y <= union.min_y
                            && b.max_x >= union.max_x
                            && b.max_y >= union.max_y,
                        "{context}: box {b:?} misses {union:?}"
                    );
                }
            }
        }
    }

    /// Run random insertions, small and large updates and removals in every
    /// world, square and fitted. Every 500 steps, refresh and call `check`
    /// with `refreshed` set; every 100 steps and at the end, call it with
    /// `refreshed` unset.
    fn exercise(seed: u64, mut check: impl FnMut(&Tree, usize, bool)) {
        let mut rng = StdRng::seed_from_u64(seed);
        for world in WORLDS {
            for mut tree in [Tree::new(world.clone()), Tree::new_fit(world.clone())] {
                let root = tree.get_bounds().clone();
                let mut items: Vec<(ItemHandle, Rectangle)> = Vec::new();
                for step in 0..4000 {
                    match rng.gen_range(0..5) {
                        0 | 1 => {
                            let bounds = random_bounds(&mut rng, &root);
                            items.push((tree.insert(bounds.clone(), step), bounds));
                        }
                        2 | 3 if !items.is_empty() => {
                            let index = rng.gen_range(0..items.len());
                            let (handle, old) = &items[index];
                            let bounds = if rng.gen_bool(0.7) {
                                nudge(&mut rng, old)
                            } else {
                                random_bounds(&mut rng, &root)
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
                    if step % 500 == 499 {
                        tree.refresh_bounds();
                        check(&tree, step, true);
                    } else if step % 100 == 0 {
                        check(&tree, step, false);
                    }
                }
                check(&tree, 4000, false);
            }
        }
    }

    #[test]
    fn boxes_cover_items_until_refreshed() {
        exercise(0, |tree, _, refreshed| assert_boxes(tree, refreshed));
    }

    /// Largest overflow of the items of `level` beyond their cells, computed
    /// item by item: negative when all keep off that edge, -infinity without
    /// items.
    fn exact_overflow(tree: &Tree, level: usize) -> [f64; 4] {
        let geometry = &tree.geometry[level];
        let node_size = (geometry.node_width, geometry.node_height);
        let grid = &tree.layers[level];
        let mut largest = [f64::NEG_INFINITY; 4];
        for y in 0..grid.rows() {
            for x in 0..grid.cols() {
                for entry in &grid[y][x].items {
                    let overflow =
                        cell_overflow(&entry.bounds, tree.root_bounds.get_min(), node_size, (x, y));
                    for (largest, overflow) in largest.iter_mut().zip(overflow) {
                        *largest = largest.max(overflow);
                    }
                }
            }
        }
        largest
    }

    fn directions(bound: DirectionalOverflow) -> [f64; 4] {
        [bound.left, bound.right, bound.top, bound.bottom]
    }

    /// The bound of every level must cover the overflow of all its items at
    /// any time, and equal the largest one right after a refresh.
    #[test]
    fn level_bounds_cover_items_until_refreshed() {
        exercise(1, |tree, step, refreshed| {
            for level in 2..=MAX_LEVEL as usize {
                let exact = exact_overflow(tree, level);
                let bound = directions(tree.overflow_bound(level));
                for direction in 0..4 {
                    let context = format!(
                        "step {step}, level {level}, direction {direction}: bound {} for overflow {}",
                        bound[direction], exact[direction]
                    );
                    if refreshed {
                        assert_eq!(bound[direction], exact[direction], "{context}");
                    } else {
                        assert!(bound[direction] >= exact[direction], "{context}");
                    }
                }
            }
        });
    }

    /// Extent of the items of every column along x and every row along y of
    /// `level`, computed item by item; `[+infinity, -infinity]` where a
    /// column or row holds no items.
    fn exact_spans(tree: &Tree, level: usize) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
        let grid = &tree.layers[level];
        let mut columns = vec![[f64::INFINITY, f64::NEG_INFINITY]; grid.cols()];
        let mut rows = vec![[f64::INFINITY, f64::NEG_INFINITY]; grid.rows()];
        for y in 0..grid.rows() {
            for x in 0..grid.cols() {
                for entry in &grid[y][x].items {
                    let b = &entry.bounds;
                    columns[x] = [columns[x][0].min(b.min_x), columns[x][1].max(b.max_x)];
                    rows[y] = [rows[y][0].min(b.min_y), rows[y][1].max(b.max_y)];
                }
            }
        }
        (columns, rows)
    }

    /// The span of every column and row must cover its items at any time, and
    /// be exactly their extent right after a refresh.
    #[test]
    fn line_spans_cover_items_until_refreshed() {
        exercise(4, |tree, step, refreshed| {
            for level in 2..=MAX_LEVEL as usize {
                let (columns, rows) = exact_spans(tree, level);
                let lines = &tree.lines[level];
                for (kind, spans, exact) in [
                    ("column", &lines.columns, &columns),
                    ("row", &lines.rows, &rows),
                ] {
                    assert_eq!(spans.len(), exact.len());
                    for (line, (span, exact)) in spans.iter().zip(exact).enumerate() {
                        let context = format!(
                            "step {step}, level {level}, {kind} {line}: span {span:?} for items {exact:?}"
                        );
                        if refreshed {
                            assert_eq!(span, exact, "{context}");
                        } else {
                            assert!(span[0] <= exact[0] && span[1] >= exact[1], "{context}");
                        }
                    }
                }
            }
        });
    }

    /// Expected copies of every node of `level` right after a refresh,
    /// computed item by item from the neighbors, in storage order.
    fn exact_halo(tree: &Tree, level: usize) -> Vec<Halo> {
        let grid = &tree.layers[level];
        let (cols, rows) = (grid.cols(), grid.rows());
        let unions: Vec<Rectangle> = grid.iter().map(union).collect();
        let union_at = |x: isize, y: isize| -> Rectangle {
            if x < 0 || y < 0 || x as usize >= cols || y as usize >= rows {
                EMPTY
            } else {
                unions[y as usize * cols + x as usize].clone()
            }
        };
        let mut exact = Vec::with_capacity(cols * rows);
        for y in 0..rows as isize {
            for x in 0..cols as isize {
                let left = union_at(x - 1, y);
                let right = union_at(x + 1, y);
                let top = union_at(x, y - 1);
                let bottom = union_at(x, y + 1);
                let top_left = union_at(x - 1, y - 1);
                let top_right = union_at(x + 1, y - 1);
                let bottom_left = union_at(x - 1, y + 1);
                let bottom_right = union_at(x + 1, y + 1);
                exact.push(Halo {
                    left: left.max_x,
                    right: right.min_x,
                    top: top.max_y,
                    bottom: bottom.min_y,
                    top_left: [top_left.max_x, top_left.max_y],
                    top_right: [top_right.min_x, top_right.max_y],
                    bottom_left: [bottom_left.max_x, bottom_left.min_y],
                    bottom_right: [bottom_right.min_x, bottom_right.min_y],
                });
            }
        }
        exact
    }

    /// Whether `copies` reach at least as far as `exact` on every slot.
    fn covers(copies: &Halo, exact: &Halo) -> bool {
        copies.left >= exact.left
            && copies.right <= exact.right
            && copies.top >= exact.top
            && copies.bottom <= exact.bottom
            && copies.top_left[0] >= exact.top_left[0]
            && copies.top_left[1] >= exact.top_left[1]
            && copies.top_right[0] <= exact.top_right[0]
            && copies.top_right[1] >= exact.top_right[1]
            && copies.bottom_left[0] >= exact.bottom_left[0]
            && copies.bottom_left[1] <= exact.bottom_left[1]
            && copies.bottom_right[0] <= exact.bottom_right[0]
            && copies.bottom_right[1] <= exact.bottom_right[1]
    }

    /// The copies in every node must cover the items of its neighbors
    /// whenever neither its row nor a row next to it is touched, equal their
    /// boxes right after a refresh, which clears the level and row marks, and
    /// never change for the missing neighbors of an edge node.
    #[test]
    fn halo_covers_neighbors_until_refreshed() {
        exercise(5, |tree, step, refreshed| {
            for (level, grid) in tree.layers.iter().enumerate().skip(1) {
                let (cols, rows) = (grid.cols(), grid.rows());
                let marks = tree.marks[level];
                let lines = &tree.lines[level];
                if refreshed {
                    assert_eq!(marks, Marks::default(), "step {step}, level {level}");
                    assert!(
                        lines.row_touched.iter().all(|&touched| !touched),
                        "step {step}, level {level}: {:?}",
                        lines.row_touched
                    );
                }
                let exact = exact_halo(tree, level);
                let copies = &tree.halo[level];
                assert_eq!(copies.len(), cols * rows);
                for (index, (copies, exact)) in copies.iter().zip(&exact).enumerate() {
                    let (x, y) = (index % cols, index / cols);
                    let context =
                        format!("step {step}, level {level}, node {index}: {copies:?} for {exact:?}");
                    if refreshed {
                        assert_eq!(copies, exact, "{context}");
                    } else if !lines.near_touched(y) {
                        assert!(covers(copies, exact), "{context}");
                    }
                    let initial = &NO_HALO;
                    if x == 0 {
                        assert_eq!(copies.left, initial.left, "{context}");
                        assert_eq!(copies.top_left, initial.top_left, "{context}");
                        assert_eq!(copies.bottom_left, initial.bottom_left, "{context}");
                    }
                    if x + 1 == cols {
                        assert_eq!(copies.right, initial.right, "{context}");
                        assert_eq!(copies.top_right, initial.top_right, "{context}");
                        assert_eq!(copies.bottom_right, initial.bottom_right, "{context}");
                    }
                    if y == 0 {
                        assert_eq!(copies.top, initial.top, "{context}");
                        assert_eq!(copies.top_left, initial.top_left, "{context}");
                        assert_eq!(copies.top_right, initial.top_right, "{context}");
                    }
                    if y + 1 == rows {
                        assert_eq!(copies.bottom, initial.bottom, "{context}");
                        assert_eq!(copies.bottom_left, initial.bottom_left, "{context}");
                        assert_eq!(copies.bottom_right, initial.bottom_right, "{context}");
                    }
                }
            }
        });
    }

    /// Coarsest level below the root that holds items, `MAX_LEVEL + 1` if none.
    fn coarsest_occupied(tree: &Tree) -> usize {
        (2..=MAX_LEVEL as usize)
            .find(|&level| !tree.geometry[level].is_empty())
            .unwrap_or(MAX_LEVEL as usize + 1)
    }

    /// `coarsest_level()` must be the coarsest level below the root that holds
    /// items after every insertion, update and removal, refreshed or not. With
    /// at most nine items, levels keep emptying and filling again.
    #[test]
    fn coarsest_level_follows_the_items() {
        let mut rng = StdRng::seed_from_u64(5);
        for world in WORLDS {
            for mut tree in [Tree::new(world.clone()), Tree::new_fit(world.clone())] {
                let root = tree.get_bounds().clone();
                let mut handles: Vec<ItemHandle> = Vec::new();
                let mut raised = 0;
                for step in 0..3000 {
                    let before = tree.coarsest_level();
                    let operation = match handles.len() {
                        0 | 1 => 0,
                        9.. => 2,
                        _ => rng.gen_range(0..3),
                    };
                    match operation {
                        0 => {
                            let bounds = random_bounds(&mut rng, &root);
                            handles.push(tree.insert(bounds, step));
                        }
                        1 => {
                            let bounds = random_bounds(&mut rng, &root);
                            tree.update(bounds, handles[rng.gen_range(0..handles.len())]);
                        }
                        _ => tree.remove(handles.swap_remove(rng.gen_range(0..handles.len()))),
                    }
                    assert_eq!(tree.coarsest_level(), coarsest_occupied(&tree), "step {step}");
                    raised += (tree.coarsest_level() > before) as usize;
                    if rng.gen_bool(0.1) {
                        tree.refresh_bounds();
                        assert_eq!(
                            tree.coarsest_level(),
                            coarsest_occupied(&tree),
                            "step {step}, refreshed"
                        );
                    }
                }
                // The coarsest level emptied out often enough to exercise clearing its bit.
                assert!(raised > 100, "the coarsest level emptied only {raised} times");
            }
        }
    }

    /// Storing an item marks its level touched and stale, taking one marks
    /// it touched, and a refresh clears both; an unchanged level stays clear.
    /// The spans and copies of a clear level are exact.
    #[test]
    fn marks_follow_stores_and_takes() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let finest = MAX_LEVEL as usize;
        let small = Rectangle::center_rect(100.0, 100.0, 4.0, 4.0);
        let big = Rectangle::center_rect(100.0, 100.0, 200.0, 200.0);
        let coarser = tree.position(&big).0;
        assert!(coarser < finest);
        let clear = |tree: &Tree| tree.marks.iter().all(|&marks| marks == Marks::default());

        assert!(clear(&tree));
        let handle = tree.insert(small.clone(), 0);
        assert_eq!(tree.marks[finest], Marks::STORED);
        tree.refresh_bounds();
        assert!(clear(&tree));

        // Replaced in its node.
        tree.update(Rectangle::center_rect(101.0, 100.0, 4.0, 4.0), handle);
        assert_eq!(tree.marks[finest], Marks::STORED);
        tree.refresh_bounds();
        assert!(clear(&tree));

        // Moved to another level: stored there, taken here.
        tree.update(big.clone(), handle);
        assert_eq!(tree.marks[coarser], Marks::STORED);
        assert_eq!(
            tree.marks[finest],
            Marks {
                touched: true,
                stale: false
            }
        );
        tree.refresh_bounds();
        assert!(clear(&tree));

        tree.remove(handle);
        assert_eq!(
            tree.marks[coarser],
            Marks {
                touched: true,
                stale: false
            }
        );
        assert_eq!(tree.marks[finest], Marks::default());
        tree.refresh_bounds();
        assert!(clear(&tree));
    }

    /// Storing an item in a row or taking one from it marks the row; the
    /// rows next to a marked row, within the grid, count as near it; a
    /// refresh clears the marks of every row.
    #[test]
    fn row_marks_follow_stores_and_takes() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let finest = MAX_LEVEL as usize;
        // Finest nodes are 16 units wide; this one is (6, 6).
        let item = Rectangle::center_rect(100.0, 100.0, 4.0, 4.0);
        assert_eq!(tree.position(&item), (finest, 6, 6));
        let marked = |tree: &Tree| -> Vec<usize> {
            (0..tree.lines[finest].row_touched.len())
                .filter(|&y| tree.lines[finest].row_touched[y])
                .collect()
        };
        let near = |tree: &Tree| -> Vec<usize> {
            (0..tree.lines[finest].row_touched.len())
                .filter(|&y| tree.lines[finest].near_touched(y))
                .collect()
        };

        let handle = tree.insert(item.clone(), 0);
        assert_eq!(marked(&tree), [6]);
        assert_eq!(near(&tree), [5, 6, 7]);
        tree.refresh_bounds();
        assert_eq!(marked(&tree), []);
        assert_eq!(near(&tree), []);

        // Moved to the row below: stored there, taken here.
        tree.update(Rectangle::center_rect(100.0, 116.0, 4.0, 4.0), handle);
        assert_eq!(marked(&tree), [6, 7]);
        assert_eq!(near(&tree), [5, 6, 7, 8]);
        tree.refresh_bounds();

        tree.remove(handle);
        assert_eq!(marked(&tree), [7]);
        tree.refresh_bounds();
        assert_eq!(marked(&tree), []);

        // At the edge of the grid, the neighborhood ends with the grid.
        tree.insert(Rectangle::center_rect(100.0, 4.0, 4.0, 4.0), 1);
        assert_eq!(marked(&tree), [0]);
        assert_eq!(near(&tree), [0, 1]);
    }

    /// An item that grows or moves towards any of the eight neighbors of its
    /// node must be found before the next refresh by a query centered in
    /// that neighbor, whose copies do not reflect it yet.
    #[test]
    fn item_reaching_into_a_neighbor_is_found_before_refresh() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        // Finest nodes are 16 units wide; (6, 6) spans 96..112.
        let home = Rectangle::center_rect(100.0, 100.0, 4.0, 4.0);
        let far = Rectangle::center_rect(500.0, 500.0, 4.0, 4.0);
        assert_eq!(tree.position(&home), (MAX_LEVEL as usize, 6, 6));
        for dy in -1..=1isize {
            for dx in -1..=1isize {
                if (dx, dy) == (0, 0) {
                    continue;
                }
                // As large as the node, centered 7 units from its center
                // towards the neighbor, and so reaching 1 unit short of the
                // neighbor's center: 103..119 for dx = 1, against 112..128.
                let reaching = Rectangle::center_rect(
                    104.0 + 7.0 * dx as f64,
                    104.0 + 7.0 * dy as f64,
                    16.0,
                    16.0,
                );
                let query = Rectangle::center_rect(
                    104.0 + 12.0 * dx as f64,
                    104.0 + 12.0 * dy as f64,
                    2.0,
                    2.0,
                );
                let context = format!("towards ({dx}, {dy})");
                assert_eq!(tree.position(&reaching), (MAX_LEVEL as usize, 6, 6), "{context}");
                assert_eq!(
                    tree.position(&query),
                    (MAX_LEVEL as usize, (6 + dx) as usize, (6 + dy) as usize),
                    "{context}"
                );

                // Grown within its node, or moved in from another node.
                for before in [&home, &far] {
                    let mut found = Vec::new();
                    let handle = tree.insert(before.clone(), 0);
                    tree.refresh_bounds();
                    tree.search(&query, |id| found.push(id));
                    assert_eq!(found, [], "{context}, {before:?}");
                    tree.update(reaching.clone(), handle);
                    tree.search(&query, |id| found.push(id));
                    assert_eq!(found, [0], "{context}, from {before:?}");
                    tree.refresh_bounds();
                    found.clear();
                    tree.search(&query, |id| found.push(id));
                    assert_eq!(found, [0], "{context}, from {before:?}, refreshed");
                    tree.remove(handle);
                    tree.refresh_bounds();
                }
            }
        }
    }

    /// An item that grows into a level coarser than any other must be found
    /// before the next refresh, also by queries on the finest level.
    #[test]
    fn item_moved_to_a_coarser_level_is_found_before_refresh() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let handle = tree.insert(Rectangle::center_rect(100.0, 100.0, 4.0, 4.0), 0);
        tree.refresh_bounds();

        let big = Rectangle::center_rect(100.0, 100.0, 200.0, 200.0);
        tree.update(big.clone(), handle);
        assert!(tree.position(&big).0 < MAX_LEVEL as usize);

        let query = Rectangle::center_rect(150.0, 150.0, 2.0, 2.0);
        assert_eq!(tree.position(&query).0, MAX_LEVEL as usize);
        let mut found = Vec::new();
        tree.search(&query, |id| found.push(id));
        assert_eq!(found, vec![0]);
    }

    /// Items that keep off their cell edges give negative bounds, before and
    /// after a refresh. Searches must still find exactly the intersecting
    /// items, including those that overlap a query by a single ulp.
    #[test]
    fn negative_level_bounds_keep_searches_exact() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        // The finest nodes are 16 units wide; every item stays at least 1.5
        // units inside its node.
        let mut items = Vec::new();
        for y in 0..64 {
            for x in 0..64 {
                let cx = (x as f64 + 0.5) * 16.0 + rng.gen_range(-2.0..2.0);
                let cy = (y as f64 + 0.5) * 16.0 + rng.gen_range(-2.0..2.0);
                let (width, height) = (rng.gen_range(3.0..9.0), rng.gen_range(3.0..9.0));
                let bounds = Rectangle::center_rect(cx, cy, width, height);
                tree.insert(bounds.clone(), items.len());
                items.push(bounds);
            }
        }

        for refreshed in [false, true] {
            if refreshed {
                tree.refresh_bounds();
            }
            let bound = directions(tree.overflow_bound(MAX_LEVEL as usize));
            assert!(
                bound.iter().all(|&reach| reach < -1.0),
                "refreshed {refreshed}: {bound:?}"
            );
            for round in 0..3000 {
                let query = if round % 2 == 0 {
                    let (x, y) = (rng.gen_range(-20.0..1044.0), rng.gen_range(-20.0..1044.0));
                    let scale = [0.5, 4.0, 16.0, 64.0][rng.gen_range(0..4)];
                    let (width, height) = (
                        scale * rng.gen_range(0.5..2.0),
                        scale * rng.gen_range(0.5..2.0),
                    );
                    Rectangle::center_rect(x, y, width, height)
                } else {
                    let b = &items[rng.gen_range(0..items.len())];
                    let width = b.get_width() * rng.gen_range(0.1..8.0);
                    let height = b.get_height() * rng.gen_range(0.1..8.0);
                    match rng.gen_range(0..4) {
                        0 => Rectangle::new(b.max_x.next_down(), b.min_y, b.max_x + width, b.max_y),
                        1 => Rectangle::new(b.min_x - width, b.min_y, b.min_x.next_up(), b.max_y),
                        2 => {
                            Rectangle::new(b.min_x, b.max_y.next_down(), b.max_x, b.max_y + height)
                        }
                        _ => Rectangle::new(b.min_x, b.min_y - height, b.max_x, b.min_y.next_up()),
                    }
                };
                let expected: Vec<usize> = (0..items.len())
                    .filter(|&i| items[i].intersects(&query))
                    .collect();
                let mut found = Vec::new();
                tree.search(&query, |id| found.push(id));
                found.sort_unstable();
                assert_eq!(found, expected, "refreshed {refreshed}, query {query:?}");
            }
        }
    }

    /// Boxes tighten only in `refresh_bounds`, except that a node left
    /// without items resets its box immediately.
    #[test]
    fn refresh_shrinks_boxes_and_emptied_nodes_reset_them() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let wide = Rectangle::center_rect(100.0, 100.0, 10.0, 10.0);
        let narrow = Rectangle::center_rect(100.0, 100.0, 8.0, 8.0);
        let handle = tree.insert(wide.clone(), 0);
        let coord @ (level, x, y) = tree.position(&wide);
        assert_eq!(
            tree.position(&narrow),
            coord,
            "both bounds belong to one node"
        );

        tree.update(narrow.clone(), handle);
        assert_eq!(tree.layers[level][y][x].bounds, wide);
        tree.refresh_bounds();
        assert_eq!(tree.layers[level][y][x].bounds, narrow);

        tree.remove(handle);
        assert_eq!(tree.layers[level][y][x].bounds, EMPTY);
    }

    /// Items with a NaN coordinate, which meet no query, go to the root. Their
    /// other sides grow its box, but no side of it, nor of the spans of its
    /// column and row, becomes NaN, whichever comes first and before and
    /// after a refresh; the box therefore still meets queries, and searches
    /// find the other items there. The root's spans are never read.
    #[test]
    fn nan_items_leave_the_root_box_to_the_others() {
        // Centered outside the world, so stored in the root.
        let outside = Rectangle::center_rect(-10.0, 100.0, 4.0, 4.0);
        let nan = [
            Rectangle::new(f64::NAN, 90.0, -5.0, 110.0),
            Rectangle::new(-15.0, f64::NAN, f64::NAN, 102.0),
            Rectangle::new(f64::NAN, f64::NAN, f64::NAN, f64::NAN),
        ];
        let query = Rectangle::center_rect(-10.0, 100.0, 2.0, 2.0);
        for nan_first in [true, false] {
            let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
            let mut handles = Vec::new();
            if nan_first {
                handles.extend(nan.iter().map(|bounds| tree.insert(bounds.clone(), 1)));
            }
            tree.insert(outside.clone(), 0);
            if !nan_first {
                handles.extend(nan.iter().map(|bounds| tree.insert(bounds.clone(), 1)));
            }
            // Replaced in place by other NaN bounds.
            tree.update(nan[2].clone(), handles[0]);
            assert_eq!(tree.position(&outside), (1, 0, 0));
            for refreshed in [false, true] {
                if refreshed {
                    tree.refresh_bounds();
                }
                let context = format!("NaN first {nan_first}, refreshed {refreshed}");
                let root = &tree.layers[1][0][0].bounds;
                let lines = &tree.lines[1];
                let sides = [root.min_x, root.min_y, root.max_x, root.max_y];
                let spans = [lines.columns[0], lines.rows[0]].concat();
                assert!(!sides.iter().chain(&spans).any(|side| side.is_nan()), "{context}: {root:?}, {spans:?}");
                // The sides the items have, as `f64::min` and `max` fold them.
                let exact = union(&tree.layers[1][0][0]);
                if refreshed {
                    assert_eq!(*root, exact, "{context}");
                } else {
                    assert!(
                        root.min_x <= exact.min_x
                            && root.min_y <= exact.min_y
                            && root.max_x >= exact.max_x
                            && root.max_y >= exact.max_y,
                        "{context}: {root:?} misses {exact:?}"
                    );
                }
                let mut found = Vec::new();
                tree.search(&query, |id| found.push(id));
                assert_eq!(found, [0], "{context}");
            }
        }
    }
}
