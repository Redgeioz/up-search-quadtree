use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::cell_order::for_each_cell_hierarchically;
use crate::layer_geometry::{
    LayerGeometry, Levels, build_levels, half_node_corners, scan_range, size_level,
};
use crate::exact_overflow_bound_grid::DirectionalOverflow;
use crate::overflow_bound::{cell_overflow, higher, lower};
use crate::rect::{EMPTY, NOWHERE, Rectangle};
use grid::*;

/// `(level, x, y)` of a node.
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<SpanOverflowBoundGridNode<T>>>;

/// [`LazyOverflowBoundGrid`] with the bound of every level refined to its
/// columns and rows: the storage, positioning and handles of [`HierarchicalLooseGrid`],
/// where every node keeps a box covering the bounds of its items, and every
/// column and row of a level the span of its items along its own axis, from
/// the smallest to the largest x of a column's items and likewise in y for a
/// row. Search reads a neighboring column or row within the fixed reach of the
/// hierarchical loose grid only if its span meets the query, which leaves out
/// whole columns and rows without reading their nodes, and then skips each
/// node whose box misses the query without reading its items.
///
/// A column or row without items has an empty span, so sparse levels leave
/// out most of their neighbors. On dense levels the spans still stop short of
/// the cell edges where every item keeps off them; otherwise the boxes do the
/// work.
///
/// Insertion and update only widen the box of the target node and the spans
/// of its column and row. Removal and moving out leave them unchanged, except
/// that an emptied node resets its box. Boxes and spans thus stay valid as
/// items move, shrink or leave, and tighten only in [`refresh_bounds`], which
/// the caller invokes when convenient, such as once per frame after the
/// updates. Stale spans cost at most what the fixed reach costs; the boxes
/// still apply.
///
/// The ancestor walk also stops at the coarsest level below the root that
/// holds items, which a bitmask of the occupied levels keeps exact at all
/// times.
///
/// Boxes and spans consist of item coordinates, so their checks compare the
/// same values as the intersection tests and need no rounding margin. Neither
/// needs cell edges, which also keeps them cheap to maintain.
///
/// [`LazyOverflowBoundGrid`]: crate::LazyOverflowBoundGrid
/// [`HierarchicalLooseGrid`]: crate::HierarchicalLooseGrid
/// [`refresh_bounds`]: Self::refresh_bounds
pub struct SpanOverflowBoundGrid<T: Copy, const MAX_LEVEL: u8> {
    layers: Layers<T>,
    /// Indexed by level, parallel to `layers`.
    geometry: Vec<LayerGeometry>,
    /// Reciprocals of the node width and height, indexed by level. Position
    /// and search multiply by them instead of dividing by the node size, and
    /// `geometry` builds its cell edges with the same arithmetic.
    inverse_node_size: Vec<(f64, f64)>,
    /// Indexed by level, parallel to `layers`.
    lines: Vec<LineSpan>,
    /// Bit `L` is set while level `L` holds items: storing an item sets the
    /// bit of its level, and taking the last item off a level clears it. The
    /// coarsest level below the root that holds items, where the ancestor walk
    /// of search stops, is therefore exact at all times, not only after a
    /// refresh. With all items on the finest level, stopping the walk there
    /// measured 10% faster than walking every level.
    occupied_levels: u64,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    /// Where the entry of each live handle is stored.
    slots: Slots<GridLocation>,
}

impl<T: Copy, const MAX_LEVEL: u8> SpanOverflowBoundGrid<T, MAX_LEVEL> {
    /// Create a new grid with the given bounds.
    ///
    /// Nodes are square, which usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> SpanOverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new grid with the given bounds.
    ///
    /// The root is `world_bounds` itself, so nodes take its aspect ratio
    /// instead of being square.
    pub fn new_fit(world_bounds: Rectangle) -> SpanOverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(
        world_bounds: Rectangle,
    ) -> SpanOverflowBoundGrid<T, MAX_LEVEL> {
        let Levels {
            layers,
            root_bounds,
            geometry,
            inverse_node_size,
        } = build_levels::<_, FIT>(&world_bounds, MAX_LEVEL, SpanOverflowBoundGridNode::new);

        // One per level; the root's spans grow but are never read, as search
        // reads the root whole.
        let lines = layers
            .iter()
            .map(|grid| LineSpan::new(grid.cols(), grid.rows()))
            .collect();

        SpanOverflowBoundGrid {
            layers,
            geometry,
            inverse_node_size,
            lines,
            occupied_levels: 0,
            root_bounds,
            world_bounds,
            slots: Slots::new(),
        }
    }

    /// Return the actual bounds of the root.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Largest overflow per direction beyond the cells of `level` since the
    /// last [`refresh_bounds`](Self::refresh_bounds), in world units, as
    /// implied by the spans of its columns and rows: negative when every item
    /// keeps that far off its cell edge, and -infinity on a level that has
    /// held no items since then. Search uses the spans directly.
    ///
    /// Panics unless `level` is in `2..=MAX_LEVEL`; the root has no neighbors
    /// to bound.
    pub fn overflow_bound(&self, level: usize) -> DirectionalOverflow {
        assert!((2..=MAX_LEVEL as usize).contains(&level), "invalid level");
        let geometry = &self.geometry[level];
        let node_size = (geometry.node_width, geometry.node_height);
        let origin = self.root_bounds.get_min();
        let lines = &self.lines[level];
        // Cell edges as `cell_overflow` computes them, so the bound matches it
        // exactly, item by item.
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

    /// On every touched level, shrink the box of every node and the span of
    /// every column and row to the items held now, undoing the growth from
    /// items that have since moved, shrunk or left. Visits every node of a
    /// touched level once, and the items of its touched rows; untouched
    /// levels are already exact and are skipped.
    ///
    /// A refreshed box is the union of its items, so the spans follow from
    /// the boxes, one per node rather than one per item.
    pub fn refresh_bounds(&mut self) {
        for level in 1..=MAX_LEVEL as usize {
            let geometry = &self.geometry[level];
            let lines = &mut self.lines[level];
            if !std::mem::take(&mut lines.touched) {
                continue;
            }
            // Every node of a level without items has already reset its box.
            if geometry.is_empty() {
                lines.clear();
                lines.row_touched.fill(false);
                continue;
            }
            let grid = &mut self.layers[level];
            // A column span crosses every row, so it is rebuilt from the
            // boxes of all rows; the boxes and span of a row are rebuilt only
            // if the row is touched.
            lines.columns.fill(NOWHERE);
            for y in 0..grid.rows() {
                if std::mem::take(&mut lines.row_touched[y]) {
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
                            lines.widen_column(x, &node.bounds);
                        }
                    }
                }
            }
        }
    }

    /// Widen the spans of the column and row of `coord` to cover `bounds`
    /// stored there, and mark the level and row touched. Every store runs
    /// it; left to the inliner, it became a call once `cover` and the spans
    /// grew with `lower` and `higher`, which cost the update path most of
    /// what those saved.
    #[inline(always)]
    fn grow_lines(&mut self, bounds: &Rectangle, (level, x, y): Coord) {
        self.lines[level].grow((x, y), bounds);
    }

    /// Find the `(level, x, y)` of the node an item with `bounds` belongs in.
    pub fn position(&self, bounds: &Rectangle) -> Coord {
        let center = bounds.get_center();
        if !self.world_bounds.contains_point(center) {
            return (1, 0, 0);
        }

        let level = self.level_for_size(bounds.get_width(), bounds.get_height());
        let (x, y) = self.cell_containing(level, center);
        (level, x, y)
    }

    /// Level whose nodes fit an item of the given size.
    ///
    /// Level `L` holds items that fit across the root at least `2^(L - 1)`
    /// and fewer than `2^L` times, clamped to `1..=MAX_LEVEL`.
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

    fn get_root(&self) -> &SpanOverflowBoundGridNode<T> {
        unsafe { self.layers.get_unchecked(1).get_unchecked(0, 0) }
    }

    /// Insert an item into the grid and return the handle that identifies
    /// it for [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord @ (level, x, y) = self.position(&bounds);
        self.grow_lines(&bounds, coord);
        let node = &mut self.layers[level][y][x];
        node.grow(&bounds);
        let handle = self
            .slots
            .allocate(GridLocation::new(coord, node.items.len()));
        node.items.push(Entry {
            bounds,
            item,
            handle,
        });
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
        self.lines[level].mark_taken(y);
        self.vacate(level);
    }

    /// Update the bounds of an item and, if necessary, its node.
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
        let new_node = &mut self.layers[nl][ny][nx];
        new_node.grow(&bounds);
        *self.slots.get_mut(handle) = GridLocation::new(curt_coord, new_node.items.len());
        new_node.items.push(Entry { bounds, ..entry });
        // `grow_lines` marked the new level; mark the old one, which lost an item.
        self.lines[level].mark_taken(y);
        self.occupy(nl);
        self.vacate(level);
    }

    /// Overwrite the bounds of an entry that stays in its node, widening the
    /// node's box and the spans of its column and row to cover them.
    fn replace_bounds(&mut self, coord: Coord, index: usize, bounds: Rectangle) {
        self.grow_lines(&bounds, coord);
        let (level, x, y) = coord;
        let node = &mut self.layers[level][y][x];
        node.grow(&bounds);
        node.items[index].bounds = bounds;
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

    /// Take the entry at `index` out of a node. The entry that fills the gap
    /// has its slot updated, so handles stay valid without a lookup.
    fn take_entry(&mut self, level: usize, x: usize, y: usize, index: usize) -> Entry<T> {
        let node = &mut self.layers[level][y][x];
        let entry = node.items.swap_remove(index);
        if let Some(moved) = node.items.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        if node.items.is_empty() {
            // With nothing left to cover, search skips the node at once.
            node.bounds = EMPTY;
        }
        entry
    }

    /// Cheap check that `position` would still return `coord` for `bounds`.
    ///
    /// [`LayerGeometry::matches_layer`] and [`LayerGeometry::matches_cell`]
    /// explain why the cached limits agree with `position` exactly.
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
    /// the `self_query` bench group measured it 8% faster than searching
    /// item by item in random order with 10,000 items, and a one-off probe
    /// outside the repository 27% with 640,000 items at the same
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

    /// Call `callback` once for every pair of distinct stored items whose
    /// bounds intersect, with the two items in either order. On its own level
    /// an item reads only the items after it in its own node and the nodes
    /// after its own in row-major order, that is the neighbor to the right
    /// and those in the row below, where a search with its bounds would read
    /// them; items in the root pair up with the root items after them. Below
    /// its level an item scans the finer levels as a search does, so a pair
    /// across levels is found once, by the item on the coarser level, and no
    /// item looks above its own level. The nodes are visited as in
    /// [`for_each_self_query`](Self::for_each_self_query).
    ///
    /// Leaving each cross-level pair to its finer item instead, which walked
    /// the ancestor path as a search does, took 0.81 ms against 0.34 ms on the
    /// mixed distribution: nearly all items lie on the finest level, and each
    /// of them paid for a walk over the occupied levels above it.
    pub fn for_each_pair(&self, mut callback: impl FnMut(T, T)) {
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
                for (index, entry) in layers[level][y][x].items.iter().enumerate() {
                    let mut report = |found| callback(entry.item, found);
                    if level == 1 {
                        self.get_root().pair_with_after(index, &entry.bounds, &mut report);
                    } else {
                        self.pair_half_stencil(level, (x, y), index, &entry.bounds, &mut report);
                    }
                    if level < finest {
                        self.pair_scan_down(level, &entry.bounds, &mut report);
                    }
                }
            },
        );
    }

    /// Search from the bottom up. Execute the callback for each item whose
    /// bounds intersect `bounds`.
    ///
    /// Follows [`HierarchicalLooseGrid::search`], except that columns and rows
    /// whose spans miss the query are left out, and a node whose box misses
    /// the query is skipped without reading its items.
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

                    // The fixed reach overshoots by about one column and row on
                    // each side; drop each such edge line whose span misses the
                    // query. Loops that drop more lines measured slower on
                    // large queries.
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

    /// Search the half stencil of [`for_each_pair`] on level `level`: node
    /// `(x, y)` after its item `start`, and of the neighbors after it in
    /// row-major order those that the ancestor walk of a search with
    /// `bounds` would read, by the same tests.
    ///
    /// [`for_each_pair`]: Self::for_each_pair
    fn pair_half_stencil(
        &self,
        level: usize,
        (x, y): (usize, usize),
        start: usize,
        bounds: &Rectangle,
        callback: &mut impl FnMut(T),
    ) {
        let grid = &self.layers[level];
        let lines = &self.lines[level];
        let grid_width = grid.cols();
        let grid_height = grid.rows();
        // SAFETY: `(x, y)` is a node of `grid`, and each neighbor is read only
        // after testing that it exists; clamped indices read an in-bounds line
        // whose result the edge tests discard.
        unsafe {
            let below = lines.rows.get_unchecked((y + 1).min(grid_height - 1))[0];
            let left_of = lines.columns.get_unchecked(x.saturating_sub(1))[1];
            let right_of = lines.columns.get_unchecked((x + 1).min(grid_width - 1))[0];
            let down = (y + 1 != grid_height) & (bounds.max_y > below);
            let before = (x != 0) & (bounds.min_x < left_of);
            let after = (x + 1 != grid_width) & (bounds.max_x > right_of);
            grid.get_unchecked(y, x).pair_with_after(start, bounds, callback);
            if after {
                grid.get_unchecked(y, x + 1).pair_with(bounds, callback);
            }
            if down {
                if before {
                    grid.get_unchecked(y + 1, x - 1).pair_with(bounds, callback);
                }
                grid.get_unchecked(y + 1, x).pair_with(bounds, callback);
                if after {
                    grid.get_unchecked(y + 1, x + 1).pair_with(bounds, callback);
                }
            }
        }
    }

    /// Scan the levels finer than `level` over `bounds` for
    /// [`for_each_pair`](Self::for_each_pair) as [`search`](Self::search)
    /// scans the levels finer than a query's, a copy so that pair queries and
    /// searches share no code.
    fn pair_scan_down(&self, level: usize, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        // As in `search`: a NaN corner meets nothing, while for a NaN minimum
        // the scan would report the items of the nodes it takes as inside
        // `bounds` untested.
        if bounds.has_nan() {
            return;
        }
        let max_level = MAX_LEVEL as usize;
        let (offset_x, offset_y) = self.root_bounds.get_min();
        let layers = &self.layers;
        // The corners of `bounds` relative to the root origin, in half nodes
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

                // The fixed reach overshoots by about one column and row on
                // each side; drop each such edge line whose span misses
                // `bounds`. Loops that drop more lines measured slower on
                // large queries.
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
                            // The node lies within `bounds`, loose bounds
                            // included, so its items need no intersection test.
                            node.pair_all(callback);
                        } else {
                            node.pair_with(bounds, callback);
                        }
                    }
                }
            });
    }

    fn search_up_3x3(&self, level: usize, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        let (mut x, mut y) = self.cell_containing(level, bounds.get_center());

        // The root is searched last; levels coarser than `coarsest` hold nothing.
        let coarsest = self.coarsest_level().min(level + 1);
        self.layers[coarsest..=level]
            .iter()
            .zip(&self.geometry[coarsest..=level])
            .zip(&self.lines[coarsest..=level])
            .rev()
            .for_each(|((grid, geometry), lines)| unsafe {
                if !geometry.is_empty() {
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();
                    // The unchecked reads below rely on the walk staying in the grid.
                    debug_assert!(
                        x < grid_width && y < grid_height,
                        "walk cell ({x}, {y}) outside {grid_width}x{grid_height}"
                    );

                    // Whether the span of each neighboring column and row meets
                    // the query. Clamped indices read an in-bounds line whose
                    // result the edge tests discard. Short-circuit branches here
                    // measured search 50% slower on a scene where these
                    // outcomes vary.
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
                // Move to the parent cell even when this level is empty.
                x >>= 1;
                y >>= 1;
            });

        self.get_root().search_items(bounds, callback);
    }
}

/// Span of the items of every column along x and of every row along y of one
/// level since the last refresh, as `[min, max]`;
/// `[+infinity, -infinity]` where no item has been, which meets no query.
struct LineSpan {
    columns: Box<[[f64; 2]]>,
    rows: Box<[[f64; 2]]>,
    /// Whether an item was stored on, replaced on or taken from the level
    /// since its last refresh, so that a box or span there may be looser
    /// than its items; the refresh skips untouched levels, which are exact.
    /// Kept here rather than in a separate array because every store already
    /// reaches this struct through `grow`.
    touched: bool,
    /// The same per row: the refresh shrinks the boxes and rebuilds the span
    /// of each touched row, and reads only the boxes of the other rows, for
    /// the column spans.
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

    /// Widen column `x` and row `y` to cover `bounds` stored there, and mark
    /// the level and the row touched.
    #[inline]
    fn grow(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.touched = true;
        self.row_touched[y] = true;
        self.widen((x, y), bounds);
    }

    /// Mark the level and row `y` touched after an item is taken from it.
    #[inline]
    fn mark_taken(&mut self, y: usize) {
        self.touched = true;
        self.row_touched[y] = true;
    }

    /// Widen column `x` and row `y` to cover `bounds` without marking
    /// anything. The refresh calls this for every box; marking there, one
    /// store per node, measured the refresh 5% slower.
    #[inline]
    fn widen(&mut self, (x, y): (usize, usize), bounds: &Rectangle) {
        self.widen_column(x, bounds);
        let row = &mut self.rows[y];
        row[0] = lower(row[0], bounds.min_y);
        row[1] = higher(row[1], bounds.max_y);
    }

    /// Widen column `x` alone to cover `bounds`.
    #[inline]
    fn widen_column(&mut self, x: usize, bounds: &Rectangle) {
        let column = &mut self.columns[x];
        column[0] = lower(column[0], bounds.min_x);
        column[1] = higher(column[1], bounds.max_x);
    }

    fn clear(&mut self) {
        self.columns.fill(NOWHERE);
        self.rows.fill(NOWHERE);
    }
}

struct SpanOverflowBoundGridNode<T: Copy> {
    /// Covers the bounds of every item in `items`; [`EMPTY`] while there are
    /// none.
    bounds: Rectangle,
    items: Vec<Entry<T>>,
}

impl<T: Copy> SpanOverflowBoundGridNode<T> {
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

    /// Shrink the box to the union of the items stored now.
    fn refresh(&mut self) {
        let Self { bounds, items } = self;
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

    /// [`search_items`](Self::search_items) for the pair queries, a copy so
    /// that they share no code with the searches.
    fn pair_with(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        if !self.bounds.intersects(bounds) {
            return;
        }
        self.items
            .iter()
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }

    /// Test the items after `index` against `bounds`, without the box, which
    /// covers the item at `index` and thus meets any query it makes.
    fn pair_with_after(&self, index: usize, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        self.items[index + 1..]
            .iter()
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }

    /// [`iter_items`](Self::iter_items) for the pair queries, a copy so that
    /// they share no code with the searches.
    fn pair_all(&self, callback: &mut impl FnMut(T)) {
        self.items.iter().for_each(|entry| callback(entry.item));
    }

    fn iter_items(&self, callback: &mut impl FnMut(T)) {
        self.items.iter().for_each(|entry| callback(entry.item));
    }
}

/// Grow `target` to cover `bounds`. A NaN coordinate leaves its side
/// unchanged, which is harmless because an item with one intersects no query.
#[inline]
fn cover(target: &mut Rectangle, bounds: &Rectangle) {
    target.min_x = lower(target.min_x, bounds.min_x);
    target.min_y = lower(target.min_y, bounds.min_y);
    target.max_x = higher(target.max_x, bounds.max_x);
    target.max_y = higher(target.max_y, bounds.max_y);
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    const MAX_LEVEL: u8 = 7;
    type Tree = SpanOverflowBoundGrid<usize, MAX_LEVEL>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    crate::grid_tests::span_tests!(SpanOverflowBoundGridNode);

    /// Storing an item on a level, replacing one there or taking one from it
    /// marks the level; a refresh clears every mark, and an unchanged level
    /// stays unmarked. `exercise` relies on this: its checks after a refresh
    /// expect every level exact, skipped ones included.
    #[test]
    fn touched_levels_are_the_ones_refreshed() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let finest = MAX_LEVEL as usize;
        let small = Rectangle::center_rect(100.0, 100.0, 4.0, 4.0);
        let big = Rectangle::center_rect(100.0, 100.0, 200.0, 200.0);
        let coarser = tree.position(&big).0;
        assert!(coarser < finest);
        let touched = |tree: &Tree| -> Vec<usize> {
            (0..tree.lines.len())
                .filter(|&level| tree.lines[level].touched)
                .collect()
        };

        assert_eq!(touched(&tree), []);
        let handle = tree.insert(small.clone(), 0);
        assert_eq!(touched(&tree), [finest]);
        tree.refresh_bounds();
        assert_eq!(touched(&tree), []);

        tree.update(Rectangle::center_rect(101.0, 100.0, 4.0, 4.0), handle);
        assert_eq!(touched(&tree), [finest]);
        tree.refresh_bounds();

        tree.update(big.clone(), handle);
        assert_eq!(touched(&tree), [coarser, finest]);
        tree.refresh_bounds();

        tree.remove(handle);
        assert_eq!(touched(&tree), [coarser]);
        tree.refresh_bounds();
        assert_eq!(touched(&tree), []);
    }

    /// Storing an item in a row or taking one from it marks the row, and a
    /// refresh clears the marks. `exercise` relies on the rest: its checks
    /// after a refresh expect every row exact, skipped ones included.
    #[test]
    fn touched_rows_are_the_ones_shrunk() {
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        let finest = MAX_LEVEL as usize;
        // Finest nodes are 16 units wide; this item belongs in (6, 6).
        let item = Rectangle::center_rect(100.0, 100.0, 4.0, 4.0);
        assert_eq!(tree.position(&item), (finest, 6, 6));
        let rows = |tree: &Tree| -> Vec<usize> {
            (0..tree.lines[finest].row_touched.len())
                .filter(|&y| tree.lines[finest].row_touched[y])
                .collect()
        };

        let handle = tree.insert(item.clone(), 0);
        assert_eq!(rows(&tree), [6]);
        tree.refresh_bounds();
        assert_eq!(rows(&tree), []);

        // Move to the row below: stored in row 7, taken from row 6.
        tree.update(Rectangle::center_rect(100.0, 116.0, 4.0, 4.0), handle);
        assert_eq!(rows(&tree), [6, 7]);
        tree.refresh_bounds();
        assert_eq!(rows(&tree), []);

        tree.remove(handle);
        assert_eq!(rows(&tree), [7]);
        tree.refresh_bounds();
        assert_eq!(rows(&tree), []);
    }
}
