use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::cell_order::for_each_cell_hierarchically;
use crate::layer_geometry::{
    LayerGeometry, Levels, build_levels, first_reaching_node, last_reaching_node, size_level,
};
use crate::exact_overflow_bound_grid::DirectionalOverflow;
use crate::overflow_bound::{LevelReach, cell_overflow, higher, lower, rounding_slack};
use crate::rect::{EMPTY, Rectangle};
use grid::*;

/// (level, x, y)
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<LazyOverflowBoundGridNode<T>>>;

/// Overflow-bound grid with lazy per-node boxes: the storage, positioning and
/// handles of [`HierarchicalLooseGrid`], where every node also keeps a box
/// covering the bounds of its items, and every level a bound per direction on
/// how far its items extend beyond their cells, which only grows until the
/// next refresh. Search shrinks the fixed reach (half a node) to that bound
/// to select candidate nodes, which excludes whole neighboring rows and
/// columns without reading them, and skips a node whose box misses the query
/// without reading its items.
///
/// A bound may be negative: when every item of a level stays inside its cell
/// edge in some direction, search enters a cell across that edge only where
/// the query crosses the edge by more than the smallest such gap, less the
/// rounding margin.
///
/// Insertion and update only grow the box of the target node and the bound
/// of its level, with one comparison per side. Removal and moving out leave
/// both unchanged, except that a node left without items resets its box.
/// Boxes and bounds therefore remain valid as items move, shrink or leave,
/// and tighten only in [`refresh_bounds`], at a time the caller chooses, e.g.
/// once per frame after the updates. A stale bound never costs more than the
/// fixed reach, and the boxes still apply.
///
/// The ancestor walk also stops at the coarsest level below the root that
/// holds items, which a bitmask of the occupied levels keeps exact at all
/// times.
///
/// Boxes consist of item coordinates, so their tests compare the same values
/// as the intersection tests and need no rounding margin; the level bounds
/// add a rounding margin.
///
/// [`HierarchicalLooseGrid`]: crate::HierarchicalLooseGrid
/// [`refresh_bounds`]: Self::refresh_bounds
pub struct LazyOverflowBoundGrid<T: Copy, const MAX_LEVEL: u8> {
    layers: Layers<T>,
    /// Indexed by level, parallel to `layers`.
    geometry: Vec<LayerGeometry>,
    /// Reciprocals of the node width and height, indexed by level. Position
    /// and search multiply by them instead of dividing by the node size, and
    /// `geometry` builds its cell edges with the same arithmetic.
    inverse_node_size: Vec<(f64, f64)>,
    /// Indexed by level, parallel to `layers`.
    reach: Vec<LevelReach>,
    /// Bit `L` is set while level `L` holds items: storing an item sets the
    /// bit of its level, and taking the last item off a level clears it. The
    /// coarsest level below the root that holds items, where the ancestor walk
    /// of search stops, is therefore exact at all times, not only after a
    /// refresh. With all items on the finest level, stopping the walk there
    /// measured 10% faster than walking every level.
    occupied_levels: u64,
    /// Rounding margin that search adds to every level bound, since its cell
    /// edges accumulate rounding differently from [`cell_overflow`].
    slack: f64,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    /// Where the entry of each live handle is stored.
    slots: Slots<GridLocation>,
}

impl<T: Copy, const MAX_LEVEL: u8> LazyOverflowBoundGrid<T, MAX_LEVEL> {
    /// Create a new grid with the given bounds.
    ///
    /// Every node is square, which usually makes search more efficient.
    pub fn new(world_bounds: Rectangle) -> LazyOverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new grid with the given bounds.
    ///
    /// The nodes take the aspect ratio of the given bounds instead of being
    /// square.
    pub fn new_fit(world_bounds: Rectangle) -> LazyOverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(world_bounds: Rectangle) -> LazyOverflowBoundGrid<T, MAX_LEVEL> {
        let Levels {
            layers,
            root_bounds,
            geometry,
            inverse_node_size,
        } = build_levels::<_, FIT>(&world_bounds, MAX_LEVEL, LazyOverflowBoundGridNode::new);

        let slack = rounding_slack(&root_bounds);
        // The root bound grows like any other but is never read, since search
        // scans the root entirely.
        let reach = geometry
            .iter()
            .map(|geometry| LevelReach::new(geometry.node_width, geometry.node_height, slack))
            .collect();

        LazyOverflowBoundGrid {
            layers,
            geometry,
            inverse_node_size,
            reach,
            occupied_levels: 0,
            slack,
            root_bounds,
            world_bounds,
            slots: Slots::new(),
        }
    }

    /// Return the actual bounds of the root.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Return the largest overflow beyond the cells seen on `level` since the
    /// last [`refresh_bounds`](Self::refresh_bounds), in world units:
    /// negative when every item stays that far inside its cell edge, and
    /// -infinity if the level has held no items since then. Search adds the
    /// rounding margin and caps the result at half a node.
    ///
    /// Panics unless `level` is in `2..=MAX_LEVEL`; the root has no neighbors
    /// to bound.
    pub fn overflow_bound(&self, level: usize) -> DirectionalOverflow {
        assert!((2..=MAX_LEVEL as usize).contains(&level), "invalid level");
        let [left, right, top, bottom] = self.reach[level].reach;
        DirectionalOverflow {
            left,
            right,
            top,
            bottom,
        }
    }

    /// Shrink the box of every node to the bounds of its current items, and
    /// the bound of every level to the largest overflow of those items,
    /// undoing the growth from items that have since moved, shrunk or left.
    /// Visits every node and item at most once.
    ///
    /// A refreshed box is the union of its items, so its overflow beyond the
    /// cell equals the largest overflow of those items bit for bit; the bounds
    /// are therefore computed from the boxes, one per node instead of one per
    /// item.
    pub fn refresh_bounds(&mut self) {
        let origin = self.root_bounds.get_min();
        for level in 1..=MAX_LEVEL as usize {
            let geometry = &self.geometry[level];
            let node_size = (geometry.node_width, geometry.node_height);
            let mut largest = [f64::NEG_INFINITY; 4];
            // Every node of a level without items has already reset its box.
            if !geometry.is_empty() {
                let grid = &mut self.layers[level];
                let cols = grid.cols();
                // Visit the nodes in storage order, i.e. row by row.
                let (mut x, mut y) = (0, 0);
                for node in grid.iter_mut() {
                    node.refresh();
                    if !node.items.is_empty() {
                        let overflow = cell_overflow(&node.bounds, origin, node_size, (x, y));
                        for (largest, overflow) in largest.iter_mut().zip(overflow) {
                            *largest = higher(*largest, overflow);
                        }
                    }
                    x += 1;
                    if x == cols {
                        x = 0;
                        y += 1;
                    }
                }
            }
            self.reach[level].set(largest, self.slack);
        }
    }

    /// Raise the bound of the level of `coord` to cover `bounds` stored at
    /// `coord`.
    #[inline]
    fn grow_reach(&mut self, bounds: &Rectangle, (level, x, y): Coord) {
        let geometry = &self.geometry[level];
        let node_size = (geometry.node_width, geometry.node_height);
        let overflow = cell_overflow(bounds, self.root_bounds.get_min(), node_size, (x, y));
        self.reach[level].grow(overflow, self.slack);
    }

    /// Find the level and coordinates of the node that stores an item with
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

    /// Return the level whose nodes fit an item of the given size.
    ///
    /// Level `L` holds items that fit across the root at least `2^(L - 1)`
    /// times but fewer than `2^L` times, clamped to `1..=MAX_LEVEL`.
    /// [`LayerGeometry`] caches the matching size limits of each level.
    fn level_for_size(&self, width: f64, height: f64) -> usize {
        // The bit length of `(root / size) as usize` on each axis, from the
        // exponents rather than by dividing.
        let level_x = size_level(self.root_bounds.get_width(), width);
        let level_y = size_level(self.root_bounds.get_height(), height);
        level_x.min(level_y).clamp(1, MAX_LEVEL as i64) as usize
    }

    /// Return the cell of `level` that contains `point`, clamped to the grid.
    fn cell_containing(&self, level: usize, point: (f64, f64)) -> (usize, usize) {
        let grid = &self.layers[level];
        let (inverse_width, inverse_height) = self.inverse_node_size[level];
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let x = (((point.0 - origin_x) * inverse_width) as usize).min(grid.cols() - 1);
        let y = (((point.1 - origin_y) * inverse_height) as usize).min(grid.rows() - 1);
        (x, y)
    }

    fn get_root(&self) -> &LazyOverflowBoundGridNode<T> {
        unsafe { self.layers.get_unchecked(1).get_unchecked(0, 0) }
    }

    /// Insert an item and return the handle that identifies it to
    /// [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord @ (level, x, y) = self.position(&bounds);
        self.grow_reach(&bounds, coord);
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

    /// Remove an item.
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let (level, x, y) = location.coord();
        let entry = self.take_entry(level, x, y, location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        self.slots.free(handle);
        self.vacate(level);
    }

    /// Update the bounds of an item and, if necessary, move it to another node.
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
        self.grow_reach(&bounds, curt_coord);
        let new_node = &mut self.layers[nl][ny][nx];
        new_node.grow(&bounds);
        *self.slots.get_mut(handle) = GridLocation::new(curt_coord, new_node.items.len());
        new_node.items.push(Entry { bounds, ..entry });
        self.occupy(nl);
        self.vacate(level);
    }

    /// Overwrite the bounds of an entry that stays in its node, and grow the
    /// box of the node and the bound of its level to cover them.
    fn replace_bounds(&mut self, coord: Coord, index: usize, bounds: Rectangle) {
        self.grow_reach(&bounds, coord);
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

    /// Take the entry at `index` out of a node by swap removal, and update the
    /// slot of the entry moved into its place, so handles remain valid without
    /// a lookup.
    fn take_entry(&mut self, level: usize, x: usize, y: usize, index: usize) -> Entry<T> {
        let node = &mut self.layers[level][y][x];
        let entry = node.items.swap_remove(index);
        if let Some(moved) = node.items.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        if node.items.is_empty() {
            // No items are left to cover; the empty box lets search reject the
            // node at once.
            node.bounds = EMPTY;
        }
        entry
    }

    /// Check cheaply that `position` would still return `coord` for `bounds`.
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
    /// the `self_query` bench group measured it 14% faster than searching
    /// item by item in random order with 10,000 items, and a one-off probe
    /// outside the repository 28% with 640,000 items at the same
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
    /// the ancestor path as a search does, took 1.06 ms against 0.39 ms on the
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

    /// Search from the bottom up, calling `callback` for each item found.
    ///
    /// Follows [`HierarchicalLooseGrid::search`], except that each level
    /// reaches beyond its cells only as far as its bound allows, and that a
    /// node whose box misses the query is skipped without reading its items.
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
            // Top-left corner, relative to the root origin.
            let min_x = bounds.min_x - offset_x;
            let min_y = bounds.min_y - offset_y;

            // Bottom-right corner, relative to the root origin.
            let max_x = bounds.max_x - offset_x;
            let max_y = bounds.max_y - offset_y;

            layers[level + 1..=max_level]
                .iter()
                .zip(&self.geometry[level + 1..=max_level])
                .zip(&self.reach[level + 1..=max_level])
                .zip(&self.inverse_node_size[level + 1..=max_level])
                .rev()
                .for_each(|(((grid, geometry), reach), &(inverse_width, inverse_height))| {
                    if geometry.is_empty() {
                        return;
                    }
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();

                    // Cells at lower coordinates than the query range count as
                    // far as their items reach in the positive direction, and
                    // vice versa.
                    // Coordinates are in nodes and the shrink in half nodes, so
                    // the shrink counts half.
                    let [left, right, top, bottom] = reach.shrink;
                    // Both ends include a node whose items reach exactly to the
                    // query, as in `scan_range`.
                    let min_coord_x = first_reaching_node(min_x * inverse_width + right * 0.5);
                    let min_coord_y = first_reaching_node(min_y * inverse_height + bottom * 0.5);

                    let max_coord_x =
                        last_reaching_node(max_x * inverse_width - left * 0.5).min(grid_width - 1);
                    let max_coord_y =
                        last_reaching_node(max_y * inverse_height - top * 0.5).min(grid_height - 1);

                    // Scan the resulting range.
                    for y in min_coord_y..=max_coord_y {
                        for x in min_coord_x..=max_coord_x {
                            let node = unsafe { grid.get_unchecked(y, x) };
                            if node.items.is_empty() {
                                continue;
                            }

                            // The shrunk range lies within the fixed-reach range,
                            // so its inner part lies within that of the latter.
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
        let geometry = &self.geometry[level];
        let (offset_x, offset_y) = self.root_bounds.get_min();
        let half_width = geometry.node_width * 0.5;
        let half_height = geometry.node_height * 0.5;
        let center_x = offset_x + geometry.node_width * (x as f64 + 0.5);
        let center_y = offset_y + geometry.node_height * (y as f64 + 0.5);
        let [left, right, top, _] = self.reach[level].shrink;
        let left_column_reach = center_x - right * half_width;
        let right_column_reach = center_x + left * half_width;
        let bottom_row_reach = center_y + top * half_height;
        let to_right = bounds.max_x >= right_column_reach && x + 1 != grid.cols();
        let to_left = bounds.min_x <= left_column_reach && x != 0;
        // SAFETY: `(x, y)` is a node of `grid`, and each neighbor is read only
        // after testing that it exists.
        unsafe {
            grid.get_unchecked(y, x).pair_with_after(start, bounds, callback);
            if to_right {
                grid.get_unchecked(y, x + 1).pair_with(bounds, callback);
            }
            if bounds.max_y >= bottom_row_reach && y + 1 != grid.rows() {
                if to_left {
                    grid.get_unchecked(y + 1, x - 1).pair_with(bounds, callback);
                }
                grid.get_unchecked(y + 1, x).pair_with(bounds, callback);
                if to_right {
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
        // Top-left corner, relative to the root origin.
        let min_x = bounds.min_x - offset_x;
        let min_y = bounds.min_y - offset_y;

        // Bottom-right corner, relative to the root origin.
        let max_x = bounds.max_x - offset_x;
        let max_y = bounds.max_y - offset_y;

        layers[level + 1..=max_level]
            .iter()
            .zip(&self.geometry[level + 1..=max_level])
            .zip(&self.reach[level + 1..=max_level])
            .zip(&self.inverse_node_size[level + 1..=max_level])
            .rev()
            .for_each(|(((grid, geometry), reach), &(inverse_width, inverse_height))| {
                if geometry.is_empty() {
                    return;
                }
                let grid_width = grid.cols();
                let grid_height = grid.rows();

                // Cells at lower coordinates than the query range count as
                // far as their items reach in the positive direction, and
                // vice versa.
                // Coordinates are in nodes and the shrink in half nodes, so
                // the shrink counts half.
                let [left, right, top, bottom] = reach.shrink;
                // Both ends include a node whose items reach exactly to the
                // query, as in `scan_range`.
                let min_coord_x = first_reaching_node(min_x * inverse_width + right * 0.5);
                let min_coord_y = first_reaching_node(min_y * inverse_height + bottom * 0.5);

                let max_coord_x =
                    last_reaching_node(max_x * inverse_width - left * 0.5).min(grid_width - 1);
                let max_coord_y =
                    last_reaching_node(max_y * inverse_height - top * 0.5).min(grid_height - 1);

                // Scan the resulting range.
                for y in min_coord_y..=max_coord_y {
                    for x in min_coord_x..=max_coord_x {
                        let node = unsafe { grid.get_unchecked(y, x) };
                        if node.items.is_empty() {
                            continue;
                        }

                        // The shrunk range lies within the fixed-reach range,
                        // so its inner part lies within that of the latter.
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

        let (offset_x, offset_y) = self.root_bounds.get_min();
        let node_width = self.geometry[level].node_width;
        let node_height = self.geometry[level].node_height;

        let mut center_x = offset_x + node_width * (x as f64 + 0.5);
        let mut center_y = offset_y + node_height * (y as f64 + 0.5);
        let mut half_width = node_width * 0.5;
        let mut half_height = node_height * 0.5;

        // The root is searched separately below; levels coarser than
        // `coarsest` hold no items.
        let coarsest = self.coarsest_level().min(level + 1);
        self.layers[coarsest..=level]
            .iter()
            .zip(&self.geometry[coarsest..=level])
            .zip(&self.reach[coarsest..=level])
            .rev()
            .for_each(|((grid, geometry), reach)| unsafe {
                if !geometry.is_empty() {
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();
                    // The unchecked reads below rely on the walk staying in the grid.
                    debug_assert!(
                        x < grid_width && y < grid_height,
                        "walk cell ({x}, {y}) outside {grid_width}x{grid_height}"
                    );

                    // Coordinates up to which the items of each neighboring
                    // column and row reach into this cell; the fixed reach ends
                    // at its center.
                    let [left, right, top, bottom] = reach.shrink;
                    let left_column_reach = center_x - right * half_width;
                    let right_column_reach = center_x + left * half_width;
                    let top_row_reach = center_y - bottom * half_height;
                    let bottom_row_reach = center_y + top * half_height;

                    // A query that starts or ends exactly on the fixed reach,
                    // the center line, still meets an item whose bounds round
                    // a float past it, so the tests include the reach. A
                    // shrunk reach already holds the rounding margin.
                    if bounds.min_y <= top_row_reach && y != 0 {
                        if bounds.min_x <= left_column_reach && x != 0 {
                            let node = grid.get_unchecked(y - 1, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y - 1, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x >= right_column_reach && x + 1 != grid_width {
                            let node = grid.get_unchecked(y - 1, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }

                    {
                        if bounds.min_x <= left_column_reach && x != 0 {
                            let node = grid.get_unchecked(y, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x >= right_column_reach && x + 1 != grid_width {
                            let node = grid.get_unchecked(y, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }

                    if bounds.max_y >= bottom_row_reach && y + 1 != grid_height {
                        if bounds.min_x <= left_column_reach && x != 0 {
                            let node = grid.get_unchecked(y + 1, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y + 1, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x >= right_column_reach && x + 1 != grid_width {
                            let node = grid.get_unchecked(y + 1, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }
                }
                // Move to the parent cell, also when this level has no items.
                center_x += if x & 1 == 0 { half_width } else { -half_width };
                center_y += if y & 1 == 0 {
                    half_height
                } else {
                    -half_height
                };
                half_width *= 2.0;
                half_height *= 2.0;
                x >>= 1;
                y >>= 1;
            });

        self.get_root().search_items(bounds, callback);
    }
}

struct LazyOverflowBoundGridNode<T: Copy> {
    /// Box covering the bounds of every item in `items`; [`EMPTY`] while there
    /// are none.
    bounds: Rectangle,
    items: Vec<Entry<T>>,
}

impl<T: Copy> LazyOverflowBoundGridNode<T> {
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

    /// Shrink the box to the union of the bounds of the current items.
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
/// unchanged, which is harmless: an item with a NaN coordinate intersects no
/// query.
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
    type Tree = LazyOverflowBoundGrid<usize, MAX_LEVEL>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    crate::grid_tests::boxed_tests!(LazyOverflowBoundGridNode);

    /// Removing the item that defines a bound leaves the bound unchanged until
    /// the next refresh, which lowers it to the overflow of the remaining items.
    #[test]
    fn bounds_only_drop_on_refresh() {
        // The nodes of the finest level are 16 units wide.
        let mut tree = Tree::new(Rectangle::new(0.0, 0.0, 1024.0, 1024.0));
        // Centered just right of the cell edge at 16, extending 3 units into
        // cell 0.
        let wide = tree.insert(Rectangle::center_rect(16.5, 8.0, 7.0, 7.0), 0);
        tree.insert(Rectangle::center_rect(40.0, 40.0, 2.0, 2.0), 1);
        let level = MAX_LEVEL as usize;
        assert_eq!(tree.overflow_bound(level).left, 3.0);

        tree.remove(wide);
        assert_eq!(tree.overflow_bound(level).left, 3.0);

        // The remaining item stays 7 units inside the left edge of its node.
        tree.refresh_bounds();
        assert_eq!(tree.overflow_bound(level).left, -7.0);
    }
}
