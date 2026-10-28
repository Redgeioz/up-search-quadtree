use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::cell_order::for_each_cell_hierarchically;
use crate::layer_geometry::{
    LayerGeometry, Levels, build_levels, first_reaching_node, last_reaching_node, size_level,
};
use crate::overflow_bound::{higher, lower};
use crate::rect::{EMPTY, Rectangle};
use grid::*;

/// (level, x, y)
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<BoxedLooseGridNode<T>>>;

/// Hierarchical loose grid with lazy per-node boxes:
/// [`LazyOverflowBoundGrid`] without its per-level bounds. The storage,
/// positioning and handles are those of [`HierarchicalLooseGrid`], and every
/// node also keeps a box covering the bounds of its items. Search selects
/// candidate nodes with the fixed reach (half a node) of the hierarchical
/// loose grid and skips a node whose box misses the query without reading its
/// items.
///
/// Insertion and update only grow the box of the target node, with one
/// comparison per side. Removal and moving out leave it unchanged, except
/// that a node left without items resets its box. Boxes therefore remain
/// valid as items move, shrink or leave, and tighten only in
/// [`refresh_bounds`], at a time the caller chooses, e.g. once per frame
/// after the updates.
///
/// The ancestor walk also stops at the coarsest level below the root that
/// holds items, which a bitmask of the occupied levels keeps exact at all
/// times.
///
/// Boxes consist of item coordinates, so their tests compare the same values
/// as the intersection tests and need no rounding margin.
///
/// [`LazyOverflowBoundGrid`]: crate::LazyOverflowBoundGrid
/// [`HierarchicalLooseGrid`]: crate::HierarchicalLooseGrid
/// [`refresh_bounds`]: Self::refresh_bounds
pub struct BoxedLooseGrid<T: Copy, const MAX_LEVEL: u8> {
    layers: Layers<T>,
    /// Indexed by level, parallel to `layers`.
    geometry: Vec<LayerGeometry>,
    /// Reciprocals of the node width and height, indexed by level. Position
    /// and search multiply by them instead of dividing by the node size, and
    /// `geometry` builds its cell edges with the same arithmetic.
    inverse_node_size: Vec<(f64, f64)>,
    /// Bit `L` is set while level `L` holds items: storing an item sets the
    /// bit of its level, and taking the last item off a level clears it. The
    /// coarsest level below the root that holds items, where the ancestor walk
    /// of search stops, is therefore exact at all times, not only after a
    /// refresh.
    occupied_levels: u64,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    /// Where the entry of each live handle is stored.
    slots: Slots<GridLocation>,
}

impl<T: Copy, const MAX_LEVEL: u8> BoxedLooseGrid<T, MAX_LEVEL> {
    /// Create a new grid with the given bounds.
    ///
    /// Every node is square, which usually makes search more efficient.
    pub fn new(world_bounds: Rectangle) -> BoxedLooseGrid<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new grid with the given bounds.
    ///
    /// The nodes take the aspect ratio of the given bounds instead of being
    /// square.
    pub fn new_fit(world_bounds: Rectangle) -> BoxedLooseGrid<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(world_bounds: Rectangle) -> BoxedLooseGrid<T, MAX_LEVEL> {
        let Levels {
            layers,
            root_bounds,
            geometry,
            inverse_node_size,
        } = build_levels::<_, FIT>(&world_bounds, MAX_LEVEL, BoxedLooseGridNode::new);

        BoxedLooseGrid {
            layers,
            geometry,
            inverse_node_size,
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

    /// Shrink the box of every node to the bounds of its current items,
    /// undoing the growth from items that have since moved, shrunk or left.
    /// Visits every node and item at most once.
    pub fn refresh_bounds(&mut self) {
        for level in 1..=MAX_LEVEL as usize {
            // Every node of a level without items has already reset its box.
            if !self.geometry[level].is_empty() {
                self.layers[level].iter_mut().for_each(BoxedLooseGridNode::refresh);
            }
        }
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

    fn get_root(&self) -> &BoxedLooseGridNode<T> {
        unsafe { self.layers.get_unchecked(1).get_unchecked(0, 0) }
    }

    /// Insert an item and return the handle that identifies it to
    /// [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord @ (level, x, y) = self.position(&bounds);
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
        let new_node = &mut self.layers[nl][ny][nx];
        new_node.grow(&bounds);
        *self.slots.get_mut(handle) = GridLocation::new(curt_coord, new_node.items.len());
        new_node.items.push(Entry { bounds, ..entry });
        self.occupy(nl);
        self.vacate(level);
    }

    /// Overwrite the bounds of an entry that stays in its node, and grow the
    /// box of the node to cover them.
    fn replace_bounds(&mut self, coord: Coord, index: usize, bounds: Rectangle) {
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
    /// searching item by item is faster.
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
    /// the ancestor path as a search does, took 1.05 ms against 0.39 ms on the
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
    /// Follows [`HierarchicalLooseGrid::search`], except that a node whose box
    /// misses the query is skipped without reading its items.
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
                .zip(&self.inverse_node_size[level + 1..=max_level])
                .rev()
                .for_each(|((grid, geometry), &(inverse_width, inverse_height))| {
                    if geometry.is_empty() {
                        return;
                    }
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();

                    // Items reach half a node beyond their cells. Both ends
                    // include a node whose items reach exactly to the query, as
                    // in `scan_range`.
                    let min_coord_x = first_reaching_node(min_x * inverse_width);
                    let min_coord_y = first_reaching_node(min_y * inverse_height);

                    let max_coord_x = last_reaching_node(max_x * inverse_width).min(grid_width - 1);
                    let max_coord_y = last_reaching_node(max_y * inverse_height).min(grid_height - 1);

                    // Scan the resulting range.
                    for y in min_coord_y..=max_coord_y {
                        for x in min_coord_x..=max_coord_x {
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
    /// `bounds` would read, by the same center-line tests.
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
        let center_x = offset_x + geometry.node_width * (x as f64 + 0.5);
        let center_y = offset_y + geometry.node_height * (y as f64 + 0.5);
        let to_right = bounds.max_x >= center_x && x + 1 != grid.cols();
        let to_left = bounds.min_x <= center_x && x != 0;
        // SAFETY: `(x, y)` is a node of `grid`, and each neighbor is read only
        // after testing that it exists.
        unsafe {
            grid.get_unchecked(y, x).pair_with_after(start, bounds, callback);
            if to_right {
                grid.get_unchecked(y, x + 1).pair_with(bounds, callback);
            }
            if bounds.max_y >= center_y && y + 1 != grid.rows() {
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
            .zip(&self.inverse_node_size[level + 1..=max_level])
            .rev()
            .for_each(|((grid, geometry), &(inverse_width, inverse_height))| {
                if geometry.is_empty() {
                    return;
                }
                let grid_width = grid.cols();
                let grid_height = grid.rows();

                // Items reach half a node beyond their cells. Both ends
                // include a node whose items reach exactly to the query, as
                // in `scan_range`.
                let min_coord_x = first_reaching_node(min_x * inverse_width);
                let min_coord_y = first_reaching_node(min_y * inverse_height);

                let max_coord_x = last_reaching_node(max_x * inverse_width).min(grid_width - 1);
                let max_coord_y = last_reaching_node(max_y * inverse_height).min(grid_height - 1);

                // Scan the resulting range.
                for y in min_coord_y..=max_coord_y {
                    for x in min_coord_x..=max_coord_x {
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
            .rev()
            .for_each(|(grid, geometry)| unsafe {
                if !geometry.is_empty() {
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();
                    // The unchecked reads below rely on the walk staying in the grid.
                    debug_assert!(
                        x < grid_width && y < grid_height,
                        "walk cell ({x}, {y}) outside {grid_width}x{grid_height}"
                    );

                    // The items of each neighboring column and row reach up to
                    // the center lines of this cell. A query that starts or
                    // ends exactly on a center line still meets an item whose
                    // bounds round a float past it, so the tests include it.
                    if bounds.min_y <= center_y && y != 0 {
                        if bounds.min_x <= center_x && x != 0 {
                            let node = grid.get_unchecked(y - 1, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y - 1, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x >= center_x && x + 1 != grid_width {
                            let node = grid.get_unchecked(y - 1, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }

                    {
                        if bounds.min_x <= center_x && x != 0 {
                            let node = grid.get_unchecked(y, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x >= center_x && x + 1 != grid_width {
                            let node = grid.get_unchecked(y, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }

                    if bounds.max_y >= center_y && y + 1 != grid_height {
                        if bounds.min_x <= center_x && x != 0 {
                            let node = grid.get_unchecked(y + 1, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y + 1, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x >= center_x && x + 1 != grid_width {
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

struct BoxedLooseGridNode<T: Copy> {
    /// Box covering the bounds of every item in `items`; [`EMPTY`] while there
    /// are none.
    bounds: Rectangle,
    items: Vec<Entry<T>>,
}

impl<T: Copy> BoxedLooseGridNode<T> {
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
    type Tree = BoxedLooseGrid<usize, MAX_LEVEL>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    crate::grid_tests::refreshed_tests!();
    crate::grid_tests::lazy_box_tests!(BoxedLooseGridNode);
}
