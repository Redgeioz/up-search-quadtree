use crate::bitmap_max_multiset::MaxMultiset;
use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::cell_order::for_each_cell_hierarchically;
use crate::layer_geometry::{
    LayerGeometry, Levels, build_levels, first_reaching_node, last_reaching_node, size_level,
};
use crate::overflow_bound::rounding_slack;
use crate::rect::Rectangle;
use grid::*;

/// (level, x, y)
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<OverflowBoundGridNode<T>>>;

/// Signed distances beyond the original cell edges, in world units: negative
/// where the bounds stay inside an edge. As level maxima, -infinity in every
/// direction on a level without items. Top is -y, bottom +y.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct DirectionalOverflow {
    pub left: f64,
    pub right: f64,
    pub top: f64,
    pub bottom: f64,
}

impl DirectionalOverflow {
    fn values(self) -> [f64; 4] {
        [self.left, self.right, self.top, self.bottom]
    }
}

struct LevelOverflow {
    directions: [MaxMultiset; 4],
    /// Fraction of half a node by which search shrinks the fixed reach in each
    /// direction, kept in step with the maxima. It is exactly zero once a
    /// maximum plus the slack covers half a node; search then matches the
    /// hierarchical loose grid.
    ///
    /// Computing it in search places a chain of dependent float operations
    /// before the neighbor checks, which measured the search of
    /// [`LazyOverflowBoundGrid`](crate::LazyOverflowBoundGrid) about 40%
    /// slower.
    shrink: [f64; 4],
    /// Reciprocal of half the node width and height.
    inverse_half_node: (f64, f64),
}

impl LevelOverflow {
    fn new(node_width: f64, node_height: f64, slack: f64) -> Self {
        let mut level = Self {
            directions: Default::default(),
            shrink: [0.0; 4],
            inverse_half_node: (2.0 / node_width, 2.0 / node_height),
        };
        level.refresh_shrink(slack);
        level
    }

    fn insert(&mut self, overflow: DirectionalOverflow, slack: f64) {
        for (direction, value) in self.directions.iter_mut().zip(overflow.values()) {
            direction.insert(value);
        }
        self.refresh_shrink(slack);
    }

    fn remove(&mut self, overflow: DirectionalOverflow, slack: f64) {
        for (direction, value) in self.directions.iter_mut().zip(overflow.values()) {
            direction.remove(value);
        }
        self.refresh_shrink(slack);
    }

    fn replace(&mut self, old: DirectionalOverflow, new: DirectionalOverflow, slack: f64) {
        for ((direction, old), new) in self
            .directions
            .iter_mut()
            .zip(old.values())
            .zip(new.values())
        {
            direction.replace(old, new);
        }
        self.refresh_shrink(slack);
    }

    /// Recompute `shrink` from the cached maxima.
    #[inline]
    fn refresh_shrink(&mut self, slack: f64) {
        let (inverse_x, inverse_y) = self.inverse_half_node;
        let inverse = [inverse_x, inverse_x, inverse_y, inverse_y];
        for ((shrink, direction), inverse) in
            self.shrink.iter_mut().zip(&self.directions).zip(inverse)
        {
            *shrink = (1.0 - (direction.maximum() + slack) * inverse).max(0.0);
        }
    }

    fn maximum(&self) -> DirectionalOverflow {
        let [left, right, top, bottom] = &self.directions;
        DirectionalOverflow {
            left: left.maximum(),
            right: right.maximum(),
            top: top.maximum(),
            bottom: bottom.maximum(),
        }
    }
}

/// Overflow-bound grid: the storage and search of
/// [`crate::HierarchicalLooseGrid`], with exact per-level directional overflow
/// maxima maintained on insert, update and remove.
///
/// Overflow is the signed distance by which an object's bounds extend beyond
/// the original edges of its cell, not beyond its loose bounds; it is negative
/// where the bounds stay inside an edge. Search follows the hierarchical loose
/// grid, but each level reaches beyond its cells only as far as its maxima
/// plus a small rounding margin. With a negative maximum, search looks past
/// the corresponding cell edge only once the query crosses it by more than
/// the magnitude of that maximum, less the margin. Square cells (`new`) and
/// fitted cells (`new_fit`) follow the positioning of the hierarchical loose
/// grid.
pub struct OverflowBoundGrid<T: Copy, const MAX_LEVEL: u8> {
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
    /// of search stops, is therefore exact at all times.
    occupied_levels: u64,
    /// Indexed by level; level zero is unused.
    overflow: Vec<LevelOverflow>,
    /// Rounding margin that search adds to every maximum, since its cell edges
    /// accumulate rounding differently from `item_overflow`.
    slack: f64,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    /// Location of the entry of each live handle.
    slots: Slots<GridLocation>,
}

impl<T: Copy, const MAX_LEVEL: u8> OverflowBoundGrid<T, MAX_LEVEL> {
    /// Create a new overflow-bound grid with the given bounds.
    ///
    /// Every node is square, which usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> OverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new overflow-bound grid with the given bounds.
    ///
    /// Nodes divide the given bounds evenly along each axis instead of being
    /// square.
    pub fn new_fit(world_bounds: Rectangle) -> OverflowBoundGrid<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(world_bounds: Rectangle) -> OverflowBoundGrid<T, MAX_LEVEL> {
        let Levels {
            layers,
            root_bounds,
            geometry,
            inverse_node_size,
        } = build_levels::<_, FIT>(&world_bounds, MAX_LEVEL, OverflowBoundGridNode::new);

        let slack = rounding_slack(&root_bounds);

        OverflowBoundGrid {
            overflow: geometry
                .iter()
                .map(|geometry| {
                    LevelOverflow::new(geometry.node_width, geometry.node_height, slack)
                })
                .collect(),
            slack,
            layers,
            geometry,
            inverse_node_size,
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

    /// Exact maximum overflow beyond the original cell edges on `level`, per
    /// direction, read in O(1). A maximum is negative when every item keeps
    /// that far inside its cell edge, and -infinity in every direction on a
    /// level without items.
    ///
    /// Panics unless `level` is in `1..=MAX_LEVEL`.
    pub fn layer_overflow(&self, level: usize) -> DirectionalOverflow {
        assert!((1..=MAX_LEVEL as usize).contains(&level), "invalid level");
        self.overflow[level].maximum()
    }

    fn item_overflow(&self, bounds: &Rectangle, (level, x, y): Coord) -> DirectionalOverflow {
        let geometry = &self.geometry[level];
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let min_x = origin_x + geometry.node_width * x as f64;
        let max_x = origin_x + geometry.node_width * (x + 1) as f64;
        let min_y = origin_y + geometry.node_height * y as f64;
        let max_y = origin_y + geometry.node_height * (y + 1) as f64;
        DirectionalOverflow {
            left: min_x - bounds.min_x,
            right: bounds.max_x - max_x,
            top: min_y - bounds.min_y,
            bottom: bounds.max_y - max_y,
        }
    }

    /// Find the level and coordinates of the node that holds an item with
    /// these bounds.
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
    /// but fewer than `2^L` times along the tighter axis, clamped to
    /// `1..=MAX_LEVEL`. [`LayerGeometry`] caches the matching size limits.
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

    fn get_root(&self) -> &OverflowBoundGridNode<T> {
        unsafe { self.layers.get_unchecked(1).get_unchecked(0, 0) }
    }

    /// Insert an item into the grid and return the handle that identifies
    /// it for [`update`](Self::update) and [`remove`](Self::remove).
    ///
    /// Positioning and overflow maintenance are O(1) for the fixed 64-bit key
    /// width; growth of the node and slot vectors is amortized O(1).
    ///
    /// Panics if a coordinate of `bounds` is NaN, which the overflow
    /// multisets cannot order.
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord @ (level, x, y) = self.position(&bounds);
        let overflow = self.item_overflow(&bounds, coord);
        let node = &mut self.layers[level][y][x];
        let handle = self
            .slots
            .allocate(GridLocation::new(coord, node.items.len()));
        node.items.push(Entry {
            bounds,
            item,
            handle,
        });
        self.occupy(level);
        self.overflow[level].insert(overflow, self.slack);
        handle
    }

    /// Remove an item from the grid and refresh its level's overflow maxima.
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let (level, x, y) = location.coord();
        let entry = self.take_entry(level, x, y, location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        let overflow = self.item_overflow(&entry.bounds, (level, x, y));
        self.overflow[level].remove(overflow, self.slack);
        self.slots.free(handle);
        self.vacate(level);
    }

    /// Update the bounds of an item, its position if necessary, and the
    /// overflow maxima.
    ///
    /// Panics if the handle has already been removed, or if a coordinate of
    /// `bounds` is NaN.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let prev_coord @ (level, x, y) = location.coord();
        let i = location.index as usize;
        let old_overflow =
            self.item_overflow(&self.layers[level][y][x].items[i].bounds, prev_coord);

        if self.stays_in_node(&bounds, prev_coord) {
            let new_overflow = self.item_overflow(&bounds, prev_coord);
            self.overflow[level].replace(old_overflow, new_overflow, self.slack);
            self.layers[level][y][x].items[i].bounds = bounds;
            return;
        }

        let curt_coord = self.position(&bounds);
        if curt_coord == prev_coord {
            let new_overflow = self.item_overflow(&bounds, prev_coord);
            self.overflow[level].replace(old_overflow, new_overflow, self.slack);
            self.layers[level][y][x].items[i].bounds = bounds;
            return;
        }

        let (nl, nx, ny) = curt_coord;
        let new_overflow = self.item_overflow(&bounds, curt_coord);
        if nl == level {
            self.overflow[level].replace(old_overflow, new_overflow, self.slack);
        } else {
            self.overflow[level].remove(old_overflow, self.slack);
            self.overflow[nl].insert(new_overflow, self.slack);
        }
        let entry = self.take_entry(level, x, y, i);
        let new_node = &mut self.layers[nl][ny][nx];
        *self.slots.get_mut(handle) = GridLocation::new(curt_coord, new_node.items.len());
        new_node.items.push(Entry { bounds, ..entry });
        self.occupy(nl);
        self.vacate(level);
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

    /// Take the entry at `index` out of a node by swap removal. The entry that
    /// fills the gap has its slot updated, so handles stay valid without a
    /// lookup.
    ///
    /// Takes plain coordinates rather than a `GridLocation`; keeping the whole
    /// location alive across `update` measured several percent slower.
    fn take_entry(&mut self, level: usize, x: usize, y: usize, index: usize) -> Entry<T> {
        let node = &mut self.layers[level][y][x];
        let entry = node.items.swap_remove(index);
        if let Some(moved) = node.items.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        entry
    }

    /// Check cheaply whether `position` would still return `coord` for
    /// `bounds`.
    ///
    /// The cached limits reproduce the level arithmetic of `position` and its
    /// multiplication by the reciprocal of the node size, truncation and
    /// clamp exactly, so a resize within the same level also passes.
    /// Returns false on levels without the update shortcut and for degenerate
    /// sizes; the caller then falls back to `position`.
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
    /// the `self_query` bench group measured it 16% faster than searching
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

    /// Search from the bottom up, calling `callback` for each item found.
    ///
    /// Follows [`HierarchicalLooseGrid::search`], except that each level
    /// reaches beyond its cells only as far as its overflow maxima, plus the
    /// rounding margin, require.
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
            // Top-left corner relative to the root origin.
            let min_x = bounds.min_x - offset_x;
            let min_y = bounds.min_y - offset_y;

            // Bottom-right corner relative to the root origin.
            let max_x = bounds.max_x - offset_x;
            let max_y = bounds.max_y - offset_y;

            // Scanning every level this way, without the ancestor walk, is slower.
            layers[level + 1..=max_level]
                .iter()
                .zip(&self.geometry[level + 1..=max_level])
                .zip(&self.overflow[level + 1..=max_level])
                .zip(&self.inverse_node_size[level + 1..=max_level])
                .rev()
                .for_each(|(((grid, geometry), overflow), &(inverse_width, inverse_height))| {
                    if geometry.is_empty() {
                        return;
                    }
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();

                    // Cells before the query matter as far as their items extend
                    // forward, cells after it as far as theirs extend backward.
                    // Coordinates are in nodes and a shrink in half nodes, so a
                    // shrink counts half.
                    let [left, right, top, bottom] = overflow.shrink;
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

                            // The range only shrinks with the maxima, so this
                            // inner part lies within that of the fixed reach.
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

        let (offset_x, offset_y) = self.root_bounds.get_min();
        let node_width = self.geometry[level].node_width;
        let node_height = self.geometry[level].node_height;

        let mut center_x = offset_x + node_width * (x as f64 + 0.5);
        let mut center_y = offset_y + node_height * (y as f64 + 0.5);
        let mut half_width = node_width * 0.5;
        let mut half_height = node_height * 0.5;

        // The root is searched last; levels coarser than `coarsest` hold nothing.
        let coarsest = self.coarsest_level().min(level + 1);
        self.layers[coarsest..=level]
            .iter()
            .zip(&self.geometry[coarsest..=level])
            .zip(&self.overflow[coarsest..=level])
            .rev()
            .for_each(|((grid, geometry), overflow)| unsafe {
                if !geometry.is_empty() {
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();
                    // The unchecked reads below rely on the walk staying in the grid.
                    debug_assert!(
                        x < grid_width && y < grid_height,
                        "walk cell ({x}, {y}) outside {grid_width}x{grid_height}"
                    );

                    // How far the items of each neighboring column and row reach
                    // into this cell; the fixed reach ends at its center.
                    let [left, right, top, bottom] = overflow.shrink;
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
                // Advance to the parent cell, also on an empty level.
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

struct OverflowBoundGridNode<T: Copy> {
    items: Vec<Entry<T>>,
}

impl<T: Copy> OverflowBoundGridNode<T> {
    fn new() -> Self {
        Self { items: Vec::new() }
    }

    fn search_items(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
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
    use crate::bitmap_max_multiset::key_of;
    use rand::prelude::*;
    use std::collections::BTreeMap;

    /// Maxima of a level without items.
    const NOWHERE: DirectionalOverflow = DirectionalOverflow {
        left: f64::NEG_INFINITY,
        right: f64::NEG_INFINITY,
        top: f64::NEG_INFINITY,
        bottom: f64::NEG_INFINITY,
    };

    const MAX_LEVEL: u8 = 7;
    type Tree = OverflowBoundGrid<usize, MAX_LEVEL>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    crate::grid_tests::exact_reach_tests!();

    /// Moves and resizes through `update` must keep every slot and per-level
    /// object count consistent with `position`, also after removals and the
    /// slot reuse that follows them.
    #[test]
    fn update_keeps_slots_and_counts_consistent() {
        let mut rng = StdRng::seed_from_u64(1);
        for mut tree in trees() {
            let mut bounds_of: Vec<Rectangle> =
                (0..500).map(|_| random_bounds(&mut rng, &tree)).collect();
            let mut handles: Vec<ItemHandle> = bounds_of
                .iter()
                .enumerate()
                .map(|(item, bounds)| tree.insert(bounds.clone(), item))
                .collect();

            let finest_node_width = tree.geometry[MAX_LEVEL as usize].node_width;
            for step_index in 0..20_000 {
                let item = rng.gen_range(0..bounds_of.len());
                let bounds = if rng.gen_bool(0.5) {
                    // A small move or resize, usually within the same node.
                    let old = &bounds_of[item];
                    let (center_x, center_y) = old.get_center();
                    let step = finest_node_width * 0.1;
                    Rectangle::center_rect(
                        center_x + rng.gen_range(-step..step),
                        center_y + rng.gen_range(-step..step),
                        old.get_width() * rng.gen_range(0.9..1.1),
                        old.get_height() * rng.gen_range(0.9..1.1),
                    )
                } else {
                    random_bounds(&mut rng, &tree)
                };
                tree.update(bounds.clone(), handles[item]);
                bounds_of[item] = bounds;
                if step_index % 250 == 0 {
                    assert_overflows(&tree);
                }
            }
            assert_consistent(&tree, &handles, &bounds_of);

            for handle in handles.drain(250..) {
                tree.remove(handle);
            }
            bounds_of.truncate(250);
            assert_consistent(&tree, &handles, &bounds_of);

            // New items take over the freed slots.
            for item in 250..400 {
                let bounds = random_bounds(&mut rng, &tree);
                handles.push(tree.insert(bounds.clone(), item));
                bounds_of.push(bounds);
            }
            assert_eq!(tree.slots.slot_count(), 500);
            assert_consistent(&tree, &handles, &bounds_of);
        }
    }

    fn assert_consistent(tree: &Tree, handles: &[ItemHandle], bounds_of: &[Rectangle]) {
        assert_overflows(tree);
        assert_eq!(tree.slots.len(), handles.len());
        for (item, (handle, bounds)) in handles.iter().zip(bounds_of).enumerate() {
            let location = tree.slots.get(*handle);
            assert_eq!(
                location.coord(),
                tree.position(bounds),
                "location of {item}"
            );
            let (level, x, y) = location.coord();
            let entry = &tree.layers[level][y][x].items[location.index as usize];
            assert_eq!(entry.item, item);
            assert_eq!(entry.handle, *handle);
            assert_eq!(entry.bounds, *bounds);
        }
        for (level, (grid, geometry)) in tree.layers.iter().zip(&tree.geometry).enumerate() {
            let stored: usize = grid.iter().map(|node| node.items.len()).sum();
            assert_eq!(
                stored, geometry.object_count,
                "object count of level {level}"
            );
        }
    }

    /// Rebuild the statistics from the live bounds, with cell edges from the
    /// root size and level rather than the mutation helpers or cached geometry.
    fn assert_overflows(tree: &Tree) {
        let root = tree.get_bounds();
        for level in 1..=MAX_LEVEL as usize {
            let mut counts: [BTreeMap<u64, usize>; 4] = Default::default();
            let divisor = (1usize << (level - 1)) as f64;
            let width = root.get_width() / divisor;
            let height = root.get_height() / divisor;
            let grid = &tree.layers[level];
            for y in 0..grid.rows() {
                for x in 0..grid.cols() {
                    for entry in &grid[y][x].items {
                        let b = &entry.bounds;
                        let distances = [
                            root.min_x + x as f64 * width - b.min_x,
                            b.max_x - (root.min_x + (x + 1) as f64 * width),
                            root.min_y + y as f64 * height - b.min_y,
                            b.max_y - (root.min_y + (y + 1) as f64 * height),
                        ];
                        for (count, distance) in counts.iter_mut().zip(distances) {
                            *count.entry(key_of(distance)).or_default() += 1;
                        }
                    }
                }
            }
            for ((actual, expected), cached_max) in tree.overflow[level]
                .directions
                .iter()
                .zip(counts)
                .zip(tree.layer_overflow(level).values())
            {
                assert_eq!(actual.counts(), expected, "counts on level {level}");
                assert_eq!(
                    key_of(cached_max),
                    expected.last_key_value().map_or(0, |(&key, _)| key),
                    "maximum on level {level}"
                );
            }

            // Search reaches as far as the maxima plus the rounding margin,
            // capped at half a node; the root is always scanned in full.
            if level >= 2 {
                let maxima = tree.layer_overflow(level).values();
                let half_nodes = [width, width, height, height].map(|size| size * 0.5);
                for direction in 0..4 {
                    let half_node = half_nodes[direction];
                    let reach = half_node * (1.0 - tree.overflow[level].shrink[direction]);
                    let expected = (maxima[direction] + tree.slack).min(half_node);
                    // On a level without items both sides are -infinity.
                    assert!(
                        reach == expected || (reach - expected).abs() <= half_node * 1e-12,
                        "reach {reach} instead of {expected} on level {level}, direction {direction}"
                    );
                }
            }
        }
    }

    #[test]
    fn four_directions_duplicates_and_maximum_removal() {
        // Level 3 has 16-unit nodes; these items sit in the one from 16 to 32.
        let mut grid = OverflowBoundGrid::<u8, 3>::new(Rectangle::new(0.0, 0.0, 64.0, 64.0));
        let inside = grid.insert(Rectangle::center_rect(24.0, 24.0, 12.0, 12.0), 0);
        assert_eq!(grid.layer_overflow(3).values(), [-2.0; 4]);
        let left_bounds = Rectangle::center_rect(17.0, 24.0, 12.0, 12.0);
        let left_a = grid.insert(left_bounds.clone(), 1);
        let left_b = grid.insert(left_bounds, 1); // Items need not be unique.
        let smaller = grid.insert(Rectangle::center_rect(20.0, 24.0, 12.0, 12.0), 2);
        let right = grid.insert(Rectangle::center_rect(31.0, 24.0, 12.0, 12.0), 3);
        let top = grid.insert(Rectangle::center_rect(24.0, 17.0, 12.0, 12.0), 4);
        let bottom = grid.insert(Rectangle::center_rect(24.0, 31.0, 12.0, 12.0), 5);
        assert_eq!(
            grid.layer_overflow(3),
            DirectionalOverflow {
                left: 5.0,
                right: 5.0,
                top: 5.0,
                bottom: 5.0,
            }
        );
        assert_eq!(grid.layer_overflow(2), NOWHERE);
        grid.remove(left_a);
        assert_eq!(grid.layer_overflow(3).left, 5.0);
        grid.remove(left_b);
        assert_eq!(grid.layer_overflow(3).left, 2.0);
        // The remaining items all keep at least 2 units inside the left edge.
        grid.remove(smaller);
        assert_eq!(grid.layer_overflow(3).left, -2.0);
        for handle in [inside, right, top, bottom] {
            grid.remove(handle);
        }
        assert_eq!(grid.layer_overflow(3), NOWHERE);
        assert!(
            grid.overflow[3]
                .directions
                .iter()
                .all(MaxMultiset::is_empty)
        );
    }

    #[test]
    fn updates_refresh_overflow_in_same_cell_across_cells_and_levels() {
        let mut grid = OverflowBoundGrid::<u8, 3>::new(Rectangle::new(0.0, 0.0, 64.0, 64.0));
        let handle = grid.insert(Rectangle::center_rect(17.0, 17.0, 12.0, 12.0), 0);
        let cases = [
            (
                Rectangle::center_rect(21.0, 21.0, 12.0, 12.0),
                (3, 1, 1),
                [1.0, -5.0, 1.0, -5.0],
            ),
            (
                Rectangle::center_rect(26.0, 26.0, 12.0, 12.0),
                (3, 1, 1),
                [-4.0, 0.0, -4.0, 0.0],
            ),
            (
                Rectangle::center_rect(31.0, 31.0, 12.0, 12.0),
                (3, 1, 1),
                [-9.0, 5.0, -9.0, 5.0],
            ),
            (
                Rectangle::center_rect(45.0, 17.0, 12.0, 12.0),
                (3, 2, 1),
                [-7.0, 3.0, 5.0, -9.0],
            ),
            (
                Rectangle::center_rect(31.0, 31.0, 24.0, 24.0),
                (2, 0, 0),
                [-19.0, 11.0, -19.0, 11.0],
            ),
            (
                Rectangle::center_rect(-10.0, -20.0, 4.0, 4.0),
                (1, 0, 0),
                [12.0, -72.0, 22.0, -82.0],
            ),
            (
                Rectangle::center_rect(24.0, 24.0, 12.0, 12.0),
                (3, 1, 1),
                [-2.0; 4],
            ),
        ];
        for (bounds, coord, distances) in cases {
            grid.update(bounds, handle);
            assert_eq!(grid.slots.get(handle).coord(), coord);
            for level in 1..=3 {
                assert_eq!(
                    grid.layer_overflow(level).values(),
                    if level == coord.0 {
                        distances
                    } else {
                        NOWHERE.values()
                    }
                );
            }
        }
        grid.remove(handle);
        for level in 1..=3 {
            assert_eq!(grid.layer_overflow(level), NOWHERE);
        }
    }

    #[test]
    fn fitted_cells_use_width_and_height_and_root_overflow_is_unbounded() {
        let mut grid =
            OverflowBoundGrid::<u8, 3>::new_fit(Rectangle::new(-100.0, 200.0, 28.0, 264.0));
        let handle = grid.insert(Rectangle::center_rect(-67.0, 217.0, 20.0, 12.0), 0);
        assert_eq!(
            grid.position(&Rectangle::center_rect(-67.0, 217.0, 20.0, 12.0)),
            (3, 1, 1)
        );
        assert_eq!(grid.layer_overflow(3).values(), [9.0, -21.0, 5.0, -9.0]);
        grid.update(Rectangle::new(-1000.0, -2000.0, 3000.0, 4000.0), handle);
        assert_eq!(grid.layer_overflow(3), NOWHERE);
        assert_eq!(
            grid.layer_overflow(1).values(),
            [900.0, 2972.0, 2200.0, 3736.0]
        );
        // A move and resize within the root takes the full-position fallback.
        grid.update(Rectangle::new(-101.0, 199.0, 29.0, 265.0), handle);
        assert_eq!(grid.layer_overflow(1).values(), [1.0; 4]);
        grid.remove(handle);
        assert_eq!(grid.layer_overflow(1), NOWHERE);
    }

    /// Items inside their cell edges give negative maxima. Searches must still
    /// find exactly the intersecting items, including those that overlap a
    /// query by a single float.
    #[test]
    fn negative_level_maxima_keep_searches_exact() {
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

        let maxima = tree.layer_overflow(MAX_LEVEL as usize).values();
        assert!(maxima.iter().all(|&reach| reach < -1.0), "{maxima:?}");
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
                    2 => Rectangle::new(b.min_x, b.max_y.next_down(), b.max_x, b.max_y + height),
                    _ => Rectangle::new(b.min_x, b.min_y - height, b.max_x, b.min_y.next_up()),
                }
            };
            let expected: Vec<usize> = (0..items.len())
                .filter(|&i| items[i].intersects(&query))
                .collect();
            let mut found = Vec::new();
            tree.search(&query, |id| found.push(id));
            found.sort_unstable();
            assert_eq!(found, expected, "query {query:?}");
        }
    }

    #[test]
    #[should_panic(expected = "invalid level")]
    fn unused_level_has_no_public_overflow() {
        let grid = Tree::new(WORLDS[0].clone());
        grid.layer_overflow(0);
    }

    #[test]
    #[should_panic(expected = "invalid level")]
    fn out_of_range_level_has_no_public_overflow() {
        let grid = Tree::new(WORLDS[0].clone());
        grid.layer_overflow(MAX_LEVEL as usize + 1);
    }
}
