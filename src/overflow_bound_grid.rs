use crate::bitmap_max_multiset::MaxMultiset;
use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::layer_geometry::LayerGeometry;
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
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        let n = 1usize
            .checked_shl((MAX_LEVEL - 1) as u32)
            .expect("`MAX_LEVEL` is too large and will cause overflow.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let world_width = world_bounds.get_width();
        let world_height = world_bounds.get_height();

        // One grid per level; level zero has no cells.
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
                    vec.push(OverflowBoundGridNode::new());
                }
            }
            layers.push(Grid::from_vec(vec, cols));
            size = 1 << n;
        }

        // Root bounds: unless fitted, a square from the world's minimum corner.
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
        geometry.push(LayerGeometry::default()); // Level zero is unused, like `layers[0]`.
        for (level, grid) in layers.iter().enumerate().skip(1) {
            geometry.push(LayerGeometry::new(
                &root_bounds,
                level,
                grid.cols(),
                grid.rows(),
            ));
        }

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
        let multiple_x = (self.root_bounds.get_width() / width) as usize;
        let multiple_y = (self.root_bounds.get_height() / height) as usize;

        // Equivalent to:
        // let multiple = multiple_x.min(multiple_y);
        // let level = if multiple != 0 {
        //     ((multiple.ilog2() + 1) as u8).min(MAX_LEVEL)
        // } else {
        //     1
        // };
        let level = usize::BITS - multiple_x.min(multiple_y).leading_zeros();
        (level as usize).clamp(1, MAX_LEVEL as usize)
    }

    /// Cell of `level` whose area contains `point`, clamped into the grid.
    fn cell_containing(&self, level: usize, point: (f64, f64)) -> (usize, usize) {
        let grid = &self.layers[level];
        let geometry = &self.geometry[level];
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let x = (((point.0 - origin_x) / geometry.node_width) as usize).min(grid.cols() - 1);
        let y = (((point.1 - origin_y) / geometry.node_height) as usize).min(grid.rows() - 1);
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
        self.geometry[level].object_count += 1;
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
        self.geometry[level].object_count -= 1;
    }

    /// Update the bounds of an item, its position if necessary, and the
    /// overflow maxima.
    ///
    /// Panics if the handle has already been removed.
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
        self.geometry[nl].object_count += 1;
        self.geometry[level].object_count -= 1;
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
    /// The cached limits reproduce the division, truncation and clamp of
    /// `position` exactly, so a resize within the same level also passes.
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

    /// Search from the bottom up, calling `callback` for each item found.
    ///
    /// Follows [`HierarchicalLooseGrid::search`], except that each level
    /// reaches beyond its cells only as far as its overflow maxima, plus the
    /// rounding margin, require.
    ///
    /// [`HierarchicalLooseGrid::search`]: crate::HierarchicalLooseGrid::search
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
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

            let calc_min = |n: f64| (n.round() as usize).saturating_sub(1);
            let calc_max = |n: f64| (n - 0.5).ceil() as usize;

            // Scanning every level this way, without the ancestor walk, is slower.
            layers[level + 1..=max_level]
                .iter()
                .zip(&self.geometry[level + 1..=max_level])
                .zip(&self.overflow[level + 1..=max_level])
                .rev()
                .for_each(|((grid, geometry), overflow)| {
                    if geometry.is_empty() {
                        return;
                    }
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();
                    let node_width = geometry.node_width;
                    let node_height = geometry.node_height;

                    // Cells before the query matter as far as their items extend
                    // forward, cells after it as far as theirs extend backward.
                    // Coordinates are in nodes, so a shrink, a fraction of half a
                    // node, counts half.
                    let [left, right, top, bottom] = overflow.shrink;
                    let min_coord_x = calc_min(min_x / node_width + right * 0.5);
                    let min_coord_y = calc_min(min_y / node_height + bottom * 0.5);

                    let max_coord_x = calc_max(max_x / node_width - left * 0.5).min(grid_width - 1);
                    let max_coord_y =
                        calc_max(max_y / node_height - top * 0.5).min(grid_height - 1);

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

        self.layers[2..=level]
            .iter()
            .zip(&self.geometry[2..=level])
            .zip(&self.overflow[2..=level])
            .rev()
            .for_each(|((grid, geometry), overflow)| unsafe {
                if !geometry.is_empty() {
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();

                    // How far the items of each neighboring column and row reach
                    // into this cell; the fixed reach ends at its center.
                    let [left, right, top, bottom] = overflow.shrink;
                    let left_column_reach = center_x - right * half_width;
                    let right_column_reach = center_x + left * half_width;
                    let top_row_reach = center_y - bottom * half_height;
                    let bottom_row_reach = center_y + top * half_height;

                    if bounds.min_y < top_row_reach && y != 0 {
                        if bounds.min_x < left_column_reach && x != 0 {
                            let node = grid.get_unchecked(y - 1, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y - 1, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x > right_column_reach && x + 1 != grid_width {
                            let node = grid.get_unchecked(y - 1, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }

                    {
                        if bounds.min_x < left_column_reach && x != 0 {
                            let node = grid.get_unchecked(y, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x > right_column_reach && x + 1 != grid_width {
                            let node = grid.get_unchecked(y, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }

                    if bounds.max_y > bottom_row_reach && y + 1 != grid_height {
                        if bounds.min_x < left_column_reach && x != 0 {
                            let node = grid.get_unchecked(y + 1, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y + 1, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x > right_column_reach && x + 1 != grid_width {
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
    use std::ops::Range;

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

    fn trees() -> Vec<Tree> {
        WORLDS
            .iter()
            .flat_map(|world| [Tree::new(world.clone()), Tree::new_fit(world.clone())])
            .collect()
    }

    /// Move `value` a few floats up or down, onto either side of a boundary.
    fn jitter(rng: &mut StdRng, mut value: f64) -> f64 {
        for _ in 0..rng.gen_range(0..4) {
            value = if rng.gen_bool(0.5) {
                value.next_up()
            } else {
                value.next_down()
            };
        }
        value
    }

    /// Jitter around one of `candidates`, or fall back to a uniform value.
    fn near_one_of(rng: &mut StdRng, candidates: &[f64], fallback: Range<f64>) -> f64 {
        if candidates.is_empty() || rng.gen_bool(0.3) {
            rng.gen_range(fallback)
        } else {
            let candidate = candidates[rng.gen_range(0..candidates.len())];
            jitter(rng, candidate)
        }
    }

    /// Random bounds biased towards the size limits and cell edges of a random
    /// level, where `position` and the cached geometry most likely disagree.
    fn random_bounds(rng: &mut StdRng, tree: &Tree) -> Rectangle {
        let root = tree.get_bounds();
        let world = &tree.world_bounds;
        let geometry = &tree.geometry[rng.gen_range(1..=MAX_LEVEL as usize)];

        // The node size is the size limit of the level; half of it is that of
        // the next finer level.
        let size_limits = |node_size: f64| -> Vec<f64> {
            if geometry.has_update_shortcut() {
                vec![node_size, node_size * 0.5]
            } else {
                Vec::new()
            }
        };
        let width_limits = size_limits(geometry.node_width);
        let height_limits = size_limits(geometry.node_height);
        let width = near_one_of(rng, &width_limits, 0.0..root.get_width() * 1.1);
        let height = near_one_of(rng, &height_limits, 0.0..root.get_height() * 1.1);

        // Only the inner edges are finite; the outer ones are +-infinity.
        let inner_edges = |edges: &[f64], origin: f64| -> Vec<f64> {
            edges
                .get(1..edges.len().saturating_sub(1))
                .unwrap_or(&[])
                .iter()
                .map(|edge| origin + edge)
                .collect()
        };
        let x_edges = inner_edges(&geometry.cell_x_edges, root.min_x);
        let y_edges = inner_edges(&geometry.cell_y_edges, root.min_y);
        let pad_x = world.get_width() * 0.05;
        let pad_y = world.get_height() * 0.05;
        let center_x = near_one_of(rng, &x_edges, world.min_x - pad_x..world.max_x + pad_x);
        let center_y = near_one_of(rng, &y_edges, world.min_y - pad_y..world.max_y + pad_y);

        Rectangle::center_rect(center_x, center_y, width, height)
    }

    /// The update shortcut must agree with `position` exactly. A false positive
    /// would leave an item in the wrong node; a false negative would only cost
    /// time, but the cached limits are meant to be exact.
    #[test]
    fn update_shortcut_agrees_with_position() {
        let mut rng = StdRng::seed_from_u64(0);
        for tree in trees() {
            for _ in 0..100_000 {
                let bounds = random_bounds(&mut rng, &tree);
                let coord @ (level, x, y) = tree.position(&bounds);

                // Degenerate sizes intentionally bypass the shortcut.
                let is_degenerate = bounds.get_width() <= 0.0 || bounds.get_height() <= 0.0;
                if tree.geometry[level].has_update_shortcut() && !is_degenerate {
                    assert!(
                        tree.stays_in_node(&bounds, coord),
                        "false negative for {bounds:?} at {coord:?}"
                    );
                }

                // Neighboring cells on the same level must be rejected.
                let grid = &tree.layers[level];
                for dy in -1..=1isize {
                    for dx in -1..=1isize {
                        let (nx, ny) = (x.wrapping_add_signed(dx), y.wrapping_add_signed(dy));
                        if (dx, dy) != (0, 0) && nx < grid.cols() && ny < grid.rows() {
                            assert!(
                                !tree.stays_in_node(&bounds, (level, nx, ny)),
                                "false positive for {bounds:?} at ({level}, {nx}, {ny})"
                            );
                        }
                    }
                }

                // The containing cell on an adjacent level must be rejected too.
                for other_level in [level - 1, level + 1] {
                    if (1..=MAX_LEVEL as usize).contains(&other_level) {
                        let (ox, oy) = tree.cell_containing(other_level, bounds.get_center());
                        assert!(
                            !tree.stays_in_node(&bounds, (other_level, ox, oy)),
                            "false positive for {bounds:?} at ({other_level}, {ox}, {oy})"
                        );
                    }
                }
            }
        }
    }

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

    /// A removed handle must be rejected even after its slot has been reused.
    #[test]
    #[should_panic(expected = "has been removed")]
    fn stale_handle_is_rejected() {
        let mut tree = Tree::new(WORLDS[0].clone());
        let bounds = Rectangle::center_rect(100.0, 100.0, 10.0, 10.0);
        let stale = tree.insert(bounds.clone(), 0);
        tree.remove(stale);
        let reused = tree.insert(bounds.clone(), 1);
        assert_ne!(reused, stale);
        tree.update(bounds, stale);
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
