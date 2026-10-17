use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::layer_geometry::LayerGeometry;
use crate::overflow_bound_grid::DirectionalOverflow;
use crate::overflow_bound::{LevelReach, cell_overflow, rounding_slack};
use crate::rect::Rectangle;
use grid::*;

/// (level, x, y)
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<LazyOverflowBoundGridNode<T>>>;

/// Overflow-bound grid with high-water bounds per level: the storage,
/// positioning and handles of [`HierarchicalLooseGrid`], with a search that
/// shrinks the fixed reach (half a node) of every level to the largest
/// overflow beyond the cells seen on that level since the last
/// [`refresh_bounds`].
///
/// A bound may be negative: when every item of a level stays inside its cell
/// edge in some direction, search enters a cell across that edge only where
/// the query crosses the edge by more than the smallest such gap, less the
/// rounding margin.
///
/// Insertion and update only raise the bounds, with one comparison per
/// direction; removal leaves them unchanged. A raised bound remains valid as
/// items move, shrink or leave, and drops only when `refresh_bounds`
/// recomputes it, at a time the caller chooses, e.g. once per frame after the
/// updates.
///
/// [`HierarchicalLooseGrid`]: crate::HierarchicalLooseGrid
/// [`refresh_bounds`]: Self::refresh_bounds
pub struct LazyOverflowBoundGrid<T: Copy, const MAX_LEVEL: u8> {
    layers: Layers<T>,
    /// Indexed by level, parallel to `layers`.
    geometry: Vec<LayerGeometry>,
    /// Indexed by level, parallel to `layers`.
    reach: Vec<LevelReach>,
    /// Rounding margin that search adds to every reach, since its cell edges
    /// accumulate rounding differently from [`cell_overflow`].
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
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let world_width = world_bounds.get_width();
        let world_height = world_bounds.get_height();

        // Initialize the layers.
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
                    vec.push(LazyOverflowBoundGridNode::new());
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

        // The root bound grows like any other but is never read, since search
        // scans the root entirely.
        let reach = geometry
            .iter()
            .map(|geometry| LevelReach::new(geometry.node_width, geometry.node_height, slack))
            .collect();

        LazyOverflowBoundGrid {
            layers,
            geometry,
            reach,
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

    /// Lower the bound of every level below the root to the largest overflow
    /// of the items now stored on it, undoing the growth from items that have
    /// since moved, shrunk or been removed. Visits every cell and item below
    /// the root at most once.
    pub fn refresh_bounds(&mut self) {
        let origin = self.root_bounds.get_min();
        for level in 2..=MAX_LEVEL as usize {
            let geometry = &self.geometry[level];
            let node_size = (geometry.node_width, geometry.node_height);
            let mut largest = [f64::NEG_INFINITY; 4];
            if !geometry.is_empty() {
                let grid = &self.layers[level];
                for y in 0..grid.rows() {
                    for x in 0..grid.cols() {
                        for entry in &grid[y][x].items {
                            let overflow = cell_overflow(&entry.bounds, origin, node_size, (x, y));
                            for (largest, overflow) in largest.iter_mut().zip(overflow) {
                                *largest = largest.max(overflow);
                            }
                        }
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
        let multiple_x = (self.root_bounds.get_width() / width) as usize;
        let multiple_y = (self.root_bounds.get_height() / height) as usize;
        let level = usize::BITS - multiple_x.min(multiple_y).leading_zeros();
        (level as usize).clamp(1, MAX_LEVEL as usize)
    }

    /// Return the cell of `level` that contains `point`, clamped to the grid.
    fn cell_containing(&self, level: usize, point: (f64, f64)) -> (usize, usize) {
        let grid = &self.layers[level];
        let geometry = &self.geometry[level];
        let (origin_x, origin_y) = self.root_bounds.get_min();
        let x = (((point.0 - origin_x) / geometry.node_width) as usize).min(grid.cols() - 1);
        let y = (((point.1 - origin_y) / geometry.node_height) as usize).min(grid.rows() - 1);
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
        let handle = self
            .slots
            .allocate(GridLocation::new(coord, node.items.len()));
        node.items.push(Entry {
            bounds,
            item,
            handle,
        });
        self.geometry[level].object_count += 1;
        handle
    }

    /// Remove an item. Its level keeps its bound until the next
    /// [`refresh_bounds`](Self::refresh_bounds).
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let (level, x, y) = location.coord();
        let entry = self.take_entry(level, x, y, location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        self.slots.free(handle);
        self.geometry[level].object_count -= 1;
    }

    /// Update the bounds of an item and, if necessary, move it to another node.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let prev_coord @ (level, x, y) = location.coord();
        let i = location.index as usize;

        if self.stays_in_node(&bounds, prev_coord) {
            self.grow_reach(&bounds, prev_coord);
            self.layers[level][y][x].items[i].bounds = bounds;
            return;
        }

        let curt_coord = self.position(&bounds);
        self.grow_reach(&bounds, curt_coord);
        if curt_coord == prev_coord {
            self.layers[level][y][x].items[i].bounds = bounds;
            return;
        }

        let (nl, nx, ny) = curt_coord;
        let entry = self.take_entry(level, x, y, i);
        let new_node = &mut self.layers[nl][ny][nx];
        *self.slots.get_mut(handle) = GridLocation::new(curt_coord, new_node.items.len());
        new_node.items.push(Entry { bounds, ..entry });
        self.geometry[nl].object_count += 1;
        self.geometry[level].object_count -= 1;
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
        entry
    }

    /// Check cheaply that `position` would still return `coord` for `bounds`.
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

    /// Search from the bottom up, calling `callback` for each item found.
    ///
    /// Follows [`HierarchicalLooseGrid::search`], except that each level
    /// reaches beyond its cells only as far as its overflow bound allows.
    ///
    /// [`HierarchicalLooseGrid::search`]: crate::HierarchicalLooseGrid::search
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
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

            let calc_min = |n: f64| (n.round() as usize).saturating_sub(1);
            let calc_max = |n: f64| (n - 0.5).ceil() as usize;

            layers[level + 1..=max_level]
                .iter()
                .zip(&self.geometry[level + 1..=max_level])
                .zip(&self.reach[level + 1..=max_level])
                .rev()
                .for_each(|((grid, geometry), reach)| {
                    if geometry.is_empty() {
                        return;
                    }
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();
                    let node_width = geometry.node_width;
                    let node_height = geometry.node_height;

                    // Cells at lower coordinates than the query range count as
                    // far as their items reach in the positive direction, and
                    // vice versa.
                    // Coordinates are in nodes and the shrink in half nodes, so
                    // the shrink counts half.
                    let [left, right, top, bottom] = reach.shrink;
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
            .zip(&self.reach[2..=level])
            .rev()
            .for_each(|((grid, geometry), reach)| unsafe {
                if !geometry.is_empty() {
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();

                    // Coordinates up to which the items of each neighboring
                    // column and row reach into this cell; the fixed reach ends
                    // at its center.
                    let [left, right, top, bottom] = reach.shrink;
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
    items: Vec<Entry<T>>,
}

impl<T: Copy> LazyOverflowBoundGridNode<T> {
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

unsafe impl<T: Copy, const MAX_LEVEL: u8> Send for LazyOverflowBoundGrid<T, MAX_LEVEL> {}
unsafe impl<T: Copy, const MAX_LEVEL: u8> Sync for LazyOverflowBoundGrid<T, MAX_LEVEL> {}

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

    /// Bounds of any size from the root down to below the finest node, centered
    /// anywhere in the root and therefore sometimes outside the world.
    fn random_bounds(rng: &mut StdRng, root: &Rectangle) -> Rectangle {
        let scale = 2f64.powf(-rng.gen_range(0.0..MAX_LEVEL as f64 + 2.0));
        let width = root.get_width() * scale * rng.gen_range(0.5..1.0);
        let height = root.get_height() * scale * rng.gen_range(0.5..1.0);
        let (x, y) = root.get_random_point(rng);
        Rectangle::center_rect(x, y, width, height)
    }

    /// A small move that usually keeps the item in its node but changes its
    /// overflow.
    fn nudge(rng: &mut StdRng, bounds: &Rectangle) -> Rectangle {
        let (x, y) = bounds.get_center();
        let (width, height) = (bounds.get_width(), bounds.get_height());
        Rectangle::center_rect(
            x + width * rng.gen_range(-0.2..0.2),
            y + height * rng.gen_range(-0.2..0.2),
            width,
            height,
        )
    }

    /// Largest overflow per direction over the entries stored on `level`.
    fn exact_overflow(tree: &Tree, level: usize) -> [f64; 4] {
        let geometry = &tree.geometry[level];
        let grid = &tree.layers[level];
        let node_size = (geometry.node_width, geometry.node_height);
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

    /// Between refreshes, every bound must cover the overflow of every item on
    /// its level; right after a refresh, it must equal the largest overflow
    /// exactly.
    #[test]
    fn bounds_cover_items_and_match_after_refresh() {
        let mut rng = StdRng::seed_from_u64(0);
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

                    if step % 100 == 0 {
                        for level in 2..=MAX_LEVEL as usize {
                            let exact = exact_overflow(&tree, level);
                            let bound = directions(tree.overflow_bound(level));
                            for direction in 0..4 {
                                assert!(
                                    bound[direction] >= exact[direction],
                                    "level {level} direction {direction}: stale bound {} below overflow {}",
                                    bound[direction],
                                    exact[direction]
                                );
                            }
                        }
                    }
                    if step % 300 == 0 {
                        tree.refresh_bounds();
                        for level in 2..=MAX_LEVEL as usize {
                            assert_eq!(
                                directions(tree.overflow_bound(level)),
                                exact_overflow(&tree, level),
                                "level {level} after refresh"
                            );
                        }
                    }
                }
            }
        }
    }

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

    /// Items that stay inside their cell edges yield negative bounds, before
    /// and after a refresh. Search must still return exactly the intersecting
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
}
