use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::layer_geometry::LayerGeometry;
use crate::rect::Rectangle;
use grid::*;

/// (level, x, y)
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<UpSearchQuadTreeNode<T>>>;

/// Optimized [`GridLooseQuadTree`] retaining only the up-search function.
///
/// The main change is to store only the items on each node to reduce the size of each node.
/// As the up-search method does not rely on intersection check of nodes, there is no need to
/// store the bounds of each node.
///
/// [`GridLooseQuadTree`]: crate::grid_loose_quadtree::GridLooseQuadTree
pub struct UpSearchQuadTree<T: Copy, const MAX_LEVEL: u8> {
    layers: Layers<T>,
    /// Indexed by level, parallel to `layers`.
    geometry: Vec<LayerGeometry>,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    /// Where the entry of each live handle is stored.
    slots: Slots<GridLocation>,
}

impl<T: Copy, const MAX_LEVEL: u8> UpSearchQuadTree<T, MAX_LEVEL> {
    /// Create a new quadtree with the given bounds.
    ///
    /// Force using a square as the bounds of each node. This usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> UpSearchQuadTree<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new quadtree with the given bounds.
    ///
    /// Make the bounds of each node fit the given bounds instead of forcing it to be square.
    pub fn new_fit(world_bounds: Rectangle) -> UpSearchQuadTree<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(world_bounds: Rectangle) -> UpSearchQuadTree<T, MAX_LEVEL> {
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let world_width = world_bounds.get_width();
        let world_height = world_bounds.get_height();

        // Initialize layers
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
                    vec.push(UpSearchQuadTreeNode::new());
                }
            }
            layers.push(Grid::from_vec(vec, cols));
            size = 1 << n;
        }

        // Determine the root bounds to use
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

        UpSearchQuadTree {
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

    /// Find the level and coordinates of the node where the item should be inserted.
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
        let multiple_x = (self.root_bounds.get_width() / width) as usize;
        let multiple_y = (self.root_bounds.get_height() / height) as usize;

        // Equals to:
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

    fn get_root(&self) -> &UpSearchQuadTreeNode<T> {
        unsafe { self.layers.get_unchecked(1).get_unchecked(0, 0) }
    }

    /// Insert an item into the quadtree and return the handle that identifies
    /// it for [`update`](Self::update) and [`remove`](Self::remove).
    ///
    /// The time complexity is O(1) since the insertion position is obtained directly
    /// by computation, instead of by judgment at each node.
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord @ (level, x, y) = self.position(&bounds);
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

    /// Remove an item from the quadtree.
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

    /// Update the bounds of an item and, if necessary, its position in the quadtree.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let prev_coord @ (level, x, y) = location.coord();
        let i = location.index as usize;

        if self.stays_in_node(&bounds, prev_coord) {
            self.layers[level][y][x].items[i].bounds = bounds;
            return;
        }

        let curt_coord = self.position(&bounds);
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

    /// Take the entry at `index` out of a node. The entry that fills the gap
    /// has its slot updated, so handles stay valid without any lookup.
    ///
    /// Takes plain coordinates rather than a `GridLocation`: keeping the whole
    /// location alive across `update` measured several percent slower.
    fn take_entry(&mut self, level: usize, x: usize, y: usize, index: usize) -> Entry<T> {
        let node = &mut self.layers[level][y][x];
        let entry = node.items.swap_remove(index);
        if let Some(moved) = node.items.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        entry
    }

    /// Cheap check that `position` would still return `coord` for `bounds`.
    ///
    /// The cached limits mirror position's division, truncation and clamp
    /// exactly, so resizing within the same level passes as well. Returns
    /// false on levels without the shortcut; the caller then falls back to
    /// `position`.
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

    /// Search from the bottom up. Execute the callback function for each item found.
    ///
    /// For details, please refer to the declaration of [`GridLooseQuadTree::search_up`]
    ///
    /// [`GridLooseQuadTree::search_up`]: crate::grid_loose_quadtree::GridLooseQuadTree::search_up
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        let max_level = MAX_LEVEL as usize;
        let level = self.level_for_size(bounds.get_width(), bounds.get_height());
        let (offset_x, offset_y) = self.root_bounds.get_min();

        let layers = &self.layers;
        if level != max_level {
            // top left
            let min_x = bounds.min_x - offset_x;
            let min_y = bounds.min_y - offset_y;

            // bottom right
            let max_x = bounds.max_x - offset_x;
            let max_y = bounds.max_y - offset_y;

            let calc_min = |n: f64| (n.round() as usize).saturating_sub(1);
            let calc_max = |n: f64| (n - 0.5).ceil() as usize;

            // Note: Searching directly from the bottom to the top using this method will be slower
            layers[level + 1..=max_level]
                .iter()
                .zip(&self.geometry[level + 1..=max_level])
                .rev()
                .for_each(|(grid, geometry)| {
                    if geometry.is_empty() {
                        return;
                    }
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();
                    let node_width = geometry.node_width;
                    let node_height = geometry.node_height;

                    let min_coord_x = calc_min(min_x / node_width);
                    let min_coord_y = calc_min(min_y / node_height);

                    let max_coord_x = calc_max(max_x / node_width).min(grid_width - 1);
                    let max_coord_y = calc_max(max_y / node_height).min(grid_height - 1);

                    // Scan the region obtained
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
                                // This node is fully contained, so the intersection checks can be skipped
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
            .rev()
            .for_each(|(grid, geometry)| unsafe {
                if !geometry.is_empty() {
                    let grid_width = grid.cols();
                    let grid_height = grid.rows();

                    if bounds.min_y < center_y && y != 0 {
                        if bounds.min_x < center_x && x != 0 {
                            let node = grid.get_unchecked(y - 1, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y - 1, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x > center_x && x + 1 != grid_width {
                            let node = grid.get_unchecked(y - 1, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }

                    {
                        if bounds.min_x < center_x && x != 0 {
                            let node = grid.get_unchecked(y, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x > center_x && x + 1 != grid_width {
                            let node = grid.get_unchecked(y, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }

                    if bounds.max_y > center_y && y + 1 != grid_height {
                        if bounds.min_x < center_x && x != 0 {
                            let node = grid.get_unchecked(y + 1, x - 1);
                            node.search_items(bounds, callback);
                        }

                        {
                            let node = grid.get_unchecked(y + 1, x);
                            node.search_items(bounds, callback);
                        }

                        if bounds.max_x > center_x && x + 1 != grid_width {
                            let node = grid.get_unchecked(y + 1, x + 1);
                            node.search_items(bounds, callback);
                        }
                    }
                }
                // Advance the ancestor path even when this layer has no objects.
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

pub struct UpSearchQuadTreeNode<T: Copy> {
    items: Vec<Entry<T>>,
}

impl<T: Copy> UpSearchQuadTreeNode<T> {
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

unsafe impl<T: Copy, const MAX_LEVEL: u8> Send for UpSearchQuadTree<T, MAX_LEVEL> {}
unsafe impl<T: Copy, const MAX_LEVEL: u8> Sync for UpSearchQuadTree<T, MAX_LEVEL> {}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::ops::Range;

    const MAX_LEVEL: u8 = 7;
    type Tree = UpSearchQuadTree<usize, MAX_LEVEL>;

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

    /// Move `value` a few floats up or down so it lands on either side of a boundary.
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
    /// level, where `position` and the cached geometry are most likely to disagree.
    fn random_bounds(rng: &mut StdRng, tree: &Tree) -> Rectangle {
        let root = tree.get_bounds();
        let world = &tree.world_bounds;
        let geometry = &tree.geometry[rng.gen_range(1..=MAX_LEVEL as usize)];

        // The node size is the size limit of the level, half of it that of the
        // next finer level.
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
    /// time, but the cached limits claim to be exact.
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

                // Neighbouring cells on the same level must be rejected.
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

    /// Moving and resizing items through `update` must keep every slot and
    /// per-level object count consistent with `position`, also after removals
    /// and the slot reuse that follows them.
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
            for _ in 0..20_000 {
                let item = rng.gen_range(0..bounds_of.len());
                let bounds = if rng.gen_bool(0.5) {
                    // A small move or resize, which usually stays in the same node.
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
}
