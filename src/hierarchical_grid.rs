//! Hierarchical grid: one loose grid per level, scanned without the ancestor walk.
//!
//! Level `L` has `2^(L - 1)` cells per axis over a square region covering the
//! world. An object is stored on the level whose cells fit it, in the cell
//! containing its center, exactly as in `HierarchicalLooseGrid::position`.
//! Only the search differs: every level is scanned over the query's cell range
//! padded by half a cell, without the 3x3 ancestor walk. The query is
//! converted to integer half-cell coordinates once, on the finest level, and
//! shifted for coarser levels; cells whose loose envelope lies inside the
//! query report their objects without intersection tests.

use crate::cell_order::for_each_cell_hierarchically;
use crate::handle::{Entry, GridLocation, ItemHandle, Slots};
use crate::layer_geometry::{LayerGeometry, size_level};
use crate::rect::Rectangle;

/// `(level, x, y)`
type Coord = (usize, usize, usize);

/// Cells of one level, in row-major order. The level's size limits and cell
/// edges are stored in `HierarchicalGrid::geometry`.
struct Layer<T> {
    cells_per_axis: usize,
    /// Reciprocal of the cell size. Positioning multiplies by it, which is
    /// cheaper than dividing; the geometry builds its cell edges with the same
    /// arithmetic, and the search derives its coordinates from the finest
    /// level's reciprocal by exact power-of-two scaling.
    inverse_cell_size: f64,
    cells: Vec<Vec<Entry<T>>>,
}

impl<T> Layer<T> {
    #[inline]
    fn cell_mut(&mut self, x: usize, y: usize) -> &mut Vec<Entry<T>> {
        &mut self.cells[y * self.cells_per_axis + x]
    }
}

pub struct HierarchicalGrid<T: Copy, const MAX_LEVEL: u8> {
    world_bounds: Rectangle,
    /// Origin of the square region of side `side` that covers the world.
    origin_x: f64,
    origin_y: f64,
    side: f64,
    /// Indexed by level; level zero is unused.
    layers: Vec<Layer<T>>,
    /// Indexed by level, parallel to `layers`.
    geometry: Vec<LayerGeometry>,
    /// Location of the entry of each live handle.
    slots: Slots<GridLocation>,
}

impl<T: Copy, const MAX_LEVEL: u8> HierarchicalGrid<T, MAX_LEVEL> {
    pub fn new(world_bounds: Rectangle) -> Self {
        assert!(MAX_LEVEL > 0, "`MAX_LEVEL` cannot be zero.");
        let finest = 1usize << (MAX_LEVEL - 1);
        assert!(
            finest.checked_mul(finest).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let side = world_bounds.get_width().max(world_bounds.get_height());
        let (origin_x, origin_y) = world_bounds.get_min();
        // The geometry needs only the size of the region. A rectangle at the
        // origin keeps both sides exactly `side`; the world's own corners may
        // not subtract back to it.
        let region = Rectangle::new(0.0, 0.0, side, side);

        let mut layers = Vec::with_capacity(MAX_LEVEL as usize + 1);
        let mut geometry = Vec::with_capacity(MAX_LEVEL as usize + 1);
        // Level zero is unused.
        layers.push(Layer {
            cells_per_axis: 0,
            inverse_cell_size: 0.0,
            cells: Vec::new(),
        });
        geometry.push(LayerGeometry::default());
        for level in 1..=MAX_LEVEL as usize {
            let cells_per_axis = 1 << (level - 1);
            let (cell_size, _) = LayerGeometry::node_size(&region, cells_per_axis, cells_per_axis);
            let inverse_cell_size = 1.0 / cell_size;
            layers.push(Layer {
                cells_per_axis,
                inverse_cell_size,
                cells: (0..cells_per_axis * cells_per_axis)
                    .map(|_| Vec::new())
                    .collect(),
            });
            geometry.push(LayerGeometry::with_reciprocal(
                &region,
                level,
                cells_per_axis,
                cells_per_axis,
                inverse_cell_size,
                inverse_cell_size,
            ));
        }

        Self {
            world_bounds,
            origin_x,
            origin_y,
            side,
            layers,
            geometry,
            slots: Slots::new(),
        }
    }

    /// Level and cell that store an object with these bounds. Objects centered
    /// outside the world go to the single cell of level 1.
    pub(crate) fn position(&self, bounds: &Rectangle) -> Coord {
        let center = bounds.get_center();
        if !self.world_bounds.contains_point(center) {
            return (1, 0, 0);
        }
        let level = self.level_for_size(bounds.get_width(), bounds.get_height());
        let (x, y) = self.cell_containing(level, center);
        (level, x, y)
    }

    /// Level whose cells fit an object of this size: the finest level `L` such
    /// that the object fits across the region at least `2^(L - 1)` times,
    /// clamped to `1..=MAX_LEVEL`, as in `HierarchicalLooseGrid`. The region
    /// and its cells are square, so the longer side alone decides.
    fn level_for_size(&self, width: f64, height: f64) -> usize {
        // The bit length of `(side / size) as usize`, from the exponents
        // rather than by dividing.
        size_level(self.side, width.max(height)).clamp(1, MAX_LEVEL as i64) as usize
    }

    /// Cell of `level` whose area contains `point`, clamped into the grid.
    fn cell_containing(&self, level: usize, point: (f64, f64)) -> (usize, usize) {
        let layer = &self.layers[level];
        let last = layer.cells_per_axis - 1;
        let x = (((point.0 - self.origin_x) * layer.inverse_cell_size) as usize).min(last);
        let y = (((point.1 - self.origin_y) * layer.inverse_cell_size) as usize).min(last);
        (x, y)
    }

    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        let coord @ (level, x, y) = self.position(&bounds);
        let cell = self.layers[level].cell_mut(x, y);
        let handle = self.slots.allocate(GridLocation::new(coord, cell.len()));
        cell.push(Entry {
            bounds,
            item,
            handle,
        });
        self.geometry[level].object_count += 1;
        handle
    }

    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let (level, x, y) = location.coord();
        let entry = self.take_entry(level, x, y, location.index as usize);
        debug_assert_eq!(entry.handle, handle);
        self.slots.free(handle);
        self.geometry[level].object_count -= 1;
    }

    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let location = self.slots.get(handle);
        let prev_coord @ (level, x, y) = location.coord();
        let i = location.index as usize;
        if self.stays_in_cell(&bounds, prev_coord) {
            self.layers[level].cell_mut(x, y)[i].bounds = bounds;
            return;
        }

        let curt_coord = self.position(&bounds);
        if curt_coord == prev_coord {
            self.layers[level].cell_mut(x, y)[i].bounds = bounds;
            return;
        }

        let (nl, nx, ny) = curt_coord;
        let entry = self.take_entry(level, x, y, i);
        let new_cell = self.layers[nl].cell_mut(nx, ny);
        *self.slots.get_mut(handle) = GridLocation::new(curt_coord, new_cell.len());
        new_cell.push(Entry { bounds, ..entry });
        self.geometry[nl].object_count += 1;
        self.geometry[level].object_count -= 1;
    }

    /// Take an entry out of its cell. The entry moved into the gap has its
    /// slot updated, so its handle stays valid without a lookup.
    fn take_entry(&mut self, level: usize, x: usize, y: usize, index: usize) -> Entry<T> {
        let cell = self.layers[level].cell_mut(x, y);
        let entry = cell.swap_remove(index);
        if let Some(moved) = cell.get(index) {
            self.slots.get_mut(moved.handle).index = index as u32;
        }
        entry
    }

    /// Check cheaply whether `position` would still return `coord` for
    /// `bounds`, using the level's cached size limits and cell edges. Returns
    /// false on levels without the shortcut, where the caller falls back to
    /// `position`.
    fn stays_in_cell(&self, bounds: &Rectangle, coord: Coord) -> bool {
        let (level, x, y) = coord;
        let geometry = &self.geometry[level];
        if !geometry.has_update_shortcut() {
            return false;
        }

        let center = bounds.get_center();
        let relative_center = (center.0 - self.origin_x, center.1 - self.origin_y);
        let is_finest_layer = level == MAX_LEVEL as usize;
        geometry.matches_layer(bounds.get_width(), bounds.get_height(), is_finest_layer)
            && geometry.matches_cell(relative_center, (x, y))
            && self.world_bounds.contains_point(center)
    }

    /// Square region of the levels, anchored at the world's min corner.
    pub fn get_bounds(&self) -> Rectangle {
        Rectangle::new(
            self.origin_x,
            self.origin_y,
            self.origin_x + self.side,
            self.origin_y + self.side,
        )
    }

    /// Search with the bounds of every stored item, as [`search`](Self::search)
    /// would item by item, and call `callback` with the item searched for and
    /// each item found, the searched one included. The items are taken cell
    /// by cell, depth first through the levels and in Morton order within
    /// each, so that consecutive searches read nearby cells and the same
    /// ancestors. The walk stops at the finest level that holds items but
    /// visits every cell down to it, so where cells far outnumber items,
    /// searching item by item is faster. On the search bench's distribution,
    /// the `self_query` bench group measured it 13% faster than searching
    /// item by item in random order with 10,000 items, and a one-off probe
    /// outside the repository 40% with 640,000 items at the same
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
            |level| (layers[level].cells_per_axis, layers[level].cells_per_axis),
            |level, x, y| {
                let layer = &layers[level];
                for entry in &layer.cells[y * layer.cells_per_axis + x] {
                    self.search(&entry.bounds, |found| callback(entry.item, found));
                }
            },
        );
    }

    /// Call `callback` for every item whose bounds intersect `bounds`,
    /// finest level first.
    ///
    /// Panics for bounds inverted along x by more than two cells of a level
    /// that holds items, from three cells on unless the clamping to the grid
    /// or an inverted y range leaves no row to scan: the columns of that
    /// level, padded by half a cell on each side, then end two or more
    /// columns before they start.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        // Every comparison with NaN is false, so a query with a NaN corner
        // meets nothing, while for a NaN minimum the scans would report the
        // items of the cells they take as inside it untested, and a NaN
        // maximum, cast to zero, could end a column range before it starts.
        if bounds.has_nan() {
            return;
        }
        // Half-cell coordinates of the query on the finest level. The cells of
        // every coarser level are unions of finest cells with the same origin,
        // so their half-cell coordinates follow by an arithmetic shift. The
        // casts truncate, which saves the C runtime call that `floor` compiles
        // to on x86-64 without SSE4.1 and rounds down except below zero, where
        // the query may extend outside the world. There every range below
        // clamps to cell 0 either way; only a query starting less than half a
        // finest cell beyond half a cell before the origin of a level has the
        // objects of that level's first column or row tested rather than
        // reported untested.
        let inverse_half_cell = self.layers[MAX_LEVEL as usize].inverse_cell_size * 2.0;
        let half = [
            ((bounds.min_x - self.origin_x) * inverse_half_cell) as i64,
            ((bounds.min_y - self.origin_y) * inverse_half_cell) as i64,
            ((bounds.max_x - self.origin_x) * inverse_half_cell) as i64,
            ((bounds.max_y - self.origin_y) * inverse_half_cell) as i64,
        ];

        for level in (2..=MAX_LEVEL as usize).rev() {
            if self.geometry[level].is_empty() {
                continue;
            }
            let layer = &self.layers[level];

            let shift = MAX_LEVEL as usize - level;
            let [min_x, min_y, max_x, max_y] = half.map(|coordinate| coordinate >> shift);
            let last = layer.cells_per_axis as i64 - 1;

            // Objects on this level are at most one cell large and centered in
            // their cell, so they extend at most half a cell (the fixed reach)
            // beyond it: pad the query by one half-cell on each side.
            let first_cx = (min_x.saturating_sub(1) >> 1).clamp(0, last) as usize;
            let first_cy = (min_y.saturating_sub(1) >> 1).clamp(0, last) as usize;
            let last_cx = (max_x.saturating_add(1) >> 1).clamp(0, last) as usize;
            let last_cy = (max_y.saturating_add(1) >> 1).clamp(0, last) as usize;

            // A cell whose loose envelope `[c - 0.5, c + 1.5]` (in cells) lies
            // inside the query reports its objects without testing them. The
            // thresholds are conservative by up to one cell on the far side.
            let inner_min_x = min_x.saturating_add(1) >> 1;
            let inner_min_y = min_y.saturating_add(1) >> 1;
            let inner_max_x = max_x.saturating_sub(3) >> 1;
            let inner_max_y = max_y.saturating_sub(3) >> 1;

            for cy in first_cy..=last_cy {
                let row_start = cy * layer.cells_per_axis;
                let row = &layer.cells[row_start + first_cx..=row_start + last_cx];
                let inner_row = (cy as i64) > inner_min_y && (cy as i64) < inner_max_y;
                for (cx, cell) in (first_cx..=last_cx).zip(row) {
                    if inner_row && (cx as i64) > inner_min_x && (cx as i64) < inner_max_x {
                        for entry in cell {
                            callback(entry.item);
                        }
                    } else {
                        for entry in cell {
                            if entry.bounds.intersects(bounds) {
                                callback(entry.item);
                            }
                        }
                    }
                }
            }
        }

        // The root cell also holds objects centered outside the world or
        // larger than it, whose extent is unbounded: always test them.
        if !self.geometry[1].is_empty() {
            for entry in &self.layers[1].cells[0] {
                if entry.bounds.intersects(bounds) {
                    callback(entry.item);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;
    use std::ops::Range;

    const MAX_LEVEL: u8 = 7;
    type Grid = HierarchicalGrid<usize, MAX_LEVEL>;

    const WORLDS: [Rectangle; 3] = [
        Rectangle::new(-128.0, -128.0, 896.0, 384.0),
        Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
        Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    ];

    fn grids() -> Vec<Grid> {
        WORLDS
            .iter()
            .map(|world| Grid::new(world.clone()))
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

    /// Jitter one of `candidates`, or draw a uniform value from `fallback`.
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
    fn random_bounds(rng: &mut StdRng, grid: &Grid) -> Rectangle {
        let world = &grid.world_bounds;
        let geometry = &grid.geometry[rng.gen_range(1..=MAX_LEVEL as usize)];

        // The cell size is the level's size limit; half of it is that of the
        // next finer level.
        let size_limits = if geometry.has_update_shortcut() {
            vec![geometry.node_width, geometry.half_node_width]
        } else {
            Vec::new()
        };
        let width = near_one_of(rng, &size_limits, 0.0..grid.side * 1.1);
        let height = near_one_of(rng, &size_limits, 0.0..grid.side * 1.1);

        // Only the inner edges are finite; the outer ones are +-infinity.
        let inner_edges = |edges: &[f64], origin: f64| -> Vec<f64> {
            edges
                .get(1..edges.len().saturating_sub(1))
                .unwrap_or(&[])
                .iter()
                .map(|edge| origin + edge)
                .collect()
        };
        let x_edges = inner_edges(&geometry.cell_x_edges, grid.origin_x);
        let y_edges = inner_edges(&geometry.cell_y_edges, grid.origin_y);
        let pad_x = world.get_width() * 0.05;
        let pad_y = world.get_height() * 0.05;
        let center_x = near_one_of(rng, &x_edges, world.min_x - pad_x..world.max_x + pad_x);
        let center_y = near_one_of(rng, &y_edges, world.min_y - pad_y..world.max_y + pad_y);

        Rectangle::center_rect(center_x, center_y, width, height)
    }

    /// The update shortcut must agree exactly with `position`, although the
    /// cell edges follow this grid's reciprocal arithmetic. A false positive
    /// would leave an item in the wrong cell; a false negative would only cost
    /// time, but the cached limits are meant to be exact.
    #[test]
    fn update_shortcut_agrees_with_position() {
        let mut rng = StdRng::seed_from_u64(0);
        for grid in grids() {
            for _ in 0..100_000 {
                let bounds = random_bounds(&mut rng, &grid);
                let coord @ (level, x, y) = grid.position(&bounds);

                // Degenerate sizes intentionally bypass the shortcut.
                let is_degenerate = bounds.get_width() <= 0.0 || bounds.get_height() <= 0.0;
                if grid.geometry[level].has_update_shortcut() && !is_degenerate {
                    assert!(
                        grid.stays_in_cell(&bounds, coord),
                        "false negative for {bounds:?} at {coord:?}"
                    );
                }

                // Neighboring cells on the same level must be rejected.
                let cells_per_axis = grid.layers[level].cells_per_axis;
                for dy in -1..=1isize {
                    for dx in -1..=1isize {
                        let (nx, ny) = (x.wrapping_add_signed(dx), y.wrapping_add_signed(dy));
                        if (dx, dy) != (0, 0) && nx < cells_per_axis && ny < cells_per_axis {
                            assert!(
                                !grid.stays_in_cell(&bounds, (level, nx, ny)),
                                "false positive for {bounds:?} at ({level}, {nx}, {ny})"
                            );
                        }
                    }
                }

                // The containing cell on an adjacent level must be rejected too.
                for other_level in [level - 1, level + 1] {
                    if (1..=MAX_LEVEL as usize).contains(&other_level) {
                        let (ox, oy) = grid.cell_containing(other_level, bounds.get_center());
                        assert!(
                            !grid.stays_in_cell(&bounds, (other_level, ox, oy)),
                            "false positive for {bounds:?} at ({other_level}, {ox}, {oy})"
                        );
                    }
                }
            }
        }
    }

    /// The search derives every level's cell range from the finest level's
    /// half-cell coordinates by shifting. For any point, the shift must yield
    /// the cell that `position` assigns, including at cell edges.
    #[test]
    fn shifted_half_cells_agree_with_position() {
        let mut rng = StdRng::seed_from_u64(2);
        for grid in grids() {
            let inverse_half_cell = grid.layers[MAX_LEVEL as usize].inverse_cell_size * 2.0;
            for _ in 0..100_000 {
                let bounds = random_bounds(&mut rng, &grid);
                let (point_x, point_y) = bounds.get_center();
                let half_x = ((point_x - grid.origin_x) * inverse_half_cell) as i64;
                let half_y = ((point_y - grid.origin_y) * inverse_half_cell) as i64;
                for level in 2..=MAX_LEVEL as usize {
                    let last = grid.layers[level].cells_per_axis as i64 - 1;
                    let shift = MAX_LEVEL as usize - level;
                    let expected = (
                        ((half_x >> shift) >> 1).clamp(0, last) as usize,
                        ((half_y >> shift) >> 1).clamp(0, last) as usize,
                    );
                    assert_eq!(
                        grid.cell_containing(level, (point_x, point_y)),
                        expected,
                        "cell of ({point_x}, {point_y}) on level {level}"
                    );
                }
            }
        }
    }

    /// Moves and resizes through `update` must keep every slot and per-level
    /// object count consistent with `position`, also after removals and the
    /// subsequent slot reuse.
    #[test]
    fn update_keeps_slots_and_counts_consistent() {
        let mut rng = StdRng::seed_from_u64(1);
        for mut grid in grids() {
            let mut bounds_of: Vec<Rectangle> =
                (0..500).map(|_| random_bounds(&mut rng, &grid)).collect();
            let mut handles: Vec<ItemHandle> = bounds_of
                .iter()
                .enumerate()
                .map(|(item, bounds)| grid.insert(bounds.clone(), item))
                .collect();

            let finest_cell_size = grid.geometry[MAX_LEVEL as usize].node_width;
            for _ in 0..20_000 {
                let item = rng.gen_range(0..bounds_of.len());
                let bounds = if rng.gen_bool(0.5) {
                    // A small move or resize, which usually stays in the same cell.
                    let old = &bounds_of[item];
                    let (center_x, center_y) = old.get_center();
                    let step = finest_cell_size * 0.1;
                    Rectangle::center_rect(
                        center_x + rng.gen_range(-step..step),
                        center_y + rng.gen_range(-step..step),
                        old.get_width() * rng.gen_range(0.9..1.1),
                        old.get_height() * rng.gen_range(0.9..1.1),
                    )
                } else {
                    random_bounds(&mut rng, &grid)
                };
                grid.update(bounds.clone(), handles[item]);
                bounds_of[item] = bounds;
            }
            assert_consistent(&grid, &handles, &bounds_of);

            for handle in handles.drain(250..) {
                grid.remove(handle);
            }
            bounds_of.truncate(250);
            assert_consistent(&grid, &handles, &bounds_of);

            // New items reuse the freed slots.
            for item in 250..400 {
                let bounds = random_bounds(&mut rng, &grid);
                handles.push(grid.insert(bounds.clone(), item));
                bounds_of.push(bounds);
            }
            assert_eq!(grid.slots.slot_count(), 500);
            assert_consistent(&grid, &handles, &bounds_of);
        }
    }

    /// A removed handle must be rejected even after its slot has been reused.
    #[test]
    #[should_panic(expected = "has been removed")]
    fn stale_handle_is_rejected() {
        let mut grid = Grid::new(WORLDS[0].clone());
        let bounds = Rectangle::center_rect(100.0, 100.0, 10.0, 10.0);
        let stale = grid.insert(bounds.clone(), 0);
        grid.remove(stale);
        let reused = grid.insert(bounds.clone(), 1);
        assert_ne!(reused, stale);
        grid.update(bounds, stale);
    }

    fn assert_consistent(grid: &Grid, handles: &[ItemHandle], bounds_of: &[Rectangle]) {
        assert_eq!(grid.slots.len(), handles.len());
        for (item, (handle, bounds)) in handles.iter().zip(bounds_of).enumerate() {
            let location = grid.slots.get(*handle);
            assert_eq!(
                location.coord(),
                grid.position(bounds),
                "location of {item}"
            );
            let (level, x, y) = location.coord();
            let layer = &grid.layers[level];
            let entry = &layer.cells[y * layer.cells_per_axis + x][location.index as usize];
            assert_eq!(entry.item, item);
            assert_eq!(entry.handle, *handle);
            assert_eq!(entry.bounds, *bounds);
        }
        for (level, (layer, geometry)) in grid.layers.iter().zip(&grid.geometry).enumerate() {
            let stored: usize = layer.cells.iter().map(Vec::len).sum();
            assert_eq!(
                stored, geometry.object_count,
                "object count of level {level}"
            );
        }
    }
}
