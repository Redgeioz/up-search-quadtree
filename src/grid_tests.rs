//! Tests that several grids share, written once. Each macro expands into a
//! grid's own `tests` module, where the tests can read the grid's private
//! fields. That module defines `Tree` and `WORLDS`, and `MAX_LEVEL` for the
//! multi-level grids, and imports `rand::prelude::*`.

/// `coarsest_occupied`, which the tests of every grid use.
macro_rules! coarsest_occupied {
    () => {
        /// Coarsest level below the root that holds items, `MAX_LEVEL + 1` if none.
        fn coarsest_occupied(tree: &Tree) -> usize {
            (2..=MAX_LEVEL as usize)
                .find(|&level| !tree.geometry[level].is_empty())
                .unwrap_or(MAX_LEVEL as usize + 1)
        }
    };
}
pub(crate) use coarsest_occupied;

/// Tests of the grids that keep no bounds to refresh, `HierarchicalLooseGrid`
/// and `ExactOverflowBoundGrid`.
macro_rules! exact_reach_tests {
    () => {
        $crate::grid_tests::coarsest_occupied!();

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
        fn near_one_of(rng: &mut StdRng, candidates: &[f64], fallback: std::ops::Range<f64>) -> f64 {
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

        /// `coarsest_level()` must be the coarsest level below the root that holds
        /// items after every insertion, update and removal. With at most nine
        /// items, levels keep emptying and filling again.
        #[test]
        fn coarsest_level_follows_the_items() {
            let mut rng = StdRng::seed_from_u64(5);
            for mut tree in trees() {
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
                            let bounds = random_bounds(&mut rng, &tree);
                            handles.push(tree.insert(bounds, step));
                        }
                        1 => {
                            let bounds = random_bounds(&mut rng, &tree);
                            tree.update(bounds, handles[rng.gen_range(0..handles.len())]);
                        }
                        _ => tree.remove(handles.swap_remove(rng.gen_range(0..handles.len()))),
                    }
                    assert_eq!(tree.coarsest_level(), coarsest_occupied(&tree), "step {step}");
                    if tree.coarsest_level() > before {
                        raised += 1;
                    }
                }
                // The coarsest level emptied out often enough to exercise clearing its bit.
                assert!(raised > 100, "the coarsest level emptied only {raised} times");
            }
        }
    };
}
pub(crate) use exact_reach_tests;

/// Tests of the grids whose bounds or boxes only shrink on refresh.
macro_rules! refreshed_tests {
    () => {
        $crate::grid_tests::coarsest_occupied!();

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
    };
}
pub(crate) use refreshed_tests;

/// The tests of `refreshed_tests`, and those of the grids whose per-level
/// bounds only shrink on refresh.
macro_rules! lazy_bound_tests {
    () => {
        $crate::grid_tests::refreshed_tests!();

        fn directions(bound: DirectionalOverflow) -> [f64; 4] {
            [bound.left, bound.right, bound.top, bound.bottom]
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
    };
}
pub(crate) use lazy_bound_tests;

/// Helpers for the grids that keep a box per node, whose node type is `$node`.
macro_rules! box_helpers {
    ($node:ident) => {
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
        fn union<T: Copy>(node: &$node<T>) -> Rectangle {
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
                        assert!(b.contains(&union), "{context}: box {b:?} misses {union:?}");
                    }
                }
            }
        }
    };
}
pub(crate) use box_helpers;

/// Tests of the grids whose boxes, of node type `$node`, only shrink on
/// refresh. Uses `random_bounds` of `refreshed_tests`.
macro_rules! lazy_box_tests {
    ($node:ident) => {
        $crate::grid_tests::box_helpers!($node);

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
    };
}
pub(crate) use lazy_box_tests;

/// The tests of `lazy_bound_tests` and `lazy_box_tests`, for the grids that
/// keep a box per node, whose node type is `$node`, besides per-level bounds.
macro_rules! boxed_tests {
    ($node:ident) => {
        $crate::grid_tests::lazy_bound_tests!();
        $crate::grid_tests::lazy_box_tests!($node);

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
    };
}
pub(crate) use boxed_tests;

/// The tests of `boxed_tests`, and those of the grids that also keep the span
/// of every row and column.
macro_rules! span_tests {
    ($node:ident) => {
        $crate::grid_tests::boxed_tests!($node);

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
                        assert!(root.contains(&exact), "{context}: {root:?} misses {exact:?}");
                    }
                    let mut found = Vec::new();
                    tree.search(&query, |id| found.push(id));
                    assert_eq!(found, [0], "{context}");
                }
            }
        }
    };
}
pub(crate) use span_tests;

/// Tests of the single-level grids `SpanGrid` and `HaloGrid`.
macro_rules! single_level_tests {
    () => {
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
            let check = |line: &str, spans: &[[f64; 2]], unions: &[[f64; 2]]| {
                for (i, (span, union)) in spans.iter().zip(unions).enumerate() {
                    if exact {
                        assert_eq!(span, union, "{line} {i}");
                    } else {
                        assert!(span[0] <= union[0] && span[1] >= union[1], "{line} {i}");
                    }
                }
            };
            check("column", &tree.lines.columns, &columns);
            check("row", &tree.lines.rows, &row_spans);
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
    };
}
pub(crate) use single_level_tests;
