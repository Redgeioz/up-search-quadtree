use crate::{ball::*, rect::*, *};

use rand::prelude::*;
use std::collections::HashSet;
use std::f64::consts::PI;

const MAX_LEVEL: u8 = 7;
const MAX_ITEMS: usize = 8;
/// Cell size of the uniform grid, a few times the usual ball diameter.
const UNIFORM_CELL_SIZE: f64 = 32.0;

const BOUND_WIDTH: f64 = 1024.0;
const BOUND_HEIGHT: f64 = 512.0;
const BALL_COUNT: usize = 2500;
const MIN_RADIUS: f64 = 1.0;
const MAX_RADIUS: f64 = 5.0;
const MIN_VELOCITY: f64 = 0.2;
const MAX_VELOCITY: f64 = 1.0;

/// Steps of the dynamic check; each moves, resizes, removes and inserts balls.
const STEPS: usize = 8;

fn world() -> Rectangle {
    Rectangle::new(-128.0, -128.0, BOUND_WIDTH - 128.0, BOUND_HEIGHT - 128.0)
}

/// Random radius, doubled up to seven times at random so that balls span many
/// levels.
fn random_radius(rng: &mut impl Rng) -> f64 {
    let mut r = rng.gen_range(MIN_RADIUS..MAX_RADIUS);
    for _ in 0..7 {
        if rng.gen_bool(0.35) {
            r *= 2.0;
        } else {
            break;
        }
    }
    r
}

/// A moving ball at a random point of `bounds`.
fn random_ball(rng: &mut impl Rng, bounds: &Rectangle) -> Ball {
    let (x, y) = bounds.get_random_point(rng);
    let r = random_radius(rng);
    let v = rng.gen_range(MIN_VELOCITY..MAX_VELOCITY);
    let angle = rng.gen_range(0.0..2.0 * PI);
    BallBuilder::new()
        .set_position(x, y)
        .set_radius(r)
        .set_velocity(v * angle.cos(), v * angle.sin())
        .build()
}

/// Every spatial structure of the crate except `StripedGrid`, holding the same
/// balls.
struct Trees {
    loose: LooseQuadTree<usize, MAX_LEVEL, MAX_ITEMS>,
    hierarchical_loose: HierarchicalLooseGrid<usize, MAX_LEVEL>,
    uniform: UniformGrid<usize>,
    hierarchical: HierarchicalGrid<usize, MAX_LEVEL>,
    overflow_bound: OverflowBoundGrid<usize, MAX_LEVEL>,
    lazy: LazyOverflowBoundGrid<usize, MAX_LEVEL>,
    boxed: BoxedOverflowBoundGrid<usize, MAX_LEVEL>,
    striped: StripedOverflowBoundGrid<usize, MAX_LEVEL>,
    incoming: IncomingOverflowBoundGrid<usize, MAX_LEVEL>,
}

/// Handles of one ball in every structure.
#[derive(Clone, Copy)]
struct Handles {
    loose: ItemHandle,
    hierarchical_loose: ItemHandle,
    uniform: ItemHandle,
    hierarchical: ItemHandle,
    overflow_bound: ItemHandle,
    lazy: ItemHandle,
    boxed: ItemHandle,
    striped: ItemHandle,
    incoming: ItemHandle,
}

impl Trees {
    fn new(world: &Rectangle) -> Self {
        Self {
            loose: LooseQuadTree::new(world.clone()),
            hierarchical_loose: HierarchicalLooseGrid::new(world.clone()),
            uniform: UniformGrid::new(world.clone(), UNIFORM_CELL_SIZE),
            hierarchical: HierarchicalGrid::new(world.clone()),
            overflow_bound: OverflowBoundGrid::new(world.clone()),
            lazy: LazyOverflowBoundGrid::new(world.clone()),
            boxed: BoxedOverflowBoundGrid::new(world.clone()),
            striped: StripedOverflowBoundGrid::new(world.clone()),
            incoming: IncomingOverflowBoundGrid::new(world.clone()),
        }
    }

    fn insert(&mut self, bounds: &Rectangle, id: usize) -> Handles {
        Handles {
            loose: self.loose.insert(bounds.clone(), id),
            hierarchical_loose: self.hierarchical_loose.insert(bounds.clone(), id),
            uniform: self.uniform.insert(bounds.clone(), id),
            hierarchical: self.hierarchical.insert(bounds.clone(), id),
            overflow_bound: self.overflow_bound.insert(bounds.clone(), id),
            lazy: self.lazy.insert(bounds.clone(), id),
            boxed: self.boxed.insert(bounds.clone(), id),
            striped: self.striped.insert(bounds.clone(), id),
            incoming: self.incoming.insert(bounds.clone(), id),
        }
    }

    fn update(&mut self, bounds: &Rectangle, handles: Handles) {
        self.loose.update(bounds.clone(), handles.loose);
        self.hierarchical_loose.update(bounds.clone(), handles.hierarchical_loose);
        self.uniform.update(bounds.clone(), handles.uniform);
        self.hierarchical.update(bounds.clone(), handles.hierarchical);
        self.overflow_bound
            .update(bounds.clone(), handles.overflow_bound);
        self.lazy.update(bounds.clone(), handles.lazy);
        self.boxed.update(bounds.clone(), handles.boxed);
        self.striped
            .update(bounds.clone(), handles.striped);
        self.incoming
            .update(bounds.clone(), handles.incoming);
    }

    fn remove(&mut self, handles: Handles) {
        self.loose.remove(handles.loose);
        self.hierarchical_loose.remove(handles.hierarchical_loose);
        self.uniform.remove(handles.uniform);
        self.hierarchical.remove(handles.hierarchical);
        self.overflow_bound.remove(handles.overflow_bound);
        self.lazy.remove(handles.lazy);
        self.boxed.remove(handles.boxed);
        self.striped.remove(handles.striped);
        self.incoming.remove(handles.incoming);
    }

    /// Tighten the bounds that shrink only on request.
    fn refresh(&mut self) {
        self.lazy.refresh_bounds();
        self.boxed.refresh_bounds();
        self.striped.refresh_bounds();
        self.incoming.refresh_bounds();
    }

    /// Items that each structure's search finds for `bounds`, excluding `id`.
    fn search(&self, bounds: &Rectangle, id: usize) -> Vec<(&'static str, HashSet<usize>)> {
        vec![
            (
                "LooseQuadTree",
                collect(id, |cb| self.loose.search(bounds, cb)),
            ),
            (
                "HierarchicalLooseGrid",
                collect(id, |cb| self.hierarchical_loose.search(bounds, cb)),
            ),
            (
                "UniformGrid",
                collect(id, |cb| self.uniform.search(bounds, cb)),
            ),
            (
                "HierarchicalGrid",
                collect(id, |cb| self.hierarchical.search(bounds, cb)),
            ),
            (
                "OverflowBoundGrid",
                collect(id, |cb| self.overflow_bound.search(bounds, cb)),
            ),
            (
                "LazyOverflowBoundGrid",
                collect(id, |cb| self.lazy.search(bounds, cb)),
            ),
            (
                "BoxedOverflowBoundGrid",
                collect(id, |cb| self.boxed.search(bounds, cb)),
            ),
            (
                "StripedOverflowBoundGrid",
                collect(id, |cb| self.striped.search(bounds, cb)),
            ),
            (
                "IncomingOverflowBoundGrid",
                collect(id, |cb| self.incoming.search(bounds, cb)),
            ),
        ]
    }
}

/// Run `search` and collect every item it reports other than `id`.
fn collect(id: usize, search: impl Fn(&mut dyn FnMut(usize))) -> HashSet<usize> {
    let mut found = HashSet::new();
    search(&mut |other: usize| {
        if other != id {
            found.insert(other);
        }
    });
    found
}

/// Every structure's search must agree with a brute-force scan of the live
/// balls.
fn assert_searches_match_brute_force(trees: &Trees, balls: &[Option<(Ball, Handles)>]) {
    let live: Vec<(usize, Rectangle)> = balls
        .iter()
        .enumerate()
        .filter_map(|(id, ball)| ball.as_ref().map(|(ball, _)| (id, ball.get_bounds())))
        .collect();

    for (id, bounds) in &live {
        let expected: HashSet<usize> = live
            .iter()
            .filter(|(other, other_bounds)| other != id && other_bounds.intersects(bounds))
            .map(|(other, _)| *other)
            .collect();
        for (name, found) in trees.search(bounds, *id) {
            assert_eq!(
                found, expected,
                "{name} disagrees with brute force for ball {id}"
            );
        }
    }
}

#[test]
fn check_quadtrees() {
    for _ in 0..25 {
        single_check();
    }
}

fn single_check() {
    let mut rng = thread_rng();
    let world = world();
    let mut trees = Trees::new(&world);

    let balls: Vec<Option<(Ball, Handles)>> = (0..BALL_COUNT)
        .map(|id| {
            let ball = random_ball(&mut rng, &world);
            let handles = trees.insert(&ball.get_bounds(), id);
            Some((ball, handles))
        })
        .collect();

    assert_searches_match_brute_force(&trees, &balls);
}

/// After moves, resizes, removals and reinsertions through handles, every
/// structure must answer searches exactly like a brute-force scan.
#[test]
fn check_updates_and_removals() {
    let mut rng = StdRng::seed_from_u64(7);
    let world = world();
    let mut trees = Trees::new(&world);

    // `None` marks a removed ball; the structures may reuse its slot for a
    // later insert.
    let mut balls: Vec<Option<(Ball, Handles)>> = Vec::new();
    for _ in 0..BALL_COUNT {
        let ball = random_ball(&mut rng, &world);
        let handles = trees.insert(&ball.get_bounds(), balls.len());
        balls.push(Some((ball, handles)));
    }

    for step in 0..STEPS {
        // Move every ball; resizing some changes their level or node.
        for (ball, handles) in balls.iter_mut().flatten() {
            ball.update(&world);
            if rng.gen_bool(0.05) {
                ball.r = random_radius(&mut rng);
            }
            trees.update(&ball.get_bounds(), *handles);
        }

        // Remove some balls and insert new ones, which reuse the freed slots.
        for _ in 0..BALL_COUNT / 20 {
            let id = rng.gen_range(0..balls.len());
            if let Some((_, handles)) = balls[id].take() {
                trees.remove(handles);
            }
        }
        for _ in 0..BALL_COUNT / 40 {
            let ball = random_ball(&mut rng, &world);
            let handles = trees.insert(&ball.get_bounds(), balls.len());
            balls.push(Some((ball, handles)));
        }

        // Check the lazy bounds both stale and right after a refresh.
        if step % 2 == 1 {
            trees.refresh();
        }
        assert_searches_match_brute_force(&trees, &balls);
    }
}

/// Structures whose search narrows the fixed reach of each level to a bound on
/// how far that level's items extend beyond their cells, or skips cells by the
/// bounding box of their items.
trait OverflowBounded {
    fn create(world: Rectangle, fit: bool) -> Self;
    fn root(&self) -> Rectangle;
    fn insert_item(&mut self, bounds: Rectangle, id: usize) -> ItemHandle;
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle);
    fn remove_item(&mut self, handle: ItemHandle);
    fn search_items(&self, bounds: &Rectangle, callback: &mut dyn FnMut(usize));
    fn refresh(&mut self) {}
}

impl OverflowBounded for OverflowBoundGrid<usize, MAX_LEVEL> {
    fn create(world: Rectangle, fit: bool) -> Self {
        if fit {
            Self::new_fit(world)
        } else {
            Self::new(world)
        }
    }
    fn root(&self) -> Rectangle {
        self.get_bounds().clone()
    }
    fn insert_item(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
        self.insert(bounds, id)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn search_items(&self, bounds: &Rectangle, callback: &mut dyn FnMut(usize)) {
        self.search(bounds, callback);
    }
}

impl OverflowBounded for LazyOverflowBoundGrid<usize, MAX_LEVEL> {
    fn create(world: Rectangle, fit: bool) -> Self {
        if fit {
            Self::new_fit(world)
        } else {
            Self::new(world)
        }
    }
    fn root(&self) -> Rectangle {
        self.get_bounds().clone()
    }
    fn insert_item(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
        self.insert(bounds, id)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn search_items(&self, bounds: &Rectangle, callback: &mut dyn FnMut(usize)) {
        self.search(bounds, callback);
    }
    fn refresh(&mut self) {
        self.refresh_bounds();
    }
}

impl OverflowBounded for BoxedOverflowBoundGrid<usize, MAX_LEVEL> {
    fn create(world: Rectangle, fit: bool) -> Self {
        if fit {
            Self::new_fit(world)
        } else {
            Self::new(world)
        }
    }
    fn root(&self) -> Rectangle {
        self.get_bounds().clone()
    }
    fn insert_item(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
        self.insert(bounds, id)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn search_items(&self, bounds: &Rectangle, callback: &mut dyn FnMut(usize)) {
        self.search(bounds, callback);
    }
    fn refresh(&mut self) {
        self.refresh_bounds();
    }
}


impl OverflowBounded for StripedOverflowBoundGrid<usize, MAX_LEVEL> {
    fn create(world: Rectangle, fit: bool) -> Self {
        if fit {
            Self::new_fit(world)
        } else {
            Self::new(world)
        }
    }
    fn root(&self) -> Rectangle {
        self.get_bounds().clone()
    }
    fn insert_item(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
        self.insert(bounds, id)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn search_items(&self, bounds: &Rectangle, callback: &mut dyn FnMut(usize)) {
        self.search(bounds, callback);
    }
    fn refresh(&mut self) {
        self.refresh_bounds();
    }
}

impl OverflowBounded for IncomingOverflowBoundGrid<usize, MAX_LEVEL> {
    fn create(world: Rectangle, fit: bool) -> Self {
        if fit {
            Self::new_fit(world)
        } else {
            Self::new(world)
        }
    }
    fn root(&self) -> Rectangle {
        self.get_bounds().clone()
    }
    fn insert_item(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
        self.insert(bounds, id)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn search_items(&self, bounds: &Rectangle, callback: &mut dyn FnMut(usize)) {
        self.search(bounds, callback);
    }
    fn refresh(&mut self) {
        self.refresh_bounds();
    }
}

/// Worlds with square and non-square roots, and with an offset origin.
const TIGHT_WORLDS: [Rectangle; 3] = [
    Rectangle::new(-128.0, -128.0, 896.0, 384.0),
    Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
    Rectangle::new(0.3, -0.7, 1000.1, 333.3),
];

/// Move `value` a few floats up or down, onto either side of an edge.
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

/// Bounds of an item on a random level below the root. Unless `saturate`
/// hits, the item stays well below its level's size limit, so that level's
/// bound shrinks below the fixed reach. Centers often lie on a cell edge,
/// where an item reaches furthest into the neighboring cell.
fn tight_item(rng: &mut StdRng, root: &Rectangle, world: &Rectangle, saturate: f64) -> Rectangle {
    let level = rng.gen_range(2..=MAX_LEVEL as u32);
    let cells = 1u32 << (level - 1);
    let node_width = root.get_width() / cells as f64;
    let node_height = root.get_height() / cells as f64;
    let (low, high) = if rng.gen_bool(saturate) {
        (0.95, 1.0)
    } else if level == MAX_LEVEL as u32 {
        (0.02, 0.4)
    } else {
        // More than half a node keeps the item on this level.
        (0.51, 0.75)
    };
    let width = node_width * rng.gen_range(low..high);
    let height = node_height * rng.gen_range(low..high);

    let mut coordinate = |min: f64, node: f64, range: std::ops::Range<f64>| {
        if rng.gen_bool(0.5) {
            let edge = min + node * rng.gen_range(1..cells) as f64;
            jitter(rng, edge)
        } else {
            rng.gen_range(range)
        }
    };
    let x = coordinate(root.min_x, node_width, world.min_x..world.max_x);
    let y = coordinate(root.min_y, node_height, world.min_y..world.max_y);
    Rectangle::center_rect(x, y, width, height)
}

/// The bounds of every live item, queries of every size, and queries that
/// overlap one live item by a single float, where a bound that is too small
/// would lose it.
fn tight_queries(
    rng: &mut StdRng,
    root: &Rectangle,
    live: &[(usize, Rectangle)],
) -> Vec<Rectangle> {
    let mut queries: Vec<Rectangle> = live.iter().map(|(_, bounds)| bounds.clone()).collect();
    for _ in 0..300 {
        let scale = 2f64.powf(-rng.gen_range(0.0..MAX_LEVEL as f64 + 1.0));
        let (x, y) = root.get_random_point(rng);
        queries.push(Rectangle::center_rect(
            x,
            y,
            root.get_width() * scale * rng.gen_range(0.2..1.0),
            root.get_height() * scale * rng.gen_range(0.2..1.0),
        ));
    }
    for _ in 0..300 {
        let (_, b) = &live[rng.gen_range(0..live.len())];
        let width = b.get_width() * rng.gen_range(0.1..3.0);
        let height = b.get_height() * rng.gen_range(0.1..3.0);
        queries.push(match rng.gen_range(0..4) {
            0 => Rectangle::new(b.max_x.next_down(), b.min_y, b.max_x + width, b.max_y),
            1 => Rectangle::new(b.min_x - width, b.min_y, b.min_x.next_up(), b.max_y),
            2 => Rectangle::new(b.min_x, b.max_y.next_down(), b.max_x, b.max_y + height),
            _ => Rectangle::new(b.min_x, b.min_y - height, b.max_x, b.min_y.next_up()),
        });
    }
    queries
}

fn assert_tight_search_matches_brute_force(
    tree: &impl OverflowBounded,
    items: &[Option<(ItemHandle, Rectangle)>],
    rng: &mut StdRng,
    context: &str,
) {
    let live: Vec<(usize, Rectangle)> = items
        .iter()
        .enumerate()
        .filter_map(|(id, item)| item.as_ref().map(|(_, bounds)| (id, bounds.clone())))
        .collect();
    for query in tight_queries(rng, &tree.root(), &live) {
        let expected: HashSet<usize> = live
            .iter()
            .filter(|(_, bounds)| bounds.intersects(&query))
            .map(|(id, _)| *id)
            .collect();
        let mut found = HashSet::new();
        tree.search_items(&query, &mut |id| {
            assert!(
                found.insert(id),
                "{context}: item {id} reported twice for {query:?}"
            );
        });
        assert_eq!(found, expected, "{context}: query {query:?}");
    }
}

fn check_tight_bounds<Tree: OverflowBounded>(name: &str) {
    let mut rng = StdRng::seed_from_u64(11);
    for world in TIGHT_WORLDS {
        for fit in [false, true] {
            for saturate in [0.0, 0.1] {
                let context = format!("{name}, world {world:?}, fit {fit}, saturate {saturate}");
                let mut tree = Tree::create(world.clone(), fit);
                let root = tree.root();
                let mut items: Vec<Option<(ItemHandle, Rectangle)>> = (0..400)
                    .map(|id| {
                        let bounds = tight_item(&mut rng, &root, &world, saturate);
                        Some((tree.insert_item(bounds.clone(), id), bounds))
                    })
                    .collect();
                assert_tight_search_matches_brute_force(&tree, &items, &mut rng, &context);

                for round in 0..3 {
                    // Small moves keep most items in their cells but change their overflow.
                    for (handle, bounds) in items.iter_mut().flatten() {
                        let (x, y) = bounds.get_center();
                        let (width, height) = (bounds.get_width(), bounds.get_height());
                        *bounds = Rectangle::center_rect(
                            x + width * rng.gen_range(-0.3..0.3),
                            y + height * rng.gen_range(-0.3..0.3),
                            width,
                            height,
                        );
                        tree.update_item(bounds.clone(), *handle);
                    }
                    // Move some items to random levels; remove some and insert new ones.
                    for _ in 0..60 {
                        let id = rng.gen_range(0..items.len());
                        if let Some((handle, bounds)) = &mut items[id] {
                            *bounds = tight_item(&mut rng, &root, &world, saturate);
                            tree.update_item(bounds.clone(), *handle);
                        }
                    }
                    for _ in 0..40 {
                        let id = rng.gen_range(0..items.len());
                        if let Some((handle, _)) = items[id].take() {
                            tree.remove_item(handle);
                        }
                    }
                    for _ in 0..30 {
                        let bounds = tight_item(&mut rng, &root, &world, saturate);
                        let handle = tree.insert_item(bounds.clone(), items.len());
                        items.push(Some((handle, bounds)));
                    }

                    let context = format!("{context}, round {round}");
                    assert_tight_search_matches_brute_force(&tree, &items, &mut rng, &context);
                    tree.refresh();
                    assert_tight_search_matches_brute_force(&tree, &items, &mut rng, &context);
                }
            }
        }
    }
}

/// Searches that use bounds below the fixed reach, or cell boxes, must still
/// find every item, including items that overlap a query by a single float.
#[test]
fn check_tight_overflow_bounds() {
    check_tight_bounds::<OverflowBoundGrid<usize, MAX_LEVEL>>("OverflowBoundGrid");
    check_tight_bounds::<LazyOverflowBoundGrid<usize, MAX_LEVEL>>("LazyOverflowBoundGrid");
    check_tight_bounds::<BoxedOverflowBoundGrid<usize, MAX_LEVEL>>("BoxedOverflowBoundGrid");
    check_tight_bounds::<StripedOverflowBoundGrid<usize, MAX_LEVEL>>(
        "StripedOverflowBoundGrid",
    );
    check_tight_bounds::<IncomingOverflowBoundGrid<usize, MAX_LEVEL>>(
        "IncomingOverflowBoundGrid",
    );
}
