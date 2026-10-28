use crate::{ball::*, rect::*, *};

use rand::prelude::*;
use std::collections::HashSet;
use std::f64::consts::PI;

const MAX_LEVEL: u8 = 7;
const MAX_ITEMS: usize = 8;
/// Cell size of the uniform grid, a few times the usual ball diameter.
const UNIFORM_CELL_SIZE: f64 = 32.0;
/// Cell sizes of the `MultiGrid` collections: three log-spaced classes over
/// the ball diameters, the largest balls going to its overflow list.
const MULTI_CELL_SIZES: [f64; 3] = [16.0, 128.0, 1024.0];

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

/// Every spatial structure of the crate except the single-level loose grids
/// `SpanGrid` and `HaloGrid`, which have tests of their own, holding
/// the same balls. Results are compared as sets, so duplicates go unnoticed
/// here; the tight-bounds and self-query checks catch them.
struct Trees {
    loose: LooseQuadTree<usize, MAX_LEVEL, MAX_ITEMS>,
    hierarchical_loose: HierarchicalLooseGrid<usize, MAX_LEVEL>,
    uniform: UniformGrid<usize>,
    two_layer: TwoLayerGrid<usize>,
    multi: MultiGrid<usize>,
    hierarchical: HierarchicalGrid<usize, MAX_LEVEL>,
    overflow_bound: ExactOverflowBoundGrid<usize, MAX_LEVEL>,
    boxed_loose: BoxedLooseGrid<usize, MAX_LEVEL>,
    lazy: LazyOverflowBoundGrid<usize, MAX_LEVEL>,
    span: SpanOverflowBoundGrid<usize, MAX_LEVEL>,
    halo: HaloOverflowBoundGrid<usize, MAX_LEVEL>,
    sweep: SortAndSweep<usize>,
}

/// Handles of one ball in every structure.
#[derive(Clone, Copy)]
struct Handles {
    loose: ItemHandle,
    hierarchical_loose: ItemHandle,
    uniform: ItemHandle,
    two_layer: ItemHandle,
    multi: ItemHandle,
    hierarchical: ItemHandle,
    overflow_bound: ItemHandle,
    boxed_loose: ItemHandle,
    lazy: ItemHandle,
    span: ItemHandle,
    halo: ItemHandle,
    sweep: ItemHandle,
}

impl Trees {
    fn new(world: &Rectangle) -> Self {
        Self {
            loose: LooseQuadTree::new(world.clone()),
            hierarchical_loose: HierarchicalLooseGrid::new(world.clone()),
            uniform: UniformGrid::new(world.clone(), UNIFORM_CELL_SIZE),
            two_layer: TwoLayerGrid::new(world.clone(), UNIFORM_CELL_SIZE),
            multi: MultiGrid::new(world.clone(), &MULTI_CELL_SIZES),
            hierarchical: HierarchicalGrid::new(world.clone()),
            overflow_bound: ExactOverflowBoundGrid::new(world.clone()),
            boxed_loose: BoxedLooseGrid::new(world.clone()),
            lazy: LazyOverflowBoundGrid::new(world.clone()),
            span: SpanOverflowBoundGrid::new(world.clone()),
            halo: HaloOverflowBoundGrid::new(world.clone()),
            sweep: SortAndSweep::new(),
        }
    }

    fn insert(&mut self, bounds: &Rectangle, id: usize) -> Handles {
        Handles {
            loose: self.loose.insert(bounds.clone(), id),
            hierarchical_loose: self.hierarchical_loose.insert(bounds.clone(), id),
            uniform: self.uniform.insert(bounds.clone(), id),
            two_layer: self.two_layer.insert(bounds.clone(), id),
            multi: self.multi.insert(bounds.clone(), id),
            hierarchical: self.hierarchical.insert(bounds.clone(), id),
            overflow_bound: self.overflow_bound.insert(bounds.clone(), id),
            boxed_loose: self.boxed_loose.insert(bounds.clone(), id),
            lazy: self.lazy.insert(bounds.clone(), id),
            span: self.span.insert(bounds.clone(), id),
            halo: self.halo.insert(bounds.clone(), id),
            sweep: self.sweep.insert(bounds.clone(), id),
        }
    }

    fn update(&mut self, bounds: &Rectangle, handles: Handles) {
        self.loose.update(bounds.clone(), handles.loose);
        self.hierarchical_loose.update(bounds.clone(), handles.hierarchical_loose);
        self.uniform.update(bounds.clone(), handles.uniform);
        self.two_layer.update(bounds.clone(), handles.two_layer);
        self.multi.update(bounds.clone(), handles.multi);
        self.hierarchical.update(bounds.clone(), handles.hierarchical);
        self.overflow_bound
            .update(bounds.clone(), handles.overflow_bound);
        self.boxed_loose.update(bounds.clone(), handles.boxed_loose);
        self.lazy.update(bounds.clone(), handles.lazy);
        self.span
            .update(bounds.clone(), handles.span);
        self.halo
            .update(bounds.clone(), handles.halo);
        self.sweep.update(bounds.clone(), handles.sweep);
    }

    fn remove(&mut self, handles: Handles) {
        self.loose.remove(handles.loose);
        self.hierarchical_loose.remove(handles.hierarchical_loose);
        self.uniform.remove(handles.uniform);
        self.two_layer.remove(handles.two_layer);
        self.multi.remove(handles.multi);
        self.hierarchical.remove(handles.hierarchical);
        self.overflow_bound.remove(handles.overflow_bound);
        self.boxed_loose.remove(handles.boxed_loose);
        self.lazy.remove(handles.lazy);
        self.span.remove(handles.span);
        self.halo.remove(handles.halo);
        self.sweep.remove(handles.sweep);
    }

    /// Tighten the bounds that shrink only on request.
    fn refresh(&mut self) {
        self.boxed_loose.refresh_bounds();
        self.lazy.refresh_bounds();
        self.span.refresh_bounds();
        self.halo.refresh_bounds();
        self.sweep.refresh_order();
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
                "TwoLayerGrid",
                collect(id, |cb| self.two_layer.search(bounds, cb)),
            ),
            (
                "MultiGrid",
                collect(id, |cb| self.multi.search(bounds, cb)),
            ),
            (
                "HierarchicalGrid",
                collect(id, |cb| self.hierarchical.search(bounds, cb)),
            ),
            (
                "ExactOverflowBoundGrid",
                collect(id, |cb| self.overflow_bound.search(bounds, cb)),
            ),
            (
                "BoxedLooseGrid",
                collect(id, |cb| self.boxed_loose.search(bounds, cb)),
            ),
            (
                "LazyOverflowBoundGrid",
                collect(id, |cb| self.lazy.search(bounds, cb)),
            ),
            (
                "SpanOverflowBoundGrid",
                collect(id, |cb| self.span.search(bounds, cb)),
            ),
            (
                "HaloOverflowBoundGrid",
                collect(id, |cb| self.halo.search(bounds, cb)),
            ),
            (
                "SortAndSweep",
                collect(id, |cb| self.sweep.search(bounds, cb)),
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

/// Every structure's search with a live ball's own bounds must find that ball.
/// Balls far from all others are lost without a brute-force search noticing.
fn assert_balls_find_themselves(trees: &Trees, balls: &[Option<(Ball, Handles)>]) {
    for (id, (ball, _)) in balls.iter().enumerate().filter_map(|(id, ball)| Some((id, ball.as_ref()?))) {
        for (name, found) in trees.search(&ball.get_bounds(), usize::MAX) {
            assert!(found.contains(&id), "{name} lost ball {id}");
        }
    }
}

/// Items centered outside the world must stay findable through inserts, splits,
/// moves across the world's edge, resizes and removals.
#[test]
fn items_centered_outside_the_world_stay_findable() {
    let mut rng = StdRng::seed_from_u64(11);
    let world = world();
    // Three quarters of this area lie outside the world.
    let around = world.scale(2.0);
    let mut trees = Trees::new(&world);

    let mut balls: Vec<Option<(Ball, Handles)>> = Vec::new();
    for _ in 0..BALL_COUNT / 5 {
        let ball = random_ball(&mut rng, &around);
        let handles = trees.insert(&ball.get_bounds(), balls.len());
        balls.push(Some((ball, handles)));
    }
    assert_searches_match_brute_force(&trees, &balls);
    assert_balls_find_themselves(&trees, &balls);

    for step in 0..STEPS {
        for (ball, handles) in balls.iter_mut().flatten() {
            // Drift without bouncing, so that balls cross the world's edge,
            // and now and then jump anywhere.
            if rng.gen_bool(0.1) {
                (ball.x, ball.y) = around.get_random_point(&mut rng);
            } else {
                ball.x += 40.0 * ball.vx;
                ball.y += 40.0 * ball.vy;
            }
            if rng.gen_bool(0.05) {
                ball.r = random_radius(&mut rng);
            }
            trees.update(&ball.get_bounds(), *handles);
        }
        for _ in 0..BALL_COUNT / 100 {
            let id = rng.gen_range(0..balls.len());
            if let Some((_, handles)) = balls[id].take() {
                trees.remove(handles);
            }
        }
        for _ in 0..BALL_COUNT / 200 {
            let ball = random_ball(&mut rng, &around);
            let handles = trees.insert(&ball.get_bounds(), balls.len());
            balls.push(Some((ball, handles)));
        }
        if step % 2 == 1 {
            trees.refresh();
        }
        assert_searches_match_brute_force(&trees, &balls);
        assert_balls_find_themselves(&trees, &balls);
    }
}

/// Splitting the root must leave the items centered outside the world in it:
/// moved into a child, whose loose bounds miss them, searches lost them.
#[test]
fn loose_quadtree_split_keeps_items_centered_outside_the_world() {
    let world = Rectangle::new(0.0, 0.0, 1000.0, 1000.0);
    let mut tree = LooseQuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(world);
    tree.insert(Rectangle::center_rect(-500.0, 500.0, 10.0, 10.0), 99);
    for i in 0..MAX_ITEMS {
        tree.insert(Rectangle::center_rect(100.0 + 80.0 * i as f64, 300.0, 10.0, 10.0), i);
    }
    let mut found = Vec::new();
    tree.search(&Rectangle::center_rect(-500.0, 500.0, 20.0, 20.0), |id| found.push(id));
    assert_eq!(found, [99]);
}

/// Every structure holding items that are `Send` and `Sync` is itself `Send`
/// and `Sync`.
#[test]
fn structures_are_send_and_sync() {
    fn assert_send_sync<S: Send + Sync>() {}
    assert_send_sync::<LooseQuadTree<usize, MAX_LEVEL, MAX_ITEMS>>();
    assert_send_sync::<HierarchicalLooseGrid<usize, MAX_LEVEL>>();
    assert_send_sync::<UniformGrid<usize>>();
    assert_send_sync::<TwoLayerGrid<usize>>();
    assert_send_sync::<MultiGrid<usize>>();
    assert_send_sync::<HierarchicalGrid<usize, MAX_LEVEL>>();
    assert_send_sync::<ExactOverflowBoundGrid<usize, MAX_LEVEL>>();
    assert_send_sync::<BoxedLooseGrid<usize, MAX_LEVEL>>();
    assert_send_sync::<LazyOverflowBoundGrid<usize, MAX_LEVEL>>();
    assert_send_sync::<SpanOverflowBoundGrid<usize, MAX_LEVEL>>();
    assert_send_sync::<HaloOverflowBoundGrid<usize, MAX_LEVEL>>();
    assert_send_sync::<SpanGrid<usize>>();
    assert_send_sync::<HaloGrid<usize>>();
    assert_send_sync::<SortAndSweep<usize>>();
}

/// A `MAX_LEVEL` of zero, or one whose finest level has more cells than
/// `usize` counts, is rejected with its own message in every build, instead of
/// an arithmetic overflow panic in debug builds and a wrapped shift in release.
macro_rules! rejects_max_level {
    ($($name:ident: $create:expr, $message:literal;)*) => {
        $(#[test]
        #[should_panic(expected = $message)]
        fn $name() {
            $create;
        })*
    };
}
rejects_max_level!(
    loose_quadtree_rejects_level_zero: LooseQuadTree::<usize, 0, MAX_ITEMS>::new(world()), "cannot be zero";
    loose_quadtree_rejects_level_65: LooseQuadTree::<usize, 65, MAX_ITEMS>::new(world()), "too large";
    hierarchical_loose_grid_rejects_level_zero: HierarchicalLooseGrid::<usize, 0>::new(world()), "cannot be zero";
    hierarchical_loose_grid_rejects_level_65: HierarchicalLooseGrid::<usize, 65>::new(world()), "too large";
    hierarchical_grid_rejects_level_zero: HierarchicalGrid::<usize, 0>::new(world()), "cannot be zero";
    hierarchical_grid_rejects_level_65: HierarchicalGrid::<usize, 65>::new(world()), "too large";
    exact_grid_rejects_level_zero: ExactOverflowBoundGrid::<usize, 0>::new(world()), "cannot be zero";
    exact_grid_rejects_level_65: ExactOverflowBoundGrid::<usize, 65>::new(world()), "too large";
    boxed_loose_grid_rejects_level_zero: BoxedLooseGrid::<usize, 0>::new(world()), "cannot be zero";
    boxed_loose_grid_rejects_level_65: BoxedLooseGrid::<usize, 65>::new(world()), "too large";
    lazy_grid_rejects_level_zero: LazyOverflowBoundGrid::<usize, 0>::new(world()), "cannot be zero";
    lazy_grid_rejects_level_65: LazyOverflowBoundGrid::<usize, 65>::new(world()), "too large";
    span_grid_rejects_level_zero: SpanOverflowBoundGrid::<usize, 0>::new(world()), "cannot be zero";
    span_grid_rejects_level_65: SpanOverflowBoundGrid::<usize, 65>::new(world()), "too large";
    halo_grid_rejects_level_zero: HaloOverflowBoundGrid::<usize, 0>::new(world()), "cannot be zero";
    halo_grid_rejects_level_65: HaloOverflowBoundGrid::<usize, 65>::new(world()), "too large";
);

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

/// The methods of `OverflowBounded` that every structure forwards to its own,
/// and `refresh` for those that tighten their bounds only on request.
macro_rules! forward_items {
    ($($refresh:ident)?) => {
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
        $(fn refresh(&mut self) {
            self.$refresh();
        })?
    };
}

/// Multi-level structures, whose roots are square or fitted to the world.
macro_rules! new_or_fit {
    ($($t:ident $(: $refresh:ident)?),* $(,)?) => {
        $(impl<const L: u8> OverflowBounded for $t<usize, L> {
            fn create(world: Rectangle, fit: bool) -> Self {
                if fit {
                    Self::new_fit(world)
                } else {
                    Self::new(world)
                }
            }
            forward_items!($($refresh)?);
        })*
    };
}
new_or_fit!(
    ExactOverflowBoundGrid,
    BoxedLooseGrid: refresh_bounds,
    LazyOverflowBoundGrid: refresh_bounds,
    SpanOverflowBoundGrid: refresh_bounds,
    HaloOverflowBoundGrid: refresh_bounds,
);

/// Cell size of the finest level of the hierarchical structures over `world`,
/// for the single-level ones. With it `tight_item` puts its finest items in
/// cells and the larger ones in the overflow node, with many centers on cell
/// edges.
fn finest_cell_size(world: &Rectangle) -> f64 {
    let side = world.get_width().max(world.get_height());
    side / (1u32 << (MAX_LEVEL - 1)) as f64
}

impl OverflowBounded for HaloGrid<usize> {
    fn create(world: Rectangle, _fit: bool) -> Self {
        let cell_size = finest_cell_size(&world);
        Self::new(world, cell_size)
    }
    forward_items!(refresh_bounds);
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

/// The bounds of every live item, `count` queries of every size, and `count`
/// queries that overlap one live item by a single float, where a bound that
/// is too small would lose it.
fn tight_queries(
    rng: &mut StdRng,
    root: &Rectangle,
    live: &[(usize, Rectangle)],
    count: usize,
) -> Vec<Rectangle> {
    let mut queries: Vec<Rectangle> = live.iter().map(|(_, bounds)| bounds.clone()).collect();
    for _ in 0..count {
        let scale = 2f64.powf(-rng.gen_range(0.0..MAX_LEVEL as f64 + 1.0));
        let (x, y) = root.get_random_point(rng);
        queries.push(Rectangle::center_rect(
            x,
            y,
            root.get_width() * scale * rng.gen_range(0.2..1.0),
            root.get_height() * scale * rng.gen_range(0.2..1.0),
        ));
    }
    for _ in 0..count {
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
    count: usize,
    context: &str,
) {
    let live: Vec<(usize, Rectangle)> = items
        .iter()
        .enumerate()
        .filter_map(|(id, item)| item.as_ref().map(|(_, bounds)| (id, bounds.clone())))
        .collect();
    for query in tight_queries(rng, &tree.root(), &live, count) {
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
                assert_tight_search_matches_brute_force(&tree, &items, &mut rng, 300, &context);

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
                    assert_tight_search_matches_brute_force(&tree, &items, &mut rng, 300, &context);
                    tree.refresh();
                    assert_tight_search_matches_brute_force(&tree, &items, &mut rng, 300, &context);
                }
            }
        }
    }
}

/// Searches that use bounds below the fixed reach, or cell boxes, must still
/// find every item, including items that overlap a query by a single float.
#[test]
fn check_tight_overflow_bounds() {
    check_tight_bounds::<ExactOverflowBoundGrid<usize, MAX_LEVEL>>("ExactOverflowBoundGrid");
    check_tight_bounds::<BoxedLooseGrid<usize, MAX_LEVEL>>("BoxedLooseGrid");
    check_tight_bounds::<LazyOverflowBoundGrid<usize, MAX_LEVEL>>("LazyOverflowBoundGrid");
    check_tight_bounds::<SpanOverflowBoundGrid<usize, MAX_LEVEL>>(
        "SpanOverflowBoundGrid",
    );
    check_tight_bounds::<HaloOverflowBoundGrid<usize, MAX_LEVEL>>(
        "HaloOverflowBoundGrid",
    );
    check_tight_bounds::<HaloGrid<usize>>("HaloGrid");
    // Its skipped tests rely on the cell ranges alone; items reaching across
    // a cell edge by a single float test that reasoning.
    check_tight_bounds::<TwoLayerGrid<usize>>("TwoLayerGrid");
    check_tight_bounds::<MultiGrid<usize>>("MultiGrid");
    // Without bounds the ancestor walk and HG's scans decide from the query's
    // corners which neighbors to read; items reaching across an edge by a
    // single float test those decisions.
    check_tight_bounds::<HierarchicalLooseGrid<usize, MAX_LEVEL>>("HierarchicalLooseGrid");
    check_tight_bounds::<HierarchicalGrid<usize, MAX_LEVEL>>("HierarchicalGrid");
}

/// Bounds of an item on a random level, the root included.
fn sparse_item(rng: &mut StdRng, root: &Rectangle, world: &Rectangle) -> Rectangle {
    if rng.gen_bool(0.1) {
        // More than half the root on both axes puts the item on the root.
        let (x, y) = world.get_random_point(rng);
        Rectangle::center_rect(x, y, root.get_width() * 0.8, root.get_height() * 0.8)
    } else {
        tight_item(rng, root, world, 0.1)
    }
}

/// With at most seven items, the levels below the root keep emptying and
/// filling again; searches must find every item after each step, before and
/// after a refresh.
fn check_sparse_levels<Tree: OverflowBounded>(name: &str) {
    let mut rng = StdRng::seed_from_u64(13);
    for world in TIGHT_WORLDS {
        for fit in [false, true] {
            let context = format!("{name}, world {world:?}, fit {fit}");
            let mut tree = Tree::create(world.clone(), fit);
            let root = tree.root();
            let mut items: Vec<Option<(ItemHandle, Rectangle)>> = Vec::new();
            for step in 0..400 {
                let live: Vec<usize> = (0..items.len()).filter(|&id| items[id].is_some()).collect();
                let operation = match live.len() {
                    0 | 1 => 0,
                    7.. => 2,
                    _ => rng.gen_range(0..3),
                };
                match operation {
                    0 => {
                        let bounds = sparse_item(&mut rng, &root, &world);
                        let handle = tree.insert_item(bounds.clone(), items.len());
                        items.push(Some((handle, bounds)));
                    }
                    1 => {
                        let id = live[rng.gen_range(0..live.len())];
                        let (handle, bounds) = items[id].as_mut().unwrap();
                        *bounds = sparse_item(&mut rng, &root, &world);
                        tree.update_item(bounds.clone(), *handle);
                    }
                    _ => {
                        let id = live[rng.gen_range(0..live.len())];
                        let (handle, _) = items[id].take().unwrap();
                        tree.remove_item(handle);
                    }
                }
                let context = format!("{context}, step {step}");
                assert_tight_search_matches_brute_force(&tree, &items, &mut rng, 40, &context);
                if rng.gen_bool(0.2) {
                    tree.refresh();
                    assert_tight_search_matches_brute_force(&tree, &items, &mut rng, 40, &context);
                }
            }
        }
    }
}

/// The ancestor walk stops at the coarsest level below the root that holds
/// items, or may hold them until the next refresh; it must not stop short of
/// one that does.
#[test]
fn check_sparse_levels_keep_searches_exact() {
    check_sparse_levels::<ExactOverflowBoundGrid<usize, MAX_LEVEL>>("ExactOverflowBoundGrid");
    check_sparse_levels::<BoxedLooseGrid<usize, MAX_LEVEL>>("BoxedLooseGrid");
    check_sparse_levels::<LazyOverflowBoundGrid<usize, MAX_LEVEL>>("LazyOverflowBoundGrid");
    check_sparse_levels::<SpanOverflowBoundGrid<usize, MAX_LEVEL>>(
        "SpanOverflowBoundGrid",
    );
    check_sparse_levels::<HaloOverflowBoundGrid<usize, MAX_LEVEL>>(
        "HaloOverflowBoundGrid",
    );
    check_sparse_levels::<HierarchicalLooseGrid<usize, MAX_LEVEL>>("HierarchicalLooseGrid");
}

/// Whether a tree holding one item finds it with a query that meets it, here
/// by a float.
fn finds_the_item<Tree: OverflowBounded>(
    fit: bool,
    world: [f64; 4],
    item: [f64; 4],
    query: [f64; 4],
) -> bool {
    let rectangle = |b: [f64; 4]| Rectangle::new(b[0], b[1], b[2], b[3]);
    let (item, query) = (rectangle(item), rectangle(query));
    assert!(item.intersects(&query));
    let mut tree = Tree::create(rectangle(world), fit);
    tree.insert_item(item, 0);
    tree.refresh();
    let mut found = false;
    tree.search_items(&query, &mut |_| found = true);
    found
}

/// Items whose bounds round a float past the half node they may reach beyond
/// their cell, queries of a coarser level that start or end on that half
/// node once taken relative to the root origin, and the scan of the finer
/// levels that must still reach them.
/// From the differential probes of the hierarchical loose grid: the first
/// item, centered on the world's edge, sits in the last column, and the
/// query's minimum rounds onto the column's reach; the second query's
/// maximum is the item's minimum but for a float; the third case is in a
/// square root, the fourth item's width rounds to its level's size; the
/// fifth is the first with the axes swapped.
#[test]
fn scans_reach_items_past_their_reach_by_a_float() {
    macro_rules! cases {
        ($($tree:ident),*) => {$(
            assert!(finds_the_item::<$tree<usize, 7>>(
                true,
                [-128.0, -128.0, 896.0, 384.0],
                [832.0, 302.8907177374069, 960.0, 334.8907177374069],
                [959.9999999999999, 302.8907177374069, 1137.9644479941, 334.8907177374069],
            ), "{}, minimum on the reach", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 5>>(
                true,
                [-128.0, -128.0, 896.0, 384.0],
                [736.0000000000002, 207.99999999999997, 800.0000000000002, 239.99999999999997],
                [736.0000000000002, 169.9099894198402, 800.0000000000002, 208.0],
            ), "{}, maximum a float past the item", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 4>>(
                false,
                [-1.0, -1.0, 1.0, 1.0],
                [
                    0.24999999999999992,
                    5.551115123125783e-17,
                    0.7499999999999999,
                    0.4999999999999999,
                ],
                [-0.2500000000000001, 0.3627828096920144, 0.2500000000000001, 0.7084902535781837],
            ), "{}, square root", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 4>>(
                true,
                [-5.0, 2.0, 295.0, 1302.0],
                [107.49999999999994, 326.99999999999994, 182.49999999999994, 977.0],
                [174.52658891816444, -323.0, 415.4734110818355, 327.0],
            ), "{}, width rounded to the level's size", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 7>>(
                true,
                [-128.0, -128.0, 384.0, 896.0],
                [302.8907177374069, 832.0, 334.8907177374069, 960.0],
                [302.8907177374069, 959.9999999999999, 334.8907177374069, 1137.9644479941],
            ), "{}, minimum y on the reach", stringify!($tree));
        )*};
    }
    cases!(
        HierarchicalLooseGrid,
        ExactOverflowBoundGrid,
        BoxedLooseGrid,
        LazyOverflowBoundGrid,
        SpanOverflowBoundGrid,
        HaloOverflowBoundGrid
    );
}

/// Items that reach a float past the half node they may reach beyond their
/// cell, the center line of the next cell, and queries of the same level
/// that end on that line from either side along either axis. The first item
/// is centered a float left of a cell edge but goes to the cell on the right,
/// as its relative center rounds onto the edge. The ancestor walk must read
/// the item's cell although the query does not cross the line. In the last
/// four cases the item's cell is a diagonal neighbor, up left, up right, down
/// left and down right, and the query starts or ends exactly on the line
/// along x.
#[test]
fn walks_read_neighbors_that_a_query_ends_at() {
    macro_rules! cases {
        ($($tree:ident),*) => {$(
            assert!(finds_the_item::<$tree<usize, 10>>(
                false,
                [-128.0, -128.0, 896.0, 384.0],
                [403.99999999999994, 197.97820269930065, 411.99999999999994, 205.97820269930065],
                [397.97465150723383, 197.97820269930065, 404.0, 205.97820269930065],
            ), "{}, maximum x on the line", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 5>>(
                false,
                [0.3, -0.7, 1000.1, 333.3],
                [375.2250000000001, 82.4246186644975, 625.1750000000001, 331.9722875558981],
                [625.175, 168.82895253072385, 660.1290045889748, 195.7744949319497],
            ), "{}, minimum x on the line", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 10>>(
                false,
                [-2553747.1238352456, -186969.24312036688, -2553747.103898442, -186969.15195679414],
                [-2553747.1226531733, -186969.2089340271, -2553747.111353074, -186969.1861431339],
                [-2553747.12170412, -186969.18614313393, -2553747.1208647257, -186969.18375089634],
            ), "{}, minimum y on the line", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 4>>(
                false,
                [-128.0, -128.0, 896.0, 384.0],
                [281.7631482395004, 63.99999999999999, 344.6212442037694, 191.99999999999997],
                [236.6073673504712, -6.135935554517005, 357.2613953315605, 64.0],
            ), "{}, maximum y on the line", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 5>>(
                false,
                [0.3, 0.3, 1000.1, 1000.1],
                [375.2250000000001, 375.2250000000001, 625.1750000000001, 625.1750000000001],
                [625.175, 625.175, 660.1290045889748, 660.1290045889748],
            ), "{}, up left", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 5>>(
                false,
                [-128.0, -128.0, 896.0, 384.0],
                [415.99999999999994, 95.99999999999999, 479.99999999999994, 159.99999999999997],
                [362.99474866598837, 159.99999999999994, 416.0, 213.49099771974548],
            ), "{}, up right", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 4>>(
                true,
                [-971.4775966428215, -535.5709852931466, 555.4357432193824, 462.41088881109783],
                [-112.58884297033185, 25.79381889049085, 78.27532451244363, 150.54155315352136],
                [78.2753245124436, -83.70577087832085, 195.92928427880582, 25.79381889049091],
            ), "{}, down left", stringify!($tree));
            assert!(finds_the_item::<$tree<usize, 4>>(
                false,
                [-128.0, -128.0, 896.0, 384.0],
                [63.99999999999999, 64.0, 192.0, 192.0],
                [-26.440239030628533, -55.33756565344977, 64.0, 64.00000000000001],
            ), "{}, down right", stringify!($tree));
        )*};
    }
    cases!(
        HierarchicalLooseGrid,
        ExactOverflowBoundGrid,
        BoxedLooseGrid,
        LazyOverflowBoundGrid,
        SpanOverflowBoundGrid,
        HaloOverflowBoundGrid
    );
}

/// The same for the scans of the single-level grids, with the cell sizes of
/// the probes that found the cases: a query larger than a cell whose maximum
/// is the item's minimum but for a float, the same at a minimum along y and
/// along x, and a cell of an eighth of a unit.
#[test]
fn single_level_scans_reach_items_past_their_reach_by_a_float() {
    let rectangle = |b: [f64; 4]| Rectangle::new(b[0], b[1], b[2], b[3]);
    let cases = [
        (
            [-128.0, -128.0, 896.0, 384.0],
            128.0,
            [64.0, 324.6986158711724, 191.99999999999997, 443.3013841288276],
            [-298.75815933894233, 324.6986158711724, 64.00000000000001, 443.3013841288276],
        ),
        (
            [-1.0, -1.0, 1.0, 1.0],
            1.0,
            [-1.3131659333683308, 0.5, -0.6868340666316687, 1.5],
            [-1.3131659333683308, 1.4999999999999998, -0.6868340666316687, 4.424846802161501],
        ),
        (
            [-1.0, -1.0, 1.0, 1.0],
            1.0,
            [0.5, -1.3131659333683308, 1.5, -0.6868340666316687],
            [1.4999999999999998, -1.3131659333683308, 4.424846802161501, -0.6868340666316687],
        ),
        (
            [-1.0, -1.0, 1.0, 1.0],
            0.125,
            [0.18749999999999994, 0.18750000000000003, 0.3124999999999999, 0.3125],
            [0.18749999999999994, -0.13567152657319528, 0.3124999999999999, 0.18750000000000006],
        ),
    ];
    for (world, cell, item, query) in cases {
        let (item, query) = (rectangle(item), rectangle(query));
        assert!(item.intersects(&query));
        let mut span = SpanGrid::new(rectangle(world), cell);
        let mut halo = HaloGrid::new(rectangle(world), cell);
        span.insert(item.clone(), 0);
        halo.insert(item.clone(), 0);
        span.refresh_bounds();
        halo.refresh_bounds();
        let (mut by_span, mut by_halo) = (false, false);
        span.search(&query, |_| by_span = true);
        halo.search(&query, |_| by_halo = true);
        assert!(by_span && by_halo, "cell {cell}: SpanGrid {by_span}, HaloGrid {by_halo}");
    }
}

/// A query with a NaN corner meets nothing, as every comparison with NaN is
/// false, so no structure may report an item for it, also not from the nodes
/// that its scans take as inside the query without testing their items.
#[test]
fn queries_with_a_nan_corner_find_nothing() {
    let mut rng = StdRng::seed_from_u64(11);
    let world = world();
    let mut trees = Trees::new(&world);
    let side = world.get_width().max(world.get_height());
    let cell = side / (1u32 << (MAX_LEVEL - 1)) as f64;
    let mut span = SpanGrid::new(world.clone(), cell);
    let mut halo = HaloGrid::new(world.clone(), cell);
    for id in 0..BALL_COUNT {
        let bounds = random_ball(&mut rng, &world).get_bounds();
        trees.insert(&bounds, id);
        span.insert(bounds.clone(), id);
        halo.insert(bounds, id);
    }
    trees.refresh();
    span.refresh_bounds();
    halo.refresh_bounds();

    let spans = [
        [-1e300, -1e300, 1e300, 1e300],
        [-128.0, -128.0, 896.0, 384.0],
        [100.0, 50.0, 300.0, 200.0],
    ];
    for corners in spans {
        for nan in 0..4 {
            let mut query = corners;
            query[nan] = f64::NAN;
            let query = Rectangle::new(query[0], query[1], query[2], query[3]);
            let mut found = trees.search(&query, usize::MAX);
            found.push(("SpanGrid", collect(usize::MAX, |cb| span.search(&query, cb))));
            found.push(("HaloGrid", collect(usize::MAX, |cb| halo.search(&query, cb))));
            let reporting: Vec<_> = found
                .iter()
                .filter(|(_, items)| !items.is_empty())
                .map(|(name, items)| format!("{name} ({})", items.len()))
                .collect();
            assert!(reporting.is_empty(), "{query:?}: {}", reporting.join(", "));
        }
    }
}

/// With one to three levels, the walk below the root is empty or short, and
/// the stop past the finest level (`MAX_LEVEL + 1`) is the end of the grid.
#[test]
fn check_sparse_levels_with_few_levels() {
    macro_rules! few_levels {
        ($($tree:ident),*) => {$(
            check_sparse_levels::<$tree<usize, 1>>(concat!(stringify!($tree), ", MAX_LEVEL 1"));
            check_sparse_levels::<$tree<usize, 2>>(concat!(stringify!($tree), ", MAX_LEVEL 2"));
            check_sparse_levels::<$tree<usize, 3>>(concat!(stringify!($tree), ", MAX_LEVEL 3"));
        )*};
    }
    few_levels!(
        ExactOverflowBoundGrid,
        BoxedLooseGrid,
        LazyOverflowBoundGrid,
        SpanOverflowBoundGrid,
        HaloOverflowBoundGrid,
        HierarchicalLooseGrid
    );
}

/// The structures without bounds below the fixed reach take part only in
/// the self-query check, with the cells of the finest level of the
/// hierarchical structures where they need a cell size.
impl OverflowBounded for UniformGrid<usize> {
    fn create(world: Rectangle, _fit: bool) -> Self {
        let cell_size = finest_cell_size(&world);
        Self::new(world, cell_size)
    }
    forward_items!();
}

impl OverflowBounded for TwoLayerGrid<usize> {
    fn create(world: Rectangle, _fit: bool) -> Self {
        let cell_size = finest_cell_size(&world);
        Self::new(world, cell_size)
    }
    forward_items!();
}

impl OverflowBounded for MultiGrid<usize> {
    fn create(world: Rectangle, _fit: bool) -> Self {
        let cell_size = finest_cell_size(&world);
        Self::new(world, &[cell_size, cell_size * 4.0, cell_size * 16.0])
    }
    forward_items!();
}

impl OverflowBounded for SpanGrid<usize> {
    fn create(world: Rectangle, _fit: bool) -> Self {
        let cell_size = finest_cell_size(&world);
        Self::new(world, cell_size)
    }
    forward_items!(refresh_bounds);
}

impl<const L: u8> OverflowBounded for HierarchicalGrid<usize, L> {
    fn create(world: Rectangle, _fit: bool) -> Self {
        Self::new(world)
    }
    forward_items!();
}

new_or_fit!(HierarchicalLooseGrid);

/// Structures that search with the bounds of every stored item in one pass.
trait SelfQuery {
    fn self_query(&self, callback: &mut dyn FnMut(usize, usize));
}

macro_rules! self_query {
    ($($t:ty),*) => {
        $(impl SelfQuery for $t {
            fn self_query(&self, callback: &mut dyn FnMut(usize, usize)) {
                self.for_each_self_query(|query, found| callback(query, found));
            }
        })*
    };
}
self_query!(
    ExactOverflowBoundGrid<usize, MAX_LEVEL>,
    BoxedLooseGrid<usize, MAX_LEVEL>,
    LazyOverflowBoundGrid<usize, MAX_LEVEL>,
    SpanOverflowBoundGrid<usize, MAX_LEVEL>,
    HaloOverflowBoundGrid<usize, MAX_LEVEL>,
    HierarchicalLooseGrid<usize, MAX_LEVEL>,
    HierarchicalGrid<usize, MAX_LEVEL>,
    UniformGrid<usize>,
    TwoLayerGrid<usize>,
    MultiGrid<usize>,
    SpanGrid<usize>,
    HaloGrid<usize>
);

/// Every pair (query, found) the self query reports must be a pair of live
/// items whose bounds intersect, the item itself included, and every such
/// pair must be reported exactly once.
fn assert_self_query_matches_brute_force(
    tree: &(impl SelfQuery + OverflowBounded),
    items: &[Option<(ItemHandle, Rectangle)>],
    context: &str,
) {
    let live: Vec<(usize, &Rectangle)> = items
        .iter()
        .enumerate()
        .filter_map(|(id, item)| item.as_ref().map(|(_, bounds)| (id, bounds)))
        .collect();
    let mut expected = Vec::new();
    for &(query, query_bounds) in &live {
        for &(found, found_bounds) in &live {
            if found_bounds.intersects(query_bounds) {
                expected.push((query, found));
            }
        }
    }
    // The same pairs as searching with every item one by one.
    let mut searched = Vec::new();
    for &(query, query_bounds) in &live {
        tree.search_items(query_bounds, &mut |found| searched.push((query, found)));
    }
    let mut reported = Vec::new();
    tree.self_query(&mut |query, found| reported.push((query, found)));
    expected.sort_unstable();
    searched.sort_unstable();
    reported.sort_unstable();
    assert_eq!(reported, searched, "{context}: against searching item by item");
    assert_eq!(reported, expected, "{context}: against brute force");
}

/// The worlds of the tight checks and one taller than wide, whose fitted
/// root has more rows than columns.
const SELF_QUERY_WORLDS: [Rectangle; 4] = [
    Rectangle::new(-128.0, -128.0, 896.0, 384.0),
    Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
    Rectangle::new(0.3, -0.7, 1000.1, 333.3),
    Rectangle::new(-5.0, 2.0, 295.0, 1302.0),
];

/// Items of the kinds `tight_item` never makes: zero-sized, larger than the
/// root, and outside the world beyond its min corner.
fn odd_items(rng: &mut StdRng, root: &Rectangle, world: &Rectangle) -> Vec<Rectangle> {
    let (x, y) = world.get_random_point(rng);
    vec![
        Rectangle::new(x, y, x, y),
        root.scale(1.5),
        Rectangle::new(
            world.min_x - 0.3 * world.get_width(),
            world.min_y - 0.2 * world.get_height(),
            world.min_x - 0.1 * world.get_width(),
            world.min_y - 0.05 * world.get_height(),
        ),
    ]
}

fn check_self_query<Tree: OverflowBounded + SelfQuery>(name: &str) {
    let mut rng = StdRng::seed_from_u64(13);
    for (world, fit) in SELF_QUERY_WORLDS.into_iter().flat_map(|world| [(world.clone(), false), (world, true)]) {
        let context = format!("{name}, world {world:?}, fit {fit}");
        let mut tree = Tree::create(world.clone(), fit);
        let root = tree.root();
        let mut items: Vec<Option<(ItemHandle, Rectangle)>> = (0..300)
            .map(|id| {
                let bounds = tight_item(&mut rng, &root, &world, 0.1);
                Some((tree.insert_item(bounds.clone(), id), bounds))
            })
            .collect();
        for bounds in odd_items(&mut rng, &root, &world) {
            let handle = tree.insert_item(bounds.clone(), items.len());
            items.push(Some((handle, bounds)));
        }
        assert_self_query_matches_brute_force(&tree, &items, &context);
        for round in 0..3 {
            for (handle, bounds) in items.iter_mut().flatten() {
                if rng.gen_bool(0.5) {
                    *bounds = tight_item(&mut rng, &root, &world, 0.1);
                    tree.update_item(bounds.clone(), *handle);
                }
            }
            for _ in 0..30 {
                let id = rng.gen_range(0..items.len());
                if let Some((handle, _)) = items[id].take() {
                    tree.remove_item(handle);
                }
            }
            for _ in 0..20 {
                let bounds = tight_item(&mut rng, &root, &world, 0.1);
                let handle = tree.insert_item(bounds.clone(), items.len());
                items.push(Some((handle, bounds)));
            }
            let context = format!("{context}, round {round}");
            assert_self_query_matches_brute_force(&tree, &items, &format!("{context}, stale"));
            tree.refresh();
            assert_self_query_matches_brute_force(&tree, &items, &format!("{context}, refreshed"));
        }
    }
}

/// Structures that report every intersecting pair of distinct items once.
trait PairQuery {
    fn pairs(&self, callback: &mut dyn FnMut(usize, usize));
}

macro_rules! pair_query {
    ($($t:ty),*) => {
        $(impl PairQuery for $t {
            fn pairs(&self, callback: &mut dyn FnMut(usize, usize)) {
                self.for_each_pair(|a, b| callback(a, b));
            }
        })*
    };
}
pair_query!(
    ExactOverflowBoundGrid<usize, MAX_LEVEL>,
    LazyOverflowBoundGrid<usize, MAX_LEVEL>,
    SpanOverflowBoundGrid<usize, MAX_LEVEL>,
    HaloOverflowBoundGrid<usize, MAX_LEVEL>,
    MultiGrid<usize>,
    HierarchicalLooseGrid<usize, MAX_LEVEL>,
    BoxedLooseGrid<usize, MAX_LEVEL>,
    HierarchicalGrid<usize, MAX_LEVEL>,
    UniformGrid<usize>,
    TwoLayerGrid<usize>,
    SpanGrid<usize>,
    HaloGrid<usize>,
    LooseQuadTree<usize, MAX_LEVEL, MAX_ITEMS>
);

impl<const L: u8, const M: usize> OverflowBounded for LooseQuadTree<usize, L, M> {
    fn create(world: Rectangle, fit: bool) -> Self {
        if fit {
            Self::new_fit(world)
        } else {
            Self::new(world)
        }
    }
    forward_items!();
}

/// `SortAndSweep`, which needs no world, with the world of the pair check as
/// its root; `SCALAR` picks the scalar sweep for its pairs.
struct Sweep<const SCALAR: bool> {
    sweep: SortAndSweep<usize>,
    world: Rectangle,
}

impl<const SCALAR: bool> OverflowBounded for Sweep<SCALAR> {
    fn create(world: Rectangle, _fit: bool) -> Self {
        Self {
            sweep: SortAndSweep::new(),
            world,
        }
    }
    fn root(&self) -> Rectangle {
        self.world.clone()
    }
    fn insert_item(&mut self, bounds: Rectangle, id: usize) -> ItemHandle {
        self.sweep.insert(bounds, id)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.sweep.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.sweep.remove(handle);
    }
    fn search_items(&self, bounds: &Rectangle, callback: &mut dyn FnMut(usize)) {
        self.sweep.search(bounds, callback);
    }
    fn refresh(&mut self) {
        self.sweep.refresh_order();
    }
}

impl<const SCALAR: bool> PairQuery for Sweep<SCALAR> {
    fn pairs(&self, callback: &mut dyn FnMut(usize, usize)) {
        if SCALAR {
            self.sweep.for_each_pair_scalar(|a, b| callback(a, b));
        } else {
            self.sweep.for_each_pair(|a, b| callback(a, b));
        }
    }
}

/// Every pair of distinct live items whose bounds intersect must be reported
/// exactly once, in either order, and no other pair; a duplicate or an item
/// paired with itself shows up as an extra entry.
fn assert_pairs_match_brute_force(
    tree: &(impl PairQuery + OverflowBounded),
    items: &[Option<(ItemHandle, Rectangle)>],
    context: &str,
) {
    let live: Vec<(usize, &Rectangle)> = items
        .iter()
        .enumerate()
        .filter_map(|(id, item)| item.as_ref().map(|(_, bounds)| (id, bounds)))
        .collect();
    let mut expected = Vec::new();
    for (i, &(a, a_bounds)) in live.iter().enumerate() {
        for &(b, b_bounds) in &live[i + 1..] {
            if a_bounds.intersects(b_bounds) {
                expected.push((a.min(b), a.max(b)));
            }
        }
    }
    let mut reported = Vec::new();
    tree.pairs(&mut |a, b| reported.push((a.min(b), a.max(b))));
    expected.sort_unstable();
    reported.sort_unstable();
    assert_eq!(reported, expected, "{context}");
}

/// As `check_self_query`, for the pairs: random tight items, odd items, then
/// rounds of updates, removals and insertions, checked stale and refreshed.
fn check_pairs<Tree: OverflowBounded + PairQuery>(name: &str) {
    let mut rng = StdRng::seed_from_u64(17);
    for (world, fit) in SELF_QUERY_WORLDS.into_iter().flat_map(|world| [(world.clone(), false), (world, true)]) {
        let context = format!("{name}, world {world:?}, fit {fit}");
        let mut tree = Tree::create(world.clone(), fit);
        let root = tree.root();
        let mut items: Vec<Option<(ItemHandle, Rectangle)>> = (0..300)
            .map(|id| {
                let bounds = tight_item(&mut rng, &root, &world, 0.1);
                Some((tree.insert_item(bounds.clone(), id), bounds))
            })
            .collect();
        for bounds in odd_items(&mut rng, &root, &world) {
            let handle = tree.insert_item(bounds.clone(), items.len());
            items.push(Some((handle, bounds)));
        }
        assert_pairs_match_brute_force(&tree, &items, &context);
        for round in 0..3 {
            for (handle, bounds) in items.iter_mut().flatten() {
                if rng.gen_bool(0.5) {
                    *bounds = tight_item(&mut rng, &root, &world, 0.1);
                    tree.update_item(bounds.clone(), *handle);
                }
            }
            for _ in 0..30 {
                let id = rng.gen_range(0..items.len());
                if let Some((handle, _)) = items[id].take() {
                    tree.remove_item(handle);
                }
            }
            for _ in 0..20 {
                let bounds = tight_item(&mut rng, &root, &world, 0.1);
                let handle = tree.insert_item(bounds.clone(), items.len());
                items.push(Some((handle, bounds)));
            }
            let context = format!("{context}, round {round}");
            assert_pairs_match_brute_force(&tree, &items, &format!("{context}, stale"));
            tree.refresh();
            assert_pairs_match_brute_force(&tree, &items, &format!("{context}, refreshed"));
        }

        // Items a fraction of a finest cell large, packed into 8 by 8 cells, so
        // that every node and neighbor direction holds intersecting pairs.
        let cell = finest_cell_size(&world);
        let (x0, y0) = world.get_min();
        let area = Rectangle::new(x0 + 2.0 * cell, y0 + 2.0 * cell, x0 + 10.0 * cell, y0 + 10.0 * cell);
        let dense = |rng: &mut StdRng| {
            let (x, y) = area.get_random_point(rng);
            let (w, h) = (cell * rng.gen_range(0.2..0.95), cell * rng.gen_range(0.2..0.95));
            Rectangle::center_rect(x, y, w, h)
        };
        for _ in 0..400 {
            let bounds = dense(&mut rng);
            let handle = tree.insert_item(bounds.clone(), items.len());
            items.push(Some((handle, bounds)));
        }
        let context = format!("{context}, dense");
        assert_pairs_match_brute_force(&tree, &items, &format!("{context}, stale"));
        tree.refresh();
        assert_pairs_match_brute_force(&tree, &items, &format!("{context}, refreshed"));
        for (handle, bounds) in items.iter_mut().flatten() {
            if rng.gen_bool(0.5) {
                let (x, y) = bounds.get_center();
                let step = cell * 0.3;
                *bounds = Rectangle::center_rect(
                    x + rng.gen_range(-step..step),
                    y + rng.gen_range(-step..step),
                    bounds.get_width(),
                    bounds.get_height(),
                );
                tree.update_item(bounds.clone(), *handle);
            }
        }
        assert_pairs_match_brute_force(&tree, &items, &format!("{context}, moved, stale"));
        tree.refresh();
        assert_pairs_match_brute_force(&tree, &items, &format!("{context}, moved, refreshed"));
    }
}

#[test]
fn pairs_match_brute_force() {
    check_pairs::<ExactOverflowBoundGrid<usize, MAX_LEVEL>>("ExactOverflowBoundGrid");
    check_pairs::<LazyOverflowBoundGrid<usize, MAX_LEVEL>>("LazyOverflowBoundGrid");
    check_pairs::<SpanOverflowBoundGrid<usize, MAX_LEVEL>>("SpanOverflowBoundGrid");
    check_pairs::<HaloOverflowBoundGrid<usize, MAX_LEVEL>>("HaloOverflowBoundGrid");
    check_pairs::<MultiGrid<usize>>("MultiGrid");
    check_pairs::<HierarchicalLooseGrid<usize, MAX_LEVEL>>("HierarchicalLooseGrid");
    check_pairs::<BoxedLooseGrid<usize, MAX_LEVEL>>("BoxedLooseGrid");
    check_pairs::<HierarchicalGrid<usize, MAX_LEVEL>>("HierarchicalGrid");
    check_pairs::<UniformGrid<usize>>("UniformGrid");
    check_pairs::<TwoLayerGrid<usize>>("TwoLayerGrid");
    check_pairs::<SpanGrid<usize>>("SpanGrid");
    check_pairs::<HaloGrid<usize>>("HaloGrid");
    check_pairs::<LooseQuadTree<usize, MAX_LEVEL, MAX_ITEMS>>("LooseQuadTree");
    check_pairs::<Sweep<false>>("SortAndSweep");
    check_pairs::<Sweep<true>>("SortAndSweep, scalar");
}

#[test]
fn self_queries_match_brute_force() {
    check_self_query::<ExactOverflowBoundGrid<usize, MAX_LEVEL>>("ExactOverflowBoundGrid");
    check_self_query::<BoxedLooseGrid<usize, MAX_LEVEL>>("BoxedLooseGrid");
    check_self_query::<LazyOverflowBoundGrid<usize, MAX_LEVEL>>("LazyOverflowBoundGrid");
    check_self_query::<SpanOverflowBoundGrid<usize, MAX_LEVEL>>("SpanOverflowBoundGrid");
    check_self_query::<HaloOverflowBoundGrid<usize, MAX_LEVEL>>("HaloOverflowBoundGrid");
    check_self_query::<HierarchicalLooseGrid<usize, MAX_LEVEL>>("HierarchicalLooseGrid");
    check_self_query::<HierarchicalGrid<usize, MAX_LEVEL>>("HierarchicalGrid");
    check_self_query::<UniformGrid<usize>>("UniformGrid");
    check_self_query::<TwoLayerGrid<usize>>("TwoLayerGrid");
    check_self_query::<MultiGrid<usize>>("MultiGrid");
    check_self_query::<SpanGrid<usize>>("SpanGrid");
    check_self_query::<HaloGrid<usize>>("HaloGrid");
}

/// Fitted worlds wider and taller than high, whose roots differ between the
/// axes.
const FITTED_WORLDS: [Rectangle; 2] = [
    Rectangle::new(0.0, 0.0, 1000.0, 300.0),
    Rectangle::new(-5.0, 2.0, 295.0, 1302.0),
];

/// The level `position` must pick, by definition: the bit length of how many
/// times the item fits across the root along the tighter axis, clamped to the
/// levels, or the root for a center outside the world.
fn level_by_division(root: &Rectangle, world: &Rectangle, bounds: &Rectangle) -> usize {
    if !world.contains_point(bounds.get_center()) {
        return 1;
    }
    let multiple_x = (root.get_width() / bounds.get_width()) as usize;
    let multiple_y = (root.get_height() / bounds.get_height()) as usize;
    let level = usize::BITS - multiple_x.min(multiple_y).leading_zeros();
    (level as usize).clamp(1, MAX_LEVEL as usize)
}

/// A side of `root_side` divided by a random power of two up to below the
/// finest level, and often exactly a level's node size or a float next to it.
fn random_side(rng: &mut StdRng, root_side: f64) -> f64 {
    let k = rng.gen_range(0..MAX_LEVEL as i32 + 2);
    let threshold = root_side / 2f64.powi(k);
    match rng.gen_range(0..4) {
        0 => threshold,
        1 => threshold.next_up(),
        2 => threshold.next_down(),
        _ => threshold * rng.gen_range(0.5..1.0),
    }
}

macro_rules! check_size_levels {
    ($($t:ident),*) => {
        $(for world in FITTED_WORLDS {
            let tree = $t::<usize, MAX_LEVEL>::new_fit(world.clone());
            let root = tree.get_bounds().clone();
            let mut rng = StdRng::seed_from_u64(19);
            for _ in 0..20000 {
                let (x, y) = world.scale(1.1).get_random_point(&mut rng);
                let width = random_side(&mut rng, root.get_width());
                let height = random_side(&mut rng, root.get_height());
                let bounds = Rectangle::center_rect(x, y, width, height);
                assert_eq!(
                    tree.position(&bounds).0,
                    level_by_division(&root, &world, &bounds),
                    "{}, world {world:?}, bounds {bounds:?}",
                    stringify!($t)
                );
            }
        })*
    };
}

/// `position` must pick the level by the items' size exactly as dividing the
/// root by it does, on both axes of roots that are not square.
#[test]
fn positions_follow_the_size_levels() {
    check_size_levels!(
        HierarchicalLooseGrid,
        ExactOverflowBoundGrid,
        BoxedLooseGrid,
        LazyOverflowBoundGrid,
        SpanOverflowBoundGrid,
        HaloOverflowBoundGrid
    );
    // The hierarchical grid always covers the world with a square region,
    // whose longer side alone decides; the definition is the same.
    for world in FITTED_WORLDS {
        let tree = HierarchicalGrid::<usize, MAX_LEVEL>::new(world.clone());
        let root = tree.get_bounds();
        let mut rng = StdRng::seed_from_u64(23);
        for _ in 0..20000 {
            let (x, y) = world.scale(1.1).get_random_point(&mut rng);
            let width = random_side(&mut rng, root.get_width());
            let height = random_side(&mut rng, root.get_height());
            let bounds = Rectangle::center_rect(x, y, width, height);
            assert_eq!(
                tree.position(&bounds).0,
                level_by_division(&root, &world, &bounds),
                "HierarchicalGrid, world {world:?}, bounds {bounds:?}"
            );
        }
    }
}
