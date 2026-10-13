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

/// Steps of the dynamic check; each one moves, resizes, removes and inserts balls.
const STEPS: usize = 8;

fn world() -> Rectangle {
    Rectangle::new(-128.0, -128.0, BOUND_WIDTH - 128.0, BOUND_HEIGHT - 128.0)
}

/// A radius that doubles a few times at random, so that balls span many levels.
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

/// Every spatial structure of the crate, holding the same balls.
struct Trees {
    normal: QuadTree<usize, MAX_LEVEL, MAX_ITEMS>,
    loose: LooseQuadTree<usize, MAX_LEVEL, MAX_ITEMS>,
    grid_loose: GridLooseQuadTree<usize, MAX_LEVEL>,
    up_search: UpSearchQuadTree<usize, MAX_LEVEL>,
    original: UpSearchQuadTreeOriginal<usize, MAX_LEVEL>,
    uniform: UniformGrid<usize>,
    multi_layer: MultiLayerGrid<usize, MAX_LEVEL>,
}

/// The handles one ball received from every tree.
#[derive(Clone, Copy)]
struct Handles {
    normal: ItemHandle,
    loose: ItemHandle,
    grid_loose: ItemHandle,
    up_search: ItemHandle,
    original: ItemHandle,
    uniform: ItemHandle,
    multi_layer: ItemHandle,
}

impl Trees {
    fn new(world: &Rectangle) -> Self {
        Self {
            normal: QuadTree::new(world.clone()),
            loose: LooseQuadTree::new(world.clone()),
            grid_loose: GridLooseQuadTree::new(world.clone()),
            up_search: UpSearchQuadTree::new(world.clone()),
            original: UpSearchQuadTreeOriginal::new(world.clone()),
            uniform: UniformGrid::new(world.clone(), UNIFORM_CELL_SIZE),
            multi_layer: MultiLayerGrid::new(world.clone()),
        }
    }

    fn insert(&mut self, bounds: &Rectangle, id: usize) -> Handles {
        Handles {
            normal: self.normal.insert(bounds.clone(), id),
            loose: self.loose.insert(bounds.clone(), id),
            grid_loose: self.grid_loose.insert(bounds.clone(), id),
            up_search: self.up_search.insert(bounds.clone(), id),
            original: self.original.insert(bounds.clone(), id),
            uniform: self.uniform.insert(bounds.clone(), id),
            multi_layer: self.multi_layer.insert(bounds.clone(), id),
        }
    }

    fn update(&mut self, bounds: &Rectangle, handles: Handles) {
        self.normal.update(bounds.clone(), handles.normal);
        self.loose.update(bounds.clone(), handles.loose);
        self.grid_loose.update(bounds.clone(), handles.grid_loose);
        self.up_search.update(bounds.clone(), handles.up_search);
        self.original.update(bounds.clone(), handles.original);
        self.uniform.update(bounds.clone(), handles.uniform);
        self.multi_layer.update(bounds.clone(), handles.multi_layer);
    }

    fn remove(&mut self, handles: Handles) {
        self.normal.remove(handles.normal);
        self.loose.remove(handles.loose);
        self.grid_loose.remove(handles.grid_loose);
        self.up_search.remove(handles.up_search);
        self.original.remove(handles.original);
        self.uniform.remove(handles.uniform);
        self.multi_layer.remove(handles.multi_layer);
    }

    /// What every search method finds for `bounds`, excluding `id` itself.
    fn search(&self, bounds: &Rectangle, id: usize) -> Vec<(&'static str, HashSet<usize>)> {
        vec![
            ("QuadTree", collect(id, |cb| self.normal.search(bounds, cb))),
            (
                "LooseQuadTree",
                collect(id, |cb| self.loose.search(bounds, cb)),
            ),
            (
                "GridLooseQuadTree::search",
                collect(id, |cb| self.grid_loose.search(bounds, cb)),
            ),
            (
                "GridLooseQuadTree::search_up",
                collect(id, |cb| self.grid_loose.search_up(bounds, cb)),
            ),
            (
                "GridLooseQuadTree::search_bidirectional",
                collect(id, |cb| self.grid_loose.search_bidirectional(bounds, cb)),
            ),
            (
                "UpSearchQuadTree",
                collect(id, |cb| self.up_search.search(bounds, cb)),
            ),
            (
                "UpSearchQuadTreeOriginal",
                collect(id, |cb| self.original.search(bounds, cb)),
            ),
            (
                "UniformGrid",
                collect(id, |cb| self.uniform.search(bounds, cb)),
            ),
            (
                "MultiLayerGrid",
                collect(id, |cb| self.multi_layer.search(bounds, cb)),
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

/// Every search method must agree with a brute-force scan over the live balls.
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

/// Moving, resizing, removing and re-inserting balls through their handles
/// must leave every tree answering searches exactly like a brute-force scan.
#[test]
fn check_updates_and_removals() {
    let mut rng = StdRng::seed_from_u64(7);
    let world = world();
    let mut trees = Trees::new(&world);

    // `None` is a removed ball; the trees may hand its slot to a later insert.
    let mut balls: Vec<Option<(Ball, Handles)>> = Vec::new();
    for _ in 0..BALL_COUNT {
        let ball = random_ball(&mut rng, &world);
        let handles = trees.insert(&ball.get_bounds(), balls.len());
        balls.push(Some((ball, handles)));
    }

    for _ in 0..STEPS {
        // Move every ball; resizing some of them changes their level or node.
        for (ball, handles) in balls.iter_mut().flatten() {
            ball.update(&world);
            if rng.gen_bool(0.05) {
                ball.r = random_radius(&mut rng);
            }
            trees.update(&ball.get_bounds(), *handles);
        }

        // Remove some balls and add new ones, which take over the freed slots.
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

        assert_searches_match_brute_force(&trees, &balls);
    }
}
