use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use rand::*;

use lib::ball::*;
use lib::rect::*;
use lib::*;

mod common;
use common::*;

const MIN_VELOCITY: f64 = 1.0;
const MAX_VELOCITY: f64 = 5.0;

/// `count` moving balls at random points, with radii drawn from
/// `MIN_RADIUS..MAX_RADIUS` that double at most `doublings` times, each time
/// with a chance of a quarter, inserted into `quadtree`.
fn gen_balls(
    quadtree: &mut impl QTree<usize>,
    count: usize,
    seed: u64,
    doublings: usize,
) -> (Vec<Ball>, Vec<ItemHandle>) {
    gen_balls_in(quadtree, &BOUNDS, count, seed, doublings)
}

/// [`gen_balls`] at random points of `world`.
fn gen_balls_in(
    quadtree: &mut impl QTree<usize>,
    world: &Rectangle,
    count: usize,
    seed: u64,
    doublings: usize,
) -> (Vec<Ball>, Vec<ItemHandle>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut balls = Vec::with_capacity(count);
    let mut handles = Vec::with_capacity(count);
    for _ in 0..count {
        let (x, y) = world.get_random_point(&mut rng);
        let mut r = rng.gen_range(MIN_RADIUS..MAX_RADIUS);
        for _ in 0..doublings {
            if rng.gen_bool(0.25) {
                r *= 2.0;
            } else {
                break;
            }
        }
        let v = rng.gen_range(MIN_VELOCITY..MAX_VELOCITY);
        let angle = rng.gen_range(0.0..std::f64::consts::TAU);

        let ball = BallBuilder::new()
            .set_position(x, y)
            .set_radius(r)
            .set_velocity(v * angle.cos(), v * angle.sin())
            .build();

        let id = balls.len();
        handles.push(quadtree.insert_item(ball.get_bounds(), id));
        balls.push(ball);
    }

    (balls, handles)
}

/// Move every ball one step and update its bounds in the tree.
fn bench_update(group: &mut BenchmarkGroup<WallTime>, name: &str, tree: impl QTree<usize>) {
    bench_update_balls(group, name, tree, 5);
}

/// [`bench_update`] for balls that double at most `doublings` times.
fn bench_update_balls(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut tree: impl QTree<usize>,
    doublings: usize,
) {
    let (mut balls, handles) = gen_balls(&mut tree, AMOUNT, SEED, doublings);

    group.bench_function(name, |b| {
        b.iter(|| {
            balls.iter_mut().zip(&handles).for_each(|(ball, handle)| {
                ball.update(&BOUNDS);
                tree.update_item(ball.get_bounds(), *handle);
            });
            tree.end_frame();
        });
    });
}

/// Remove every ball and reinsert it at its current position.
fn bench_reinsert(group: &mut BenchmarkGroup<WallTime>, name: &str, mut tree: impl QTree<usize>) {
    let (balls, mut handles) = gen_balls(&mut tree, AMOUNT, SEED, 5);

    group.bench_function(name, |b| {
        b.iter(|| {
            balls
                .iter()
                .zip(handles.iter_mut())
                .enumerate()
                .for_each(|(id, (ball, handle))| {
                    tree.remove_item(*handle);
                    *handle = tree.insert_item(ball.get_bounds(), id);
                });
            tree.end_frame();
        });
    });
}

/// Runs `$bench` on every structure, in the group `$name`. In the list, a
/// cell size marks a single-layer grid.
macro_rules! bench_all {
    ($c:expr, $name:literal, $bench:ident) => {
        bench_all!($c, $name, $bench, [
            LooseQuadTree,
            HierarchicalLooseGrid,
            ExactOverflowBoundGrid,
            BoxedLooseGrid,
            LazyOverflowBoundGrid,
            SpanOverflowBoundGrid,
            HaloOverflowBoundGrid,
            SpanGrid(GRID_CELL_SIZE),
            HaloGrid(GRID_CELL_SIZE),
            UniformGrid(GRID_CELL_SIZE),
            TwoLayerGrid(GRID_CELL_SIZE),
            MultiGrid(&MULTI_CELL_SIZES),
            HierarchicalGrid,
        ])
    };
    ($c:expr, $name:literal, $bench:ident, [$($t:ident $(($cell:expr))?),* $(,)?]) => {{
        let mut group = $c.benchmark_group($name);
        $($bench(&mut group, stringify!($t), tree!($t $(, $cell)?));)*
        group.finish();
    }};
}

fn update(c: &mut Criterion) {
    bench_all!(c, "update", bench_update);
    let mut group = c.benchmark_group("update");
    bench_update(&mut group, "MultiGrid/2", tree!(MultiGrid, &MULTI_CELL_SIZES_2));
    bench_update(&mut group, "SortAndSweep", SortAndSweep::new());
    bench_update(&mut group, "SortAndSweep/unsorted", UnsortedSweep(SortAndSweep::new()));
    group.finish();
}

/// [`bench_update`] for [`LARGE_AMOUNT`] balls in [`LARGE_BOUNDS`].
fn bench_update_large(group: &mut BenchmarkGroup<WallTime>, name: &str, mut tree: impl QTree<usize>) {
    let (mut balls, handles) = gen_balls_in(&mut tree, &LARGE_BOUNDS, LARGE_AMOUNT, SEED, 5);

    group.bench_function(name, |b| {
        b.iter(|| {
            balls.iter_mut().zip(&handles).for_each(|(ball, handle)| {
                ball.update(&LARGE_BOUNDS);
                tree.update_item(ball.get_bounds(), *handle);
            });
            tree.end_frame();
        });
    });
}

/// [`update`] in the scene of `pairs_160k`, sixteen times as large at the same
/// density, for the structures compared there.
fn update_160k(c: &mut Criterion) {
    let mut group = c.benchmark_group("update_160k");
    group.sample_size(20);
    macro_rules! arm {
        ($name:expr, $tree:expr) => {
            bench_update_large(&mut group, $name, $tree)
        };
    }
    arm!("HierarchicalLooseGrid", HierarchicalLooseGrid::<usize, LARGE_MAX_LEVEL>::new(LARGE_BOUNDS.clone()));
    arm!("HaloOverflowBoundGrid", HaloOverflowBoundGrid::<usize, LARGE_MAX_LEVEL>::new(LARGE_BOUNDS.clone()));
    arm!("UniformGrid", UniformGrid::<usize>::new(LARGE_BOUNDS.clone(), GRID_CELL_SIZE));
    arm!("TwoLayerGrid", TwoLayerGrid::<usize>::new(LARGE_BOUNDS.clone(), GRID_CELL_SIZE));
    arm!("SortAndSweep", SortAndSweep::new());
    arm!("SortAndSweep/unsorted", UnsortedSweep(SortAndSweep::new()));
    group.finish();
}

fn reinsert(c: &mut Criterion) {
    bench_all!(c, "reinsert", bench_reinsert);
}

/// Time only the updates, and the `end_frame` after each pass, on identical
/// precomputed transitions. Each iteration goes A -> B -> A, so every
/// structure starts and ends each iteration with the same bounds, whatever
/// iteration count Criterion chooses for it. The reported duration covers
/// 2 * AMOUNT updates; ball movement, bounds construction, insertion and
/// random-number generation lie outside the timer.
fn bench_update_pair(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut tree: impl QTree<usize>,
    from: &[Rectangle],
    to: &[Rectangle],
) {
    let handles: Vec<_> = from
        .iter()
        .enumerate()
        .map(|(id, bounds)| tree.insert_item(bounds.clone(), id))
        .collect();
    group.bench_function(name, |b| {
        b.iter(|| {
            for bounds in [black_box(to), black_box(from)] {
                for (bounds, handle) in bounds.iter().zip(&handles) {
                    tree.update_item(bounds.clone(), *handle);
                }
                tree.end_frame();
            }
            black_box(&tree);
        });
    });
}

fn update_compare(c: &mut Criterion) {
    let mut baseline = tree!(HierarchicalLooseGrid);
    let (mut balls, _) = gen_balls(&mut baseline, AMOUNT, SEED, 5);
    let from: Vec<_> = balls.iter().map(Ball::get_bounds).collect();
    for ball in &mut balls {
        ball.update(&BOUNDS);
    }
    let moved: Vec<_> = balls.iter().map(Ball::get_bounds).collect();
    let cell_size = GRID_CELL_SIZE;
    let crossed: Vec<_> = moved
        .iter()
        .map(|bounds| {
            let (x, y) = bounds.get_center();
            Rectangle::center_rect(
                (x + cell_size).rem_euclid(BOUND_WIDTH),
                (y + cell_size).rem_euclid(BOUND_HEIGHT),
                bounds.get_width(),
                bounds.get_height(),
            )
        })
        .collect();
    let resized: Vec<_> = moved
        .iter()
        .enumerate()
        .map(|(id, bounds)| bounds.scale([0.25, 0.5, 2.0, 4.0][id % 4]))
        .collect();
    let mut group = c.benchmark_group("update_compare");
    group.throughput(Throughput::Elements((2 * AMOUNT) as u64));
    for (scenario, to) in [
        ("small_move", &moved),
        ("cell_move", &crossed),
        ("move_resize", &resized),
    ] {
        let mut same_cell = 0;
        let mut changed_cell = 0;
        let mut changed_level = 0;
        for (a, b) in from.iter().zip(to) {
            let a = baseline.position(a);
            let b = baseline.position(b);
            if a == b {
                same_cell += 1;
            } else if a.0 == b.0 {
                changed_cell += 1;
            } else {
                changed_level += 1;
            }
        }
        eprintln!(
            "{scenario}: same_cell={same_cell}, changed_cell={changed_cell}, changed_level={changed_level}"
        );
        macro_rules! arm {
            ($t:ident $(, $cell:expr)?) => {
                let name = format!("{scenario}/{}", stringify!($t));
                bench_update_pair(&mut group, &name, tree!($t $(, $cell)?), &from, to);
            };
        }
        arm!(HierarchicalLooseGrid);
        arm!(ExactOverflowBoundGrid);
        arm!(BoxedLooseGrid);
        arm!(LazyOverflowBoundGrid);
        arm!(SpanOverflowBoundGrid);
        arm!(HaloOverflowBoundGrid);
        arm!(UniformGrid, GRID_CELL_SIZE);
        arm!(TwoLayerGrid, GRID_CELL_SIZE);
        arm!(SpanGrid, GRID_CELL_SIZE);
        arm!(HaloGrid, GRID_CELL_SIZE);
    }
    group.finish();
}

/// Cell size of the single-level grids for the small balls, which all fit it.
const HALF_CELL_SIZE: f64 = GRID_CELL_SIZE / 2.0;

/// [`update`] for balls that never double, all fitting a cell of
/// [`HALF_CELL_SIZE`], in the single-level structures; `MultiGrid` with one
/// collection is a single-level loose grid without bounds.
fn small_update(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_update");
    macro_rules! arm {
        ($name:expr, $tree:expr) => {
            bench_update_balls(&mut group, $name, $tree, 0)
        };
    }
    arm!("MultiGrid/156", tree!(MultiGrid, &[HALF_CELL_SIZE]));
    arm!("UniformGrid/156", tree!(UniformGrid, HALF_CELL_SIZE));
    arm!("TwoLayerGrid/156", tree!(TwoLayerGrid, HALF_CELL_SIZE));
    arm!("SpanGrid/156", tree!(SpanGrid, HALF_CELL_SIZE));
    arm!("HaloGrid/156", tree!(HaloGrid, HALF_CELL_SIZE));
    group.finish();
}

criterion_group!(bench, update, reinsert, update_compare, small_update, update_160k);
criterion_main!(bench);
