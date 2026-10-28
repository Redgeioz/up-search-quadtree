use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use rand::*;
use std::hint::black_box;

use lib::ball::*;
use lib::rect::*;
use lib::*;

mod common;
use common::*;

/// Half the finest cell size, which the small balls still fit.
const HALF_CELL_SIZE: f64 = GRID_CELL_SIZE / 2.0;

/// The bounds of `AMOUNT` balls at random points, with radii drawn from
/// `MIN_RADIUS..MAX_RADIUS` that double at most `doublings` times, each time
/// with a chance of a quarter.
fn ball_bounds(doublings: usize) -> Vec<Rectangle> {
    ball_bounds_in(&BOUNDS, AMOUNT, doublings)
}

/// [`ball_bounds`] for `count` balls in `world`.
fn ball_bounds_in(world: &Rectangle, count: usize, doublings: usize) -> Vec<Rectangle> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    (0..count)
        .map(|_| {
            let (x, y) = world.get_random_point(&mut rng);
            let mut r = rng.gen_range(MIN_RADIUS..MAX_RADIUS);
            for _ in 0..doublings {
                if rng.gen_bool(0.25) {
                    r *= 2.0;
                } else {
                    break;
                }
            }
            let ball = BallBuilder::new().set_position(x, y).set_radius(r).build();
            ball.get_bounds()
        })
        .collect()
}

/// Insert the balls of [`ball_bounds`] into `tree`, numbered in order, and
/// return their bounds.
fn gen_balls(tree: &mut impl QTree<usize>, doublings: usize) -> Vec<Rectangle> {
    let bounds_cache = ball_bounds(doublings);
    for (id, bounds) in bounds_cache.iter().enumerate() {
        tree.insert_item(bounds.clone(), id);
    }
    bounds_cache
}

/// Search with every ball's bounds among balls that double up to five times.
fn bench_search<Tree: QTree<usize>>(c: &mut Criterion, name: &str, mut tree: Tree) {
    let bounds_cache = gen_balls(&mut tree, 5);
    // In the lazy overflow bound grids, insertion alone leaves boxes, spans
    // and bounds exact and refreshing changes nothing, except that in
    // `HaloOverflowBoundGrid` only the refresh fills the copies of the
    // neighbors' sides and clears the stale marks.
    tree.end_frame();

    c.bench_function(name, |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                tree.search_items(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn search(c: &mut Criterion) {
    macro_rules! arm {
        ($t:ident $(, $cell:expr)?) => {
            bench_search(c, stringify!($t), tree!($t $(, $cell)?))
        };
    }
    arm!(LooseQuadTree);
    arm!(HierarchicalLooseGrid);
    arm!(UniformGrid, GRID_CELL_SIZE);
    arm!(TwoLayerGrid, GRID_CELL_SIZE);
    arm!(MultiGrid, &MULTI_CELL_SIZES);
    bench_search(c, "MultiGrid/2", tree!(MultiGrid, &MULTI_CELL_SIZES_2));
    arm!(HierarchicalGrid);
    arm!(ExactOverflowBoundGrid);
    arm!(BoxedLooseGrid);
    arm!(LazyOverflowBoundGrid);
    arm!(SpanOverflowBoundGrid);
    arm!(HaloOverflowBoundGrid);
    arm!(SpanGrid, GRID_CELL_SIZE);
    arm!(HaloGrid, GRID_CELL_SIZE);
}

/// Search with every ball's bounds among balls that never double. All fit far
/// below the finest cell size, where overflow bounds narrow the scan.
fn bench_small_balls<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut tree: Tree,
) {
    let bounds_cache = gen_balls(&mut tree, 0);
    tree.end_frame();

    group.bench_function(name, |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                tree.search_items(bounds, |id| {
                    black_box(id);
                })
            });
        });
    });
}

fn small_balls(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_balls");
    macro_rules! arm {
        ($t:ident) => {
            bench_small_balls(&mut group, stringify!($t), tree!($t))
        };
        ($t:ident / $label:expr, $cell:expr) => {
            bench_small_balls(
                &mut group,
                &format!("{}/{}", stringify!($t), $label),
                tree!($t, $cell),
            )
        };
    }
    arm!(HierarchicalLooseGrid);
    arm!(ExactOverflowBoundGrid);
    arm!(BoxedLooseGrid);
    arm!(LazyOverflowBoundGrid);
    arm!(SpanOverflowBoundGrid);
    arm!(HaloOverflowBoundGrid);
    for (name, cell) in [("312", GRID_CELL_SIZE), ("156", HALF_CELL_SIZE)] {
        bench_small_balls(&mut group, &format!("MultiGrid/{name}"), tree!(MultiGrid, &[cell]));
        arm!(UniformGrid / name, cell);
        arm!(TwoLayerGrid / name, cell);
        arm!(SpanGrid / name, cell);
        arm!(HaloGrid / name, cell);
    }
    group.finish();
}

/// Items that keep off their cell edges: two per finest cell, near its center
/// and well below its size, so no item reaches the edges of its cell.
fn gen_aligned(tree: &mut impl QTree<usize>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let cells = (BOUND_WIDTH / GRID_CELL_SIZE) as usize;
    let mut id = 0;
    for y in 0..cells {
        for x in 0..cells {
            for _ in 0..2 {
                let cx = (x as f64 + 0.5) * GRID_CELL_SIZE + rng.gen_range(-30.0..30.0);
                let cy = (y as f64 + 0.5) * GRID_CELL_SIZE + rng.gen_range(-30.0..30.0);
                let w = rng.gen_range(60.0..180.0);
                let h = rng.gen_range(60.0..180.0);
                tree.insert_item(Rectangle::center_rect(cx, cy, w, h), id);
                id += 1;
            }
        }
    }
}

/// Query rectangles anywhere in the world, with sides drawn from `sizes`.
fn aligned_queries(count: usize, sizes: std::ops::Range<f64>, seed: u64) -> Vec<Rectangle> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let (x, y) = BOUNDS.get_random_point(&mut rng);
            let w = rng.gen_range(sizes.clone());
            let h = rng.gen_range(sizes.clone());
            Rectangle::center_rect(x, y, w, h)
        })
        .collect()
}

fn bench_aligned<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut tree: Tree,
) {
    gen_aligned(&mut tree);
    tree.end_frame();
    let small = aligned_queries(AMOUNT, 50.0..300.0, 1);
    let large = aligned_queries(AMOUNT / 5, 400.0..1500.0, 2);
    // Shared by both loops, so that they time one and the same search routine.
    let search = |tree: &Tree, bounds: &Rectangle| {
        tree.search_items(bounds, |id| {
            black_box(id);
        })
    };
    group.bench_function(format!("small/{name}"), |b| {
        b.iter(|| small.iter().for_each(|bounds| search(&tree, bounds)));
    });
    group.bench_function(format!("large/{name}"), |b| {
        b.iter(|| large.iter().for_each(|bounds| search(&tree, bounds)));
    });
}

fn aligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("aligned");
    macro_rules! arm {
        ($t:ident) => {
            bench_aligned(&mut group, stringify!($t), tree!($t))
        };
        ($t:ident / $label:expr, $cell:expr) => {
            bench_aligned(
                &mut group,
                &format!("{}/{}", stringify!($t), $label),
                tree!($t, $cell),
            )
        };
    }
    arm!(HierarchicalLooseGrid);
    arm!(ExactOverflowBoundGrid);
    arm!(BoxedLooseGrid);
    arm!(LazyOverflowBoundGrid);
    arm!(SpanOverflowBoundGrid);
    arm!(HaloOverflowBoundGrid);
    arm!(UniformGrid / "312", GRID_CELL_SIZE);
    arm!(TwoLayerGrid / "312", GRID_CELL_SIZE);
    arm!(SpanGrid / "312", GRID_CELL_SIZE);
    arm!(HaloGrid / "312", GRID_CELL_SIZE);
    group.finish();
}

/// Small balls on the finest level and `count` squares on one coarser level.
/// The width and height of each square are drawn independently from `side`,
/// so the squares are in general rectangles.
fn gen_two_tier(count: usize, side: std::ops::Range<f64>) -> (Vec<Rectangle>, Vec<Rectangle>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED);
    let small = (0..AMOUNT)
        .map(|_| {
            let (x, y) = BOUNDS.get_random_point(&mut rng);
            let r = rng.gen_range(MIN_RADIUS..MAX_RADIUS);
            Rectangle::center_rect(x, y, r * 2.0, r * 2.0)
        })
        .collect();
    let large = (0..count)
        .map(|_| {
            let (x, y) = BOUNDS.get_random_point(&mut rng);
            let w = rng.gen_range(side.clone());
            let h = rng.gen_range(side.clone());
            Rectangle::center_rect(x, y, w, h)
        })
        .collect();
    (small, large)
}

fn bench_two_tier<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut tree: Tree,
    (small, large): &(Vec<Rectangle>, Vec<Rectangle>),
) {
    for (id, bounds) in small.iter().chain(large).enumerate() {
        tree.insert_item(bounds.clone(), id);
    }
    tree.end_frame();
    // Shared by both loops, so that they time one and the same search routine.
    let search = |tree: &Tree, bounds: &Rectangle| {
        tree.search_items(bounds, |id| {
            black_box(id);
        })
    };
    group.bench_function(format!("small/{name}"), |b| {
        b.iter(|| small.iter().for_each(|bounds| search(&tree, bounds)));
    });
    group.bench_function(format!("large/{name}"), |b| {
        b.iter(|| large.iter().for_each(|bounds| search(&tree, bounds)));
    });
}

/// Search with the balls and with the squares of a two-tier scene, then with
/// the balls after the squares have shrunk onto the finest level without a
/// refresh. This empties the squares' level, a state that lazy structures
/// tidy up only in a refresh.
fn two_tier(c: &mut Criterion, name: &str, scene: (Vec<Rectangle>, Vec<Rectangle>)) {
    let mut group = c.benchmark_group(name);
    macro_rules! arm {
        ($bench:ident, $t:ident) => {
            $bench(&mut group, stringify!($t), tree!($t), &scene)
        };
    }
    arm!(bench_two_tier, HierarchicalLooseGrid);
    arm!(bench_two_tier, ExactOverflowBoundGrid);
    arm!(bench_two_tier, BoxedLooseGrid);
    arm!(bench_two_tier, LazyOverflowBoundGrid);
    arm!(bench_two_tier, SpanOverflowBoundGrid);
    arm!(bench_two_tier, HaloOverflowBoundGrid);

    arm!(bench_stale, BoxedLooseGrid);
    arm!(bench_stale, LazyOverflowBoundGrid);
    arm!(bench_stale, SpanOverflowBoundGrid);
    arm!(bench_stale, HaloOverflowBoundGrid);
    group.finish();
}

/// Search with the balls after the squares have shrunk onto the finest level
/// without a refresh, which empties their level.
fn bench_stale<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut grid: Tree,
    (small, large): &(Vec<Rectangle>, Vec<Rectangle>),
) {
    for (id, bounds) in small.iter().enumerate() {
        grid.insert_item(bounds.clone(), id);
    }
    let handles: Vec<_> = large
        .iter()
        .enumerate()
        .map(|(id, bounds)| grid.insert_item(bounds.clone(), AMOUNT + id))
        .collect();
    for (bounds, handle) in large.iter().zip(handles) {
        let (x, y) = bounds.get_center();
        grid.update_item(Rectangle::center_rect(x, y, 80.0, 80.0), handle);
    }
    group.bench_function(format!("stale/{name}"), |b| {
        b.iter(|| {
            small.iter().for_each(|bounds| {
                grid.search_items(bounds, |id| {
                    black_box(id);
                })
            })
        });
    });
}

/// Search with the balls of the search distribution after every `stride`th
/// of them has moved 1 to 5 units in some direction since the last refresh,
/// as queries do between the updates of a frame and its refresh.
fn bench_moving<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut grid: Tree,
    (label, stride): (&str, usize),
) {
    let mut bounds = ball_bounds(5);
    let handles: Vec<_> = bounds
        .iter()
        .enumerate()
        .map(|(id, bounds)| grid.insert_item(bounds.clone(), id))
        .collect();
    grid.end_frame();
    let mut rng = rand::rngs::StdRng::seed_from_u64(SEED + 1);
    for (bounds, handle) in bounds.iter_mut().zip(handles).step_by(stride) {
        let angle = rng.gen_range(0.0..std::f64::consts::TAU);
        let distance = rng.gen_range(1.0..5.0);
        let (x, y) = bounds.get_center();
        *bounds = Rectangle::center_rect(
            (x + distance * angle.cos()).clamp(0.0, BOUND_WIDTH.next_down()),
            (y + distance * angle.sin()).clamp(0.0, BOUND_HEIGHT.next_down()),
            bounds.get_width(),
            bounds.get_height(),
        );
        grid.update_item(bounds.clone(), handle);
    }
    group.bench_function(format!("{label}/{name}"), |b| {
        b.iter(|| {
            bounds.iter().for_each(|bounds| {
                grid.search_items(bounds, |id| {
                    black_box(id);
                })
            })
        });
    });
}

/// The search distribution with a tenth of the balls, or all of them, moved
/// since the last refresh.
fn moving(c: &mut Criterion) {
    let mut group = c.benchmark_group("moving");
    for scene in [("tenth", 10), ("all", 1)] {
        macro_rules! arm {
            ($t:ident) => {
                bench_moving(&mut group, stringify!($t), tree!($t), scene)
            };
        }
        arm!(BoxedLooseGrid);
        arm!(LazyOverflowBoundGrid);
        arm!(SpanOverflowBoundGrid);
        arm!(HaloOverflowBoundGrid);
    }
    group.finish();
}

/// 64 squares on level 3, with the levels between them and the balls empty:
/// the ancestor walk from the finest level passes three empty levels before
/// it reaches the squares.
fn gap(c: &mut Criterion) {
    two_tier(c, "gap", gen_two_tier(64, 2600.0..4900.0));
}

/// 1,024 squares on level 6, about one per cell, directly above the balls:
/// the ancestor walk from the finest level reads one more level and skips
/// none.
fn adjacent(c: &mut Criterion) {
    two_tier(c, "adjacent", gen_two_tier(1024, 320.0..620.0));
}

/// Search with the bounds of every item of the distribution: one search per
/// item in the items' order, which is random in space, against one self
/// query, which takes the items cell by cell.
fn bench_self_query<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut tree: Tree,
    self_query: impl Fn(&Tree),
) {
    let bounds_cache = gen_balls(&mut tree, 5);
    tree.end_frame();
    group.bench_function(format!("loop/{name}"), |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                tree.search_items(bounds, |id| {
                    black_box(id);
                })
            })
        });
    });
    group.bench_function(format!("batch/{name}"), |b| b.iter(|| self_query(&tree)));
}

fn self_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("self_query");
    macro_rules! arm {
        ($t:ident $(, $cell:expr)?) => {
            bench_self_query(&mut group, stringify!($t), tree!($t $(, $cell)?), |tree| {
                tree.for_each_self_query(|query, id| {
                    black_box((query, id));
                })
            })
        };
    }
    arm!(ExactOverflowBoundGrid);
    arm!(BoxedLooseGrid);
    arm!(LazyOverflowBoundGrid);
    arm!(SpanOverflowBoundGrid);
    arm!(HaloOverflowBoundGrid);
    arm!(HierarchicalLooseGrid);
    arm!(HierarchicalGrid);
    arm!(UniformGrid, GRID_CELL_SIZE);
    arm!(TwoLayerGrid, GRID_CELL_SIZE);
    arm!(SpanGrid, GRID_CELL_SIZE);
    arm!(HaloGrid, GRID_CELL_SIZE);
    group.finish();
}

/// Report every intersecting pair of the balls of [`ball_bounds`], doubled up
/// to `doublings` times, once, with each structure's pair rules; the pair
/// count, printed once, must agree.
fn bench_pairs<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    tree: Tree,
    doublings: usize,
    pairs: impl Fn(&Tree, &mut dyn FnMut(usize, usize)),
) {
    bench_pairs_on(group, name, tree, &ball_bounds(doublings), pairs);
}

/// [`bench_pairs`] for the items `bounds`, numbered in order.
fn bench_pairs_on<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut tree: Tree,
    bounds: &[Rectangle],
    pairs: impl Fn(&Tree, &mut dyn FnMut(usize, usize)),
) {
    for (id, bounds) in bounds.iter().enumerate() {
        tree.insert_item(bounds.clone(), id);
    }
    tree.end_frame();
    let mut count = 0usize;
    pairs(&tree, &mut |_, _| count += 1);
    eprintln!("{name}: {count} pairs");
    group.bench_function(name, |b| {
        b.iter(|| {
            pairs(&tree, &mut |a, b| {
                black_box((a, b));
            })
        })
    });
}

fn pairs(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairs");
    macro_rules! arm {
        ($name:expr, $tree:expr) => {
            bench_pairs(&mut group, $name, $tree, 5, |tree, callback| {
                tree.for_each_pair(|a, b| callback(a, b))
            })
        };
    }
    arm!("ExactOverflowBoundGrid", tree!(ExactOverflowBoundGrid));
    arm!("LazyOverflowBoundGrid", tree!(LazyOverflowBoundGrid));
    arm!("SpanOverflowBoundGrid", tree!(SpanOverflowBoundGrid));
    arm!("HaloOverflowBoundGrid", tree!(HaloOverflowBoundGrid));
    arm!("MultiGrid", tree!(MultiGrid, &MULTI_CELL_SIZES));
    arm!("MultiGrid/2", tree!(MultiGrid, &MULTI_CELL_SIZES_2));
    arm!("HierarchicalLooseGrid", tree!(HierarchicalLooseGrid));
    arm!("BoxedLooseGrid", tree!(BoxedLooseGrid));
    arm!("HierarchicalGrid", tree!(HierarchicalGrid));
    arm!("UniformGrid", tree!(UniformGrid, GRID_CELL_SIZE));
    arm!("TwoLayerGrid", tree!(TwoLayerGrid, GRID_CELL_SIZE));
    arm!("SpanGrid", tree!(SpanGrid, GRID_CELL_SIZE));
    arm!("HaloGrid", tree!(HaloGrid, GRID_CELL_SIZE));
    arm!("LooseQuadTree", tree!(LooseQuadTree));
    arm!("SortAndSweep", SortAndSweep::new());
    bench_pairs(&mut group, "SortAndSweep/scalar", SortAndSweep::new(), 5, |tree, callback| {
        tree.for_each_pair_scalar(|a, b| callback(a, b))
    });
    arm!("SortAndSweep/unsorted", UnsortedSweep(SortAndSweep::new()));
    group.finish();
}

/// [`pairs`] in a scene sixteen times as large at the same density, for the
/// grids compared with the sweep, whose candidates on x grow with the number
/// of items per column of the world.
fn pairs_160k(c: &mut Criterion) {
    let mut group = c.benchmark_group("pairs_160k");
    group.sample_size(10);
    let bounds = ball_bounds_in(&LARGE_BOUNDS, LARGE_AMOUNT, 5);
    macro_rules! arm {
        ($name:expr, $tree:expr) => {
            bench_pairs_on(&mut group, $name, $tree, &bounds, |tree, callback| {
                tree.for_each_pair(|a, b| callback(a, b))
            })
        };
    }
    arm!("HierarchicalLooseGrid", HierarchicalLooseGrid::<usize, LARGE_MAX_LEVEL>::new(LARGE_BOUNDS.clone()));
    arm!("HaloOverflowBoundGrid", HaloOverflowBoundGrid::<usize, LARGE_MAX_LEVEL>::new(LARGE_BOUNDS.clone()));
    arm!("UniformGrid", UniformGrid::<usize>::new(LARGE_BOUNDS.clone(), GRID_CELL_SIZE));
    arm!("TwoLayerGrid", TwoLayerGrid::<usize>::new(LARGE_BOUNDS.clone(), GRID_CELL_SIZE));
    arm!("SortAndSweep", SortAndSweep::new());
    bench_pairs_on(&mut group, "SortAndSweep/scalar", SortAndSweep::new(), &bounds, |tree, callback| {
        tree.for_each_pair_scalar(|a, b| callback(a, b))
    });
    arm!("SortAndSweep/unsorted", UnsortedSweep(SortAndSweep::new()));
    group.finish();
}

/// The pairs of the small balls, which all fit a cell, in the single-level
/// grids with cells of 156.25 units; `MultiGrid` with one collection is a
/// single-level loose grid without bounds.
fn small_pairs(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_pairs");
    macro_rules! arm {
        ($name:expr, $tree:expr) => {
            bench_pairs(&mut group, $name, $tree, 0, |tree, callback| {
                tree.for_each_pair(|a, b| callback(a, b))
            })
        };
    }
    arm!("MultiGrid/156", tree!(MultiGrid, &[HALF_CELL_SIZE]));
    arm!("UniformGrid/156", tree!(UniformGrid, HALF_CELL_SIZE));
    arm!("TwoLayerGrid/156", tree!(TwoLayerGrid, HALF_CELL_SIZE));
    arm!("SpanGrid/156", tree!(SpanGrid, HALF_CELL_SIZE));
    arm!("HaloGrid/156", tree!(HaloGrid, HALF_CELL_SIZE));
    group.finish();
}

criterion_group!(
    bench,
    search,
    small_balls,
    aligned,
    gap,
    adjacent,
    moving,
    self_query,
    pairs,
    small_pairs,
    pairs_160k,
);
criterion_main!(bench);
