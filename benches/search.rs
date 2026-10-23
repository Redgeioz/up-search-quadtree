use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use rand::*;
use std::hint::black_box;

use lib::ball::*;
use lib::rect::*;
use lib::*;

const MAX_LEVEL: u8 = 7;
const MAX_ITEMS: usize = 8;

const BOUND_WIDTH: f64 = 20000.0;
const BOUND_HEIGHT: f64 = 20000.0;

const AMOUNT: usize = 10000;
const SEED: u64 = 0;

// Diameters of 50 to 100, the size of levels 8 to 9; with `MAX_LEVEL` 7,
// balls that do not double sit on the finest level.
const MIN_RADIUS: f64 = 25.0;
const MAX_RADIUS: f64 = 50.0;

const BOUNDS: Rectangle = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);

/// Cell size of the uniform grids: the finest cell size of the hierarchical
/// structures, 312.5.
const GRID_CELL_SIZE: f64 = BOUND_WIDTH / (1u32 << (MAX_LEVEL - 1)) as f64;

trait QTree<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T);
    /// Called once after a scene is built, like the end of a frame, so that
    /// lazy structures bring their bounds up to date.
    fn end_frame(&mut self) {}
}

impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> QTree<T>
    for LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for HierarchicalLooseGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for OverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for LazyOverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for BoxedOverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for SpanOverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for HaloOverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

impl<T: Copy> QTree<T> for UniformGrid<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for HierarchicalGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy> QTree<T> for SpanGrid<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

impl<T: Copy> QTree<T> for HaloGrid<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

/// Half the finest cell size, which the small balls still fit.
const HALF_CELL_SIZE: f64 = GRID_CELL_SIZE / 2.0;

fn gen_balls(quadtree: &mut impl QTree<usize>, count: usize, seed: u64) -> Vec<Rectangle> {
    gen_balls_doubling(quadtree, count, seed, 5)
}

/// Like [`gen_balls`], but the radius doubles at most `doublings` times.
fn gen_balls_doubling(
    quadtree: &mut impl QTree<usize>,
    count: usize,
    seed: u64,
    doublings: usize,
) -> Vec<Rectangle> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut balls = Vec::with_capacity(count);
    for _ in 0..count {
        let (x, y) = BOUNDS.get_random_point(&mut rng);
        let mut r = rng.gen_range(MIN_RADIUS..MAX_RADIUS);
        for _ in 0..doublings {
            if rng.gen_bool(0.25) {
                r *= 2.0;
            } else {
                break;
            }
        }

        let ball = BallBuilder::new().set_position(x, y).set_radius(r).build();

        let id = balls.len();
        let bounds = ball.get_bounds();
        balls.push(ball);
        quadtree.insert_item(bounds.clone(), id);
    }

    balls
        .iter()
        .map(|ball| ball.get_bounds())
        .collect::<Vec<_>>()
}

fn quadtree_loose(c: &mut Criterion) {
    let mut quadtree_loose = LooseQuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut quadtree_loose, AMOUNT, SEED);

    c.bench_function("LooseQuadTree", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_loose.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn grid_hierarchical_loose(c: &mut Criterion) {
    let mut grid_hierarchical_loose =
        HierarchicalLooseGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut grid_hierarchical_loose, AMOUNT, SEED);

    c.bench_function("HierarchicalLooseGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid_hierarchical_loose.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn grid_overflow_exact(c: &mut Criterion) {
    let mut grid = OverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);

    c.bench_function("OverflowBoundGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn grid_overflow_lazy(c: &mut Criterion) {
    let mut grid = LazyOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);
    // Insertion alone leaves the bounds exact; refreshing changes nothing.
    grid.refresh_bounds();

    c.bench_function("LazyOverflowBoundGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn grid_overflow_boxed(c: &mut Criterion) {
    let mut grid = BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);
    // Insertion alone leaves boxes and bounds exact; refreshing changes nothing.
    grid.refresh_bounds();

    c.bench_function("BoxedOverflowBoundGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn grid_overflow_span(c: &mut Criterion) {
    let mut grid = SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);
    // Insertion alone leaves boxes and spans exact; refreshing changes nothing.
    grid.refresh_bounds();

    c.bench_function("SpanOverflowBoundGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn grid_overflow_halo(c: &mut Criterion) {
    let mut grid = HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);
    // Insertion leaves boxes and spans exact, but only the refresh fills the
    // copies of the neighbors' sides and clears the stale marks.
    grid.refresh_bounds();

    c.bench_function("HaloOverflowBoundGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn span_grid(c: &mut Criterion) {
    let mut grid = SpanGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE);
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);
    grid.refresh_bounds();

    c.bench_function("SpanGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn halo_grid(c: &mut Criterion) {
    let mut grid = HaloGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE);
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);
    grid.refresh_bounds();

    c.bench_function("HaloGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn grid_uniform(c: &mut Criterion) {
    let mut grid = UniformGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE);
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);

    c.bench_function("UniformGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn grid_hierarchical(c: &mut Criterion) {
    let mut grid = HierarchicalGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut grid, AMOUNT, SEED);

    c.bench_function("HierarchicalGrid", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

/// Search with every ball's bounds among balls that never double. All fit far
/// below the finest cell size, where overflow bounds narrow the scan.
fn bench_small_balls<Tree: QTree<usize>>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut tree: Tree,
    search: impl Fn(&Tree, &Rectangle),
) {
    let bounds_cache = gen_balls_doubling(&mut tree, AMOUNT, SEED, 0);
    tree.end_frame();

    group.bench_function(name, |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| search(&tree, bounds));
        });
    });
}

fn small_balls(c: &mut Criterion) {
    let mut group = c.benchmark_group("small_balls");
    bench_small_balls(
        &mut group,
        "HierarchicalLooseGrid",
        HierarchicalLooseGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_small_balls(
        &mut group,
        "OverflowBoundGrid",
        OverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_small_balls(
        &mut group,
        "LazyOverflowBoundGrid",
        LazyOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_small_balls(
        &mut group,
        "BoxedOverflowBoundGrid",
        BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_small_balls(
        &mut group,
        "SpanOverflowBoundGrid",
        SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_small_balls(
        &mut group,
        "HaloOverflowBoundGrid",
        HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    for (name, cell) in [("312", GRID_CELL_SIZE), ("156", HALF_CELL_SIZE)] {
        bench_small_balls(
            &mut group,
            &format!("UniformGrid/{name}"),
            UniformGrid::<usize>::new(BOUNDS.clone(), cell),
            |tree, bounds| {
                tree.search(bounds, |id| {
                    black_box(id);
                })
            },
        );
        bench_small_balls(
            &mut group,
            &format!("SpanGrid/{name}"),
            SpanGrid::<usize>::new(BOUNDS.clone(), cell),
            |tree, bounds| {
                tree.search(bounds, |id| {
                    black_box(id);
                })
            },
        );
        bench_small_balls(
            &mut group,
            &format!("HaloGrid/{name}"),
            HaloGrid::<usize>::new(BOUNDS.clone(), cell),
            |tree, bounds| {
                tree.search(bounds, |id| {
                    black_box(id);
                })
            },
        );
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
    search: impl Fn(&Tree, &Rectangle),
) {
    gen_aligned(&mut tree);
    tree.end_frame();
    let small = aligned_queries(AMOUNT, 50.0..300.0, 1);
    let large = aligned_queries(AMOUNT / 5, 400.0..1500.0, 2);
    group.bench_function(format!("small/{name}"), |b| {
        b.iter(|| small.iter().for_each(|bounds| search(&tree, bounds)));
    });
    group.bench_function(format!("large/{name}"), |b| {
        b.iter(|| large.iter().for_each(|bounds| search(&tree, bounds)));
    });
}

fn aligned(c: &mut Criterion) {
    let mut group = c.benchmark_group("aligned");
    bench_aligned(
        &mut group,
        "HierarchicalLooseGrid",
        HierarchicalLooseGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_aligned(
        &mut group,
        "OverflowBoundGrid",
        OverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_aligned(
        &mut group,
        "LazyOverflowBoundGrid",
        LazyOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_aligned(
        &mut group,
        "BoxedOverflowBoundGrid",
        BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_aligned(
        &mut group,
        "SpanOverflowBoundGrid",
        SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_aligned(
        &mut group,
        "HaloOverflowBoundGrid",
        HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_aligned(
        &mut group,
        "UniformGrid/312",
        UniformGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_aligned(
        &mut group,
        "SpanGrid/312",
        SpanGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_aligned(
        &mut group,
        "HaloGrid/312",
        HaloGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
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
    search: impl Fn(&Tree, &Rectangle),
) {
    for (id, bounds) in small.iter().chain(large).enumerate() {
        tree.insert_item(bounds.clone(), id);
    }
    tree.end_frame();
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
    bench_two_tier(
        &mut group,
        "HierarchicalLooseGrid",
        HierarchicalLooseGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_two_tier(
        &mut group,
        "OverflowBoundGrid",
        OverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_two_tier(
        &mut group,
        "LazyOverflowBoundGrid",
        LazyOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_two_tier(
        &mut group,
        "BoxedOverflowBoundGrid",
        BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_two_tier(
        &mut group,
        "SpanOverflowBoundGrid",
        SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_two_tier(
        &mut group,
        "HaloOverflowBoundGrid",
        HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |tree, bounds| {
            tree.search(bounds, |id| {
                black_box(id);
            })
        },
    );

    bench_stale(
        &mut group,
        "BoxedOverflowBoundGrid",
        BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |grid, bounds, id| grid.insert(bounds, id),
        |grid, bounds, handle| grid.update(bounds, handle),
        |grid, bounds| {
            grid.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_stale(
        &mut group,
        "SpanOverflowBoundGrid",
        SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |grid, bounds, id| grid.insert(bounds, id),
        |grid, bounds, handle| grid.update(bounds, handle),
        |grid, bounds| {
            grid.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    bench_stale(
        &mut group,
        "HaloOverflowBoundGrid",
        HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
        &scene,
        |grid, bounds, id| grid.insert(bounds, id),
        |grid, bounds, handle| grid.update(bounds, handle),
        |grid, bounds| {
            grid.search(bounds, |id| {
                black_box(id);
            })
        },
    );
    group.finish();
}

/// Search with the balls after the squares have shrunk onto the finest level
/// without a refresh, which empties their level.
fn bench_stale<Tree>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut grid: Tree,
    (small, large): &(Vec<Rectangle>, Vec<Rectangle>),
    insert: impl Fn(&mut Tree, Rectangle, usize) -> ItemHandle,
    update: impl Fn(&mut Tree, Rectangle, ItemHandle),
    search: impl Fn(&Tree, &Rectangle),
) {
    for (id, bounds) in small.iter().enumerate() {
        insert(&mut grid, bounds.clone(), id);
    }
    let handles: Vec<_> = large
        .iter()
        .enumerate()
        .map(|(id, bounds)| insert(&mut grid, bounds.clone(), AMOUNT + id))
        .collect();
    for (bounds, handle) in large.iter().zip(handles) {
        let (x, y) = bounds.get_center();
        update(&mut grid, Rectangle::center_rect(x, y, 80.0, 80.0), handle);
    }
    group.bench_function(format!("stale/{name}"), |b| {
        b.iter(|| small.iter().for_each(|bounds| search(&grid, bounds)));
    });
}

/// Takes the items of a scene without storing them, so that [`gen_balls`] can
/// make the bounds of a scene that is then built with handles.
struct NoTree;

impl QTree<usize> for NoTree {
    fn insert_item(&mut self, _: Rectangle, _: usize) {}
}

/// Search with the balls of the search distribution after every `stride`th
/// of them has moved 1 to 5 units in some direction since the last refresh,
/// as queries do between the updates of a frame and its refresh.
#[allow(clippy::too_many_arguments)]
fn bench_moving<Tree>(
    group: &mut BenchmarkGroup<WallTime>,
    name: &str,
    mut grid: Tree,
    (label, stride): (&str, usize),
    insert: impl Fn(&mut Tree, Rectangle, usize) -> ItemHandle,
    update: impl Fn(&mut Tree, Rectangle, ItemHandle),
    refresh: impl Fn(&mut Tree),
    search: impl Fn(&Tree, &Rectangle),
) {
    let mut bounds = gen_balls(&mut NoTree, AMOUNT, SEED);
    let handles: Vec<_> = bounds
        .iter()
        .enumerate()
        .map(|(id, bounds)| insert(&mut grid, bounds.clone(), id))
        .collect();
    refresh(&mut grid);
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
        update(&mut grid, bounds.clone(), handle);
    }
    group.bench_function(format!("{label}/{name}"), |b| {
        b.iter(|| bounds.iter().for_each(|bounds| search(&grid, bounds)));
    });
}

/// The search distribution with a tenth of the balls, or all of them, moved
/// since the last refresh.
fn moving(c: &mut Criterion) {
    let mut group = c.benchmark_group("moving");
    for scene in [("tenth", 10), ("all", 1)] {
        bench_moving(
            &mut group,
            "BoxedOverflowBoundGrid",
            BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            scene,
            |grid, bounds, id| grid.insert(bounds, id),
            |grid, bounds, handle| grid.update(bounds, handle),
            |grid| grid.refresh_bounds(),
            |grid, bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                })
            },
        );
        bench_moving(
            &mut group,
            "SpanOverflowBoundGrid",
            SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            scene,
            |grid, bounds, id| grid.insert(bounds, id),
            |grid, bounds, handle| grid.update(bounds, handle),
            |grid| grid.refresh_bounds(),
            |grid, bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                })
            },
        );
        bench_moving(
            &mut group,
            "HaloOverflowBoundGrid",
            HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            scene,
            |grid, bounds, id| grid.insert(bounds, id),
            |grid, bounds, handle| grid.update(bounds, handle),
            |grid| grid.refresh_bounds(),
            |grid, bounds| {
                grid.search(bounds, |id| {
                    black_box(id);
                })
            },
        );
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
    search: impl Fn(&Tree, &Rectangle),
    self_query: impl Fn(&Tree),
) {
    let bounds_cache = gen_balls(&mut tree, AMOUNT, SEED);
    tree.end_frame();
    group.bench_function(format!("loop/{name}"), |b| {
        b.iter(|| bounds_cache.iter().for_each(|bounds| search(&tree, bounds)));
    });
    group.bench_function(format!("batch/{name}"), |b| b.iter(|| self_query(&tree)));
}

fn self_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("self_query");
    macro_rules! arm {
        ($t:ident) => {
            bench_self_query(
                &mut group,
                stringify!($t),
                $t::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
                |tree, bounds| {
                    tree.search(bounds, |id| {
                        black_box(id);
                    })
                },
                |tree| {
                    tree.for_each_self_query(|query, id| {
                        black_box((query, id));
                    })
                },
            )
        };
    }
    arm!(OverflowBoundGrid);
    arm!(LazyOverflowBoundGrid);
    arm!(BoxedOverflowBoundGrid);
    arm!(SpanOverflowBoundGrid);
    arm!(HaloOverflowBoundGrid);
    arm!(HierarchicalLooseGrid);
    arm!(HierarchicalGrid);
    macro_rules! single {
        ($t:ident) => {
            bench_self_query(
                &mut group,
                stringify!($t),
                $t::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
                |tree, bounds| {
                    tree.search(bounds, |id| {
                        black_box(id);
                    })
                },
                |tree| {
                    tree.for_each_self_query(|query, id| {
                        black_box((query, id));
                    })
                },
            )
        };
    }
    single!(UniformGrid);
    single!(SpanGrid);
    single!(HaloGrid);
    group.finish();
}

criterion_group!(
    bench,
    quadtree_loose,
    grid_hierarchical_loose,
    grid_uniform,
    grid_hierarchical,
    grid_overflow_exact,
    grid_overflow_lazy,
    grid_overflow_boxed,
    grid_overflow_span,
    grid_overflow_halo,
    grid_uniform,
    span_grid,
    halo_grid,
    small_balls,
    aligned,
    gap,
    adjacent,
    moving,
    self_query,
);
criterion_main!(bench);
