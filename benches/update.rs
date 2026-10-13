use criterion::measurement::WallTime;
use criterion::{BenchmarkGroup, Criterion, criterion_group, criterion_main};
use rand::*;

use lib::ball::*;
use lib::rect::*;
use lib::*;

const MAX_LEVEL: u8 = 7;
const MAX_ITEMS: usize = 8;

const BOUND_WIDTH: f64 = 20000.0;
const BOUND_HEIGHT: f64 = 20000.0;

const AMOUNT: usize = 10000;
const SEED: u64 = 0;

const MIN_RADIUS: f64 = 25.0;
const MAX_RADIUS: f64 = 50.0;
const MIN_VELOCITY: f64 = 1.0;
const MAX_VELOCITY: f64 = 5.0;

const BOUNDS: Rectangle = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);

/// Cell size of the uniform grids: the finest node size of the quadtrees.
const GRID_CELL_SIZE: f64 = BOUND_WIDTH / (1u32 << (MAX_LEVEL - 1)) as f64;

/// Uniform access to the item management of every tree.
trait QTree<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle;
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle);
    fn remove_item(&mut self, handle: ItemHandle);
}

impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> QTree<T>
    for QuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
}

impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> QTree<T>
    for LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for GridLooseQuadTree<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for UpSearchQuadTree<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for UpSearchQuadTreeOriginal<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
}

impl<T: Copy> QTree<T> for UniformGrid<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for MultiLayerGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
}

fn gen_balls(
    quadtree: &mut impl QTree<usize>,
    count: usize,
    seed: u64,
) -> (Vec<Ball>, Vec<ItemHandle>) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut balls = Vec::with_capacity(count);
    let mut handles = Vec::with_capacity(count);
    for _ in 0..count {
        let (x, y) = BOUNDS.get_random_point(&mut rng);
        let mut r = rng.gen_range(MIN_RADIUS..MAX_RADIUS);
        for _ in 0..5 {
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
fn bench_update(group: &mut BenchmarkGroup<WallTime>, name: &str, mut tree: impl QTree<usize>) {
    let (mut balls, handles) = gen_balls(&mut tree, AMOUNT, SEED);

    group.bench_function(name, |b| {
        b.iter(|| {
            balls.iter_mut().zip(&handles).for_each(|(ball, handle)| {
                ball.update(&BOUNDS);
                tree.update_item(ball.get_bounds(), *handle);
            });
        });
    });
}

/// Remove every ball and insert it again where it is.
fn bench_reinsert(group: &mut BenchmarkGroup<WallTime>, name: &str, mut tree: impl QTree<usize>) {
    let (balls, mut handles) = gen_balls(&mut tree, AMOUNT, SEED);

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
        });
    });
}

fn update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");
    bench_update(
        &mut group,
        "QuadTree",
        QuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "LooseQuadTree",
        LooseQuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "GridLooseQuadTree",
        GridLooseQuadTree::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "UpSearchQuadTree",
        UpSearchQuadTree::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "UpSearchQuadTreeOriginal",
        UpSearchQuadTreeOriginal::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "UniformGrid",
        UniformGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
    );
    bench_update(
        &mut group,
        "MultiLayerGrid",
        MultiLayerGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    group.finish();
}

fn reinsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("reinsert");
    bench_reinsert(
        &mut group,
        "QuadTree",
        QuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "LooseQuadTree",
        LooseQuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "GridLooseQuadTree",
        GridLooseQuadTree::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "UpSearchQuadTree",
        UpSearchQuadTree::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "UpSearchQuadTreeOriginal",
        UpSearchQuadTreeOriginal::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "UniformGrid",
        UniformGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
    );
    bench_reinsert(
        &mut group,
        "MultiLayerGrid",
        MultiLayerGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    group.finish();
}

criterion_group!(bench, update, reinsert);
criterion_main!(bench);
