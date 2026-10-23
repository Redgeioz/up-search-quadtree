use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
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

/// Cell size of the uniform grids: the finest cell size of the hierarchical
/// structures, 312.5.
const GRID_CELL_SIZE: f64 = BOUND_WIDTH / (1u32 << (MAX_LEVEL - 1)) as f64;

/// Common interface to the item management of every structure.
trait QTree<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle;
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle);
    fn remove_item(&mut self, handle: ItemHandle);
    /// Called after each round of updates, like the end of a frame.
    fn end_frame(&mut self) {}
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

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for HierarchicalLooseGrid<T, MAX_LEVEL> {
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

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for OverflowBoundGrid<T, MAX_LEVEL> {
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

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for LazyOverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for BoxedOverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}


impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for SpanOverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for HaloOverflowBoundGrid<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
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

impl<T: Copy, const MAX_LEVEL: u8> QTree<T> for HierarchicalGrid<T, MAX_LEVEL> {
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

impl<T: Copy> QTree<T> for SpanGrid<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
    }
}

impl<T: Copy> QTree<T> for HaloGrid<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        self.insert(bounds, item)
    }
    fn update_item(&mut self, bounds: Rectangle, handle: ItemHandle) {
        self.update(bounds, handle);
    }
    fn remove_item(&mut self, handle: ItemHandle) {
        self.remove(handle);
    }
    fn end_frame(&mut self) {
        self.refresh_bounds();
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
            tree.end_frame();
        });
    });
}

/// Remove every ball and reinsert it at its current position.
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
            tree.end_frame();
        });
    });
}

fn update(c: &mut Criterion) {
    let mut group = c.benchmark_group("update");
    bench_update(
        &mut group,
        "LooseQuadTree",
        LooseQuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "HierarchicalLooseGrid",
        HierarchicalLooseGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "OverflowBoundGrid",
        OverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "LazyOverflowBoundGrid",
        LazyOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "BoxedOverflowBoundGrid",
        BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "SpanOverflowBoundGrid",
        SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "HaloOverflowBoundGrid",
        HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_update(
        &mut group,
        "SpanGrid",
        SpanGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
    );
    bench_update(
        &mut group,
        "HaloGrid",
        HaloGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
    );
    bench_update(
        &mut group,
        "UniformGrid",
        UniformGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
    );
    bench_update(
        &mut group,
        "HierarchicalGrid",
        HierarchicalGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    group.finish();
}

fn reinsert(c: &mut Criterion) {
    let mut group = c.benchmark_group("reinsert");
    bench_reinsert(
        &mut group,
        "LooseQuadTree",
        LooseQuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "HierarchicalLooseGrid",
        HierarchicalLooseGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "OverflowBoundGrid",
        OverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "LazyOverflowBoundGrid",
        LazyOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "BoxedOverflowBoundGrid",
        BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "SpanOverflowBoundGrid",
        SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "HaloOverflowBoundGrid",
        HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    bench_reinsert(
        &mut group,
        "SpanGrid",
        SpanGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
    );
    bench_reinsert(
        &mut group,
        "HaloGrid",
        HaloGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
    );
    bench_reinsert(
        &mut group,
        "UniformGrid",
        UniformGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
    );
    bench_reinsert(
        &mut group,
        "HierarchicalGrid",
        HierarchicalGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
    );
    group.finish();
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
    let mut baseline = HierarchicalLooseGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let (mut balls, _) = gen_balls(&mut baseline, AMOUNT, SEED);
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
        bench_update_pair(
            &mut group,
            &format!("{scenario}/HierarchicalLooseGrid"),
            HierarchicalLooseGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            &from,
            to,
        );
        bench_update_pair(
            &mut group,
            &format!("{scenario}/OverflowBoundGrid"),
            OverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            &from,
            to,
        );
        bench_update_pair(
            &mut group,
            &format!("{scenario}/LazyOverflowBoundGrid"),
            LazyOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            &from,
            to,
        );
        bench_update_pair(
            &mut group,
            &format!("{scenario}/BoxedOverflowBoundGrid"),
            BoxedOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            &from,
            to,
        );
        bench_update_pair(
            &mut group,
            &format!("{scenario}/SpanOverflowBoundGrid"),
            SpanOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            &from,
            to,
        );
        bench_update_pair(
            &mut group,
            &format!("{scenario}/HaloOverflowBoundGrid"),
            HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(BOUNDS.clone()),
            &from,
            to,
        );
        bench_update_pair(
            &mut group,
            &format!("{scenario}/UniformGrid"),
            UniformGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
            &from,
            to,
        );
        bench_update_pair(
            &mut group,
            &format!("{scenario}/SpanGrid"),
            SpanGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
            &from,
            to,
        );
        bench_update_pair(
            &mut group,
            &format!("{scenario}/HaloGrid"),
            HaloGrid::<usize>::new(BOUNDS.clone(), GRID_CELL_SIZE),
            &from,
            to,
        );
    }
    group.finish();
}

criterion_group!(bench, update, reinsert, update_compare);
criterion_main!(bench);
