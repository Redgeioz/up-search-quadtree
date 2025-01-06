use criterion::{criterion_group, criterion_main, Criterion};
use rand::*;
use std::hash::Hash;
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

// level 9~10 size
const MIN_RADIUS: f64 = 25.0;
const MAX_RADIUS: f64 = 50.0;

const BOUNDS: Rectangle = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);

trait QTree<T> {
    fn insert_item(&mut self, bounds: Rectangle, item: T);
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> QTree<T>
    for QuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> QTree<T>
    for LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> QTree<T> for GridLooseQuadTree<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> QTree<T> for UpSearchQuadTree<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> QTree<T> for UpSearchQuadTreeOriginal<T, MAX_LEVEL> {
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

fn gen_balls(quadtree: &mut impl QTree<usize>, count: usize, seed: u64) -> Vec<Rectangle> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut balls = Vec::with_capacity(count);
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

fn quadtree(c: &mut Criterion) {
    let mut quadtree_normal = QuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut quadtree_normal, AMOUNT, SEED);

    c.bench_function("QuadTree", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_normal.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn quadtree_loose(c: &mut Criterion) {
    let mut quadtree_loose = LooseQuadTree::<usize, MAX_LEVEL, 8>::new(BOUNDS.clone());
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

fn quadtree_gl(c: &mut Criterion) {
    let mut quadtree_gl = GridLooseQuadTree::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut quadtree_gl, AMOUNT, SEED);

    c.bench_function("GridLooseQuadTree", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_gl.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn quadtree_us(c: &mut Criterion) {
    let mut quadtree_us = UpSearchQuadTree::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut quadtree_us, AMOUNT, SEED);

    c.bench_function("UpSearchQuadTree", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_us.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

fn quadtree_uso(c: &mut Criterion) {
    let mut quadtree_uso = UpSearchQuadTreeOriginal::<usize, MAX_LEVEL>::new(BOUNDS.clone());
    let bounds_cache = gen_balls(&mut quadtree_uso, AMOUNT, SEED);

    c.bench_function("UpSearchQuadTreeOriginal", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_uso.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

criterion_group!(bench, quadtree, quadtree_loose, quadtree_gl, quadtree_us);
criterion_main!(bench);
