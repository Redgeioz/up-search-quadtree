#![feature(adt_const_params)]
#![feature(const_option)]
use std::hash::Hash;
use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};

use lib::ball::*;
use lib::grid_loose_quadtree::GridLooseQuadTree;
use lib::quadtree::QuadTree;
use lib::rect::*;
use lib::up_search_quadtree::UpSearchQuadTree;
use lib::Looseness;
use rand::*;

const MAX_LEVEL: u8 = 7;
const MAX_ITEMS: usize = 8;
const LOOSENESS: Looseness = Looseness::from_f64(2.0).unwrap();

const BOUND_WIDTH: f64 = 20000.0;
const BOUND_HEIGHT: f64 = 20000.0;

const AMOUNT_SMALL: usize = 2000;
const AMOUNT_MEDIUM: usize = 5000;
const AMOUNT_LARGE: usize = 12500;

// level 9~10 size
const MIN_RADIUS: f64 = 25.0;
const MAX_RADIUS: f64 = 50.0;

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

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const LOOSENESS: Looseness> QTree<T>
    for GridLooseQuadTree<T, MAX_LEVEL, LOOSENESS>
{
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const LOOSENESS: Looseness> QTree<T>
    for UpSearchQuadTree<T, MAX_LEVEL, LOOSENESS>
{
    fn insert_item(&mut self, bounds: Rectangle, item: T) {
        self.insert(bounds, item);
    }
}

fn gen_balls(
    quadtree: &mut impl QTree<usize>,
    bounds: &Rectangle,
    count: usize,
    seed: u64,
) -> Vec<Rectangle> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut balls = Vec::with_capacity(count);
    for _ in 0..count {
        let (x, y) = bounds.get_random_point(&mut rng);
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
    let mut g = c.benchmark_group("QuadTree");

    let bounds = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);

    let mut quadtree_normal = QuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_normal, &bounds, AMOUNT_SMALL, 1);

    g.bench_function("Search (small amount)", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_normal.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });

    let mut quadtree_normal = QuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_normal, &bounds, AMOUNT_MEDIUM, 2);

    g.bench_function("Search (medium amount)", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_normal.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });

    let mut quadtree_normal = QuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_normal, &bounds, AMOUNT_LARGE, 3);

    g.bench_function("Search (large amount)", |b| {
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
    let mut g = c.benchmark_group("GridLooseQuadTree");

    let bounds = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);

    let mut quadtree_loose = GridLooseQuadTree::<usize, MAX_LEVEL, LOOSENESS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_loose, &bounds, AMOUNT_SMALL, 1);

    g.bench_function("Search (samll amount)", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_loose.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });

    // g.bench_function("Search up (samll amount)", |b| {
    //     b.iter(|| {
    //         bounds_cache.iter().for_each(|bounds| {
    //             quadtree_loose.search_up(bounds, |id| {
    //                 black_box(id);
    //             });
    //         });
    //     });
    // });

    let mut quadtree_loose = GridLooseQuadTree::<usize, MAX_LEVEL, LOOSENESS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_loose, &bounds, AMOUNT_MEDIUM, 2);

    g.bench_function("Search (medium amount)", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_loose.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });

    // g.bench_function("Search up (medium amount)", |b| {
    //     b.iter(|| {
    //         bounds_cache.iter().for_each(|bounds| {
    //             quadtree_loose.search_up(bounds, |id| {
    //                 black_box(id);
    //             });
    //         });
    //     });
    // });

    let mut quadtree_loose = GridLooseQuadTree::<usize, MAX_LEVEL, LOOSENESS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_loose, &bounds, AMOUNT_LARGE, 3);

    g.bench_function("Search (large amount)", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_loose.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });

    // g.bench_function("Search up (large amount)", |b| {
    //     b.iter(|| {
    //         bounds_cache.iter().for_each(|bounds| {
    //             quadtree_loose.search_up(bounds, |id| {
    //                 black_box(id);
    //             });
    //         });
    //     });
    // });
}

fn quadtree_us(c: &mut Criterion) {
    let mut g = c.benchmark_group("UpSearchQuadTree");

    let bounds = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);

    let mut quadtree_us = UpSearchQuadTree::<usize, MAX_LEVEL, LOOSENESS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_us, &bounds, AMOUNT_SMALL, 1);

    g.bench_function("Search (small amount)", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_us.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });

    let mut quadtree_us = UpSearchQuadTree::<usize, MAX_LEVEL, LOOSENESS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_us, &bounds, AMOUNT_MEDIUM, 2);

    g.bench_function("Search (medium amount)", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_us.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });

    let mut quadtree_us = UpSearchQuadTree::<usize, MAX_LEVEL, LOOSENESS>::new(bounds.clone());
    let bounds_cache = gen_balls(&mut quadtree_us, &bounds, AMOUNT_LARGE, 3);

    g.bench_function("Search (large amount)", |b| {
        b.iter(|| {
            bounds_cache.iter().for_each(|bounds| {
                quadtree_us.search(bounds, |id| {
                    black_box(id);
                });
            });
        });
    });
}

criterion_group!(bench, quadtree, quadtree_loose, quadtree_us);
criterion_main!(bench);
