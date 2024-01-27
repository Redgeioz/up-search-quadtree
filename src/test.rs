use crate::ball::*;
use crate::grid_loose_quadtree::GridLooseQuadTree;
use crate::quadtree::QuadTree;
use crate::rect::*;
use crate::up_search_quadtree::UpSearchQuadTree;
use rand::*;
use std::collections::HashSet;
use std::f64::consts::PI;

const MAX_LEVEL: u8 = 7;
const MAX_ITEMS: usize = 8;

const BOUND_WIDTH: f64 = 1024.0;
const BOUND_HEIGHT: f64 = 512.0;
const BALL_COUNT: usize = 2500;
const MIN_RADIUS: f64 = 1.0;
const MAX_RADIUS: f64 = 5.0;
const MIN_VELOCITY: f64 = 0.2;
const MAX_VELOCITY: f64 = 1.0;

#[test]
fn check_quadtrees() {
    for _ in 0..25 {
        single_check();
    }
}

fn single_check() {
    let bounds = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);
    let mut quadtree_loose = GridLooseQuadTree::<usize, MAX_LEVEL>::new(bounds.clone());
    let mut quadtree_us = UpSearchQuadTree::<usize, MAX_LEVEL>::new(bounds.clone());
    let mut quadtree_normal = QuadTree::<usize, MAX_LEVEL, MAX_ITEMS>::new(bounds.clone());

    let mut balls = Vec::with_capacity(BALL_COUNT);
    for _ in 0..BALL_COUNT {
        let (x, y) = bounds.get_random_point(&mut thread_rng());
        let mut r = thread_rng().gen_range(MIN_RADIUS..MAX_RADIUS);
        for _ in 0..7 {
            if thread_rng().gen_bool(0.35) {
                r *= 2.0;
            } else {
                break;
            }
        }
        let v = thread_rng().gen_range(MIN_VELOCITY..MAX_VELOCITY);
        let angle = thread_rng().gen_range(0.0..2.0 * PI);
        let vx = v * angle.cos();
        let vy = v * angle.sin();
        let ball = BallBuilder::new()
            .set_position(x, y)
            .set_radius(r)
            .set_velocity(vx, vy)
            .build();

        let id = balls.len();
        let bounds = ball.get_bounds();
        balls.push(ball);
        quadtree_loose.insert(bounds.clone(), id);
        quadtree_us.insert(bounds.clone(), id);
        quadtree_normal.insert(bounds, id);
    }

    balls.iter().enumerate().for_each(|(id, ball)| {
        let mut set_1 = HashSet::new();
        let mut set_2 = HashSet::new();
        let mut set_3 = HashSet::new();
        let mut set_4 = HashSet::new();
        let mut set_5 = HashSet::new();

        let bounds = ball.get_bounds();

        quadtree_us.search(&bounds, |other_id| {
            if id == other_id {
                return;
            }

            set_1.insert(other_id);
        });

        quadtree_loose.search_bidirectional(&bounds, |other_id| {
            if id == other_id {
                return;
            }

            set_2.insert(other_id);
        });

        quadtree_loose.search_up(&bounds, |other_id| {
            if id == other_id {
                return;
            }

            set_3.insert(other_id);
        });

        quadtree_loose.search(&bounds, |other_id| {
            if id == other_id {
                return;
            }

            set_4.insert(other_id);
        });

        quadtree_normal.search(&bounds, |other_id| {
            if id == other_id {
                return;
            }

            set_5.insert(other_id);
        });

        balls.iter().enumerate().for_each(|(other_id, other)| {
            if id == other_id {
                return;
            }

            if !ball.get_bounds().intersects(&other.get_bounds()) {
                return;
            }

            assert!(set_1.contains(&other_id));
            assert!(set_2.contains(&other_id));
            assert!(set_3.contains(&other_id));
            assert!(set_4.contains(&other_id));
            assert!(set_5.contains(&other_id));
        });
    });
}
