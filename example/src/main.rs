use bevy::{
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
};
use lib::ball::*;
use lib::rect::*;
use lib::*;
use rand::*;
use std::f64::consts::PI;

const MAX_LEVEL: u8 = 7;

const BOUND_WIDTH: f64 = 1024.0;
const BOUND_HEIGHT: f64 = 512.0;
const BALL_COUNT: usize = 2500;
const MIN_RADIUS: f64 = 1.0;
const MAX_RADIUS: f64 = 4.0;
const MIN_VELOCITY: f64 = 0.2;
const MAX_VELOCITY: f64 = 1.0;

type QTree = UpSearchQuadTree<usize, MAX_LEVEL>;

#[derive(Resource)]
struct Data {
    pub quadtree: QTree,
    pub balls: Vec<Ball>,
    pub world_bounds: Rectangle,
    pub last_print: f32,
}

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugins(FrameTimeDiagnosticsPlugin)
        .add_systems(Startup, setup)
        .add_systems(Update, update)
        .add_systems(Last, print_fps)
        .run();
}

fn setup(mut commands: Commands) {
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let bounds = Rectangle::new(0.0, 0.0, BOUND_WIDTH, BOUND_HEIGHT);
    let mut quadtree = QTree::new(bounds.clone());

    let mut balls = Vec::with_capacity(BALL_COUNT);
    for _ in 0..BALL_COUNT {
        let (x, y) = bounds.get_random_point(&mut rng);
        let mut r = rng.gen_range(MIN_RADIUS..MAX_RADIUS);
        for _ in 0..5 {
            if rng.gen_bool(0.25) {
                r *= 2.0;
            } else {
                break;
            }
        }
        let v = rng.gen_range(MIN_VELOCITY..MAX_VELOCITY);
        let angle = rng.gen_range(0.0..2.0 * PI);
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
        quadtree.insert(bounds.clone(), id);
    }

    let mut camera = Camera2dBundle::default();
    camera.transform.translation.x = (BOUND_WIDTH / 2.0) as f32;
    camera.transform.translation.y = (BOUND_HEIGHT / 2.0) as f32;
    commands.spawn(camera);
    commands.insert_resource(Data {
        quadtree,
        balls,
        world_bounds: bounds,
        last_print: 0.0,
    });
}

fn update(mut res: ResMut<Data>, mut gizmos: Gizmos) {
    let Data {
        quadtree,
        balls,
        world_bounds,
        last_print: _,
    } = &mut *res;

    for id in 0..balls.len() {
        let ball = balls.get_mut(id).unwrap();
        let bounds = ball.get_bounds();

        ball.update(world_bounds);
        quadtree.search(&bounds, |other_id| {
            if id == other_id {
                return;
            }

            // let [this, other] = balls.get_many_mut([id, other_id]).unwrap();
            let (this, other) = if other_id > id {
                let (s1, s2) = balls.split_at_mut(other_id);
                (&mut s1[id], &mut s2[0])
            } else {
                let (s1, s2) = balls.split_at_mut(id);
                (&mut s2[0], &mut s1[other_id])
            };

            this.collide(other);
        });

        let ball = balls.get_mut(id).unwrap();

        quadtree.update(ball.get_bounds(), id);
    }

    {
        let bounds = world_bounds;
        let (x, y) = bounds.get_center();
        let (x, y) = (x as f32, y as f32);
        let (w, h) = (bounds.get_width() as f32, bounds.get_height() as f32);
        gizmos.rect_2d(Vec2::new(x, y), 0.0, Vec2::new(w, h), Color::BLUE);
    }

    balls.iter().for_each(|ball| {
        let (x, y) = (ball.x as f32, ball.y as f32);
        let radius = ball.r as f32;
        gizmos.circle_2d(Vec2::new(x, y), radius, Color::GREEN);
    });
}

fn print_fps(mut res: ResMut<Data>, diagnostics: Res<DiagnosticsStore>, time: Res<Time>) {
    if let Some(fps) = diagnostics.get(FrameTimeDiagnosticsPlugin::FPS) {
        if let Some(average) = fps.average() {
            let time = time.elapsed_seconds();
            if time - res.last_print > 0.5 {
                println!("FPS: {:.2}", average);
                res.last_print = time;
            }
        }
    }
}
