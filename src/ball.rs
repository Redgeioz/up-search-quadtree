use crate::rect::Rectangle;

#[derive(Default)]
pub struct BallBuilder {
    ball: Ball,
}

impl BallBuilder {
    pub fn new() -> BallBuilder {
        BallBuilder::default()
    }

    pub fn set_position(mut self, x: f64, y: f64) -> Self {
        self.ball.x = x;
        self.ball.y = y;
        self
    }

    pub fn set_radius(mut self, r: f64) -> Self {
        self.ball.r = r;
        self.ball.mass = r.powi(2) / 100.0;
        self
    }

    pub fn set_velocity(mut self, vx: f64, vy: f64) -> Self {
        self.ball.vx = vx;
        self.ball.vy = vy;
        self
    }

    pub fn build(self) -> Ball {
        self.ball
    }
}

#[derive(Default)]
pub struct Ball {
    pub x: f64,
    pub y: f64,
    pub r: f64,
    pub mass: f64,
    pub vx: f64,
    pub vy: f64,
}

impl Ball {
    pub fn get_bounds(&self) -> Rectangle {
        Rectangle::new(
            self.x - self.r,
            self.y - self.r,
            self.x + self.r,
            self.y + self.r,
        )
    }

    pub fn update(&mut self, bounds: &Rectangle) {
        self.x += self.vx;
        self.y += self.vy;
        self.bounce(bounds);
    }

    fn bounce(&mut self, bounds: &Rectangle) {
        let r = self.r;
        if self.x < bounds.min_x + r {
            self.x = bounds.min_x + r;
            self.vx = -self.vx;
        } else if self.x > bounds.max_x - r {
            self.x = bounds.max_x - r;
            self.vx = -self.vx;
        }
        if self.y < bounds.min_y + r {
            self.y = bounds.min_y + r;
            self.vy = -self.vy;
        } else if self.y > bounds.max_y - r {
            self.y = bounds.max_y - r;
            self.vy = -self.vy;
        }
    }

    pub fn collide(&mut self, other: &mut Ball) {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        let dist = (dx * dx + dy * dy).sqrt();
        let diff = self.r + other.r - dist;
        if diff > 0.0 {
            let (cos, sin) = if dist.is_normal() {
                (dx / dist, dy / dist)
            } else {
                (1.0, 0.0)
            };

            let p = 2.0 * (self.vx * cos + self.vy * sin - other.vx * cos - other.vy * sin)
                / (self.mass + other.mass);
            self.vx -= p * other.mass * cos;
            self.vy -= p * other.mass * sin;
            other.vx += p * self.mass * cos;
            other.vy += p * self.mass * sin;

            let w1 = self.mass / (self.mass + other.mass);
            let w2 = other.mass / (self.mass + other.mass);

            self.x += w2 * cos * diff;
            self.y += w2 * sin * diff;
            other.x -= w1 * cos * diff;
            other.y -= w1 * sin * diff;
        }
    }
}
