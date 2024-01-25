use rand::Rng;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rectangle {
    // Top left corner
    pub min_x: f64,
    pub min_y: f64,

    // Bottom right corner
    pub max_x: f64,
    pub max_y: f64,
}

impl Rectangle {
    #[inline]
    pub fn new(min_x: f64, min_y: f64, max_x: f64, max_y: f64) -> Rectangle {
        Rectangle {
            min_x,
            min_y,
            max_x,
            max_y,
        }
    }

    #[inline]
    pub fn center_rect(x: f64, y: f64, w: f64, h: f64) -> Rectangle {
        let min_x = x - w / 2.0;
        let min_y = y - h / 2.0;
        let max_x = x + w / 2.0;
        let max_y = y + h / 2.0;
        Rectangle::new(min_x, min_y, max_x, max_y)
    }

    #[inline]
    pub fn get_min(&self) -> (f64, f64) {
        (self.min_x, self.min_y)
    }

    #[inline]
    pub fn get_max(&self) -> (f64, f64) {
        (self.max_x, self.max_y)
    }

    #[inline]
    pub fn get_width(&self) -> f64 {
        self.max_x - self.min_x
    }

    #[inline]
    pub fn get_height(&self) -> f64 {
        self.max_y - self.min_y
    }

    #[inline]
    pub fn get_center(&self) -> (f64, f64) {
        let x = (self.min_x + self.max_x) / 2.0;
        let y = (self.min_y + self.max_y) / 2.0;
        (x, y)
    }

    #[inline]
    pub fn get_random_point(&self, rng: &mut impl Rng) -> (f64, f64) {
        let x = rng.gen_range(0.0..1.0) * (self.max_x - self.min_x) + self.min_x;
        let y = rng.gen_range(0.0..1.0) * (self.max_y - self.min_y) + self.min_y;
        (x, y)
    }

    #[inline]
    pub fn scale(&self, scale: f64) -> Rectangle {
        let (x, y) = self.get_center();
        let w = self.get_width() * scale;
        let h = self.get_height() * scale;
        Rectangle::center_rect(x, y, w, h)
    }

    #[inline]
    pub fn contains(&self, b: &Rectangle) -> bool {
        self.min_x <= b.min_x
            && b.max_x <= self.max_x
            && self.min_y <= b.min_y
            && b.max_y <= self.max_y
    }

    #[inline]
    pub fn contains_point(&self, p: (f64, f64)) -> bool {
        let (x, y) = p;
        self.min_x <= x && x <= self.max_x && self.min_y <= y && y <= self.max_y
    }

    #[inline]
    pub fn intersects(&self, b: &Rectangle) -> bool {
        self.min_x < b.max_x && b.min_x < self.max_x && self.min_y < b.max_y && b.min_y < self.max_y
    }
}
