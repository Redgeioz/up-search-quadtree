use super::Looseness;
use crate::rect::Rectangle;
use grid::*;
use std::collections::HashMap;
use std::hash::Hash;
use std::mem::{transmute, MaybeUninit};
use std::ptr;

type Coord = (usize, usize, usize);
type Grids<T, const LOOSENESS: Looseness> = Vec<Grid<*mut GridLooseQuadTreeNode<T, LOOSENESS>>>;

/// An implementation of a loose quadtree with grid.
pub struct GridLooseQuadTree<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const LOOSENESS: Looseness> {
    root: Box<GridLooseQuadTreeNode<T, LOOSENESS>>,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    grids: Box<Grids<T, LOOSENESS>>,
    items: HashMap<T, Coord>,
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const LOOSENESS: Looseness>
    GridLooseQuadTree<T, MAX_LEVEL, LOOSENESS>
{
    /// Create a new quadtree with the given bounds.
    ///
    /// Force using a square as the bounds of each node. This usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> GridLooseQuadTree<T, MAX_LEVEL, LOOSENESS> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new quadtree with the given bounds.
    ///
    /// Make the bounds of each node fit the given bounds instead of forcing it to be square.
    pub fn new_fit(world_bounds: Rectangle) -> GridLooseQuadTree<T, MAX_LEVEL, LOOSENESS> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(
        world_bounds: Rectangle,
    ) -> GridLooseQuadTree<T, MAX_LEVEL, LOOSENESS> {
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let mut root_width = world_bounds.get_width();
        let mut root_height = world_bounds.get_height();

        // Initialize grids
        let mut grids = Box::new(Vec::with_capacity(MAX_LEVEL as usize + 1));
        let mut size = 0;
        for n in 0..=MAX_LEVEL {
            let (mut rows, mut cols) = (size, size);
            if !FIT {
                if root_width > root_height {
                    rows = (size as f64 * root_height / root_width).ceil() as usize;
                } else if root_height > root_width {
                    cols = (size as f64 / root_height * root_width).ceil() as usize;
                }
            }
            grids.push(Grid::init(rows, cols, ptr::null_mut()));
            size = 1 << n;
        }

        // Determine the root bounds to use
        let root_bounds = if !FIT {
            let len = root_width.max(root_height);
            root_width = len;
            root_height = len;

            let (min_x, min_y) = world_bounds.get_min();
            let center_x = min_x + len * 0.5;
            let center_y = min_y + len * 0.5;
            Rectangle::center_rect(center_x, center_y, root_width, root_height)
        } else {
            world_bounds.clone()
        };

        let looseness = LOOSENESS.to_f64();
        let loose_bounds = root_bounds.scale(looseness);

        // Initialize all nodes
        let mut this = GridLooseQuadTree {
            root: Box::new(GridLooseQuadTreeNode::new(loose_bounds)),
            root_bounds,
            world_bounds,
            items: HashMap::new(),
            grids,
        };

        (this.grids[1])[0][0] = this.root.as_mut() as *mut GridLooseQuadTreeNode<T, LOOSENESS>;
        this.root
            .split(&mut this.grids, (1, 0, 0), MAX_LEVEL as usize);
        this
    }

    /// Return the real bounds of the root.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Find the level and coordinates of the node where the item should be inserted.
    pub fn position(&self, bounds: &Rectangle) -> Coord {
        if !self.world_bounds.contains_point(bounds.get_center()) {
            return (1, 0, 0);
        }

        let width = bounds.get_width();
        let height = bounds.get_height();
        let root_bounds = &self.root_bounds;
        let root_width = root_bounds.get_width();
        let root_height = root_bounds.get_height();
        let multiple_x = (root_width / width) as usize;
        let multiple_y = (root_height / height) as usize;

        // Equals to:
        // let multiple = multiple_x.min(multiple_y);
        // let level = if multiple != 0 {
        //     ((multiple.ilog2() + 1) as u8).min(MAX_LEVEL)
        // } else {
        //     1
        // };
        let level = usize::BITS - multiple_x.min(multiple_y).leading_zeros();
        let level = (level as usize).clamp(1, MAX_LEVEL as usize);

        let (x, y) = bounds.get_center();
        let (offset_x, offset_y) = root_bounds.get_min();

        let grid = self.grids.get(level).unwrap();
        let grid_width = grid.cols();
        let grid_height = grid.rows();

        let edge_max_node_num = grid_width.max(grid_height) as f64;
        let node_width = root_width / edge_max_node_num;
        let node_height = root_height / edge_max_node_num;

        let coord_x = (((x - offset_x) / node_width) as usize).min(grid_width - 1);
        let coord_y = (((y - offset_y) / node_height) as usize).min(grid_height - 1);

        (level, coord_x, coord_y)
    }

    fn get_node(&self, level: usize, x: usize, y: usize) -> &GridLooseQuadTreeNode<T, LOOSENESS> {
        unsafe { &*self.grids[level][y][x] }
    }

    fn get_node_mut(
        &mut self,
        level: usize,
        x: usize,
        y: usize,
    ) -> &mut GridLooseQuadTreeNode<T, LOOSENESS> {
        unsafe { &mut *self.grids[level][y][x] }
    }

    /// Insert an item into the quadtree.
    ///
    /// The time complexity is O(1) since the insertion position is obtained directly
    /// by computation, instead of by judgment at each node.
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> Coord {
        let coord = self.position(&bounds);
        self.items.insert(item, coord);

        let (level, x, y) = coord;
        let node = self.get_node_mut(level, x, y);

        node.add(bounds, item);
        coord
    }

    /// Remove an item from the quadtree.
    pub fn remove(&mut self, item: T) {
        let coord = self.items.get(&item).expect("Removal item not found.");
        let (level, x, y) = *coord;
        let node = self.get_node_mut(level, x, y);

        node.remove(item);

        self.items.remove(&item);
    }

    /// Update the bounds of an item and, if necessary, its position in the quadtree.
    pub fn update(&mut self, bounds: Rectangle, item: T) {
        let curt_coord = self.position(&bounds);
        let prev_coord = self.items.get_mut(&item).expect("Update item not found.");

        if curt_coord == *prev_coord {
            let (level, x, y) = *prev_coord;
            self.get_node_mut(level, x, y).update(bounds, item);
            return;
        }

        let prev_coord = {
            let prev = *prev_coord;
            *prev_coord = curt_coord;

            prev
        };

        unsafe {
            let get_node_mut = |coord: Coord| {
                let (level, x, y) = coord;
                let ptr = self.grids[level][y][x];
                &mut *ptr
            };

            let new_node = get_node_mut(curt_coord);
            let old_node = get_node_mut(prev_coord);

            old_node.remove(item);
            new_node.add(bounds, item);
        }
    }

    /// Search from the bottom up. Execute the callback function for each item found.
    ///
    /// As an extension of [`search_bidirectional`], at each of the remaining layers, we
    /// can directly calculate the range of nodes that intersect the search region, rather
    /// than making a separate judgment for each node about whether it intersects the
    /// search region.
    ///
    /// Notebly, in addition to being faster than the traditional method, the items it finds
    /// are relatively ordered in terms of size due to the fact that it searches each level
    /// from the bottom up, rather than each branch from the top. This might be useful when
    /// sorting is required.
    ///
    /// [`search_bidirectional`]: GridLooseQuadTree::search_bidirectional
    pub fn search_up(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        let width = bounds.get_width();
        let height = bounds.get_height();
        let root_bounds = &self.root_bounds;
        let root_width = root_bounds.get_width();
        let root_height = root_bounds.get_height();
        let multiple_x = (root_width / width) as usize;
        let multiple_y = (root_height / height) as usize;

        let level = usize::BITS - multiple_x.min(multiple_y).leading_zeros();
        let level = (level as usize).clamp(1, MAX_LEVEL as usize);

        let (offset_x, offset_y) = root_bounds.get_min();

        let grids = &self.grids;
        let looseness = LOOSENESS.to_f64();
        if level != MAX_LEVEL as usize {
            // top left
            let min_x = bounds.min_x - offset_x;
            let min_y = bounds.min_y - offset_y;

            // bottom right
            let max_x = bounds.max_x - offset_x;
            let max_y = bounds.max_y - offset_y;

            let calc_min = |n: f64| ((n + (1.0 - (looseness - 1.0) / 2.0)).trunc() - 1.0) as usize;
            let calc_max = |n: f64| (n + (looseness - 1.0) / 2.0).trunc() as usize;

            // Note: Searching directly from the bottom to the top using this method will be slower
            for level in ((level + 1)..=MAX_LEVEL as usize).rev() {
                let grid = unsafe { grids.get_unchecked(level) };

                let grid_width = grid.cols();
                let grid_height = grid.rows();

                let edge_max_node_num = grid_width.max(grid_height) as f64;
                let node_width = root_width / edge_max_node_num;
                let node_height = root_height / edge_max_node_num;

                let min_coord_x = (calc_min(min_x / node_width)).min(grid_width - 1);
                let min_coord_y = (calc_min(min_y / node_height)).min(grid_height - 1);

                let max_coord_x = (calc_max(max_x / node_width)).min(grid_width - 1);
                let max_coord_y = (calc_max(max_y / node_height)).min(grid_height - 1);

                // Scan the region obtained
                for y in min_coord_y..=max_coord_y {
                    for x in min_coord_x..=max_coord_x {
                        let node = unsafe { &**grid.get_unchecked(y, x) };
                        node.search_items(bounds, &mut callback);
                    }
                }
            }
        }

        if level == 1 {
            self.root.search_items(bounds, &mut callback);
            return;
        }

        let (x, y) = bounds.get_center();

        let grid = grids.get(level).unwrap();
        let grid_width = grid.cols();
        let grid_height = grid.rows();

        let edge_max_node_num = grid_width.max(grid_height) as f64;
        let node_width = root_width / edge_max_node_num;
        let node_height = root_height / edge_max_node_num;

        let coord_x = (((x - offset_x) / node_width) as usize).min(grid_width - 1);
        let coord_y = (((y - offset_y) / node_height) as usize).min(grid_height - 1);

        let node = self.get_node(level, coord_x, coord_y);
        node.search_up_3x3::<false>(grids, (level, coord_x, coord_y), bounds, &mut callback);
    }

    /// Search down and then up from the level where the search area is located.
    /// Execute the callback function for each item found.
    ///
    /// The idea behind this approach is that, similar to the method for insertion,
    /// firstly we can treat the search region as an item and find its level and
    /// coordinates. From the obtained position upwards, the path is determined,
    /// so we only need to check 3x3 areas centered on this path. This skips a lot
    /// of conditional judgments. For the remainder, use the traditional method.
    pub fn search_bidirectional(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        let width = bounds.get_width();
        let height = bounds.get_height();
        let root_bounds = &self.root_bounds;
        let root_width = root_bounds.get_width();
        let root_height = root_bounds.get_height();
        let multiple_x = (root_width / width) as usize;
        let multiple_y = (root_height / height) as usize;

        let level = usize::BITS - multiple_x.min(multiple_y).leading_zeros();
        let level = (level as usize).clamp(1, MAX_LEVEL as usize);

        if level <= 2 {
            self.root.search_down(bounds, &mut callback);
            return;
        }

        let (x, y) = bounds.get_center();
        let (offset_x, offset_y) = root_bounds.get_min();

        let grids = &self.grids;
        let grid = grids.get(level).unwrap();
        let grid_width = grid.cols();
        let grid_height = grid.rows();

        let edge_max_node_num = grid_width.max(grid_height) as f64;
        let node_width = root_width / edge_max_node_num;
        let node_height = root_height / edge_max_node_num;

        let coord_x = (((x - offset_x) / node_width) as usize).min(grid_width - 1);
        let coord_y = (((y - offset_y) / node_height) as usize).min(grid_height - 1);

        let node = self.get_node(level, coord_x, coord_y);
        node.search_up_3x3::<true>(grids, (level, coord_x, coord_y), bounds, &mut callback);
    }

    /// Traditional method. Search from the top down.
    /// Execute the callback function for each item found.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        self.root.search_down(bounds, &mut callback);
    }
}

struct GridLooseQuadTreeNode<T: Copy + Eq, const LOOSENESS: Looseness> {
    loose_bounds: Rectangle,
    items: Vec<(Rectangle, T)>,
    children: Option<Box<[Self; 4]>>,
}

impl<T: Copy + Eq, const LOOSENESS: Looseness> GridLooseQuadTreeNode<T, LOOSENESS> {
    fn new(loose_bounds: Rectangle) -> Self {
        Self {
            loose_bounds,
            items: Vec::new(),
            children: None,
        }
    }

    fn add(&mut self, bounds: Rectangle, item: T) {
        self.items.push((bounds, item));
    }

    fn update(&mut self, bounds: Rectangle, item: T) {
        let (b, _) = self
            .items
            .iter_mut()
            .find(|(_, stored)| *stored == item)
            .expect("Item not found");

        *b = bounds;
    }

    fn remove(&mut self, item: T) {
        let i = self
            .items
            .iter()
            .position(|(_, stored)| *stored == item)
            .expect("Item not found.");

        self.items.remove(i);
    }

    /// Initialize all nodes at once
    fn split(&mut self, grids: &mut Grids<T, LOOSENESS>, position: Coord, max_level: usize) {
        let (level, coord_x, coord_y) = position;
        if level >= max_level {
            return;
        }

        let looseness = LOOSENESS.to_f64();
        let (center_x, center_y) = self.loose_bounds.get_center();

        // wh of loose bounds of child nodes
        let width = self.loose_bounds.get_width() / 2.0;
        let height = self.loose_bounds.get_height() / 2.0;

        let offset_x = width / looseness / 2.0;
        let offset_y = height / looseness / 2.0;

        let mut children = Box::new([
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
            MaybeUninit::uninit(),
        ]);

        for i in 0..4 {
            #[rustfmt::skip]
            let sign_x = ((i &  1) * 2) as f64 - 1.0;
            let sign_y = ((i >> 1) * 2) as f64 - 1.0;

            let rect = Rectangle::center_rect(
                center_x + sign_x * offset_x,
                center_y + sign_y * offset_y,
                width,
                height,
            );
            let node = GridLooseQuadTreeNode::new(rect);

            // Calculate the coordinates of child nodes.
            //   i   |  i & 1  |  i >> 1 |
            // ——————|—————————|—————————|
            //  0b00 |    0    |    0    |
            //  0b01 |    1    |    0    |
            //  0b10 |    0    |    1    |
            //  0b11 |    1    |    1    |
            // ——————|—————————|—————————|
            #[rustfmt::skip]
            let coord_x = (coord_x << 1) + (i &  1);
            let coord_y = (coord_y << 1) + (i >> 1);

            children[i].write(node);

            let node = unsafe { children[i].assume_init_mut() };

            let next_level = level + 1;
            if grids
                .get(next_level)
                .unwrap()
                .get(coord_y, coord_x)
                .is_some()
            {
                node.split(grids, (next_level, coord_x, coord_y), max_level);
                grids[next_level][coord_y][coord_x] = node as *mut Self;
            }
        }

        let children = unsafe { transmute::<_, Box<[Self; 4]>>(children) };

        self.children = Some(children);
    }

    fn search_up_3x3<const DOWN: bool>(
        &self,
        grids: &Grids<T, LOOSENESS>,
        position: Coord,
        bounds: &Rectangle,
        callback: &mut impl FnMut(T),
    ) {
        let (level, x, y) = position;
        let mut search = |node: &Self| {
            if DOWN {
                node.search_down(bounds, callback);
            } else {
                node.search_items(bounds, callback);
            }
        };

        unsafe {
            let grid = grids.get_unchecked(level);
            let grid_width = grid.cols();
            let grid_height = grid.rows();
            let (center_x, center_y) = self.loose_bounds.get_center();

            let looseness = LOOSENESS.to_f64();
            let width = self.loose_bounds.get_width() / looseness;
            let height = self.loose_bounds.get_height() / looseness;

            // range 0 ~ 0.5
            let coeff = 0.5 - (looseness - 1.0) / 2.0;

            let y_top = center_y - height * coeff;
            let y_bottom = center_y + height * coeff;
            let x_left = center_x - width * coeff;
            let x_right = center_x + width * coeff;

            if bounds.min_y < y_top && y != 0 {
                if bounds.min_x < x_left && x != 0 {
                    let node = &**grid.get_unchecked(y - 1, x - 1);
                    search(node);
                }

                {
                    let node = &**grid.get_unchecked(y - 1, x);
                    search(node);
                }

                if bounds.max_x > x_right && x + 1 != grid_width {
                    let node = &**grid.get_unchecked(y - 1, x + 1);
                    search(node);
                }
            }

            {
                if bounds.min_x < x_left && x != 0 {
                    let node = &**grid.get_unchecked(y, x - 1);
                    search(node);
                }

                {
                    search(self);
                }

                if bounds.max_x > x_right && x + 1 != grid_width {
                    let node = &**grid.get_unchecked(y, x + 1);
                    search(node);
                }
            }

            if bounds.max_y > y_bottom && y + 1 != grid_height {
                if bounds.min_x < x_left && x != 0 {
                    let node = &**grid.get_unchecked(y + 1, x - 1);
                    search(node);
                }

                {
                    let node = &**grid.get_unchecked(y + 1, x);
                    search(node);
                }

                if bounds.max_x > x_right && x + 1 != grid_width {
                    let node = &**grid.get_unchecked(y + 1, x + 1);
                    search(node);
                }
            }

            let upper_grid = grids.get_unchecked(level - 1);
            let parent_coord_x = x >> 1;
            let parent_coord_y = y >> 1;
            let parent = &**upper_grid.get_unchecked(parent_coord_y, parent_coord_x);
            let next_pos = (level - 1, parent_coord_x, parent_coord_y);

            if level == 2 {
                parent.search_items(bounds, callback);
            } else {
                parent.search_up_3x3::<false>(grids, next_pos, bounds, callback);
            }
        }
    }

    fn search_down(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        self.children.iter().for_each(|children| {
            let [tl, tr, bl, br] = children.as_ref();
            let (min_x, min_y) = tl.loose_bounds.get_min();
            let (x_left, y_top) = br.loose_bounds.get_min();
            let (max_x, max_y) = br.loose_bounds.get_max();
            let (x_right, y_bottom) = tl.loose_bounds.get_max();

            let children_bounds = Rectangle::new(min_x, min_y, max_x, max_y);
            if !children_bounds.intersects(bounds) {
                return;
            }

            if bounds.min_y < y_bottom {
                if bounds.min_x < x_right {
                    tl.search_down(bounds, callback);
                }
                if bounds.max_x > x_left {
                    tr.search_down(bounds, callback);
                }
            }
            if bounds.max_y > y_top {
                if bounds.min_x < x_right {
                    bl.search_down(bounds, callback);
                }
                if bounds.max_x > x_left {
                    br.search_down(bounds, callback);
                }
            }
        });

        self.search_items(bounds, callback);
    }

    fn search_items(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        for (b, item) in self.items.iter() {
            if b.intersects(bounds) {
                callback(*item);
            }
        }
    }
}

unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const LOOSENESS: Looseness> Send
    for GridLooseQuadTree<T, MAX_LEVEL, LOOSENESS>
{
}
unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const LOOSENESS: Looseness> Sync
    for GridLooseQuadTree<T, MAX_LEVEL, LOOSENESS>
{
}
