use crate::rect::Rectangle;
use grid::*;
use std::collections::HashMap;
use std::hash::Hash;

type Coord = (usize, usize, usize);
type Grids<T> = Vec<Grid<UpSearchQuadTreeNode<T>>>;

/// An optimized [`GridLooseQuadTree`] retaining only the up-search function
///
/// One change is to store all nodes in grids to speed up the search by continuous memory data
/// layout. Although, for the traditional branch search method, it causes the four child nodes
/// of a parent node to be non-contiguous in memory layout, thus slowing down the search. The
/// up-search method, which traverses the entire possible range at each level, does not face
/// this problem.
///
/// Another change is to store only the center of the bounds on each node, as opposed to the
/// entire rectangle, which can reduce the size of each node.
///
/// [`GridLooseQuadTree`]: crate::grid_loose_quadtree::GridLooseQuadTree
pub struct UpSearchQuadTree<T: Copy + Eq + Hash, const MAX_LEVEL: u8> {
    grids: Grids<T>,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    items: HashMap<T, Coord>,
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> UpSearchQuadTree<T, MAX_LEVEL> {
    /// Create a new quadtree with the given bounds.
    ///
    /// Force using a square as the bounds of each node. This usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> UpSearchQuadTree<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new quadtree with the given bounds.
    ///
    /// Make the bounds of each node fit the given bounds instead of forcing it to be square.
    pub fn new_fit(world_bounds: Rectangle) -> UpSearchQuadTree<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(world_bounds: Rectangle) -> UpSearchQuadTree<T, MAX_LEVEL> {
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let mut root_width = world_bounds.get_width();
        let mut root_height = world_bounds.get_height();

        // Initialize grids
        let mut grids = Vec::with_capacity(MAX_LEVEL as usize + 1);
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
            let mut vec = Vec::with_capacity(rows * cols);
            for _ in 0..rows {
                for _ in 0..cols {
                    vec.push(UpSearchQuadTreeNode::new());
                }
            }
            grids.push(Grid::from_vec(vec, cols));
            size = 1 << n;
        }

        // Determine the root bounds to use
        let root_bounds;
        if !FIT {
            let len = root_width.max(root_height);
            root_width = len;
            root_height = len;

            let (min_x, min_y) = world_bounds.get_min();
            let center_x = min_x + len * 0.5;
            let center_y = min_y + len * 0.5;
            root_bounds = Rectangle::center_rect(center_x, center_y, root_width, root_height)
        } else {
            root_bounds = world_bounds.clone();
        }

        UpSearchQuadTree {
            grids,
            root_bounds,
            world_bounds,
            items: HashMap::new(),
        }
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

    fn get_root(&self) -> &UpSearchQuadTreeNode<T> {
        unsafe { self.grids.get_unchecked(1).get_unchecked(0, 0) }
    }

    fn get_node_mut(&mut self, level: usize, x: usize, y: usize) -> &mut UpSearchQuadTreeNode<T> {
        &mut self.grids[level][y][x]
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

        let (nl, nx, ny) = curt_coord;
        let (ol, ox, oy) = prev_coord;

        self.grids[ol][oy][ox].remove(item);
        self.grids[nl][ny][nx].add(bounds, item);
    }

    /// Search from the bottom up. Execute the callback function for each item found.
    ///
    /// For details, please refer to the declaration of [`GridLooseQuadTree::search_up`]
    ///
    /// [`GridLooseQuadTree::search_up`]: crate::grid_loose_quadtree::GridLooseQuadTree::search_up
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        let width = bounds.get_width();
        let height = bounds.get_height();
        let root_bounds = &self.root_bounds;
        let root_width = root_bounds.get_width();
        let root_height = root_bounds.get_height();
        let multiple_x = (root_width / width) as usize;
        let multiple_y = (root_height / height) as usize;

        let max_level = MAX_LEVEL as usize;
        let level = usize::BITS - multiple_x.min(multiple_y).leading_zeros();
        let level = (level as usize).clamp(1, max_level);

        let (offset_x, offset_y) = root_bounds.get_min();

        let grids = &self.grids;
        if level != max_level {
            // top left
            let min_x = bounds.min_x - offset_x;
            let min_y = bounds.min_y - offset_y;

            // bottom right
            let max_x = bounds.max_x - offset_x;
            let max_y = bounds.max_y - offset_y;

            let calc_min = |n: f64| ((n + 0.5).trunc() - 1.0) as usize;
            let calc_max = |n: f64| (n + 0.5).trunc() as usize;

            // Note: Searching directly from the bottom to the top using this method will be slower
            grids[level + 1..=max_level].iter().rev().for_each(|grid| {
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
                        let node = unsafe { grid.get_unchecked(y, x) };

                        if y > min_coord_y + 1
                            && y < max_coord_y - 1
                            && x > min_coord_x + 1
                            && x < max_coord_x - 1
                        {
                            // This node is fully contained, so the intersection checks can be skipped
                            node.iter_items(&mut callback);
                        } else {
                            node.search_items(bounds, &mut callback);
                        }
                    }
                }
            });
        }

        if level == 1 {
            self.get_root().search_items(bounds, &mut callback);
            return;
        }

        self.search_up_3x3(level, bounds, &mut callback);
    }

    fn search_up_3x3(&self, level: usize, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        let (bx, by) = bounds.get_center();
        let grid = self.grids.get(level).unwrap();
        let grid_width = grid.cols();
        let grid_height = grid.rows();

        let edge_max_node_num = grid_width.max(grid_height) as f64;

        let root_bounds = &self.root_bounds;
        let (offset_x, offset_y) = root_bounds.get_min();
        let root_width = root_bounds.get_width();
        let root_height = root_bounds.get_height();

        let mut node_width = root_width / edge_max_node_num;
        let mut node_height = root_height / edge_max_node_num;

        let mut x = (((bx - offset_x) / node_width) as usize).min(grid_width - 1);
        let mut y = (((by - offset_y) / node_height) as usize).min(grid_height - 1);

        self.grids[2..=level].iter().rev().for_each(|grid| unsafe {
            let grid_width = grid.cols();
            let grid_height = grid.rows();

            let center_x = offset_x + node_width * (x as f64 + 0.5);
            let center_y = offset_y + node_height * (y as f64 + 0.5);

            if bounds.min_y < center_y && y != 0 {
                if bounds.min_x < center_x && x != 0 {
                    let node = grid.get_unchecked(y - 1, x - 1);
                    node.search_items(bounds, callback);
                }

                {
                    let node = grid.get_unchecked(y - 1, x);
                    node.search_items(bounds, callback);
                }

                if bounds.max_x > center_x && x + 1 != grid_width {
                    let node = grid.get_unchecked(y - 1, x + 1);
                    node.search_items(bounds, callback);
                }
            }

            {
                if bounds.min_x < center_x && x != 0 {
                    let node = grid.get_unchecked(y, x - 1);
                    node.search_items(bounds, callback);
                }

                {
                    let node = grid.get_unchecked(y, x);
                    node.search_items(bounds, callback);
                }

                if bounds.max_x > center_x && x + 1 != grid_width {
                    let node = grid.get_unchecked(y, x + 1);
                    node.search_items(bounds, callback);
                }
            }

            if bounds.max_y > center_y && y + 1 != grid_height {
                if bounds.min_x < center_x && x != 0 {
                    let node = grid.get_unchecked(y + 1, x - 1);
                    node.search_items(bounds, callback);
                }

                {
                    let node = grid.get_unchecked(y + 1, x);
                    node.search_items(bounds, callback);
                }

                if bounds.max_x > center_x && x + 1 != grid_width {
                    let node = grid.get_unchecked(y + 1, x + 1);
                    node.search_items(bounds, callback);
                }
            }

            node_width *= 2.0;
            node_height *= 2.0;
            x /= 2;
            y /= 2;
        });

        self.get_root().search_items(bounds, callback);
    }
}

struct UpSearchQuadTreeNode<T: Copy + Eq> {
    items: Vec<(Rectangle, T)>,
}

impl<T: Copy + Eq> UpSearchQuadTreeNode<T> {
    fn new() -> Self {
        Self { items: Vec::new() }
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

    fn search_items(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        self.items
            .iter()
            .filter(|(b, _)| b.intersects(bounds))
            .for_each(|(_, item)| callback(*item));
    }

    fn iter_items(&self, callback: &mut impl FnMut(T)) {
        self.items.iter().for_each(|(_, item)| callback(*item));
    }
}

unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> Send for UpSearchQuadTree<T, MAX_LEVEL> {}
unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> Sync for UpSearchQuadTree<T, MAX_LEVEL> {}
