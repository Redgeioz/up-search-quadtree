use crate::rect::Rectangle;

use grid::*;
use std::collections::HashMap;
use std::hash::Hash;

/// # Examples
/// ```ignore
/// let a = 0b1010_1101;
/// let b = 0b1011_0111;
/// let (common, shift) = find_common(a, b);
///
/// assert_eq!(common, 0b101);
/// assert_eq!(common << shift, 0b1010_0000);
/// ```
fn find_common(a: usize, b: usize) -> (usize, usize) {
    let shift_steps = usize::BITS as usize - (a ^ b).leading_zeros() as usize;
    (a >> shift_steps, shift_steps)
}

/// (level, x, y, i)
type Location = (usize, usize, usize, usize);
/// (level, x, y)
type Coord = (usize, usize, usize);
type Layers<T> = Vec<Grid<UpSearchQuadTreeNode<T>>>;

pub struct UpSearchQuadTreeOriginal<T: Copy + Eq + Hash, const MAX_LEVEL: u8> {
    layers: Layers<T>,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    items: HashMap<T, Location>,
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> UpSearchQuadTreeOriginal<T, MAX_LEVEL> {
    /// Create a new quadtree with the given bounds.
    ///
    /// Force using a square as the bounds of each node. This usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> UpSearchQuadTreeOriginal<T, MAX_LEVEL> {
        Self::create::<false>(world_bounds)
    }

    /// Create a new quadtree with the given bounds.
    ///
    /// Make the bounds of each node fit the given bounds instead of forcing it to be square.
    pub fn new_fit(world_bounds: Rectangle) -> UpSearchQuadTreeOriginal<T, MAX_LEVEL> {
        Self::create::<true>(world_bounds)
    }

    fn create<const FIT: bool>(world_bounds: Rectangle) -> UpSearchQuadTreeOriginal<T, MAX_LEVEL> {
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let mut root_width = world_bounds.get_width();
        let mut root_height = world_bounds.get_height();

        // Initialize layers
        let mut layers = Vec::with_capacity(MAX_LEVEL as usize + 1);
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
            layers.push(Grid::from_vec(vec, cols));
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

        UpSearchQuadTreeOriginal {
            layers,
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
        if !self.world_bounds.contains(bounds) {
            return (1, 0, 0);
        }

        let root_bounds = &self.root_bounds;
        let root_width = root_bounds.get_width();
        let root_height = root_bounds.get_height();

        let (offset_x, offset_y) = root_bounds.get_min();

        let max_level = MAX_LEVEL as usize;
        let grid = self.layers.get(max_level).unwrap();
        let grid_width = grid.cols();
        let grid_height = grid.rows();

        let edge_max_node_num = grid_width.max(grid_height) as f64;
        let node_width = root_width / edge_max_node_num;
        let node_height = root_height / edge_max_node_num;

        // top left
        let min_x = bounds.min_x - offset_x;
        let min_y = bounds.min_y - offset_y;

        // bottom right
        let max_x = bounds.max_x - offset_x;
        let max_y = bounds.max_y - offset_y;

        // Find the coordinates of the nodes where the top left and bottom right
        // of the given region are located at the max level respectively
        let min_coord_x = ((min_x / node_width) as usize).min(grid_width - 1);
        let min_coord_y = ((min_y / node_height) as usize).min(grid_height - 1);

        let max_coord_x = ((max_x / node_width).ceil() as usize)
            .saturating_sub(1)
            .min(grid_width - 1);
        let max_coord_y = ((max_y / node_height).ceil() as usize)
            .saturating_sub(1)
            .min(grid_height - 1);

        // The next step is a process of finding the lowest common ancestor of the two nodes.
        // Constantly calculate the value of these two coordinates at the upper level and compare
        // them, if they end up equal, the resulting coordinate is the position where the item
        // should be inserted.
        //
        // It is like this:
        // let mut level = max_level;
        // while min_coord_x != max_coord_x || min_coord_y != max_coord_y {
        //     min_coord_x /= 2;
        //     min_coord_y /= 2;
        //     max_coord_x /= 2;
        //     max_coord_y /= 2;
        //     level -= 1;
        // }
        //
        // It can be simplified to an O(1) operation by shifting two binary numbers to the right
        // to find the common part as below:
        let (_, shift_steps_x) = find_common(min_coord_x, max_coord_x);
        let (_, shift_steps_y) = find_common(min_coord_y, max_coord_y);

        let shift_steps = shift_steps_x.max(shift_steps_y);
        let level = max_level - shift_steps;
        let coord_x = min_coord_x >> shift_steps;
        let coord_y = min_coord_y >> shift_steps;

        (level, coord_x, coord_y)
    }

    fn get_root(&self) -> &UpSearchQuadTreeNode<T> {
        unsafe { self.layers.get_unchecked(1).get_unchecked(0, 0) }
    }

    fn get_node_mut(&mut self, level: usize, x: usize, y: usize) -> &mut UpSearchQuadTreeNode<T> {
        &mut self.layers[level][y][x]
    }

    /// Insert an item into the quadtree.
    ///
    /// The time complexity is O(1) since the insertion position is obtained directly
    /// by computation, instead of by judgment at each node.
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> Coord {
        let coord = self.position(&bounds);

        let (level, x, y) = coord;
        let node = self.get_node_mut(level, x, y);

        let location = (coord.0, coord.1, coord.2, node.items.len());
        node.add(bounds, item);
        self.items.insert(item, location);
        coord
    }

    /// Remove an item from the quadtree.
    pub fn remove(&mut self, item: T) {
        let location = self.items.get(&item).expect("Removal item not found.");
        let (level, x, y, i) = *location;
        let node = self.get_node_mut(level, x, y);

        assert!(
            node.items[i].1 == item,
            "Item not found at the expected location."
        );

        let last_idx = node.items.len() - 1;
        if i != last_idx {
            node.items.swap_remove(i);
            let moved_item = node.items[i].1.clone();
            self.items.get_mut(&moved_item).unwrap().3 = i;
        } else {
            node.items.pop();
        }

        self.items.remove(&item);
    }

    /// Update the bounds of an item and, if necessary, its position in the quadtree.
    pub fn update(&mut self, bounds: Rectangle, item: T) {
        let curt_coord = self.position(&bounds);
        let prev_location = self.items.get_mut(&item).expect("Update item not found.");
        let (level, x, y, i) = *prev_location;
        let prev_coord = (level, x, y);

        if curt_coord == prev_coord {
            self.get_node_mut(level, x, y).update(bounds, i);
            return;
        }

        let (nl, nx, ny) = curt_coord;
        let (ol, ox, oy) = prev_coord;

        let new_node = &mut self.layers[nl][ny][nx];

        prev_location.0 = nl;
        prev_location.1 = nx;
        prev_location.2 = ny;
        prev_location.3 = new_node.items.len();

        new_node.add(bounds, item);

        let old_node = &mut self.layers[ol][oy][ox];
        let last_idx = old_node.items.len() - 1;
        if i != last_idx {
            old_node.items.swap_remove(i);
            let moved_item = old_node.items[i].1.clone();
            self.items.get_mut(&moved_item).unwrap().3 = i;
        } else {
            old_node.items.pop();
        }
    }

    /// Search from the bottom up. Execute the callback function for each item found.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        if !self.world_bounds.intersects(bounds) {
            self.get_root().search_items(bounds, &mut callback);
            return;
        }

        let root_bounds = &self.root_bounds;
        let root_width = root_bounds.get_width();
        let root_height = root_bounds.get_height();

        let (offset_x, offset_y) = root_bounds.get_min();

        let max_level = MAX_LEVEL as usize;
        let grid = self.layers.get(max_level).unwrap();
        let grid_width = grid.cols();
        let grid_height = grid.rows();

        let edge_max_node_num = grid_width.max(grid_height) as f64;
        let node_width = root_width / edge_max_node_num;
        let node_height = root_height / edge_max_node_num;

        // top left
        let min_x = bounds.min_x - offset_x;
        let min_y = bounds.min_y - offset_y;

        // bottom right
        let max_x = bounds.max_x - offset_x;
        let max_y = bounds.max_y - offset_y;

        // Find the coordinates of the nodes where the top left and bottom right
        // of the given region are located at the max level respectively
        let mut min_coord_x = ((min_x / node_width) as usize).min(grid_width - 1);
        let mut min_coord_y = ((min_y / node_height) as usize).min(grid_height - 1);

        let mut max_coord_x = ((max_x / node_width).ceil() as usize)
            .saturating_sub(1)
            .min(grid_width - 1);
        let mut max_coord_y = ((max_y / node_height).ceil() as usize)
            .saturating_sub(1)
            .min(grid_height - 1);

        let (_, shift_steps_x) = find_common(min_coord_x, max_coord_x);
        let (_, shift_steps_y) = find_common(min_coord_y, max_coord_y);

        let shift_steps = shift_steps_x.max(shift_steps_y);
        let level = max_level - shift_steps;

        let layers = &self.layers;
        if level != max_level {
            layers[level + 1..=max_level].iter().rev().for_each(|grid| {
                for y in min_coord_y..=max_coord_y {
                    for x in min_coord_x..=max_coord_x {
                        let node = unsafe { grid.get_unchecked(y, x) };

                        if y > min_coord_y && y < max_coord_y && x > min_coord_x && x < max_coord_x
                        {
                            // This node is fully contained, so the intersection checks can be skipped
                            node.iter_items(&mut callback);
                        } else {
                            node.search_items(bounds, &mut callback);
                        }
                    }
                }

                min_coord_x >>= 1;
                min_coord_y >>= 1;
                max_coord_x >>= 1;
                max_coord_y >>= 1;
            });
        }

        let mut coord_x = min_coord_x;
        let mut coord_y = min_coord_y;

        self.layers[1..=level].iter().rev().for_each(|grid| unsafe {
            grid.get_unchecked(coord_y, coord_x)
                .search_items(bounds, &mut callback);

            coord_x >>= 1;
            coord_y >>= 1;
        });
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

    fn update(&mut self, bounds: Rectangle, index: usize) {
        let (b, _) = &mut self.items[index];
        *b = bounds;
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

unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> Send
    for UpSearchQuadTreeOriginal<T, MAX_LEVEL>
{
}
unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8> Sync
    for UpSearchQuadTreeOriginal<T, MAX_LEVEL>
{
}
