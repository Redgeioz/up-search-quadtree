use crate::rect::Rectangle;

use std::collections::HashMap;
use std::hash::Hash;
use std::ptr;

/// An implementation of a quadtree that can be dynamically updated.
pub struct QuadTree<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> {
    root: Box<QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>>,
    world_bounds: Rectangle,
    items: HashMap<T, *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>>,
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize>
    QuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
    /// Create a new quadtree with the given bounds.
    ///
    /// Force using a square as the bounds of each node. This usually makes searches more efficient.
    pub fn new(world_bounds: Rectangle) -> QuadTree<T, MAX_LEVEL, MAX_ITEMS> {
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let bounds_width = world_bounds.get_width();
        let bounds_height = world_bounds.get_height();
        let len = bounds_width.max(bounds_height);
        let (min_x, min_y) = world_bounds.get_min();
        let center_x = min_x + len * 0.5;
        let center_y = min_y + len * 0.5;

        let root_bounds = Rectangle::center_rect(center_x, center_y, len, len);

        QuadTree {
            root: Box::new(QuadTreeNode::<T, MAX_LEVEL, MAX_ITEMS>::new(
                1,
                root_bounds,
                ptr::null_mut(),
            )),
            world_bounds,
            items: HashMap::new(),
        }
    }

    /// Create a new quadtree with the given bounds.
    ///
    /// Make the bounds of each node fit the given bounds instead of forcing it to be square.
    pub fn new_fit(world_bounds: Rectangle) -> QuadTree<T, MAX_LEVEL, MAX_ITEMS> {
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        QuadTree {
            root: Box::new(QuadTreeNode::<T, MAX_LEVEL, MAX_ITEMS>::new(
                1,
                world_bounds.clone(),
                ptr::null_mut(),
            )),
            world_bounds,
            items: HashMap::new(),
        }
    }

    /// Return the bounds of the root.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root.bounds
    }

    /// Insert an item into the quadtree.
    pub fn insert(&mut self, bounds: Rectangle, item: T) {
        self.items.insert(item, std::ptr::null_mut());
        if self.world_bounds.contains(&bounds) {
            self.root.insert(bounds, item, &mut self.items);
        } else {
            self.root.add(bounds, item, &mut self.items);
        }
    }

    /// Remove an item from the quadtree.
    pub fn remove(&mut self, item: T) {
        let node = self.items.get(&item).expect("Removal item not found.");

        let node = unsafe { &mut **node };
        node.remove(item);

        self.items.remove(&item);
        QuadTree::merge(node);
    }

    /// Update the bounds of an item and, if necessary, its position in the quadtree.
    pub fn update(&mut self, bounds: Rectangle, item: T) {
        let item_locations = &mut self.items;
        let old_node = item_locations.get(&item).expect("Update item not found.");

        unsafe {
            let old_node = *old_node;
            let new_node = {
                let mut new_node = &mut *old_node;
                while !new_node.bounds.contains(&bounds) && new_node.level != 1 {
                    new_node = &mut *new_node.parent;
                }

                let mut new_node = new_node as *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                while let Some(children) = (*new_node).children.as_mut() {
                    let [tl, tr, bl, br] = children.as_mut();
                    let (x, y) = tl.bounds.get_max();
                    if bounds.max_y < y {
                        if bounds.max_x < x {
                            new_node = tl as *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                            continue;
                        }
                        if bounds.min_x > x {
                            new_node = tr as *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                            continue;
                        }
                    } else if bounds.min_y > y {
                        if bounds.max_x < x {
                            new_node = bl as *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                            continue;
                        }
                        if bounds.min_x > x {
                            new_node = br as *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                            continue;
                        }
                    }
                    break;
                }

                if new_node == old_node {
                    (*old_node).update(bounds, item);
                    return;
                }

                new_node
            };

            let old_node = &mut *old_node;
            let new_node = &mut *new_node;

            old_node.remove(item);
            new_node.add(bounds, item, item_locations);

            QuadTree::merge(old_node);
        }
    }

    fn merge(mut node: &mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>) {
        while node.level > 1 {
            if let Some(children) = node.children.as_ref() {
                for child in children.iter() {
                    if child.children.is_some() || !child.items.is_empty() {
                        return;
                    }
                }
                node.children = None;
            } else {
                node = unsafe { &mut *node.parent };
            }
        }
    }

    /// Search in the quadtree by the given area. Execute the callback function for each item found.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        self.root.search(bounds, &mut callback);
    }
}

pub struct QuadTreeNode<T: Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> {
    level: u8,
    items: Vec<(Rectangle, T)>,
    bounds: Rectangle,
    children: Option<Box<[QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>; 4]>>,
    parent: *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>,
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize>
    QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>
{
    fn new(
        lv: u8,
        bounds: Rectangle,
        parent: *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>,
    ) -> QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS> {
        QuadTreeNode {
            level: lv,
            items: Vec::with_capacity(MAX_ITEMS),
            bounds,
            children: None,
            parent,
        }
    }

    fn create_child(&mut self, rect: Rectangle) -> QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS> {
        QuadTreeNode::new(self.level + 1, rect, self as *mut Self)
    }

    fn add(&mut self, bounds: Rectangle, item: T, item_locations: &mut HashMap<T, *mut Self>) {
        *item_locations.get_mut(&item).unwrap() = self as *mut Self;
        self.items.push((bounds, item));
        self.split(item_locations);
    }

    fn update(&mut self, bounds: Rectangle, item: T) {
        let (b, _) = self
            .items
            .iter_mut()
            .find(|(_, stored)| *stored == item)
            .expect("Item not found");

        *b = bounds;
    }

    fn remove(&mut self, item: T) -> T {
        let i = self
            .items
            .iter()
            .position(|(_, stored)| *stored == item)
            .expect("Item not found.");

        let (_, item) = self.items.remove(i);
        item
    }

    fn split(&mut self, item_locations: &mut HashMap<T, *mut Self>) {
        if self.children.is_some() || self.items.len() <= MAX_ITEMS || self.level >= MAX_LEVEL {
            return;
        }

        let (center_x, center_y) = self.bounds.get_center();
        let width = self.bounds.get_width() / 2.0;
        let height = self.bounds.get_height() / 2.0;
        let offset_x = width / 2.0;
        let offset_y = height / 2.0;

        let children = [0, 1, 2, 3].map(|i| {
            #[rustfmt::skip]
            let sign_x = ((i &  1) * 2) as f64 - 1.0;
            let sign_y = ((i >> 1) * 2) as f64 - 1.0;

            self.create_child(Rectangle::center_rect(
                center_x + sign_x * offset_x,
                center_y + sign_y * offset_y,
                width,
                height,
            ))
        });

        self.children = Some(Box::new(children));
        let items = std::mem::replace(&mut self.items, Vec::with_capacity(MAX_ITEMS));
        items
            .into_iter()
            .for_each(|(bounds, item)| self.insert(bounds, item, item_locations));
    }

    fn insert(&mut self, bounds: Rectangle, item: T, item_locations: &mut HashMap<T, *mut Self>) {
        if let Some(children) = self.children.as_mut() {
            let [tl, tr, bl, br] = children.as_mut();
            // The center of this node
            let (x, y) = tl.bounds.get_max();
            if bounds.max_y < y {
                if bounds.max_x < x {
                    return tl.insert(bounds, item, item_locations);
                }
                if bounds.min_x > x {
                    return tr.insert(bounds, item, item_locations);
                }
            } else if bounds.min_y > y {
                if bounds.max_x < x {
                    return bl.insert(bounds, item, item_locations);
                }
                if bounds.min_x > x {
                    return br.insert(bounds, item, item_locations);
                }
            }
        }

        // The item is not contained in any child.
        // So insert it into self here.
        self.add(bounds, item, item_locations);
    }

    fn search(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        self.children.iter().for_each(|children| {
            let [tl, tr, bl, br] = children.as_ref();
            // The center of this node
            let (x, y) = tl.bounds.get_max();
            if bounds.min_y < y {
                if bounds.min_x < x {
                    tl.search(bounds, callback);
                }
                if bounds.max_x > x {
                    tr.search(bounds, callback);
                }
            }
            if bounds.max_y > y {
                if bounds.min_x < x {
                    bl.search(bounds, callback);
                }
                if bounds.max_x > x {
                    br.search(bounds, callback);
                }
            }
        });

        self.search_items(bounds, callback);
    }

    fn search_items(&self, bounds: &Rectangle, callback: &mut impl FnMut(T)) {
        self.items
            .iter()
            .filter(|(b, _)| b.intersects(bounds))
            .for_each(|(_, item)| callback(*item));
    }
}

unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> Send
    for QuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
}
unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> Sync
    for QuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
}
