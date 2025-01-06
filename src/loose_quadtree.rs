use crate::rect::Rectangle;

use std::collections::HashMap;
use std::hash::Hash;
use std::ptr;

pub struct LooseQuadTree<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> {
    pub root: Box<LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>>,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    items: HashMap<T, *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>>,
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize>
    LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
    pub fn new(world_bounds: Rectangle) -> LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS> {
        Self::create::<false>(world_bounds)
    }

    pub fn new_fit(world_bounds: Rectangle) -> LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS> {
        Self::create::<true>(world_bounds)
    }

    pub fn create<const FIT: bool>(
        world_bounds: Rectangle,
    ) -> LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS> {
        let n = 1usize << (MAX_LEVEL - 1);
        assert!(MAX_LEVEL as usize > 0, "`MAX_LEVEL` cannot be zero.");
        assert!(
            n.checked_mul(n).is_some(),
            "`MAX_LEVEL` is too large and will cause overflow."
        );

        let mut root_width = world_bounds.get_width();
        let mut root_height = world_bounds.get_height();

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

        let looseness = 2.0;
        let loose_bounds = root_bounds.scale(looseness);

        LooseQuadTree {
            root: Box::new(LooseQuadTreeNode::<T, MAX_LEVEL, MAX_ITEMS>::new(
                1,
                loose_bounds,
                ptr::null_mut(),
            )),
            root_bounds,
            world_bounds,
            items: HashMap::new(),
        }
    }

    /// Return the real bounds of the root.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Insert an item into the LooseQuadTree.
    pub fn insert(&mut self, bounds: Rectangle, item: T) {
        self.items.insert(item, std::ptr::null_mut());
        if self.world_bounds.contains_point(bounds.get_center()) {
            self.root.insert(bounds, item, &mut self.items);
        } else {
            self.root.add(bounds, item, &mut self.items);
        }
    }

    /// Remove an item from the LooseQuadTree.
    pub fn remove(&mut self, item: T) {
        let node = self.items.get(&item).expect("Removal item not found.");

        let node = unsafe { &mut **node };
        node.remove(item);

        self.items.remove(&item);
    }

    /// Update the bounds of an item and, if necessary, its position in the LooseQuadTree.
    pub fn update(&mut self, bounds: Rectangle, item: T) {
        let item_locations = &mut self.items;
        let old_node = item_locations.get(&item).expect("Update item not found.");

        unsafe {
            let old_node = *old_node;
            let new_node = {
                let mut new_node = &mut *old_node;
                let (x, y) = bounds.get_center();
                while !new_node.loose_bounds.scale(0.5).contains_point((x, y))
                    && new_node.level != 1
                {
                    new_node = &mut *new_node.parent;
                }

                let width = bounds.get_width();
                let height = bounds.get_height();
                let mut new_node = new_node as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                while let Some(children) = (*new_node).children.as_mut() {
                    if width > (*new_node).loose_bounds.get_width() / 4.0
                        || height > (*new_node).loose_bounds.get_height() / 4.0
                    {
                        break;
                    }

                    let [tl, tr, bl, br] = children.as_mut();
                    let (x, y) = bounds.get_center();
                    let (center_x, center_y) = (*new_node).loose_bounds.get_center();
                    if y < center_y {
                        if x < center_x {
                            new_node = tl as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                            continue;
                        } else {
                            new_node = tr as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                            continue;
                        }
                    } else if x < center_x {
                        new_node = bl as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                        continue;
                    } else {
                        new_node = br as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                        continue;
                    }
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

            LooseQuadTree::merge(old_node);
        }
    }

    fn merge(mut node: &mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>) {
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

    /// Search in the LooseQuadTree by the given area. Execute the callback function for each item found.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        self.root.search(bounds, &mut callback);
    }
}

pub struct LooseQuadTreeNode<T: Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> {
    pub level: u8,
    pub items: Vec<(Rectangle, T)>,
    pub loose_bounds: Rectangle,
    pub children: Option<Box<[LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>; 4]>>,
    parent: *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>,
}

impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize>
    LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>
{
    fn new(
        lv: u8,
        loose_bounds: Rectangle,
        parent: *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>,
    ) -> LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS> {
        LooseQuadTreeNode {
            level: lv,
            items: Vec::with_capacity(MAX_ITEMS),
            loose_bounds,
            children: None,
            parent,
        }
    }

    fn create_child(&mut self, rect: Rectangle) -> LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS> {
        LooseQuadTreeNode::new(self.level + 1, rect, self as *mut Self)
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

        let (center_x, center_y) = self.loose_bounds.get_center();
        let width = self.loose_bounds.get_width() / 2.0;
        let height = self.loose_bounds.get_height() / 2.0;
        let offset_x = width / 4.0;
        let offset_y = height / 4.0;

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
        items.into_iter().for_each(|(bounds, item)| {
            self.insert(bounds, item, item_locations);
        });
    }

    fn insert(&mut self, bounds: Rectangle, item: T, item_locations: &mut HashMap<T, *mut Self>) {
        if let Some(children) = self.children.as_mut() {
            if bounds.get_width() <= self.loose_bounds.get_width() / 4.0
                && bounds.get_height() <= self.loose_bounds.get_height() / 4.0
            {
                let [tl, tr, bl, br] = children.as_mut();
                let (x, y) = bounds.get_center();
                let (center_x, center_y) = self.loose_bounds.get_center();
                if y < center_y {
                    if x < center_x {
                        return tl.insert(bounds, item, item_locations);
                    } else {
                        return tr.insert(bounds, item, item_locations);
                    }
                } else if x < center_x {
                    return bl.insert(bounds, item, item_locations);
                } else {
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
                    tl.search(bounds, callback);
                }
                if bounds.max_x > x_left {
                    tr.search(bounds, callback);
                }
            }
            if bounds.max_y > y_top {
                if bounds.min_x < x_right {
                    bl.search(bounds, callback);
                }
                if bounds.max_x > x_left {
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
    for LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
}
unsafe impl<T: Copy + Eq + Hash, const MAX_LEVEL: u8, const MAX_ITEMS: usize> Sync
    for LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
}
