use crate::handle::{Entry, ItemHandle, NodeLocation, Slots};
use crate::rect::Rectangle;

use std::ptr;

/// Where the entry of each live handle is stored.
type NodeSlots<T, const MAX_LEVEL: u8, const MAX_ITEMS: usize> =
    Slots<NodeLocation<QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>>>;

/// An implementation of a quadtree that can be dynamically updated.
pub struct QuadTree<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> {
    root: Box<QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>>,
    world_bounds: Rectangle,
    slots: NodeSlots<T, MAX_LEVEL, MAX_ITEMS>,
}

impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> QuadTree<T, MAX_LEVEL, MAX_ITEMS> {
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
            slots: Slots::new(),
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
            slots: Slots::new(),
        }
    }

    /// Return the bounds of the root.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root.bounds
    }

    /// Insert an item into the quadtree and return the handle that identifies
    /// it for [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        // `add` records the location once the entry has reached its node.
        let handle = self.slots.allocate(NodeLocation::detached());
        let entry = Entry {
            bounds,
            item,
            handle,
        };
        if self.world_bounds.contains(&entry.bounds) {
            self.root.insert(entry, &mut self.slots);
        } else {
            self.root.add(entry, &mut self.slots);
        }
        handle
    }

    /// Remove an item from the quadtree.
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let NodeLocation { node, index } = self.slots.get(handle);

        let node = unsafe { &mut *node };
        let entry = node.take_entry(index as usize, &mut self.slots);
        debug_assert_eq!(entry.handle, handle);

        self.slots.free(handle);
        QuadTree::merge(node);
    }

    /// Update the bounds of an item and, if necessary, its position in the quadtree.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let NodeLocation {
            node: old_node,
            index,
        } = self.slots.get(handle);

        unsafe {
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
                    let old_node = &mut *old_node;
                    old_node.items[index as usize].bounds = bounds;
                    return;
                }

                new_node
            };

            let old_node = &mut *old_node;
            let new_node = &mut *new_node;

            let entry = old_node.take_entry(index as usize, &mut self.slots);
            new_node.add(Entry { bounds, ..entry }, &mut self.slots);

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

pub struct QuadTreeNode<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> {
    level: u8,
    items: Vec<Entry<T>>,
    bounds: Rectangle,
    children: Option<Box<[QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>; 4]>>,
    parent: *mut QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>,
}

impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> QuadTreeNode<T, MAX_LEVEL, MAX_ITEMS> {
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

    /// Store `entry` in this node and record where it went.
    fn add(&mut self, entry: Entry<T>, slots: &mut NodeSlots<T, MAX_LEVEL, MAX_ITEMS>) {
        *slots.get_mut(entry.handle) = NodeLocation {
            node: self,
            index: self.items.len() as u32,
        };
        self.items.push(entry);
        self.split(slots);
    }

    /// Take the entry at `index` out of this node. The entry that fills the
    /// gap has its slot updated, so handles stay valid without any lookup.
    fn take_entry(
        &mut self,
        index: usize,
        slots: &mut NodeSlots<T, MAX_LEVEL, MAX_ITEMS>,
    ) -> Entry<T> {
        let entry = self.items.swap_remove(index);
        if let Some(moved) = self.items.get(index) {
            slots.get_mut(moved.handle).index = index as u32;
        }
        entry
    }

    fn split(&mut self, slots: &mut NodeSlots<T, MAX_LEVEL, MAX_ITEMS>) {
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
            .for_each(|entry| self.insert(entry, slots));
    }

    fn insert(&mut self, entry: Entry<T>, slots: &mut NodeSlots<T, MAX_LEVEL, MAX_ITEMS>) {
        if let Some(children) = self.children.as_mut() {
            let [tl, tr, bl, br] = children.as_mut();
            // The center of this node
            let (x, y) = tl.bounds.get_max();
            let bounds = &entry.bounds;
            if bounds.max_y < y {
                if bounds.max_x < x {
                    return tl.insert(entry, slots);
                }
                if bounds.min_x > x {
                    return tr.insert(entry, slots);
                }
            } else if bounds.min_y > y {
                if bounds.max_x < x {
                    return bl.insert(entry, slots);
                }
                if bounds.min_x > x {
                    return br.insert(entry, slots);
                }
            }
        }

        // The item is not contained in any child.
        // So insert it into self here.
        self.add(entry, slots);
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
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }
}

unsafe impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> Send
    for QuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
}
unsafe impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> Sync
    for QuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
}
