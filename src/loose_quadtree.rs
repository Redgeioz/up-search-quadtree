use crate::handle::{Entry, ItemHandle, NodeLocation, Slots};
use crate::rect::Rectangle;

use std::ptr;

/// Location of the entry of each live handle.
type NodeSlots<T, const MAX_LEVEL: u8, const MAX_ITEMS: usize> =
    Slots<NodeLocation<LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>>>;

pub struct LooseQuadTree<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> {
    pub root: Box<LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>>,
    root_bounds: Rectangle,
    world_bounds: Rectangle,
    slots: NodeSlots<T, MAX_LEVEL, MAX_ITEMS>,
}

impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS> {
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
            slots: Slots::new(),
        }
    }

    /// Return the bounds of the root without looseness.
    pub fn get_bounds(&self) -> &Rectangle {
        &self.root_bounds
    }

    /// Insert an item into the tree and return the handle that identifies it
    /// for [`update`](Self::update) and [`remove`](Self::remove).
    pub fn insert(&mut self, bounds: Rectangle, item: T) -> ItemHandle {
        // `add` records the location once the entry has reached its node.
        let handle = self.slots.allocate(NodeLocation::detached());
        let entry = Entry {
            bounds,
            item,
            handle,
        };
        if self.world_bounds.contains_point(entry.bounds.get_center()) {
            self.root.insert(entry, &mut self.slots);
        } else {
            self.root.add(entry, &mut self.slots);
        }
        handle
    }

    /// Remove an item from the tree.
    ///
    /// Panics if the handle has already been removed.
    pub fn remove(&mut self, handle: ItemHandle) {
        let NodeLocation { node, index } = self.slots.get(handle);

        let node = unsafe { &mut *node };
        let entry = node.take_entry(index as usize, &mut self.slots);
        debug_assert_eq!(entry.handle, handle);

        self.slots.free(handle);
        LooseQuadTree::merge(node);
    }

    /// Update the bounds of an item and, if necessary, its node.
    ///
    /// Panics if the handle has already been removed.
    pub fn update(&mut self, bounds: Rectangle, handle: ItemHandle) {
        let NodeLocation {
            node: old_node,
            index,
        } = self.slots.get(handle);

        unsafe {
            let new_node = {
                let (x, y) = bounds.get_center();
                let width = bounds.get_width();
                let height = bounds.get_height();

                let new_node = if !self.world_bounds.contains_point((x, y)) {
                    // As in `insert`, an item centered outside the world belongs
                    // to the root.
                    &mut *self.root as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>
                } else {
                    // Climb while `insert` would not descend this far: a node holds
                    // only items that fit half its loose bounds and are centered in
                    // its unloosened area.
                    let mut new_node = &mut *old_node;
                    while new_node.level != 1
                        && (width > new_node.loose_bounds.get_width() / 2.0
                            || height > new_node.loose_bounds.get_height() / 2.0
                            || !new_node.loose_bounds.scale(0.5).contains_point((x, y)))
                    {
                        new_node = &mut *new_node.parent;
                    }

                    // Then descend exactly as `insert` does.
                    let mut new_node = new_node as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                    while let Some(children) = (*new_node).children.as_mut() {
                        if width > (*new_node).loose_bounds.get_width() / 4.0
                            || height > (*new_node).loose_bounds.get_height() / 4.0
                        {
                            break;
                        }

                        let [tl, tr, bl, br] = children.as_mut();
                        let (center_x, center_y) = (*new_node).loose_bounds.get_center();
                        if y < center_y {
                            if x < center_x {
                                new_node = tl as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                            } else {
                                new_node = tr as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                            }
                        } else if x < center_x {
                            new_node = bl as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                        } else {
                            new_node = br as *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>;
                        }
                    }
                    new_node
                };

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

    /// Search for items whose bounds intersect `bounds`, calling `callback`
    /// for each.
    pub fn search(&self, bounds: &Rectangle, mut callback: impl FnMut(T)) {
        self.root.search(bounds, &mut callback);
    }
}

pub struct LooseQuadTreeNode<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> {
    pub level: u8,
    items: Vec<Entry<T>>,
    pub loose_bounds: Rectangle,
    pub children: Option<Box<[LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>; 4]>>,
    parent: *mut LooseQuadTreeNode<T, MAX_LEVEL, MAX_ITEMS>,
}

impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize>
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

    /// Store `entry` in this node and record its location.
    fn add(&mut self, entry: Entry<T>, slots: &mut NodeSlots<T, MAX_LEVEL, MAX_ITEMS>) {
        *slots.get_mut(entry.handle) = NodeLocation {
            node: self,
            index: self.items.len() as u32,
        };
        self.items.push(entry);
        self.split(slots);
    }

    /// Take the entry at `index` out of this node. The entry moved into the
    /// gap has its slot updated, so its handle stays valid without a lookup.
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
        items.into_iter().for_each(|entry| {
            self.insert(entry, slots);
        });
    }

    fn insert(&mut self, entry: Entry<T>, slots: &mut NodeSlots<T, MAX_LEVEL, MAX_ITEMS>) {
        if let Some(children) = self.children.as_mut() {
            let bounds = &entry.bounds;
            if bounds.get_width() <= self.loose_bounds.get_width() / 4.0
                && bounds.get_height() <= self.loose_bounds.get_height() / 4.0
            {
                let [tl, tr, bl, br] = children.as_mut();
                let (x, y) = bounds.get_center();
                let (center_x, center_y) = self.loose_bounds.get_center();
                if y < center_y {
                    if x < center_x {
                        return tl.insert(entry, slots);
                    } else {
                        return tr.insert(entry, slots);
                    }
                } else if x < center_x {
                    return bl.insert(entry, slots);
                } else {
                    return br.insert(entry, slots);
                }
            }
        }

        // No child fits the item, or the node has none: store it here.
        self.add(entry, slots);
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
            .filter(|entry| entry.bounds.intersects(bounds))
            .for_each(|entry| callback(entry.item));
    }
}

unsafe impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> Send
    for LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
}
unsafe impl<T: Copy, const MAX_LEVEL: u8, const MAX_ITEMS: usize> Sync
    for LooseQuadTree<T, MAX_LEVEL, MAX_ITEMS>
{
}
