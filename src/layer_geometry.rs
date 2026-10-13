//! Per-level geometry of the grid-based structures: exact size limits and
//! cell edges that let `update` confirm an item stays in its cell without
//! repeating the positioning arithmetic.

use crate::rect::Rectangle;

/// Per-level constants derived from the root bounds, plus the number of
/// objects currently stored on that level.
///
/// Everything except `object_count` is fixed for the lifetime of the tree.
/// The node size and the cell edges agree bit-for-bit with the arithmetic in
/// [`UpSearchQuadTree::position`](crate::UpSearchQuadTree::position), which lets `update` confirm that an item
/// stays in its node without repeating that computation.
#[derive(Default)]
pub(crate) struct LayerGeometry {
    // `search` reads only these first three fields of every level it visits.
    pub(crate) object_count: usize,
    /// Size of a cell, i.e. the root size divided by `2^(level - 1)`.
    ///
    /// This is also the largest object (inclusive) that `position` assigns to
    /// this level; see `matches_layer` for why the comparison is exact.
    pub(crate) node_width: f64,
    pub(crate) node_height: f64,
    /// Half the node size, i.e. the largest object of the next finer level.
    /// Cached rather than multiplied in `matches_layer`, which measured
    /// `update` 28% slower.
    pub(crate) half_node_width: f64,
    pub(crate) half_node_height: f64,
    /// Lower edge of every cell along each axis, relative to the root origin.
    /// Cell `i` covers the half-open interval `edges[i]..edges[i + 1]`.
    /// Left empty on the root and on degenerate layers to disable the update shortcut.
    ///
    /// Boxed slices keep the struct at 72 bytes; with `Vec`s it is 88 bytes,
    /// which measured `search` a few percent slower.
    pub(crate) cell_x_edges: Box<[f64]>,
    pub(crate) cell_y_edges: Box<[f64]>,
}

impl LayerGeometry {
    /// Geometry of `level`, whose grid has `cols` by `rows` cells, for a
    /// structure whose `position` divides coordinates by the node size.
    pub(crate) fn new(root_bounds: &Rectangle, level: usize, cols: usize, rows: usize) -> Self {
        // Preserve division here and in position/search: multiplying by a
        // reciprocal can round across a cell boundary.
        let (node_width, node_height) = Self::node_size(root_bounds, cols, rows);
        Self::build(
            root_bounds,
            level,
            cols,
            rows,
            |x| (x / node_width) as usize,
            |y| (y / node_height) as usize,
        )
    }

    /// Geometry of `level` for a structure whose `position` multiplies
    /// coordinates by these reciprocals of the node size instead of dividing.
    /// The cell edges then follow that arithmetic.
    pub(crate) fn with_reciprocal(
        root_bounds: &Rectangle,
        level: usize,
        cols: usize,
        rows: usize,
        inverse_node_width: f64,
        inverse_node_height: f64,
    ) -> Self {
        Self::build(
            root_bounds,
            level,
            cols,
            rows,
            |x| (x * inverse_node_width) as usize,
            |y| (y * inverse_node_height) as usize,
        )
    }

    /// Node size of a level whose grid has `cols` by `rows` cells: the root
    /// size divided by the larger count, since a grid trimmed to a non-square
    /// world keeps the cell size of the full square grid.
    pub(crate) fn node_size(root_bounds: &Rectangle, cols: usize, rows: usize) -> (f64, f64) {
        let cells_per_axis = cols.max(rows) as f64;
        (
            root_bounds.get_width() / cells_per_axis,
            root_bounds.get_height() / cells_per_axis,
        )
    }

    /// `cell_of_x` and `cell_of_y` map a coordinate relative to the root
    /// origin to its cell exactly as the structure's `position` does.
    fn build(
        root_bounds: &Rectangle,
        level: usize,
        cols: usize,
        rows: usize,
        cell_of_x: impl Fn(f64) -> usize,
        cell_of_y: impl Fn(f64) -> usize,
    ) -> Self {
        let root_width = root_bounds.get_width();
        let root_height = root_bounds.get_height();
        let (node_width, node_height) = Self::node_size(root_bounds, cols, rows);

        // A root item may have come from outside the world, and degenerate or
        // subnormal sizes break the exactness that `matches_layer` relies on.
        // Both keep using full positioning.
        let is_root = level == 1;
        let sizes_are_usable = [root_width, root_height, node_width, node_height]
            .into_iter()
            .all(|size| size.is_normal() && size > 0.0);
        if is_root || !sizes_are_usable {
            return Self {
                node_width,
                node_height,
                ..Self::default()
            };
        }

        Self {
            object_count: 0,
            node_width,
            node_height,
            half_node_width: node_width * 0.5,
            half_node_height: node_height * 0.5,
            cell_x_edges: Self::cell_edges(node_width, cols, cell_of_x),
            cell_y_edges: Self::cell_edges(node_height, rows, cell_of_y),
        }
    }

    // The predicates below are not generic but run inside the generic tree
    // methods, which other crates instantiate. `#[inline]` keeps them
    // inlinable there; without it `update` measured a few percent slower.

    /// Whether `matches_layer` and `matches_cell` are usable on this level.
    #[inline]
    pub(crate) fn has_update_shortcut(&self) -> bool {
        !self.cell_x_edges.is_empty()
    }

    #[inline]
    pub(crate) fn is_empty(&self) -> bool {
        self.object_count == 0
    }

    /// Whether `position` assigns an object of this size to this level.
    ///
    /// `position` puts an object here when `(root / size) as usize` lies in
    /// `2^(level - 1)..2^level`. Comparing against the node size reproduces
    /// that exactly: `node_width` is `root / 2^(level - 1)` without rounding,
    /// dividing the root by it yields exactly `2^(level - 1)`, and the next
    /// float above it already yields less. The halved node size (also exact)
    /// is the limit of the next finer level in the same way.
    #[inline]
    pub(crate) fn matches_layer(&self, width: f64, height: f64, is_finest_layer: bool) -> bool {
        // Degenerate sizes fall through to full positioning.
        if !(width > 0.0 && height > 0.0) {
            return false;
        }
        let fits_this_level = width <= self.node_width && height <= self.node_height;
        if !fits_this_level {
            return false;
        }
        // The finest level keeps everything that would otherwise go further
        // down. Elsewhere the object must be too large for the next finer level.
        is_finest_layer || width > self.half_node_width || height > self.half_node_height
    }

    /// Whether `position` maps a center at this offset from the root origin to `cell`.
    #[inline]
    pub(crate) fn matches_cell(&self, relative_center: (f64, f64), cell: (usize, usize)) -> bool {
        let (center_x, center_y) = relative_center;
        let (x, y) = cell;
        (self.cell_x_edges[x]..self.cell_x_edges[x + 1]).contains(&center_x)
            && (self.cell_y_edges[y]..self.cell_y_edges[y + 1]).contains(&center_y)
    }

    /// Lower edge of every cell as seen by `position`: `edges[i]` is the
    /// smallest relative coordinate `c` with `cell_of(c) == i`.
    ///
    /// `cell_size * i` may land on either side of the cast boundary, so walk
    /// through adjacent floats until the boundary is found.
    fn cell_edges(cell_size: f64, cell_count: usize, cell_of: impl Fn(f64) -> usize) -> Box<[f64]> {
        let mut edges = Vec::with_capacity(cell_count + 1);
        edges.push(f64::NEG_INFINITY); // Negative casts saturate to cell zero.
        for index in 1..cell_count {
            let mut edge = cell_size * index as f64;
            while cell_of(edge) < index {
                edge = edge.next_up();
            }
            while cell_of(edge.next_down()) >= index {
                edge = edge.next_down();
            }
            edges.push(edge);
        }
        edges.push(f64::INFINITY); // The last cell clamps all larger coordinates.
        edges.into_boxed_slice()
    }
}
