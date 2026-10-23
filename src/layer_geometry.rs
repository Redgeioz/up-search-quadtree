//! Per-level geometry of the grid-based structures: exact size limits and
//! cell edges with which `update` confirms that an item stays in its cell
//! without repeating the positioning arithmetic.

use crate::rect::Rectangle;

/// Per-level constants derived from the root bounds, plus the number of
/// objects currently stored on the level.
///
/// Everything except `object_count` is fixed for the lifetime of the tree.
/// The node size and cell edges agree bit for bit with the positioning
/// arithmetic of the owning structure, such as
/// [`HierarchicalLooseGrid::position`](crate::HierarchicalLooseGrid::position),
/// so `update` can confirm that an item stays in its node without repeating
/// that computation.
#[derive(Default)]
pub(crate) struct LayerGeometry {
    // `search` reads only the first three fields of each level it visits.
    pub(crate) object_count: usize,
    /// Size of a cell: the root size divided by `2^(level - 1)`.
    ///
    /// Also the largest object size (inclusive) that `position` assigns to
    /// this level; `matches_layer` explains why the comparison is exact.
    pub(crate) node_width: f64,
    pub(crate) node_height: f64,
    /// Half the node size: the largest object size of the next finer level.
    /// Cached; multiplying in `matches_layer` instead measured `update` 28%
    /// slower.
    pub(crate) half_node_width: f64,
    pub(crate) half_node_height: f64,
    /// Lower edge of every cell along each axis, relative to the root origin;
    /// cell `i` covers the half-open interval `edges[i]..edges[i + 1]`. Empty
    /// on the root and on degenerate levels, which disables the update
    /// shortcut.
    ///
    /// Boxed slices keep the struct at 72 bytes; `Vec`s make it 88 bytes,
    /// which measured `search` a few percent slower.
    pub(crate) cell_x_edges: Box<[f64]>,
    pub(crate) cell_y_edges: Box<[f64]>,
}

impl LayerGeometry {
    /// Geometry of `level`, whose grid has `cols` by `rows` cells, for a
    /// structure whose `position` multiplies coordinates by these reciprocals
    /// of the node size; the cell edges follow that arithmetic. `position`
    /// must use the same reciprocals, or the update shortcut could keep an
    /// item in a cell that `position` would not choose: multiplying by
    /// another value, or dividing by the node size, can round across a cell
    /// boundary.
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
    /// size divided by the larger count, as a grid trimmed to a non-square
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

        // Root items may come from outside the world, and degenerate or
        // subnormal sizes break the exactness `matches_layer` relies on; both
        // cases keep full positioning.
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

    // The predicates below are not generic but run inside generic tree
    // methods, which other crates instantiate; `#[inline]` keeps them
    // inlinable there. Without it, `update` measured a few percent slower.

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
    /// `position` assigns an object here when `(root / size) as usize` lies in
    /// `2^(level - 1)..2^level`. Comparing with the node size reproduces this
    /// exactly: `node_width` is `root / 2^(level - 1)` without rounding, so
    /// dividing the root by it yields exactly `2^(level - 1)`, and dividing by
    /// the next float above it yields less. The halved node size, also exact,
    /// bounds the next finer level in the same way.
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
        // The finest level also takes everything smaller; any other level
        // requires the object to be too large for the next finer level.
        is_finest_layer || width > self.half_node_width || height > self.half_node_height
    }

    /// Whether `position` maps a center at this root-relative offset to `cell`.
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
    /// `cell_size * i` may land on either side of the cast boundary, so the
    /// boundary is located by stepping through adjacent floats.
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

/// The level an item of extent `size` along one axis of a root of extent
/// `root` belongs on, before clamping to the levels of the structure:
/// `usize::BITS - ((root / size) as usize).leading_zeros()`, the bit length of
/// how many times the item fits across the root, which every hierarchical
/// structure's `position` computes on both axes.
///
/// For a normal positive `size` and `root` it follows from the exponents
/// without the division. The rounded quotient reaches `2^k` exactly when
/// `size * 2^k <= root`, as `matches_layer` explains for `k` a level, and that
/// holds when the exponent of `size` plus `k` is below that of `root`, or
/// equal to it with a mantissa no larger. The largest such `k`, plus one, is
/// the bit length; as `size` is normal, `root / 2^k` stays normal up to that
/// `k`, which the argument of `matches_layer` needs. The result may fall
/// below zero or exceed 64 where the division's bit length stops at those
/// bounds; clamped to the levels, the two agree. Zero, subnormal, negative,
/// infinite and NaN sizes and roots keep the division.
#[inline]
pub(crate) fn size_level(root: f64, size: f64) -> i64 {
    const MANTISSA: u64 = (1 << 52) - 1;
    let (root_bits, size_bits) = (root.to_bits(), size.to_bits());
    // With the sign bit, which is clear for positive numbers; normal positive
    // numbers have a biased exponent of 1 to 2046.
    let (root_exponent, size_exponent) = ((root_bits >> 52) as i64, (size_bits >> 52) as i64);
    if (1..=2046).contains(&size_exponent) && (1..=2046).contains(&root_exponent) {
        let larger_mantissa = (size_bits & MANTISSA) > (root_bits & MANTISSA);
        root_exponent - size_exponent - larger_mantissa as i64 + 1
    } else {
        (usize::BITS - ((root / size) as usize).leading_zeros()) as i64
    }
}

/// Corners `[min_x, min_y, max_x, max_y]` of a query, given relative to the
/// root origin, in half nodes of the finest level: the minima rounded up and
/// the maxima truncated toward zero. [`scan_range`] derives the nodes to scan
/// on every level from them. Rounding each level's coordinates instead
/// measured large aligned queries 27-34% slower in the hierarchical loose,
/// column-and-row and neighbor grids, for the scans before they included
/// their ends, which took `round` and `ceil` on every level; both compile to
/// C runtime calls on x86-64 without SSE4.1. Truncation, which saves the call
/// `floor` would make, rounds the maxima down except below zero, where
/// [`scan_range`] ends at node zero for either. The minima take their
/// ceilings from truncation too, by [`ceil_from_truncation`], which saves the
/// two calls to `ceil`. Against `ceil` itself, this measured large aligned
/// queries 2-6% faster in the hierarchical loose and neighbor grids, with and
/// without code alignment forced. The mixed distribution of the hierarchical
/// loose grid measured 2-5% slower in default builds, in all six rounds of
/// two runs, as an earlier version had by 4.5%, but about 3% faster with
/// alignment forced, which points to code layout; the other queries moved
/// within the few percent by which binaries of the same code differ.
#[inline]
pub(crate) fn half_node_corners(relative: [f64; 4], finest_inverse: (f64, f64)) -> [i64; 4] {
    let (half_x, half_y) = (2.0 * finest_inverse.0, 2.0 * finest_inverse.1);
    let (min_x, min_y) = (relative[0] * half_x, relative[1] * half_y);
    [
        ceil_from_truncation(min_x, min_x as i64),
        ceil_from_truncation(min_y, min_y as i64),
        (relative[2] * half_x) as i64,
        (relative[3] * half_y) as i64,
    ]
}

/// Nodes `[min_x, min_y, max_x, max_y]` that the scan of a level `shift`
/// levels coarser than the finest visits for a query with the given
/// [`half_node_corners`], the maxima clamped to `last`, the last column and
/// row. The items of a node reach half a node past it, so for a corner `n`
/// in nodes of that level these are `ceil(n - 1.5)` and `floor(n + 0.5)`:
/// the nodes whose items reach the query, and also a node that the query
/// starts exactly half a node past or ends exactly half a node before, as an
/// item whose bounds round a float past that half node still meets such a
/// query. The scan used to stop short of that node, at `round(n) - 1` and
/// `ceil(n - 0.5)`. Rounding can take an item further past its half node,
/// as when its width rounds to the node size, so a query that ends a float
/// before that half node may still miss it. Both bounds follow by shifts
/// alone: the reciprocal node sizes of the levels differ by exact powers of
/// two, so `n` on that level is the finest-level value scaled by `2^-shift`
/// without rounding, and floor and ceil of a quotient by a power of two
/// follow from those of the dividend. Only a minimum beyond the last node
/// may differ, for a corner at infinity or `2^63` half nodes and more, which
/// leaves the range empty either way.
///
/// A node at least two nodes in from both ends of the range lies, with the
/// half node its items may reach on each side, inside the query, ends
/// included. Its items, centered in it and no larger, then meet the query
/// without a test, points included: half a node separates their centers from
/// the query's edges. That takes corners that compare; the searches return
/// before scanning for a query with a NaN corner, which meets nothing.
#[inline]
pub(crate) fn scan_range(half: [i64; 4], shift: u32, last: (usize, usize)) -> [usize; 4] {
    // ceil(n - 1.5) is ceil(2n) / 2, rounded down, less one, and ceil(2n) on
    // this level is the finest ceil(2n) divided by 2^shift, rounded up;
    // saturating keeps a query at infinity in range.
    let min = |half: i64| {
        ((half.saturating_add((1 << shift) - 1) >> (shift + 1)) - 1).max(0) as usize
    };
    // floor(n + 0.5) is floor(2n + 1) / 2, rounded down.
    let max = |half: i64, last: usize| {
        (((half >> shift).saturating_add(1)) >> 1).clamp(0, last as i64) as usize
    };
    [min(half[0]), min(half[1]), max(half[2], last.0), max(half[3], last.1)]
}

/// First node whose items may reach a query, for `n` the query's minimum in
/// nodes of the grid or level scanned, plus half the shrink of the level's
/// bound, if any: `ceil(n - 1.5)`, zero at least, as in [`scan_range`]. The
/// items of a node reach half a node past it, and one whose bounds round a
/// float further still meets a query that starts exactly there, which the
/// scans used to leave out by starting at `round(n) - 1`. `ceil(n - 1.5)` is
/// `ceil(2n)` halved, rounded down, less one; the ceiling comes from the
/// truncation by [`ceil_from_truncation`], without a call to `ceil`. From
/// `2^62` nodes on this gives `2^62 - 2`, beyond every node.
#[inline]
pub(crate) fn first_reaching_node(n: f64) -> usize {
    let twice = 2.0 * n;
    ((ceil_from_truncation(twice, twice as i64) >> 1) - 1).max(0) as usize
}

/// Last node whose items may reach a query, for `n` the query's maximum in
/// nodes of the grid or level scanned, less half the shrink of the level's
/// bound, if any: `floor(n + 0.5)`, zero at least, as in [`scan_range`],
/// which includes a node that the query ends exactly half a node before.
/// `floor(n + 0.5)` is `floor(2n) + 1` halved and rounded down; the cast,
/// which truncates, stands in for `floor`, as the two differ only below zero,
/// where both give zero. From `2^62` nodes on this gives `2^62 - 1` where
/// the cast to `usize` would saturate higher; the scans clamp both to their
/// last node.
#[inline]
pub(crate) fn last_reaching_node(n: f64) -> usize {
    let twice = (2.0 * n) as i64;
    (twice.saturating_add(1) >> 1).max(0) as usize
}

/// `value.ceil() as i64`, from `truncated`, which is `value as i64`: one more
/// where the truncation fell short, which happens only above zero. `ceil`
/// compiles to a C runtime call on x86-64 without SSE4.1. Beyond `2^53` the
/// floats are whole and the comparison exact; the addition saturates where
/// the cast did, at `+infinity` and from `2^63` on.
#[inline]
pub(crate) fn ceil_from_truncation(value: f64, truncated: i64) -> i64 {
    truncated.saturating_add(((truncated as f64) < value) as i64)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::prelude::*;

    /// The definition `size_level` must reproduce.
    fn by_division(root: f64, size: f64) -> i64 {
        (usize::BITS - ((root / size) as usize).leading_zeros()) as i64
    }

    /// Up to `count` floats on either side of `value`, and `value` itself.
    fn around(value: f64, count: usize) -> Vec<f64> {
        let mut values = vec![value];
        let (mut up, mut down) = (value, value);
        for _ in 0..count {
            up = up.next_up();
            down = down.next_down();
            values.extend([up, down]);
        }
        values
    }

    /// `ceil(n - 1.5)` and `floor(n + 0.5)`, saturated at zero, computed
    /// exactly. `n - 1.5` and `n + 0.5` may round onto a whole number, and
    /// from `2^52` on cannot even hold the half; the whole part and the
    /// fraction, both exact, give the rounding instead, the sum taken in
    /// integers that neither round nor overflow.
    fn reaching_by_rounding(n: f64) -> (usize, usize) {
        let floor = n.floor();
        let (whole, fraction) = (floor as i128, n - floor);
        let clamp = |v: i128| v.clamp(0, u64::MAX as i128) as usize;
        (
            clamp(whole.saturating_sub(1).saturating_add((fraction > 0.5) as i128)),
            clamp(whole.saturating_add((fraction >= 0.5) as i128)),
        )
    }

    /// `scan_range` must give exactly what rounding the product with each
    /// level's own reciprocal node size gives, `ceil(n - 1.5)` and
    /// `floor(n + 0.5)`, for reciprocals built as the grids build them, and
    /// for corners on and next to node and half-node edges, outside the root,
    /// infinite and NaN. Minima beyond the last node all leave the range
    /// empty, so they are compared capped at `last + 1`.
    #[test]
    fn scan_range_agrees_with_the_rounding_of_each_level() {
        let calc_min = |n: f64| reaching_by_rounding(n).0;
        let calc_max = |n: f64, last: usize| reaching_by_rounding(n).1.min(last);
        let mut rng = StdRng::seed_from_u64(9);
        let roots = [
            Rectangle::new(0.0, 0.0, 20000.0, 20000.0),
            Rectangle::new(-128.0, -128.0, 896.0, 384.0),
            Rectangle::new(0.3, -0.7, 1000.1, 333.3),
            Rectangle::new(-3.7e5, 1.1e4, 2.9e6, 1.7e6),
        ];
        for max_level in [1usize, 2, 7, 10] {
            for root in &roots {
                // Reciprocals of every level as the grids compute them.
                let inverses: Vec<(f64, f64)> = (1..=max_level)
                    .map(|level| {
                        let cells = 1usize << (level - 1);
                        let (width, height) = LayerGeometry::node_size(root, cells, cells);
                        (1.0 / width, 1.0 / height)
                    })
                    .collect();
                let finest = inverses[max_level - 1];
                for _ in 0..20_000 {
                    // A coordinate in finest half nodes, often on or a few
                    // floats away from an edge, sometimes far outside.
                    let mut coordinate = |inverse: f64| -> f64 {
                        let half = 0.5 / inverse;
                        let value = match rng.gen_range(0..10) {
                            0 => rng.gen_range(-3.0..3.0) * half,
                            1 => f64::INFINITY,
                            2 => f64::NEG_INFINITY,
                            3 => f64::NAN,
                            4..=6 => rng.gen_range(-4i64..(4 << max_level)) as f64 * half,
                            _ => rng.gen_range(-0.1..1.1) * half * (2 << max_level) as f64,
                        };
                        let mut value = value;
                        for _ in 0..rng.gen_range(0..3) {
                            value = if rng.gen_bool(0.5) { value.next_up() } else { value.next_down() };
                        }
                        value
                    };
                    let relative = [
                        coordinate(finest.0),
                        coordinate(finest.1),
                        coordinate(finest.0),
                        coordinate(finest.1),
                    ];
                    let half = half_node_corners(relative, finest);
                    for level in 1..=max_level {
                        let (inverse_x, inverse_y) = inverses[level - 1];
                        let last = (1usize << (level - 1)) - 1;
                        let expected = [
                            calc_min(relative[0] * inverse_x).min(last + 1),
                            calc_min(relative[1] * inverse_y).min(last + 1),
                            calc_max(relative[2] * inverse_x, last),
                            calc_max(relative[3] * inverse_y, last),
                        ];
                        let shift = (max_level - level) as u32;
                        let [min_x, min_y, max_x, max_y] = scan_range(half, shift, (last, last));
                        assert_eq!(
                            [min_x.min(last + 1), min_y.min(last + 1), max_x, max_y],
                            expected,
                            "level {level} of {max_level}, root {root:?}, corners {relative:?}"
                        );
                    }
                }
            }
        }
    }

    /// `first_reaching_node` and `last_reaching_node` must give what
    /// `ceil(n - 1.5)` and `floor(n + 0.5)`, saturated at zero, give: for `n`
    /// on, next to and halfway between whole nodes, negative, near the powers
    /// of two where floats stop holding halves and where the integer casts
    /// saturate, random over the whole range of floats, and the special
    /// values. Results from `2^61` nodes on, beyond the last node of any grid,
    /// are compared capped there, as the scans clamp them to their last node
    /// or leave the range empty.
    #[test]
    fn reaching_nodes_agree_with_rounding() {
        let first = |n: f64| reaching_by_rounding(n).0;
        let last = |n: f64| reaching_by_rounding(n).1;
        let cap = 1usize << 61;
        let mut values = vec![
            0.0,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
        ];
        for whole in -10..=10 {
            values.extend([whole as f64, whole as f64 + 0.5]);
        }
        for power in 0..70 {
            let value = 2f64.powi(power);
            values.extend([value, value + 0.5, value - 0.5, value * 1.5]);
        }
        let mut rng = StdRng::seed_from_u64(31);
        for _ in 0..20000 {
            values.push(rng.gen_range(-3.0..300.0));
            values.push(rng.gen_range(-3i32..300) as f64 * 0.5);
            values.push(2f64.powf(rng.gen_range(-1074.0..1023.0)) * rng.gen_range(1.0..2.0));
        }
        for value in values {
            for n in around(value, 3).into_iter().flat_map(|n| [n, -n]) {
                assert_eq!(
                    first_reaching_node(n).min(cap),
                    first(n).min(cap),
                    "first, n = {n:e}"
                );
                assert_eq!(
                    last_reaching_node(n).min(cap),
                    last(n).min(cap),
                    "last, n = {n:e}"
                );
            }
        }
    }

    /// `ceil_from_truncation` must give what `ceil` and the saturating cast
    /// give: for values on, next to and halfway between whole numbers, on both
    /// sides of zero, near the powers of two where floats stop holding
    /// fractions and where the cast saturates, random over the whole range of
    /// floats, and the special values.
    #[test]
    fn ceil_from_truncation_agrees_with_ceil() {
        let mut values = vec![
            0.0,
            f64::MIN_POSITIVE,
            5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NAN,
        ];
        for whole in -10..=10 {
            values.extend([whole as f64, whole as f64 + 0.5]);
        }
        for power in 0..70 {
            let value = 2f64.powi(power);
            values.extend([value, value + 0.5, value - 0.5, value * 1.5]);
        }
        let mut rng = StdRng::seed_from_u64(29);
        for _ in 0..20000 {
            values.push(rng.gen_range(-3.0..300.0));
            values.push(rng.gen_range(-3i32..300) as f64 * 0.5);
            values.push(2f64.powf(rng.gen_range(-1074.0..1023.0)) * rng.gen_range(1.0..2.0));
        }
        for value in values {
            for v in around(value, 3).into_iter().flat_map(|v| [v, -v]) {
                assert_eq!(
                    ceil_from_truncation(v, v as i64),
                    v.ceil() as i64,
                    "v = {v:e}"
                );
            }
        }
    }

    /// Every level threshold `root / 2^k`, a few floats around it, random
    /// sizes over the whole range of floats and the special values all give
    /// the bit length of the division, up to the bounds where it stops.
    #[test]
    fn size_level_agrees_with_the_division() {
        let roots = [
            1024.0, 20000.0, 999.8, 334.0, 512.0, 1e-10, 3.7e5, 1e300,
            // Roots whose thresholds `root / 2^k` underflow for large `k`.
            2f64.powi(-958), 1e-300, f64::MIN_POSITIVE,
        ];
        let specials = [
            0.0,
            -0.0,
            f64::MIN_POSITIVE,
            f64::MIN_POSITIVE / 2.0,
            5e-324,
            f64::MAX,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
            -1.0,
            -1e-300,
        ];
        let mut rng = StdRng::seed_from_u64(17);
        for root in roots {
            let mut sizes: Vec<f64> = specials.to_vec();
            sizes.extend(around(root, 4));
            sizes.extend(around(root * 2.0, 4));
            for k in 0..70 {
                sizes.extend(around(root / 2f64.powi(k), 4));
            }
            for _ in 0..20000 {
                let exponent = rng.gen_range(-1074.0..1023.0);
                sizes.push(2f64.powf(exponent) * rng.gen_range(1.0..2.0));
            }
            for size in sizes {
                assert_eq!(
                    size_level(root, size).clamp(0, 64),
                    by_division(root, size),
                    "root {root:e}, size {size:e}"
                );
            }
        }
    }
}
