//! Bounds on how far items extend beyond their cells, shared by
//! [`LazyOverflowBoundGrid`], [`BoxedOverflowBoundGrid`],
//! [`SpanOverflowBoundGrid`] and the other overflow-bound grids.
//!
//! Positioning by size keeps every item within half a node of its cell, up
//! to floating-point rounding, and the hierarchical loose grid always scans
//! that far, with no margin for that rounding. Search needs only an
//! upper bound on the actual distances to scan less, so the lazy grids do not
//! track them exactly.
//!
//! Every array here lists the directions as left (-x), right (+x), top (-y)
//! and bottom (+y).
//!
//! [`LazyOverflowBoundGrid`]: crate::LazyOverflowBoundGrid
//! [`BoxedOverflowBoundGrid`]: crate::BoxedOverflowBoundGrid
//! [`SpanOverflowBoundGrid`]: crate::SpanOverflowBoundGrid

use crate::rect::Rectangle;

/// Distance by which `bounds` extends beyond each edge of `cell`, in the
/// direction order of this module; negative where it stays inside that edge.
#[inline]
pub(crate) fn cell_overflow(
    bounds: &Rectangle,
    origin: (f64, f64),
    node_size: (f64, f64),
    cell: (usize, usize),
) -> [f64; 4] {
    // Cell indices fit in `u32`, as in `GridLocation`; `u32` converts to `f64`
    // in one instruction, `usize` in several. Both conversions are exact.
    let (x, y) = (cell.0 as u32 as f64, cell.1 as u32 as f64);
    let min_x = origin.0 + node_size.0 * x;
    let max_x = origin.0 + node_size.0 * (x + 1.0);
    let min_y = origin.1 + node_size.1 * y;
    let max_y = origin.1 + node_size.1 * (y + 1.0);
    [
        min_x - bounds.min_x,
        bounds.max_x - max_x,
        min_y - bounds.min_y,
        bounds.max_y - max_y,
    ]
}

/// `low.min(value)` for a `low` that is never NaN, such as the lower end of a
/// box or span grown from infinity: a NaN `value` leaves `low`, as with
/// `f64::min`. Of two equal zeros it always keeps `low`, while `min` may
/// return either, so the sign of a zero end can differ from what `min` would
/// give; no comparison sees it.
/// `min` also handles a NaN `low`, with six more instructions around its
/// `minsd` on x86-64. The column-and-row and neighbor grids grow a box and
/// two spans with every store; using these there instead of `min` and `max`
/// cut their update times by 3-13%. In the boxed, lazy and single-level grids,
/// which grow boxes, spans or refreshed level bounds the same way, they cut the
/// update benchmarks by 5-15% on one machine and by up to 5% on another
/// (`bench-results/lower-higher-port-2026-10-24.md`).
#[inline(always)]
pub(crate) fn lower(low: f64, value: f64) -> f64 {
    if value < low { value } else { low }
}

/// `high.max(value)` for a `high` that is never NaN; see [`lower`].
#[inline(always)]
pub(crate) fn higher(high: f64, value: f64) -> f64 {
    if value > high { value } else { high }
}

/// Rounding margin that search adds to every bound, since its cell edges
/// accumulate rounding differently from [`cell_overflow`].
///
/// Items below the root stay within half the root of it, and the cell edges
/// of search are off by a few units in the last place of their coordinates;
/// 2^-40 of the largest root coordinate covers that by a wide margin.
pub(crate) fn rounding_slack(root_bounds: &Rectangle) -> f64 {
    let largest_coordinate = [
        root_bounds.min_x,
        root_bounds.min_y,
        root_bounds.max_x,
        root_bounds.max_y,
    ]
    .into_iter()
    .fold(0.0f64, |largest, coordinate| largest.max(coordinate.abs()));
    largest_coordinate * 2f64.powi(-40)
}

/// High-water overflow bound of the items of one level.
pub(crate) struct LevelReach {
    /// Largest overflow per direction since the last refresh: negative when
    /// every item stays inside that edge of its cell, and -infinity if the
    /// level has held no items since then. It may exceed the overflow of
    /// every item still present.
    pub(crate) reach: [f64; 4],
    /// Multiple of half a node by which search shrinks the fixed reach in
    /// each direction, kept in step with `reach`. It exceeds one for a
    /// negative bound, and is exactly zero once the reach plus the slack
    /// covers half a node, so that search then matches the hierarchical loose
    /// grid.
    ///
    /// Computing it in search puts a chain of dependent float operations
    /// ahead of the neighbor checks; with 10,000 small balls on the finest
    /// level, search then measured 0.95 ms instead of 0.68 ms.
    pub(crate) shrink: [f64; 4],
    /// Reciprocal of half the node width and height.
    inverse_half_node: (f64, f64),
}

impl LevelReach {
    /// Create the bound of a level without items, which reaches nowhere
    /// (-infinity); the first item sets it.
    pub(crate) fn new(node_width: f64, node_height: f64, slack: f64) -> Self {
        let mut level = Self {
            reach: [f64::NEG_INFINITY; 4],
            shrink: [0.0; 4],
            inverse_half_node: (2.0 / node_width, 2.0 / node_height),
        };
        level.set([f64::NEG_INFINITY; 4], slack);
        level
    }

    /// Raise the reach to cover `overflow`. Once a level is populated, this
    /// rarely raises anything, so updates mostly skip the stores.
    #[inline]
    pub(crate) fn grow(&mut self, overflow: [f64; 4], slack: f64) {
        for (direction, overflow) in overflow.into_iter().enumerate() {
            if overflow > self.reach[direction] {
                self.reach[direction] = overflow;
                self.shrink[direction] = self.shrink_of(direction, slack);
            }
        }
    }

    pub(crate) fn set(&mut self, reach: [f64; 4], slack: f64) {
        self.reach = reach;
        for direction in 0..4 {
            self.shrink[direction] = self.shrink_of(direction, slack);
        }
    }

    #[inline]
    fn shrink_of(&self, direction: usize, slack: f64) -> f64 {
        let (inverse_x, inverse_y) = self.inverse_half_node;
        let inverse = if direction < 2 { inverse_x } else { inverse_y };
        (1.0 - (self.reach[direction] + slack) * inverse).max(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `lower` and `higher` agree with `f64::min` and `f64::max` for the
    /// infinities, both zeros, ordinary numbers and NaN as the value, as long
    /// as the value they extend is not NaN, and of two equal zeros they keep
    /// the one they extend.
    #[test]
    fn lower_and_higher_match_min_and_max_for_a_number() {
        let values = [
            f64::NEG_INFINITY,
            -1.5,
            -0.0,
            0.0,
            f64::MIN_POSITIVE,
            2.5,
            f64::INFINITY,
            f64::NAN,
        ];
        for &extent in &values[..values.len() - 1] {
            for &value in &values {
                let context = format!("extent {extent}, value {value}");
                // Signed zeros compare equal; `min` and `max` may return either.
                assert_eq!(lower(extent, value), extent.min(value), "{context}");
                assert_eq!(higher(extent, value), extent.max(value), "{context}");
                assert!(!lower(extent, value).is_nan(), "{context}");
                assert!(!higher(extent, value).is_nan(), "{context}");
            }
        }
        for (extent, value) in [(0.0, -0.0), (-0.0, 0.0)] {
            let context = format!("extent {extent:?}, value {value:?}");
            assert_eq!(lower(extent, value).to_bits(), f64::to_bits(extent), "{context}");
            assert_eq!(higher(extent, value).to_bits(), f64::to_bits(extent), "{context}");
        }
    }
}
