//! Orders in which batch operations visit the cells of the levels.

/// Call `visit` with every cell `(level, x, y)` of the levels `1..=finest`
/// of a hierarchical grid, where level `n` has `dims(n)` cells and covers
/// the cells `2x..2x + 2` by `2y..2y + 2` of level `n + 1` with its cell
/// `(x, y)`: depth first from the single cell of level 1, each cell before
/// the four it covers, in Morton order. The finest level is so visited in
/// Morton order, with each coarser cell just before the first finest cell it
/// covers, and the cells of all levels around a place come together.
///
/// Every cell of level `n + 1` lies under a cell of level `n` as long as no
/// level has more than twice the columns or rows of the next coarser one.
pub(crate) fn for_each_cell_hierarchically(
    finest: usize,
    dims: impl Fn(usize) -> (usize, usize),
    mut visit: impl FnMut(usize, usize, usize),
) {
    fn walk(
        (level, x, y): (usize, usize, usize),
        finest: usize,
        dims: &impl Fn(usize) -> (usize, usize),
        visit: &mut impl FnMut(usize, usize, usize),
    ) {
        visit(level, x, y);
        if level == finest {
            return;
        }
        let (cols, rows) = dims(level + 1);
        for (cx, cy) in [(2 * x, 2 * y), (2 * x + 1, 2 * y), (2 * x, 2 * y + 1), (2 * x + 1, 2 * y + 1)] {
            if cx < cols && cy < rows {
                walk((level + 1, cx, cy), finest, dims, visit);
            }
        }
    }
    if finest >= 1 && dims(1).0 > 0 && dims(1).1 > 0 {
        walk((1, 0, 0), finest, &dims, &mut visit);
    }
}

/// Call `visit` with every cell `(x, y)` of a single grid of `cols × rows`
/// cells, in Morton order: as the finest level of
/// [`for_each_cell_hierarchically`] under coarser levels that halve it,
/// rounding up, down to a single cell, which skips the parts of the
/// enclosing power-of-two square outside the grid.
pub(crate) fn for_each_cell_in_morton_order(cols: usize, rows: usize, mut visit: impl FnMut(usize, usize)) {
    if cols == 0 || rows == 0 {
        return;
    }
    let finest = cols.max(rows).next_power_of_two().trailing_zeros() as usize + 1;
    let dims = |level: usize| {
        let shift = finest - level;
        (cols.div_ceil(1 << shift), rows.div_ceil(1 << shift))
    };
    for_each_cell_hierarchically(finest, dims, |level, x, y| {
        if level == finest {
            visit(x, y);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn levels(finest: usize, dims: impl Fn(usize) -> (usize, usize)) -> Vec<(usize, usize, usize)> {
        let mut cells = Vec::new();
        for_each_cell_hierarchically(finest, dims, |level, x, y| cells.push((level, x, y)));
        cells
    }

    /// Levels of 1 × 1, 2 × 2 and 4 × 4 cells: each cell comes right before
    /// the cells it covers, which follow in Morton order.
    #[test]
    fn levels_are_visited_parent_first_in_z_order() {
        let square = |level: usize| (1 << (level - 1), 1 << (level - 1));
        assert_eq!(
            levels(3, square),
            vec![
                (1, 0, 0),
                (2, 0, 0), (3, 0, 0), (3, 1, 0), (3, 0, 1), (3, 1, 1),
                (2, 1, 0), (3, 2, 0), (3, 3, 0), (3, 2, 1), (3, 3, 1),
                (2, 0, 1), (3, 0, 2), (3, 1, 2), (3, 0, 3), (3, 1, 3),
                (2, 1, 1), (3, 2, 2), (3, 3, 2), (3, 2, 3), (3, 3, 3),
            ]
        );
    }

    /// Levels as the grids of a world three times as wide as high build
    /// them, with the rows rounded up: every cell of every level once.
    #[test]
    fn wide_levels_visit_each_cell_once() {
        let wide = |level: usize| {
            let cols = 1usize << (level - 1);
            (cols, (cols as f64 / 3.0).ceil() as usize)
        };
        let mut cells = levels(7, wide);
        let total: usize = (1..=7).map(|level| wide(level).0 * wide(level).1).sum();
        assert_eq!(cells.len(), total);
        cells.sort();
        cells.dedup();
        assert_eq!(cells.len(), total);
        assert!(cells.iter().all(|&(level, x, y)| x < wide(level).0 && y < wide(level).1));
    }

    fn order(cols: usize, rows: usize) -> Vec<(usize, usize)> {
        let mut cells = Vec::new();
        for_each_cell_in_morton_order(cols, rows, |x, y| cells.push((x, y)));
        cells
    }

    #[test]
    fn square_grid_is_visited_in_z_order() {
        assert_eq!(
            order(4, 4),
            vec![
                (0, 0), (1, 0), (0, 1), (1, 1),
                (2, 0), (3, 0), (2, 1), (3, 1),
                (0, 2), (1, 2), (0, 3), (1, 3),
                (2, 2), (3, 2), (2, 3), (3, 3),
            ]
        );
    }

    /// Cells outside a non-square or odd-sized grid are left out, and the
    /// others keep their Morton order.
    #[test]
    fn other_grids_visit_each_cell_once() {
        assert_eq!(order(3, 2), vec![(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]);
        assert_eq!(order(1, 1), vec![(0, 0)]);
        assert_eq!(order(0, 0), vec![]);
        for (cols, rows) in [(5, 3), (64, 32), (7, 9), (1, 6)] {
            let mut cells = order(cols, rows);
            assert_eq!(cells.len(), cols * rows);
            cells.sort();
            cells.dedup();
            assert_eq!(cells.len(), cols * rows);
            assert!(cells.iter().all(|&(x, y)| x < cols && y < rows));
        }
    }
}
