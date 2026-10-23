# Directional Overflow Bounds

Pruning region queries in hierarchical loose grids with measured per-direction bounds on how far stored items extend beyond their cells, maintained per level, per column and row, and per neighboring cell. Implemented in Rust for moving 2D axis-aligned boxes.

## Motivation

A hierarchical loose grid assigns each item to the level matching its size and, within that level, to the cell containing its center. An item may thus extend up to half a cell beyond its cell, and every search pads each cell by that amount. The padding reflects the worst case rather than the stored items; measuring the actual maximum extent on each level yields a per-level looseness.

A single value per level, however, conflates four directions. An item near the right edge of its cell extends only rightward, while the search consults each direction separately: cells to the left of a query matter only as far as their items extend rightward, and cells to the right only as far as theirs extend leftward. Keeping the maximum extent of each direction apart yields four bounds per level, the **directional overflow bounds**, so that a large extent in one direction no longer widens the other three.

A per-level bound is set by the farthest-reaching item on the level and, on densely populated levels, approaches the worst case again. The bounds are therefore refined to ever smaller groups of items, each consulted where the search selects cells:

| Bounds kept | Structure |
|---|---|
| None: half a cell in every direction | `HierarchicalLooseGrid` |
| Per level | `OverflowBoundGrid` (exact), `LazyOverflowBoundGrid`; `BoxedOverflowBoundGrid` adds a box per cell |
| Per column and per row | `SpanOverflowBoundGrid` |
| Per neighbor of every cell | `HaloOverflowBoundGrid` |

On 10,000 balls of mixed sizes, each queried with its own bounds, per-neighbor bounds halve the search time of the plain grid (1.00 ms vs. 1.98 ms) at twice its update cost (149 µs vs. 76 µs per 10,000 moves, including one refresh). Full measurements are given under [Results](#results).

## The hierarchical loose grid

`HierarchicalLooseGrid` is the base shared by all bounded grids.

**Positioning.** Level `L` (1 ≤ `L` ≤ `MAX_LEVEL`) divides the root into `2^(L-1)` cells per axis. Each item is assigned in O(1) to the finest level whose cells are no smaller than the item, and to the cell containing its center. Items smaller than the finest cells stay on the finest level; items centered outside the world or larger than a level-2 cell go to the root. Being no larger than its cell and centered in it, an item extends at most half a cell beyond each edge, so each cell acts as a loose cell of twice its size.

**Search.** The query is positioned like an item, at level `q`.

- On each level finer than `q`, the cells within half a cell of the query are scanned. Cells at least two cells inside this range lie, loose bounds included, entirely within the query; their items are reported without intersection tests.
- On level `q` and above, the query spans at most one cell, so only the cell containing its center and the eight around it can hold intersecting items; these cells follow the ancestor path of the query's cell. The search walks this path and visits a neighbor only if the query crosses the cell's center line toward it, which selects the same cells as half-cell padding.
- The root is scanned in full.

No node bounds are tested, so nodes store only their items. As levels are visited from the finest upward, smaller items tend to be reported first, though items within a level are unordered.

**Handles.** `insert` returns an `ItemHandle` recording the item's location, so `update` and `remove` reach it in O(1) without hashing or scanning its node. An update within the same cell only rewrites the bounds. Items need only be `Copy`.

## Directional bounds per level

For an item `b` stored in cell `[x0, x1] × [y0, y1]`, with top denoting negative y:

```text
left = x0 - b.min_x      right  = b.max_x - x1
top  = y0 - b.min_y      bottom = b.max_y - y1

L, R, T, B = maxima of left, right, top and bottom over the items of a level
```

Values are negative where an item stays inside the edge. The left side of the search uses `R`, the right side `L`. With the origin at zero, cell width `s` and a query `[a, b]` in x:

```text
finer levels: scan columns floor((a - R) / s) through ceil((b + L) / s) - 1

ancestor walk, in the query's cell [x0, x1]:
    visit the left neighbors  only if a < x0 + R
    visit the right neighbors only if b > x1 - L
```

Rows use `B` above the query and `T` below. Each bound is widened by a rounding margin of `2^-40` times the largest root coordinate and capped at half a cell, where the rules reduce to those of the plain grid.

Relative to the plain grid, and before the second to fourth changes noted below the results tables, these bounds cut search time by 46% for balls well inside the finest cells, by 56% and 32% for small and large queries over items near their cell centers, and by 27% to 32% for the ball queries of the two-tier scenes. On the mixed distribution they had no effect, as some items on every level extend nearly half a cell. A single looseness per level, the largest of the four bounds applied in all directions, performed equally on scenes with symmetrically placed items and 3% to 8% worse on the two-tier scenes and the mixed distribution.

**Negative bounds.** If every item on a level keeps clear of a cell edge, the bound is negative, and the search crosses that edge only when the query exceeds it by more than the clearance. On the aligned scene, whose items stay at least 36 units inside their 312.5-unit cells, admitting negative bounds made small queries 17% to 20% faster than clamping them at zero.

**Maintenance.** Both structures share the same search.

- `OverflowBoundGrid` keeps the bounds exact. Each level and direction holds a counted multiset of all item distances: a compressed radix-64 bitmap over order-preserving 64-bit keys ([`bitmap_max_multiset.rs`](src/bitmap_max_multiset.rs)), with at most 11 branches per path, giving O(1) insertion, removal and maximum for the fixed key width. Each item contributes four entries, and a move replaces those that change, usually all four; updating 10,000 balls takes 4.36 ms.
- `LazyOverflowBoundGrid` raises a bound on insertion and update, with one comparison per direction, and leaves it unchanged on removal; `refresh_bounds` recomputes it from the items, e.g. once per frame. A raised bound remains valid as items move, shrink or leave, so results are always exact and only pruning degrades until the next refresh. Updating 10,000 balls takes 126 µs, refresh included.

**Cell boxes.** `BoxedOverflowBoundGrid` adds to the lazy per-level bounds a box per node covering its items, grown on insertion and update and shrunk by `refresh_bounds`. The level bounds select the nodes to visit; a visited node whose box misses the query is skipped without reading its items. The ancestor walk also stops at the coarsest non-root level that may hold items. Relative to the lazy grid, which did not stop there at the time, search time fell by 19% on the mixed distribution, 15% on small balls, 26% on small aligned queries and 33% for the rectangle queries of the adjacent scene.

## Directional bounds per column and row

`SpanOverflowBoundGrid` maintains for every column of a level the span from the smallest to the largest x of its items, and likewise in y for every row. The ends of a column's span are its own `L` and `R` in absolute coordinates; rows carry `T` and `B`. The ancestor walk visits a neighboring column or row only if its span meets the query, the scan of finer levels drops boundary columns and rows whose spans miss the query, and the cell boxes of the boxed grid skip the remaining nodes that miss it.

Spans consist of item coordinates, so comparisons need neither cell edges nor a rounding margin, and maintenance needs no arithmetic. On sparse levels most columns and rows are empty and never visited; on dense levels a column still spans many items, but no longer stands for the entire level.

Insertion and update widen the node's box and the spans of its column and row, and mark the level and row as touched. `refresh_bounds` skips untouched levels; elsewhere it shrinks the boxes and rebuilds the spans of touched rows only, reading the boxes of other rows solely for the column spans. When moving items stay within a band of rows, or a large static population occupies its own level, a refresh costs about a quarter of a full one.

Relative to the lazy per-level bounds, and before the second change noted below the results tables, search time fell by 38% on the mixed distribution, 24% on small balls, 34% on small aligned queries, 23% and 26% for the ball queries of the two-tier scenes and 24% for the rectangle queries of the adjacent scene, and rose by 9% on large aligned queries. Updating 10,000 balls takes 140 µs, vs. 126 µs for the lazy grid.

## Directional bounds per neighboring cell

With per-column and per-row bounds, nearly all remaining waste lies in the ancestor walk: in a counting model of the mixed distribution, a query visits 12.0 nodes, 11.4 of them in the walk and 9.5 of those to no effect, since the span of a dense column or row almost always meets the query.

`HaloOverflowBoundGrid` stores at every node the bounds of its eight neighbors facing it: the largest `max_x` among the items of its left neighbor, the smallest `min_x` of its right neighbor, their counterparts in y, and two values per diagonal neighbor, twelve in total. Since the walk reads the query's own node on every level anyway, these values alone decide which neighbors to visit. They are kept in an array beside the grid rather than in the nodes, so that nodes remain as compact as in the column-and-row grid for scans and updates.

Insertion and update maintain boxes and spans as in the column-and-row grid and mark the level stale and the row touched; the copies held by neighbors are left as they are. On a stale level, queries in or next to a touched row judge neighbors by the spans, all others by the copies. `refresh_bounds` shrinks the boxes of touched rows and recopies the facing sides into those rows and their neighbors. Before a refresh, search therefore runs at the speed of the column-and-row grid (rows "before a refresh" below).

Relative to the per-column and per-row bounds, search time fell by 20% on the mixed distribution, 39% for the rectangle queries of the adjacent scene, 16% and 9% for the ball queries of the two-tier scenes and 4% on small balls, and rose by 4% on small aligned queries, where the spans already exclude everything. Updates cost 1.03 to 1.12 times those of the column-and-row grid, as the refresh copies the facing sides of eight neighbors per node. The structure suits frames that update, refresh and then query.

## Results

Measured in a single session on an AMD Ryzen 7 9800X3D (rustc 1.99.0, Windows 11) with Criterion defaults. World 20,000 × 20,000, 7 levels, finest cells 312.5 units wide. Columns: `HierarchicalLooseGrid` (none), `LazyOverflowBoundGrid` and `OverflowBoundGrid` (per level), `BoxedOverflowBoundGrid`, `SpanOverflowBoundGrid` (per column and row) and `HaloOverflowBoundGrid` (per neighbor).

Search time in µs:

| Scene and queries | None | Level, lazy | Level, exact | Boxed | Column and row | Neighbor |
|---|---:|---:|---:|---:|---:|---:|
| Mixed distribution, 10,000 | 1978 | 2029 | 2040 | 1643 | 1254 | 1003 |
| Small balls, 10,000 | 1256 | 684 | 683 | 579 | 522 | 500 |
| Aligned, 10,000 small | 719 | 319 | 318 | 237 | 212 | 220 |
| Aligned, 2,000 large | 327 | 223 | 217 | 220 | 242 | 233 |
| Gap, 10,000 balls | 1567 | 1146 | 1131 | 1036 | 886 | 810 |
| Adjacent, 10,000 balls | 1759 | 1191 | 1195 | 1006 | 878 | 739 |
| Adjacent, 1,024 rectangles | 292 | 216 | 220 | 145 | 164 | 100 |
| Gap, balls before a refresh | | | | 580 | 499 | 501 |
| Adjacent, balls before a refresh | | | | 571 | 512 | 517 |

The 64 rectangle queries of the gap scene take 27 to 31 µs in every structure.

Item management in µs. `update` moves all 10,000 balls one step and refreshes once; `reinsert` removes and reinserts every ball; the last three rows apply 20,000 precomputed updates (small moves, one-cell moves, moves with resizing) with two refreshes.

| Operation | None | Level, lazy | Level, exact | Boxed | Column and row | Neighbor |
|---|---:|---:|---:|---:|---:|---:|
| `update` | 76 | 126 | 4359 | 147 | 140 | 149 |
| `reinsert` | 271 | 327 | 4231 | 368 | 385 | 398 |
| Small moves | 119 | 200 | 8547 | 237 | 223 | 250 |
| One-cell moves | 385 | 490 | 9414 | 554 | 557 | 578 |
| Moves with resizing | 188 | 298 | 8874 | 322 | 318 | 333 |

These tables predate seven changes. First, every grid now finds an item's level from the float exponents of its size and of the root, and its cell by multiplying with the reciprocal of the cell size, where most of them divided before. In later runs (three interleaved rounds, 2 s measurements), `reinsert` took 174 µs without bounds (`HierarchicalLooseGrid`) and 212, 250, 256 and 279 µs for the lazy, boxed, column-and-row and neighbor grids, 30% to 36% less than above. `update` and the precomputed moves stayed within 5%, and search became up to 17% faster on aligned queries. `HierarchicalGrid` reinserts in 157 µs, `SpanGrid` in 215 µs and `HaloGrid` in 223 µs. Details are in [docs/multi-level-optimization-directions.md](docs/multi-level-optimization-directions.md). Second, the grid without bounds and the grids with exact and lazy level bounds now also stop the ancestor walk at the coarsest level below the root that holds items (for the lazy grid, that may hold items until the next refresh), as the other three overflow-bound grids already did. Over two runs of three interleaved rounds, this cut search time on small aligned queries by 40-44% without bounds, 21-24% with exact and 21-31% with lazy bounds. Without bounds it also cut search time on small balls by 23-24%, on large aligned queries by 18-21% and in the adjacent scene by 9-14%, and with lazy bounds on the rectangle queries of the adjacent scene by 7-13%. Every other difference was under 5% in at least one of the runs, within the noise of the unchanged boxed grid; this includes the mixed distribution, where only level 2 is empty, the gap scene and the self query. Without bounds the gain is larger because there the step to the parent cell compiles to a branch on the parity of the cell coordinates, which random queries cannot predict, while the bounded grids compute it without a branch; making only that step branch-free gained as much in a separate measurement. Updates stayed within 3%, except that one-cell moves without bounds were about 2% slower in every run. Third, the grid without bounds now skips that step after the coarsest level, where its result is unused. In three more runs this cut its search time by a further 6-7% on small balls, 13-16% on small aligned queries and 4-6% for the ball queries of the adjacent scene; the other differences stayed within the 3% by which the same source varies between binaries. A branch-free step with the same skip was a trade-off: 8-9% faster for the ball queries of the gap scene, which cross three empty levels, but 2% slower on the mixed distribution and 3% slower in the batched self query. In the bounded grids, the same skip turned their branch-free step into a branch and made scenes with empty levels 8-37% slower, so they keep the step. Fourth, the grid without bounds and the column-and-row and neighbor grids now compute the query's corners once, in half cells of the finest level, and derive the scan range of every finer level by shifts, bit for bit what rounding on each level gave, as `HierarchicalGrid` already did. In three runs this cut search time on large aligned queries by 32-34%, 27% and 31-32%, and by 12-15% for the rectangles of the gap scene without bounds; the mixed distribution became 2-4% faster. On x86-64 without SSE4.1, `round`, `floor` and `ceil` compile to C runtime calls. Called in isolation, `round` takes about 6 ns against 1.3 ns for `floor`, but in a scan, where its result bounds the loop, it measured about 16 ns: the Windows C runtime's `round` clears the fraction with 2-byte stores and reads the result back in one 8-byte load, which store forwarding cannot serve, and branches on whether the fraction is one half, which random queries mispredict. Fifth, the grids with exact, lazy and boxed bounds and `SpanGrid` and `HaloGrid`, which still called `round` once per axis for every scanned level or query, now take the first node of the scan from `floor`, with the same result. In three interleaved rounds, confirmed by a reviewer's runs, this cut search time on large aligned queries by 23-29% in all five, and on the rectangle queries of the adjacent scene by 8-15% with exact and lazy bounds and by about 17% with boxes, measured in one process, as criterion's figures for the boxes swing between 29% and 47% in this bimodal scene. The batched self query of `SpanGrid` became 1-3% slower, from a change in register allocation; the other differences stayed within the noise of the unchanged grids. Sixth, every grid now truncates where it took `floor`: for the first node of the scans above, for the minima of the corners from which the grid without bounds and the column-and-row and neighbor grids derive their scans, and for all four corners of a query in `HierarchicalGrid`. Truncation differs from `floor` only below zero, where every scan range clamps to its first node alike, so results are unchanged. Only `HierarchicalGrid`, for a query starting just over half a cell of some level before its origin, now tests the objects of that level's first column or row instead of reporting them untested; for queries with a NaN coordinate, which the grids answer in part with objects they report untested, it therefore returns fewer of those. In three to five interleaved rounds, and in five with code alignment forced, this cut search time on large aligned queries by a further 4-11% in the grids with bounds and in `SpanGrid` and `HaloGrid` and by 2% without bounds, and made every query of `HierarchicalGrid` about 3% faster. Small aligned queries, which do not scan and run none of the changed code, moved by -6% to +10% depending on the grid, from changes in the generated code; with alignment forced, the boxed grid stayed about 5% slower there. Seventh, the column-and-row and neighbor grids now grow their cell boxes and their column and row spans with a comparison and a select instead of `f64::min` and `f64::max`, which on x86-64 spend six more instructions each on handling a NaN receiver; these extents, grown from infinity, never hold one. In three interleaved rounds, with and without code alignment forced, and in six rounds by a reviewer, this cut update times by 3-13% in both grids, with search unchanged. The column-and-row grid needed its span update forced inline, which the shorter code had otherwise turned into a call. Details are in section 36 of [bench-results/overflow-bound-grids-2026-10-13.md](bench-results/overflow-bound-grids-2026-10-13.md). The comparisons in the sections above were measured before the second to seventh changes. In a run after the fourth change, the exact bounds cut search time relative to the grid without bounds by 29% instead of 46% on small balls, by 34% instead of 56% on small aligned queries and by 25% and 17% instead of 27% to 32% for the ball queries of the gap and adjacent scenes, but were 9% slower instead of 32% faster on large aligned queries, whose scan the exact grid then still rounded with `round` on every level; after the fifth change they were 14-19% faster there, and after the sixth about 23%. Relative to the lazy bounds, the boxes cut search time on small aligned queries by 1-17% instead of 26%, depending on the run, and the column and row bounds by 15-18% instead of 34%. Details are in [docs/cross-structure-optimizations.md](docs/cross-structure-optimizations.md).

**Selection.** All structures return exact results at all times, up to floating-point rounding, and refreshing only restores pruning. The loose grids search a fixed reach of half a cell without a rounding margin, so an item that overlaps a query by a few units in the last place, right at that reach, can be missed.

- Update, refresh, then query each frame: `HaloOverflowBoundGrid`.
- Queries interleaved with updates, without a natural refresh point: `SpanOverflowBoundGrid`, refreshed when convenient, or `HierarchicalLooseGrid` when updates dominate.
- Exact per-level extents at any time (`layer_overflow`): `OverflowBoundGrid`, at a high update cost.
- Items of uniform size: `UniformGrid` or `SpanGrid`, with the cell size tuned to the items.

## Usage

The library target is named `lib`.

```rust
use lib::rect::Rectangle;
use lib::*;

const MAX_LEVEL: u8 = 7;

let world = Rectangle::new(0.0, 0.0, 20000.0, 20000.0);
let mut grid = HaloOverflowBoundGrid::<usize, MAX_LEVEL>::new(world);

let handle = grid.insert(Rectangle::center_rect(100.0, 100.0, 50.0, 50.0), 0);
grid.update(Rectangle::center_rect(103.0, 101.0, 50.0, 50.0), handle);
grid.refresh_bounds(); // Once per frame, after the updates.

grid.search(&Rectangle::center_rect(120.0, 90.0, 80.0, 80.0), |id| println!("{id}"));
let bound = grid.overflow_bound(MAX_LEVEL as usize); // Left, right, top, bottom.
grid.remove(handle);
```

- `new` uses square cells spanning the larger side of the world; `new_fit` fits the cells to the world.
- Using a handle after `remove` panics, even if its slot has been reused.
- `refresh_bounds` is provided by the lazy structures: `LazyOverflowBoundGrid`, `BoxedOverflowBoundGrid`, `SpanOverflowBoundGrid`, `HaloOverflowBoundGrid`, `SpanGrid` and `HaloGrid`.
- Every structure except `LooseQuadTree` also provides `for_each_self_query(|item, found| ...)`, which searches with the bounds of every stored item, cell by cell: the multi-level grids go depth first through the levels and in Morton order within each, the single-level ones in Morton order. It visits every cell (in the multi-level grids, on every level down to the finest one holding items), so it pays off only when most cells hold items. Against searching item by item in random order, the `self_query` bench group (10,000 items of the mixed distribution) found it 7-16% faster for the multi-level grids, up to 8% faster for the single-level ones. A one-off probe outside the repository found it 22-43% faster for all of them with 640,000 items at the same density, where the random order misses the cache. `UniformGrid` gains the least because each search first reads its query's bounds from its object table.
- In the four lazy overflow-bound grids, `overflow_bound(level)` returns a level's current bounds, for levels 2 to `MAX_LEVEL`; `OverflowBoundGrid::layer_overflow(level)` returns the exact maxima, for levels 1 to `MAX_LEVEL`. Empty levels report negative infinity.

## Other structures

- [`LooseQuadTree`](src/loose_quadtree.rs): a loose quadtree with conventional top-down search over node bounds. Mixed distribution 3.54 ms; update 77 µs; reinsertion 413 µs.
- [`HierarchicalGrid`](src/hierarchical_grid.rs): the positioning of the hierarchical loose grid, scanning every level over the query range padded by half a cell, without the ancestor walk. Cheapest maintenance here (reinsertion 165 µs); mixed distribution 1.89 ms.
- [`UniformGrid`](src/uniform_grid.rs): a single level registering each item in every cell it overlaps. With 312.5-unit cells it takes 1.03 ms on the mixed distribution and is fastest on large aligned queries (138 µs), but it depends on the cell size, and one-cell moves and resizing cost 628 and 701 µs, vs. 578 and 333 µs with per-neighbor bounds.
- [`SpanGrid`](src/span_grid.rs): per-column and per-row bounds on a single level of chosen cell size, for scenes whose items all fit a cell; larger items go to one overflow node read by every search.
- [`HaloGrid`](src/halo_grid.rs): `SpanGrid` with the per-neighbor copies of `HaloOverflowBoundGrid`, so that its 3 × 3 walk reads a neighbor only if that neighbor's own box may reach the query. Small balls take 225 µs on 156-unit cells, vs. 372 µs in `SpanGrid` (1.29 instead of 4.01 nodes read per query), and 441 vs. 465 µs on 312-unit cells. Aligned small queries are slower, 185 vs. 144 µs, because the spans already prune nearly as well and the copies only add a read. Updates are 2-8% slower.

## Benchmarks

```bash
cargo bench --bench search
cargo bench --bench update
```

Some mixed-distribution benchmarks are commented out in `criterion_group!` in [`benches/search.rs`](benches/search.rs); uncomment them to reproduce the tables. Scenes:

- **Mixed distribution:** 10,000 balls of radius 25 to 50, each doubled up to five times with probability 0.25 per step, populating several levels; each ball's bounds serve as a query.
- **Small balls:** as above without doubling; every ball lies well inside a finest cell.
- **Aligned:** two items per finest cell, within 30 units of its center, 60 to 180 units per side; queries of 50 to 300 units (small) and 400 to 1,500 units (large) per side.
- **Gap:** small balls on the finest level and 64 rectangles with sides of 2,600 to 4,900 units on level 3, with the levels in between empty.
- **Adjacent:** small balls and 1,024 rectangles with sides of 320 to 620 units on level 6, directly above them.
- **Before a refresh:** the two-tier scenes after the rectangles have shrunk onto the finest level, queried without a refresh.

The single per-level looseness compared above is not part of the crate; it was measured with `LazyOverflowBoundGrid` modified to apply the largest of its four bounds in all directions.

The design notes in [`docs/`](docs) and the measurement log [`bench-results/overflow-bound-grids-2026-10-13.md`](bench-results/overflow-bound-grids-2026-10-13.md), both in Chinese, document every variant examined, including those that performed worse. `SpanOverflowBoundGrid`, `HaloOverflowBoundGrid`, `SpanGrid` and `HaloGrid` were called `StripedOverflowBoundGrid`, `IncomingOverflowBoundGrid`, `StripedGrid` and `IncomingGrid` until 2026-10-21; the documents use the new names, while raw logs, data files and archived sources keep the old ones. [`bench-results/loose-grid-optimization-2026-10-13.md`](bench-results/loose-grid-optimization-2026-10-13.md) and [`bench-results/overflow-bound-grid-update-2026-10-13.md`](bench-results/overflow-bound-grid-update-2026-10-13.md) concern the bitmap of `OverflowBoundGrid` before its distances became signed.

## License

[MIT](LICENSE).
