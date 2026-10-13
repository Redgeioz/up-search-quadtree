# Up-search Quadtree

This repository presents an upward-searching method that reduces intersection checks between the query region and node bounds during quadtree searches.

## Principle

Combining a loose quadtree (with a looseness factor of 2) with a multilayer grid allows an object's insertion position to be computed in O(1) time from its size and center. The search proceeds upward from the lowest layer. First, determine the query region's insertion point and use it as the dividing point between different upward-search methods.

1. **Below the insertion layer — calculate a search range for each layer.** Start at the lowest layer and calculate the range of nodes to search.  Scan the nodes in this range, then move up one layer and calculate a new range using that layer's cell dimensions. Repeat until all layers below the insertion layer have been searched.

2. **At the insertion layer and above — follow the path to the root and search neighboring nodes.** Since the path is already determined by the insertion point, simply search the insertion node and its neighbors within the surrounding 3 × 3 block, starting at the insertion layer. Then move up to its parent and search in the same way, repeating until the root is reached.

Both methods access nodes directly through grid coordinates, avoiding the recursive node-bound intersection checks used to discover branches in a traditional search.

## Advantages

- **Fewer search-path decisions.** Grid coordinates directly identify nodes and their parents, avoiding the node-bound intersection checks used to select branches during recursive search.
- **Smaller nodes.** `UpSearchQuadTree` stores object bounds and IDs without storing node bounds or child pointers.
- **Low object-management overhead.** Insertion positions are computed in O(1) time. `insert` returns an `ItemHandle` that records where the object is stored, so `update` and `remove` reach the object directly in O(1) time, without hashing the object or scanning its node. The object type only needs to be `Copy`.
- **Approximate ordering by object size.** Scanning from the lowest layer upward tends to return smaller objects before larger ones, although objects within each layer are not sorted by size.

## Disadvantages

- **Memory overhead from preallocated grids.** Empty nodes also occupy space, making the structure less economical for sparse scenes. For square world bounds, adding a layer roughly quadruples the grid storage.
- **Empty-cell scans for large queries.** Deeper layers are scanned over rectangular ranges, including empty cells. Search cost therefore depends on the number of grid cells covered as well as the number of objects.
- **Sensitivity to object distribution.** Objects concentrated in a few nodes, or many large objects stored in upper layers, can require numerous candidate intersection checks. In the worst case, all stored objects may need to be examined.

## Usage

Every quadtree in this repository shares the same item management: `insert` hands back an `ItemHandle`, which `update` and `remove` take instead of the item itself.

```rust
let mut tree = UpSearchQuadTree::<usize, MAX_LEVEL>::new(world_bounds);

let handle = tree.insert(ball.get_bounds(), id);
tree.search(&query_bounds, |other_id| { /* ... */ });
tree.update(ball.get_bounds(), handle);
tree.remove(handle);
```

Handles are small `Copy` values. Using a handle after `remove` panics, even if its slot has been reused by a later `insert`.

## Quadtrees Included in This Repository

[`quadtree.rs`](src/quadtree.rs) implements a regular quadtree with traditional downward search. [`loose_quadtree.rs`](src/loose_quadtree.rs) implements a loose quadtree using the same search direction.

[`grid_loose_quadtree.rs`](src/grid_loose_quadtree.rs) combines a loose quadtree with a multilayer grid and supports downward, upward, and bidirectional search. [`up_search_quadtree.rs`](src/up_search_quadtree.rs) builds on this implementation, retaining only upward search and removing stored node bounds to reduce the size of each node.

[`up_search_quadtree_original.rs`](src/up_search_quadtree_original.rs) implements the original idea using a regular, non-loose quadtree. It computes the lowest common ancestor of the two lowest-layer corner nodes in O(1) time using bit operations, allowing the insertion position to be determined directly. This original variant did not show a substantial performance improvement in earlier comparisons, but its simpler structure makes it a useful reference for understanding upward search.

## Benchmark

```bash
cargo bench
```

The following results were previously recorded for 10,000 searches among 10,000 balls of different sizes, using each ball's bounding rectangle as the query:

| QuadTree | LooseQuadTree | UpSearchQuadTree |
|:--------:|:-------------:|:----------------:|
| 4.62 ms  | 3.36 ms       | 2.28 ms          |

`cargo bench --bench update` measures item management on the same 10,000 balls: moving every ball one step and updating its bounds (`update`), and removing every ball and inserting it again (`reinsert`). Both go through handles, so no hashing is involved:

|          | QuadTree | LooseQuadTree | UpSearchQuadTree |
|:--------:|:--------:|:-------------:|:----------------:|
| update   | 87 µs    | 77 µs         | 75 µs            |
| reinsert | 541 µs   | 415 µs        | 275 µs           |
