# Up-search Quadtree

This repository aims to show a new method for efficiently searching a quadtree.

## Principle

The combination of a loose quadtree with a multilayer grid enables O(1) time complexity for insertion. Based on this feature, we can rapidly determine the insertion position for the search region. From this point upwards to the top, the path is predetermined, requiring only the search of a 3x3 area along this path. Consequently, a significant number of intersection checks can be skipped. For the remaining levels, the coordinates of nodes intersecting with the search region can be directly calculated, eliminating the need for individually checking each node for intersection.

## Advantages
- Very fast search
- Smaller size for each node
- Items found are relatively ordered in terms of size

## Files

Including an example, a simple benchmark on searching, and three types of quadtree  implementations.

Among them, `quadtree.rs` is a normal quadtree implementation. `grid_loose_quadtree.rs` is a loose quadtree combined with a multilayers grid, which also implements the up-search function. Additionally, `up_search_quadtree.rs` is the result obtained by retaining only the up-search function and optimization based on `grid_loose_quadtree.rs`.

*Compilation requires Rust **nightly** 1.72 or higher.*

## Benchmark

```bash
cargo bench
```

Search 12500 times among 12500 balls of different sizes using the bounds of each of them:

| Quadtree| GridLooseQuadtree | UpSearchQuadtree |
|:-------:|:-----------------:|:----------------:|
| 10.24ms | 6.67ms            | 4.38ms           |

Of these, the first two results are obtained using the traditional method.

## Example

```bash
cd example
```

```bash
cargo run
```
or
```bash
cargo run --release
```