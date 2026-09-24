# HNSW Implementation Guide

The implementation consists of:

- `include/vectordb/indexes/hnsw_index.hpp`
- `src/indexes/hnsw_index.cpp`

Like `RandomProjectionLshIndex`, `HnswIndex` was built and tested directly
against `VectorStore` before being wired into `Collection`. It is the
project's first graph-based ANN index (Malkov & Yashunin,
"Efficient and robust approximate nearest neighbor search using Hierarchical
Navigable Small World graphs") and supports all three metrics (`L2`, `Dot`,
`Cosine`), unlike the cosine-only LSH index.

## 1. The data structure

HNSW is a multi-layer proximity graph:

```txt
layer 2:        A ------------- D
                |               |
layer 1:        A ----- C ----- D ----- F
                |        \      |       |
layer 0:  ..all nodes, densely connected..
```

Upper layers are sparse and let search jump across the space quickly; layer 0
contains every node and is where the final candidate set is refined. Each
node gets a random top level `l` at insertion time and exists at every layer
from 0 up to `l`.

Adjacency is stored as `[node][layer] -> neighbor internal IDs`
(`adjacency_`), with node `i`'s outer vector sized `node_level_[i] + 1`. The
node with the highest level so far is the `entry_point_`; every search starts
there and descends.

## 2. Two primitives

Everything else composes these two building blocks.

### `search_layer(query, entry_points, ef, layer)`

Greedy best-first search on one layer (Algorithm 2 in the paper): a
closest-first candidate min-heap drives exploration, a worst-first result
max-heap caps the running best set at `ef`, and a visited set avoids
re-scoring a node. It stops once the best unexplored candidate is worse than
the current worst kept result. Returns up to `ef` nodes, closest-first.

Since "closer" depends on the metric (`L2` is smaller-is-closer, `Dot`/
`Cosine` are larger-is-closer), every score is oriented once via
`index_detail::higher_is_better(metric)` into an `orient_score` where smaller
always means closer. This lets both heaps and every comparison in the file
use one consistent direction instead of branching on metric everywhere.

### `select_neighbors_heuristic(base_id, candidates, m)`

Algorithm 4's diversity heuristic. Sort candidates by distance to `base_id`,
then greedily keep a candidate only if it is closer to `base_id` than to
every neighbor already selected; stop at `m` neighbors. This spreads
connections across the space instead of greedily picking the `m` closest
points, which tend to cluster and hurt long-range navigability.

## 3. Insertion (`add`)

Per Algorithm 1:

1. Assign a random level `l = floor(-ln(U(0,1)) * mL)`, `mL = 1/ln(M)` (`M ==
   1` falls back to `ln(2)` so the formula stays well-defined for that
   degenerate but valid configuration).
2. Greedily descend from `entry_point_`'s layer down to `l+1` with `ef = 1`,
   tracking a single best entry point per layer.
3. For each layer from `min(l, max_level_)` down to `0`: run
   `search_layer(ef_construction)` from the current entry set, pick up to
   `M` (or `2*M` at layer 0 — `M_max0`) neighbors via the heuristic, and wire
   bidirectional edges. Any existing neighbor that now exceeds its own cap
   gets pruned by re-running the heuristic on its updated list.
4. If `l > max_level_`, the new node becomes the entry point.

### Exception safety

`Index::add()` must leave the previous searchable state untouched if it
throws. HNSW insertion touches several existing nodes' neighbor lists at
once, so it is split into two phases:

- `prepare_insert()` is read-only against the committed graph and returns an
  `InsertPlan`: the new node's own per-layer neighbor lists, plus a list of
  `(node, layer, new_neighbor_list)` updates for every existing node whose
  list changes. All allocation and distance computation happens here, and it
  may throw (invalid vectors, `bad_alloc`).
- `apply_insert()` commits a plan with vector `push_back`/move-assignment
  only. `add()` reserves capacity for the one new adjacency/level slot before
  calling it, so the commit itself cannot fail once reached.

`build()` follows the same shape but on fully local containers (fresh
adjacency, level, entry point, and RNG state seeded from `config_.seed`),
committed into the members with `std::move` only after every vector has been
inserted successfully. An empty store is a valid no-op — `Collection`'s
constructor calls `build()` before any vectors exist.

## 4. Search (`search`)

Per Algorithm 5: greedily descend from `entry_point_` to layer 1 with `ef =
1`, then run `search_layer` at layer 0 with `ef = max(ef_search, top_k)`.
Feed the results into `index_detail::TopKAccumulator` (already metric-aware,
with the project's usual lower-internal-id tie-break) and return the top
`top_k`.

## 5. Determinism

The only randomness is level assignment, drawn from a `std::mt19937_64`
seeded with `config_.seed`. Level sampling uses a hand-rolled 53-bit uniform
double derived directly from the generator's bits rather than
`std::uniform_real_distribution`, whose sampling sequence is not specified
byte-for-byte across standard library implementations — the same reasoning
LSH already applies to its projection sampler.

Two consequences fall out of this for free:

- `build()` twice on the same store reproduces the identical graph (a fresh
  local RNG is always seeded from `config_.seed`).
- A full `build()` and an equivalent sequence of `add()` calls on an
  initially empty, then incrementally grown, store produce identical graphs:
  `rng_` is a member that keeps advancing across calls, so `add()` after an
  empty `build()` continues the exact same draw sequence the full build
  would have used for those vectors.

## 6. Invariants

After a successful `build()` or `add()`:

1. `adjacency_.size() == node_level_.size() == indexed_vector_count_`.
2. `adjacency_[i].size() == node_level_[i] + 1`.
3. Every ID in `adjacency_[i][layer]` is `< indexed_vector_count_` and has
   `layer < adjacency_[that ID].size()`.
4. `entry_point_` has the maximum level in the graph (`max_level_`); ties
   keep whichever node became the entry point first.
5. `is_built_` is true only when the above hold.

## 7. Metric notes

`Dot` is not a true metric (no triangle inequality), so the graph's greedy
routing assumption can lose some recall compared to `L2`/`Cosine`. This is a
known, documented tradeoff rather than a blocking issue — the same caveat
IVF-Flat and LSH apply to metrics outside their primary design target.

## 8. Common bugs

- Using the new node's own cap (`M`/`2*M`) when pruning an *existing*
  neighbor at a different layer — the cap depends on the layer being edited,
  not on which node is being inserted.
- Running the heuristic against distance to the query instead of distance to
  `base_id` when pruning an existing neighbor's list.
- Forgetting the `layer < adjacency_[node].size()` bound check in
  `search_layer` — nodes inserted before a given layer existed simply have no
  entry there.
- Mutating `adjacency_`/`node_level_`/`entry_point_`/`max_level_` before
  every throwing step of insertion has completed.
- Seeding `rng_` fresh on every `add()` instead of letting it advance across
  calls — breaks the full-build/incremental-add equivalence.

## 9. Complexity

Let `N` be the vector count, `D` the dimension, `M` the paper's `M`, `efc`
`ef_construction`, and `efs` `ef_search`.

| Operation | Time | Extra index memory |
| --- | --- | --- |
| Insert (`add`) | `O(efc * M * D * log(N))` amortized | `O(M)` per node, `O(2M)` at layer 0 |
| Search | `O(efs * M * D * log(N))` amortized | `O(efs)` |
| Full `build()` | `O(N)` insertions | `O(N * M)` total edges |

The original vectors stay in `VectorStore`; HNSW only adds the adjacency
lists and one level per indexed vector.

## 10. Integration and persistence

`Collection` selects `IndexKind::Hnsw` through `CollectionOptions::hnsw`
exactly like LSH selects `IndexKind::RandomProjectionLsh` — no metric guard,
since all three are supported. Persistence bumps the collection format to
**version 3** and assigns HNSW index-kind code `2` (LSH keeps `1`, Flat keeps
`0`). Only the four `HnswConfig` fields (`M`, `ef_construction`, `ef_search`,
`seed`) are serialized; the graph itself is never written to disk. Loading a
collection replays every vector through `Collection::insert()` in the
original internal-ID order, which drives `HnswIndex::add()` — with the same
seed and the same insertion order, this deterministically reconstructs the
exact same graph. This mirrors how LSH already persists as configuration
only, and keeps the on-disk format independent of implementation details
like the exact heap/adjacency layout.

Versions 1 and 2 continue to load exactly as before; a version-2 (or
version-1) file can never contain HNSW metadata, so loading rejects an
`Hnsw` index-kind code paired with a format version below 3.

## 11. Tests and benchmarks

`tests/hnsw_index_tests.cpp` mirrors the LSH test checklist: config
validation, build/add lifecycle errors, query-dimension validation,
`top_k == 0` and empty-index behavior, determinism (same seed twice, full
build vs. incremental add), duplicate/out-of-order `add()` rejection, stale-
store detection, and a `MatchesFlatIndex...AcrossMetrics` recall test run for
all three metrics on a small, well-separated dataset. `tests/collection_tests.cpp`
and `tests/persistence_tests.cpp` cover index selection, the config accessor,
and a save/load round trip.

`benchmarks/hnsw_recall_benchmarks.cpp` mirrors
`benchmarks/lsh_recall_benchmarks.cpp`: `FlatIndex` ground truth, `hnsw_build_ms`,
`recall_at_k`, and a portable `index_payload_bytes` estimate, sweeping `M`,
`ef_construction`, and `ef_search` independently. `BM_Glove25HnswSearch` in
`benchmarks/glove_25_benchmarks.cpp` runs the same sweep against the GloVe-25
dataset alongside the existing flat and LSH cases. On the repository's
synthetic uniform-random benchmark data, HNSW recall lands well above LSH's
at comparable latency, and degrades gracefully as `M`/`ef_search` are
increased relative to dataset size — the expected ANN recall/latency
tradeoff, not a defect.
