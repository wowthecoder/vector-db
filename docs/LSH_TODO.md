# Random Projection / LSH Implementation Guide

The implementation consists of:

- `include/vectordb/indexes/random_projection_lsh_index.hpp`
- `src/indexes/random_projection_lsh_index.cpp`

The index was implemented and tested directly against `VectorStore` before
being connected to `Collection`. Flat search remains the default, while callers
can now select LSH explicitly through `CollectionOptions`.

## Scope the first version

Start with random-hyperplane locality-sensitive hashing for cosine similarity.
For a random projection vector `r`, one hash bit is:

```txt
bit(x, r) = 1 when dot(x, r) >= 0, otherwise 0
```

Concatenate `num_bits_per_table` bits into a `std::uint64_t` signature. Similar
directions are more likely to have the same signature. Each independent table
uses its own random projections, giving a close vector several chances to
collide with the query.

Do not claim that this hash family provides the same guarantees for raw dot
product or L2 distance. Keep those metrics unsupported initially, or implement
and document a metric-appropriate transformation or hash family later.

## 1. Generate deterministic projections

Implement `generate_random_projections()`.

- Seed one `std::mt19937_64` from `config_.seed`.
- Draw projection coordinates from a standard normal distribution.
- Generate exactly:

  ```txt
  num_tables * num_bits_per_table * vectors_.dim()
  ```

  floats.
- Store them in `projections_` using the documented `[table][bit][dimension]`
  layout.
- Generate projections in a stable loop order so a fixed seed is reproducible.

One possible offset formula is:

```txt
((table * num_bits_per_table) + bit) * dimension
```

Normalizing each projection is optional for sign-based hashing because a
positive scale does not change which side of the hyperplane a vector occupies.

## 2. Compute a signature

Implement `compute_signature(vector, table_index)`.

For each bit in that table:

1. Find the corresponding projection in `projections_`.
2. Compute its dot product with the input vector.
3. Set the bit when the result is non-negative.

Use an unsigned shift:

```cpp
signature |= (std::uint64_t{1} << bit_index);
```

Validate the vector dimension and table index while bringing up the code. The
constructor already limits signatures to 64 bits.

Useful signature tests include:

- the same vector always produces the same signature;
- rebuilding with the same seed reproduces all signatures;
- opposite non-zero vectors produce complementary bits unless a dot product is
  exactly zero;
- changing the seed changes at least some signatures.

## 3. Build the hash tables

Implement `build()` with replacement semantics:

1. Generate a fresh projection array.
2. Create `config_.num_tables` empty hash tables.
3. For every internal vector ID and every table, compute a signature and append
   the ID to that signature's bucket.
4. Set `is_built_` only after all new state is complete.

Prefer building into local temporary containers and moving them into the
members at the end. If allocation or hashing throws, the object then retains
its previous valid state instead of appearing successfully rebuilt with partial
tables.

For the first version, require callers to rebuild after adding vectors to the
referenced `VectorStore`. Incremental insertion can be added after correctness
is established.

## 4. Collect candidates

Implement `collect_candidates(query)`.

- Compute the query signature independently for every table.
- Look up the exactly matching bucket in each table.
- Merge IDs without returning duplicates.
- Stop or truncate according to `config_.num_candidates`.
- Make candidate selection deterministic. Do not let
  `std::unordered_map` iteration order determine which IDs survive a limit.

A simple deterministic MVP is to visit tables in numeric order, preserve IDs in
bucket insertion order, and use a set only to track which IDs have already been
seen.

Exact-bucket lookup can return too few candidates. Once the MVP works, add
multi-probe search by checking signatures with Hamming distance one, then two.
Keep probe count separate from `num_candidates`: one controls bucket lookup
work, while the other controls exact scoring work.

## 5. Score and rerank exactly

LSH should only choose candidates. Final result scores must use the original
vectors and the requested metric.

Implement `score_vector()` and `select_top_k()` using the helpers in
`vectordb/distance.hpp`. For the cosine MVP:

- higher similarity is better;
- return scores in descending order;
- use the lower internal ID as the tie-breaker, matching `FlatIndex`;
- return at most `min(top_k, candidate_count)` results.

Use a fixed-size heap as in `FlatIndex`, but feed candidate scores directly into
it. There is no need to allocate a second array containing every score.

Then finish `search()` by collecting candidates and reranking them. Define the
behavior for an empty candidate set explicitly; returning an empty result is a
reasonable starting point.

## 6. Add correctness tests

Create `tests/random_projection_lsh_index_tests.cpp` and add it to
`tests/CMakeLists.txt`. Cover at least:

- invalid configuration values;
- searching before `build()`;
- query dimension validation;
- deterministic builds with a fixed seed;
- `top_k == 0`;
- an empty vector store;
- no duplicate IDs when a candidate appears in several tables;
- score ordering and internal-ID tie-breaking;
- rebuilding after the vector store changes.

For a small, deliberately easy cosine dataset, assert that the expected nearest
neighbor is found. Do not require perfect recall for arbitrary random datasets;
LSH is approximate, and such a test will become flaky as parameters change.

## 7. Measure recall against `FlatIndex` — implemented

Use `FlatIndex` as ground truth. For each query, compare its exact top `k` IDs
with the LSH result:

```txt
recall@k = |exact IDs intersect approximate IDs| / k
```

Benchmark these controls independently:

- `num_tables`: more collision opportunities, memory, and hashing work;
- `num_bits_per_table`: narrower buckets and usually fewer candidates;
- `num_candidates`: more exact scoring work and usually higher recall;
- dataset size and dimension;
- cosine query latency, build time, and index memory.

Always report recall beside latency. An approximate search time without its
result quality is not meaningful.

The implementation is in `benchmarks/lsh_recall_benchmarks.cpp`. It reports
`recall_at_k`, per-query latency, query throughput, `lsh_build_ms`, and the
portable logical `index_payload_bytes` estimate. See `benchmarks/README.md` for
build commands, counter definitions, and the focused benchmark command.

## 8. Integrate only after the standalone index works — implemented

`Index` is the runtime-polymorphic search contract implemented by `FlatIndex`
and `RandomProjectionLshIndex`. `Collection` owns a `std::unique_ptr<Index>` and
selects its implementation from `CollectionOptions`; flat remains the default.

The integration preserves these properties:

- brute force remains available as the exact default and ground truth;
- callers select approximate search explicitly;
- inserting a vector updates every LSH table through `Index::add`;
- persistence version 2 stores the index kind and LSH configuration, then
  deterministically reconstructs the tables while loading;
- persistence version 1 remains readable and defaults to flat search;
- public search results still translate internal IDs to external IDs in one
  place.

## Suggested implementation order

1. Projection generation and signature tests.
2. Full index build and bucket tests.
3. Candidate collection and deduplication.
4. Exact cosine reranking and `search()`.
5. Comparison with `FlatIndex` and recall measurement.
6. Multi-probe lookup and tuning.
7. Collection integration and persistence.
8. Metric-specific alternatives for L2 or dot product.
