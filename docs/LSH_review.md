Overall, this is a solid educational MVP: the structure is understandable, exact reranking is correct, configuration is deterministic within one runtime, persistence works, and the tests cover the public behavior well. The largest problems are candidate selection quality, metric enforcement, and index lifecycle safety.

## High-priority findings

1. Candidate limiting discards the value of multiple tables.

In [collect_candidates()](/home/wowthecoder/vector-db/src/indexes/random_projection_lsh_index.cpp:143), candidates are accepted in table and insertion order. Once `num_candidates` is reached, the function immediately returns.

Consequences:

- Earlier tables dominate.
- Lower internal IDs tend to be favored.
- A vector colliding in several tables receives no higher priority.
- Adding tables may not improve recall if the first table fills the limit.
- A true neighbor can be excluded before exact reranking.

This likely explains much of the low benchmark recall.

A better approach is:

1. Visit every matching bucket.
2. Count collisions per internal ID.
3. Sort or partially select IDs by collision count.
4. Exact-score the best `num_candidates`.

```cpp
std::unordered_map<std::uint64_t, std::size_t> collision_counts;

for (each matching bucket) {
    for (const auto id : bucket) {
        ++collision_counts[id];
    }
}

// Choose highest collision counts, with internal ID as tie-breaker.
```

After that, add multi-probe search by checking nearby signatures, starting with one-bit Hamming neighbors.

2. The standalone LSH index accepts inappropriate metrics.

The header describes random-hyperplane LSH for cosine similarity, but the [constructor](/home/wowthecoder/vector-db/src/indexes/random_projection_lsh_index.cpp:13) accepts L2 and dot product. `Collection` rejects them, but direct users of `RandomProjectionLshIndex` can still select them.

Exact reranking with L2 does not fix candidate selection: the hash family is still based on angular similarity.

Enforce cosine inside the LSH constructor:

```cpp
if (metric != Metric::Cosine) {
    throw std::invalid_argument(
        "Random-projection LSH supports cosine similarity only");
}
```

Metric-specific L2 or maximum-inner-product schemes should be separate index implementations or hash policies.

3. Rebuilding is not exception-safe.

[build()](/home/wowthecoder/vector-db/src/indexes/random_projection_lsh_index.cpp:29) replaces `projections_` first, then constructs `new_tables`.

If table construction throws during a rebuild:

- `projections_` contains the new projections.
- `tables_` still contains buckets made with the old projections.
- `is_built_` may remain `true`.

Subsequent searches would hash queries using projections that do not correspond to the stored tables.

Generate both structures locally and commit them together:

```cpp
auto new_projections = generate_random_projections();
auto new_tables = build_tables(new_projections);

projections_ = std::move(new_projections);
tables_ = std::move(new_tables);
is_built_ = true;
```

4. Incremental insertion is not transactional.

[Collection::insert()](/home/wowthecoder/vector-db/src/collection.cpp:42) mutates the vector store and ID maps before calling `index_->add()`. If LSH insertion throws after updating only some tables, the collection and index diverge.

Possible policies:

- Mark the index stale and rebuild before the next search.
- Make insertion transactional with rollback support in `VectorStore`.
- Have LSH compute signatures and prepare allocations before committing.
- Document that allocation failure can require an index rebuild.

Also, `Index::add()` is ambiguous. `add_vector()` or `on_vector_added()` better communicates that the vector must already exist in the store.

5. Zero vectors fail too late.

A zero vector hashes to an all-one signature because every projection dot product is zero and the test uses `>= 0`. It is accepted during build or insertion, but cosine scoring later throws in [cosine_similarity()](/home/wowthecoder/vector-db/src/distance.cpp:29).

Worse, a stored zero vector can make a non-zero query fail if it enters the candidate set.

Reject zero vectors when inserting into a cosine collection, or normalize vectors on insertion and reject them there. The current test expecting a search-time failure preserves an inconvenient behavior rather than preventing it.

## Efficiency findings

- [collect_candidates()](/home/wowthecoder/vector-db/src/indexes/random_projection_lsh_index.cpp:143) should reserve capacity for its vector and set:

  ```cpp
  const auto capacity =
      std::min(config_.num_candidates, vectors_.size());

  candidates.reserve(capacity);
  unique_ids.reserve(capacity);
  ```

- `compute_signature()` recalculates `table_offset` inside the bit loop. Hoist it outside.

- Projection count multiplication at [generate_random_projections()](/home/wowthecoder/vector-db/src/indexes/random_projection_lsh_index.cpp:102) can overflow `size_t`. Validate multiplication before allocation and place practical limits on table count.

- Build performs many `unordered_map` rehashes and bucket-vector allocations. Reserving an estimated number of signatures per table would help.

- Search first allocates scored results for every candidate, then constructs a second top-k heap in [select_top_k()](/home/wowthecoder/vector-db/src/indexes/index_utils.cpp:40). A streaming `TopKAccumulator::consider(result)` would reduce temporary memory from `O(C + k)` to `O(k)`.

- Cosine scoring recalculates the query norm and candidate norm for every candidate. Normalize vectors once or cache stored norms, then final cosine scoring becomes a dot product against one normalized query.

- `std::normal_distribution` is deterministic for repeated runs using the same standard-library implementation, but its exact sequence is not guaranteed across standard libraries. Persisting only the seed may therefore rebuild different projections on another platform. Serialize projections or use a precisely specified sampling transform if cross-platform reproducibility matters.

## Benchmark review

The recall formula and separation of setup from timed search are correct. Exact ground truth is calculated before timing, and latency measures only LSH search.

The main benchmark weakness is its dataset. Vectors and queries are independent uniform random points generated in [benchmark_utils.cpp](/home/wowthecoder/vector-db/benchmarks/benchmark_utils.cpp:13). In high dimensions, their nearest-neighbor structure is weak and unlike embedding datasets. Combined with exact-bucket-only probing, this naturally produces recall close to zero.

Add at least three dataset modes:

- Uniform random, retained as a stress baseline.
- Clustered synthetic vectors, with queries perturbed around cluster members.
- A real normalized embedding dataset or standard ANN dataset.

Other benchmark improvements:

- Increase recall queries from 50 to at least 500–1,000 for more stable estimates.
- Report `recall_at_k_percent`; the raw fraction is formatted confusingly by Google Benchmark.
- Report average candidates scored, matching buckets, and unique collisions.
- Add paired `FlatIndex` latency on the exact same vectors and queries.
- Add a separate build benchmark instead of only exposing one build-time counter.
- Measure actual resident/index allocation or keep clearly labelling the current memory counter as a lower-bound payload estimate.
- Process the complete query set per benchmark iteration, or guarantee enough iterations to cycle through every query. Short runs currently may time only an initial subset at [lsh_recall_benchmarks.cpp](/home/wowthecoder/vector-db/benchmarks/lsh_recall_benchmarks.cpp:125).

## Test review

The suite is already strong in these areas:

- Configuration boundaries
- Build lifecycle
- Fixed-seed determinism
- Candidate deduplication and limits
- Exact reranking and tie-breaking
- 64-bit signatures
- Incremental insertion
- Collection integration
- Persistence and legacy format loading

Important missing tests:

- Direct LSH construction with L2/dot must fail.
- Full build and incremental insertion should produce identical results for the same dataset.
- Duplicate or out-of-order calls to `add()` should be rejected or explicitly supported.
- Invalid internal IDs passed to `add()`.
- Known signature results using injected, fixed projections; current tests cannot directly catch projection-layout errors.
- Property tests over many seeds asserting:
  - unique result IDs;
  - exact score values;
  - correct descending order;
  - result count bounded by `top_k` and candidate limit.
- Candidate collision-count ranking once implemented.
- Multi-probe lookup behavior.
- Projection-size overflow and practical configuration limits.
- Truncated version-2 persistence metadata and each invalid LSH configuration field.
- Automated AddressSanitizer and UndefinedBehaviorSanitizer runs.
- Unit tests for the benchmark’s `recall_at_k()` helper; it is currently local to the benchmark source.

## Coding-style suggestions

- Remove comments that restate container types, such as the explanation around [table insertion](/home/wowthecoder/vector-db/src/indexes/random_projection_lsh_index.cpp:43).
- Rename local `vector` variables to `stored_vector` or `candidate_vector`.
- Remove redundant `vectordb::` qualification while already inside `namespace vectordb`.
- Document `Index::build()` and `Index::add()` lifecycle requirements in [index.hpp](/home/wowthecoder/vector-db/include/vectordb/indexes/index.hpp:17).
- Track `indexed_vector_count_` so the index can detect stale stores, duplicate additions, and out-of-order IDs.

My recommended implementation order is:

1. Enforce cosine and reject zero vectors.
2. Replace first-seen candidate truncation with collision-count ranking.
3. Make rebuild state atomic.
4. Add multi-probe lookup.
5. Cache normalization/norms and stream scores into top-k.
6. Upgrade the benchmark datasets and diagnostic counters.
7. Add property, lifecycle, and sanitizer tests.