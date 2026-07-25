# IVF-Flat Implementation Guide

This is a learning guide for the scaffold in:

- `include/vectordb/indexes/inverted_file_flat_index.hpp`
- `src/indexes/inverted_file_flat_index.cpp`
- `tests/inverted_file_flat_index_tests.cpp`

The scaffold compiles, validates its configuration, and exposes its build state.
The algorithmic methods deliberately throw `std::logic_error`. Replace those
stubs one milestone at a time and add the suggested tests as you go.

## 1. What you are building

An inverted-file index divides the vector space into coarse regions. Each
region is represented by a centroid and owns a posting list of vector IDs.

```txt
training vectors
      |
      v
  k-means training
      |
      +------ centroids: c0, c1, ... c(nlist - 1)
      |
      v
assign every vector to its nearest centroid
      |
      +------ list 0: [7, 18, 31, ...]
      +------ list 1: [0,  4, 22, ...]
      `------ ...
```

At query time:

```txt
query
  |
  +-- compare with every centroid
  |
  +-- select the nearest nprobe lists
  |
  +-- gather the IDs in those lists
  |
  `-- compute exact L2 distance to those original vectors
          |
          `-- return exact top-k within the candidate set
```

The "Flat" in IVF-Flat means the vectors in the selected posting lists are
scored exactly and are not compressed. This is different from IVF-PQ, which
stores product-quantized codes and uses approximate candidate scores.

IVF-Flat is still an approximate nearest-neighbor algorithm because it ignores
vectors assigned to lists that were not probed.

## 2. Scope of the first version

Keep the first version intentionally narrow:

- Support `Metric::L2` only.
- Train the coarse centroids with Lloyd's k-means algorithm.
- Store internal vector IDs in each posting list.
- Assign each stored vector to exactly one list.
- Probe exactly `num_probes` distinct lists.
- Use exact L2 distance for final reranking.
- Keep flat search as the ground truth.
- Develop `InvertedFileFlatIndex` directly against `VectorStore` before
  connecting it to `Collection`.

The common IVF names and this scaffold's names are:

| IVF literature | Scaffold field | Meaning |
| --- | --- | --- |
| `nlist` | `num_lists` | Number of coarse centroids/posting lists |
| `nprobe` | `num_probes` | Number of posting lists visited per query |
| training iterations | `max_iterations` | Maximum Lloyd iterations |

Start with fixed-iteration training. Once correctness is established, you can
stop early when assignments no longer change or centroid movement falls below
a documented tolerance.

## 3. Understand the state layout

`centroids_` is a flat `std::vector<float>` in
`[list][dimension]` order. The offset of coordinate `dimension_index` in
centroid `list_id` is:

```txt
list_id * vectors_.dim() + dimension_index
```

You can construct a view of one centroid like this:

```cpp
const std::span<const float> centroid(
    centroids.data() + list_id * vectors_.dim(),
    vectors_.dim());
```

`posting_lists_[list_id]` stores the internal IDs assigned to that centroid.
The original coordinates stay in `vectors_`.

Maintain these invariants after every successful `build()` or `add()`:

1. `centroids_.size() == num_lists * vectors_.dim()`.
2. `posting_lists_.size() == num_lists`.
3. Every indexed ID occurs in exactly one posting list.
4. Every posting-list ID is less than `indexed_vector_count_`.
5. `indexed_vector_count_` is the number of vectors represented by the index.
6. `is_built_` is true only when all the other invariants hold.

## 4. Milestone 1: initialize centroids

Implement `initialize_centroids()`.

Before selecting centroids, reject a training set with fewer vectors than
`config_.num_lists`. Reusing the same vector as two initial centroids creates
empty clusters immediately and hides configuration mistakes.

### Simple starting strategy

Select `num_lists` distinct stored vectors using a generator seeded with
`config_.seed`, then copy their coordinates into the centroid array.

This is enough to learn the rest of the algorithm, but random initialization
can produce weak clusters. It may also converge to different local minima for
different seeds.

### Better follow-up: k-means++

After the simple version works, implement k-means++:

1. Select the first centroid using the seed.
2. For every vector, track its squared distance to the nearest selected
   centroid.
3. Select the next centroid with probability proportional to that squared
   distance.
4. Update the nearest-distance cache.
5. Repeat until `num_lists` centroids have been selected.

If you want tests that are exactly reproducible across different standard
library implementations, be careful with standard distributions: their
sampling sequences are not all specified byte-for-byte. One alternative is a
small documented sampler driven directly by `std::mt19937_64` output.

### Tests to add

- Building with fewer vectors than lists fails clearly.
- The returned centroid array has `num_lists * dimension` floats.
- A fixed seed produces the same initialization twice.
- Selected initial centroids are distinct vector IDs.
- A different seed changes at least one initial centroid on a suitable dataset.

The helper is private. Prefer testing it indirectly through deterministic
search behavior. If you temporarily expose internals while learning, remove
that exposure when the public behavior is covered.

## 5. Milestone 2: find the nearest centroid

Implement `find_nearest_centroid(input_vector, centroids)`.

For every centroid:

1. Compute the L2 distance from `input_vector`.
2. Keep the ID with the smallest distance.
3. If distances tie, keep the lower centroid ID.

For assignment, squared L2 distance is sufficient:

```txt
sum((input[d] - centroid[d])^2)
```

The square root is monotonic, so it cannot change which centroid is nearest.
Avoiding it makes the training loop cheaper. Final result scores must still use
the repository's normal L2 scoring so IVF results have the same score semantics
as `FlatIndex`.

Validate assumptions while bringing the code up:

- `input_vector.size() == vectors_.dim()`;
- the centroid array is non-empty;
- `centroids.size()` is divisible by `vectors_.dim()`.

Later, internal callers can rely on established invariants if you prefer to
remove redundant checks.

### Tests to add

- A point closest to the first centroid returns list 0.
- A point closest to the last centroid returns the last list.
- An exact-distance tie chooses the lower list ID.
- A multi-dimensional example catches an incorrect flat-array offset.

## 6. Milestone 3: train with Lloyd's algorithm

Implement `train_centroids()`.

Lloyd's algorithm alternates between assignment and update:

```txt
centroids = initialize_centroids()

repeat at most max_iterations times:
    sums   = num_lists * dimension zeroes
    counts = num_lists zeroes

    for each stored vector:
        list_id = nearest centroid
        add every coordinate to sums[list_id]
        increment counts[list_id]

    for each list:
        centroid[list] = sums[list] / counts[list]
```

Allocate `sums` and `counts` once outside the loop if convenient, then clear
them at the start of each iteration.

### Empty clusters

An empty cluster has `counts[list_id] == 0`, so dividing by its count is invalid.
Pick and document one policy. A useful deterministic policy is:

1. Find the vector whose current assigned-centroid distance is largest.
2. Copy that vector into the empty centroid.
3. If several vectors tie, choose the lower internal ID.
4. Do not use the same replacement vector for multiple empty clusters in the
   same iteration.

A simpler first attempt may keep the old centroid. That avoids invalid floats
but can leave a permanently unused list. If you start there, write a test and
upgrade the policy after the basic loop works.

### Early stopping

You have two reasonable choices:

- Always run exactly `max_iterations`. This is simple and deterministic.
- Stop when no vector changes assignment. Cache the previous assignment for
  every vector and compare each iteration.

Avoid an exact floating-point equality check between old and new centroids.
Assignment stability or a clearly defined movement tolerance is easier to
reason about.

### Overflow and allocation checks

Before allocating the centroid or sum arrays, check that:

```txt
num_lists * vectors_.dim()
```

fits in `std::size_t`. The LSH implementation's `checked_multiply()` is a useful
local example, though you should avoid duplicating it if you decide to promote
the helper into shared code.

### Tests to add

Use a tiny dataset with two obvious groups, for example points near `(0, 0)`
and points near `(10, 10)`.

- Training separates the two groups.
- Rebuilding with the same seed produces identical public search results.
- Training never creates `NaN` or infinite coordinates.
- An empty-cluster dataset follows your chosen policy.
- `max_iterations == 1` performs exactly one update.
- Oversized centroid storage fails with `std::length_error`.

Do not assert exact centroid coordinates for a random initialization unless the
initial selection is part of the documented contract. Tests based on final
assignments are less brittle.

## 7. Milestone 4: build posting lists

Implement `build_posting_lists(centroids)`.

Create exactly `num_lists` empty vectors. Visit internal IDs in increasing
order, find the nearest centroid, and append each ID to that centroid's list.
Increasing order gives stable list contents and deterministic tie behavior.

This method should build and return local state. Do not mutate
`posting_lists_` yet. Keeping the work local is important for the strong
exception guarantee in the next milestone.

### Tests to add

- Every vector appears once.
- No vector appears in two lists.
- Total posting-list sizes equal `vectors_.size()`.
- IDs within each list are in increasing order after a full build.
- Two clearly separated groups land in different lists.
- Equidistant vectors follow the centroid tie-break rule.

## 8. Milestone 5: make `build()` transactional

Implement `build()` by composing the earlier helpers:

```txt
new_centroids    = train_centroids()
new_posting_lists = build_posting_lists(new_centroids)

commit new_centroids
commit new_posting_lists
set indexed_vector_count
set is_built
```

Do all allocations and fallible work before changing the members. Commit with
moves only after both temporary structures are complete. If training or list
construction throws during a rebuild, the previous search-visible index must
remain usable.

Suggested lifecycle behavior:

- Require at least `num_lists` training vectors.
- Permit rebuilding an already-built index.
- Set `indexed_vector_count_ = vectors_.size()` only on successful commit.
- Never set `is_built_` early.

### Tests to add

- A successful build sets `is_built()` to true.
- Search before build throws `std::logic_error`.
- A failed initial build leaves `is_built()` false.
- Rebuilding after the store changes indexes all current vectors.
- A deliberately failed rebuild preserves the prior searchable state.

The last test may require a naturally failing condition or a small test seam.
Do not add production behavior solely to force an allocation failure.

## 9. Milestone 6: select lists for a query

Implement `select_lists(query)`.

Compute the query's L2 distance to all centroids and select the nearest
`config_.num_probes` IDs.

For the first implementation, storing all `(list_id, distance)` pairs and using
`std::partial_sort` is clear:

```txt
better(a, b):
    smaller distance wins
    if equal, smaller list ID wins
```

The selected IDs should be returned nearest-first. That order is useful for
debugging and for future adaptive probing, although exact reranking does not
depend on it.

Complexity is `O(num_lists * dimension + num_lists log(num_probes))` with a
partial sort. A fixed-size heap can reduce selection storage later.

### Tests to add

- `num_probes == 1` selects the nearest list.
- `num_probes == num_lists` selects every list.
- Ties use the lower list ID.
- Returned list IDs are unique.
- Selected lists are ordered by centroid distance.

## 10. Milestone 7: exact candidate reranking

Implement `search(query, top_k)`.

Validate the query dimension before checking `top_k`. This matches the existing
indexes: an invalid query remains invalid even if the caller asks for zero
results.

Recommended order:

1. Validate `query.size() == vectors_.dim()`.
2. Require `is_built_`.
3. Require `indexed_vector_count_ == vectors_.size()` so an externally changed
   store cannot be searched through a stale index.
4. Return empty for `top_k == 0`.
5. Select the nearest lists.
6. Visit every internal ID in those posting lists.
7. Score its original vector with
   `index_detail::score_vector(metric_, ...)`.
8. Stream each result into `index_detail::TopKAccumulator`.
9. Return `TopKAccumulator::finish()`.

Include the private `index_utils.hpp` from the source file, as `FlatIndex`
does. The accumulator already implements:

- lower scores first for L2;
- internal ID as the deterministic tie-breaker;
- bounded `O(top_k)` result storage;
- final sorted output.

Because each vector belongs to exactly one posting list and selected list IDs
are distinct, candidate IDs should not need deduplication. If duplicates occur,
an invariant is broken; hiding that with a set makes the underlying bug harder
to find.

The result count can be smaller than `top_k` when the probed lists contain too
few vectors. That is normal for an approximate index.

### Tests to add

- Query dimension validation happens before early returns.
- Searching before build fails.
- `top_k == 0` returns no results.
- Results are ordered by increasing L2 distance.
- Equal scores use lower internal IDs first.
- Result IDs are unique.
- Result count is at most `top_k`.
- Every returned score equals the exact score from `FlatIndex`.
- `num_probes == num_lists` matches `FlatIndex` exactly for all `top_k`.
- An easy clustered dataset finds the expected nearest neighbor with one probe.

The `num_probes == num_lists` test is especially valuable: IVF then examines
every stored vector, so any mismatch points to scoring or top-k logic rather
than approximation.

## 11. Milestone 8: incremental `add()`

The `Index::add()` contract says that the vector has already been appended to
`VectorStore`, exactly once, in internal-ID order.

Implement this policy:

1. Require a successful build.
2. Require `internal_id == indexed_vector_count_`.
3. Require `internal_id < vectors_.size()`.
4. Find the new vector's nearest trained centroid.
5. Ensure appending to that posting list can succeed.
6. Append the ID.
7. Increment `indexed_vector_count_` only after the append succeeds.

Do not retrain centroids on every insertion. That would be expensive and would
also require moving existing IDs whenever centroids change. Instead, document
that recall can degrade as the data distribution drifts and that callers should
rebuild periodically.

For exception safety, reserve the selected posting list before changing
`indexed_vector_count_`. If reserve or append throws, the prior index state must
remain unchanged.

### Tests to add

- Adding before build fails.
- Adding the next vector makes it searchable.
- Duplicate and out-of-order adds fail.
- Full build and incremental additions produce the same search results when
  they use the same fixed centroids.
- A stale store is detected if a vector is appended without calling `add()`.
- Rebuilding after many additions indexes the complete store.

## 12. Measure approximation quality

Correct ordering within the candidate set does not guarantee useful recall.
Compare IVF-Flat with `FlatIndex`:

```txt
recall@k =
    number of exact top-k IDs present in approximate results
    --------------------------------------------------------
                              k
```

Report recall beside query latency. Tune these independently:

- `num_lists`: more lists make each posting list smaller but increase centroid
  storage, training work, and coarse query comparisons.
- `num_probes`: more probes increase candidate work and usually improve recall.
- `max_iterations`: more iterations may improve clustering but only affect
  build time after convergence.
- initialization and seed: weak initial centroids can produce imbalanced lists.

Also report:

- average candidates scored per query;
- minimum, maximum, and average posting-list size;
- number of empty lists;
- build/training time;
- flat-search latency on the same data;
- dataset size and dimension.

Useful sanity checks:

- Increasing `num_probes` should not reduce the candidate set.
- At `num_probes == num_lists`, recall must be 1.0.
- Candidate count should be much smaller than the dataset for IVF to have a
  chance of beating flat search.

Use both clustered synthetic data and a real embedding/ANN dataset. Uniform
random vectors are useful as a stress case but often do not have clustering
structure that favors IVF.

## 13. Complexity and memory model

Let:

- `N` be the number of vectors;
- `D` be the dimension;
- `L` be `num_lists`;
- `P` be `num_probes`;
- `I` be the number of training iterations;
- `C` be the number of candidates in the probed lists.

Approximate costs are:

| Operation | Time | Extra index memory |
| --- | --- | --- |
| Training | `O(I * N * L * D)` | `O(L * D + N)` |
| Posting-list build | `O(N * L * D)` | `O(N + L)` IDs/containers |
| Query coarse pass | `O(L * D)` | `O(L)` simple implementation |
| Query exact rerank | `O(C * D + C log k)` or heap equivalent | `O(k)` |
| Incremental add | `O(L * D)` | one ID |

The original vectors use `O(N * D)` memory in `VectorStore`. IVF-Flat adds
centroids plus one internal ID per indexed vector; it does not create a second
copy of every vector.

## 14. Common bugs

Watch for these failure modes:

- Using `num_probes` as the number of candidate vectors instead of lists.
- Returning centroid distance as the result score.
- Forgetting exact reranking against original vectors.
- Treating larger L2 distance as better.
- Mixing centroid indices with vector internal IDs.
- Computing a flat centroid offset with `num_lists` instead of dimension.
- Dividing by zero for an empty cluster.
- Appending a vector to more than one posting list.
- Mutating member state halfway through a failed rebuild.
- Updating `indexed_vector_count_` before a fallible posting-list append.
- Searching a stale index after `VectorStore` changed directly.
- Assuming all posting lists are balanced.
- Claiming IVF-Flat is exact when `num_probes < num_lists`.

## 15. Integration with `Collection` comes last

Do not add `IndexKind::InvertedFileFlat` yet. The current `Collection`
constructor immediately calls `index_->build()` while its `VectorStore` is
empty. A k-means IVF index normally needs at least `num_lists` training vectors,
so simply adding a switch case would make collection construction fail.

Once the standalone index is correct, choose an explicit lifecycle design:

1. **Explicit training API:** callers insert training vectors, then call
   `collection.build_index()`.
2. **Lazy training:** collection uses flat search until enough vectors exist,
   then trains IVF.
3. **External training set:** construct IVF with representative vectors before
   normal insertion.
4. **Tiny-data fallback:** use flat search while the dataset is smaller than
   `num_lists`.

An explicit training API is the clearest design, but it changes the public
collection contract. Whichever design you choose, then update:

- `IndexKind`;
- `CollectionOptions`;
- the `make_index()` factory;
- configuration accessors;
- collection tests;
- the persistence version and metadata validation;
- README usage examples;
- benchmarks.

Do not reuse persistence index-kind numeric values. Preserve version-1 and
version-2 loading behavior, and only write IVF metadata in a newly documented
format version.

## 16. Suggested implementation order

Use this sequence so each step has a small debugging surface:

1. Implement squared-distance and nearest-centroid lookup.
2. Implement deterministic centroid initialization.
3. Run Lloyd training with a documented empty-cluster policy.
4. Build posting lists and verify their invariants.
5. Commit trained state transactionally in `build()`.
6. Select the nearest `num_probes` lists.
7. Exact-rerank candidates with `TopKAccumulator`.
8. Implement lifecycle-safe incremental insertion.
9. Compare full probing with `FlatIndex`.
10. Measure recall/latency and list balance.
11. Add k-means++ or other training improvements.
12. Design collection and persistence integration.

When all algorithmic TODOs are complete, this command finds remaining scaffold
markers:

```sh
rg 'TODO\\(IVF-' include/vectordb/indexes/inverted_file_flat_index.hpp \
    src/indexes/inverted_file_flat_index.cpp
```
