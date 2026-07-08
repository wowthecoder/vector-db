Your repo is at a good “toy DB core” stage. The next mistake to avoid is adding HNSW/IVF/etc directly into `VectorStore`. Before that, split the project into **storage**, **distance**, **index interface**, **benchmarking**, and later **embedding/API/UI**.

## Phase 1 — Stabilize the core API

Implement a clean abstraction like:

```cpp
class VectorIndex {
public:
    virtual void add(std::span<const float> vector, uint64_t id) = 0;
    virtual std::vector<SearchResult> search(std::span<const float> query, size_t k) const = 0;
    virtual void build() {}
    virtual void save(const std::filesystem::path& path) const = 0;
    virtual void load(const std::filesystem::path& path) = 0;
    virtual ~VectorIndex() = default;
};
```

Your `VectorStore` should own vectors, ids, and metadata. Indexes should be replaceable modules that know how to search the vectors.

Suggested new files:

```txt
include/vectordb/
  types.hpp              // ids, SearchResult, Metric enum
  vector_store.hpp       // raw vector storage
  index.hpp              // VectorIndex interface
  indexes/
    brute_force.hpp
    flat.hpp
    hnsw.hpp             // later
    ivf_flat.hpp         // later
  io/
    binary_io.hpp
  benchmark/
    benchmark.hpp

src/
  indexes/
    brute_force.cpp
    flat.cpp
  io/
    binary_io.cpp
```

Also make sure your CMake uses C++20 if you use `std::span`:

```cmake
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

## Phase 2 — Implement exact search properly

Before ANN algorithms, build a strong exact baseline.

Implement:

1. **Flat / brute-force index**

   * Stores all vectors.
   * Computes distance to every vector.
   * Returns top-k.
   * This becomes your correctness reference.

2. **Top-k selection**

   * Use a fixed-size max heap for top-k instead of sorting all distances.
   * Sorting all distances is okay for testing, but too slow for realistic benchmarks.

3. **Batch search**

   * Search many queries at once.
   * Useful for benchmarking and later API use.

4. **Distance metrics**

   * L2 squared distance.
   * Cosine distance.
   * Inner product / maximum inner product search.
   * For cosine, normalize vectors once on insert/build, not every query unless necessary.

5. **Persistence**

   * Save/load vectors and ids.
   * Save/load index state later.

Done criteria: exact search returns the same results as a slow reference implementation in tests.

## Phase 3 — Build the benchmark suite early

Do this before implementing advanced indexes. Otherwise you will not know whether your “optimization” helped.

Benchmark dimensions:

```txt
Dataset size:       1k, 10k, 100k, 1M later
Dimension:          32, 128, 384, 768, 1536
Metric:             L2, cosine, inner product
k:                  1, 10, 100
Query count:        100, 1000, 10000
Output metrics:     latency p50/p95/p99, QPS, recall@k, memory, build time
```

Use brute force as ground truth, then compare ANN algorithms against it.

ANN-Benchmarks is a useful reference because it is specifically designed to compare approximate nearest-neighbor algorithms across datasets and parameter settings, and it reports quality/performance tradeoffs rather than only raw speed. ([ann-benchmarks.com][1])

I would structure your benchmark folder like:

```txt
benchmarks/
  CMakeLists.txt
  bench_flat.cpp
  bench_recall.cpp
  datasets/
    generate_random.cpp
    load_fvecs.cpp
    load_hdf5.cpp   // optional later
```

Output benchmark results to CSV:

```txt
algorithm,dataset_size,dim,k,metric,build_ms,query_p50_us,query_p95_us,qps,recall,memory_mb
flat,100000,128,10,cosine,0,850,1200,1176,1.0,51.2
```

This makes it easy to plot later in Python.

## Phase 4 — First ANN algorithms

Implement algorithms in this order:

### 1. Random projection / LSH

Good learning project. Not usually the strongest baseline, but simple and teaches approximate search.

Add:

```txt
indexes/lsh.hpp
indexes/lsh.cpp
```

Benchmark knobs:

```txt
num_tables
num_bits_per_table
num_candidates
```

### 2. IVF-Flat

This is the first “real vector DB style” index.

Idea:

1. Run k-means to create clusters.
2. Assign each vector to nearest centroid.
3. At query time, search only the closest `nprobe` clusters.
4. Compute exact distances inside those clusters.

This teaches clustering, inverted lists, build/search tradeoffs, and recall/latency tuning.

Files:

```txt
indexes/ivf_flat.hpp
indexes/ivf_flat.cpp
clustering/kmeans.hpp
clustering/kmeans.cpp
```

Benchmark knobs:

```txt
num_clusters
nprobe
kmeans_iterations
```

### 3. HNSW

This should probably be your first major “serious” ANN implementation.

HNSW is a graph-based approximate nearest-neighbor method using hierarchical navigable small-world graphs; the original paper describes a multilayer graph where upper layers help route quickly and lower layers refine the search. ([arXiv][2])

Files:

```txt
indexes/hnsw.hpp
indexes/hnsw.cpp
```

Benchmark knobs:

```txt
M
ef_construction
ef_search
```

This is where your project becomes interesting.

## Phase 5 — Compression and memory-efficient indexes

After flat, IVF, and HNSW, add compression.

Implement:

1. **Scalar quantization**

   * Store vectors as `uint8_t` or `int8_t`.
   * Compare approximate distances.
   * Optional final rerank using original float vectors.

2. **Product quantization**

   * Split vector into subspaces.
   * Quantize each subspace.
   * Use lookup tables at query time.

3. **IVF-PQ**

   * IVF for pruning.
   * PQ for compressed candidate scoring.
   * Optional reranking.

This phase matters because real vector DBs are often memory-bound, not just compute-bound.

## Phase 6 — Disk-based / large-scale search

Once in-memory search works, implement a disk-aware path.

DiskANN is a Microsoft research/project line focused on scalable vector indexing for large web/enterprise-scale search, including cost-effective search where the full index may not fit comfortably in RAM. ([GitHub][3])

You do not need to copy DiskANN immediately. Instead, build toward it:

```txt
storage/
  mmap_vector_store.hpp
  mmap_vector_store.cpp
indexes/
  disk_graph.hpp
  disk_graph.cpp
```

Start with:

1. Memory-mapped vector storage.
2. Graph index in RAM, vectors on disk.
3. Candidate reranking by fetching vectors from disk.
4. Later: disk-resident graph layout.

## Phase 7 — Add document storage and semantic search

Only after your vector layer is usable, add document semantics.

You need these layers:

```txt
Document
  id
  title
  body
  metadata
  chunks[]

Chunk
  chunk_id
  document_id
  text
  embedding
```

Implement:

1. **Text chunking**

   * Fixed token/window chunking.
   * Overlap support.
   * Store chunk → document mapping.

2. **Embedding ingestion**

   * First support loading precomputed embeddings from JSON/CSV.
   * Then add a Python ingestion script or C++ HTTP client for embedding APIs.
   * Keep embedding generation separate from the C++ vector index at first.

3. **Search result format**

   * Return chunk text, document id, score, and metadata.
   * Later add hybrid search with keyword filters.

Suggested structure:

```txt
apps/
  semantic_search_cli/
    main.cpp

server/
  vectordb_server.cpp

python/
  embed_documents.py
  ingest.py

web/
  simple-ui/
```

## Phase 8 — API server

Start simple. Do not build a distributed database yet.

Expose endpoints like:

```txt
POST /collections
POST /collections/{name}/vectors
POST /collections/{name}/search
POST /collections/{name}/documents
POST /collections/{name}/semantic-search
GET  /collections/{name}/stats
```

Use something lightweight first:

* C++ HTTP server: Crow, Drogon, Pistache, or Boost.Beast.
* Or use a Python FastAPI wrapper that calls your C++ library.

For learning C++ systems, C++ server is good. For quickly building a product-like demo, Python FastAPI + C++ shared library is easier.

## Phase 9 — UI

Your first UI can be very simple:

```txt
Upload documents
Choose embedding model
Build index
Search query
Show top chunks
Show latency / algorithm / score
```

The UI should let you switch between:

```txt
Flat
IVF-Flat
HNSW
IVF-PQ
Disk-based graph
```

That will make your project much more impressive because users can see the speed/recall tradeoff directly.

## Phase 10 — Research-paper implementation track

Once your framework is stable, pick papers and implement them as plugins. Good candidates to study include HNSW, DiskANN-style graph search, ScaNN-style pruning/quantization, and newer HNSW variants. Google’s ScaNN combines search-space pruning and quantization for efficient vector similarity search, especially for maximum inner product search and related metrics. ([Google Research][4])

Keep each paper implementation isolated:

```txt
indexes/
  paper_x/
    paper_x_index.hpp
    paper_x_index.cpp
    README.md
```

For every paper/index, require:

```txt
build()
search()
save()
load()
benchmark config
paper notes
```

## My recommended immediate next steps

Do these next, in order:

1. Create `index.hpp` with a `VectorIndex` interface.
2. Move brute-force search into `indexes/brute_force.hpp/.cpp`.
3. Add `SearchResult`, `Metric`, and `VectorId` types.
4. Add proper top-k heap search.
5. Add benchmark executable for flat search.
6. Add CSV benchmark output.
7. Add tests comparing brute-force results against a slow reference.
8. Add IVF-Flat.
9. Add HNSW.
10. Add document chunking + semantic search demo.

A good near-term target is:

> “I can load 100k vectors, run brute force, IVF-Flat, and HNSW, then produce a CSV comparing recall@10, p95 latency, QPS, build time, and memory.”

That would be a strong foundation before you move into SOTA/paper implementations.

[1]: https://ann-benchmarks.com/index.html?utm_source=chatgpt.com "ANN-Benchmarks"
[2]: https://arxiv.org/abs/1603.09320?utm_source=chatgpt.com "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
[3]: https://github.com/microsoft/diskann?utm_source=chatgpt.com "DiskANN3: A Composable Vector Indexing Library"
[4]: https://research.google/blog/announcing-scann-efficient-vector-similarity-search/?utm_source=chatgpt.com "Announcing ScaNN: Efficient Vector Similarity Search"
