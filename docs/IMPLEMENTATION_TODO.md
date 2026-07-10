# Batch Search and Persistence TODO

The public APIs and source files are scaffolded, but intentionally throw
`std::logic_error`. Implement them in the order below.

## 1. FlatIndex batch search

Implement `FlatIndex::batch_search` in `src/indexes/flat_index.cpp`.

`queries` is a flattened sequence of query vectors:

```txt
[query 0 floats][query 1 floats]...[query N floats]
```

Each query contains `vectors_.dim()` floats.

TODO:

1. Reject input when `queries.size()` is not divisible by `vectors_.dim()`.
2. Compute `query_count` from the flattened input size.
3. Reserve space for `query_count` result lists.
4. For each query, create a `std::span<const float>` over the corresponding
   `dim` floats.
5. Call the existing `search(query, top_k)` and append its result.
6. Return an empty outer vector for an empty query span.

Start with the simple loop above. Parallel execution can be added and measured
later without changing the API.

## 2. Collection batch search

Implement `Collection::batch_search` in `src/collection.cpp`.

TODO:

1. Call `index_.batch_search(queries, top_k)`.
2. Reserve one public result list per internal result list.
3. Convert every `InternalSearchResult` to `SearchResult`, using
   `internal_to_external_` in the same way as `Collection::search`.
4. Preserve query order and best-to-worst result order.

Avoid duplicating search calculations in `Collection`; its job is to translate
IDs and delegate searching to the index.

## 3. Choose a binary file format

Implement persistence in `src/io/collection_io.cpp`. Start with a small,
versioned binary format:

```txt
magic bytes
format version
metric
dimension
vector count

for each vector:
    external ID byte length
    external ID bytes
    dimension float values
```

Use fixed-width integer fields such as `std::uint32_t` and `std::uint64_t`.
Document whether numeric fields use the host byte order or a fixed byte order.
A fixed byte order is more portable, but host byte order is acceptable for a
first local-only version if the limitation is documented.

Do not write raw `Collection`, `std::vector`, or `std::string` object memory.
Those objects contain implementation-specific pointers and bookkeeping.

## 4. Save

Implement `Collection::save`.

TODO:

1. Open `path` as a binary `std::ofstream`.
2. Throw a useful exception if opening fails.
3. Write and validate each field listed in the format above.
4. Read each stored vector with `vectors_.get(internal_id)`.
5. Obtain its external ID from `internal_to_external_`.
6. Check the stream state after writing and report incomplete writes.

Writing to a temporary file and renaming it after success is a useful later
improvement because it avoids leaving a partially written target file.

## 5. Load

Implement `Collection::load`.

TODO:

1. Open `path` as a binary `std::ifstream`.
2. Validate magic bytes and reject unsupported versions.
3. Validate the metric, dimension, count, and every string length before
   allocating memory.
4. Create the result with
   `std::make_unique<Collection>(dimension, metric)`.
5. Read each ID and vector, then call `insert`.
6. Reject truncated files, duplicate or empty IDs, invalid dimensions, and
   unreasonable lengths/counts.
7. Return the `std::unique_ptr` only after the complete file is valid.

`load` returns a pointer because `FlatIndex` contains a reference to its
collection's `VectorStore`. Copying or moving a `Collection` naively could make
that reference point at the wrong store, so those operations are explicitly
deleted for now.

## 6. Tests to add

Executable specifications are scaffolded in:

```txt
tests/batch_search_tests.cpp
tests/persistence_tests.cpp
```

Their names start with `DISABLED_`, so GoogleTest compiles and discovers them
without running them while the methods still throw `std::logic_error`.

After implementing a behavior, remove `DISABLED_` from the relevant test name
and run:

```sh
cmake --build build
ctest --test-dir build --output-on-failure
```

After the initial tests pass, add malformed persistence files that verify
truncated data, wrong magic bytes, unsupported versions, duplicate IDs, and
unreasonable stored lengths are rejected.

Use a unique path under the operating system's temporary directory and remove
test files after each persistence test.
