# Benchmark Implementation Guide

The `benchmarks/` directory contains a compiling Google Benchmark skeleton for
insertion, single search, batch search, save, and load. Every benchmark is
registered with a small initial set of arguments, but deliberately calls
`SkipWithError` until you implement it.

## Build and run

Keep benchmarks in a release build. Debug builds mainly measure disabled
optimizations and assertion overhead.

```sh
cmake -S . -B build-benchmarks \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECTORDB_BUILD_TESTS=OFF \
    -DVECTORDB_BUILD_BENCHMARKS=ON
cmake --build build-benchmarks
./build-benchmarks/benchmarks/vectordb_benchmarks
```

CMake uses an installed Google Benchmark package when one is available and
otherwise fetches version 1.8.3. Useful runner options include:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks --benchmark_list_tests
./build-benchmarks/benchmarks/vectordb_benchmarks --benchmark_filter=BatchSearch
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true
```

## Implement the shared data helpers first

Fill in `benchmark_utils.cpp` before the benchmark bodies.

`make_vector` should use a seeded pseudo-random generator and a fixed
distribution, for example uniform floats in `[-1, 1]`. A seed makes runs
comparable and failures reproducible. Do not use `std::random_device` for every
value: it is slow and makes each run different. For cosine benchmarks, either
avoid all-zero vectors or normalize generated vectors consistently.

`make_collection` can construct a `Collection`, generate `vector_count`
vectors, and insert them under deterministic IDs such as `vector_0`. Keep this
work outside the timed region of search and persistence-load benchmarks.

`make_temporary_path` should combine `std::filesystem::temp_directory_path()`
with a sufficiently unique filename. Persistence benchmarks should remove
their files even when an exception occurs; a tiny RAII cleanup class is useful.

## Insert benchmark

Read vector count and dimension from `state.range(0)` and `state.range(1)`.
Generate all vectors before the timed loop. Inside each iteration, construct a
new collection and insert that prepared data:

```cpp
for (auto _ : state)
{
    vectordb::Collection collection(dimension, vectordb::Metric::L2);
    // Insert every prepared vector.
    benchmark::DoNotOptimize(collection.size());
}
```

Constructing the collection inside the loop is intentional here because each
iteration needs an empty destination. Decide whether string-ID construction is
part of the operation you want to measure. If it is not, prepare IDs alongside
the vectors.

After the loop, report throughput with `state.SetItemsProcessed` using the
number of inserted vectors across all iterations. `SetBytesProcessed` can also
report the vector payload (`count * dimension * sizeof(float)`), but remember
that IDs and container overhead make actual memory traffic larger.

## Single-search benchmark

Create and populate the collection before entering the timed loop. Generate
the query once, then retain the returned results so the compiler cannot discard
the call:

```cpp
for (auto _ : state)
{
    auto results = collection->search(query, top_k);
    benchmark::DoNotOptimize(results);
}
```

Start with L2 so the first numbers are easy to understand. Once the harness is
stable, add registrations or a small argument-name helper for Dot and Cosine.
Measure several collection sizes, dimensions, and `top_k` values independently;
changing all three at once makes regressions harder to diagnose.

## Batch-search benchmark

The API expects one flat float buffer:

```txt
[query 0][query 1]...[query N-1]
```

Allocate `query_count * dimension` floats before the loop and fill each query
deterministically. Benchmark only `batch_search` in the timed loop. Report the
total number of queries with `SetItemsProcessed`, not just the number of
batches. This makes batch sizes easier to compare.

A useful follow-up comparison is repeated single search over the exact same
query buffer. Give it a separate benchmark name and identical arguments. The
ratio then shows whether batching changes overhead or only provides API
convenience.

## Persistence benchmarks

For save, build the collection before timing. Each iteration should save to the
temporary path. File removal or truncation setup is housekeeping, so place it
outside the measurement or bracket it with `state.PauseTiming()` and
`state.ResumeTiming()`. If you want durable-storage latency rather than stream
write latency, that is a separate benchmark and requires an explicit flush-to-
disk policy.

For load, create a valid file once before the loop. The timed operation should
be only `Collection::load(path)`. Keep the returned pointer alive through
`DoNotOptimize`, and destroy it before the next iteration if you want allocation
and destruction costs to be consistent. Remove the fixture file after the
benchmark completes.

Filesystem benchmarks are noisy because operating-system caches affect the
result. Treat warm-cache load as the default measurement and label it that way.
Reliable cold-cache measurement is platform-specific and usually needs elevated
system operations, so do not mix it into this small portable suite.

## Measurement hygiene

- Remove `SkipWithError` only after the corresponding body is implemented.
- Use `std::int64_t` conversions carefully when reading `state.range(...)`, then
  validate before converting to `std::size_t`.
- Keep random generation, fixture population, validation, logging, and cleanup
  outside timed loops unless they are the subject of the benchmark.
- Use `DoNotOptimize` on returned values and `ClobberMemory` only when you truly
  need to prevent reordering of memory effects.
- Run correctness tests before collecting performance data. A fast incorrect
  implementation is not a useful baseline.
- Keep the machine as idle as practical and use repetitions. Compare medians,
  not a single best run.
- Add larger argument sets gradually. The current values are intentionally small
  enough for quick local feedback.

## A sensible implementation order

1. Implement deterministic helpers and temporary-file cleanup.
2. Implement single search; it has the simplest fixture lifetime.
3. Implement batch search and add repeated-single-search comparison.
4. Implement insertion and its throughput counters.
5. Implement warm-cache save/load benchmarks.
6. Record a release-build baseline before optimizing the library.

