#include <benchmark/benchmark.h>

#include "benchmark_utils.hpp"
#include "vectordb/types.hpp"

namespace {

void BM_CollectionSave(benchmark::State &state) {
    const int64_t count = state.range(0);
    const int64_t dimension = state.range(1);

    // Populate once and time writes to a temporary file.
    constexpr std::uint32_t seed = 42;
    auto collection = vectordb::benchmarks::make_collection(
        count, dimension, vectordb::Metric::L2, seed);
    const auto path = vectordb::benchmarks::make_temporary_path();
    const vectordb::benchmarks::TemporaryFileCleanup cleanup(path);

    for (auto _ : state) {
        collection->save(path);
    }
}

void BM_CollectionLoad(benchmark::State &state) {
    // state.range(0): collection size
    // state.range(1): vector dimension
    const int64_t count = state.range(0);
    const int64_t dimension = state.range(1);

    // Save one fixture file, then time repeated Collection::load calls.
    constexpr std::uint32_t seed = 42;
    auto collection = vectordb::benchmarks::make_collection(
        count, dimension, vectordb::Metric::L2, seed);
    const auto path = vectordb::benchmarks::make_temporary_path();
    const vectordb::benchmarks::TemporaryFileCleanup cleanup(path);

    collection->save(path);

    for (auto _ : state) {
        auto loaded = vectordb::Collection::load(path);
        benchmark::DoNotOptimize(loaded);
    }
}

}  // namespace

BENCHMARK(BM_CollectionSave)
    ->Args({1'000, 128})
    ->Args({10'000, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();

BENCHMARK(BM_CollectionLoad)
    ->Args({1'000, 128})
    ->Args({10'000, 128})
    ->Unit(benchmark::kMillisecond)
    ->UseRealTime();
