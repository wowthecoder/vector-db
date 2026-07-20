#include <benchmark/benchmark.h>

#include "benchmark_utils.hpp"
#include "vectordb/types.hpp"

namespace {

template <vectordb::Metric metric>
void BM_CollectionSearch(benchmark::State &state) {
    // state.range(0): collection size
    // state.range(1): vector dimension
    // state.range(2): top_k
    const int64_t count = state.range(0);
    const int64_t dimension = state.range(1);
    const int64_t top_k = state.range(2);

    constexpr std::uint32_t seed = 42;
    auto collection =
        vectordb::benchmarks::make_collection(count, dimension, metric, seed);
    auto query = vectordb::benchmarks::make_vector(dimension, seed + count);

    // Populate once, then time repeated Collection::search calls.
    for (auto _ : state) {
        auto results = collection->search(query, top_k);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations());
}

}  // namespace

BENCHMARK_TEMPLATE(BM_CollectionSearch, vectordb::Metric::L2)
    ->ArgNames({"count", "dimension", "top_k"})

    // Collection-size scaling
    ->Args({1'000, 128, 10})
    ->Args({10'000, 128, 10})
    ->Args({100'000, 128, 10})

    // Dimension scaling; count and top_k remain fixed
    ->Args({10'000, 64, 10})
    ->Args({10'000, 256, 10})
    ->Args({10'000, 768, 10})

    // top_k scaling; count and dimension remain fixed
    ->Args({10'000, 128, 1})
    ->Args({10'000, 128, 50})
    ->Args({10'000, 128, 100})

    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_CollectionSearch, vectordb::Metric::Dot)
    ->ArgNames({"count", "dimension", "top_k"})

    ->Args({1'000, 128, 10})
    ->Args({10'000, 128, 10})
    ->Args({100'000, 128, 10})

    ->Args({10'000, 64, 10})
    ->Args({10'000, 256, 10})
    ->Args({10'000, 768, 10})

    ->Args({10'000, 128, 1})
    ->Args({10'000, 128, 50})
    ->Args({10'000, 128, 100})

    ->Unit(benchmark::kMicrosecond);

BENCHMARK_TEMPLATE(BM_CollectionSearch, vectordb::Metric::Cosine)
    ->ArgNames({"count", "dimension", "top_k"})

    ->Args({1'000, 128, 10})
    ->Args({10'000, 128, 10})
    ->Args({100'000, 128, 10})

    ->Args({10'000, 64, 10})
    ->Args({10'000, 256, 10})
    ->Args({10'000, 768, 10})

    ->Args({10'000, 128, 1})
    ->Args({10'000, 128, 50})
    ->Args({10'000, 128, 100})

    ->Unit(benchmark::kMicrosecond);
