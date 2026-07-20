#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "benchmark_utils.hpp"
#include "vectordb/types.hpp"

namespace {

std::vector<float> make_queries(std::size_t query_count,
                                std::size_t dimension) {
    constexpr std::uint32_t query_seed = 1'000'000;

    std::vector<float> queries;
    queries.reserve(query_count * dimension);

    for (std::size_t i = 0; i < query_count; ++i) {
        const auto query = vectordb::benchmarks::make_vector(
            dimension, query_seed + static_cast<std::uint32_t>(i));
        queries.insert(queries.end(), query.begin(), query.end());
    }

    return queries;
}

template <vectordb::Metric metric>
void BM_CollectionBatchSearch(benchmark::State &state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dimension = static_cast<std::size_t>(state.range(1));
    const auto query_count = static_cast<std::size_t>(state.range(2));
    const auto top_k = static_cast<std::size_t>(state.range(3));

    constexpr std::uint32_t collection_seed = 42;
    auto collection = vectordb::benchmarks::make_collection(
        count, dimension, metric, collection_seed);
    const auto queries = make_queries(query_count, dimension);

    for (auto _ : state) {
        auto results = collection->batch_search(queries, top_k);
        benchmark::DoNotOptimize(results);
    }

    state.SetItemsProcessed(state.iterations() *
                            static_cast<std::int64_t>(query_count));
}

template <vectordb::Metric metric>
void BM_RepeatedSingleSearch(benchmark::State &state) {
    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dimension = static_cast<std::size_t>(state.range(1));
    const auto query_count = static_cast<std::size_t>(state.range(2));
    const auto top_k = static_cast<std::size_t>(state.range(3));

    constexpr std::uint32_t collection_seed = 42;
    auto collection = vectordb::benchmarks::make_collection(
        count, dimension, metric, collection_seed);
    const auto queries = make_queries(query_count, dimension);

    for (auto _ : state) {
        for (std::size_t i = 0; i < query_count; ++i) {
            const std::span<const float> query(queries.data() + i * dimension,
                                               dimension);
            auto results = collection->search(query, top_k);
            benchmark::DoNotOptimize(results);
        }
    }

    state.SetItemsProcessed(state.iterations() *
                            static_cast<std::int64_t>(query_count));
}

void apply_batch_search_arguments(benchmark::internal::Benchmark *benchmark) {
    benchmark
        ->ArgNames({"count", "dimension", "query_count", "top_k"})
        // Collection-size scaling.
        ->Args({1'000, 128, 8, 10})
        ->Args({10'000, 128, 8, 10})
        ->Args({100'000, 128, 8, 10})

        // Dimension scaling.
        ->Args({10'000, 64, 8, 10})
        ->Args({10'000, 256, 8, 10})
        ->Args({10'000, 768, 8, 10})

        // Batch-size scaling.
        ->Args({10'000, 128, 1, 10})
        ->Args({10'000, 128, 32, 10})

        // top_k scaling.
        ->Args({10'000, 128, 8, 1})
        ->Args({10'000, 128, 8, 50})
        ->Args({10'000, 128, 8, 100});
}

}  // namespace

BENCHMARK_TEMPLATE(BM_CollectionBatchSearch, vectordb::Metric::L2)
    ->Apply(apply_batch_search_arguments)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_CollectionBatchSearch, vectordb::Metric::Dot)
    ->Apply(apply_batch_search_arguments)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_CollectionBatchSearch, vectordb::Metric::Cosine)
    ->Apply(apply_batch_search_arguments)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_TEMPLATE(BM_RepeatedSingleSearch, vectordb::Metric::L2)
    ->Apply(apply_batch_search_arguments)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_RepeatedSingleSearch, vectordb::Metric::Dot)
    ->Apply(apply_batch_search_arguments)
    ->Unit(benchmark::kMillisecond);
BENCHMARK_TEMPLATE(BM_RepeatedSingleSearch, vectordb::Metric::Cosine)
    ->Apply(apply_batch_search_arguments)
    ->Unit(benchmark::kMillisecond);
