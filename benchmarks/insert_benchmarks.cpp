#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>

#include "benchmark_utils.hpp"

namespace {

void BM_CollectionInsert(benchmark::State &state) {
    // state.range(0): vectors inserted per iteration
    // state.range(1): vector dimension
    if (state.range(0) <= 0 || state.range(1) <= 0) {
        state.SkipWithError("count and dimension must be positive");
        return;
    }

    const auto count = static_cast<std::size_t>(state.range(0));
    const auto dimension = static_cast<std::size_t>(state.range(1));

    std::vector<std::vector<float>> vectors;
    std::vector<std::string> ids;
    vectors.reserve(count);
    ids.reserve(count);

    for (std::size_t i = 0; i < count; ++i) {
        vectors.push_back(vectordb::benchmarks::make_vector(dimension, 42 + i));
        ids.push_back("vector_" + std::to_string(i));
    }

    // Time insertion only, excluding collection construction and destruction.
    for (auto _ : state) {
        state.PauseTiming();
        {
            vectordb::Collection collection(dimension, vectordb::Metric::L2);
            state.ResumeTiming();

            // Insert every prepared vector.
            for (std::size_t i = 0; i < count; ++i) {
                collection.insert(ids[i], vectors[i]);
            }

            // Prevent removal of the insertion code above.
            benchmark::DoNotOptimize(collection.size());
            state.PauseTiming();
        }  // collection is destroyed while timing is paused
        state.ResumeTiming();
    }

    state.SetItemsProcessed(state.iterations() *
                            static_cast<std::int64_t>(count));
}

}  // namespace

BENCHMARK(BM_CollectionInsert)
    ->Args({1'000, 128})
    ->Args({10'000, 128})
    ->Unit(benchmark::kMillisecond);
