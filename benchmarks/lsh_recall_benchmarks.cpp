#include <benchmark/benchmark.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <unordered_set>
#include <vector>

#include "benchmark_utils.hpp"
#include "vectordb/indexes/flat_index.hpp"
#include "vectordb/indexes/random_projection_lsh_index.hpp"
#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

namespace {

double recall_at_k(
    const std::vector<std::vector<vectordb::InternalSearchResult>> &exact,
    const std::vector<std::vector<vectordb::InternalSearchResult>>
        &approximate) {
    std::size_t matches = 0;
    std::size_t expected = 0;

    for (std::size_t query_index = 0; query_index < exact.size();
         ++query_index) {
        std::unordered_set<std::uint64_t> exact_ids;
        exact_ids.reserve(exact[query_index].size());
        for (const auto &result : exact[query_index]) {
            exact_ids.insert(result.internal_id);
        }

        expected += exact[query_index].size();
        std::unordered_set<std::uint64_t> approximate_ids;
        approximate_ids.reserve(approximate[query_index].size());
        for (const auto &result : approximate[query_index]) {
            approximate_ids.insert(result.internal_id);
        }
        for (const std::uint64_t internal_id : approximate_ids) {
            matches += exact_ids.contains(internal_id) ? 1 : 0;
        }
    }

    return expected == 0
               ? 1.0
               : static_cast<double>(matches) / static_cast<double>(expected);
}

double logical_index_payload_bytes(
    std::size_t vector_count, std::size_t dimension,
    const vectordb::RandomProjectionLshConfig &config) {
    const double projection_bytes =
        static_cast<double>(config.num_tables) *
        static_cast<double>(config.num_bits_per_table) *
        static_cast<double>(dimension) * sizeof(float);
    const double membership_bytes = static_cast<double>(config.num_tables) *
                                    static_cast<double>(vector_count) *
                                    sizeof(std::uint64_t);
    return projection_bytes + membership_bytes;
}

void BM_RandomProjectionLshSearch(benchmark::State &state) {
    // state.range(0): vector count
    // state.range(1): vector dimension
    // state.range(2): top_k
    // state.range(3): number of hash tables
    // state.range(4): bits per table
    // state.range(5): maximum exact-scoring candidates
    // state.range(6): number of queries used for recall measurement
    const auto vector_count = static_cast<std::size_t>(state.range(0));
    const auto dimension = static_cast<std::size_t>(state.range(1));
    const auto top_k = static_cast<std::size_t>(state.range(2));
    const auto num_tables = static_cast<std::size_t>(state.range(3));
    const auto num_bits_per_table = static_cast<std::size_t>(state.range(4));
    const auto num_candidates = static_cast<std::size_t>(state.range(5));
    const auto query_count = static_cast<std::size_t>(state.range(6));

    constexpr std::uint32_t data_seed = 42;
    constexpr std::uint64_t projection_seed = 1729;

    vectordb::VectorStore vectors(dimension);
    for (std::size_t index = 0; index < vector_count; ++index) {
        vectors.add(vectordb::benchmarks::make_vector(
            dimension, data_seed + static_cast<std::uint32_t>(index)));
    }

    std::vector<std::vector<float>> queries;
    queries.reserve(query_count);
    for (std::size_t index = 0; index < query_count; ++index) {
        queries.push_back(vectordb::benchmarks::make_vector(
            dimension,
            data_seed + static_cast<std::uint32_t>(vector_count + index)));
    }

    const vectordb::FlatIndex exact_index(vectors, vectordb::Metric::Cosine);
    std::vector<std::vector<vectordb::InternalSearchResult>> exact_results;
    exact_results.reserve(query_count);
    for (const auto &query : queries) {
        exact_results.push_back(exact_index.search(query, top_k));
    }

    const vectordb::RandomProjectionLshConfig config{
        .num_tables = num_tables,
        .num_bits_per_table = num_bits_per_table,
        .num_candidates = num_candidates,
        .seed = projection_seed,
    };
    vectordb::RandomProjectionLshIndex lsh_index(
        vectors, vectordb::Metric::Cosine, config);

    const auto build_start = std::chrono::steady_clock::now();
    lsh_index.build();
    const auto build_end = std::chrono::steady_clock::now();
    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start)
            .count();

    std::vector<std::vector<vectordb::InternalSearchResult>>
        approximate_results;
    approximate_results.reserve(query_count);
    for (const auto &query : queries) {
        approximate_results.push_back(lsh_index.search(query, top_k));
    }
    const double recall = recall_at_k(exact_results, approximate_results);

    std::size_t query_index = 0;
    for (auto _ : state) {
        auto results = lsh_index.search(queries[query_index], top_k);
        benchmark::DoNotOptimize(results);
        query_index = (query_index + 1) % queries.size();
    }

    state.SetItemsProcessed(state.iterations());
    state.counters["recall_at_k"] = recall;
    state.counters["lsh_build_ms"] = build_ms;
    state.counters["index_payload_bytes"] =
        logical_index_payload_bytes(vector_count, dimension, config);
}

void apply_lsh_arguments(benchmark::internal::Benchmark *benchmark) {
    // Baseline: count=10k, dimension=128, k=10, tables=8, bits=8,
    // candidates=100, queries=50. Each following case changes one control.
    benchmark
        ->ArgNames({"count", "dimension", "top_k", "num_tables", "num_bits",
                    "num_candidates", "query_count"})
        ->Args({10'000, 128, 10, 8, 8, 100, 50})

        // Dataset-size scaling.
        ->Args({1'000, 128, 10, 8, 8, 100, 50})
        ->Args({50'000, 128, 10, 8, 8, 100, 50})

        // Dimension scaling.
        ->Args({10'000, 32, 10, 8, 8, 100, 50})
        ->Args({10'000, 384, 10, 8, 8, 100, 50})

        // Hash-table scaling.
        ->Args({10'000, 128, 10, 2, 8, 100, 50})
        ->Args({10'000, 128, 10, 4, 8, 100, 50})
        ->Args({10'000, 128, 10, 16, 8, 100, 50})

        // Signature-width scaling.
        ->Args({10'000, 128, 10, 8, 4, 100, 50})
        ->Args({10'000, 128, 10, 8, 12, 100, 50})
        ->Args({10'000, 128, 10, 8, 16, 100, 50})

        // Candidate-limit scaling.
        ->Args({10'000, 128, 10, 8, 8, 10, 50})
        ->Args({10'000, 128, 10, 8, 8, 25, 50})
        ->Args({10'000, 128, 10, 8, 8, 400, 50})

        // Result-count scaling.
        ->Args({10'000, 128, 1, 8, 8, 100, 50})
        ->Args({10'000, 128, 50, 8, 8, 100, 50});
}

}  // namespace

BENCHMARK(BM_RandomProjectionLshSearch)
    ->Apply(apply_lsh_arguments)
    ->Unit(benchmark::kMicrosecond);
