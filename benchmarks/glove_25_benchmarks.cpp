#include <benchmark/benchmark.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_set>
#include <vector>

#include "ann_dataset.hpp"
#include "vectordb/indexes/flat_index.hpp"
#include "vectordb/indexes/random_projection_lsh_index.hpp"
#include "vectordb/types.hpp"

namespace {

#ifndef VECTORDB_GLOVE25_DEFAULT_PATH
#define VECTORDB_GLOVE25_DEFAULT_PATH "benchmark-data/glove-25-angular.vdbann"
#endif

constexpr std::uint64_t kProjectionSeed = 1729;
constexpr std::size_t kFlatValidationQueries = 10;

struct CachedDataset {
    std::unique_ptr<vectordb::benchmarks::AnnDataset> dataset;
    std::string error;
};

std::filesystem::path dataset_path() {
    const char *override_path =
        std::getenv("VECTORDB_GLOVE25_PREPARED_DATASET");
    return override_path != nullptr && override_path[0] != '\0'
               ? std::filesystem::path(override_path)
               : std::filesystem::path(VECTORDB_GLOVE25_DEFAULT_PATH);
}

const CachedDataset &cached_dataset() {
    static const CachedDataset cached = [] {
        CachedDataset result;
        try {
            result.dataset = std::make_unique<vectordb::benchmarks::AnnDataset>(
                vectordb::benchmarks::AnnDataset::load(dataset_path()));
        } catch (const std::exception &error) {
            result.error = std::string(error.what()) +
                           ". Prepare it with: python3 benchmarks/"
                           "prepare_ann_dataset.py benchmark-data/"
                           "glove-25-angular.hdf5 benchmark-data/"
                           "glove-25-angular.vdbann";
        }
        return result;
    }();
    return cached;
}

const vectordb::benchmarks::AnnDataset *get_dataset(benchmark::State &state) {
    const CachedDataset &cached = cached_dataset();
    if (!cached.dataset) {
        state.SkipWithError(cached.error);
        return nullptr;
    }
    return cached.dataset.get();
}

double recall_at_k(
    const vectordb::benchmarks::AnnDataset &dataset,
    const std::vector<std::vector<vectordb::InternalSearchResult>> &results,
    std::size_t top_k) {
    std::size_t matches = 0;

    for (std::size_t query_index = 0; query_index < results.size();
         ++query_index) {
        const std::span<const std::uint64_t> expected =
            dataset.neighbors(query_index).first(top_k);
        std::unordered_set<std::uint64_t> expected_ids(expected.begin(),
                                                       expected.end());
        std::unordered_set<std::uint64_t> returned_ids;
        returned_ids.reserve(results[query_index].size());

        for (const auto &result : results[query_index]) {
            if (returned_ids.insert(result.internal_id).second &&
                expected_ids.contains(result.internal_id)) {
                ++matches;
            }
        }
    }

    const std::size_t expected_count = results.size() * top_k;
    return expected_count == 0 ? 1.0
                               : static_cast<double>(matches) /
                                     static_cast<double>(expected_count);
}

template <typename Index>
double measure_recall(const vectordb::benchmarks::AnnDataset &dataset,
                      const Index &index, std::size_t top_k,
                      std::size_t query_count) {
    std::vector<std::vector<vectordb::InternalSearchResult>> results;
    results.reserve(query_count);
    for (std::size_t query_index = 0; query_index < query_count;
         ++query_index) {
        results.push_back(index.search(dataset.query(query_index), top_k));
    }
    return recall_at_k(dataset, results, top_k);
}

void set_common_counters(benchmark::State &state,
                         const vectordb::benchmarks::AnnDataset &dataset,
                         std::size_t recall_queries, double recall) {
    state.SetItemsProcessed(state.iterations());
    state.counters["dataset_vectors"] =
        static_cast<double>(dataset.vectors().size());
    state.counters["dimension"] = static_cast<double>(dataset.dimension());
    state.counters["recall_queries"] = static_cast<double>(recall_queries);
    state.counters["recall_at_k"] = recall;
}

void BM_Glove25FlatSearch(benchmark::State &state) {
    const auto *dataset = get_dataset(state);
    if (dataset == nullptr) {
        return;
    }

    const auto top_k = static_cast<std::size_t>(state.range(0));
    const auto requested_query_pool = static_cast<std::size_t>(state.range(1));
    if (top_k > dataset->neighbors_per_query()) {
        state.SkipWithError(
            "top_k exceeds the dataset's ground-truth neighbor count");
        return;
    }

    const std::size_t query_pool =
        std::min(requested_query_pool, dataset->query_count());
    if (query_pool == 0) {
        state.SkipWithError("query pool must contain at least one query");
        return;
    }

    const vectordb::FlatIndex index(dataset->vectors(),
                                    vectordb::Metric::Cosine);
    const std::size_t validation_queries =
        std::min(kFlatValidationQueries, query_pool);
    const double recall =
        measure_recall(*dataset, index, top_k, validation_queries);

    std::size_t query_index = 0;
    for (auto _ : state) {
        auto results = index.search(dataset->query(query_index), top_k);
        benchmark::DoNotOptimize(results);
        query_index = (query_index + 1) % query_pool;
    }

    set_common_counters(state, *dataset, validation_queries, recall);
    state.counters["index_build_ms"] = 0.0;
}

void BM_Glove25LshSearch(benchmark::State &state) {
    const auto *dataset = get_dataset(state);
    if (dataset == nullptr) {
        return;
    }

    const auto top_k = static_cast<std::size_t>(state.range(0));
    const auto num_tables = static_cast<std::size_t>(state.range(1));
    const auto num_bits = static_cast<std::size_t>(state.range(2));
    const auto num_candidates = static_cast<std::size_t>(state.range(3));
    const auto requested_query_pool = static_cast<std::size_t>(state.range(4));
    const auto requested_recall_queries =
        static_cast<std::size_t>(state.range(5));

    if (top_k > dataset->neighbors_per_query()) {
        state.SkipWithError(
            "top_k exceeds the dataset's ground-truth neighbor count");
        return;
    }

    const std::size_t query_pool =
        std::min(requested_query_pool, dataset->query_count());
    const std::size_t recall_queries =
        std::min(requested_recall_queries, dataset->query_count());
    if (query_pool == 0 || recall_queries == 0) {
        state.SkipWithError(
            "query pool and recall query count must both be nonzero");
        return;
    }

    const vectordb::RandomProjectionLshConfig config{
        .num_tables = num_tables,
        .num_bits_per_table = num_bits,
        .num_candidates = num_candidates,
        .seed = kProjectionSeed,
    };
    vectordb::RandomProjectionLshIndex index(dataset->vectors(),
                                             vectordb::Metric::Cosine, config);

    const auto build_start = std::chrono::steady_clock::now();
    index.build();
    const auto build_end = std::chrono::steady_clock::now();
    const double build_ms =
        std::chrono::duration<double, std::milli>(build_end - build_start)
            .count();

    const double recall =
        measure_recall(*dataset, index, top_k, recall_queries);

    std::size_t query_index = 0;
    for (auto _ : state) {
        auto results = index.search(dataset->query(query_index), top_k);
        benchmark::DoNotOptimize(results);
        query_index = (query_index + 1) % query_pool;
    }

    set_common_counters(state, *dataset, recall_queries, recall);
    state.counters["index_build_ms"] = build_ms;
    state.counters["lsh_build_ms"] = build_ms;
}

void apply_glove_lsh_arguments(benchmark::internal::Benchmark *benchmark) {
    benchmark
        ->ArgNames({"top_k", "num_tables", "num_bits", "num_candidates",
                    "query_pool", "recall_queries"})

        // Baseline.
        ->Args({10, 8, 12, 1'000, 100, 1'000})

        // Hash-table scaling.
        ->Args({10, 4, 12, 1'000, 100, 1'000})
        ->Args({10, 16, 12, 1'000, 100, 1'000})

        // Signature-width scaling.
        ->Args({10, 8, 10, 1'000, 100, 1'000})
        ->Args({10, 8, 14, 1'000, 100, 1'000})

        // Candidate-limit scaling.
        ->Args({10, 8, 12, 100, 100, 1'000})
        ->Args({10, 8, 12, 5'000, 100, 1'000});
}

}  // namespace

BENCHMARK(BM_Glove25FlatSearch)
    ->ArgNames({"top_k", "query_pool"})
    ->Args({10, 100})
    ->Iterations(100)
    ->Unit(benchmark::kMillisecond);

BENCHMARK(BM_Glove25LshSearch)
    ->Apply(apply_glove_lsh_arguments)
    ->Iterations(100)
    ->Unit(benchmark::kMicrosecond);
