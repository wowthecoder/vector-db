#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <unordered_map>
#include <vector>

#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

namespace vectordb {

struct RandomProjectionLshConfig {
    std::size_t num_tables = 8;
    std::size_t num_bits_per_table = 12;
    std::size_t num_candidates = 100;
    std::uint64_t seed = 42;
};

// Approximate nearest-neighbor index based on random-hyperplane LSH.
//
// The first implementation should target cosine similarity. See
// docs/LSH_TODO.md for the intended implementation sequence and the changes
// needed to support other metrics correctly.
class RandomProjectionLshIndex {
   public:
    RandomProjectionLshIndex(
        const VectorStore &vectors, Metric metric,
        RandomProjectionLshConfig config = RandomProjectionLshConfig{});

    // Rebuilds projections and hash tables from the current VectorStore.
    void build();

    std::vector<InternalSearchResult> search(std::span<const float> query,
                                             std::size_t top_k) const;

    bool is_built() const;
    const RandomProjectionLshConfig &config() const;

   private:
    using Signature = std::uint64_t;
    using Bucket = std::vector<std::uint64_t>;
    using HashTable = std::unordered_map<Signature, Bucket>;

    void generate_random_projections();
    Signature compute_signature(std::span<const float> vector,
                                std::size_t table_index) const;
    std::vector<std::uint64_t> collect_candidates(
        std::span<const float> query) const;
    float score_vector(const float *query, const float *candidate) const;
    std::vector<InternalSearchResult> select_top_k(
        std::span<const std::uint64_t> candidate_ids,
        std::span<const float> query, std::size_t top_k) const;

    const VectorStore &vectors_;
    Metric metric_;
    RandomProjectionLshConfig config_;

    // Flattened as [table][bit][dimension].
    std::vector<float> projections_;
    std::vector<HashTable> tables_;
    bool is_built_ = false;
};

}  // namespace vectordb
