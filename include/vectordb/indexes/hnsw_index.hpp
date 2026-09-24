#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

#include "vectordb/indexes/index.hpp"
#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

namespace vectordb {

struct HnswConfig {
    static constexpr std::size_t max_M = 512;

    std::size_t M = 16;
    std::size_t ef_construction = 200;
    std::size_t ef_search = 50;
    // Drives level assignment and is the only source of randomness, so a
    // fixed seed reproduces the exact same graph.
    std::uint64_t seed = 42;
};

// Approximate nearest-neighbor index based on Hierarchical Navigable Small
// World graphs (Malkov & Yashunin). See docs/HNSW_TODO.md for the algorithm
// walkthrough and implementation notes.
class HnswIndex : public Index {
   public:
    HnswIndex(const VectorStore &vectors, Metric metric,
             HnswConfig config = HnswConfig{});

    // Rebuilds the whole graph from the current VectorStore with replacement
    // semantics.
    void build() override;
    void add(std::uint64_t internal_id) override;

    std::vector<InternalSearchResult> search(std::span<const float> query,
                                             std::size_t top_k) const override;

    bool is_built() const;
    const HnswConfig &config() const;

   private:
    // A node scored against a fixed reference point, oriented so a smaller
    // orient_score always means "closer", regardless of metric.
    struct ScoredNode {
        std::uint64_t id;
        float orient_score;
    };

    // A pending change to one existing node's neighbor list at one layer,
    // computed ahead of any mutation so `add()` can commit without throwing.
    struct NeighborUpdate {
        std::uint64_t node_id;
        std::size_t layer;
        std::vector<std::uint64_t> neighbors;
    };

    // Everything a new insertion needs to commit, computed against the
    // read-only graph state so failures never leave a partially-mutated
    // graph.
    struct InsertPlan {
        std::size_t level = 0;
        // Chosen neighbors for the new node, indexed by layer (size level+1).
        std::vector<std::vector<std::uint64_t>> own_neighbors;
        std::vector<NeighborUpdate> neighbor_updates;
        bool becomes_entry_point = false;
    };

    // Greedy best-first search on one layer. Returns up to `ef` nodes,
    // closest-first.
    std::vector<ScoredNode> search_layer(
        std::span<const float> query,
        const std::vector<std::uint64_t> &entry_points, std::size_t ef,
        std::size_t layer,
        const std::vector<std::vector<std::vector<std::uint64_t>>> &adjacency,
        bool prefer_higher) const;

    // Algorithm 4 diversity heuristic: keeps a candidate only if it is
    // closer to `base_id` than to any neighbor already selected.
    std::vector<std::uint64_t> select_neighbors_heuristic(
        std::uint64_t base_id, const std::vector<std::uint64_t> &candidates,
        std::size_t m) const;

    // Computes the full set of graph changes for inserting `internal_id`
    // without mutating anything. May throw (allocation, invalid vectors);
    // the caller commits the result with `apply_insert`.
    InsertPlan prepare_insert(
        const std::vector<std::vector<std::vector<std::uint64_t>>> &adjacency,
        std::uint64_t entry_point, std::size_t max_level,
        bool has_entry_point, std::mt19937_64 &rng,
        std::uint64_t internal_id) const;

    // Applies a previously computed plan. Callers reserve capacity for the
    // new node ahead of time so this never needs to allocate.
    static void apply_insert(
        std::vector<std::vector<std::vector<std::uint64_t>>> &adjacency,
        std::vector<std::size_t> &node_level, std::uint64_t &entry_point,
        std::size_t &max_level, bool &has_entry_point,
        std::uint64_t internal_id, InsertPlan plan);

    const VectorStore &vectors_;
    Metric metric_;
    HnswConfig config_;

    // Adjacency indexed as [node][layer] -> neighbor internal IDs. Node i's
    // outer vector has size node_level_[i] + 1.
    std::vector<std::vector<std::vector<std::uint64_t>>> adjacency_;
    std::vector<std::size_t> node_level_;
    std::uint64_t entry_point_ = 0;
    std::size_t max_level_ = 0;
    bool has_entry_point_ = false;
    std::size_t indexed_vector_count_ = 0;
    bool is_built_ = false;
    std::mt19937_64 rng_;
};

}  // namespace vectordb
