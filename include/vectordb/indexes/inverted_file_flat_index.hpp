#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectordb/indexes/index.hpp"
#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

namespace vectordb {

struct InvertedFileFlatConfig {
    // Usually called nlist: the number of Voronoi cells in the coarse
    // quantizer.
    std::size_t num_lists = 64;

    // Usually called nprobe: the number of nearby cells searched per query.
    std::size_t num_probes = 4;

    // Maximum number of Lloyd iterations used while training the centroids.
    std::size_t max_iterations = 20;

    // Makes centroid initialization reproducible.
    std::uint64_t seed = 42;
};

// Learning scaffold for an IVF-Flat approximate nearest-neighbor index.
//
// The first implementation intentionally targets L2 distance only. Posting
// lists contain internal IDs, so candidate vectors remain uncompressed in the
// referenced VectorStore ("Flat"). See docs/IVF_FLAT_TODO.md.
class InvertedFileFlatIndex : public Index {
   public:
    InvertedFileFlatIndex(
        const VectorStore &vectors, Metric metric,
        InvertedFileFlatConfig config = InvertedFileFlatConfig{});

    // Trains the coarse quantizer and replaces all posting lists.
    void build() override;

    // Assigns one newly appended vector to its nearest trained centroid.
    void add(std::uint64_t internal_id) override;

    // Probes the nearest coarse lists and exactly reranks their vectors.
    std::vector<InternalSearchResult> search(std::span<const float> query,
                                             std::size_t top_k) const override;

    bool is_built() const;
    const InvertedFileFlatConfig &config() const;

   private:
    using ListId = std::size_t;
    using PostingList = std::vector<std::uint64_t>;

    // TODO(IVF-1): Select num_lists initial centroids.
    std::vector<float> initialize_centroids() const;

    // TODO(IVF-2): Return the nearest centroid in the supplied flat array.
    ListId find_nearest_centroid(std::span<const float> input_vector,
                                 std::span<const float> centroids) const;

    // TODO(IVF-3): Run Lloyd's algorithm and return trained centroids.
    std::vector<float> train_centroids() const;

    // TODO(IVF-4): Assign every stored vector to exactly one posting list.
    std::vector<PostingList> build_posting_lists(
        std::span<const float> centroids) const;

    // TODO(IVF-5): Return the num_probes nearest centroid IDs.
    std::vector<ListId> select_lists(std::span<const float> query) const;

    const VectorStore &vectors_;
    Metric metric_;
    InvertedFileFlatConfig config_;

    // Flattened as [list][dimension].
    std::vector<float> centroids_;
    std::vector<PostingList> posting_lists_;
    std::size_t indexed_vector_count_ = 0;
    bool is_built_ = false;
};

}  // namespace vectordb
