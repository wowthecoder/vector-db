#pragma once

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

#include "vectordb/indexes/index.hpp"
#include "vectordb/indexes/random_projection_lsh_index.hpp"
#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

namespace vectordb {

struct CollectionOptions {
    IndexKind index_kind = IndexKind::Flat;
    RandomProjectionLshConfig lsh;
};

class Collection {
   public:
    Collection(std::size_t dim, Metric metric);
    Collection(std::size_t dim, Metric metric, CollectionOptions options);
    Collection(const Collection &) = delete;
    Collection &operator=(const Collection &) = delete;
    Collection(Collection &&) = delete;
    Collection &operator=(Collection &&) = delete;

    void insert(const std::string &external_id, std::span<const float> vector);

    std::vector<SearchResult> search(std::span<const float> query,
                                     std::size_t top_k) const;
    std::vector<std::vector<SearchResult>> batch_search(
        std::span<const float> queries, std::size_t top_k) const;

    void save(const std::filesystem::path &path) const;
    static std::unique_ptr<Collection> load(const std::filesystem::path &path);

    std::size_t size() const;
    std::size_t dim() const;
    Metric metric() const;
    IndexKind index_kind() const;
    const RandomProjectionLshConfig &lsh_config() const;

   private:
    Metric metric_;
    VectorStore vectors_;
    CollectionOptions options_;
    std::unique_ptr<Index> index_;

    std::vector<SearchResult> internal_to_external_list(
        const std::vector<InternalSearchResult> &internal_results) const;

    std::unordered_map<std::string, std::uint64_t> external_to_internal_;
    std::vector<std::string> internal_to_external_;
};

}  // namespace vectordb
