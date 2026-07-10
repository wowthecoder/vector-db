#pragma once

#include "vectordb/indexes/flat_index.hpp"
#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace vectordb
{

    class Collection
    {
    public:
        Collection(std::size_t dim, Metric metric);
        Collection(const Collection &) = delete;
        Collection &operator=(const Collection &) = delete;
        Collection(Collection &&) = delete;
        Collection &operator=(Collection &&) = delete;

        void insert(const std::string &external_id, std::span<const float> vector);

        std::vector<SearchResult> search(std::span<const float> query, std::size_t top_k) const;
        std::vector<std::vector<SearchResult>> batch_search(
            std::span<const float> queries,
            std::size_t top_k) const;

        void save(const std::filesystem::path &path) const;
        static std::unique_ptr<Collection> load(const std::filesystem::path &path);

        std::size_t size() const;
        std::size_t dim() const;
        Metric metric() const;

    private:
        Metric metric_;
        VectorStore vectors_;
        FlatIndex index_;

        std::vector<SearchResult> internal_to_external_list(std::vector<InternalSearchResult> internal_results) const;

        std::unordered_map<std::string, std::uint64_t> external_to_internal_;
        std::vector<std::string> internal_to_external_;
    };

}
