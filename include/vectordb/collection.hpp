#pragma once

#include "vectordb/indexes/flat_index.hpp"
#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

#include <cstddef>
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

        void insert(const std::string &external_id, std::span<const float> vector);

        std::vector<SearchResult> search(std::span<const float> query, std::size_t top_k) const;

        std::size_t size() const;
        std::size_t dim() const;

    private:
        VectorStore vectors_;
        FlatIndex index_;

        std::unordered_map<std::string, std::uint64_t> external_to_internal_;
        std::vector<std::string> internal_to_external_;
    };

}
