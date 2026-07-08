#pragma once

#include "vectordb/distance.hpp"
#include "vectordb/vector_store.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <unordered_map>
#include <vector>

namespace vectordb
{

    struct SearchResult
    {
        std::string external_id;
        std::uint64_t internal_id;
        float score;
    };

    class Collection
    {
    public:
        Collection(std::size_t dim, Metric metric);

        void insert(const std::string &external_id, std::span<const float> vector);

        std::vector<SearchResult> search(std::span<const float> query, std::size_t top_k) const;

        std::size_t size() const;
        std::size_t dim() const;

    private:
        float score_vector(const float *a, const float *b) const;
        bool higher_is_better() const;

        Metric metric_;
        VectorStore vectors_;

        std::unordered_map<std::string, std::uint64_t> external_to_internal_;
        std::vector<std::string> internal_to_external_;
    };

}