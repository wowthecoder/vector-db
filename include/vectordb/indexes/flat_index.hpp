#pragma once

#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace vectordb
{

    class FlatIndex
    {
    public:
        FlatIndex(const VectorStore &vectors, Metric metric);

        std::vector<InternalSearchResult> search(std::span<const float> query, std::size_t top_k) const;
        std::vector<std::vector<InternalSearchResult>> batch_search(
            std::span<const float> queries,
            std::size_t top_k) const;

    private:
        float score_vector(const float *a, const float *b) const;
        bool higher_is_better() const;

        const VectorStore &vectors_;
        Metric metric_;
    };

}
