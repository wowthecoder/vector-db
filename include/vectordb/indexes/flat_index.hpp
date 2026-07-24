#pragma once

#include <cstddef>
#include <span>
#include <vector>

#include "vectordb/indexes/index.hpp"
#include "vectordb/types.hpp"
#include "vectordb/vector_store.hpp"

namespace vectordb {

class FlatIndex : public Index {
   public:
    FlatIndex(const VectorStore &vectors, Metric metric);

    void build() override;
    void add(std::uint64_t internal_id) override;
    std::vector<InternalSearchResult> search(std::span<const float> query,
                                             std::size_t top_k) const override;
    std::vector<std::vector<InternalSearchResult>> batch_search(
        std::span<const float> queries, std::size_t top_k) const;

   private:
    const VectorStore &vectors_;
    Metric metric_;
};

}  // namespace vectordb
