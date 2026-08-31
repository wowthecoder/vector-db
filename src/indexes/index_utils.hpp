#pragma once

#include <cstddef>
#include <vector>

#include "vectordb/types.hpp"

namespace vectordb::index_detail {

class TopKAccumulator {
   public:
    TopKAccumulator(std::size_t capacity, Metric metric);

    void consider(InternalSearchResult result);
    std::vector<InternalSearchResult> finish();

   private:
    bool is_better(const InternalSearchResult &left,
                   const InternalSearchResult &right) const;

    std::size_t capacity_;
    bool prefer_higher_;
    std::vector<InternalSearchResult> heap_;
};

float score_vector(Metric metric, const float *a, const float *b,
                   std::size_t dimension);

// True when a larger score is closer for the given metric (Dot, Cosine).
// False when a smaller score is closer (L2). Shared so every index orients
// its heaps and comparators the same way as TopKAccumulator.
bool higher_is_better(Metric metric);

}  // namespace vectordb::index_detail
