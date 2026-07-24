#pragma once

#include <cstddef>
#include <vector>

#include "vectordb/types.hpp"

namespace vectordb::index_detail {

float score_vector(Metric metric, const float *a, const float *b,
                   std::size_t dimension);

std::vector<InternalSearchResult> select_top_k(
    std::vector<InternalSearchResult> results, std::size_t top_k,
    Metric metric);

}  // namespace vectordb::index_detail