#pragma once

#include <cstdint>
#include <string>

namespace vectordb {

enum class Metric { L2, Dot, Cosine };

struct InternalSearchResult {
    std::uint64_t internal_id;
    float score;
};

struct SearchResult {
    std::string external_id;
    std::uint64_t internal_id;
    float score;
};

}  // namespace vectordb
