#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

#include "vectordb/types.hpp"

namespace vectordb {

enum class IndexKind {
    Flat,
    RandomProjectionLsh,
};

class Index {
   public:
    virtual ~Index() = default;

    virtual void build() = 0;
    virtual void add(std::uint64_t internal_id) = 0;
    virtual std::vector<InternalSearchResult> search(
        std::span<const float> query, std::size_t top_k) const = 0;
};

}  // namespace vectordb
