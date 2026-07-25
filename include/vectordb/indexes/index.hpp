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

    // Synchronizes the index with all vectors currently in its store. Derived
    // indexes may require a successful build before additions or searches. If
    // rebuilding throws, the previous search-visible state must remain intact.
    virtual void build() = 0;

    // Adds the next vector after it has been appended to the store. Vectors
    // must be added exactly once in internal-ID order. If insertion throws,
    // the previous search-visible index state must remain unchanged.
    virtual void add(std::uint64_t internal_id) = 0;

    virtual std::vector<InternalSearchResult> search(
        std::span<const float> query, std::size_t top_k) const = 0;
};

}  // namespace vectordb
