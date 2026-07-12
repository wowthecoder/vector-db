#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace vectordb {

class VectorStore {
   public:
    explicit VectorStore(std::size_t dim);

    std::uint64_t add(std::span<const float> vector);

    const float *get(std::uint64_t internal_id) const;

    std::size_t size() const;
    std::size_t dim() const;

   private:
    std::size_t dim_;
    std::vector<float> data_;
};

}  // namespace vectordb