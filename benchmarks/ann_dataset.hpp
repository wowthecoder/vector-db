#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "vectordb/vector_store.hpp"

namespace vectordb::benchmarks {

// In-memory view of an ANN-Benchmarks dataset prepared by
// prepare_ann_dataset.py. Training vectors are loaded directly into the
// VectorStore so the million-vector GloVe dataset is not retained twice.
class AnnDataset {
   public:
    static AnnDataset load(const std::filesystem::path &path);

    const VectorStore &vectors() const;
    std::size_t dimension() const;
    std::size_t query_count() const;
    std::size_t neighbors_per_query() const;

    std::span<const float> query(std::size_t query_index) const;
    std::span<const std::uint64_t> neighbors(std::size_t query_index) const;

   private:
    explicit AnnDataset(std::size_t dimension);

    VectorStore vectors_;
    std::vector<float> queries_;
    std::vector<std::uint64_t> neighbors_;
    std::size_t neighbors_per_query_ = 0;
};

}  // namespace vectordb::benchmarks
