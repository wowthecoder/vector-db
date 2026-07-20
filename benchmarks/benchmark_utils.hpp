#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

#include "vectordb/collection.hpp"

namespace vectordb::benchmarks {

class TemporaryFileCleanup {
   public:
    explicit TemporaryFileCleanup(std::filesystem::path path)
        : path_(std::move(path)) {}

    ~TemporaryFileCleanup() noexcept {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    TemporaryFileCleanup(const TemporaryFileCleanup &) = delete;
    TemporaryFileCleanup &operator=(const TemporaryFileCleanup &) = delete;

   private:
    std::filesystem::path path_;
};

std::vector<float> make_vector(std::size_t dimension, std::uint32_t seed);

std::unique_ptr<Collection> make_collection(std::size_t vector_count,
                                            std::size_t dimension,
                                            Metric metric, std::uint32_t seed);

std::filesystem::path make_temporary_path();

}  // namespace vectordb::benchmarks
