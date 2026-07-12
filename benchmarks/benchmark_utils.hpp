#pragma once

#include "vectordb/collection.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace vectordb::benchmarks
{

    // TODO: Generate repeatable input from seed. Keep generation outside timed loops.
    std::vector<float> make_vector(std::size_t dimension, std::uint32_t seed);

    // TODO: Build a populated collection with stable, unique external IDs.
    std::unique_ptr<Collection> make_collection(
        std::size_t vector_count,
        std::size_t dimension,
        Metric metric,
        std::uint32_t seed);

    // TODO: Return a collision-resistant path in the system temporary directory.
    std::filesystem::path make_temporary_path();

}

