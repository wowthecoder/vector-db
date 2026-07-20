#include "benchmark_utils.hpp"

#include <atomic>
#include <chrono>
#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>

namespace vectordb::benchmarks {

std::vector<float> make_vector(std::size_t dimension, std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::vector<float> values(dimension);

    for (float &value : values) {
        value = distribution(generator);
    }

    return values;
}

std::unique_ptr<Collection> make_collection(std::size_t vector_count,
                                            std::size_t dimension,
                                            Metric metric, std::uint32_t seed) {
    auto collection = std::make_unique<Collection>(dimension, metric);

    for (size_t i = 0; i < vector_count; ++i) {
        auto vector = make_vector(dimension, seed + i);
        collection->insert("vector_" + std::to_string(i), vector);
    }

    return collection;
}

std::filesystem::path make_temporary_path() {
    static std::atomic_uint64_t sequence{0};

    const auto timestamp =
        std::chrono::steady_clock::now().time_since_epoch().count();

    const std::string filename = "vectordb_benchmark_" +
                                 std::to_string(timestamp) + "_" +
                                 std::to_string(sequence++) + ".bin";

    return std::filesystem::temp_directory_path() / filename;
}

}  // namespace vectordb::benchmarks
