#include "benchmark_utils.hpp"

#include <cstddef>
#include <random>
#include <stdexcept>
#include <string>

namespace vectordb::benchmarks {
// class TemporaryFileCleanup
// {
// public:
//     explicit TemporaryFileCleanup(std::filesystem::path path) :
//     path_(std::move(path)){

//     }

// private:
//     std::filesystem::path path_
// }

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
    auto vector = make_vector(dimension, seed);
    collection->insert("vector_" + std::to_string(i), vector);
  }

  return collection;
}

std::filesystem::path make_temporary_path() {
  throw std::logic_error("TODO: implement benchmark temporary paths");
}

}  // namespace vectordb::benchmarks
