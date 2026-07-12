#include "benchmark_utils.hpp"

#include <stdexcept>

namespace vectordb::benchmarks
{

    std::vector<float> make_vector(std::size_t dimension, std::uint32_t seed)
    {
        (void)dimension;
        (void)seed;
        throw std::logic_error("TODO: implement benchmark vector generation");
    }

    std::unique_ptr<Collection> make_collection(
        std::size_t vector_count,
        std::size_t dimension,
        Metric metric,
        std::uint32_t seed)
    {
        (void)vector_count;
        (void)dimension;
        (void)metric;
        (void)seed;
        throw std::logic_error("TODO: implement benchmark collection generation");
    }

    std::filesystem::path make_temporary_path()
    {
        throw std::logic_error("TODO: implement benchmark temporary paths");
    }

}

