#include "ann_dataset.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace vectordb::benchmarks {
namespace {

constexpr std::array<char, 8> kMagic{'V', 'D', 'B', 'A', 'N', 'N', '0', '1'};
constexpr std::uint32_t kFormatVersion = 1;
constexpr std::uint32_t kAngularDistance = 1;
constexpr std::size_t kTrainingChunkRows = 16'384;

struct FileHeader {
    std::uint32_t version;
    std::uint32_t dimension;
    std::uint64_t train_count;
    std::uint64_t query_count;
    std::uint32_t neighbors_per_query;
    std::uint32_t distance;
};

static_assert(std::is_trivially_copyable_v<FileHeader>);
static_assert(sizeof(FileHeader) == 32);

std::size_t checked_size(std::uint64_t left, std::uint64_t right,
                         const char *description) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::runtime_error(std::string(description) +
                                 " is too large for this platform");
    }
    return static_cast<std::size_t>(left * right);
}

template <typename T>
void read_exact(std::ifstream &input, T *destination, std::size_t count,
                const char *description) {
    const std::size_t bytes = checked_size(count, sizeof(T), description);
    if (bytes >
        static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max())) {
        throw std::runtime_error(std::string(description) +
                                 " exceeds the stream size limit");
    }

    input.read(reinterpret_cast<char *>(destination),
               static_cast<std::streamsize>(bytes));
    if (!input) {
        throw std::runtime_error(std::string("Could not read ") + description);
    }
}

}  // namespace

AnnDataset::AnnDataset(std::size_t dimension) : vectors_(dimension) {}

AnnDataset AnnDataset::load(const std::filesystem::path &path) {
    if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error(
            "Prepared ANN datasets currently require a little-endian host");
    }

    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Could not open prepared ANN dataset: " +
                                 path.string());
    }

    std::array<char, kMagic.size()> magic{};
    read_exact(input, magic.data(), magic.size(), "dataset magic");
    if (magic != kMagic) {
        throw std::runtime_error(
            "Prepared ANN dataset has an invalid file signature");
    }

    FileHeader header{};
    read_exact(input, &header, 1, "dataset header");
    if (header.version != kFormatVersion) {
        throw std::runtime_error("Unsupported prepared ANN dataset version: " +
                                 std::to_string(header.version));
    }
    if (header.dimension == 0 || header.train_count == 0 ||
        header.query_count == 0 || header.neighbors_per_query == 0) {
        throw std::runtime_error(
            "Prepared ANN dataset contains an empty dimension or array");
    }
    if (header.neighbors_per_query > header.train_count) {
        throw std::runtime_error(
            "Ground-truth neighbor count exceeds training vector count");
    }
    if (header.distance != kAngularDistance) {
        throw std::runtime_error(
            "Prepared ANN dataset does not use angular distance");
    }

    AnnDataset dataset(header.dimension);
    const std::size_t dimension = header.dimension;
    const std::size_t train_count =
        checked_size(header.train_count, 1, "training vector count");

    const std::size_t chunk_rows = std::min(kTrainingChunkRows, train_count);
    std::vector<float> training_chunk(
        checked_size(chunk_rows, dimension, "training chunk"));

    for (std::size_t first_row = 0; first_row < train_count;
         first_row += chunk_rows) {
        const std::size_t rows = std::min(chunk_rows, train_count - first_row);
        read_exact(input, training_chunk.data(),
                   checked_size(rows, dimension, "training chunk"),
                   "training vectors");

        for (std::size_t row = 0; row < rows; ++row) {
            dataset.vectors_.add(std::span<const float>(
                training_chunk.data() + row * dimension, dimension));
        }
    }

    const std::size_t query_value_count =
        checked_size(header.query_count, dimension, "query vectors");
    dataset.queries_.resize(query_value_count);
    read_exact(input, dataset.queries_.data(), dataset.queries_.size(),
               "query vectors");

    dataset.neighbors_per_query_ = header.neighbors_per_query;
    const std::size_t neighbor_count =
        checked_size(header.query_count, header.neighbors_per_query,
                     "ground-truth neighbors");
    dataset.neighbors_.resize(neighbor_count);
    read_exact(input, dataset.neighbors_.data(), dataset.neighbors_.size(),
               "ground-truth neighbors");

    if (input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error(
            "Prepared ANN dataset contains unexpected trailing data");
    }

    for (const std::uint64_t neighbor : dataset.neighbors_) {
        if (neighbor >= header.train_count) {
            throw std::runtime_error(
                "Ground-truth neighbor id exceeds training vector count");
        }
    }

    return dataset;
}

const VectorStore &AnnDataset::vectors() const { return vectors_; }

std::size_t AnnDataset::dimension() const { return vectors_.dim(); }

std::size_t AnnDataset::query_count() const {
    return queries_.size() / dimension();
}

std::size_t AnnDataset::neighbors_per_query() const {
    return neighbors_per_query_;
}

std::span<const float> AnnDataset::query(std::size_t query_index) const {
    if (query_index >= query_count()) {
        throw std::out_of_range("ANN dataset query index is out of range");
    }
    return std::span<const float>(queries_.data() + query_index * dimension(),
                                  dimension());
}

std::span<const std::uint64_t> AnnDataset::neighbors(
    std::size_t query_index) const {
    if (query_index >= query_count()) {
        throw std::out_of_range("ANN dataset query index is out of range");
    }
    return std::span<const std::uint64_t>(
        neighbors_.data() + query_index * neighbors_per_query_,
        neighbors_per_query_);
}

}  // namespace vectordb::benchmarks
