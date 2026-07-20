#include <array>
#include <cstdint>
#include <fstream>
#include <istream>
#include <ostream>
#include <stdexcept>
#include <string>

#include "vectordb/collection.hpp"

namespace {
constexpr std::array<char, 8> k_magic_bytes{'V', 'D', 'B', 'C',
                                            'O', 'L', 'L', '\0'};
constexpr std::uint32_t k_version = 1;
// Format safety limits checked before allocating data from an input file.
constexpr std::uint64_t k_max_dimension = 65536;
constexpr std::uint64_t k_max_vector_count = 10'000'000;
constexpr std::uint32_t k_max_external_id_length = 65536;

/*
Format v1, little-endian:
magic bytes: 8 bytes
version: uint32
metric: uint32 (0 for L2, 1 for Dot, 2 for Cosine)
dimension: uint64
vector count: uint64
for every vector, write the following:
    external id byte length: uint32
    external id bytes
    vector values: dimension float32 values
*/

/*
write_u32/write_u64/read_u32/read_u64
-> one fixed-size numeric value
write_exact/read_exact
-> a block of N bytes
*/

void read_exact(std::istream &in, char *data, std::size_t size) {
    in.read(data, static_cast<std::streamsize>(size));
    if (!in) {
        throw std::runtime_error("Unexpected end of collection file");
    }
}

void write_exact(std::ostream &out, const char *data, std::size_t size) {
    out.write(data, static_cast<std::streamsize>(size));
    if (!out) {
        throw std::runtime_error("Failed to write collection file");
    }
}

void write_u32(std::ostream &out, std::uint32_t value) {
    std::array<char, 4> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (i * 8)) & 0xff);
    }

    write_exact(out, bytes.data(), bytes.size());
}

void write_u64(std::ostream &out, std::uint64_t value) {
    std::array<char, 8> bytes{};
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        bytes[i] = static_cast<char>((value >> (i * 8)) & 0xff);
    }

    write_exact(out, bytes.data(), bytes.size());
}

std::uint32_t read_u32(std::istream &in) {
    std::array<char, 4> bytes{};
    read_exact(in, bytes.data(), bytes.size());

    std::uint32_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |=
            static_cast<std::uint32_t>(static_cast<unsigned char>(bytes[i]))
            << (i * 8);
    }

    return value;
}

std::uint64_t read_u64(std::istream &in) {
    std::array<char, 8> bytes{};
    read_exact(in, bytes.data(), bytes.size());

    std::uint64_t value = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        value |=
            static_cast<std::uint64_t>(static_cast<unsigned char>(bytes[i]))
            << (i * 8);
    }

    return value;
}
}  // namespace

namespace vectordb {
void Collection::save(const std::filesystem::path &path) const {
    if (dim() > k_max_dimension) {
        throw std::runtime_error(
            "Vector dimension exceeds maximum supported dimension of " +
            std::to_string(k_max_dimension));
    }
    if (size() > k_max_vector_count) {
        throw std::runtime_error(
            "Vector count exceeds maximum supported count of " +
            std::to_string(k_max_vector_count));
    }
    for (const auto &external_id : internal_to_external_) {
        if (external_id.size() > k_max_external_id_length) {
            throw std::runtime_error(
                "External ID length exceeds maximum supported length of " +
                std::to_string(k_max_external_id_length));
        }
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    write_exact(out, k_magic_bytes.data(), 8 * sizeof(char));
    write_u32(out, k_version);
    std::uint32_t metric_code;

    switch (metric_) {
        case Metric::L2:
            metric_code = 0;
            break;
        case Metric::Dot:
            metric_code = 1;
            break;
        case Metric::Cosine:
            metric_code = 2;
            break;
        default:
            throw std::runtime_error("Unsupported collection metric");
    }

    write_u32(out, metric_code);
    write_u64(out, static_cast<std::uint64_t>(dim()));
    write_u64(out, static_cast<std::uint64_t>(size()));

    for (std::uint64_t internal_id = 0; internal_id < size(); ++internal_id) {
        const std::string &external_id =
            internal_to_external_.at(static_cast<std::size_t>(internal_id));

        const auto id_length = static_cast<std::uint32_t>(external_id.size());

        write_u32(out, id_length);
        write_exact(out, external_id.data(), external_id.size());

        const float *vector = vectors_.get(internal_id);

        write_exact(out, reinterpret_cast<const char *>(vector),
                    dim() * sizeof(float));
    }
}

std::unique_ptr<Collection> Collection::load(
    const std::filesystem::path &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + path.string());
    }

    std::array<char, k_magic_bytes.size()> magic{};
    read_exact(in, magic.data(), magic.size());

    if (magic != k_magic_bytes) {
        throw std::runtime_error("Invalid collection file magic bytes");
    }

    const std::uint32_t version = read_u32(in);

    if (version != k_version) {
        throw std::runtime_error("Unsupported collection file version: " +
                                 std::to_string(version));
    }

    const std::uint32_t metric_code = read_u32(in);
    Metric metric;

    switch (metric_code) {
        case 0:
            metric = Metric::L2;
            break;
        case 1:
            metric = Metric::Dot;
            break;
        case 2:
            metric = Metric::Cosine;
            break;
        default:
            throw std::runtime_error(
                "Invalid Metric code, should be either 0, 1 or 2");
    }

    const std::uint64_t dimension = read_u64(in);

    if (dimension == 0) {
        throw std::runtime_error("Vector dimension cannot be zero");
    }
    if (dimension > k_max_dimension) {
        throw std::runtime_error(
            "Vector dimension exceeds maximum supported dimension of " +
            std::to_string(k_max_dimension));
    }

    const std::uint64_t count = read_u64(in);
    if (count > k_max_vector_count) {
        throw std::runtime_error(
            "Vector count exceeds maximum supported count of " +
            std::to_string(k_max_vector_count));
    }

    auto collection = std::make_unique<Collection>(dimension, metric);

    for (std::uint64_t internal_id = 0; internal_id < count; ++internal_id) {
        const std::uint32_t id_length = read_u32(in);

        if (id_length == 0) {
            throw std::runtime_error("External ID length cannot be zero");
        }
        if (id_length > k_max_external_id_length) {
            throw std::runtime_error(
                "External ID length exceeds maximum supported length of " +
                std::to_string(k_max_external_id_length));
        }

        std::string external_id(id_length, '\0');
        read_exact(in, external_id.data(), id_length);

        std::vector<float> values(dimension);
        read_exact(in, reinterpret_cast<char *>(values.data()),
                   dimension * sizeof(float));

        collection->insert(external_id, values);
    }

    return collection;
}

}  // namespace vectordb
