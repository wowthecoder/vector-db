#include "vectordb/indexes/random_projection_lsh_index.hpp"

#include <random>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "vectordb/distance.hpp"

namespace vectordb {

RandomProjectionLshIndex::RandomProjectionLshIndex(
    const VectorStore &vectors, Metric metric, RandomProjectionLshConfig config)
    : vectors_(vectors), metric_(metric), config_(std::move(config)) {
    if (config_.num_tables == 0) {
        throw std::invalid_argument("LSH requires at least one hash table");
    }
    if (config_.num_bits_per_table == 0 || config_.num_bits_per_table > 64) {
        throw std::invalid_argument(
            "LSH bits per table must be between 1 and 64");
    }
    if (config_.num_candidates == 0) {
        throw std::invalid_argument(
            "LSH candidate limit must be greater than zero");
    }
}

void RandomProjectionLshIndex::build() {
    generate_random_projections();

    std::vector<HashTable> new_tables(config_.num_tables);

    for (std::uint64_t internal_id = 0; internal_id < vectors_.size();
         ++internal_id) {
        const std::span<const float> vector(vectors_.get(internal_id),
                                            vectors_.dim());

        for (std::size_t table_index = 0; table_index < config_.num_tables;
             ++table_index) {
            const Signature signature = compute_signature(vector, table_index);

            // new_tables[table_index] is a hash table of <Signature, Bucket>
            // Bucket is a list(vector) of integers(internal ids)
            new_tables[table_index][signature].push_back(internal_id);
        }
    }

    tables_ = std::move(new_tables);
    is_built_ = true;
}

std::vector<InternalSearchResult> RandomProjectionLshIndex::search(
    std::span<const float> query, std::size_t top_k) const {
    if (query.size() != vectors_.dim()) {
        throw std::invalid_argument(
            "Query dimension does not match index dimension");
    }
    if (top_k == 0 || vectors_.size() == 0) {
        return {};
    }
    if (!is_built_) {
        throw std::logic_error("LSH index must be built before searching");
    }

    // TODO: Collect candidate IDs from the query's buckets, cap the candidate
    // set according to config_, then score and rank the candidates exactly.
    throw std::logic_error(
        "RandomProjectionLshIndex::search is not implemented");
}

bool RandomProjectionLshIndex::is_built() const { return is_built_; }

const RandomProjectionLshConfig &RandomProjectionLshIndex::config() const {
    return config_;
}

// TODO: Define the private implementation helpers declared in the header.
void RandomProjectionLshIndex::generate_random_projections() {
    std::mt19937_64 generator(config_.seed);
    std::normal_distribution<float> distribution(0.0f, 1.0f);
    const std::size_t size =
        config_.num_tables * config_.num_bits_per_table * vectors_.dim();
    projections_.clear();
    projections_.reserve(size);

    // idx = ((table * config_.num_bits_per_table) + bit) * vectors_.dim() +
    // dimension;
    for (std::size_t table = 0; table < config_.num_tables; ++table) {
        for (std::size_t bit = 0; bit < config_.num_bits_per_table; ++bit) {
            for (std::size_t dimension = 0; dimension < vectors_.dim();
                 ++dimension) {
                projections_.push_back(distribution(generator));
            }
        }
    }
}

RandomProjectionLshIndex::Signature RandomProjectionLshIndex::compute_signature(
    std::span<const float> vector, std::size_t table_index) const {
    Signature signature = 0;

    for (std::size_t bit = 0; bit < config_.num_bits_per_table; ++bit) {
        const std::size_t table_offset =
            table_index * config_.num_bits_per_table * vectors_.dim();
        const std::size_t projection_offset =
            table_offset + bit * vectors_.dim();
        std::span<const float> projection(
            projections_.data() + projection_offset, vectors_.dim());
        const float dot = vectordb::dot_product(
            projection.data(), vector.data(), projection.size());
        if (dot >= 0.0f) {
            signature |= (std::uint64_t{1} << bit);
        }
    }

    return signature;
}

std::vector<std::uint64_t> RandomProjectionLshIndex::collect_candidates(
    std::span<const float> query) const {
    std::unordered_set<std::uint64_t> unique_ids;
    std::vector<std::uint64_t> candidates;

    for (std::size_t table_index = 0; table_index < config_.num_tables;
         ++table_index) {
        const Signature signature = compute_signature(query, table_index);
        const HashTable &table = tables_.at(table_index);
        const auto bucket_it = table.find(signature);

        if (bucket_it == table.end()) {
            continue;
        }

        const Bucket &bucket = bucket_it->second;

        // set.insert(value).second is only true when value does NOT exist in
        // the set before insertion
        for (const std::uint64_t internal_id : bucket) {
            if (!unique_ids.insert(internal_id).second) {
                continue;
            }

            candidates.push_back(internal_id);

            if (candidates.size() == config_.num_candidates) {
                return candidates;
            }
        }
    }

    return candidates;
}

}  // namespace vectordb
