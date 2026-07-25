#include "vectordb/indexes/random_projection_lsh_index.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

#include "index_utils.hpp"
#include "vectordb/distance.hpp"

namespace vectordb {
namespace {

std::size_t checked_multiply(std::size_t left, std::size_t right,
                             const char *description) {
    if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
        throw std::length_error(std::string(description) + " is too large");
    }
    return left * right;
}

float inverse_norm(std::span<const float> input_vector) {
    const float squared_norm = dot_product(
        input_vector.data(), input_vector.data(), input_vector.size());
    if (squared_norm == 0.0f) {
        throw std::invalid_argument(
            "Random-projection LSH does not support zero vectors");
    }
    return 1.0f / std::sqrt(squared_norm);
}

float deterministic_normal_approximation(std::mt19937_64 &generator) {
    // The sum of 12 U[0, 1] samples minus 6 approximates N(0, 1).
    // Using fixed high-order engine bits and power-of-two scaling avoids the
    // implementation-defined sequence of std::normal_distribution.
    constexpr std::uint64_t sample_mask = (std::uint64_t{1} << 24) - 1;
    constexpr std::size_t sample_count = 12;
    constexpr double midpoint =
        static_cast<double>(sample_count * sample_mask) / 2.0;
    constexpr double scale = 1.0 / static_cast<double>(std::uint64_t{1} << 24);

    std::uint64_t sum = 0;
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
        sum += (generator() >> 40) & sample_mask;
    }
    return static_cast<float>((static_cast<double>(sum) - midpoint) * scale);
}

std::size_t estimated_signature_count(std::size_t vector_count,
                                      std::size_t bits_per_table) {
    if (vector_count == 0) {
        return 0;
    }

    constexpr std::size_t size_t_bits =
        std::numeric_limits<std::size_t>::digits;
    if (bits_per_table >= size_t_bits) {
        return vector_count;
    }

    return std::min(vector_count, std::size_t{1} << bits_per_table);
}

}  // namespace

RandomProjectionLshIndex::RandomProjectionLshIndex(
    const VectorStore &vectors, Metric metric, RandomProjectionLshConfig config)
    : vectors_(vectors), metric_(metric), config_(std::move(config)) {
    if (metric_ != Metric::Cosine) {
        throw std::invalid_argument(
            "Random-projection LSH supports cosine similarity only");
    }
    if (config_.num_tables == 0) {
        throw std::invalid_argument("LSH requires at least one hash table");
    }
    if (config_.num_tables > RandomProjectionLshConfig::max_num_tables) {
        throw std::invalid_argument(
            "LSH table count exceeds maximum supported value of " +
            std::to_string(RandomProjectionLshConfig::max_num_tables));
    }
    if (config_.num_bits_per_table == 0 || config_.num_bits_per_table > 64) {
        throw std::invalid_argument(
            "LSH bits per table must be between 1 and 64");
    }
    if (config_.num_candidates == 0) {
        throw std::invalid_argument(
            "LSH candidate limit must be greater than zero");
    }

    static_cast<void>(projection_count());
}

void RandomProjectionLshIndex::build() {
    auto new_projections = generate_random_projections();
    std::vector<HashTable> new_tables(config_.num_tables);
    std::vector<float> new_inverse_vector_norms;
    new_inverse_vector_norms.reserve(vectors_.size());

    const std::size_t signature_count =
        estimated_signature_count(vectors_.size(), config_.num_bits_per_table);
    const std::size_t expected_bucket_size =
        signature_count == 0
            ? 0
            : vectors_.size() / signature_count +
                  (vectors_.size() % signature_count != 0 ? 1 : 0);

    for (HashTable &table : new_tables) {
        table.reserve(signature_count);
    }

    for (std::uint64_t internal_id = 0; internal_id < vectors_.size();
         ++internal_id) {
        const std::span<const float> stored_vector(vectors_.get(internal_id),
                                                   vectors_.dim());
        new_inverse_vector_norms.push_back(inverse_norm(stored_vector));

        for (std::size_t table_index = 0; table_index < config_.num_tables;
             ++table_index) {
            const Signature signature =
                compute_signature(stored_vector, table_index, new_projections);
            auto [bucket_it, inserted] =
                new_tables[table_index].try_emplace(signature);
            if (inserted) {
                bucket_it->second.reserve(expected_bucket_size);
            }
            bucket_it->second.push_back(internal_id);
        }
    }

    projections_ = std::move(new_projections);
    tables_ = std::move(new_tables);
    inverse_vector_norms_ = std::move(new_inverse_vector_norms);
    indexed_vector_count_ = vectors_.size();
    is_built_ = true;
}

void RandomProjectionLshIndex::add(std::uint64_t internal_id) {
    if (!is_built_) {
        throw std::logic_error("LSH index must be built before insertion");
    }

    if (internal_id != indexed_vector_count_) {
        throw std::logic_error(
            "LSH vectors must be added once in internal ID order");
    }

    const std::span<const float> stored_vector(vectors_.get(internal_id),
                                               vectors_.dim());
    const float stored_vector_inverse_norm = inverse_norm(stored_vector);

    std::vector<Signature> signatures;
    signatures.reserve(config_.num_tables);

    for (std::size_t table_index = 0; table_index < config_.num_tables;
         ++table_index) {
        signatures.push_back(
            compute_signature(stored_vector, table_index, projections_));
    }

    std::vector<Bucket *> buckets;
    buckets.reserve(config_.num_tables);
    if (inverse_vector_norms_.size() == inverse_vector_norms_.max_size()) {
        throw std::length_error("LSH norm cache exceeds maximum size");
    }
    inverse_vector_norms_.reserve(inverse_vector_norms_.size() + 1);

    for (std::size_t table_index = 0; table_index < config_.num_tables;
         ++table_index) {
        auto [bucket_it, inserted] =
            tables_.at(table_index).try_emplace(signatures[table_index]);
        static_cast<void>(inserted);

        Bucket &bucket = bucket_it->second;
        if (bucket.size() == bucket.max_size()) {
            throw std::length_error("LSH bucket exceeds maximum size");
        }
        bucket.reserve(bucket.size() + 1);
        buckets.push_back(&bucket);
    }

    inverse_vector_norms_.push_back(stored_vector_inverse_norm);
    for (Bucket *bucket : buckets) {
        bucket->push_back(internal_id);
    }
    ++indexed_vector_count_;
}

std::vector<InternalSearchResult> RandomProjectionLshIndex::search(
    std::span<const float> query, std::size_t top_k) const {
    if (query.size() != vectors_.dim()) {
        throw std::invalid_argument(
            "Query dimension does not match index dimension");
    }
    if (!is_built_) {
        throw std::logic_error("LSH index must be built before searching");
    }
    if (indexed_vector_count_ != vectors_.size()) {
        throw std::logic_error(
            "LSH index is stale and must be rebuilt before searching");
    }

    const float query_inverse_norm = inverse_norm(query);
    if (top_k == 0 || vectors_.size() == 0) {
        return {};
    }

    const std::vector<std::uint64_t> candidate_ids = collect_candidates(query);
    index_detail::TopKAccumulator results(std::min(top_k, candidate_ids.size()),
                                          metric_);

    for (const std::uint64_t internal_id : candidate_ids) {
        const float *candidate_vector = vectors_.get(internal_id);
        const float score =
            dot_product(query.data(), candidate_vector, vectors_.dim()) *
            query_inverse_norm *
            inverse_vector_norms_.at(static_cast<std::size_t>(internal_id));
        results.consider({
            internal_id,
            score,
        });
    }

    return results.finish();
}

bool RandomProjectionLshIndex::is_built() const { return is_built_; }

const RandomProjectionLshConfig &RandomProjectionLshIndex::config() const {
    return config_;
}

std::vector<float> RandomProjectionLshIndex::generate_random_projections()
    const {
    std::mt19937_64 generator(config_.seed);
    std::vector<float> projections;
    projections.reserve(projection_count());

    // idx = ((table * config_.num_bits_per_table) + bit) * vectors_.dim() +
    // dimension;
    for (std::size_t table = 0; table < config_.num_tables; ++table) {
        for (std::size_t bit = 0; bit < config_.num_bits_per_table; ++bit) {
            for (std::size_t dimension = 0; dimension < vectors_.dim();
                 ++dimension) {
                projections.push_back(
                    deterministic_normal_approximation(generator));
            }
        }
    }

    return projections;
}

std::size_t RandomProjectionLshIndex::projection_count() const {
    const std::size_t projection_vectors = checked_multiply(
        config_.num_tables, config_.num_bits_per_table, "LSH projection count");
    return checked_multiply(projection_vectors, vectors_.dim(),
                            "LSH projection count");
}

RandomProjectionLshIndex::Signature RandomProjectionLshIndex::compute_signature(
    std::span<const float> input_vector, std::size_t table_index,
    std::span<const float> projections) const {
    Signature signature = 0;
    const std::size_t table_offset =
        table_index * config_.num_bits_per_table * vectors_.dim();

    for (std::size_t bit = 0; bit < config_.num_bits_per_table; ++bit) {
        const std::size_t projection_offset =
            table_offset + bit * vectors_.dim();
        std::span<const float> projection(
            projections.data() + projection_offset, vectors_.dim());
        const float dot = dot_product(projection.data(), input_vector.data(),
                                      projection.size());
        if (dot >= 0.0f) {
            signature |= (std::uint64_t{1} << bit);
        }
    }

    return signature;
}

std::vector<std::uint64_t> RandomProjectionLshIndex::collect_candidates(
    std::span<const float> query) const {
    std::unordered_map<std::uint64_t, std::size_t> collision_counts;
    collision_counts.reserve(std::min(config_.num_candidates, vectors_.size()));

    // Inspect every matching bucket in every table.
    for (std::size_t table_index = 0; table_index < config_.num_tables;
         ++table_index) {
        const Signature signature =
            compute_signature(query, table_index, projections_);
        const HashTable &table = tables_.at(table_index);
        const auto bucket_it = table.find(signature);

        if (bucket_it == table.end()) {
            continue;
        }

        const Bucket &bucket = bucket_it->second;

        for (const std::uint64_t internal_id : bucket) {
            ++collision_counts[internal_id];
        }
    }

    // Store each unique ID together with its collision count.
    using RankedCandidate = std::pair<std::uint64_t, std::size_t>;
    std::vector<RankedCandidate> ranked_candidates;
    ranked_candidates.reserve(collision_counts.size());

    for (const auto &[internal_id, collision_count] : collision_counts) {
        ranked_candidates.emplace_back(internal_id, collision_count);
    }

    // More collisions are better. For equal counts, prefer the lower ID so
    // candidate selection remains deterministic.
    const auto better_candidate = [](const RankedCandidate &left,
                                     const RankedCandidate &right) {
        if (left.second != right.second) {
            return left.second > right.second;
        }

        return left.first < right.first;
    };

    const std::size_t candidate_count =
        std::min(config_.num_candidates, ranked_candidates.size());

    std::partial_sort(ranked_candidates.begin(),
                      ranked_candidates.begin() + candidate_count,
                      ranked_candidates.end(), better_candidate);

    // remove trailing elements after candidate_count
    ranked_candidates.resize(candidate_count);

    std::vector<std::uint64_t> candidates;
    candidates.reserve(candidate_count);

    for (const auto &ranked_candidate : ranked_candidates) {
        candidates.push_back(ranked_candidate.first);
    }

    return candidates;
}

}  // namespace vectordb
