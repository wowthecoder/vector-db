#include "vectordb/collection.hpp"

#include <stdexcept>
#include <utility>

#include "vectordb/indexes/flat_index.hpp"

namespace vectordb {
namespace {

std::unique_ptr<Index> make_index(const VectorStore &vectors, Metric metric,
                                  const CollectionOptions &options) {
    switch (options.index_kind) {
        case IndexKind::Flat:
            return std::make_unique<FlatIndex>(vectors, metric);
        case IndexKind::RandomProjectionLsh:
            if (metric != Metric::Cosine) {
                throw std::invalid_argument(
                    "Random-projection LSH currently supports cosine only");
            }
            return std::make_unique<RandomProjectionLshIndex>(vectors, metric,
                                                              options.lsh);
    }

    throw std::invalid_argument("Unsupported index kind");
}

}  // namespace

Collection::Collection(std::size_t dim, Metric metric)
    : Collection(dim, metric, CollectionOptions{}) {}

Collection::Collection(std::size_t dim, Metric metric,
                       CollectionOptions options)
    : metric_(metric),
      vectors_(dim),
      options_(std::move(options)),
      index_(make_index(vectors_, metric_, options_)) {
    index_->build();
}

void Collection::insert(const std::string &external_id,
                        std::span<const float> vector) {
    if (external_id.empty()) {
        throw std::invalid_argument("External id must not be empty");
    }

    if (external_to_internal_.contains(external_id)) {
        throw std::invalid_argument("External id already exists");
    }

    const std::uint64_t internal_id = vectors_.add(vector);
    external_to_internal_.emplace(external_id, internal_id);
    internal_to_external_.push_back(external_id);
    index_->add(internal_id);
}

std::vector<SearchResult> Collection::search(std::span<const float> query,
                                             std::size_t top_k) const {
    if (query.size() != dim()) {
        throw std::invalid_argument(
            "Query dimension does not match collection dimension");
    }

    const auto internal_results = index_->search(query, top_k);

    return internal_to_external_list(internal_results);
}

std::vector<std::vector<SearchResult>> Collection::batch_search(
    std::span<const float> queries, std::size_t top_k) const {
    if (queries.size() % dim() != 0) {
        throw std::invalid_argument(
            "Query list should be divisible by vector dimension");
    }
    if (queries.empty()) {
        return {};
    }

    const std::size_t query_count = queries.size() / dim();
    std::vector<std::vector<SearchResult>> results;
    results.reserve(query_count);

    for (std::size_t i = 0; i < query_count; ++i) {
        const std::span<const float> query(queries.data() + (i * dim()), dim());
        results.push_back(
            internal_to_external_list(index_->search(query, top_k)));
    }

    return results;
}

std::vector<SearchResult> Collection::internal_to_external_list(
    const std::vector<InternalSearchResult> &internal_results) const {
    std::vector<SearchResult> results;
    results.reserve(internal_results.size());

    for (const auto &result : internal_results) {
        results.push_back({
            internal_to_external_.at(
                static_cast<std::size_t>(result.internal_id)),
            result.internal_id,
            result.score,
        });
    }

    return results;
}

std::size_t Collection::size() const { return vectors_.size(); }

std::size_t Collection::dim() const { return vectors_.dim(); }

Metric Collection::metric() const { return metric_; }

IndexKind Collection::index_kind() const { return options_.index_kind; }

const RandomProjectionLshConfig &Collection::lsh_config() const {
    return options_.lsh;
}

}  // namespace vectordb
