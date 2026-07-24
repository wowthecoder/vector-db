#include "vectordb/indexes/flat_index.hpp"

#include <stdexcept>
#include <utility>

#include "index_utils.hpp"

namespace vectordb {

FlatIndex::FlatIndex(const VectorStore &vectors, Metric metric)
    : vectors_(vectors), metric_(metric) {}

void FlatIndex::build() {}

void FlatIndex::add(std::uint64_t) {}

std::vector<InternalSearchResult> FlatIndex::search(
    std::span<const float> query, std::size_t top_k) const {
    if (query.size() != vectors_.dim()) {
        throw std::invalid_argument(
            "Query dimension does not match index dimension");
    }

    std::vector<InternalSearchResult> results;
    results.reserve(vectors_.size());

    for (std::uint64_t internal_id = 0; internal_id < vectors_.size();
         ++internal_id) {
        const float *candidate = vectors_.get(internal_id);
        results.push_back({
            internal_id,
            index_detail::score_vector(metric_, query.data(), candidate,
                                       vectors_.dim()),
        });
    }

    return index_detail::select_top_k(std::move(results), top_k, metric_);
}

std::vector<std::vector<InternalSearchResult>> FlatIndex::batch_search(
    std::span<const float> queries, std::size_t top_k) const {
    if (queries.size() % vectors_.dim() != 0) {
        throw std::invalid_argument(
            "Query list should be divisible by vector dimension");
    }
    if (queries.empty()) {
        return {};
    }

    std::size_t query_count = queries.size() / vectors_.dim();
    std::vector<std::vector<InternalSearchResult>> results;
    results.reserve(query_count);

    for (std::size_t i = 0; i < query_count; ++i) {
        std::span<const float> query(queries.data() + (i * vectors_.dim()),
                                     vectors_.dim());
        results.push_back(search(query, top_k));
    }

    return results;
}

}  // namespace vectordb
