#include "vectordb/indexes/flat_index.hpp"

#include <algorithm>
#include <stdexcept>

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
    if (top_k == 0 || vectors_.size() == 0) {
        return {};
    }

    index_detail::TopKAccumulator results(std::min(top_k, vectors_.size()),
                                          metric_);

    for (std::uint64_t internal_id = 0; internal_id < vectors_.size();
         ++internal_id) {
        const float *candidate_vector = vectors_.get(internal_id);
        results.consider({
            internal_id,
            index_detail::score_vector(metric_, query.data(), candidate_vector,
                                       vectors_.dim()),
        });
    }

    return results.finish();
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
