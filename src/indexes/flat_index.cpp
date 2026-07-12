#include "vectordb/indexes/flat_index.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

#include "vectordb/distance.hpp"

namespace vectordb {
namespace {

bool is_better_result(const InternalSearchResult &a,
                      const InternalSearchResult &b, bool higher_is_better) {
    if (a.score == b.score) {
        return a.internal_id < b.internal_id;
    }

    return higher_is_better ? a.score > b.score : a.score < b.score;
}

std::vector<InternalSearchResult> select_top_k(
    std::vector<InternalSearchResult> results, std::size_t top_k,
    bool higher_is_better) {
    const auto better = [higher_is_better](const InternalSearchResult &a,
                                           const InternalSearchResult &b) {
        return is_better_result(a, b, higher_is_better);
    };

    if (top_k == 0 || results.empty()) {
        return {};
    }

    std::vector<InternalSearchResult> heap;
    heap.reserve(std::min(top_k, results.size()));

    for (const auto &result : results) {
        if (heap.size() < top_k) {
            heap.push_back(result);
            std::push_heap(heap.begin(), heap.end(), better);
        } else if (better(result, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), better);
            // item to pop is moved to heap.back()
            heap.back() = result;
            std::push_heap(heap.begin(), heap.end(), better);
        }
    }
    std::sort(heap.begin(), heap.end(), better);

    return heap;
}

}  // namespace

FlatIndex::FlatIndex(const VectorStore &vectors, Metric metric)
    : vectors_(vectors), metric_(metric) {}

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
            score_vector(query.data(), candidate),
        });
    }

    return select_top_k(std::move(results), top_k, higher_is_better());
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

float FlatIndex::score_vector(const float *a, const float *b) const {
    switch (metric_) {
        case Metric::L2:
            return l2_distance(a, b, vectors_.dim());
        case Metric::Dot:
            return dot_product(a, b, vectors_.dim());
        case Metric::Cosine:
            return cosine_similarity(a, b, vectors_.dim());
    }

    throw std::invalid_argument("Unsupported metric");
}

bool FlatIndex::higher_is_better() const {
    return metric_ == Metric::Dot || metric_ == Metric::Cosine;
}

}  // namespace vectordb
