#include "index_utils.hpp"

#include <algorithm>
#include <stdexcept>

#include "vectordb/distance.hpp"

namespace vectordb::index_detail {
namespace {

bool higher_is_better(Metric metric) {
    return metric == Metric::Dot || metric == Metric::Cosine;
}

bool is_better_result(const InternalSearchResult &a,
                      const InternalSearchResult &b, bool higher_is_better) {
    if (a.score == b.score) {
        return a.internal_id < b.internal_id;
    }

    return higher_is_better ? a.score > b.score : a.score < b.score;
}

}  // namespace

float score_vector(Metric metric, const float *a, const float *b,
                   std::size_t dimension) {
    switch (metric) {
        case Metric::L2:
            return l2_distance(a, b, dimension);
        case Metric::Dot:
            return dot_product(a, b, dimension);
        case Metric::Cosine:
            return cosine_similarity(a, b, dimension);
    }

    throw std::invalid_argument("Unsupported metric");
}

std::vector<InternalSearchResult> select_top_k(
    std::vector<InternalSearchResult> results, std::size_t top_k,
    Metric metric) {
    const bool prefer_higher = higher_is_better(metric);

    const auto better = [prefer_higher](const InternalSearchResult &a,
                                        const InternalSearchResult &b) {
        return is_better_result(a, b, prefer_higher);
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
            heap.back() = result;
            std::push_heap(heap.begin(), heap.end(), better);
        }
    }

    std::sort(heap.begin(), heap.end(), better);
    return heap;
}

}  // namespace vectordb::index_detail