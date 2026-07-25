#include "index_utils.hpp"

#include <algorithm>
#include <stdexcept>
#include <utility>

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

TopKAccumulator::TopKAccumulator(std::size_t capacity, Metric metric)
    : capacity_(capacity), prefer_higher_(higher_is_better(metric)) {
    heap_.reserve(capacity_);
}

bool TopKAccumulator::is_better(const InternalSearchResult &left,
                                const InternalSearchResult &right) const {
    return is_better_result(left, right, prefer_higher_);
}

void TopKAccumulator::consider(InternalSearchResult result) {
    if (capacity_ == 0) {
        return;
    }

    const auto better = [this](const InternalSearchResult &left,
                               const InternalSearchResult &right) {
        return is_better(left, right);
    };

    if (heap_.size() < capacity_) {
        heap_.push_back(result);
        std::push_heap(heap_.begin(), heap_.end(), better);
    } else if (is_better(result, heap_.front())) {
        std::pop_heap(heap_.begin(), heap_.end(), better);
        heap_.back() = result;
        std::push_heap(heap_.begin(), heap_.end(), better);
    }
}

std::vector<InternalSearchResult> TopKAccumulator::finish() {
    const auto better = [this](const InternalSearchResult &left,
                               const InternalSearchResult &right) {
        return is_better(left, right);
    };

    std::sort(heap_.begin(), heap_.end(), better);
    return std::move(heap_);
}

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

}  // namespace vectordb::index_detail
