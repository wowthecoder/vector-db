#include "vectordb/indexes/inverted_file_flat_index.hpp"

#include <stdexcept>
#include <utility>

namespace vectordb {

InvertedFileFlatIndex::InvertedFileFlatIndex(const VectorStore &vectors,
                                             Metric metric,
                                             InvertedFileFlatConfig config)
    : vectors_(vectors), metric_(metric), config_(std::move(config)) {
    if (metric_ != Metric::L2) {
        throw std::invalid_argument(
            "IVF-Flat currently supports L2 distance only");
    }
    if (config_.num_lists == 0) {
        throw std::invalid_argument(
            "IVF-Flat requires at least one posting list");
    }
    if (config_.num_probes == 0 || config_.num_probes > config_.num_lists) {
        throw std::invalid_argument(
            "IVF-Flat probe count must be between 1 and the list count");
    }
    if (config_.max_iterations == 0) {
        throw std::invalid_argument(
            "IVF-Flat requires at least one training iteration");
    }
}

void InvertedFileFlatIndex::build() {
    // TODO(IVF-6): Train into local state, build local posting lists, then
    // commit both together. Follow the exception-safety notes in the guide.
    throw std::logic_error(
        "IVF-Flat build is not implemented; see docs/IVF_FLAT_TODO.md");
}

void InvertedFileFlatIndex::add(std::uint64_t) {
    // TODO(IVF-7): Validate lifecycle/order, find the nearest centroid, and
    // append the ID without partially mutating the index on failure.
    throw std::logic_error(
        "IVF-Flat add is not implemented; see docs/IVF_FLAT_TODO.md");
}

std::vector<InternalSearchResult> InvertedFileFlatIndex::search(
    std::span<const float>, std::size_t) const {
    // TODO(IVF-8): Validate, select lists, and exactly score their candidates
    // with index_detail::TopKAccumulator.
    throw std::logic_error(
        "IVF-Flat search is not implemented; see docs/IVF_FLAT_TODO.md");
}

bool InvertedFileFlatIndex::is_built() const { return is_built_; }

const InvertedFileFlatConfig &InvertedFileFlatIndex::config() const {
    return config_;
}

std::vector<float> InvertedFileFlatIndex::initialize_centroids() const {
    // TODO(IVF-1): Implement this milestone.
    throw std::logic_error(
        "IVF-Flat centroid initialization is not implemented");
}

InvertedFileFlatIndex::ListId InvertedFileFlatIndex::find_nearest_centroid(
    std::span<const float>, std::span<const float>) const {
    // TODO(IVF-2): Implement this milestone.
    throw std::logic_error("IVF-Flat centroid lookup is not implemented");
}

std::vector<float> InvertedFileFlatIndex::train_centroids() const {
    // TODO(IVF-3): Implement this milestone.
    throw std::logic_error("IVF-Flat centroid training is not implemented");
}

std::vector<InvertedFileFlatIndex::PostingList>
InvertedFileFlatIndex::build_posting_lists(std::span<const float>) const {
    // TODO(IVF-4): Implement this milestone.
    throw std::logic_error("IVF-Flat posting-list build is not implemented");
}

std::vector<InvertedFileFlatIndex::ListId> InvertedFileFlatIndex::select_lists(
    std::span<const float>) const {
    // TODO(IVF-5): Implement this milestone.
    throw std::logic_error("IVF-Flat list selection is not implemented");
}

}  // namespace vectordb
