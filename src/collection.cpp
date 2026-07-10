#include "vectordb/collection.hpp"

#include <stdexcept>

namespace vectordb
{

    Collection::Collection(std::size_t dim, Metric metric)
        : metric_(metric),
          vectors_(dim),
          index_(vectors_, metric)
    {
    }

    void Collection::insert(const std::string &external_id, std::span<const float> vector)
    {
        if (external_id.empty())
        {
            throw std::invalid_argument("External id must not be empty");
        }

        if (external_to_internal_.contains(external_id))
        {
            throw std::invalid_argument("External id already exists");
        }

        const std::uint64_t internal_id = vectors_.add(vector);
        external_to_internal_.emplace(external_id, internal_id);
        internal_to_external_.push_back(external_id);
    }

    std::vector<SearchResult> Collection::search(std::span<const float> query, std::size_t top_k) const
    {
        if (query.size() != dim())
        {
            throw std::invalid_argument("Query dimension does not match collection dimension");
        }

        const auto internal_results = index_.search(query, top_k);

        return internal_to_external_list(internal_results);
    }

    std::vector<std::vector<SearchResult>> Collection::batch_search(
        std::span<const float> queries,
        std::size_t top_k) const
    {
        const auto internal_results = index_.batch_search(queries, top_k);
        std::vector<std::vector<SearchResult>> results;
        results.reserve(internal_results.size());
        for (const auto &internal_result : internal_results)
        {
            const auto result = internal_to_external_list(internal_result);
            results.push_back(result);
        }

        return results;
    }

    std::vector<SearchResult> Collection::internal_to_external_list(
        std::vector<InternalSearchResult> internal_results) const
    {
        std::vector<SearchResult> results;
        results.reserve(internal_results.size());

        for (const auto &result : internal_results)
        {
            results.push_back({
                internal_to_external_.at(static_cast<std::size_t>(result.internal_id)),
                result.internal_id,
                result.score,
            });
        }

        return results;
    }

    std::size_t Collection::size() const
    {
        return vectors_.size();
    }

    std::size_t Collection::dim() const
    {
        return vectors_.dim();
    }

    Metric Collection::metric() const
    {
        return metric_;
    }

}
