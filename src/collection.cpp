#include "vectordb/collection.hpp"

#include <algorithm>
#include <stdexcept>

namespace vectordb
{

    Collection::Collection(std::size_t dim, Metric metric)
        : metric_(metric),
          vectors_(dim)
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

        std::vector<SearchResult> results;
        results.reserve(size());

        for (std::uint64_t internal_id = 0; internal_id < size(); ++internal_id)
        {
            const float *candidate = vectors_.get(internal_id);
            results.push_back({
                internal_to_external_.at(static_cast<std::size_t>(internal_id)),
                internal_id,
                score_vector(query.data(), candidate),
            });
        }

        const auto better = [this](const SearchResult &a, const SearchResult &b) {
            if (a.score == b.score)
            {
                return a.internal_id < b.internal_id;
            }

            return higher_is_better() ? a.score > b.score : a.score < b.score;
        };

        if (top_k < results.size())
        {
            std::partial_sort(results.begin(), results.begin() + static_cast<std::ptrdiff_t>(top_k), results.end(), better);
            results.resize(top_k);
        }
        else
        {
            std::sort(results.begin(), results.end(), better);
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

    float Collection::score_vector(const float *a, const float *b) const
    {
        switch (metric_)
        {
        case Metric::L2:
            return l2_distance(a, b, dim());
        case Metric::Dot:
            return dot_product(a, b, dim());
        case Metric::Cosine:
            return cosine_similarity(a, b, dim());
        }

        throw std::invalid_argument("Unsupported metric");
    }

    bool Collection::higher_is_better() const
    {
        return metric_ == Metric::Dot || metric_ == Metric::Cosine;
    }

}
