#include "vectordb/distance.hpp"

#include <cmath>
#include <stdexcept>

namespace vectordb
{

    float l2_distance(const float *a, const float *b, std::size_t dim)
    {
        float sum = 0.0f;

        for (std::size_t i = 0; i < dim; ++i)
        {
            const float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return std::sqrt(sum);
    }

    float dot_product(const float *a, const float *b, std::size_t dim)
    {
        float res = 0.0f;

        for (std::size_t i = 0; i < dim; ++i)
        {
            res += a[i] * b[i];
        }

        return res;
    }

    float cosine_similarity(const float *a, const float *b, std::size_t dim)
    {
        float numerator = dot_product(a, b, dim);
        float norm_a = 0.0f;
        float norm_b = 0.0f;

        for (std::size_t i = 0; i < dim; ++i)
        {
            norm_a += a[i] * a[i];
            norm_b += b[i] * b[i];
        }

        if (norm_a == 0.0f || norm_b == 0.0f)
        {
            throw std::invalid_argument("Cannot compute cosine similarity with zero vector");
        }

        norm_a = std::sqrt(norm_a);
        norm_b = std::sqrt(norm_b);

        return numerator / (norm_a * norm_b);
    }

    void normalize_in_place(std::vector<float> &v)
    {
        float norm_sq = 0.0f;

        for (float x : v)
        {
            norm_sq += x * x;
        }

        norm_sq = std::sqrt(norm_sq);

        if (norm_sq == 0.0f)
        {
            throw std::invalid_argument("Cannot normalize zero vector");
        }

        for (float &x : v)
        {
            x /= norm_sq;
        }
    }

}
