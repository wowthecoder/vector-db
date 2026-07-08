#pragma once

#include "vectordb/types.hpp"

#include <cstddef>
#include <vector>

namespace vectordb
{

    float l2_distance(const float *a, const float *b, std::size_t dim);
    float dot_product(const float *a, const float *b, std::size_t dim);
    // Computes cosine similarity for arbitrary, non-normalized vectors.
    // This is slower because it computes both vector norms.
    // Useful for tests, validation, or one-off calls.
    float cosine_similarity(const float *a, const float *b, std::size_t dim);

    // L2 or unit vector normalization
    // Normalize so that L2 of vector = 1
    void normalize_in_place(std::vector<float> &v);
}
