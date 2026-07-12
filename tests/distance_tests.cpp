#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "vectordb/distance.hpp"

TEST(DistanceTest, ComputesDistancesAndSimilarities) {
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, 6.0f, 3.0f};

    EXPECT_NEAR(vectordb::l2_distance(a.data(), b.data(), a.size()), 5.0f,
                0.0001f);
    EXPECT_NEAR(vectordb::dot_product(a.data(), b.data(), a.size()), 25.0f,
                0.0001f);
    EXPECT_NEAR(vectordb::cosine_similarity(a.data(), a.data(), a.size()), 1.0f,
                0.0001f);
}

TEST(DistanceTest, NormalizesVectors) {
    std::vector<float> normalized{3.0f, 4.0f};

    vectordb::normalize_in_place(normalized);

    EXPECT_NEAR(normalized[0], 0.6f, 0.0001f);
    EXPECT_NEAR(normalized[1], 0.8f, 0.0001f);
}

TEST(DistanceTest, RejectsZeroVectorsForCosineOperations) {
    const std::vector<float> zero{0.0f, 0.0f};
    const std::vector<float> non_zero{1.0f, 0.0f};

    EXPECT_THROW(
        vectordb::cosine_similarity(zero.data(), non_zero.data(), zero.size()),
        std::invalid_argument);
    EXPECT_THROW(
        {
            std::vector<float> value(2, 0.0f);
            vectordb::normalize_in_place(value);
        },
        std::invalid_argument);
}
