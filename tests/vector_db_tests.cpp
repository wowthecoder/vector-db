#include "vectordb/collection.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/vector_store.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(DistanceTest, ComputesDistancesAndSimilarities)
{
    const std::vector<float> a{1.0f, 2.0f, 3.0f};
    const std::vector<float> b{4.0f, 6.0f, 3.0f};

    EXPECT_NEAR(vectordb::l2_distance(a.data(), b.data(), a.size()), 5.0f, 0.0001f);
    EXPECT_NEAR(vectordb::dot_product(a.data(), b.data(), a.size()), 25.0f, 0.0001f);
    EXPECT_NEAR(vectordb::cosine_similarity(a.data(), a.data(), a.size()), 1.0f, 0.0001f);
}

TEST(DistanceTest, NormalizesVectors)
{
    std::vector<float> normalized{3.0f, 4.0f};

    vectordb::normalize_in_place(normalized);

    EXPECT_NEAR(normalized[0], 0.6f, 0.0001f);
    EXPECT_NEAR(normalized[1], 0.8f, 0.0001f);
}

TEST(DistanceTest, RejectsZeroVectorsForCosineOperations)
{
    const std::vector<float> zero{0.0f, 0.0f};
    const std::vector<float> non_zero{1.0f, 0.0f};

    EXPECT_THROW(vectordb::cosine_similarity(zero.data(), non_zero.data(), zero.size()), std::invalid_argument);
    EXPECT_THROW(
        {
            std::vector<float> value(2, 0.0f);
            vectordb::normalize_in_place(value);
        },
        std::invalid_argument);
}

TEST(VectorStoreTest, AddsAndRetrievesVectors)
{
    vectordb::VectorStore store(3);

    const std::vector<float> first{1.0f, 2.0f, 3.0f};
    const std::vector<float> second{4.0f, 5.0f, 6.0f};

    EXPECT_EQ(store.add(first), 0);
    EXPECT_EQ(store.add(second), 1);
    EXPECT_EQ(store.size(), 2);
    EXPECT_EQ(store.dim(), 3);
    EXPECT_NEAR(store.get(1)[2], 6.0f, 0.0001f);
}

TEST(VectorStoreTest, ValidatesDimensionsAndIds)
{
    vectordb::VectorStore store(3);
    const std::vector<float> wrong_dim{1.0f, 2.0f};

    EXPECT_THROW(store.add(wrong_dim), std::invalid_argument);
    EXPECT_THROW(store.get(0), std::out_of_range);
    EXPECT_THROW(vectordb::VectorStore invalid(0), std::invalid_argument);
}

TEST(CollectionTest, RanksL2SearchByLowestDistance)
{
    vectordb::Collection collection(2, vectordb::Metric::L2);

    collection.insert("near", std::vector<float>{1.0f, 1.0f});
    collection.insert("far", std::vector<float>{5.0f, 5.0f});
    collection.insert("exact", std::vector<float>{0.0f, 0.0f});

    const auto results = collection.search(std::vector<float>{0.0f, 0.0f}, 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].external_id, "exact");
    EXPECT_EQ(results[1].external_id, "near");
    EXPECT_NEAR(results[0].score, 0.0f, 0.0001f);
}

TEST(CollectionTest, RanksDotSearchByHighestScore)
{
    vectordb::Collection collection(2, vectordb::Metric::Dot);

    collection.insert("x", std::vector<float>{1.0f, 0.0f});
    collection.insert("y", std::vector<float>{0.0f, 1.0f});
    collection.insert("big_x", std::vector<float>{2.0f, 0.0f});

    const auto results = collection.search(std::vector<float>{1.0f, 0.0f}, 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].external_id, "big_x");
    EXPECT_EQ(results[1].external_id, "x");
}

TEST(CollectionTest, RanksCosineSearchByHighestSimilarity)
{
    vectordb::Collection collection(2, vectordb::Metric::Cosine);

    collection.insert("same", std::vector<float>{2.0f, 0.0f});
    collection.insert("orthogonal", std::vector<float>{0.0f, 3.0f});

    const auto results = collection.search(std::vector<float>{1.0f, 0.0f}, 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].external_id, "same");
    EXPECT_NEAR(results[0].score, 1.0f, 0.0001f);
}

TEST(CollectionTest, ValidatesInputs)
{
    vectordb::Collection collection(2, vectordb::Metric::L2);

    collection.insert("id", std::vector<float>{1.0f, 2.0f});

    EXPECT_EQ(collection.size(), 1);
    EXPECT_EQ(collection.dim(), 2);
    EXPECT_TRUE(collection.search(std::vector<float>{1.0f, 2.0f}, 0).empty());
    EXPECT_THROW(collection.insert("id", std::vector<float>{3.0f, 4.0f}), std::invalid_argument);
    EXPECT_THROW(collection.insert("", std::vector<float>{3.0f, 4.0f}), std::invalid_argument);
    EXPECT_THROW(collection.search(std::vector<float>{1.0f}, 1), std::invalid_argument);
}
