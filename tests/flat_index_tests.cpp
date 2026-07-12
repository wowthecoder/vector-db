#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "vectordb/indexes/flat_index.hpp"

TEST(FlatIndexTest, SearchesStoredVectorsByL2Distance) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{3.0f, 0.0f});
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{2.0f, 0.0f});

    const vectordb::FlatIndex index(vectors, vectordb::Metric::L2);
    const auto results = index.search(std::vector<float>{0.0f, 0.0f}, 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].internal_id, 1);
    EXPECT_NEAR(results[0].score, 1.0f, 0.0001f);
    EXPECT_EQ(results[1].internal_id, 2);
    EXPECT_NEAR(results[1].score, 2.0f, 0.0001f);
}

TEST(FlatIndexTest, SearchesStoredVectorsByDotProduct) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{0.0f, 1.0f});
    vectors.add(std::vector<float>{2.0f, 0.0f});

    const vectordb::FlatIndex index(vectors, vectordb::Metric::Dot);
    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].internal_id, 2);
    EXPECT_NEAR(results[0].score, 2.0f, 0.0001f);
    EXPECT_EQ(results[1].internal_id, 0);
    EXPECT_NEAR(results[1].score, 1.0f, 0.0001f);
}

TEST(FlatIndexTest, UsesInternalIdAsTieBreaker) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{1.0f, 0.0f});

    const vectordb::FlatIndex index(vectors, vectordb::Metric::Dot);
    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].internal_id, 0);
    EXPECT_EQ(results[1].internal_id, 1);
    EXPECT_EQ(results[2].internal_id, 2);
}

TEST(FlatIndexTest, ValidatesQueryDimension) {
    vectordb::VectorStore vectors(2);
    const vectordb::FlatIndex index(vectors, vectordb::Metric::L2);

    EXPECT_THROW(index.search(std::vector<float>{1.0f}, 1),
                 std::invalid_argument);
}
