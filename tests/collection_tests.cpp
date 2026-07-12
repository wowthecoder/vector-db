#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "vectordb/collection.hpp"

TEST(CollectionTest, EmptySearchReturnsNoResults) {
    vectordb::Collection collection(3, vectordb::Metric::L2);

    const auto results =
        collection.search(std::vector<float>{1.0f, 2.0f, 3.0f}, 10);

    EXPECT_TRUE(results.empty());
    EXPECT_EQ(collection.size(), 0);
    EXPECT_EQ(collection.dim(), 3);
}

TEST(CollectionTest, RanksL2SearchByLowestDistance) {
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

TEST(CollectionTest, TopKLargerThanCollectionReturnsAllSortedResults) {
    vectordb::Collection collection(2, vectordb::Metric::L2);

    collection.insert("middle", std::vector<float>{2.0f, 0.0f});
    collection.insert("nearest", std::vector<float>{1.0f, 0.0f});
    collection.insert("farthest", std::vector<float>{3.0f, 0.0f});

    const auto results = collection.search(std::vector<float>{0.0f, 0.0f}, 10);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].external_id, "nearest");
    EXPECT_EQ(results[1].external_id, "middle");
    EXPECT_EQ(results[2].external_id, "farthest");
}

TEST(CollectionTest, TiesAreOrderedByInternalId) {
    vectordb::Collection collection(2, vectordb::Metric::Dot);

    collection.insert("first", std::vector<float>{1.0f, 0.0f});
    collection.insert("second", std::vector<float>{1.0f, 0.0f});
    collection.insert("third", std::vector<float>{1.0f, 0.0f});

    const auto results = collection.search(std::vector<float>{1.0f, 0.0f}, 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].external_id, "first");
    EXPECT_EQ(results[0].internal_id, 0);
    EXPECT_EQ(results[1].external_id, "second");
    EXPECT_EQ(results[1].internal_id, 1);
    EXPECT_EQ(results[2].external_id, "third");
    EXPECT_EQ(results[2].internal_id, 2);
}

TEST(CollectionTest, RanksDotSearchByHighestScore) {
    vectordb::Collection collection(2, vectordb::Metric::Dot);

    collection.insert("x", std::vector<float>{1.0f, 0.0f});
    collection.insert("y", std::vector<float>{0.0f, 1.0f});
    collection.insert("big_x", std::vector<float>{2.0f, 0.0f});

    const auto results = collection.search(std::vector<float>{1.0f, 0.0f}, 3);

    ASSERT_EQ(results.size(), 3);
    EXPECT_EQ(results[0].external_id, "big_x");
    EXPECT_EQ(results[1].external_id, "x");
}

TEST(CollectionTest, RanksCosineSearchByHighestSimilarity) {
    vectordb::Collection collection(2, vectordb::Metric::Cosine);

    collection.insert("same", std::vector<float>{2.0f, 0.0f});
    collection.insert("orthogonal", std::vector<float>{0.0f, 3.0f});

    const auto results = collection.search(std::vector<float>{1.0f, 0.0f}, 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].external_id, "same");
    EXPECT_NEAR(results[0].score, 1.0f, 0.0001f);
}

TEST(CollectionTest, ValidatesInputs) {
    vectordb::Collection collection(2, vectordb::Metric::L2);

    collection.insert("id", std::vector<float>{1.0f, 2.0f});

    EXPECT_EQ(collection.size(), 1);
    EXPECT_EQ(collection.dim(), 2);
    EXPECT_TRUE(collection.search(std::vector<float>{1.0f, 2.0f}, 0).empty());
    EXPECT_THROW(collection.insert("id", std::vector<float>{3.0f, 4.0f}),
                 std::invalid_argument);
    EXPECT_THROW(collection.insert("", std::vector<float>{3.0f, 4.0f}),
                 std::invalid_argument);
    EXPECT_THROW(collection.search(std::vector<float>{1.0f}, 1),
                 std::invalid_argument);
}

TEST(CollectionTest, FailedInsertDoesNotConsumeExternalId) {
    vectordb::Collection collection(2, vectordb::Metric::L2);

    EXPECT_THROW(collection.insert("retry", std::vector<float>{1.0f}),
                 std::invalid_argument);
    EXPECT_EQ(collection.size(), 0);

    collection.insert("retry", std::vector<float>{1.0f, 2.0f});

    const auto results = collection.search(std::vector<float>{1.0f, 2.0f}, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results[0].external_id, "retry");
    EXPECT_EQ(results[0].internal_id, 0);
}
