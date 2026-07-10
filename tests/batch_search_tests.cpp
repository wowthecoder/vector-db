#include "vectordb/collection.hpp"

#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

TEST(BatchSearchTest, SearchesMultipleQueriesInInputOrder)
{
    vectordb::Collection collection(2, vectordb::Metric::L2);
    collection.insert("zero", std::vector<float>{0.0f, 0.0f});
    collection.insert("one", std::vector<float>{1.0f, 0.0f});
    collection.insert("three", std::vector<float>{3.0f, 0.0f});

    const std::vector<float> queries{
        0.0f,
        0.0f,
        3.0f,
        0.0f,
    };
    const auto batches = collection.batch_search(queries, 2);

    ASSERT_EQ(batches.size(), 2);
    ASSERT_EQ(batches[0].size(), 2);
    ASSERT_EQ(batches[1].size(), 2);

    EXPECT_EQ(batches[0][0].external_id, "zero");
    EXPECT_EQ(batches[0][1].external_id, "one");
    EXPECT_EQ(batches[1][0].external_id, "three");
    EXPECT_EQ(batches[1][1].external_id, "one");
}

TEST(BatchSearchTest, MatchesRepeatedSingleSearchesForDotProduct)
{
    vectordb::Collection collection(2, vectordb::Metric::Dot);
    collection.insert("x", std::vector<float>{2.0f, 0.0f});
    collection.insert("y", std::vector<float>{0.0f, 3.0f});
    collection.insert("xy", std::vector<float>{1.0f, 1.0f});

    const std::vector<float> queries{
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const auto batches = collection.batch_search(queries, 2);
    const auto first_expected = collection.search(
        std::span<const float>(queries.data(), collection.dim()), 2);
    const auto second_expected = collection.search(
        std::span<const float>(queries.data() + collection.dim(), collection.dim()), 2);

    ASSERT_EQ(batches.size(), 2);
    ASSERT_EQ(batches[0].size(), first_expected.size());
    ASSERT_EQ(batches[1].size(), second_expected.size());

    for (std::size_t i = 0; i < first_expected.size(); ++i)
    {
        EXPECT_EQ(batches[0][i].external_id, first_expected[i].external_id);
        EXPECT_FLOAT_EQ(batches[0][i].score, first_expected[i].score);
        EXPECT_EQ(batches[1][i].external_id, second_expected[i].external_id);
        EXPECT_FLOAT_EQ(batches[1][i].score, second_expected[i].score);
    }
}

TEST(BatchSearchTest, HandlesEmptyInputAndZeroTopK)
{
    vectordb::Collection collection(2, vectordb::Metric::L2);
    collection.insert("point", std::vector<float>{1.0f, 1.0f});

    EXPECT_TRUE(collection.batch_search({}, 1).empty());

    const std::vector<float> queries{
        0.0f,
        0.0f,
        1.0f,
        1.0f,
    };
    const auto batches = collection.batch_search(queries, 0);

    ASSERT_EQ(batches.size(), 2);
    EXPECT_TRUE(batches[0].empty());
    EXPECT_TRUE(batches[1].empty());
}

TEST(BatchSearchTest, RejectsIncompleteQueryVector)
{
    vectordb::Collection collection(3, vectordb::Metric::L2);
    const std::vector<float> incomplete_queries{1.0f, 2.0f, 3.0f, 4.0f};

    EXPECT_THROW(collection.batch_search(incomplete_queries, 1), std::invalid_argument);
}
