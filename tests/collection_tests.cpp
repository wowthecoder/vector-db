#include <gtest/gtest.h>

#include <stdexcept>
#include <vector>

#include "vectordb/collection.hpp"

namespace {

vectordb::CollectionOptions lsh_options() {
    return {
        .index_kind = vectordb::IndexKind::RandomProjectionLsh,
        .lsh =
            {
                .num_tables = 4,
                .num_bits_per_table = 4,
                .num_candidates = 10,
                .seed = 42,
            },
    };
}

}  // namespace

TEST(CollectionTest, EmptySearchReturnsNoResults) {
    vectordb::Collection collection(3, vectordb::Metric::L2);

    const auto results =
        collection.search(std::vector<float>{1.0f, 2.0f, 3.0f}, 10);

    EXPECT_TRUE(results.empty());
    EXPECT_EQ(collection.size(), 0);
    EXPECT_EQ(collection.dim(), 3);
    EXPECT_EQ(collection.index_kind(), vectordb::IndexKind::Flat);
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

TEST(CollectionTest, SearchesWithRandomProjectionLshIndex) {
    vectordb::Collection collection(2, vectordb::Metric::Cosine, lsh_options());

    collection.insert("same", std::vector<float>{1.0f, 0.0f});
    collection.insert("orthogonal", std::vector<float>{0.0f, 1.0f});

    const auto results = collection.search(std::vector<float>{1.0f, 0.0f}, 2);

    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().external_id, "same");
    EXPECT_EQ(results.front().internal_id, 0);
    EXPECT_NEAR(results.front().score, 1.0f, 0.0001f);
    EXPECT_EQ(collection.index_kind(),
              vectordb::IndexKind::RandomProjectionLsh);
    EXPECT_EQ(collection.lsh_config().num_tables, 4);
    EXPECT_EQ(collection.lsh_config().num_bits_per_table, 4);
    EXPECT_EQ(collection.lsh_config().num_candidates, 10);
    EXPECT_EQ(collection.lsh_config().seed, 42);
}

TEST(CollectionTest, MakesNewInsertsImmediatelySearchableByLsh) {
    vectordb::Collection collection(2, vectordb::Metric::Cosine, lsh_options());

    collection.insert("first", std::vector<float>{1.0f, 0.0f});
    const auto first_results =
        collection.search(std::vector<float>{1.0f, 0.0f}, 1);
    ASSERT_EQ(first_results.size(), 1);
    EXPECT_EQ(first_results.front().external_id, "first");

    collection.insert("second", std::vector<float>{0.0f, 1.0f});
    const auto second_results =
        collection.search(std::vector<float>{0.0f, 1.0f}, 1);

    ASSERT_EQ(second_results.size(), 1);
    EXPECT_EQ(second_results.front().external_id, "second");
}

TEST(CollectionTest, BatchSearchWorksWithLshIndex) {
    vectordb::Collection collection(2, vectordb::Metric::Cosine, lsh_options());
    collection.insert("x", std::vector<float>{1.0f, 0.0f});
    collection.insert("y", std::vector<float>{0.0f, 1.0f});

    const std::vector<float> queries{
        1.0f,
        0.0f,
        0.0f,
        1.0f,
    };
    const auto results = collection.batch_search(queries, 1);

    ASSERT_EQ(results.size(), 2);
    ASSERT_EQ(results[0].size(), 1);
    ASSERT_EQ(results[1].size(), 1);
    EXPECT_EQ(results[0].front().external_id, "x");
    EXPECT_EQ(results[1].front().external_id, "y");
}

TEST(CollectionTest, RejectsUnsupportedLshMetrics) {
    EXPECT_THROW(vectordb::Collection(2, vectordb::Metric::L2, lsh_options()),
                 std::invalid_argument);
    EXPECT_THROW(vectordb::Collection(2, vectordb::Metric::Dot, lsh_options()),
                 std::invalid_argument);
}
