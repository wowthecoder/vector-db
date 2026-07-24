#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include "vectordb/distance.hpp"
#include "vectordb/indexes/flat_index.hpp"
#include "vectordb/indexes/random_projection_lsh_index.hpp"

namespace {

vectordb::RandomProjectionLshConfig test_config() {
    return {
        .num_tables = 4,
        .num_bits_per_table = 4,
        .num_candidates = 10,
        .seed = 42,
    };
}

void expect_same_results(
    const std::vector<vectordb::InternalSearchResult> &left,
    const std::vector<vectordb::InternalSearchResult> &right) {
    ASSERT_EQ(left.size(), right.size());

    for (std::size_t i = 0; i < left.size(); ++i) {
        EXPECT_EQ(left[i].internal_id, right[i].internal_id);
        EXPECT_FLOAT_EQ(left[i].score, right[i].score);
    }
}

}  // namespace

TEST(RandomProjectionLshIndexTest, RejectsInvalidConfiguration) {
    const vectordb::VectorStore vectors(2);

    auto config = test_config();
    config.num_tables = 0;
    EXPECT_THROW(vectordb::RandomProjectionLshIndex(
                     vectors, vectordb::Metric::Cosine, config),
                 std::invalid_argument);

    config = test_config();
    config.num_bits_per_table = 0;
    EXPECT_THROW(vectordb::RandomProjectionLshIndex(
                     vectors, vectordb::Metric::Cosine, config),
                 std::invalid_argument);

    config = test_config();
    config.num_bits_per_table = 65;
    EXPECT_THROW(vectordb::RandomProjectionLshIndex(
                     vectors, vectordb::Metric::Cosine, config),
                 std::invalid_argument);

    config = test_config();
    config.num_candidates = 0;
    EXPECT_THROW(vectordb::RandomProjectionLshIndex(
                     vectors, vectordb::Metric::Cosine, config),
                 std::invalid_argument);
}

TEST(RandomProjectionLshIndexTest, ExposesConfigurationAndBuildState) {
    const vectordb::VectorStore vectors(3);
    const auto config = test_config();
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             config);

    EXPECT_FALSE(index.is_built());
    EXPECT_EQ(index.config().num_tables, config.num_tables);
    EXPECT_EQ(index.config().num_bits_per_table, config.num_bits_per_table);
    EXPECT_EQ(index.config().num_candidates, config.num_candidates);
    EXPECT_EQ(index.config().seed, config.seed);

    index.build();
    EXPECT_TRUE(index.is_built());
}

TEST(RandomProjectionLshIndexTest, RequiresBuildBeforeSearching) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});

    const vectordb::RandomProjectionLshIndex index(
        vectors, vectordb::Metric::Cosine, test_config());

    EXPECT_THROW(index.search(std::vector<float>{1.0f, 0.0f}, 1),
                 std::logic_error);
}

TEST(RandomProjectionLshIndexTest, AddsOneVectorAfterAnEmptyBuild) {
    vectordb::VectorStore vectors(2);
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    const std::uint64_t internal_id =
        vectors.add(std::vector<float>{1.0f, 0.0f});
    index.add(internal_id);

    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results.front().internal_id, internal_id);
}

TEST(RandomProjectionLshIndexTest, RequiresBuildBeforeAdding) {
    vectordb::VectorStore vectors(2);
    const std::uint64_t internal_id =
        vectors.add(std::vector<float>{1.0f, 0.0f});
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());

    EXPECT_THROW(index.add(internal_id), std::logic_error);
}

TEST(RandomProjectionLshIndexTest, EmptyBuiltIndexReturnsNoResults) {
    const vectordb::VectorStore vectors(2);
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    EXPECT_TRUE(index.search(std::vector<float>{1.0f, 0.0f}, 10).empty());
}

TEST(RandomProjectionLshIndexTest, ZeroTopKReturnsNoResults) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});

    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    EXPECT_TRUE(index.search(std::vector<float>{1.0f, 0.0f}, 0).empty());
}

TEST(RandomProjectionLshIndexTest, FindsAnIdenticalCosineVector) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{0.0f, 1.0f});
    vectors.add(std::vector<float>{-1.0f, 0.0f});

    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 3);

    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().internal_id, 0);
    EXPECT_NEAR(results.front().score, 1.0f, 0.0001f);
}

TEST(RandomProjectionLshIndexTest, ReturnsAtMostTopKResults) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{1.0f, 0.0f});

    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].internal_id, 0);
    EXPECT_EQ(results[1].internal_id, 1);
}

TEST(RandomProjectionLshIndexTest, DeduplicatesCandidatesAndUsesIdTieBreaks) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{1.0f, 0.0f});

    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 10);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].internal_id, 0);
    EXPECT_EQ(results[1].internal_id, 1);

    std::unordered_set<std::uint64_t> result_ids;
    for (const auto &result : results) {
        EXPECT_TRUE(result_ids.insert(result.internal_id).second);
    }
}

TEST(RandomProjectionLshIndexTest, RespectsCandidateLimit) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{1.0f, 0.0f});

    auto config = test_config();
    config.num_candidates = 1;
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             config);
    index.build();

    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 10);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results.front().internal_id, 0);
}

TEST(RandomProjectionLshIndexTest, ReranksCandidatesByExactCosineScore) {
    vectordb::VectorStore vectors(2);
    const std::array<std::vector<float>, 4> stored_vectors{
        std::vector<float>{1.0f, 0.0f},
        std::vector<float>{-1.0f, 0.0f},
        std::vector<float>{0.0f, 1.0f},
        std::vector<float>{0.0f, -1.0f},
    };
    for (const auto &vector : stored_vectors) {
        vectors.add(vector);
    }

    auto config = test_config();
    config.num_tables = 1;
    config.num_bits_per_table = 1;
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             config);
    index.build();

    bool observed_multiple_candidates = false;
    for (std::size_t query_id = 0; query_id < stored_vectors.size();
         ++query_id) {
        const auto &query = stored_vectors[query_id];
        const auto results = index.search(query, stored_vectors.size());

        ASSERT_FALSE(results.empty());
        EXPECT_EQ(results.front().internal_id, query_id);
        EXPECT_NEAR(results.front().score, 1.0f, 0.0001f);
        observed_multiple_candidates |= results.size() > 1;

        for (std::size_t i = 0; i < results.size(); ++i) {
            const float expected_score = vectordb::cosine_similarity(
                query.data(), vectors.get(results[i].internal_id),
                vectors.dim());
            EXPECT_NEAR(results[i].score, expected_score, 0.0001f);

            if (i > 0) {
                EXPECT_GE(results[i - 1].score, results[i].score);
            }
        }
    }

    EXPECT_TRUE(observed_multiple_candidates);
}

TEST(RandomProjectionLshIndexTest, ValidatesQueryDimension) {
    const vectordb::VectorStore vectors(2);
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    EXPECT_THROW(index.search(std::vector<float>{1.0f}, 1),
                 std::invalid_argument);
}

TEST(RandomProjectionLshIndexTest, ValidatesQueryDimensionEvenWhenTopKIsZero) {
    const vectordb::VectorStore vectors(2);
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    EXPECT_THROW(index.search(std::vector<float>{1.0f}, 0),
                 std::invalid_argument);
}

TEST(RandomProjectionLshIndexTest, SameSeedProducesSameSearchResults) {
    vectordb::VectorStore vectors(3);
    vectors.add(std::vector<float>{1.0f, 0.0f, 0.0f});
    vectors.add(std::vector<float>{0.8f, 0.2f, 0.0f});
    vectors.add(std::vector<float>{0.0f, 1.0f, 0.0f});
    vectors.add(std::vector<float>{0.0f, 0.0f, 1.0f});
    vectors.add(std::vector<float>{-1.0f, 0.0f, 0.0f});

    vectordb::RandomProjectionLshIndex first(vectors, vectordb::Metric::Cosine,
                                             test_config());
    vectordb::RandomProjectionLshIndex second(vectors, vectordb::Metric::Cosine,
                                              test_config());
    first.build();
    second.build();

    const std::array<std::vector<float>, 3> queries{
        std::vector<float>{1.0f, 0.0f, 0.0f},
        std::vector<float>{0.0f, 1.0f, 0.0f},
        std::vector<float>{0.0f, 0.0f, 1.0f},
    };
    for (const auto &query : queries) {
        expect_same_results(first.search(query, 4), second.search(query, 4));
    }
}

TEST(RandomProjectionLshIndexTest, RebuildingWithSameSeedIsDeterministic) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{0.8f, 0.2f});
    vectors.add(std::vector<float>{0.0f, 1.0f});

    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();
    const auto before = index.search(std::vector<float>{1.0f, 0.0f}, 3);

    index.build();
    const auto after = index.search(std::vector<float>{1.0f, 0.0f}, 3);

    expect_same_results(before, after);
}

TEST(RandomProjectionLshIndexTest, SupportsSixtyFourBitSignatures) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 2.0f});

    auto config = test_config();
    config.num_tables = 1;
    config.num_bits_per_table = 64;
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             config);
    index.build();

    const auto results = index.search(std::vector<float>{1.0f, 2.0f}, 1);

    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results.front().internal_id, 0);
    EXPECT_NEAR(results.front().score, 1.0f, 0.0001f);
}

TEST(RandomProjectionLshIndexTest,
     ReturnsEmptyWhenNoExactSignatureBucketMatches) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 2.0f});

    auto config = test_config();
    config.num_tables = 1;
    config.num_bits_per_table = 64;
    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             config);
    index.build();

    const auto results = index.search(std::vector<float>{-1.0f, -2.0f}, 1);

    EXPECT_TRUE(results.empty());
}

TEST(RandomProjectionLshIndexTest, PropagatesInvalidCosineVectorErrors) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{0.0f, 0.0f});

    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    EXPECT_THROW(index.search(std::vector<float>{0.0f, 0.0f}, 1),
                 std::invalid_argument);
}

TEST(RandomProjectionLshIndexTest, MatchesFlatIndexForEasyExactTopOneQueries) {
    vectordb::VectorStore vectors(3);
    const std::array<std::vector<float>, 6> stored_vectors{
        std::vector<float>{1.0f, 0.0f, 0.0f},
        std::vector<float>{-1.0f, 0.0f, 0.0f},
        std::vector<float>{0.0f, 1.0f, 0.0f},
        std::vector<float>{0.0f, -1.0f, 0.0f},
        std::vector<float>{0.0f, 0.0f, 1.0f},
        std::vector<float>{0.0f, 0.0f, -1.0f},
    };
    for (const auto &vector : stored_vectors) {
        vectors.add(vector);
    }

    vectordb::FlatIndex flat(vectors, vectordb::Metric::Cosine);
    auto config = test_config();
    config.num_candidates = stored_vectors.size();
    vectordb::RandomProjectionLshIndex lsh(vectors, vectordb::Metric::Cosine,
                                           config);
    lsh.build();

    for (const auto &query : stored_vectors) {
        const auto exact = flat.search(query, 1);
        const auto approximate = lsh.search(query, 1);

        ASSERT_EQ(exact.size(), 1);
        ASSERT_EQ(approximate.size(), 1);
        EXPECT_EQ(approximate.front().internal_id, exact.front().internal_id);
        EXPECT_NEAR(approximate.front().score, exact.front().score, 0.0001f);
    }
}

TEST(RandomProjectionLshIndexTest, RebuildsAfterVectorsAreAdded) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{0.0f, 1.0f});

    vectordb::RandomProjectionLshIndex index(vectors, vectordb::Metric::Cosine,
                                             test_config());
    index.build();

    const std::uint64_t added_id = vectors.add(std::vector<float>{1.0f, 0.0f});

    const auto stale_results = index.search(std::vector<float>{1.0f, 0.0f}, 2);
    for (const auto &result : stale_results) {
        EXPECT_NE(result.internal_id, added_id);
    }

    index.build();

    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 2);

    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().internal_id, added_id);
}
