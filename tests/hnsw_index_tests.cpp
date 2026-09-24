#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vectordb/distance.hpp"
#include "vectordb/indexes/flat_index.hpp"
#include "vectordb/indexes/hnsw_index.hpp"

namespace {

vectordb::HnswConfig test_config() {
    return {
        .M = 8,
        .ef_construction = 32,
        .ef_search = 16,
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

TEST(HnswIndexTest, RejectsInvalidConfiguration) {
    const vectordb::VectorStore vectors(2);

    auto config = test_config();
    config.M = 0;
    EXPECT_THROW(
        vectordb::HnswIndex(vectors, vectordb::Metric::L2, config),
        std::invalid_argument);

    config = test_config();
    config.M = vectordb::HnswConfig::max_M + 1;
    EXPECT_THROW(
        vectordb::HnswIndex(vectors, vectordb::Metric::L2, config),
        std::invalid_argument);

    config = test_config();
    config.ef_construction = 0;
    EXPECT_THROW(
        vectordb::HnswIndex(vectors, vectordb::Metric::L2, config),
        std::invalid_argument);

    config = test_config();
    config.ef_search = 0;
    EXPECT_THROW(
        vectordb::HnswIndex(vectors, vectordb::Metric::L2, config),
        std::invalid_argument);
}

TEST(HnswIndexTest, ExposesConfigurationAndBuildState) {
    const vectordb::VectorStore vectors(3);
    const auto config = test_config();
    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, config);

    EXPECT_FALSE(index.is_built());
    EXPECT_EQ(index.config().M, config.M);
    EXPECT_EQ(index.config().ef_construction, config.ef_construction);
    EXPECT_EQ(index.config().ef_search, config.ef_search);
    EXPECT_EQ(index.config().seed, config.seed);

    index.build();
    EXPECT_TRUE(index.is_built());
}

TEST(HnswIndexTest, RequiresBuildBeforeSearching) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});

    const vectordb::HnswIndex index(vectors, vectordb::Metric::L2,
                                    test_config());

    EXPECT_THROW(index.search(std::vector<float>{1.0f, 0.0f}, 1),
                 std::logic_error);
}

TEST(HnswIndexTest, RequiresBuildBeforeAdding) {
    vectordb::VectorStore vectors(2);
    const std::uint64_t internal_id =
        vectors.add(std::vector<float>{1.0f, 0.0f});
    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());

    EXPECT_THROW(index.add(internal_id), std::logic_error);
}

TEST(HnswIndexTest, RejectsDuplicateAndOutOfOrderAdds) {
    vectordb::VectorStore vectors(2);
    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();

    const std::uint64_t first_id = vectors.add(std::vector<float>{1.0f, 0.0f});
    const std::uint64_t second_id = vectors.add(std::vector<float>{0.0f, 1.0f});

    EXPECT_THROW(index.add(second_id), std::logic_error);

    index.add(first_id);
    EXPECT_THROW(index.add(first_id), std::logic_error);
    index.add(second_id);

    EXPECT_EQ(
        index.search(std::vector<float>{1.0f, 0.0f}, 2).front().internal_id,
        first_id);
}

TEST(HnswIndexTest, EmptyBuiltIndexReturnsNoResults) {
    const vectordb::VectorStore vectors(2);
    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();

    EXPECT_TRUE(index.search(std::vector<float>{1.0f, 0.0f}, 10).empty());
}

TEST(HnswIndexTest, ZeroTopKReturnsNoResults) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});

    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();

    EXPECT_TRUE(index.search(std::vector<float>{1.0f, 0.0f}, 0).empty());
}

TEST(HnswIndexTest, ValidatesQueryDimension) {
    const vectordb::VectorStore vectors(2);
    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();

    EXPECT_THROW(index.search(std::vector<float>{1.0f}, 1),
                 std::invalid_argument);
}

TEST(HnswIndexTest, ValidatesQueryDimensionEvenWhenTopKIsZero) {
    const vectordb::VectorStore vectors(2);
    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();

    EXPECT_THROW(index.search(std::vector<float>{1.0f}, 0),
                 std::invalid_argument);
}

TEST(HnswIndexTest, AddsOneVectorAfterAnEmptyBuild) {
    vectordb::VectorStore vectors(2);
    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();

    const std::uint64_t internal_id =
        vectors.add(std::vector<float>{1.0f, 0.0f});
    index.add(internal_id);

    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 1);
    ASSERT_EQ(results.size(), 1);
    EXPECT_EQ(results.front().internal_id, internal_id);
}

TEST(HnswIndexTest, ReturnsAtMostTopKResults) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{2.0f, 0.0f});
    vectors.add(std::vector<float>{3.0f, 0.0f});

    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();

    const auto results = index.search(std::vector<float>{0.0f, 0.0f}, 2);

    ASSERT_EQ(results.size(), 2);
    EXPECT_EQ(results[0].internal_id, 0);
    EXPECT_EQ(results[1].internal_id, 1);
}

TEST(HnswIndexTest, DetectsStaleStoreAndRebuilds) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{0.0f, 1.0f});

    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();

    const std::uint64_t added_id = vectors.add(std::vector<float>{1.0f, 0.0f});

    EXPECT_THROW(index.search(std::vector<float>{1.0f, 0.0f}, 2),
                 std::logic_error);

    index.build();

    const auto results = index.search(std::vector<float>{1.0f, 0.0f}, 2);

    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results.front().internal_id, added_id);
}

TEST(HnswIndexTest, FailedRebuildPreservesPreviousIndexState) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});

    vectordb::HnswIndex index(vectors, vectordb::Metric::Cosine,
                              test_config());
    index.build();

    vectors.add(std::vector<float>{0.0f, 0.0f});

    EXPECT_THROW(index.build(), std::invalid_argument);
    EXPECT_TRUE(index.is_built());
    EXPECT_THROW(index.search(std::vector<float>{1.0f, 0.0f}, 1),
                 std::logic_error);
}

TEST(HnswIndexTest, SameSeedProducesSameSearchResults) {
    vectordb::VectorStore vectors(3);
    vectors.add(std::vector<float>{1.0f, 0.0f, 0.0f});
    vectors.add(std::vector<float>{0.8f, 0.2f, 0.0f});
    vectors.add(std::vector<float>{0.0f, 1.0f, 0.0f});
    vectors.add(std::vector<float>{0.0f, 0.0f, 1.0f});
    vectors.add(std::vector<float>{-1.0f, 0.0f, 0.0f});

    vectordb::HnswIndex first(vectors, vectordb::Metric::L2, test_config());
    vectordb::HnswIndex second(vectors, vectordb::Metric::L2, test_config());
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

TEST(HnswIndexTest, RebuildingWithSameSeedIsDeterministic) {
    vectordb::VectorStore vectors(2);
    vectors.add(std::vector<float>{1.0f, 0.0f});
    vectors.add(std::vector<float>{0.8f, 0.2f});
    vectors.add(std::vector<float>{0.0f, 1.0f});

    vectordb::HnswIndex index(vectors, vectordb::Metric::L2, test_config());
    index.build();
    const auto before = index.search(std::vector<float>{1.0f, 0.0f}, 3);

    index.build();
    const auto after = index.search(std::vector<float>{1.0f, 0.0f}, 3);

    expect_same_results(before, after);
}

TEST(HnswIndexTest, FullBuildAndIncrementalAddsProduceSameResults) {
    const std::array<std::vector<float>, 8> stored_vectors{
        std::vector<float>{1.0f, 0.0f, 0.0f},
        std::vector<float>{0.8f, 0.2f, 0.0f},
        std::vector<float>{0.0f, 1.0f, 0.0f},
        std::vector<float>{0.0f, 0.0f, 1.0f},
        std::vector<float>{-1.0f, 0.0f, 0.0f},
        std::vector<float>{0.2f, 0.3f, 0.9f},
        std::vector<float>{-0.4f, 0.6f, 0.1f},
        std::vector<float>{0.5f, -0.5f, 0.3f},
    };

    vectordb::VectorStore fully_built_vectors(3);
    for (const auto &stored_vector : stored_vectors) {
        fully_built_vectors.add(stored_vector);
    }
    vectordb::HnswIndex fully_built_index(
        fully_built_vectors, vectordb::Metric::L2, test_config());
    fully_built_index.build();

    vectordb::VectorStore incrementally_added_vectors(3);
    vectordb::HnswIndex incrementally_built_index(
        incrementally_added_vectors, vectordb::Metric::L2, test_config());
    incrementally_built_index.build();
    for (const auto &stored_vector : stored_vectors) {
        const std::uint64_t internal_id =
            incrementally_added_vectors.add(stored_vector);
        incrementally_built_index.add(internal_id);
    }

    for (const auto &query : stored_vectors) {
        expect_same_results(
            fully_built_index.search(query, stored_vectors.size()),
            incrementally_built_index.search(query, stored_vectors.size()));
    }
}

TEST(HnswIndexTest, MatchesFlatIndexForEasyExactTopOneQueriesAcrossMetrics) {
    const std::array<std::vector<float>, 6> stored_vectors{
        std::vector<float>{10.0f, 0.0f, 0.0f},
        std::vector<float>{-10.0f, 0.0f, 0.0f},
        std::vector<float>{0.0f, 10.0f, 0.0f},
        std::vector<float>{0.0f, -10.0f, 0.0f},
        std::vector<float>{0.0f, 0.0f, 10.0f},
        std::vector<float>{0.0f, 0.0f, -10.0f},
    };

    for (const vectordb::Metric metric :
        {vectordb::Metric::L2, vectordb::Metric::Dot,
         vectordb::Metric::Cosine}) {
        vectordb::VectorStore vectors(3);
        for (const auto &stored_vector : stored_vectors) {
            vectors.add(stored_vector);
        }

        vectordb::FlatIndex flat(vectors, metric);
        vectordb::HnswIndex hnsw(vectors, metric, test_config());
        hnsw.build();

        for (const auto &query : stored_vectors) {
            const auto exact = flat.search(query, 1);
            const auto approximate = hnsw.search(query, 1);

            ASSERT_EQ(exact.size(), 1);
            ASSERT_EQ(approximate.size(), 1);
            EXPECT_EQ(approximate.front().internal_id, exact.front().internal_id);
            EXPECT_NEAR(approximate.front().score, exact.front().score, 0.0001f);
        }
    }
}

TEST(HnswIndexTest, RecallOnSyntheticClustersMatchesFlatIndexCloselyForL2) {
    constexpr std::size_t cluster_count = 4;
    constexpr std::size_t points_per_cluster = 20;
    vectordb::VectorStore vectors(2);

    const std::array<std::pair<float, float>, cluster_count> centers{
        std::make_pair(0.0f, 0.0f),
        std::make_pair(20.0f, 0.0f),
        std::make_pair(0.0f, 20.0f),
        std::make_pair(20.0f, 20.0f),
    };

    for (const auto &[cx, cy] : centers) {
        for (std::size_t i = 0; i < points_per_cluster; ++i) {
            const float offset = static_cast<float>(i) * 0.05f;
            vectors.add(std::vector<float>{cx + offset, cy + offset});
        }
    }

    vectordb::FlatIndex flat(vectors, vectordb::Metric::L2);
    auto config = test_config();
    config.ef_search = 40;
    vectordb::HnswIndex hnsw(vectors, vectordb::Metric::L2, config);
    hnsw.build();

    std::size_t matches = 0;
    std::size_t total = 0;
    constexpr std::size_t top_k = 5;

    for (const auto &[cx, cy] : centers) {
        const std::vector<float> query{cx, cy};
        const auto exact = flat.search(query, top_k);
        const auto approximate = hnsw.search(query, top_k);

        std::unordered_set<std::uint64_t> exact_ids;
        for (const auto &result : exact) {
            exact_ids.insert(result.internal_id);
        }
        for (const auto &result : approximate) {
            matches += exact_ids.contains(result.internal_id) ? 1 : 0;
        }
        total += exact.size();
    }

    const double recall = static_cast<double>(matches) / static_cast<double>(total);
    EXPECT_GE(recall, 0.9);
}
