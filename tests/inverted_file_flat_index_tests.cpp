#include <gtest/gtest.h>

#include <stdexcept>

#include "vectordb/indexes/inverted_file_flat_index.hpp"

namespace {

vectordb::InvertedFileFlatConfig test_config() {
    return {
        .num_lists = 2,
        .num_probes = 1,
        .max_iterations = 10,
        .seed = 42,
    };
}

}  // namespace

TEST(InvertedFileFlatIndexTest, ExposesConfigurationAndInitialBuildState) {
    const vectordb::VectorStore vectors(3);
    const auto config = test_config();
    const vectordb::InvertedFileFlatIndex index(vectors, vectordb::Metric::L2,
                                                config);

    EXPECT_FALSE(index.is_built());
    EXPECT_EQ(index.config().num_lists, config.num_lists);
    EXPECT_EQ(index.config().num_probes, config.num_probes);
    EXPECT_EQ(index.config().max_iterations, config.max_iterations);
    EXPECT_EQ(index.config().seed, config.seed);
}

TEST(InvertedFileFlatIndexTest, RejectsUnsupportedMetrics) {
    const vectordb::VectorStore vectors(3);

    EXPECT_THROW(vectordb::InvertedFileFlatIndex(vectors, vectordb::Metric::Dot,
                                                 test_config()),
                 std::invalid_argument);
    EXPECT_THROW(vectordb::InvertedFileFlatIndex(
                     vectors, vectordb::Metric::Cosine, test_config()),
                 std::invalid_argument);
}

TEST(InvertedFileFlatIndexTest, RejectsInvalidConfiguration) {
    const vectordb::VectorStore vectors(3);

    auto config = test_config();
    config.num_lists = 0;
    EXPECT_THROW(
        vectordb::InvertedFileFlatIndex(vectors, vectordb::Metric::L2, config),
        std::invalid_argument);

    config = test_config();
    config.num_probes = 0;
    EXPECT_THROW(
        vectordb::InvertedFileFlatIndex(vectors, vectordb::Metric::L2, config),
        std::invalid_argument);

    config = test_config();
    config.num_probes = config.num_lists + 1;
    EXPECT_THROW(
        vectordb::InvertedFileFlatIndex(vectors, vectordb::Metric::L2, config),
        std::invalid_argument);

    config = test_config();
    config.max_iterations = 0;
    EXPECT_THROW(
        vectordb::InvertedFileFlatIndex(vectors, vectordb::Metric::L2, config),
        std::invalid_argument);
}
