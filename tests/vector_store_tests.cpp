#include <gtest/gtest.h>

#include <array>
#include <stdexcept>
#include <vector>

#include "vectordb/vector_store.hpp"

TEST(VectorStoreTest, AddsAndRetrievesVectors) {
    vectordb::VectorStore store(3);

    const std::vector<float> first{1.0f, 2.0f, 3.0f};
    const std::vector<float> second{4.0f, 5.0f, 6.0f};

    EXPECT_EQ(store.add(first), 0);
    EXPECT_EQ(store.add(second), 1);
    EXPECT_EQ(store.size(), 2);
    EXPECT_EQ(store.dim(), 3);
    EXPECT_NEAR(store.get(1)[2], 6.0f, 0.0001f);
}

TEST(VectorStoreTest, AssignsSequentialIdsAndPreservesStoredValues) {
    vectordb::VectorStore store(2);

    EXPECT_EQ(store.add(std::array<float, 2>{1.0f, 2.0f}), 0);
    EXPECT_EQ(store.add(std::array<float, 2>{3.0f, 4.0f}), 1);
    EXPECT_EQ(store.add(std::array<float, 2>{5.0f, 6.0f}), 2);

    EXPECT_EQ(store.size(), 3);
    EXPECT_NEAR(store.get(0)[0], 1.0f, 0.0001f);
    EXPECT_NEAR(store.get(0)[1], 2.0f, 0.0001f);
    EXPECT_NEAR(store.get(1)[0], 3.0f, 0.0001f);
    EXPECT_NEAR(store.get(1)[1], 4.0f, 0.0001f);
    EXPECT_NEAR(store.get(2)[0], 5.0f, 0.0001f);
    EXPECT_NEAR(store.get(2)[1], 6.0f, 0.0001f);
}

TEST(VectorStoreTest, ValidatesDimensionsAndIds) {
    vectordb::VectorStore store(3);
    const std::vector<float> wrong_dim{1.0f, 2.0f};

    EXPECT_THROW(store.add(wrong_dim), std::invalid_argument);
    EXPECT_THROW(store.get(0), std::out_of_range);
    EXPECT_THROW(vectordb::VectorStore invalid(0), std::invalid_argument);
}
