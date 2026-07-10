#include "vectordb/collection.hpp"

#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

namespace
{

    class TemporaryFile
    {
    public:
        explicit TemporaryFile(const std::string &test_name)
            : path_(std::filesystem::temp_directory_path() /
                    ("vectordb_" + test_name + "_" +
                     std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) +
                     ".bin"))
        {
        }

        ~TemporaryFile()
        {
            std::error_code error;
            std::filesystem::remove(path_, error);
        }

        const std::filesystem::path &path() const
        {
            return path_;
        }

    private:
        std::filesystem::path path_;
    };

}

TEST(PersistenceTest, DISABLED_RoundTripPreservesConfigurationAndSearchResults)
{
    const TemporaryFile file("round_trip");
    vectordb::Collection original(3, vectordb::Metric::Dot);
    original.insert("first", std::vector<float>{2.0f, 0.0f, 0.0f});
    original.insert("second", std::vector<float>{0.0f, 3.0f, 0.0f});
    original.insert("third", std::vector<float>{1.0f, 1.0f, 0.0f});

    const std::vector<float> query{1.0f, 0.0f, 0.0f};
    const auto expected = original.search(query, 3);

    original.save(file.path());
    const auto loaded = vectordb::Collection::load(file.path());

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->dim(), original.dim());
    EXPECT_EQ(loaded->size(), original.size());
    EXPECT_EQ(loaded->metric(), original.metric());

    const auto actual = loaded->search(query, 3);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_EQ(actual[i].external_id, expected[i].external_id);
        EXPECT_EQ(actual[i].internal_id, expected[i].internal_id);
        EXPECT_FLOAT_EQ(actual[i].score, expected[i].score);
    }
}

TEST(PersistenceTest, DISABLED_RoundTripPreservesEmptyCollection)
{
    const TemporaryFile file("empty");
    const vectordb::Collection original(4, vectordb::Metric::Cosine);

    original.save(file.path());
    const auto loaded = vectordb::Collection::load(file.path());

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->dim(), 4);
    EXPECT_EQ(loaded->metric(), vectordb::Metric::Cosine);
    EXPECT_EQ(loaded->size(), 0);
}

TEST(PersistenceTest, DISABLED_LoadRejectsMissingFile)
{
    const TemporaryFile file("missing");

    EXPECT_THROW(vectordb::Collection::load(file.path()), std::exception);
}
