#include <gtest/gtest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include "vectordb/collection.hpp"

namespace {

constexpr std::streamoff k_magic_offset = 0;
constexpr std::streamoff k_version_offset = 8;
constexpr std::streamoff k_metric_offset = 12;
constexpr std::streamoff k_first_record_offset = 32;

void overwrite_bytes(const std::filesystem::path &path, std::streamoff offset,
                     const std::string &bytes) {
    std::fstream file(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!file) {
        throw std::runtime_error("Failed to open test collection file");
    }

    file.seekp(offset);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) {
        throw std::runtime_error("Failed to modify test collection file");
    }
}

class TemporaryFile {
   public:
    explicit TemporaryFile(const std::string &test_name)
        : path_(std::filesystem::temp_directory_path() /
                ("vectordb_" + test_name + "_" +
                 std::to_string(std::chrono::steady_clock::now()
                                    .time_since_epoch()
                                    .count()) +
                 ".bin")) {}

    ~TemporaryFile() {
        std::error_code error;
        std::filesystem::remove(path_, error);
    }

    const std::filesystem::path &path() const { return path_; }

   private:
    std::filesystem::path path_;
};

}  // namespace

TEST(PersistenceTest, RoundTripPreservesConfigurationAndSearchResults) {
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
    for (std::size_t i = 0; i < expected.size(); ++i) {
        EXPECT_EQ(actual[i].external_id, expected[i].external_id);
        EXPECT_EQ(actual[i].internal_id, expected[i].internal_id);
        EXPECT_FLOAT_EQ(actual[i].score, expected[i].score);
    }
}

TEST(PersistenceTest, RoundTripPreservesEmptyCollection) {
    const TemporaryFile file("empty");
    const vectordb::Collection original(4, vectordb::Metric::Cosine);

    original.save(file.path());
    const auto loaded = vectordb::Collection::load(file.path());

    ASSERT_NE(loaded, nullptr);
    EXPECT_EQ(loaded->dim(), 4);
    EXPECT_EQ(loaded->metric(), vectordb::Metric::Cosine);
    EXPECT_EQ(loaded->size(), 0);
}

TEST(PersistenceTest, LoadRejectsMissingFile) {
    const TemporaryFile file("missing");

    EXPECT_THROW(vectordb::Collection::load(file.path()), std::exception);
}

TEST(PersistenceTest, LoadRejectsWrongMagicBytes) {
    const TemporaryFile file("wrong_magic");
    const vectordb::Collection collection(2, vectordb::Metric::L2);
    collection.save(file.path());
    overwrite_bytes(file.path(), k_magic_offset,
                    std::string{'N', 'O', 'T', 'V', 'D', 'B', '\0', '\0'});

    EXPECT_THROW(vectordb::Collection::load(file.path()), std::runtime_error);
}

TEST(PersistenceTest, LoadRejectsUnsupportedVersion) {
    const TemporaryFile file("unsupported_version");
    const vectordb::Collection collection(2, vectordb::Metric::L2);
    collection.save(file.path());
    overwrite_bytes(file.path(), k_version_offset,
                    std::string{'\x02', '\0', '\0', '\0'});

    EXPECT_THROW(vectordb::Collection::load(file.path()), std::runtime_error);
}

TEST(PersistenceTest, LoadRejectsInvalidMetricCode) {
    const TemporaryFile file("invalid_metric");
    const vectordb::Collection collection(2, vectordb::Metric::L2);
    collection.save(file.path());
    overwrite_bytes(file.path(), k_metric_offset,
                    std::string{'\x63', '\0', '\0', '\0'});

    EXPECT_THROW(vectordb::Collection::load(file.path()), std::runtime_error);
}

TEST(PersistenceTest, LoadRejectsTruncatedVectorData) {
    const TemporaryFile file("truncated_vector");
    vectordb::Collection collection(2, vectordb::Metric::L2);
    collection.insert("partial", std::vector<float>{1.0f, 2.0f});
    collection.save(file.path());
    std::filesystem::resize_file(file.path(),
                                 std::filesystem::file_size(file.path()) - 1);

    EXPECT_THROW(vectordb::Collection::load(file.path()), std::runtime_error);
}

TEST(PersistenceTest, LoadRejectsDuplicateExternalIds) {
    const TemporaryFile file("duplicate_id");
    vectordb::Collection collection(2, vectordb::Metric::L2);
    collection.insert("alpha", std::vector<float>{1.0f, 2.0f});
    collection.insert("bravo", std::vector<float>{3.0f, 4.0f});
    collection.save(file.path());

    const std::streamoff first_record_size = 4 + 5 + (2 * sizeof(float));
    const std::streamoff second_id_offset =
        k_first_record_offset + first_record_size + 4;
    overwrite_bytes(file.path(), second_id_offset, "alpha");

    EXPECT_THROW(vectordb::Collection::load(file.path()),
                 std::invalid_argument);
}

TEST(PersistenceTest, LoadRejectsEmptyExternalId) {
    const TemporaryFile file("empty_id");
    vectordb::Collection collection(2, vectordb::Metric::L2);
    collection.insert("valid", std::vector<float>{1.0f, 2.0f});
    collection.save(file.path());
    overwrite_bytes(file.path(), k_first_record_offset,
                    std::string{'\0', '\0', '\0', '\0'});

    EXPECT_THROW(vectordb::Collection::load(file.path()), std::runtime_error);
}
