#include "vectordb/collection.hpp"
#include "vectordb/distance.hpp"
#include "vectordb/vector_store.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    int failures = 0;

    void fail(const std::string &message)
    {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }

    void expect_true(bool condition, const std::string &message)
    {
        if (!condition)
        {
            fail(message);
        }
    }

    void expect_near(float actual, float expected, float tolerance, const std::string &message)
    {
        if (std::fabs(actual - expected) > tolerance)
        {
            fail(message + " expected " + std::to_string(expected) + " got " + std::to_string(actual));
        }
    }

    template <typename Exception, typename Function>
    void expect_throws(Function function, const std::string &message)
    {
        try
        {
            function();
            fail(message + " did not throw");
        }
        catch (const Exception &)
        {
        }
        catch (...)
        {
            fail(message + " threw the wrong exception type");
        }
    }

    void test_distance_functions()
    {
        const std::vector<float> a{1.0f, 2.0f, 3.0f};
        const std::vector<float> b{4.0f, 6.0f, 3.0f};

        expect_near(vectordb::l2_distance(a.data(), b.data(), a.size()), 5.0f, 0.0001f, "l2 distance");
        expect_near(vectordb::dot_product(a.data(), b.data(), a.size()), 25.0f, 0.0001f, "dot product");
        expect_near(vectordb::cosine_similarity(a.data(), a.data(), a.size()), 1.0f, 0.0001f, "cosine self similarity");

        std::vector<float> normalized{3.0f, 4.0f};
        vectordb::normalize_in_place(normalized);
        expect_near(normalized[0], 0.6f, 0.0001f, "normalized x");
        expect_near(normalized[1], 0.8f, 0.0001f, "normalized y");

        const std::vector<float> zero{0.0f, 0.0f};
        expect_throws<std::invalid_argument>(
            [&] { vectordb::cosine_similarity(zero.data(), normalized.data(), zero.size()); },
            "cosine rejects zero vector");
        expect_throws<std::invalid_argument>(
            [&] {
                std::vector<float> value{0.0f, 0.0f};
                vectordb::normalize_in_place(value);
            },
            "normalize rejects zero vector");
    }

    void test_vector_store()
    {
        vectordb::VectorStore store(3);

        const std::vector<float> first{1.0f, 2.0f, 3.0f};
        const std::vector<float> second{4.0f, 5.0f, 6.0f};

        expect_true(store.add(first) == 0, "first vector id");
        expect_true(store.add(second) == 1, "second vector id");
        expect_true(store.size() == 2, "store size");
        expect_true(store.dim() == 3, "store dimension");
        expect_near(store.get(1)[2], 6.0f, 0.0001f, "stored vector value");

        const std::vector<float> wrong_dim{1.0f, 2.0f};
        expect_throws<std::invalid_argument>([&] { store.add(wrong_dim); }, "store rejects wrong dimensions");
        expect_throws<std::out_of_range>([&] { store.get(2); }, "store rejects missing vector id");
        expect_throws<std::invalid_argument>([] { vectordb::VectorStore invalid(0); }, "store rejects zero dimension");
    }

    void test_collection_l2_search()
    {
        vectordb::Collection collection(2, vectordb::Metric::L2);

        collection.insert("near", std::vector<float>{1.0f, 1.0f});
        collection.insert("far", std::vector<float>{5.0f, 5.0f});
        collection.insert("exact", std::vector<float>{0.0f, 0.0f});

        const auto results = collection.search(std::vector<float>{0.0f, 0.0f}, 2);
        expect_true(results.size() == 2, "l2 top k size");
        expect_true(results[0].external_id == "exact", "l2 best result");
        expect_true(results[1].external_id == "near", "l2 second result");
        expect_near(results[0].score, 0.0f, 0.0001f, "l2 score");
    }

    void test_collection_dot_and_cosine_search()
    {
        vectordb::Collection dot_collection(2, vectordb::Metric::Dot);
        dot_collection.insert("x", std::vector<float>{1.0f, 0.0f});
        dot_collection.insert("y", std::vector<float>{0.0f, 1.0f});
        dot_collection.insert("big_x", std::vector<float>{2.0f, 0.0f});

        auto dot_results = dot_collection.search(std::vector<float>{1.0f, 0.0f}, 3);
        expect_true(dot_results[0].external_id == "big_x", "dot product ranks highest score first");
        expect_true(dot_results[1].external_id == "x", "dot product ranks second highest score second");

        vectordb::Collection cosine_collection(2, vectordb::Metric::Cosine);
        cosine_collection.insert("same", std::vector<float>{2.0f, 0.0f});
        cosine_collection.insert("orthogonal", std::vector<float>{0.0f, 3.0f});

        auto cosine_results = cosine_collection.search(std::vector<float>{1.0f, 0.0f}, 2);
        expect_true(cosine_results[0].external_id == "same", "cosine ranks highest similarity first");
        expect_near(cosine_results[0].score, 1.0f, 0.0001f, "cosine score");
    }

    void test_collection_validation()
    {
        vectordb::Collection collection(2, vectordb::Metric::L2);

        collection.insert("id", std::vector<float>{1.0f, 2.0f});

        expect_true(collection.size() == 1, "collection size");
        expect_true(collection.dim() == 2, "collection dimension");
        expect_true(collection.search(std::vector<float>{1.0f, 2.0f}, 0).empty(), "top zero returns no results");
        expect_throws<std::invalid_argument>(
            [&] { collection.insert("id", std::vector<float>{3.0f, 4.0f}); },
            "collection rejects duplicate external id");
        expect_throws<std::invalid_argument>(
            [&] { collection.insert("", std::vector<float>{3.0f, 4.0f}); },
            "collection rejects empty external id");
        expect_throws<std::invalid_argument>(
            [&] { collection.search(std::vector<float>{1.0f}, 1); },
            "collection rejects wrong query dimension");
    }
}

int main()
{
    test_distance_functions();
    test_vector_store();
    test_collection_l2_search();
    test_collection_dot_and_cosine_search();
    test_collection_validation();

    if (failures != 0)
    {
        std::cerr << failures << " test failure(s)\n";
        return 1;
    }

    std::cout << "All vectordb tests passed\n";
    return 0;
}
