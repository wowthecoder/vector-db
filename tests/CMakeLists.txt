find_package(GTest QUIET)

if(NOT GTest_FOUND)
    include(FetchContent)

    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )

    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

include(GoogleTest)

add_executable(vectordb_tests
    batch_search_tests.cpp
    collection_tests.cpp
    distance_tests.cpp
    flat_index_tests.cpp
    persistence_tests.cpp
    vector_store_tests.cpp
)

target_link_libraries(vectordb_tests
    PRIVATE
        vectordb
        GTest::gtest_main
)

target_compile_options(vectordb_tests PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

gtest_discover_tests(vectordb_tests)
