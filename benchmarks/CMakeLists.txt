find_package(benchmark CONFIG QUIET)

if(NOT benchmark_FOUND)
    include(FetchContent)

    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        googlebenchmark
        URL https://github.com/google/benchmark/archive/refs/tags/v1.8.3.zip
    )

    FetchContent_MakeAvailable(googlebenchmark)
endif()

add_executable(vectordb_benchmarks
    ann_dataset.cpp
    batch_search_benchmarks.cpp
    benchmark_utils.cpp
    glove_25_benchmarks.cpp
    insert_benchmarks.cpp
    lsh_recall_benchmarks.cpp
    persistence_benchmarks.cpp
    search_benchmarks.cpp
)

target_link_libraries(vectordb_benchmarks
    PRIVATE
        vectordb
        benchmark::benchmark_main
)

target_compile_options(vectordb_benchmarks PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

target_compile_definitions(vectordb_benchmarks PRIVATE
    VECTORDB_GLOVE25_DEFAULT_PATH="${PROJECT_SOURCE_DIR}/benchmark-data/glove-25-angular.vdbann"
)
