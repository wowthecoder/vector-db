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
    batch_search_benchmarks.cpp
    benchmark_utils.cpp
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
