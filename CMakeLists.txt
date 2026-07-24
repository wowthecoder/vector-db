cmake_minimum_required(VERSION 3.22)

project(vector_db
    VERSION 0.1.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(VECTORDB_BUILD_TESTS "Build tests" ON)
option(VECTORDB_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(VECTORDB_BUILD_CLI "Build CLI" OFF)

add_library(vectordb
    src/distance.cpp
    src/indexes/flat_index.cpp
    src/indexes/index_utils.cpp
    src/indexes/random_projection_lsh_index.cpp
    src/io/collection_io.cpp
    src/vector_store.cpp
    src/collection.cpp
)

target_include_directories(vectordb
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_options(vectordb PRIVATE
    -Wall
    -Wextra
    -Wpedantic
)

if(VECTORDB_BUILD_CLI AND EXISTS "${PROJECT_SOURCE_DIR}/cli/main.cpp")
    add_executable(vectordb_cli
        cli/main.cpp
    )

    target_link_libraries(vectordb_cli
        PRIVATE vectordb
    )
endif()

if(VECTORDB_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(VECTORDB_BUILD_BENCHMARKS AND EXISTS "${PROJECT_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()

find_program(CLANG_FORMAT_EXECUTABLE
    NAMES clang-format
)

if(CLANG_FORMAT_EXECUTABLE)
    file(GLOB_RECURSE PROJECT_FORMAT_FILES
        CONFIGURE_DEPENDS
        "${CMAKE_SOURCE_DIR}/benchmarks/*.h"
        "${CMAKE_SOURCE_DIR}/benchmarks/*.hpp"
        "${CMAKE_SOURCE_DIR}/benchmarks/*.c"
        "${CMAKE_SOURCE_DIR}/benchmarks/*.cc"
        "${CMAKE_SOURCE_DIR}/benchmarks/*.cpp"
        "${CMAKE_SOURCE_DIR}/include/*.h"
        "${CMAKE_SOURCE_DIR}/include/*.hpp"
        "${CMAKE_SOURCE_DIR}/src/*.c"
        "${CMAKE_SOURCE_DIR}/src/*.cc"
        "${CMAKE_SOURCE_DIR}/src/*.cpp"
        "${CMAKE_SOURCE_DIR}/tests/*.c"
        "${CMAKE_SOURCE_DIR}/tests/*.cc"
        "${CMAKE_SOURCE_DIR}/tests/*.cpp"
    )

    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXECUTABLE}
                -i
                -style=file
                ${PROJECT_FORMAT_FILES}
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Formatting C/C++ source files"
        VERBATIM
    )
else()
    message(WARNING "clang-format was not found; the format target is unavailable")
endif()
