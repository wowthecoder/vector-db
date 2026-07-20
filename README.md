# vector_db

A small C++20 vector database library with an in-memory vector store, distance
helpers, and collection search over L2 distance, dot product, or cosine
similarity.

## Requirements

- CMake 3.22 or newer
- A C++20 compiler such as GCC, Clang, or MSVC
- GoogleTest development package discoverable by CMake, or network access during
  CMake configure so it can be fetched automatically

## Build

Configure and build from the repository root:

```sh
cmake -S . -B build -G Ninja
cmake --build build
```

Tests are enabled by default and use GoogleTest. CMake first looks for an
installed GoogleTest package and falls back to downloading GoogleTest with
FetchContent when needed. The CLI and benchmarks options are off by default.

Useful CMake options:

```sh
cmake -S . -B build -DVECTORDB_BUILD_TESTS=ON
cmake -S . -B build -DVECTORDB_BUILD_TESTS=OFF
cmake -S . -B build-benchmarks -DCMAKE_BUILD_TYPE=Release \
    -DVECTORDB_BUILD_TESTS=OFF -DVECTORDB_BUILD_BENCHMARKS=ON
```

The benchmark targets are intentionally scaffolded for implementation. See
[`docs/BENCHMARKS.md`](docs/BENCHMARKS.md) for the build commands and guidance.

## Format Source Code

The formatting target requires `clang-format`. Configure the project, then run
the `format` target:

```sh
cmake -S . -B build -G Ninja
cmake --build build --target format
```

This formats the project's C and C++ source files in place using the
repository's `.clang-format` configuration, which is based on Google C++ style.

## Run Tests

After building, run the GoogleTest suite with CTest:

```sh
ctest --test-dir build --output-on-failure
```

You can also run the test executable directly:

```sh
./build/tests/vectordb_tests
```

## Basic Usage

```cpp
#include "vectordb/collection.hpp"

#include <vector>

int main()
{
    vectordb::Collection collection(3, vectordb::Metric::Cosine);

    collection.insert("first", std::vector<float>{1.0f, 0.0f, 0.0f});
    collection.insert("second", std::vector<float>{0.0f, 1.0f, 0.0f});

    auto results = collection.search(std::vector<float>{1.0f, 0.0f, 0.0f}, 1);
    return results.empty() ? 1 : 0;
}
```
