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
FetchContent when needed. The CLI and benchmarks options are currently off by
default because this repository does not include those targets yet.

Useful CMake options:

```sh
cmake -S . -B build -DVECTORDB_BUILD_TESTS=ON
cmake -S . -B build -DVECTORDB_BUILD_TESTS=OFF
```

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
