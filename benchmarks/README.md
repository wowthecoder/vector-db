The `benchmarks/` directory contains Google Benchmarks for insertion, single
search, batch search, repeated single search, save, and load. The sections
below document the intended measurement boundaries and remaining improvements.

## Build and run

Keep benchmarks in a release build. Debug builds mainly measure disabled
optimizations and assertion overhead. Run the correctness tests before
collecting benchmark results:

```sh
cmake -S . -B build -DVECTORDB_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Configure a separate release build for benchmarks, then build only the
benchmark executable:

```sh
cmake -S . -B build-benchmarks \
    -DCMAKE_BUILD_TYPE=Release \
    -DVECTORDB_BUILD_TESTS=OFF \
    -DVECTORDB_BUILD_BENCHMARKS=ON
cmake --build build-benchmarks --target vectordb_benchmarks
```

CMake uses an installed Google Benchmark package when one is available and
otherwise fetches version 1.8.3. Run these commands from the repository root.

List every registered case before starting a potentially long full run:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks --benchmark_list_tests
```

Run the complete suite or one benchmark family:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_CollectionInsert'
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_CollectionSearch'
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_(CollectionBatchSearch|RepeatedSingleSearch)'
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_Collection(Save|Load)'
```

Use repetitions and compare aggregate medians for a useful local baseline:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true
```

Save machine-readable results for later comparison:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true \
    --benchmark_out=benchmark-results.json \
    --benchmark_out_format=json
```

## Generate an HTML report

Turn a Google Benchmark JSON file into a self-contained dashboard:

```sh
python3 benchmarks/generate_report.py benchmark-results.json \
    --output benchmark-report.html
```

The generator requires Python 3.10 or newer and has no third-party package
dependencies.

Open `benchmark-report.html` in any modern browser. The report contains run
metadata and measurement warnings, a canonical-workload table, search-scaling
charts, batch-versus-repeated-search comparisons, insertion and persistence
charts, and a sortable, filterable table of every benchmark case. All CSS,
JavaScript, and SVG charts are embedded in the HTML; generation requires only
the Python standard library and viewing the report does not require a network
connection.

When an input contains repeated benchmark runs, the report uses aggregate
medians and displays the wall-time coefficient of variation. If aggregate rows
are absent, it calculates those statistics from the raw repetitions. A
single-repetition input remains supported, but the dashboard marks it as a
trend-only measurement.

Run the report-generator tests with:

```sh
python3 -m unittest discover -s benchmarks -p 'test_generate_report.py' -v
```

For a quick smoke run while developing the harness, select a small case and
reduce the minimum measurement time. Do not use this shortened run as a
performance baseline:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_CollectionInsert/1000/128$' \
    --benchmark_min_time=0.01s
```

Run benchmark processes one at a time on an otherwise idle machine. Parallel
runs contend for CPU, memory bandwidth, allocator locks, and filesystem caches,
which makes comparisons unreliable.
