The `benchmarks/` directory contains Google Benchmarks for insertion, exact
single search, random-projection LSH search and recall, batch search, repeated
single search, save, and load. The sections below document the measurement
boundaries and remaining improvements.

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
    --benchmark_filter='BM_RandomProjectionLshSearch'
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

## LSH recall benchmark

`BM_RandomProjectionLshSearch` times one cosine query per benchmark iteration.
Fixture generation, exact `FlatIndex` searches, LSH construction, and recall
calculation occur outside the timed loop. Each case reports:

- `recall_at_k`: fraction from 0 to 1 of exact top-k IDs returned by LSH;
- `items_per_second`: timed LSH queries per second;
- `lsh_build_ms`: one deterministic index build;
- `index_payload_bytes`: projection floats plus stored table memberships.

`index_payload_bytes` is a portable logical payload measurement, not allocator
resident memory. It excludes hash nodes, bucket arrays, vector capacity, and
allocator bookkeeping.

The registered cases vary dataset size, dimension, `top_k`, table count,
signature width, and candidate limit one at a time around a canonical workload.
Run only this family with:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_RandomProjectionLshSearch' \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true
```

Recall is computed over 50 fixed-seed queries. It is a quality measurement for
comparing configurations, not a promise that arbitrary workloads have the same
recall.

## GloVe-25 ANN dataset benchmark

The GloVe benchmark measures Flat and random-projection LSH search against the
same 1,183,514-vector cosine dataset and the exact ground truth distributed by
[ANN-Benchmarks](https://ann-benchmarks.com/). Dataset loading, Flat validation,
LSH construction, and recall evaluation occur outside the timed search loop.
The prepared dataset is loaded once and shared by all cases in the process.

Download `glove-25-angular.hdf5` into `benchmark-data/`, install the conversion
script's dependencies, and prepare the dependency-free input used by C++:

```sh
python3 -m pip install h5py numpy
python3 benchmarks/prepare_ann_dataset.py \
    benchmark-data/glove-25-angular.hdf5 \
    benchmark-data/glove-25-angular.vdbann
```

Both files under `benchmark-data/` are ignored by Git. The conversion preserves
all training vectors, all queries, and all supplied ground-truth neighbors.

Run the Flat baseline:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_Glove25FlatSearch' \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true
```

Run the LSH parameter sweep:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_Glove25LshSearch' \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true
```

Save the combined Flat/LSH run and generate its latency and recall charts plus
the detailed report table:

```sh
./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_Glove25' \
    --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true \
    --benchmark_out=glove25-benchmark-results.json \
    --benchmark_out_format=json
python3 benchmarks/generate_report.py glove25-benchmark-results.json \
    --output glove25-benchmark-report.html
```

Each search case reports `items_per_second`, `recall_at_k`, the number of
queries used for recall, dataset size, and dimension. LSH cases also report
`index_build_ms`. Flat and LSH time the same first 100 queries once per
repetition, which prevents short Google Benchmark calibration runs from timing
only the first few queries. The Flat benchmark validates ten queries against
the supplied ground truth before timing; the LSH cases calculate recall over
1,000 queries. The registered LSH cases independently vary table count,
signature width, and candidate limit around an 8-table, 12-bit,
1,000-candidate baseline.

The default prepared dataset path is absolute and derived from the CMake source
directory. To use another prepared file without rebuilding, set:

```sh
VECTORDB_GLOVE25_PREPARED_DATASET=/path/to/glove-25-angular.vdbann \
    ./build-benchmarks/benchmarks/vectordb_benchmarks \
    --benchmark_filter='BM_Glove25'
```

The full dataset makes setup and LSH builds substantially slower than the
synthetic benchmarks. Use a benchmark filter while developing, and do not use
`--benchmark_min_time` results as a published performance baseline.

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
charts, an LSH latency/recall table, GloVe Flat-versus-LSH latency and recall
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
