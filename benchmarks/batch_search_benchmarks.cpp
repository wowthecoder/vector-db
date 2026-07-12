#include <benchmark/benchmark.h>

namespace
{

    void BM_CollectionBatchSearch(benchmark::State &state)
    {
        // state.range(0): collection size
        // state.range(1): vector dimension
        // state.range(2): queries per batch
        // state.range(3): top_k
        // TODO: Build flattened queries once, then time Collection::batch_search.
        state.SkipWithError("TODO: implement batch search benchmark");
    }

}

BENCHMARK(BM_CollectionBatchSearch)
    ->Args({1'000, 128, 8, 10})
    ->Args({10'000, 128, 32, 10})
    ->Unit(benchmark::kMillisecond);

