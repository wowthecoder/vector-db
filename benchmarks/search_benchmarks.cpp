#include <benchmark/benchmark.h>

namespace
{

    void BM_CollectionSearch(benchmark::State &state)
    {
        // state.range(0): collection size
        // state.range(1): vector dimension
        // state.range(2): top_k
        // TODO: Populate once, then time repeated Collection::search calls.
        state.SkipWithError("TODO: implement single-query search benchmark");
    }

}

BENCHMARK(BM_CollectionSearch)
    ->Args({1'000, 128, 10})
    ->Args({10'000, 128, 10})
    ->Unit(benchmark::kMicrosecond);

